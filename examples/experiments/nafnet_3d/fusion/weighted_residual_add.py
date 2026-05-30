import numpy as np
from functools import partial

import paddle

paddle.enable_compat(scope={"triton"})

import triton
import triton.language as tl

from .autotune import BEST_CONFIG, tune_config, tensor_size, event_time


# Forward is pure elementwise, fixed config is fine.
BLOCK_M_FWD = 8


@triton.jit
def fused_weighted_residual_add_forward(
    residual_ptr,
    x_ptr,
    weight_ptr,
    out_ptr,
    channels: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """out = residual + x * weight[broadcast over rows]."""
    pid = tl.program_id(0)
    row_offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    c = tl.arange(0, BLOCK_C)
    offsets = row_offsets[:, None] * channels + c[None, :]

    residual = tl.load(residual_ptr + offsets).to(tl.float32)
    x = tl.load(x_ptr + offsets).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)

    out = residual + x * weight[None, :]
    tl.store(out_ptr + offsets, out)


@triton.jit
def fused_weighted_residual_add_backward(
    dout_ptr,
    x_ptr,
    weight_ptr,
    dx_ptr,
    partial_ptr,
    channels: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
    LOOP_M: tl.constexpr,
):
    """One pass over dout: write dx and accumulate partial dweight.

    dx = dout * weight, dweight = sum(dout * x, axis=rows). Each program covers
    BLOCK_M * LOOP_M rows of the full ``channels``-wide row space.
    """
    pid_m = tl.program_id(0)
    row_in_block = tl.arange(0, BLOCK_M)
    c = tl.arange(0, BLOCK_C)

    weight = tl.load(weight_ptr + c).to(tl.float32)
    dweight = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for loop_idx in tl.range(0, LOOP_M):
        row_offsets = (pid_m * LOOP_M + loop_idx) * BLOCK_M + row_in_block
        offsets = row_offsets[:, None] * channels + c[None, :]

        dout = tl.load(dout_ptr + offsets).to(tl.float32)
        x = tl.load(x_ptr + offsets).to(tl.float32)

        dx = dout * weight[None, :]
        tl.store(dx_ptr + offsets, dx)
        dweight += tl.sum(dout * x, axis=0)

    base = pid_m * channels + c
    tl.store(partial_ptr + base, dweight)


################################################################################
# Kernel wrappers
################################################################################


def _is_power_of_2(value):
    return value > 0 and (value & (value - 1)) == 0


def _check_args(residual, x, weight):
    assert residual.shape == x.shape, (
        f"residual and x shape mismatch: {residual.shape} vs {x.shape}"
    )
    channels = x.shape[-1]
    rows = int(np.prod(x.shape[:-1]))
    assert weight.shape == [channels]
    assert _is_power_of_2(channels), f"channels must be power of 2, got {channels}"
    return rows, channels


def weighted_residual_add_forward(residual, x, weight):
    rows, channels = _check_args(residual, x, weight)
    assert rows % BLOCK_M_FWD == 0, f"rows must be divisible by {BLOCK_M_FWD}, got {rows}"

    out = paddle.empty_like(x)
    grid = (rows // BLOCK_M_FWD,)
    fused_weighted_residual_add_forward[grid](
        residual,
        x,
        weight,
        out,
        channels,
        BLOCK_M=BLOCK_M_FWD,
        BLOCK_C=channels,
    )
    return out


def _weighted_residual_add_backward(dout, x, weight, block_m, loop_m):
    rows, channels = _check_args(dout, x, weight)
    assert rows % block_m == 0, f"rows must be divisible by {block_m}, got {rows}"
    assert (rows // block_m) % loop_m == 0, (
        f"rows // block_m must be divisible by loop_m, got {rows // block_m} and {loop_m}"
    )

    dx = paddle.empty_like(dout)
    num_partials = rows // (block_m * loop_m)
    partial = paddle.empty([num_partials, channels], dtype="float32")
    fused_weighted_residual_add_backward[(num_partials,)](
        dout,
        x,
        weight,
        dx,
        partial,
        channels,
        BLOCK_M=block_m,
        BLOCK_C=channels,
        LOOP_M=loop_m,
    )
    dweight = partial.sum(axis=0).cast(x.dtype)
    return dx, dweight


def weighted_residual_add_backward(dout, x, weight):
    key = ("weighted_residual_add_backward", tuple(x.shape))
    config = BEST_CONFIG.get(key)

    if config is None:
        (block_m, loop_m), best_time, tuning_time = tune_config(
            partial(_weighted_residual_add_backward, dout, x, weight),
            [1, 2, 4, 8, 16, 32],
            [1, 2, 4, 8, 16, 32, 64],
        )
        BEST_CONFIG[key] = (block_m, loop_m)
        throughput = tensor_size(x) * 3 / best_time / 1e6
        print(f"[autotune] (weighted_residual_add_backward) shape={key[1]} "
              f"{block_m=} {loop_m=} {throughput=:.0f}GB/s {tuning_time=:.3f}s")
    else:
        block_m, loop_m = config

    return _weighted_residual_add_backward(dout, x, weight, block_m, loop_m)


class FusedWeightedResidualAddTriton(paddle.autograd.PyLayer):
    """Triton fused (residual + x * weight) with autograd support."""

    @staticmethod
    def forward(ctx, residual, x, weight):
        ctx.save_for_backward(x, weight)
        return weighted_residual_add_forward(residual, x, weight)

    @staticmethod
    def backward(ctx, dout):
        x, weight = ctx.saved_tensor()
        d_x, d_weight = weighted_residual_add_backward(dout, x, weight)
        return dout, d_x, d_weight


def _ref_forward(residual, x, weight):
    return residual + x * weight


if __name__ == "__main__":
    paddle.seed(2026)
    paddle.set_printoptions(linewidth=160)
    paddle.set_flags({"FLAGS_share_tensor_for_grad_tensor_holder": True})
    dtype = "bfloat16"
    N, H, W, C = 64, 256, 256, 64

    for depth in range(6):
        residual = paddle.randn([N, H, W, C], dtype=dtype).requires_grad_()
        x = paddle.randn([N, H, W, C], dtype=dtype).requires_grad_()
        weight = paddle.randn([C], dtype=dtype).requires_grad_()
        residual_ref = residual.detach().requires_grad_()
        x_ref = x.detach().requires_grad_()
        weight_ref = weight.detach().requires_grad_()
        out_grad = paddle.randn_like(x) * 0.1
        print("-" * 30, f"shape={(N, H, W, C)}", "-" * 30)

        # paddle ref
        out_ref = _ref_forward(residual_ref, x_ref, weight_ref)
        out_ref.backward(out_grad)

        # triton
        out = FusedWeightedResidualAddTriton.apply(residual, x, weight)
        out.backward(out_grad)

        out_diff = (out - out_ref).abs().float()
        residual_diff = (residual.grad - residual_ref.grad).abs().float()
        x_diff = (x.grad - x_ref.grad).abs().float()
        weight_diff = (weight.grad - weight_ref.grad).abs().float()
        print("out_diff avg:", out_diff.mean().item(), "max:", out_diff.max().item())
        print("d_residual_diff avg:", residual_diff.mean().item(), "max:", residual_diff.max().item())
        print("d_x_diff avg:", x_diff.mean().item(), "max:", x_diff.max().item())
        print("d_weight_diff avg:", weight_diff.mean().item(), "max:", weight_diff.max().item())

        # benchmark paddle
        t = event_time(lambda: (
            r := residual.detach().requires_grad_(),
            xx := x.detach().requires_grad_(),
            ww := weight.detach().requires_grad_(),
            o := _ref_forward(r, xx, ww),
            o.backward(out_grad),
        ))
        print("paddle_time:", t)

        # benchmark triton
        t = event_time(lambda: (
            r := residual.detach().requires_grad_(),
            xx := x.detach().requires_grad_(),
            ww := weight.detach().requires_grad_(),
            o := FusedWeightedResidualAddTriton.apply(r, xx, ww),
            o.backward(out_grad),
        ))
        print("triton_time:", t)

        H, W, C = H // 2, W // 2, C * 2
