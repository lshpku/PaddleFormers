import time
import numpy as np
from functools import partial

import paddle
from paddle import nn

paddle.enable_compat(scope={"triton"})

import triton
import triton.language as tl

from .autotune import BEST_CONFIG, tune_config, tensor_size, event_time


@triton.jit
def fused_layernorm_forward_multirow(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    mean_ptr,
    rstd_ptr,
    channels: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """LayerNorm forward for multiple NHWC rows per program."""
    pid = tl.program_id(0)
    row_offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    c = tl.arange(0, BLOCK_C)
    offsets = row_offsets[:, None] * channels + c[None, :]

    x = tl.load(x_ptr + offsets).to(tl.float32)
    mean = tl.sum(x, axis=1) / channels
    centered = x - mean[:, None]
    variance = tl.sum(centered * centered, axis=1) / channels
    rstd = tl.rsqrt(variance + eps)

    weight = tl.load(weight_ptr + c).to(tl.float32)
    bias = tl.load(bias_ptr + c).to(tl.float32)
    out = centered * rstd[:, None] * weight[None, :] + bias[None, :]

    tl.store(out_ptr + offsets, out)
    tl.store(mean_ptr + row_offsets, mean)
    tl.store(rstd_ptr + row_offsets, rstd)


@triton.jit
def fused_layernorm_dx_multirow(
    dy_ptr,
    x_ptr,
    weight_ptr,
    mean_ptr,
    rstd_ptr,
    dx_ptr,
    channels: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """LayerNorm dx for multiple NHWC rows per program."""
    pid = tl.program_id(0)
    row_offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    c = tl.arange(0, BLOCK_C)
    offsets = row_offsets[:, None] * channels + c[None, :]

    dy = tl.load(dy_ptr + offsets).to(tl.float32)
    x = tl.load(x_ptr + offsets).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)
    mean = tl.load(mean_ptr + row_offsets).to(tl.float32)
    rstd = tl.load(rstd_ptr + row_offsets).to(tl.float32)

    x_hat = (x - mean[:, None]) * rstd[:, None]
    wdy = dy * weight[None, :]
    mean_wdy = tl.sum(wdy, axis=1) / channels
    mean_wdy_xhat = tl.sum(wdy * x_hat, axis=1) / channels
    dx = (wdy - mean_wdy[:, None] - x_hat * mean_wdy_xhat[:, None]) * rstd[:, None]

    tl.store(dx_ptr + offsets, dx)


@triton.jit
def fused_layernorm_dweight_dbias_partial_multirow(
    dy_ptr,
    x_ptr,
    mean_ptr,
    rstd_ptr,
    partial_ptr,
    channels: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
    LOOP_M: tl.constexpr,
):
    """Accumulate partial dweight/dbias for BLOCK_M * LOOP_M NHWC rows."""
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    row_in_block = tl.arange(0, BLOCK_M)
    c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    dweight = tl.zeros((BLOCK_C,), dtype=tl.float32)
    dbias = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for loop_idx in tl.range(0, LOOP_M):
        row_offsets = (pid_m * LOOP_M + loop_idx) * BLOCK_M + row_in_block
        offsets = row_offsets[:, None] * channels + c[None, :]

        dy = tl.load(dy_ptr + offsets).to(tl.float32)
        x = tl.load(x_ptr + offsets).to(tl.float32)
        mean = tl.load(mean_ptr + row_offsets).to(tl.float32)
        rstd = tl.load(rstd_ptr + row_offsets).to(tl.float32)
        x_hat = (x - mean[:, None]) * rstd[:, None]

        dweight += tl.sum(dy * x_hat, axis=0)
        dbias += tl.sum(dy, axis=0)

    base = (pid_m * 2 * channels) + c
    tl.store(partial_ptr + base, dweight)
    tl.store(partial_ptr + base + channels, dbias)


@triton.jit
def fused_layernorm_dx_dweight_dbias_partial_multirow(
    dy_ptr,
    x_ptr,
    weight_ptr,
    mean_ptr,
    rstd_ptr,
    dx_ptr,
    partial_ptr,
    channels: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
    LOOP_M: tl.constexpr,
):
    """Compute full dx and partial dweight/dbias for BLOCK_M * LOOP_M NHWC rows."""
    pid_m = tl.program_id(0)
    row_in_block = tl.arange(0, BLOCK_M)
    c = tl.arange(0, BLOCK_C)

    weight = tl.load(weight_ptr + c).to(tl.float32)
    dweight = tl.zeros((BLOCK_C,), dtype=tl.float32)
    dbias = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for loop_idx in tl.range(0, LOOP_M):
        row_offsets = (pid_m * LOOP_M + loop_idx) * BLOCK_M + row_in_block
        offsets = row_offsets[:, None] * channels + c[None, :]

        dy = tl.load(dy_ptr + offsets).to(tl.float32)
        x = tl.load(x_ptr + offsets).to(tl.float32)
        mean = tl.load(mean_ptr + row_offsets).to(tl.float32)
        rstd = tl.load(rstd_ptr + row_offsets).to(tl.float32)

        x_hat = (x - mean[:, None]) * rstd[:, None]
        wdy = dy * weight[None, :]
        mean_wdy = tl.sum(wdy, axis=1) / channels
        mean_wdy_xhat = tl.sum(wdy * x_hat, axis=1) / channels
        dx = (wdy - mean_wdy[:, None] - x_hat * mean_wdy_xhat[:, None]) * rstd[:, None]

        tl.store(dx_ptr + offsets, dx)
        dweight += tl.sum(dy * x_hat, axis=0)
        dbias += tl.sum(dy, axis=0)

    base = (pid_m * 2 * channels) + c
    tl.store(partial_ptr + base, dweight)
    tl.store(partial_ptr + base + channels, dbias)


################################################################################
# Kernel wrappers
################################################################################


def _is_power_of_2(value):
    return value > 0 and (value & (value - 1)) == 0


def _check_layernorm_args(x, weight=None, bias=None, block_m=1):
    assert len(x.shape) == 4, f"x must be NHWC, got shape {x.shape}"
    channels = x.shape[-1]
    rows = int(np.prod(x.shape[:-1]))
    if weight is not None:
        assert weight.shape == [channels]
    if bias is not None:
        assert bias.shape == [channels]
    assert _is_power_of_2(channels), f"channels must be power of 2, got {channels}"
    assert rows % block_m == 0, f"rows must be divisible by block_m, got {rows} and {block_m}"
    return rows, channels


def _layernorm_forward(x, weight, bias, eps, block_m):
    rows, channels = _check_layernorm_args(x, weight, bias, block_m)

    out = paddle.empty_like(x)
    mean = paddle.empty([rows], dtype="float32")
    rstd = paddle.empty([rows], dtype="float32")
    grid = (rows // block_m,)
    fused_layernorm_forward_multirow[grid](
        x,
        weight,
        bias,
        out,
        mean,
        rstd,
        channels,
        eps,
        BLOCK_M=block_m,
        BLOCK_C=channels,
    )
    return out, mean, rstd


def _layernorm_dx(dy, x, weight, mean, rstd, block_m):
    rows, channels = _check_layernorm_args(x, weight, block_m=block_m)
    assert dy.shape == x.shape
    assert mean.shape == [rows]
    assert rstd.shape == [rows]

    dx = paddle.empty_like(x)
    grid = (rows // block_m,)
    fused_layernorm_dx_multirow[grid](
        dy,
        x,
        weight,
        mean,
        rstd,
        dx,
        channels,
        BLOCK_M=block_m,
        BLOCK_C=channels,
    )
    return dx


def _check_backward_reduce_args(dy, x, mean, rstd, block_m, loop_m):
    rows, channels = _check_layernorm_args(x, block_m=block_m)
    assert dy.shape == x.shape
    assert mean.shape == [rows]
    assert rstd.shape == [rows]
    assert loop_m > 0
    assert (rows // block_m) % loop_m == 0, (
        f"rows // block_m must be divisible by loop_m, got {rows // block_m} and {loop_m}"
    )
    return rows, channels


def _layernorm_dweight_dbias(dy, x, mean, rstd, block_m, loop_m):
    rows, channels = _check_backward_reduce_args(dy, x, mean, rstd, block_m, loop_m)

    num_partials = rows // (block_m * loop_m)
    partial = paddle.empty([num_partials, 2, channels], dtype="float32")
    block_c = min(channels, 1024)
    fused_layernorm_dweight_dbias_partial_multirow[(num_partials, channels // block_c)](
        dy,
        x,
        mean,
        rstd,
        partial,
        channels,
        BLOCK_M=block_m,
        BLOCK_C=block_c,
        LOOP_M=loop_m,
    )
    reduced = partial.sum(axis=0).cast(x.dtype)
    return reduced[0], reduced[1]


def _layernorm_dx_dweight_dbias(dy, x, weight, mean, rstd, block_m, loop_m):
    rows, channels = _check_backward_reduce_args(dy, x, mean, rstd, block_m, loop_m)
    assert weight.shape == [channels]

    num_partials = rows // (block_m * loop_m)
    dx = paddle.empty_like(x)
    partial = paddle.empty([num_partials, 2, channels], dtype="float32")
    fused_layernorm_dx_dweight_dbias_partial_multirow[(num_partials,)](
        dy,
        x,
        weight,
        mean,
        rstd,
        dx,
        partial,
        channels,
        BLOCK_M=block_m,
        BLOCK_C=channels,
        LOOP_M=loop_m,
    )
    reduced = partial.sum(axis=0).cast(x.dtype)
    return dx, reduced[0], reduced[1]


################################################################################
# Autotune wrappers
################################################################################


def layernorm_forward(x, weight, bias, eps=1e-5):
    key = ("layernorm_forward", tuple(x.shape))
    block_m = BEST_CONFIG.get(key)

    if block_m is None:
        block_m, best_time, tuning_time = tune_config(
            partial(_layernorm_forward, x, weight, bias, eps),
            [1, 2, 4, 8, 16, 32],
        )
        BEST_CONFIG[key] = block_m
        throughput = tensor_size(x) * 2 / best_time / 1e6
        print(f"[autotune] (layernorm_forward) shape={key[1]} "
              f"{block_m=} {throughput=:.0f}GB/s {tuning_time=:.3f}s")

    return _layernorm_forward(x, weight, bias, eps, block_m)


def layernorm_dx(dy, x, weight, mean, rstd):
    key = ("layernorm_dx", tuple(x.shape))
    block_m = BEST_CONFIG.get(key)

    if block_m is None:
        block_m, best_time, tuning_time = tune_config(
            partial(_layernorm_dx, dy, x, weight, mean, rstd),
            [1, 2, 4, 8, 16, 32],
        )
        BEST_CONFIG[key] = block_m
        throughput = (tensor_size(x) * 3 + tensor_size(mean) * 2) / best_time / 1e6
        print(f"[autotune] (layernorm_dx) shape={key[1]} "
              f"{block_m=} {throughput=:.0f}GB/s {tuning_time=:.3f}s")

    return _layernorm_dx(dy, x, weight, mean, rstd, block_m)


def layernorm_dweight_dbias(out_grad, x, mean, rstd):
    key = ("layernorm_dweight_dbias", tuple(x.shape))
    config = BEST_CONFIG.get(key)

    if config is None:
        (block_m, loop_m), best_time, tuning_time = tune_config(
            partial(_layernorm_dweight_dbias, out_grad, x, mean, rstd),
            [1, 2, 4, 8, 16, 32],
            [1, 2, 4, 8, 16, 32, 64],
        )
        BEST_CONFIG[key] = (block_m, loop_m)
        throughput = (tensor_size(x) * 2 + tensor_size(mean) * 2) / best_time / 1e6
        print(f"[autotune] (layernorm_dweight_dbias) shape={key[1]} "
              f"{block_m=} {loop_m=} {throughput=:.0f}GB/s {tuning_time=:.3f}s")
    else:
        block_m, loop_m = config

    return _layernorm_dweight_dbias(out_grad, x, mean, rstd, block_m, loop_m)


def layernorm_dx_dweight_dbias(out_grad, x, weight, mean, rstd):
    key = ("layernorm_dx_dweight_dbias", tuple(x.shape))
    config = BEST_CONFIG.get(key)

    if config is None:
        (block_m, loop_m), best_time, tuning_time = tune_config(
            partial(_layernorm_dx_dweight_dbias, out_grad, x, weight, mean, rstd),
            [1, 2, 4, 8, 16, 32],
            [1, 2, 4, 8, 16, 32, 64],
        )
        BEST_CONFIG[key] = (block_m, loop_m)
        throughput = (tensor_size(x) * 3 + tensor_size(mean) * 2) / best_time / 1e6
        print(f"[autotune] (layernorm_dx_dweight_dbias) shape={key[1]} "
              f"{block_m=} {loop_m=} {throughput=:.0f}GB/s {tuning_time=:.3f}s")
    else:
        block_m, loop_m = config

    return _layernorm_dx_dweight_dbias(out_grad, x, weight, mean, rstd, block_m, loop_m)


def layernorm_backward(out_grad, x, weight, mean, rstd):
    key = ("layernorm_backward", tuple(x.shape))
    fuse_dx_dw = BEST_CONFIG.get(key)

    if fuse_dx_dw is None:
        fusion_time = event_time(
            lambda: layernorm_dx_dweight_dbias(out_grad, x, weight, mean, rstd)
        )
        non_fusion_time = event_time(
            lambda: (
                layernorm_dx(out_grad, x, weight, mean, rstd),
                layernorm_dweight_dbias(out_grad, x, mean, rstd),
            )
        )
        fuse_dx_dw = fusion_time < non_fusion_time
        BEST_CONFIG[key] = fuse_dx_dw
        print(f"[autotune] (layernorm_backward) shape={key[1]} fuse_dx_dw={int(fuse_dx_dw)}"
              f" time_compare={fusion_time:.3f}:{non_fusion_time:.3f}ms")

    if fuse_dx_dw:
        return layernorm_dx_dweight_dbias(out_grad, x, weight, mean, rstd)
    else:
        dx = layernorm_dx(out_grad, x, weight, mean, rstd)
        dweight, dbias = layernorm_dweight_dbias(out_grad, x, mean, rstd)
        return dx, dweight, dbias


class FusedLayerNormTriton(paddle.autograd.PyLayer):
    """Triton NHWC LayerNorm with autograd support."""

    @staticmethod
    def forward(ctx, x, weight, bias, eps=1e-5):
        out, mean, rstd = layernorm_forward(x, weight, bias, eps)
        ctx.save_for_backward(x, weight, mean, rstd)
        return out

    @staticmethod
    def backward(ctx, dy):
        x, weight, mean, rstd = ctx.saved_tensor()
        dx, dweight, dbias = layernorm_backward(dy, x, weight, mean, rstd)
        return dx, dweight, dbias


if __name__ == "__main__":
    paddle.seed(2026)
    paddle.set_printoptions(linewidth=160)
    paddle.set_flags({"FLAGS_share_tensor_for_grad_tensor_holder": True})
    dtype = "bfloat16"
    N, H, W, C = 64, 256, 256, 64

    for depth in range(6):
        x = paddle.randn([N, H, W, C], dtype=dtype).requires_grad_()
        x_ref = x.detach().requires_grad_()
        out_grad = paddle.randn_like(x) * 0.1
        print("-" * 30, f"shape={(N, H, W, C)}", "-" * 30)

        layer = nn.LayerNorm(C)
        layer.weight.data += paddle.randn_like(layer.weight) * 0.1
        layer.bias.data += paddle.randn_like(layer.bias) * 0.1
        layer = paddle.amp.decorate(layer, level="O2", dtype="bfloat16")

        # run paddle
        out_ref = layer(x_ref)
        out_ref.backward(out_grad)
        weight_grad_ref = layer.weight.grad.clone()
        bias_grad_ref = layer.bias.grad.clone()
        layer.clear_gradients(False)

        # run triton
        out = FusedLayerNormTriton.apply(x, layer.weight, layer.bias)
        out.backward(out_grad)

        # check accuracy
        out_diff = (out - out_ref).abs().float()
        x_grad_diff = (x.grad - x_ref.grad).abs().float()
        weight_grad_diff = (layer.weight.grad - weight_grad_ref).abs().float()
        bias_grad_diff = (layer.bias.grad - bias_grad_ref).abs().float()
        print("out_diff avg:", out_diff.mean().item(), "max:", out_diff.max().item())
        print("dx_diff avg:", x_grad_diff.mean().item(), "max:", x_grad_diff.max().item())
        print("dweight_diff avg:", weight_grad_diff.mean().item(), "max:", weight_grad_diff.max().item())
        print("dbias_diff avg:", bias_grad_diff.mean().item(), "max:", bias_grad_diff.max().item())

        layer.clear_gradients(False)

        # benchmark
        t = event_time(lambda: (
            inp := x.detach().requires_grad_(),
            out := layer(inp),
            out.backward(out_grad),
            layer.clear_gradients(False),
        ))
        print("paddle_time:", t)

        t = event_time(lambda: (
            inp := x.detach().requires_grad_(),
            out := FusedLayerNormTriton.apply(inp, layer.weight, layer.bias),
            out.backward(out_grad),
            layer.clear_gradients(False),
        ))
        print("triton_time:", t)

        H, W, C = H // 2, W // 2, C * 2
