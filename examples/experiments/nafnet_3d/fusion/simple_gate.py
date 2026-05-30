import numpy as np

import paddle

paddle.enable_compat(scope={"triton"})

import triton
import triton.language as tl

from .autotune import event_time, tensor_size


# Fixed parameters: SimpleGate is purely elementwise without reductions, so the
# launch config does not matter much. We pick a small BLOCK_M to keep occupancy
# high while still amortising the program launch overhead.
BLOCK_M = 8


@triton.jit
def fused_simple_gate_forward(
    x_ptr,
    out_ptr,
    channels: tl.constexpr,
    half_channels: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C_HALF: tl.constexpr,
):
    """SimpleGate forward: out[..., :C/2] = x[..., :C/2] * x[..., C/2:]."""
    pid = tl.program_id(0)
    row_offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    c = tl.arange(0, BLOCK_C_HALF)

    p_offsets = row_offsets[:, None] * channels + c[None, :]
    q_offsets = p_offsets + half_channels
    out_offsets = row_offsets[:, None] * half_channels + c[None, :]

    p = tl.load(x_ptr + p_offsets)
    q = tl.load(x_ptr + q_offsets)
    tl.store(out_ptr + out_offsets, p * q)


@triton.jit
def fused_simple_gate_backward(
    x_ptr,
    grad_ptr,
    dx_ptr,
    channels: tl.constexpr,
    half_channels: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C_HALF: tl.constexpr,
):
    """SimpleGate backward: dx_p = grad * q, dx_q = grad * p."""
    pid = tl.program_id(0)
    row_offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    c = tl.arange(0, BLOCK_C_HALF)

    p_offsets = row_offsets[:, None] * channels + c[None, :]
    q_offsets = p_offsets + half_channels
    grad_offsets = row_offsets[:, None] * half_channels + c[None, :]

    p = tl.load(x_ptr + p_offsets)
    q = tl.load(x_ptr + q_offsets)
    grad = tl.load(grad_ptr + grad_offsets)

    tl.store(dx_ptr + p_offsets, grad * q)
    tl.store(dx_ptr + q_offsets, grad * p)


################################################################################
# Kernel wrappers
################################################################################


def _is_power_of_2(value):
    return value > 0 and (value & (value - 1)) == 0


def _check_args(x):
    assert x.shape[-1] % 2 == 0, f"last dim must be even, got {x.shape[-1]}"
    channels = x.shape[-1]
    half_channels = channels // 2
    rows = int(np.prod(x.shape[:-1]))
    assert _is_power_of_2(half_channels), (
        f"half channels must be power of 2, got {half_channels}"
    )
    assert rows % BLOCK_M == 0, f"rows must be divisible by {BLOCK_M}, got {rows}"
    return rows, channels, half_channels


def simple_gate_forward(x):
    rows, channels, half_channels = _check_args(x)

    out_shape = list(x.shape)
    out_shape[-1] = half_channels
    out = paddle.empty(out_shape, dtype=x.dtype)

    grid = (rows // BLOCK_M,)
    fused_simple_gate_forward[grid](
        x,
        out,
        channels,
        half_channels,
        BLOCK_M=BLOCK_M,
        BLOCK_C_HALF=half_channels,
    )
    return out


def simple_gate_backward(x, grad):
    rows, channels, half_channels = _check_args(x)
    assert grad.shape == x.shape[:-1] + [half_channels]

    dx = paddle.empty_like(x)
    grid = (rows // BLOCK_M,)
    fused_simple_gate_backward[grid](
        x,
        grad,
        dx,
        channels,
        half_channels,
        BLOCK_M=BLOCK_M,
        BLOCK_C_HALF=half_channels,
    )
    return dx


class FusedSimpleGateTriton(paddle.autograd.PyLayer):
    """Triton SimpleGate with autograd support."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return simple_gate_forward(x)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensor()
        return simple_gate_backward(x, grad)


def _ref_forward(x):
    p, q = paddle.chunk(x, 2, axis=-1)
    return p * q


if __name__ == "__main__":
    paddle.seed(2026)
    paddle.set_printoptions(linewidth=160)
    paddle.set_flags({"FLAGS_share_tensor_for_grad_tensor_holder": True})
    dtype = "bfloat16"
    N, H, W, C = 64, 256, 256, 64

    for depth in range(6):
        x = paddle.randn([N, H, W, C], dtype=dtype).requires_grad_()
        x_ref = x.detach().requires_grad_()
        out_grad = paddle.randn([N, H, W, C // 2], dtype=dtype) * 0.1
        print("-" * 30, f"shape={(N, H, W, C)}", "-" * 30)

        # run paddle reference
        out_ref = _ref_forward(x_ref)
        out_ref.backward(out_grad)

        # run triton
        out = FusedSimpleGateTriton.apply(x)
        out.backward(out_grad)

        # check accuracy
        out_diff = (out - out_ref).abs().float()
        x_grad_diff = (x.grad - x_ref.grad).abs().float()
        print("out_diff avg:", out_diff.mean().item(), "max:", out_diff.max().item())
        print("dx_diff avg:", x_grad_diff.mean().item(), "max:", x_grad_diff.max().item())

        # benchmark paddle
        t = event_time(lambda: (
            inp := x.detach().requires_grad_(),
            o := _ref_forward(inp),
            o.backward(out_grad),
        ))
        print("paddle_time:", t)

        # benchmark triton
        t = event_time(lambda: (
            inp := x.detach().requires_grad_(),
            o := FusedSimpleGateTriton.apply(inp),
            o.backward(out_grad),
        ))
        print("triton_time:", t)

        # report throughput (read x + write out + read grad + read x + write dx)
        total_bytes = tensor_size(x) * 3 + tensor_size(out_grad) * 2
        print(f"triton_throughput: {total_bytes / t / 1e6:.0f} GB/s")

        H, W, C = H // 2, W // 2, C * 2
