from functools import partial

import paddle

paddle.enable_compat(scope={"triton"})

import triton
import triton.language as tl

from .autotune import BEST_CONFIG, tune_config, tensor_size, event_time


# Backward is purely elementwise after broadcasting dpool, so fixed config is fine.
BLOCK_M_BWD = 8


@triton.jit
def fused_simple_gate_avg_pool_forward(
    x_ptr,
    gate_ptr,
    partial_ptr,
    spatial_size: tl.constexpr,  # H * W
    channels: tl.constexpr,
    half_channels: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C_HALF: tl.constexpr,
    LOOP_M: tl.constexpr,
):
    """Fused SimpleGate + AdaptiveAvgPool2d(1) forward.

    Each program handles one (n, spatial_block) pair, processing BLOCK_M * LOOP_M
    spatial rows:
      - loads x[n, hw, :C] in NHWC layout
      - computes gate = x[..., :C/2] * x[..., C/2:]  (SimpleGate)
      - stores gate rows to gate_ptr
      - accumulates sum(gate, over rows) into partial_ptr for the avg-pool

    Post-kernel: sum partials over spatial_blocks, divide by spatial_size to get
    pool_out[n, c] = mean_{h,w}(gate[n, h, w, c]).
    """
    pid = tl.program_id(0)
    num_partials = spatial_size // (BLOCK_M * LOOP_M)
    n = pid // num_partials
    spatial_block = pid % num_partials

    c = tl.arange(0, BLOCK_C_HALF)
    partial_sum = tl.zeros((BLOCK_C_HALF,), dtype=tl.float32)

    for loop_idx in tl.range(0, LOOP_M):
        row_in_block = tl.arange(0, BLOCK_M)
        hw_offsets = (spatial_block * LOOP_M + loop_idx) * BLOCK_M + row_in_block

        # x[n, hw, c] in NHWC layout: n * spatial_size * channels + hw * channels + c
        x_base = n * spatial_size * channels
        p_offsets = x_base + hw_offsets[:, None] * channels + c[None, :]
        q_offsets = p_offsets + half_channels

        p = tl.load(x_ptr + p_offsets).to(tl.float32)
        q = tl.load(x_ptr + q_offsets).to(tl.float32)
        gate = p * q

        # Store gate[n, hw, c] in [N, spatial_size, half_channels] layout
        gate_base = n * spatial_size * half_channels
        gate_offsets = gate_base + hw_offsets[:, None] * half_channels + c[None, :]
        tl.store(gate_ptr + gate_offsets, gate)

        partial_sum += tl.sum(gate, axis=0)

    # Store partial: partial[n * num_partials + spatial_block, c]
    partial_base = pid * half_channels + c
    tl.store(partial_ptr + partial_base, partial_sum)


@triton.jit
def fused_simple_gate_avg_pool_backward(
    x_ptr,
    dgate_ptr,
    dpool_ptr,
    dx_ptr,
    spatial_size: tl.constexpr,  # H * W
    channels: tl.constexpr,
    half_channels: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C_HALF: tl.constexpr,
):
    """Backward pass.

    Given dgate[n,hw,c] and dpool[n,c] (broadcast over hw):
      d_combined[n,hw,c] = dgate[n,hw,c] + dpool[n,c] / spatial_size
      dx[n,hw,c]         = d_combined * x[n,hw,c+half_C]
      dx[n,hw,c+half_C]  = d_combined * x[n,hw,c]
    """
    pid = tl.program_id(0)
    row_offsets = pid * BLOCK_M + tl.arange(0, BLOCK_M)   # into [0, N * spatial_size)
    c = tl.arange(0, BLOCK_C_HALF)

    # Load x[n, hw, c] and x[n, hw, c+half_C]
    p_offsets = row_offsets[:, None] * channels + c[None, :]
    q_offsets = p_offsets + half_channels
    p = tl.load(x_ptr + p_offsets).to(tl.float32)
    q = tl.load(x_ptr + q_offsets).to(tl.float32)

    # Load dgate[n, hw, c]
    dgate_offsets = row_offsets[:, None] * half_channels + c[None, :]
    dgate = tl.load(dgate_ptr + dgate_offsets).to(tl.float32)

    # Load dpool[n, c] and broadcast over hw.
    # dpool_ptr points to [N, half_channels] (contiguous), n = row // spatial_size.
    n = row_offsets // spatial_size
    dpool_offsets = n[:, None] * half_channels + c[None, :]
    dpool = tl.load(dpool_ptr + dpool_offsets).to(tl.float32)

    # Combined gradient: dgate contribution + broadcast dpool contribution
    # Note: this FMA may introduce diff with eager mode.
    d_combined = dgate + dpool * (1.0 / spatial_size)

    tl.store(dx_ptr + p_offsets, d_combined * q)
    tl.store(dx_ptr + q_offsets, d_combined * p)


################################################################################
# Kernel wrappers
################################################################################


def _is_power_of_2(value):
    return value > 0 and (value & (value - 1)) == 0


def _check_args(x):
    assert x.ndim == 4, f"expected 4D tensor [N,H,W,C], got {x.ndim}D"
    N, H, W, C = x.shape
    assert C % 2 == 0, f"last dim must be even, got {C}"
    half_C = C // 2
    assert _is_power_of_2(half_C), f"half channels must be power of 2, got {half_C}"
    spatial_size = H * W
    return N, H, W, C, half_C, spatial_size


def _simple_gate_avg_pool_forward(x, block_m, loop_m):
    N, H, W, C, half_C, spatial_size = _check_args(x)
    assert spatial_size % (block_m * loop_m) == 0, (
        f"H*W={spatial_size} must be divisible by block_m*loop_m={block_m*loop_m}"
    )

    num_partials = spatial_size // (block_m * loop_m)
    gate_out = paddle.empty([N, H, W, half_C], dtype=x.dtype)
    partial = paddle.empty([N, num_partials, half_C], dtype="float32")

    grid = (N * num_partials,)
    fused_simple_gate_avg_pool_forward[grid](
        x, gate_out, partial,
        spatial_size,
        channels=C, half_channels=half_C,
        BLOCK_M=block_m, BLOCK_C_HALF=half_C, LOOP_M=loop_m,
    )

    # Sum partial accumulators per batch item, normalise
    pool_out = (
        partial.sum(axis=1) / spatial_size
    ).cast(x.dtype).reshape([N, 1, 1, half_C])
    return gate_out, pool_out


def simple_gate_avg_pool_forward(x):
    N, H, W, C, half_C, spatial_size = _check_args(x)
    key = ("simple_gate_avg_pool_forward", tuple(x.shape))
    config = BEST_CONFIG.get(key)

    if config is None:
        (block_m, loop_m), best_time, tuning_time = tune_config(
            partial(_simple_gate_avg_pool_forward, x),
            [1, 2, 4, 8, 16, 32],
            [1, 2, 4, 8, 16, 32, 64],
        )
        BEST_CONFIG[key] = (block_m, loop_m)
        throughput = (tensor_size(x) + tensor_size(x) // 2) / best_time / 1e6
        print(f"[autotune] (simple_gate_avg_pool_forward) shape={key[1]} "
              f"{block_m=} {loop_m=} {throughput=:.0f}GB/s {tuning_time=:.3f}s")
    else:
        block_m, loop_m = config

    return _simple_gate_avg_pool_forward(x, block_m, loop_m)


def simple_gate_avg_pool_backward(x, dgate, dpool):
    N, H, W, C, half_C, spatial_size = _check_args(x)
    total_rows = N * spatial_size
    assert total_rows % BLOCK_M_BWD == 0, (
        f"N*H*W={total_rows} must be divisible by BLOCK_M_BWD={BLOCK_M_BWD}"
    )
    assert list(dgate.shape) == [N, H, W, half_C], (
        f"dgate shape mismatch: expected {[N, H, W, half_C]}, got {list(dgate.shape)}"
    )
    assert list(dpool.shape) == [N, 1, 1, half_C], (
        f"dpool shape mismatch: expected {[N, 1, 1, half_C]}, got {list(dpool.shape)}"
    )

    dx = paddle.empty_like(x)

    grid = (total_rows // BLOCK_M_BWD,)
    fused_simple_gate_avg_pool_backward[grid](
        x, dgate, dpool, dx,
        spatial_size,
        channels=C, half_channels=half_C,
        BLOCK_M=BLOCK_M_BWD, BLOCK_C_HALF=half_C,
    )
    return dx


class FusedSimpleGateAvgPoolTriton(paddle.autograd.PyLayer):
    """Triton fused SimpleGate + AdaptiveAvgPool2d(1) with autograd support.

    Input:  x  shape [N, H, W, C]    (C must be even; C//2 must be power of 2)
    Output: gate_out  shape [N, H, W, C//2]   — x[..,:C/2] * x[..,C/2:]
            pool_out  shape [N, 1, 1, C//2]   — spatial average of gate_out

    Both outputs are computed in a single kernel pass over x, saving the
    separate write+read of gate_out that an unfused avg_pool would require.
    """

    @staticmethod
    def forward(ctx, x):
        gate_out, pool_out = simple_gate_avg_pool_forward(x)
        ctx.save_for_backward(x)
        return gate_out, pool_out

    @staticmethod
    def backward(ctx, dgate, dpool):
        (x,) = ctx.saved_tensor()
        return simple_gate_avg_pool_backward(x, dgate, dpool)


def _ref_forward(x):
    p, q = paddle.chunk(x, 2, axis=-1)
    x = p * q
    pool = paddle.nn.AdaptiveAvgPool2d(1, data_format="NHWC")
    return x, pool(x)


if __name__ == "__main__":
    paddle.seed(2026)
    paddle.set_flags({"FLAGS_share_tensor_for_grad_tensor_holder": True})
    dtype = "bfloat16"
    N, H, W, C = 64, 256, 256, 128

    for depth in range(6):
        x = paddle.ones([N, H, W, C], dtype=dtype).requires_grad_()
        x_ref = x.detach().requires_grad_()
        dgate_grad = paddle.randn([N, H, W, C // 2], dtype=dtype) * 0.1
        dpool_grad = paddle.randn([N, 1, 1, C // 2], dtype=dtype) * 0.1
        print("-" * 30, f"shape={(N, H, W, C)}", "-" * 30)

        # paddle
        gate_ref, pool_ref = _ref_forward(x_ref)
        paddle.autograd.backward([gate_ref, pool_ref], [dgate_grad, dpool_grad])

        # triton
        gate_out, pool_out = FusedSimpleGateAvgPoolTriton.apply(x)
        paddle.autograd.backward([gate_out, pool_out], [dgate_grad, dpool_grad])

        # accuracy
        gate_diff = (gate_out - gate_ref).abs().float()
        pool_diff = (pool_out - pool_ref).abs().float()
        dx_diff = (x.grad - x_ref.grad).abs().float()
        print("gate_diff avg:", gate_diff.mean().item(), "max:", gate_diff.max().item())
        print("pool_diff avg:", pool_diff.mean().item(), "max:", pool_diff.max().item())
        print("dx_diff avg:", dx_diff.mean().item(), "max:", dx_diff.max().item())

        # benchmark paddle
        t = event_time(lambda: (
            inp := x.detach().requires_grad_(),
            res := _ref_forward(inp),
            paddle.autograd.backward(res, [dgate_grad, dpool_grad]),
        ))
        print("paddle_time:", t)

        # benchmark triton
        t = event_time(lambda: (
            inp := x.detach().requires_grad_(),
            res := FusedSimpleGateAvgPoolTriton.apply(inp),
            paddle.autograd.backward(res, [dgate_grad, dpool_grad]),
        ))
        print("triton_time:", t)

        H, W, C = H // 2, W // 2, C * 2
