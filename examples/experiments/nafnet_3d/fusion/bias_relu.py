import paddle

paddle.enable_compat(scope={"triton"})

import triton
import triton.language as tl


@triton.jit
def fused_bias_relu_fwd_kernel(
    X_ptr,
    B_ptr,
    Y_ptr,
    actual_c: tl.constexpr,
    LOOP_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """前向 kernel: y = relu(x + bias), tile = [BLOCK_M, BLOCK_C], 每 program 跑 LOOP_K 个 tile."""
    pid = tl.program_id(0)

    rows = tl.arange(0, BLOCK_M)  # [BLOCK_M]
    cols = tl.arange(0, BLOCK_C)  # [BLOCK_C]
    col_mask = cols < actual_c  # [BLOCK_C]

    # bias 寄存器常驻, 复用到所有 tile
    b = tl.load(B_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)  # [BLOCK_C]

    base_row = pid * LOOP_K * BLOCK_M
    mask = col_mask[None, :]

    for k in tl.static_range(LOOP_K):
        row_start = base_row + k * BLOCK_M
        offs = (row_start + rows)[:, None] * actual_c + cols[None, :]

        x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = x + b[None, :]
        y = tl.where(y > 0.0, y, 0.0)
        tl.store(Y_ptr + offs, y, mask=mask)


@triton.jit
def fused_bias_relu_bwd_kernel(
    DY_ptr,
    Y_ptr,
    DX_ptr,
    actual_c: tl.constexpr,
    LOOP_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """
    反向 kernel: dx_ij = dy_ij * 1[y_ij > 0]
    利用 y = relu(x+bias) 的 mask 等价于 (x+bias)>0, 无需重算 x+bias.
    bias 不需要梯度, 不计算 db.
    """
    pid = tl.program_id(0)

    rows = tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_C)
    col_mask = cols < actual_c

    base_row = pid * LOOP_K * BLOCK_M
    mask = col_mask[None, :]

    for k in tl.static_range(LOOP_K):
        row_start = base_row + k * BLOCK_M
        offs = (row_start + rows)[:, None] * actual_c + cols[None, :]

        y = tl.load(Y_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY_ptr + offs, mask=mask, other=0.0).to(tl.float32)

        dx = tl.where(y > 0.0, dy, 0.0)
        tl.store(DX_ptr + offs, dx, mask=mask)


def _pick_block_m(block_c: int) -> int:
    """根据 C 的 next_power_of_2 选 BLOCK_M, 目标 ~1024 元素/tile."""
    target = 1024
    bm = max(1, target // block_c)
    # clamp 到 [1, 16], 16 是 N*H*W 必须整除的上限
    bm = min(bm, 16)
    # 必须是 2 的幂
    return 1 << (bm.bit_length() - 1)


def _pick_loop_k(num_row_blocks: int, target: int = 8) -> int:
    """
    选择能整除 num_row_blocks 的最大 LOOP_K (≤ target, 2 的幂).
    目标 8; 不整除时降到 4/2/1 兜底.
    """
    k = target
    while k > 1 and num_row_blocks % k != 0:
        k //= 2
    return k


class FusedBiasReluTriton(paddle.autograd.PyLayer):
    """Triton Fused Bias + ReLU with autograd support (bias no grad)."""

    @staticmethod
    def forward(ctx, x, bias):
        """forward: y = relu(x + bias)"""
        assert bias.stop_gradient, "bias mustn't requires grad"
        assert x.shape[-1:] == bias.shape, (
            f"last dim of x ({x.shape[-1]}) must equal bias size ({bias.shape})"
        )

        orig_shape = x.shape
        c = orig_shape[-1]
        n1 = 1
        for s in orig_shape[:-1]:
            n1 *= s
        assert n1 % 16 == 0, (
            f"N*H*W (got {n1}) must be a multiple of 16 for tiled BLOCK_M loading"
        )

        block_c = triton.next_power_of_2(c)
        block_m = _pick_block_m(block_c)

        y = paddle.empty(orig_shape, dtype=x.dtype)

        # 确定循环: 每个 program 跑 LOOP_K 个 [BLOCK_M, BLOCK_C] tile
        num_row_blocks = n1 // block_m
        loop_k = _pick_loop_k(num_row_blocks, target=8)
        num_programs = num_row_blocks // loop_k

        fused_bias_relu_fwd_kernel[(num_programs,)](
            x,
            bias,
            y,
            c,
            LOOP_K=loop_k,
            BLOCK_M=block_m,
            BLOCK_C=block_c,
        )

        ctx.save_for_backward(y)
        ctx.c = c
        ctx.block_c = block_c
        ctx.block_m = block_m
        ctx.loop_k = loop_k
        ctx.num_programs = num_programs
        return y

    @staticmethod
    def backward(ctx, dy):
        """backward: dx = dy * 1[y > 0]; bias 不计梯度"""
        (y,) = ctx.saved_tensor()
        c = ctx.c
        block_c = ctx.block_c
        block_m = ctx.block_m
        loop_k = ctx.loop_k
        num_programs = ctx.num_programs

        dx = paddle.empty(dy.shape, dtype=dy.dtype)

        fused_bias_relu_bwd_kernel[(num_programs,)](
            dy,
            y,
            dx,
            c,
            LOOP_K=loop_k,
            BLOCK_M=block_m,
            BLOCK_C=block_c,
        )

        return dx


def _ref_forward(x, bias):
    return paddle.nn.functional.relu(x + bias)


if __name__ == "__main__":
    from .autotune import event_time

    N, H, W, C = 64, 256, 256, 64

    for i in range(4):
        x = paddle.randn([N, H, W, C], "bfloat16")
        b = paddle.randn([C], "bfloat16")

        xr = x.detach().requires_grad_()
        yr = _ref_forward(xr, b)
        yr.backward()

        xt = x.detach().requires_grad_()
        yt = FusedBiasReluTriton.apply(xt, b)
        yt.backward()

        assert paddle.all(yr == yt)
        assert paddle.all(xr.grad == xt.grad)

        t = event_time(lambda: FusedBiasReluTriton.apply(xt, b))
        tp = x.size * x.itemsize * 2 / t / 1e6

        print(f"[forward] shape: {x.shape} tp: {tp:.1f} GB/s")

        H, W, C = H // 2, W // 2, C * 2
