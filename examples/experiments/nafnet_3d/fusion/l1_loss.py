import paddle
from paddle import Tensor

paddle.enable_compat(scope={"triton"})

import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernels — 1D flat reduce, shape agnostic
# ---------------------------------------------------------------------------
#
# grad_sign packing scheme:
#   每个 program 有 BLOCK_SIZE 条 lane，每条 lane 在 LOOP_K=4 次循环中各处理 1 个 elem，
#   因此每条 lane 共有 4 个 sign(diff)，每个 sign 只有 3 个状态({-1,0,+1})，
#   用 2-bit 编码:  +1 -> 0b01, -1 -> 0b10, 0 -> 0b00
#   4 个 2-bit 正好打包到一个 uint8 中（位 [2k+1:2k] 存放 k 次循环的 sign）。
#
# 因此 grad_sign 的字节数 = size / LOOP_K
# 索引: 第 pid 个 program 的第 c 条 lane 的 packed byte 位于
#         GradSignPtr + pid * BLOCK_SIZE + c
# ---------------------------------------------------------------------------


@triton.jit
def l1_loss_fwd_reduce_bwd_kernel(
    InputPtr,
    LabelPtr,
    PartialSumPtr,  # [num_programs] float32
    GradSignPtr,    # [num_programs * BLOCK_SIZE] uint8 (packed 4x2bit)
    size: tl.constexpr,  # total number of elements
    LOOP_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel: 同时计算 partial sum (|input-label|) 和 packed grad_sign."""
    tl.static_assert(LOOP_K == 4, "LOOP_K must be 4 for 2-bit packing into uint8")

    pid = tl.program_id(0)

    start = pid * LOOP_K * BLOCK_SIZE
    cols = tl.arange(0, BLOCK_SIZE)

    partial = 0.0
    packed = tl.zeros([BLOCK_SIZE], dtype=tl.uint8)

    for k in tl.static_range(LOOP_K):
        offs = start + k * BLOCK_SIZE + cols
        mask = offs < size

        x = tl.load(InputPtr + offs, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(LabelPtr + offs, mask=mask, other=0.0).to(tl.float32)

        diff = x - y
        abs_diff = tl.abs(diff)

        # accumulate for loss
        partial += tl.sum(abs_diff)

        # encode sign(diff): +1 -> 1, -1 -> 2, 0 -> 0
        code = tl.where(diff > 0.0, 1, tl.where(diff < 0.0, 2, 0)).to(tl.uint8)
        packed |= code << (2 * k)

    tl.store(GradSignPtr + pid * BLOCK_SIZE + cols, packed)
    tl.store(PartialSumPtr + pid, partial)


@triton.jit
def l1_loss_bwd_kernel(
    GradOutPtr,  # scalar tensor
    GradSignPtr,  # packed uint8
    GradInputPtr,
    size: tl.constexpr,  # total number of elements
    LOOP_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Backward: dx = grad_out * sign(input - label) / N，从 packed grad_sign 解码."""
    tl.static_assert(LOOP_K == 4, "LOOP_K must be 4 for 2-bit packing into uint8")

    pid = tl.program_id(0)

    start = pid * LOOP_K * BLOCK_SIZE
    cols = tl.arange(0, BLOCK_SIZE)

    go = tl.load(GradOutPtr).to(tl.float32)
    scale = go / tl.full([1], float(size), dtype=tl.float32)

    # 一次性载入该 program 全部 packed sign(BLOCK_SIZE 字节)
    packed = tl.load(GradSignPtr + pid * BLOCK_SIZE + cols)

    for k in tl.static_range(LOOP_K):
        offs = start + k * BLOCK_SIZE + cols
        mask = offs < size

        code = (packed >> (2 * k)) & 0x3
        # decode: 0 -> 0, 1 -> +1, 2 -> -1
        sign = tl.where(code == 1, 1.0,
                        tl.where(code == 2, -1.0, 0.0))
        grad = scale * sign

        tl.store(GradInputPtr + offs, grad, mask=mask)


# ---------------------------------------------------------------------------
# PyLayer
# ---------------------------------------------------------------------------


class FusedL1LossTriton(paddle.autograd.PyLayer):
    """Fused L1 Loss with 2-bit packed grad_sign cache."""

    @staticmethod
    def forward(ctx, input: Tensor, label: Tensor):
        assert input.shape == label.shape
        assert label.stop_gradient, "label mustn't require grad"
        size = input.size

        # default config — LOOP_K must be 4 for 2-bit packing
        block_size = 1024
        loop_k = 4

        chunk = block_size * loop_k
        assert size % chunk == 0, (
            f"size should be multiple of chunk, got {size} and {chunk}"
        )
        num_programs = size // chunk

        # --- Allocate ---
        partial = paddle.empty([num_programs], dtype="float32")
        # packed grad_sign: 每条 lane 一个 uint8 (容纳 4 个 2-bit sign)
        grad_sign = paddle.empty([num_programs * block_size], dtype="uint8")

        # --- Fused kernel: partial reduce + packed grad compute ---
        l1_loss_fwd_reduce_bwd_kernel[(num_programs,)](
            input, label, partial, grad_sign,
            size=size,
            LOOP_K=loop_k,
            BLOCK_SIZE=block_size,
        )

        # --- Stage 2: paddle.sum -> mean ---
        loss = paddle.sum(partial) / float(size)

        ctx.save_for_backward(grad_sign)

        ctx.dtype = input.dtype
        ctx.shape = input.shape
        ctx.size = size
        ctx.num_programs = num_programs
        ctx.block_size = block_size
        ctx.loop_k = loop_k

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        (grad_sign,) = ctx.saved_tensor()

        grad_input = paddle.empty(ctx.shape, dtype=ctx.dtype)

        l1_loss_bwd_kernel[(ctx.num_programs,)](
            grad_output, grad_sign, grad_input,
            size=ctx.size,
            LOOP_K=ctx.loop_k,
            BLOCK_SIZE=ctx.block_size,
        )

        return grad_input, None


# ---------------------------------------------------------------------------
# Reference forward
# ---------------------------------------------------------------------------

def _ref_forward(input, label):
    assert input.shape == label.shape
    assert input.size % 1024 == 0
    assert label.stop_gradient, "label mustn't require grad"
    return paddle.abs(input - label).abs().mean(dtype="float32")


# ---------------------------------------------------------------------------
# Precision test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from .autotune import event_time
    paddle.seed(2026)

    N, H, W, C = 64, 256, 256, 64

    for i in range(4):
        p = paddle.randn([N, H, W, C], "bfloat16")
        q = paddle.randn([N, H, W, C], "bfloat16")
        grad = paddle.to_tensor(1e6, dtype="float32")

        # --- Reference ---
        pr = p.detach().requires_grad_()
        yr = _ref_forward(pr, q)
        yr.backward(grad)

        # --- Triton ---
        pt = p.detach().requires_grad_()
        size = pt.size
        yt = FusedL1LossTriton.apply(pt, q)
        yt.backward(grad)

        # --- Compare ---
        y_diff = (yr - yt).abs().max().item()
        g_diff = (pr.grad - pt.grad).abs().max().item()

        print(f"[iter {i}] shape={N}x{H}x{W}x{C} {y_diff=} {g_diff=}"
              f" y_ref={yr.item():.6f} y_triton={yt.item():.6f}")

        assert y_diff < 1e-5, f"Forward mismatch: {y_diff}"
        assert g_diff == 0, f"Backward mismatch: {g_diff}"

        t = event_time(lambda: (
            o := FusedL1LossTriton.apply(pt.detach().requires_grad_(), q),
            o.backward(),
        ))
        tp = p.size * p.itemsize * 3 / t / 1e6

        print(f"tp: {tp:.1f} GB/s")

        H, W, C = H // 2, W // 2, C * 2

    print("All precision tests passed.")
