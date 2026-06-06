import paddle
from paddle import Tensor

paddle.enable_compat(scope={"triton"})

import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernels — 1D flat reduce, shape agnostic
# ---------------------------------------------------------------------------


@triton.jit
def l1_loss_fwd_reduce_bwd_kernel(
    InputPtr,
    LabelPtr,
    PartialSumPtr,  # [num_programs] float32
    GradSignPtr,
    size: tl.constexpr,  # total number of elements
    LOOP_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel: 同时计算 partial sum (|input-label|) 和 grad_input (sign)."""
    pid = tl.program_id(0)

    start = pid * LOOP_K * BLOCK_SIZE
    cols = tl.arange(0, BLOCK_SIZE)

    partial = 0.0

    for k in tl.static_range(LOOP_K):
        offs = start + k * BLOCK_SIZE + cols
        mask = offs < size

        x = tl.load(InputPtr + offs, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(LabelPtr + offs, mask=mask, other=0.0).to(tl.float32)

        diff = x - y
        abs_diff = tl.abs(diff)

        # accumulate for loss
        partial += tl.sum(abs_diff)

        # sign(diff)
        sign = tl.where(diff > 0.0, 1, -1)
        sign = tl.where(diff == 0.0, 0, sign)

        tl.store(GradSignPtr + offs, sign, mask=mask)

    tl.store(PartialSumPtr + pid, partial)


@triton.jit
def l1_loss_bwd_kernel(
    GradOutPtr,  # scalar tensor
    GradSignPtr,
    GradInputPtr,
    size: tl.constexpr,  # total number of elements
    LOOP_K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Backward: dx = grad_out * sign(input - label) / N."""
    pid = tl.program_id(0)

    start = pid * LOOP_K * BLOCK_SIZE
    cols = tl.arange(0, BLOCK_SIZE)

    go = tl.load(GradOutPtr).to(tl.float32)
    scale = go / tl.full([1], float(size), dtype=tl.float32)

    for k in tl.static_range(LOOP_K):
        offs = start + k * BLOCK_SIZE + cols
        mask = offs < size

        sign = tl.load(GradSignPtr + offs, mask=mask, other=0)
        grad = scale * sign

        tl.store(GradInputPtr + offs, grad, mask=mask)


# ---------------------------------------------------------------------------
# PyLayer
# ---------------------------------------------------------------------------


class FusedL1LossTriton(paddle.autograd.PyLayer):
    """Fused L1 Loss with cached grad bitmap."""

    @staticmethod
    def forward(ctx, input: Tensor, label: Tensor):
        assert input.shape == label.shape
        assert label.stop_gradient, "label mustn't require grad"
        size = input.size
        assert size % 1024 == 0, f"size ({size}) must be multiple of 1024"

        # default config
        block_size = 1024
        loop_k = 8

        chunk = block_size * loop_k
        assert size % chunk == 0, (
            f"size should be multiple of chunk, got {size} and {chunk}"
        )
        num_programs = size // chunk

        # --- Allocate ---
        partial = paddle.empty([num_programs], dtype="float32")
        grad_sign = paddle.empty(input.shape, dtype="int8")

        # --- Fused kernel: partial reduce + grad compute ---
        l1_loss_fwd_reduce_bwd_kernel[(num_programs,)](
            input, label, partial, grad_sign,
            size=size,
            LOOP_K=loop_k,
            BLOCK_SIZE=block_size,
        )

        # --- Stage 2: paddle.sum -> mean ---
        loss = paddle.sum(partial) / float(size)

        # 把预计算好的 grad_sign 存到 ctx
        ctx.save_for_backward(grad_sign)

        ctx.dtype = input.dtype
        ctx.num_programs = num_programs
        ctx.block_size = block_size
        ctx.loop_k = loop_k

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        (grad_sign,) = ctx.saved_tensor()

        grad_input = paddle.empty(grad_sign.shape, dtype=ctx.dtype)

        l1_loss_bwd_kernel[(ctx.num_programs,)](
            grad_output, grad_sign, grad_input,
            size=grad_sign.size,
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
