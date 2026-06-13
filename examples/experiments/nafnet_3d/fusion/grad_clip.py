import paddle
from paddle import Tensor
from paddle.nn import Parameter, Linear, Sequential
from async_utils import to_device

paddle.enable_compat(scope={"triton"})

import triton
import triton.language as tl


# 两档 BLOCK_SIZE：< THRESHOLD 走小 block 单 program，>= THRESHOLD 走大 block + SHARE 拆分。
SMALL_BLOCK = 1024
LARGE_BLOCK = 524288
THRESHOLD = 16384
SMALL_SHARE = 1
LARGE_SHARE = 64       # 524288 / 64 = 8192 elements per program
ROW_SIZE = 512


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------
# 每个 chunk 由 (addrs[chunk_id], sizes[chunk_id]) 描述，最多 BLOCK_SIZE 个 elem。
# 通过 SHARE 把一个 chunk 拆给 SHARE 个 program 协作处理（不共享输出，不需要 atomic）：
#   pid 范围 = num_chunks * SHARE
#   chunk_id = pid // SHARE,  share_id = pid % SHARE
#   每 program 负责 chunk 内 [share_id*SPAN, (share_id+1)*SPAN) 这段，SPAN = BLOCK_SIZE//SHARE
# 输出 partial_norm[pid]，主机端再 sum。SHARE=1 时退化成原来一个 program 一个 chunk。
# ---------------------------------------------------------------------------


@triton.jit
def sum_norm_kernel(
    AddrPtr,         # int64[num_chunks]
    SizePtr,         # int64[num_chunks]
    PartialNormPtr,  # float32[num_chunks * SHARE]
    BLOCK_SIZE: tl.constexpr,
    ROW_SIZE: tl.constexpr,
    SHARE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    chunk_id = pid // SHARE
    share_id = pid % SHARE
    SPAN: tl.constexpr = BLOCK_SIZE // SHARE

    addr = tl.load(AddrPtr + chunk_id)
    size = tl.load(SizePtr + chunk_id)
    base = tl.cast(addr, tl.pointer_type(DTYPE), bitcast=True)
    start = share_id * SPAN

    cols = tl.arange(0, ROW_SIZE)
    acc = tl.zeros([ROW_SIZE], dtype=tl.float32)

    for i in tl.static_range(0, SPAN, ROW_SIZE):
        offs = start + i + cols
        mask = offs < size
        x = tl.load(base + offs, mask=mask, other=0.0).to(tl.float32)
        acc += x * x

    tl.store(PartialNormPtr + pid, tl.sum(acc))


@triton.jit
def apply_clip_kernel(
    AddrPtr,    # int64[num_chunks]
    SizePtr,    # int64[num_chunks]
    RatioPtr,   # float32[1]
    BLOCK_SIZE: tl.constexpr,
    ROW_SIZE: tl.constexpr,
    SHARE: tl.constexpr,
    DTYPE: tl.constexpr,
):
    ratio = tl.load(RatioPtr).to(tl.float32)
    if ratio == 1.0:
        return

    pid = tl.program_id(0)
    chunk_id = pid // SHARE
    share_id = pid % SHARE
    SPAN: tl.constexpr = BLOCK_SIZE // SHARE

    addr = tl.load(AddrPtr + chunk_id)
    size = tl.load(SizePtr + chunk_id)
    base = tl.cast(addr, tl.pointer_type(DTYPE), bitcast=True)
    start = share_id * SPAN

    cols = tl.arange(0, ROW_SIZE)
    for i in tl.static_range(0, SPAN, ROW_SIZE):
        offs = start + i + cols
        mask = offs < size
        x = tl.load(base + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(base + offs, x * ratio, mask=mask)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------


_TL_DTYPE = {
    paddle.float32: tl.float32,
    paddle.bfloat16: tl.bfloat16,
}


def _block_for(size: int) -> tuple[int, int]:
    """Pick (block_size, share) tier based on grad element count."""
    if size < THRESHOLD:
        return SMALL_BLOCK, SMALL_SHARE
    return LARGE_BLOCK, LARGE_SHARE


@paddle.no_grad()
def clip_grad_norm_(parameters: list[Parameter], max_norm: float) -> Tensor:
    # group grads by (block_size, share, dtype). Each launch needs uniform constexprs.
    groups: dict[tuple, list[Tensor]] = {}
    for param in parameters:
        grad = param.grad
        if grad is None or not grad._is_initialized():
            continue
        assert grad.dtype in _TL_DTYPE, f"unsupported grad dtype {grad.dtype}"
        block_size, share = _block_for(grad.size)
        groups.setdefault((block_size, share, grad.dtype), []).append(grad)

    # build chunk lists, slice the global partial_norm buffer per group.
    plans = []  # (block_size, share, dtype, addrs, sizes, off, n_programs)
    total_programs = 0
    for (block_size, share, dtype), grads in groups.items():
        addr_list, size_list = [], []
        for grad in grads:
            for off in range(0, grad.size, block_size):
                addr_list.append(grad.data_ptr() + off * grad.itemsize)
                size_list.append(min(block_size, grad.size - off))
        n_chunks = len(addr_list)
        n_programs = n_chunks * share
        addrs = to_device(addr_list, dtype="int64")
        sizes = to_device(size_list, dtype="int64")
        plans.append((block_size, share, dtype, addrs, sizes, total_programs, n_programs))
        total_programs += n_programs

    partial_norms = paddle.empty([total_programs], dtype="float32")
    for block_size, share, dtype, addrs, sizes, off, n_progs in plans:
        sum_norm_kernel[(n_progs,)](
            addrs, sizes, partial_norms[off:off + n_progs],
            BLOCK_SIZE=block_size,
            ROW_SIZE=ROW_SIZE,
            SHARE=share,
            DTYPE=_TL_DTYPE[dtype],
        )

    global_norm = partial_norms.sum().sqrt()

    # standard clip: ratio = min(max_norm / (global_norm + eps), 1.0)
    ratio = (max_norm / (global_norm + 1e-6)).clip(max=1.0)

    for block_size, share, dtype, addrs, sizes, _off, n_progs in plans:
        apply_clip_kernel[(n_progs,)](
            addrs, sizes, ratio,
            BLOCK_SIZE=block_size,
            ROW_SIZE=ROW_SIZE,
            SHARE=share,
            DTYPE=_TL_DTYPE[dtype],
        )

    return global_norm


@paddle.no_grad()
def _ref_clip_grad_norm_(grads: list[Tensor], max_norm: float) -> Tensor:
    sq = paddle.stack([(g.float() ** 2).sum() for g in grads]).sum()
    global_norm = sq.sqrt()
    max_norm = paddle.to_tensor(max_norm, dtype="float32")
    ratio = max_norm / paddle.maximum(global_norm + 1e-6, max_norm)
    for g in grads:
        g.scale_(ratio)
    return global_norm


# ---------------------------------------------------------------------------
# Precision tests
# ---------------------------------------------------------------------------


def _build_model():
    return Sequential(
        # small (< THRESHOLD): exercises SMALL_BLOCK tier
        Linear(32, 64),       # 2048 elem weight (>1024 → 2 small chunks)
        Linear(64, 17),       # 1088 elem
        Linear(17, 257),      # 4369 elem
        Linear(257, 33),      # 8481 elem
        # large (>= THRESHOLD): exercises LARGE_BLOCK + SHARE tier
        Linear(33, 1024),     # 33792 elem (1 large chunk)
        Linear(1024, 1024),   # 1048576 elem (2 large chunks, each split SHARE ways)
    )


def _run_case(name: str, cast_bf16: bool, max_norm: float):
    paddle.seed(2026)
    x = paddle.randn([4, 32])

    model = _build_model()
    if cast_bf16:
        # Cast a subset of layers to bfloat16 to mimic AMP-O2 mixed-dtype grads.
        for layer in list(model)[3:]:
            layer.to(dtype="bfloat16")

    params = model.parameters()

    def _fwd_bwd():
        h = x
        for layer in model:
            if layer.weight.dtype == paddle.bfloat16:
                h = h.astype("bfloat16")
            h = layer(h)
        h.float().sum().backward()

    # reference
    _fwd_bwd()
    ref_grads = [p.grad.clone() for p in params]
    ref_norm = _ref_clip_grad_norm_(ref_grads, max_norm=max_norm)
    model.clear_gradients()

    # triton
    _fwd_bwd()
    tri_norm = clip_grad_norm_(params, max_norm=max_norm)
    tri_grads = [p.grad for p in params]

    norm_diff = (ref_norm - tri_norm).abs().item()
    grad_diff = max(
        (rg.float() - tg.float()).abs().max().item()
        for rg, tg in zip(ref_grads, tri_grads)
    )
    print(f"[{name}] norm ref={ref_norm.item():.6f} tri={tri_norm.item():.6f} "
          f"diff={norm_diff:.3e}  grad_max_diff={grad_diff:.3e}")

    norm_tol = 1e-1 if cast_bf16 else 1e-3
    grad_tol = 5e-3 if cast_bf16 else 1e-5
    assert norm_diff < norm_tol, f"[{name}] global_norm mismatch: {norm_diff}"
    assert grad_diff < grad_tol, f"[{name}] grad mismatch: {grad_diff}"


if __name__ == "__main__":
    _run_case("fp32-clip", cast_bf16=False, max_norm=10.0)
    _run_case("fp32-noop", cast_bf16=False, max_norm=1e9)
    _run_case("mixed-clip", cast_bf16=True, max_norm=10.0)
    _run_case("mixed-noop", cast_bf16=True, max_norm=1e9)
    print("All precision tests passed.")
