"""
USE_TRITON_FUSION=1 ENABLE_NVPROF=1 \
nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi -x true --cuda-event-trace=false \
python model.py
"""
import os
import paddle
from paddle.autograd import PyLayer

ENABLE_NVPROF = paddle.utils.strtobool(os.getenv("ENABLE_NVPROF", "0"))


def _unwrap_output(tensors, clone_leaf=False):
    output = []
    for tensor in tensors:
        if clone_leaf and tensor.is_leaf:
            print("[warn] clone leaf tensor:", list(tensor.shape), tensor.dtype.name.lower())
            tensor = tensor.clone()
        output.append(tensor)
    return output[0] if len(output) == 1 else output


class _NvtxBegin(PyLayer):
    @staticmethod
    def forward(ctx, name, *tensors):
        nvtx_push(name + "_fw")
        _nvtx_stack.append(name)
        ctx.name = name
        return _unwrap_output(tensors, clone_leaf=True)

    @staticmethod
    def backward(ctx, *grads):
        nvtx_pop()
        assert _nvtx_stack, f"nvtx not closed: {ctx.name}"
        name = _nvtx_stack.pop()
        assert name == ctx.name, f"nvtx not match: {name} and {ctx.name}"
        return _unwrap_output(grads)


class _NvtxEnd(PyLayer):
    @staticmethod
    def forward(ctx, name, *tensors):
        nvtx_pop()
        ctx.name = name
        assert _nvtx_stack, f"nvtx not closed: {ctx.name}"
        name = _nvtx_stack.pop()
        assert name == ctx.name, f"nvtx not match: {name} and {ctx.name}"
        return _unwrap_output(tensors)

    @staticmethod
    def backward(ctx, *grads):
        nvtx_push(ctx.name + "_bw")
        _nvtx_stack.append(ctx.name)
        return _unwrap_output(grads)


_nvtx_stack = []

if ENABLE_NVPROF:
    nvtx_start = paddle.base.core.nvprof_start
    nvtx_stop = paddle.base.core.nvprof_stop
    nvtx_push = paddle.base.core.nvprof_nvtx_push
    nvtx_pop = paddle.base.core.nvprof_nvtx_pop
    nvtx_begin = _NvtxBegin.apply
    nvtx_end = _NvtxEnd.apply

else:
    nvtx_start = str
    nvtx_stop = str
    nvtx_push = str
    nvtx_pop = str
    nvtx_begin = lambda name, *tensors: _unwrap_output(tensors)
    nvtx_end = nvtx_begin
