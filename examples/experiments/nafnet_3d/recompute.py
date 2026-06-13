import paddle
from paddle.framework import core
from paddle.distributed.fleet.recompute.recompute import detach_variable
from nvprof import nvtx_push, nvtx_pop


class RecomputeWithoutOutputFunction(paddle.autograd.PyLayer):
    """Autograd wrapper for RecomputeWithoutOutput."""

    @staticmethod
    def forward(ctx, run_function, owner, *args):
        with paddle.no_grad():
            outputs = run_function(*args)
        ctx.save_for_backward(*detach_variable(args))
        owner.ctx = ctx
        return outputs

    @staticmethod
    def backward(ctx, *output_grads):
        inputs = ctx.inputs
        outputs = ctx.outputs

        flag = "FLAGS_share_tensor_for_grad_tensor_holder"
        paddle.set_flags({flag: True})
        nvtx_push("backward")

        with paddle.amp.auto_cast(enable=False):
            paddle.autograd.backward(outputs, output_grads)

        nvtx_pop()
        nvtx_pop()  # _recompute begin
        paddle.set_flags({flag: False})

        ctx.outputs = None
        ctx.inputs = None
        grads = tuple(
            inp.grad for inp in inputs if isinstance(inp, paddle.Tensor)
        )
        return grads


class RecomputeWithoutOutput:
    """Manage forward-output discard and backward-time recompute to save memory."""

    def __init__(self, name):
        self.name = name
        self.run_function = None
        self.ctx = None
        self.outputs = None

    def recompute(self, run_function, *args):
        self.run_function = run_function

        tracer = paddle.base.framework._dygraph_tracer()
        self.is_fw_autocast = tracer._amp_level > core.AmpLevel.O0
        self.amp_white_list, self.amp_black_list = tracer._get_amp_op_list()

        outputs = RecomputeWithoutOutputFunction.apply(
            run_function, self, *args
        )
        self.outputs = outputs if isinstance(outputs, tuple) else (outputs,)
        return outputs

    def _recompute(self, grad):
        nvtx_push(self.name + "_bw")
        inputs = detach_variable(self.ctx.saved_tensor())

        with paddle.amp.auto_cast(
            enable=self.is_fw_autocast,
            custom_white_list=self.amp_white_list,
            custom_black_list=self.amp_black_list,
            level="O2",
            dtype="bfloat16",
        ):
            nvtx_push("forward")
            outputs = self.run_function(*inputs)
            nvtx_pop()

        if isinstance(outputs, paddle.Tensor):
            outputs = (outputs,)

        for stale_output, recomputed_output in zip(self.outputs, outputs):
            recomputed_output._share_buffer_to(stale_output)

        self.ctx.inputs = inputs
        self.ctx.outputs = outputs
        self.outputs = None
        self.ctx = None
        self.run_function = None

    def discard_output_and_register_recompute(self, hook_tensor):
        for output in self.outputs:
            output._clear_data()

        if not hook_tensor.stop_gradient:
            hook_tensor.register_hook(self._recompute)
