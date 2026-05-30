import paddle
import contextlib

paddle.set_flags({"FLAGS_share_tensor_for_grad_tensor_holder": True})


@contextlib.contextmanager
def event_duration(name="", repeat=1):
    for i in range(10):
        paddle.randn([1024, 1024, 1024])
    begin = paddle.device.Event(enable_timing=True)
    begin.record()
    try:
        yield
    finally:
        end = paddle.device.Event(enable_timing=True)
        end.record()
        end.synchronize()
        print(f"{name}: {begin.elapsed_time(end) / repeat:.3f} ms")


def detach_with_grad(x):
    x = x.detach()
    x.stop_gradient = False
    return x


conv2d_nchw = paddle.nn.Conv2D(
    in_channels=128,
    out_channels=128,
    kernel_size=3,
    padding=1,
    groups=128,
    data_format="NCHW",
    dtype="bfloat16",
)
x = paddle.randn([32, 128, 256, 256], dtype="bfloat16")
out = conv2d_nchw(detach_with_grad(x))
out_grad = paddle.randn_like(out)
out.backward(out_grad)

with event_duration("conv2d_nchw", 10):
    for i in range(10):
        out = conv2d_nchw(detach_with_grad(x))
        out.backward(out_grad)
