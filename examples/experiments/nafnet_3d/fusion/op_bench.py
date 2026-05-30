import contextlib

import paddle

from depthwise_conv import (
    depthwise_conv_k3p1_dx,
    depthwise_conv_k3p1_dw_dbias,
    depthwise_conv_k3p1_forward,
)

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


def pick_block_c(channels):
    if channels % 128 == 0:
        return 128
    if channels % 64 == 0:
        return 64
    return 32


def build_cases(max_cases=None):
    cases = []
    num_levels = 5
    for batch in (16, 32, 64):
        for base_channels in (32, 64):
            height = 256
            width = 256
            channels = base_channels
            for layer in range(num_levels):
                cases.append((batch, height, width, channels, layer))
                height //= 2
                width //= 2
                channels *= 2
                if max_cases and len(cases) >= max_cases:
                    return cases
    return cases


def run_paddle_case(batch, height, width, channels, repeat):
    conv = paddle.nn.Conv2D(
        in_channels=channels,
        out_channels=channels,
        kernel_size=3,
        padding=1,
        groups=channels,
        data_format="NCHW",
        dtype="bfloat16",
    )
    x = paddle.randn([batch, channels, height, width], dtype="bfloat16")
    out = conv(detach_with_grad(x))
    out_grad = paddle.randn_like(out)
    out.backward(out_grad)

    with event_duration(f"paddle_nchw N={batch} H={height} W={width} C={channels}", repeat):
        for i in range(repeat):
            out = conv(detach_with_grad(x))
            out.backward(out_grad)


def run_triton_case(batch, height, width, channels, repeat):
    block_c = pick_block_c(channels)
    weight = paddle.randn([channels, 1, 3, 3], dtype="bfloat16")
    bias = paddle.randn([channels], dtype="bfloat16")
    x = paddle.randn([batch, height, width, channels], dtype="bfloat16")
    out = depthwise_conv_k3p1_forward(x, weight, bias, block_c=block_c)
    out_grad = paddle.randn_like(out)
    depthwise_conv_k3p1_dx(out_grad, weight, block_c=block_c)
    depthwise_conv_k3p1_dw_dbias(x, out_grad, block_c=block_c)

    name = f"triton_nhwc N={batch} H={height} W={width} C={channels} block_c={block_c}"
    with event_duration(name, repeat):
        for i in range(repeat):
            out = depthwise_conv_k3p1_forward(x, weight, bias, block_c=block_c)
            depthwise_conv_k3p1_dx(out_grad, weight, block_c=block_c)
            depthwise_conv_k3p1_dw_dbias(x, out_grad, block_c=block_c)


if __name__ == "__main__":
    paddle.seed(2026)
    repeat = 10
    for case_id, (batch, height, width, channels, layer) in enumerate(build_cases()):
        print(f"\ncase {case_id}: layer={layer}, N={batch}, H={height}, W={width}, C={channels}")
        run_paddle_case(batch, height, width, channels, repeat)
        run_triton_case(batch, height, width, channels, repeat)
