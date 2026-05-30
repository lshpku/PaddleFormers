"""
Triton NHWC depthwise conv2d k=3 p=1 specialized forward and backward kernels.
"""

import paddle

paddle.enable_compat(scope={"triton"})

import triton
import triton.language as tl


@triton.jit
def _load_nhwc_vec(ptr, n, h, w, c, height: tl.constexpr, width: tl.constexpr, channels: tl.constexpr):
    mask = (h >= 0) & (h < height) & (w >= 0) & (w < width)
    offsets = ((n * height + h) * width + w) * channels + c
    return tl.load(ptr + offsets, mask=mask, other=0.0).to(tl.float32)


@triton.jit
def fused_depthwise_conv_k3p1_scheduled_block_k4(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_elements: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    channels: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """NHWC depthwise conv2d forward specialized for output tile 4x4."""
    pid_hw = tl.program_id(0)
    pid_nc = tl.program_id(1)

    num_w_tiles = width // 4
    tile_h = pid_hw // num_w_tiles
    tile_w = pid_hw - tile_h * num_w_tiles
    n = pid_nc // tl.cdiv(channels, BLOCK_C)
    c_base = (pid_nc - n * tl.cdiv(channels, BLOCK_C)) * BLOCK_C
    c = c_base + tl.arange(0, BLOCK_C)

    h0 = tile_h * 4 - 1
    w0 = tile_w * 4 - 1
    out_h = tile_h * 4
    out_w = tile_w * 4

    bias = tl.load(bias_ptr + c).to(tl.float32)
    wt = c * 9
    k00 = tl.load(weight_ptr + wt + 0).to(tl.float32)
    k01 = tl.load(weight_ptr + wt + 1).to(tl.float32)
    k02 = tl.load(weight_ptr + wt + 2).to(tl.float32)
    k10 = tl.load(weight_ptr + wt + 3).to(tl.float32)
    k11 = tl.load(weight_ptr + wt + 4).to(tl.float32)
    k12 = tl.load(weight_ptr + wt + 5).to(tl.float32)
    k20 = tl.load(weight_ptr + wt + 6).to(tl.float32)
    k21 = tl.load(weight_ptr + wt + 7).to(tl.float32)
    k22 = tl.load(weight_ptr + wt + 8).to(tl.float32)

    x00 = _load_nhwc_vec(x_ptr, n, h0 + 0, w0 + 0, c, height, width, channels)
    x01 = _load_nhwc_vec(x_ptr, n, h0 + 0, w0 + 1, c, height, width, channels)
    x02 = _load_nhwc_vec(x_ptr, n, h0 + 0, w0 + 2, c, height, width, channels)
    x03 = _load_nhwc_vec(x_ptr, n, h0 + 0, w0 + 3, c, height, width, channels)
    x04 = _load_nhwc_vec(x_ptr, n, h0 + 0, w0 + 4, c, height, width, channels)
    x05 = _load_nhwc_vec(x_ptr, n, h0 + 0, w0 + 5, c, height, width, channels)
    x10 = _load_nhwc_vec(x_ptr, n, h0 + 1, w0 + 0, c, height, width, channels)
    x11 = _load_nhwc_vec(x_ptr, n, h0 + 1, w0 + 1, c, height, width, channels)
    x12 = _load_nhwc_vec(x_ptr, n, h0 + 1, w0 + 2, c, height, width, channels)
    x13 = _load_nhwc_vec(x_ptr, n, h0 + 1, w0 + 3, c, height, width, channels)
    x14 = _load_nhwc_vec(x_ptr, n, h0 + 1, w0 + 4, c, height, width, channels)
    x15 = _load_nhwc_vec(x_ptr, n, h0 + 1, w0 + 5, c, height, width, channels)
    x20 = _load_nhwc_vec(x_ptr, n, h0 + 2, w0 + 0, c, height, width, channels)
    x21 = _load_nhwc_vec(x_ptr, n, h0 + 2, w0 + 1, c, height, width, channels)
    x22 = _load_nhwc_vec(x_ptr, n, h0 + 2, w0 + 2, c, height, width, channels)
    x23 = _load_nhwc_vec(x_ptr, n, h0 + 2, w0 + 3, c, height, width, channels)
    x24 = _load_nhwc_vec(x_ptr, n, h0 + 2, w0 + 4, c, height, width, channels)
    x25 = _load_nhwc_vec(x_ptr, n, h0 + 2, w0 + 5, c, height, width, channels)

    acc0 = bias + x00 * k00 + x01 * k01 + x02 * k02 + x10 * k10 + x11 * k11 + x12 * k12 + x20 * k20 + x21 * k21 + x22 * k22
    acc1 = bias + x01 * k00 + x02 * k01 + x03 * k02 + x11 * k10 + x12 * k11 + x13 * k12 + x21 * k20 + x22 * k21 + x23 * k22
    acc2 = bias + x02 * k00 + x03 * k01 + x04 * k02 + x12 * k10 + x13 * k11 + x14 * k12 + x22 * k20 + x23 * k21 + x24 * k22
    acc3 = bias + x03 * k00 + x04 * k01 + x05 * k02 + x13 * k10 + x14 * k11 + x15 * k12 + x23 * k20 + x24 * k21 + x25 * k22
    tl.store(out_ptr + ((n * height + out_h + 0) * width + out_w + 0) * channels + c, acc0)
    tl.store(out_ptr + ((n * height + out_h + 0) * width + out_w + 1) * channels + c, acc1)
    tl.store(out_ptr + ((n * height + out_h + 0) * width + out_w + 2) * channels + c, acc2)
    tl.store(out_ptr + ((n * height + out_h + 0) * width + out_w + 3) * channels + c, acc3)

    x30 = _load_nhwc_vec(x_ptr, n, h0 + 3, w0 + 0, c, height, width, channels)
    x31 = _load_nhwc_vec(x_ptr, n, h0 + 3, w0 + 1, c, height, width, channels)
    x32 = _load_nhwc_vec(x_ptr, n, h0 + 3, w0 + 2, c, height, width, channels)
    x33 = _load_nhwc_vec(x_ptr, n, h0 + 3, w0 + 3, c, height, width, channels)
    x34 = _load_nhwc_vec(x_ptr, n, h0 + 3, w0 + 4, c, height, width, channels)
    x35 = _load_nhwc_vec(x_ptr, n, h0 + 3, w0 + 5, c, height, width, channels)

    acc0 = bias + x10 * k00 + x11 * k01 + x12 * k02 + x20 * k10 + x21 * k11 + x22 * k12 + x30 * k20 + x31 * k21 + x32 * k22
    acc1 = bias + x11 * k00 + x12 * k01 + x13 * k02 + x21 * k10 + x22 * k11 + x23 * k12 + x31 * k20 + x32 * k21 + x33 * k22
    acc2 = bias + x12 * k00 + x13 * k01 + x14 * k02 + x22 * k10 + x23 * k11 + x24 * k12 + x32 * k20 + x33 * k21 + x34 * k22
    acc3 = bias + x13 * k00 + x14 * k01 + x15 * k02 + x23 * k10 + x24 * k11 + x25 * k12 + x33 * k20 + x34 * k21 + x35 * k22
    tl.store(out_ptr + ((n * height + out_h + 1) * width + out_w + 0) * channels + c, acc0)
    tl.store(out_ptr + ((n * height + out_h + 1) * width + out_w + 1) * channels + c, acc1)
    tl.store(out_ptr + ((n * height + out_h + 1) * width + out_w + 2) * channels + c, acc2)
    tl.store(out_ptr + ((n * height + out_h + 1) * width + out_w + 3) * channels + c, acc3)

    x40 = _load_nhwc_vec(x_ptr, n, h0 + 4, w0 + 0, c, height, width, channels)
    x41 = _load_nhwc_vec(x_ptr, n, h0 + 4, w0 + 1, c, height, width, channels)
    x42 = _load_nhwc_vec(x_ptr, n, h0 + 4, w0 + 2, c, height, width, channels)
    x43 = _load_nhwc_vec(x_ptr, n, h0 + 4, w0 + 3, c, height, width, channels)
    x44 = _load_nhwc_vec(x_ptr, n, h0 + 4, w0 + 4, c, height, width, channels)
    x45 = _load_nhwc_vec(x_ptr, n, h0 + 4, w0 + 5, c, height, width, channels)

    acc0 = bias + x20 * k00 + x21 * k01 + x22 * k02 + x30 * k10 + x31 * k11 + x32 * k12 + x40 * k20 + x41 * k21 + x42 * k22
    acc1 = bias + x21 * k00 + x22 * k01 + x23 * k02 + x31 * k10 + x32 * k11 + x33 * k12 + x41 * k20 + x42 * k21 + x43 * k22
    acc2 = bias + x22 * k00 + x23 * k01 + x24 * k02 + x32 * k10 + x33 * k11 + x34 * k12 + x42 * k20 + x43 * k21 + x44 * k22
    acc3 = bias + x23 * k00 + x24 * k01 + x25 * k02 + x33 * k10 + x34 * k11 + x35 * k12 + x43 * k20 + x44 * k21 + x45 * k22
    tl.store(out_ptr + ((n * height + out_h + 2) * width + out_w + 0) * channels + c, acc0)
    tl.store(out_ptr + ((n * height + out_h + 2) * width + out_w + 1) * channels + c, acc1)
    tl.store(out_ptr + ((n * height + out_h + 2) * width + out_w + 2) * channels + c, acc2)
    tl.store(out_ptr + ((n * height + out_h + 2) * width + out_w + 3) * channels + c, acc3)

    x50 = _load_nhwc_vec(x_ptr, n, h0 + 5, w0 + 0, c, height, width, channels)
    x51 = _load_nhwc_vec(x_ptr, n, h0 + 5, w0 + 1, c, height, width, channels)
    x52 = _load_nhwc_vec(x_ptr, n, h0 + 5, w0 + 2, c, height, width, channels)
    x53 = _load_nhwc_vec(x_ptr, n, h0 + 5, w0 + 3, c, height, width, channels)
    x54 = _load_nhwc_vec(x_ptr, n, h0 + 5, w0 + 4, c, height, width, channels)
    x55 = _load_nhwc_vec(x_ptr, n, h0 + 5, w0 + 5, c, height, width, channels)

    acc0 = bias + x30 * k00 + x31 * k01 + x32 * k02 + x40 * k10 + x41 * k11 + x42 * k12 + x50 * k20 + x51 * k21 + x52 * k22
    acc1 = bias + x31 * k00 + x32 * k01 + x33 * k02 + x41 * k10 + x42 * k11 + x43 * k12 + x51 * k20 + x52 * k21 + x53 * k22
    acc2 = bias + x32 * k00 + x33 * k01 + x34 * k02 + x42 * k10 + x43 * k11 + x44 * k12 + x52 * k20 + x53 * k21 + x54 * k22
    acc3 = bias + x33 * k00 + x34 * k01 + x35 * k02 + x43 * k10 + x44 * k11 + x45 * k12 + x53 * k20 + x54 * k21 + x55 * k22
    tl.store(out_ptr + ((n * height + out_h + 3) * width + out_w + 0) * channels + c, acc0)
    tl.store(out_ptr + ((n * height + out_h + 3) * width + out_w + 1) * channels + c, acc1)
    tl.store(out_ptr + ((n * height + out_h + 3) * width + out_w + 2) * channels + c, acc2)
    tl.store(out_ptr + ((n * height + out_h + 3) * width + out_w + 3) * channels + c, acc3)


@triton.jit
def fused_depthwise_conv_k3p1_dx_scheduled_block_k4(
    out_grad_ptr,
    weight_ptr,
    x_grad_ptr,
    n_elements: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    channels: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """NHWC depthwise conv2d input-grad specialized for output tile 4x4."""
    pid_hw = tl.program_id(0)
    pid_nc = tl.program_id(1)

    num_w_tiles = width // 4
    tile_h = pid_hw // num_w_tiles
    tile_w = pid_hw - tile_h * num_w_tiles
    n = pid_nc // tl.cdiv(channels, BLOCK_C)
    c_base = (pid_nc - n * tl.cdiv(channels, BLOCK_C)) * BLOCK_C
    c = c_base + tl.arange(0, BLOCK_C)

    h0 = tile_h * 4 - 1
    w0 = tile_w * 4 - 1
    out_h = tile_h * 4
    out_w = tile_w * 4

    wt = c * 9
    k00 = tl.load(weight_ptr + wt + 8).to(tl.float32)
    k01 = tl.load(weight_ptr + wt + 7).to(tl.float32)
    k02 = tl.load(weight_ptr + wt + 6).to(tl.float32)
    k10 = tl.load(weight_ptr + wt + 5).to(tl.float32)
    k11 = tl.load(weight_ptr + wt + 4).to(tl.float32)
    k12 = tl.load(weight_ptr + wt + 3).to(tl.float32)
    k20 = tl.load(weight_ptr + wt + 2).to(tl.float32)
    k21 = tl.load(weight_ptr + wt + 1).to(tl.float32)
    k22 = tl.load(weight_ptr + wt + 0).to(tl.float32)

    x00 = _load_nhwc_vec(out_grad_ptr, n, h0 + 0, w0 + 0, c, height, width, channels)
    x01 = _load_nhwc_vec(out_grad_ptr, n, h0 + 0, w0 + 1, c, height, width, channels)
    x02 = _load_nhwc_vec(out_grad_ptr, n, h0 + 0, w0 + 2, c, height, width, channels)
    x03 = _load_nhwc_vec(out_grad_ptr, n, h0 + 0, w0 + 3, c, height, width, channels)
    x04 = _load_nhwc_vec(out_grad_ptr, n, h0 + 0, w0 + 4, c, height, width, channels)
    x05 = _load_nhwc_vec(out_grad_ptr, n, h0 + 0, w0 + 5, c, height, width, channels)
    x10 = _load_nhwc_vec(out_grad_ptr, n, h0 + 1, w0 + 0, c, height, width, channels)
    x11 = _load_nhwc_vec(out_grad_ptr, n, h0 + 1, w0 + 1, c, height, width, channels)
    x12 = _load_nhwc_vec(out_grad_ptr, n, h0 + 1, w0 + 2, c, height, width, channels)
    x13 = _load_nhwc_vec(out_grad_ptr, n, h0 + 1, w0 + 3, c, height, width, channels)
    x14 = _load_nhwc_vec(out_grad_ptr, n, h0 + 1, w0 + 4, c, height, width, channels)
    x15 = _load_nhwc_vec(out_grad_ptr, n, h0 + 1, w0 + 5, c, height, width, channels)
    x20 = _load_nhwc_vec(out_grad_ptr, n, h0 + 2, w0 + 0, c, height, width, channels)
    x21 = _load_nhwc_vec(out_grad_ptr, n, h0 + 2, w0 + 1, c, height, width, channels)
    x22 = _load_nhwc_vec(out_grad_ptr, n, h0 + 2, w0 + 2, c, height, width, channels)
    x23 = _load_nhwc_vec(out_grad_ptr, n, h0 + 2, w0 + 3, c, height, width, channels)
    x24 = _load_nhwc_vec(out_grad_ptr, n, h0 + 2, w0 + 4, c, height, width, channels)
    x25 = _load_nhwc_vec(out_grad_ptr, n, h0 + 2, w0 + 5, c, height, width, channels)

    acc0 = x00 * k00 + x01 * k01 + x02 * k02 + x10 * k10 + x11 * k11 + x12 * k12 + x20 * k20 + x21 * k21 + x22 * k22
    acc1 = x01 * k00 + x02 * k01 + x03 * k02 + x11 * k10 + x12 * k11 + x13 * k12 + x21 * k20 + x22 * k21 + x23 * k22
    acc2 = x02 * k00 + x03 * k01 + x04 * k02 + x12 * k10 + x13 * k11 + x14 * k12 + x22 * k20 + x23 * k21 + x24 * k22
    acc3 = x03 * k00 + x04 * k01 + x05 * k02 + x13 * k10 + x14 * k11 + x15 * k12 + x23 * k20 + x24 * k21 + x25 * k22
    tl.store(x_grad_ptr + ((n * height + out_h + 0) * width + out_w + 0) * channels + c, acc0)
    tl.store(x_grad_ptr + ((n * height + out_h + 0) * width + out_w + 1) * channels + c, acc1)
    tl.store(x_grad_ptr + ((n * height + out_h + 0) * width + out_w + 2) * channels + c, acc2)
    tl.store(x_grad_ptr + ((n * height + out_h + 0) * width + out_w + 3) * channels + c, acc3)

    x30 = _load_nhwc_vec(out_grad_ptr, n, h0 + 3, w0 + 0, c, height, width, channels)
    x31 = _load_nhwc_vec(out_grad_ptr, n, h0 + 3, w0 + 1, c, height, width, channels)
    x32 = _load_nhwc_vec(out_grad_ptr, n, h0 + 3, w0 + 2, c, height, width, channels)
    x33 = _load_nhwc_vec(out_grad_ptr, n, h0 + 3, w0 + 3, c, height, width, channels)
    x34 = _load_nhwc_vec(out_grad_ptr, n, h0 + 3, w0 + 4, c, height, width, channels)
    x35 = _load_nhwc_vec(out_grad_ptr, n, h0 + 3, w0 + 5, c, height, width, channels)

    acc0 = x10 * k00 + x11 * k01 + x12 * k02 + x20 * k10 + x21 * k11 + x22 * k12 + x30 * k20 + x31 * k21 + x32 * k22
    acc1 = x11 * k00 + x12 * k01 + x13 * k02 + x21 * k10 + x22 * k11 + x23 * k12 + x31 * k20 + x32 * k21 + x33 * k22
    acc2 = x12 * k00 + x13 * k01 + x14 * k02 + x22 * k10 + x23 * k11 + x24 * k12 + x32 * k20 + x33 * k21 + x34 * k22
    acc3 = x13 * k00 + x14 * k01 + x15 * k02 + x23 * k10 + x24 * k11 + x25 * k12 + x33 * k20 + x34 * k21 + x35 * k22
    tl.store(x_grad_ptr + ((n * height + out_h + 1) * width + out_w + 0) * channels + c, acc0)
    tl.store(x_grad_ptr + ((n * height + out_h + 1) * width + out_w + 1) * channels + c, acc1)
    tl.store(x_grad_ptr + ((n * height + out_h + 1) * width + out_w + 2) * channels + c, acc2)
    tl.store(x_grad_ptr + ((n * height + out_h + 1) * width + out_w + 3) * channels + c, acc3)

    x40 = _load_nhwc_vec(out_grad_ptr, n, h0 + 4, w0 + 0, c, height, width, channels)
    x41 = _load_nhwc_vec(out_grad_ptr, n, h0 + 4, w0 + 1, c, height, width, channels)
    x42 = _load_nhwc_vec(out_grad_ptr, n, h0 + 4, w0 + 2, c, height, width, channels)
    x43 = _load_nhwc_vec(out_grad_ptr, n, h0 + 4, w0 + 3, c, height, width, channels)
    x44 = _load_nhwc_vec(out_grad_ptr, n, h0 + 4, w0 + 4, c, height, width, channels)
    x45 = _load_nhwc_vec(out_grad_ptr, n, h0 + 4, w0 + 5, c, height, width, channels)

    acc0 = x20 * k00 + x21 * k01 + x22 * k02 + x30 * k10 + x31 * k11 + x32 * k12 + x40 * k20 + x41 * k21 + x42 * k22
    acc1 = x21 * k00 + x22 * k01 + x23 * k02 + x31 * k10 + x32 * k11 + x33 * k12 + x41 * k20 + x42 * k21 + x43 * k22
    acc2 = x22 * k00 + x23 * k01 + x24 * k02 + x32 * k10 + x33 * k11 + x34 * k12 + x42 * k20 + x43 * k21 + x44 * k22
    acc3 = x23 * k00 + x24 * k01 + x25 * k02 + x33 * k10 + x34 * k11 + x35 * k12 + x43 * k20 + x44 * k21 + x45 * k22
    tl.store(x_grad_ptr + ((n * height + out_h + 2) * width + out_w + 0) * channels + c, acc0)
    tl.store(x_grad_ptr + ((n * height + out_h + 2) * width + out_w + 1) * channels + c, acc1)
    tl.store(x_grad_ptr + ((n * height + out_h + 2) * width + out_w + 2) * channels + c, acc2)
    tl.store(x_grad_ptr + ((n * height + out_h + 2) * width + out_w + 3) * channels + c, acc3)

    x50 = _load_nhwc_vec(out_grad_ptr, n, h0 + 5, w0 + 0, c, height, width, channels)
    x51 = _load_nhwc_vec(out_grad_ptr, n, h0 + 5, w0 + 1, c, height, width, channels)
    x52 = _load_nhwc_vec(out_grad_ptr, n, h0 + 5, w0 + 2, c, height, width, channels)
    x53 = _load_nhwc_vec(out_grad_ptr, n, h0 + 5, w0 + 3, c, height, width, channels)
    x54 = _load_nhwc_vec(out_grad_ptr, n, h0 + 5, w0 + 4, c, height, width, channels)
    x55 = _load_nhwc_vec(out_grad_ptr, n, h0 + 5, w0 + 5, c, height, width, channels)

    acc0 = x30 * k00 + x31 * k01 + x32 * k02 + x40 * k10 + x41 * k11 + x42 * k12 + x50 * k20 + x51 * k21 + x52 * k22
    acc1 = x31 * k00 + x32 * k01 + x33 * k02 + x41 * k10 + x42 * k11 + x43 * k12 + x51 * k20 + x52 * k21 + x53 * k22
    acc2 = x32 * k00 + x33 * k01 + x34 * k02 + x42 * k10 + x43 * k11 + x44 * k12 + x52 * k20 + x53 * k21 + x54 * k22
    acc3 = x33 * k00 + x34 * k01 + x35 * k02 + x43 * k10 + x44 * k11 + x45 * k12 + x53 * k20 + x54 * k21 + x55 * k22
    tl.store(x_grad_ptr + ((n * height + out_h + 3) * width + out_w + 0) * channels + c, acc0)
    tl.store(x_grad_ptr + ((n * height + out_h + 3) * width + out_w + 1) * channels + c, acc1)
    tl.store(x_grad_ptr + ((n * height + out_h + 3) * width + out_w + 2) * channels + c, acc2)
    tl.store(x_grad_ptr + ((n * height + out_h + 3) * width + out_w + 3) * channels + c, acc3)


@triton.jit
def fused_depthwise_conv_k3p1_dw_partial_scan2(
    x_ptr,
    out_grad_ptr,
    partial_ptr,
    n_elements: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    channels: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Stage 1: scan two adjacent rows and emit partial dw/dbias [N * H/2, 10, C]."""
    pid_nh2 = tl.program_id(0)
    pid_c = tl.program_id(1)

    h_pairs = height // 2
    n = pid_nh2 // h_pairs
    h_pair = pid_nh2 - n * h_pairs
    h = h_pair * 2
    c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    base = (pid_nh2 * 10 * channels) + c

    acc00 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc01 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc02 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc10 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc11 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc12 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc20 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc21 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc22 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc_bias = tl.zeros((BLOCK_C,), dtype=tl.float32)

    x0_l = _load_nhwc_vec(x_ptr, n, h - 1, -1, c, height, width, channels)
    x0_m = _load_nhwc_vec(x_ptr, n, h - 1, 0, c, height, width, channels)
    x1_l = _load_nhwc_vec(x_ptr, n, h + 0, -1, c, height, width, channels)
    x1_m = _load_nhwc_vec(x_ptr, n, h + 0, 0, c, height, width, channels)
    x2_l = _load_nhwc_vec(x_ptr, n, h + 1, -1, c, height, width, channels)
    x2_m = _load_nhwc_vec(x_ptr, n, h + 1, 0, c, height, width, channels)
    x3_l = _load_nhwc_vec(x_ptr, n, h + 2, -1, c, height, width, channels)
    x3_m = _load_nhwc_vec(x_ptr, n, h + 2, 0, c, height, width, channels)

    for w in tl.range(0, width):
        x0_r = _load_nhwc_vec(x_ptr, n, h - 1, w + 1, c, height, width, channels)
        x1_r = _load_nhwc_vec(x_ptr, n, h + 0, w + 1, c, height, width, channels)
        x2_r = _load_nhwc_vec(x_ptr, n, h + 1, w + 1, c, height, width, channels)
        x3_r = _load_nhwc_vec(x_ptr, n, h + 2, w + 1, c, height, width, channels)
        dy0 = _load_nhwc_vec(out_grad_ptr, n, h + 0, w, c, height, width, channels)
        dy1 = _load_nhwc_vec(out_grad_ptr, n, h + 1, w, c, height, width, channels)
        acc_bias += dy0 + dy1

        acc00 += dy0 * x0_l + dy1 * x1_l
        acc01 += dy0 * x0_m + dy1 * x1_m
        acc02 += dy0 * x0_r + dy1 * x1_r
        acc10 += dy0 * x1_l + dy1 * x2_l
        acc11 += dy0 * x1_m + dy1 * x2_m
        acc12 += dy0 * x1_r + dy1 * x2_r
        acc20 += dy0 * x2_l + dy1 * x3_l
        acc21 += dy0 * x2_m + dy1 * x3_m
        acc22 += dy0 * x2_r + dy1 * x3_r

        x0_l = x0_m
        x0_m = x0_r
        x1_l = x1_m
        x1_m = x1_r
        x2_l = x2_m
        x2_m = x2_r
        x3_l = x3_m
        x3_m = x3_r

    tl.store(partial_ptr + base + 0 * channels, acc00)
    tl.store(partial_ptr + base + 1 * channels, acc01)
    tl.store(partial_ptr + base + 2 * channels, acc02)
    tl.store(partial_ptr + base + 3 * channels, acc10)
    tl.store(partial_ptr + base + 4 * channels, acc11)
    tl.store(partial_ptr + base + 5 * channels, acc12)
    tl.store(partial_ptr + base + 6 * channels, acc20)
    tl.store(partial_ptr + base + 7 * channels, acc21)
    tl.store(partial_ptr + base + 8 * channels, acc22)
    tl.store(partial_ptr + base + 9 * channels, acc_bias)


@triton.jit
def fused_depthwise_conv_k3p1_dw_reduce(
    partial_ptr,
    weight_grad_ptr,
    num_partial_tiles: tl.constexpr,
    channels: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Reduce partial[:, 0:9, :] into weight_grad[C, 1, 3, 3]."""
    pid_k = tl.program_id(0)
    pid_c = tl.program_id(1)
    c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    m_offsets = tl.arange(0, BLOCK_M)
    acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)

    for start in range(0, num_partial_tiles, BLOCK_M):
        m = start + m_offsets
        offsets = (m[:, None] * 10 * channels) + pid_k * channels + c[None, :]
        mask = m[:, None] < num_partial_tiles
        vals = tl.load(partial_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        acc += vals

    tl.store(weight_grad_ptr + c * 9 + pid_k, tl.sum(acc, axis=0))


@triton.jit
def fused_depthwise_conv_k3p1_dbias_reduce(
    partial_ptr,
    bias_grad_ptr,
    num_partial_tiles: tl.constexpr,
    channels: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Reduce partial[:, 9, :] into bias_grad[C]."""
    pid_c = tl.program_id(0)
    c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    m_offsets = tl.arange(0, BLOCK_M)
    acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)

    for start in range(0, num_partial_tiles, BLOCK_M):
        m = start + m_offsets
        offsets = (m[:, None] * 10 * channels) + 9 * channels + c[None, :]
        mask = m[:, None] < num_partial_tiles
        vals = tl.load(partial_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        acc += vals

    tl.store(bias_grad_ptr + c, tl.sum(acc, axis=0))


def _pick_block_c(channels):
    if channels % 128 == 0:
        return 128
    if channels % 64 == 0:
        return 64
    return 32


def depthwise_conv_k3p1_forward(x, weight, bias, block_c):
    n, height, width, channels = x.shape
    assert weight.shape == [channels, 1, 3, 3]
    assert bias.shape == [channels]
    assert channels % block_c == 0 and height % 4 == 0 and width % 4 == 0

    out = paddle.empty_like(x)
    grid = ((height // 4) * (width // 4), n * (channels // block_c))
    fused_depthwise_conv_k3p1_scheduled_block_k4[grid](
        x, weight, bias, out, n, height, width, channels, BLOCK_C=block_c
    )
    return out


def depthwise_conv_k3p1_dx(out_grad, weight, block_c):
    n, height, width, channels = out_grad.shape
    assert weight.shape == [channels, 1, 3, 3]
    assert channels % block_c == 0 and height % 4 == 0 and width % 4 == 0

    x_grad = paddle.empty_like(out_grad)
    grid = ((height // 4) * (width // 4), n * (channels // block_c))
    fused_depthwise_conv_k3p1_dx_scheduled_block_k4[grid](
        out_grad, weight, x_grad, n, height, width, channels, BLOCK_C=block_c
    )
    return x_grad


def depthwise_conv_k3p1_dw_dbias(x, out_grad, block_c):
    n, height, width, channels = x.shape
    assert x.shape == out_grad.shape
    assert height % 2 == 0 and channels % block_c == 0

    num_partial_tiles = n * (height // 2)
    partial = paddle.empty([num_partial_tiles, 10, channels], dtype="float32")

    fused_depthwise_conv_k3p1_dw_partial_scan2[(num_partial_tiles, channels // block_c)](
        x, out_grad, partial, n, height, width, channels, BLOCK_C=block_c
    )

    reduced = partial.sum(axis=0)
    weight_grad = reduced[:9].T.contiguous().reshape([channels, 1, 3, 3])
    bias_grad = reduced[9]
    return weight_grad, bias_grad


class FusedDepthwiseConvK3P1Triton(paddle.autograd.PyLayer):
    """Triton NHWC depthwise conv2d k=3 p=1 with autograd support."""

    @staticmethod
    def forward(ctx, x, weight, bias):
        """forward"""
        assert len(x.shape) == 4, f"x must be NHWC, got shape {x.shape}"
        assert len(weight.shape) == 4, f"weight must be [C, 1, 3, 3], got shape {weight.shape}"
        assert bias is not None, "bias is required"

        channels = x.shape[-1]
        block_c = _pick_block_c(channels)
        out = depthwise_conv_k3p1_forward(x, weight, bias, block_c=block_c)
        ctx.save_for_backward(x, weight)
        ctx.block_c = block_c
        return out

    @staticmethod
    def backward(ctx, out_grad):
        """backward"""
        x, weight = ctx.saved_tensor()
        x_grad = depthwise_conv_k3p1_dx(out_grad, weight, block_c=ctx.block_c)
        weight_grad, bias_grad = depthwise_conv_k3p1_dw_dbias(x, out_grad, block_c=ctx.block_c)
        return x_grad, weight_grad, bias_grad


if __name__ == "__main__":
    paddle.seed(2026)
    dtype = "bfloat16"
    n, height, width, channels = 32, 256, 256, 128

    x = paddle.randn([n, height, width, channels], dtype=dtype)
    conv = paddle.nn.Conv2D(
        in_channels=channels,
        out_channels=channels,
        kernel_size=3,
        padding=1,
        groups=channels,
        data_format="NCHW",
        dtype=dtype,
    )
    conv_ref = paddle.nn.Conv2D(
        in_channels=channels,
        out_channels=channels,
        kernel_size=3,
        padding=1,
        groups=channels,
        data_format="NCHW",
        dtype="float32",
    )
    conv_ref.weight.set_value(conv.weight.float())
    conv_ref.bias.set_value(conv.bias.float())

    x_triton = x.detach()
    x_triton.stop_gradient = False
    x_ref_nchw = x.detach().transpose([0, 3, 1, 2]).float()
    x_ref_nchw.stop_gradient = False

    actual = FusedDepthwiseConvK3P1Triton.apply(x_triton, conv.weight, conv.bias)
    expected_nchw = conv_ref(x_ref_nchw)
    expected = expected_nchw.transpose([0, 2, 3, 1])
    out_grad = paddle.randn_like(actual) * 1e-3

    actual.backward(out_grad)
    expected_nchw.backward(out_grad.transpose([0, 3, 1, 2]))

    forward_diff = (actual - expected).abs()
    dx_diff = (x_triton.grad - x_ref_nchw.grad.transpose([0, 2, 3, 1])).abs()
    dw_diff = (conv.weight.grad - conv_ref.weight.grad).abs()
    dbias_diff = (conv.bias.grad - conv_ref.bias.grad).abs()

    print(f"forward max_diff={forward_diff.max().item():.6f}, mean_diff={forward_diff.mean().item():.6f}")
    print(f"dx max_diff={dx_diff.max().item():.6f}, mean_diff={dx_diff.mean().item():.6f}")
    print(f"dw max_diff={dw_diff.max().item():.6f}, mean_diff={dw_diff.mean().item():.6f}")
    print(f"dbias max_diff={dbias_diff.max().item():.6f}, mean_diff={dbias_diff.mean().item():.6f}")

    assert forward_diff.max().item() < 0.05
    assert dx_diff.max().item() < 0.05
    assert dw_diff.max().item() < 0.05
    assert dbias_diff.max().item() < 0.05
