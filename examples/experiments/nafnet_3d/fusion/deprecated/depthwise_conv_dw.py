"""
Deprecated Triton NHWC depthwise conv2d k=3 p=1 weight-grad prototype kernels.
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
def fused_depthwise_conv_k3p1_dw_partial_block_k4(
    x_ptr,
    out_grad_ptr,
    partial_ptr,
    n_elements: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    channels: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_nc = tl.program_id(1)

    num_hw_tiles = (height // 4) * (width // 4)
    num_w_tiles = width // 4
    n = pid_nc // tl.cdiv(channels, BLOCK_C)
    c_base = (pid_nc - n * tl.cdiv(channels, BLOCK_C)) * BLOCK_C
    tile_h = pid_hw // num_w_tiles
    tile_w = pid_hw - tile_h * num_w_tiles
    partial_tile = n * num_hw_tiles + pid_hw
    c = c_base + tl.arange(0, BLOCK_C)

    oh = tile_h * 4
    ow = tile_w * 4
    h0 = oh - 1
    w0 = ow - 1

    dy00 = _load_nhwc_vec(out_grad_ptr, n, oh + 0, ow + 0, c, height, width, channels)
    dy01 = _load_nhwc_vec(out_grad_ptr, n, oh + 0, ow + 1, c, height, width, channels)
    dy02 = _load_nhwc_vec(out_grad_ptr, n, oh + 0, ow + 2, c, height, width, channels)
    dy03 = _load_nhwc_vec(out_grad_ptr, n, oh + 0, ow + 3, c, height, width, channels)
    dy10 = _load_nhwc_vec(out_grad_ptr, n, oh + 1, ow + 0, c, height, width, channels)
    dy11 = _load_nhwc_vec(out_grad_ptr, n, oh + 1, ow + 1, c, height, width, channels)
    dy12 = _load_nhwc_vec(out_grad_ptr, n, oh + 1, ow + 2, c, height, width, channels)
    dy13 = _load_nhwc_vec(out_grad_ptr, n, oh + 1, ow + 3, c, height, width, channels)
    dy20 = _load_nhwc_vec(out_grad_ptr, n, oh + 2, ow + 0, c, height, width, channels)
    dy21 = _load_nhwc_vec(out_grad_ptr, n, oh + 2, ow + 1, c, height, width, channels)
    dy22 = _load_nhwc_vec(out_grad_ptr, n, oh + 2, ow + 2, c, height, width, channels)
    dy23 = _load_nhwc_vec(out_grad_ptr, n, oh + 2, ow + 3, c, height, width, channels)
    dy30 = _load_nhwc_vec(out_grad_ptr, n, oh + 3, ow + 0, c, height, width, channels)
    dy31 = _load_nhwc_vec(out_grad_ptr, n, oh + 3, ow + 1, c, height, width, channels)
    dy32 = _load_nhwc_vec(out_grad_ptr, n, oh + 3, ow + 2, c, height, width, channels)
    dy33 = _load_nhwc_vec(out_grad_ptr, n, oh + 3, ow + 3, c, height, width, channels)

    base = (partial_tile * 9 * channels) + c

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
    x30 = _load_nhwc_vec(x_ptr, n, h0 + 3, w0 + 0, c, height, width, channels)
    x31 = _load_nhwc_vec(x_ptr, n, h0 + 3, w0 + 1, c, height, width, channels)
    x32 = _load_nhwc_vec(x_ptr, n, h0 + 3, w0 + 2, c, height, width, channels)
    x33 = _load_nhwc_vec(x_ptr, n, h0 + 3, w0 + 3, c, height, width, channels)
    x34 = _load_nhwc_vec(x_ptr, n, h0 + 3, w0 + 4, c, height, width, channels)
    x35 = _load_nhwc_vec(x_ptr, n, h0 + 3, w0 + 5, c, height, width, channels)

    acc0 = dy00 * x00 + dy01 * x01 + dy02 * x02 + dy03 * x03 + dy10 * x10 + dy11 * x11 + dy12 * x12 + dy13 * x13 + dy20 * x20 + dy21 * x21 + dy22 * x22 + dy23 * x23 + dy30 * x30 + dy31 * x31 + dy32 * x32 + dy33 * x33
    acc1 = dy00 * x01 + dy01 * x02 + dy02 * x03 + dy03 * x04 + dy10 * x11 + dy11 * x12 + dy12 * x13 + dy13 * x14 + dy20 * x21 + dy21 * x22 + dy22 * x23 + dy23 * x24 + dy30 * x31 + dy31 * x32 + dy32 * x33 + dy33 * x34
    acc2 = dy00 * x02 + dy01 * x03 + dy02 * x04 + dy03 * x05 + dy10 * x12 + dy11 * x13 + dy12 * x14 + dy13 * x15 + dy20 * x22 + dy21 * x23 + dy22 * x24 + dy23 * x25 + dy30 * x32 + dy31 * x33 + dy32 * x34 + dy33 * x35
    tl.store(partial_ptr + base + 0 * channels, acc0)
    tl.store(partial_ptr + base + 1 * channels, acc1)
    tl.store(partial_ptr + base + 2 * channels, acc2)

    x40 = _load_nhwc_vec(x_ptr, n, h0 + 4, w0 + 0, c, height, width, channels)
    x41 = _load_nhwc_vec(x_ptr, n, h0 + 4, w0 + 1, c, height, width, channels)
    x42 = _load_nhwc_vec(x_ptr, n, h0 + 4, w0 + 2, c, height, width, channels)
    x43 = _load_nhwc_vec(x_ptr, n, h0 + 4, w0 + 3, c, height, width, channels)
    x44 = _load_nhwc_vec(x_ptr, n, h0 + 4, w0 + 4, c, height, width, channels)
    x45 = _load_nhwc_vec(x_ptr, n, h0 + 4, w0 + 5, c, height, width, channels)

    acc0 = dy00 * x10 + dy01 * x11 + dy02 * x12 + dy03 * x13 + dy10 * x20 + dy11 * x21 + dy12 * x22 + dy13 * x23 + dy20 * x30 + dy21 * x31 + dy22 * x32 + dy23 * x33 + dy30 * x40 + dy31 * x41 + dy32 * x42 + dy33 * x43
    acc1 = dy00 * x11 + dy01 * x12 + dy02 * x13 + dy03 * x14 + dy10 * x21 + dy11 * x22 + dy12 * x23 + dy13 * x24 + dy20 * x31 + dy21 * x32 + dy22 * x33 + dy23 * x34 + dy30 * x41 + dy31 * x42 + dy32 * x43 + dy33 * x44
    acc2 = dy00 * x12 + dy01 * x13 + dy02 * x14 + dy03 * x15 + dy10 * x22 + dy11 * x23 + dy12 * x24 + dy13 * x25 + dy20 * x32 + dy21 * x33 + dy22 * x34 + dy23 * x35 + dy30 * x42 + dy31 * x43 + dy32 * x44 + dy33 * x45
    tl.store(partial_ptr + base + 3 * channels, acc0)
    tl.store(partial_ptr + base + 4 * channels, acc1)
    tl.store(partial_ptr + base + 5 * channels, acc2)

    x50 = _load_nhwc_vec(x_ptr, n, h0 + 5, w0 + 0, c, height, width, channels)
    x51 = _load_nhwc_vec(x_ptr, n, h0 + 5, w0 + 1, c, height, width, channels)
    x52 = _load_nhwc_vec(x_ptr, n, h0 + 5, w0 + 2, c, height, width, channels)
    x53 = _load_nhwc_vec(x_ptr, n, h0 + 5, w0 + 3, c, height, width, channels)
    x54 = _load_nhwc_vec(x_ptr, n, h0 + 5, w0 + 4, c, height, width, channels)
    x55 = _load_nhwc_vec(x_ptr, n, h0 + 5, w0 + 5, c, height, width, channels)

    acc0 = dy00 * x20 + dy01 * x21 + dy02 * x22 + dy03 * x23 + dy10 * x30 + dy11 * x31 + dy12 * x32 + dy13 * x33 + dy20 * x40 + dy21 * x41 + dy22 * x42 + dy23 * x43 + dy30 * x50 + dy31 * x51 + dy32 * x52 + dy33 * x53
    acc1 = dy00 * x21 + dy01 * x22 + dy02 * x23 + dy03 * x24 + dy10 * x31 + dy11 * x32 + dy12 * x33 + dy13 * x34 + dy20 * x41 + dy21 * x42 + dy22 * x43 + dy23 * x44 + dy30 * x51 + dy31 * x52 + dy32 * x53 + dy33 * x54
    acc2 = dy00 * x22 + dy01 * x23 + dy02 * x24 + dy03 * x25 + dy10 * x32 + dy11 * x33 + dy12 * x34 + dy13 * x35 + dy20 * x42 + dy21 * x43 + dy22 * x44 + dy23 * x45 + dy30 * x52 + dy31 * x53 + dy32 * x54 + dy33 * x55
    tl.store(partial_ptr + base + 6 * channels, acc0)
    tl.store(partial_ptr + base + 7 * channels, acc1)
    tl.store(partial_ptr + base + 8 * channels, acc2)


@triton.jit
def fused_depthwise_conv_k3p1_dw_partial_w_scan(
    x_ptr,
    out_grad_ptr,
    partial_ptr,
    n_elements: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    channels: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_nh = tl.program_id(0)
    pid_c = tl.program_id(1)

    n = pid_nh // height
    h = pid_nh - n * height
    c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    base = (pid_nh * 9 * channels) + c

    acc00 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc01 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc02 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc10 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc11 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc12 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc20 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc21 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    acc22 = tl.zeros((BLOCK_C,), dtype=tl.float32)

    x0_l = _load_nhwc_vec(x_ptr, n, h - 1, -1, c, height, width, channels)
    x0_m = _load_nhwc_vec(x_ptr, n, h - 1, 0, c, height, width, channels)
    x1_l = _load_nhwc_vec(x_ptr, n, h + 0, -1, c, height, width, channels)
    x1_m = _load_nhwc_vec(x_ptr, n, h + 0, 0, c, height, width, channels)
    x2_l = _load_nhwc_vec(x_ptr, n, h + 1, -1, c, height, width, channels)
    x2_m = _load_nhwc_vec(x_ptr, n, h + 1, 0, c, height, width, channels)

    for w in tl.range(0, width):
        x0_r = _load_nhwc_vec(x_ptr, n, h - 1, w + 1, c, height, width, channels)
        x1_r = _load_nhwc_vec(x_ptr, n, h + 0, w + 1, c, height, width, channels)
        x2_r = _load_nhwc_vec(x_ptr, n, h + 1, w + 1, c, height, width, channels)
        dy = _load_nhwc_vec(out_grad_ptr, n, h, w, c, height, width, channels)

        acc00 += dy * x0_l
        acc01 += dy * x0_m
        acc02 += dy * x0_r
        acc10 += dy * x1_l
        acc11 += dy * x1_m
        acc12 += dy * x1_r
        acc20 += dy * x2_l
        acc21 += dy * x2_m
        acc22 += dy * x2_r

        x0_l = x0_m
        x0_m = x0_r
        x1_l = x1_m
        x1_m = x1_r
        x2_l = x2_m
        x2_m = x2_r

    tl.store(partial_ptr + base + 0 * channels, acc00)
    tl.store(partial_ptr + base + 1 * channels, acc01)
    tl.store(partial_ptr + base + 2 * channels, acc02)
    tl.store(partial_ptr + base + 3 * channels, acc10)
    tl.store(partial_ptr + base + 4 * channels, acc11)
    tl.store(partial_ptr + base + 5 * channels, acc12)
    tl.store(partial_ptr + base + 6 * channels, acc20)
    tl.store(partial_ptr + base + 7 * channels, acc21)
    tl.store(partial_ptr + base + 8 * channels, acc22)


def depthwise_conv_k3p1_dw_triton(x, out_grad, block_c=128, reduce_block_m=64, method="block_k4"):
    n, height, width, channels = x.shape
    assert method in ("block_k4", "w_scan")
    if method == "w_scan":
        num_partial_tiles = n * height
        partial_grid = (n * height, channels // block_c)
        partial_kernel = fused_depthwise_conv_k3p1_dw_partial_w_scan
    else:
        num_partial_tiles = n * (height // 4) * (width // 4)
        partial_grid = ((height // 4) * (width // 4), n * (channels // block_c))
        partial_kernel = fused_depthwise_conv_k3p1_dw_partial_block_k4

    partial = paddle.empty([num_partial_tiles, 9, channels], dtype="float32")
    weight_grad = paddle.empty([channels, 1, 3, 3], dtype="float32")
    partial_kernel[partial_grid](
        x, out_grad, partial, n, height, width, channels, BLOCK_C=block_c
    )

    for partial_index in range(9):
        # Kept as a deprecated call path; active implementation lives in ../depthwise_conv.py.
        pass
    return weight_grad
