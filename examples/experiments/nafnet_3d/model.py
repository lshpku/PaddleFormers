# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import paddle
from paddle import framework, nn
from paddle.autograd import PyLayer
from paddle.distributed.fleet.utils import recompute

from config import ModelConfig
from fusion import (
    USE_TRITON_FUSION,
    FusedDepthwiseConvK3P1Triton,
    FusedLayerNormTriton,
    FusedSimpleGateAvgPoolTriton,
    FusedSimpleGateTriton,
    FusedWeightedResidualAddTriton,
)
from nvprof import nvtx_start, nvtx_stop, nvtx_push, nvtx_pop
from recompute import RecomputeWithoutOutput


def simple_gate(x):
    if USE_TRITON_FUSION:
        return FusedSimpleGateTriton.apply(x)
    p, q = paddle.chunk(x, 2, axis=-1)
    return p * q


def simple_gate_avg_pool(x, pool):
    if USE_TRITON_FUSION:
        return FusedSimpleGateAvgPoolTriton.apply(x)
    x = simple_gate(x)
    return x, pool(x)


class DepthwiseConvK3P1(nn.Conv2d):
    def __init__(self, channels):
        super().__init__(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            data_format="NCHW"
        )

    def forward(self, x):
        if USE_TRITON_FUSION:
            return FusedDepthwiseConvK3P1Triton.apply(x, self.weight, self.bias)
        x = x.transpose([0, 3, 1, 2])
        x = super().forward(x)
        x = x.transpose([0, 2, 3, 1])
        return x


class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        if USE_TRITON_FUSION:
            return FusedLayerNormTriton.apply(x, self.weight, self.bias)
        return super().forward(x)


Linear = paddle.incubate.nn.FusedLinear


def concat_mask(inp, mask):
    if mask is None:
        return inp
    if mask.shape != inp.shape:
        shape = list(mask.shape)
        shape[1] = inp.shape[1]
        mask = mask.broadcast_to(shape)
    return paddle.concat([inp, mask.cast(inp.dtype)], axis=-1)


def weighted_residual_add(residual, x, weight):
    if USE_TRITON_FUSION:
        return FusedWeightedResidualAddTriton.apply(residual, x, weight)
    return residual + x * weight


class NAFBlock(nn.Layer):
    def __init__(self, c, cfg):
        super().__init__()
        self.c = c
        dw_channel = c * cfg.dw_expand
        self.conv1 = Linear(in_features=c, out_features=dw_channel)
        self.conv2 = DepthwiseConvK3P1(dw_channel)
        self.conv3 = Linear(in_features=dw_channel // 2, out_features=c)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1, data_format="NHWC"),
            Linear(in_features=dw_channel // 2, out_features=dw_channel // 2),
        )

        ffn_channel = cfg.ffn_expand * c
        self.conv4 = Linear(in_features=c, out_features=ffn_channel)
        self.conv5 = Linear(in_features=ffn_channel // 2, out_features=c)

        self.norm1 = LayerNorm(c)
        self.norm2 = LayerNorm(c)

        self.beta = nn.Parameter(paddle.zeros(c))
        self.gamma = nn.Parameter(paddle.zeros(c))

    def _forward_impl(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x, x_pool = simple_gate_avg_pool(x, self.sca[0])
        x = x * self.sca[1](x_pool)
        x = self.conv3(x)

        mid = weighted_residual_add(inp, x, self.beta)

        x = self.norm2(mid)
        x = self.conv4(x)
        x = simple_gate(x)
        x = self.conv5(x)

        return mid, x

    def forward(self, inp):
        N, H, W, C = inp.shape
        name = f"block_{N}x{H}x{W}x{C}"
        nvtx_push(name + "_fw")

        ctx = RecomputeWithoutOutput(name)
        mid, x = ctx.recompute(self._forward_impl, inp)

        out = weighted_residual_add(mid, x, self.gamma)
        ctx.discard_output_and_register_recompute(out)

        nvtx_pop()
        return out


class NAFNet3D(nn.Layer):
    def __init__(self, cfg):
        super().__init__()

        self.intro = nn.Conv2D(
            in_channels=cfg.img_channel + (1 if cfg.input_mask else 0),
            out_channels=cfg.width, kernel_size=3, padding=1, data_format="NHWC",
        )
        self.ending = nn.Conv2D(
            in_channels=cfg.width, out_channels=cfg.img_channel,
            kernel_size=3, padding=1, data_format="NHWC",
        )

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = cfg.width
        for num in cfg.enc_blk_nums:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan, cfg) for _ in range(num)]))
            self.downs.append(nn.Conv2D(chan, 2 * chan, 2, 2, data_format="NHWC"))
            chan = chan * 2

        self.middle_blks = nn.Sequential(*[NAFBlock(chan, cfg) for _ in range(cfg.middle_blk_num)])

        for num in cfg.dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2D(chan, chan * 2, 1, bias=False, data_format="NHWC"),
                    nn.PixelShuffle(2, data_format="NHWC"),
                )
            )
            chan = chan // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan, cfg) for _ in range(num)]))

    def forward(self, inp, mask=None):
        N, T, _, _, _ = inp.shape

        def squeeze_time(x):
            _, _, H, W, C = x.shape
            return x.reshape([N * T, H, W, C])

        def unsqueeze_time(x):
            _, H, W, C = x.shape
            return x.reshape([N, T, H, W, C])

        x = self.intro(squeeze_time(concat_mask(inp, mask)))

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = unsqueeze_time(x) + inp

        return x


from paddle.base.core import CUDAGraph
from paddle.device.cuda import graphs


class GudaGraphRunner(PyLayer):
    @staticmethod
    def forward(ctx, _, model):
        ctx.model = model
        return ctx.model.output

    @staticmethod
    def backward(ctx, grad):
        ctx.model.output_grad.copy_(grad)
        ctx.model.bwd_graph.replay()
        return None


def copy_or_clone(a, b):
    if a is None:
        return b.detach().clone()
    a.copy_(b)
    return a


class GudaGraphModel(NAFNet3D):
    WARMUP = 3

    def __init__(self, cfg):
        super().__init__(cfg)

        pool_id = CUDAGraph.gen_new_memory_pool_id()
        print("[cudagraph] new pool_id:", pool_id)
        self.fwd_graph = graphs.CUDAGraph(pool_id=pool_id)
        self.bwd_graph = graphs.CUDAGraph(pool_id=pool_id)

        self.inp = None
        self.mask = None
        self.output = None
        self.output_grad = None

    def _warmup(self):
        output = super().forward(self.inp, self.mask)
        grad = paddle.randn_like(output)
        output.backward(grad)
        self.clear_gradients(set_to_zero=False)

    def _forward_impl(self):
        self.fwd_graph.replay()
        # PyLayer requires at least one tensor input
        x = paddle.empty([0])
        x.stop_gradient = False
        return GudaGraphRunner.apply(x, self)

    def forward(self, inp, mask):
        self.inp = copy_or_clone(self.inp, inp)
        self.mask = copy_or_clone(self.mask, mask)

        if self.output is not None:
            return self._forward_impl()

        print(f"[cudagraph] warmup for {self.WARMUP} times")
        for i in range(self.WARMUP):
            paddle.base.core.nvprof_nvtx_push("warmup")
            self._warmup()
            paddle.base.core.nvprof_nvtx_pop()

        print("[cudagraph] capturing forward graph")
        paddle.device.synchronize()
        self.fwd_graph.capture_begin()
        self.output = super().forward(self.inp, self.mask)
        self.fwd_graph.capture_end()

        print("[cudagraph] capturing backward graph")
        self.output_grad = paddle.empty_like(self.output)
        paddle.device.synchronize()
        self.bwd_graph.capture_begin()
        self.output.backward(self.output_grad)
        self.bwd_graph.capture_end()

        print("[cudagraph] done capturing")
        paddle.device.synchronize()

        return self._forward_impl()


if __name__ == "__main__":
    cfg = ModelConfig(
        input_mask=True,
        width=64,
        middle_blk_num=8,
        enc_blk_nums=[2, 4, 8, 16],
        dec_blk_nums=[16, 8, 4, 4],
    )

    # model = NAFNet3D(cfg)
    model = GudaGraphModel(cfg)
    optimizer = paddle.optimizer.AdamW(
        parameters=model.parameters(),
        multi_precision=True,
    )
    model, optimizer = paddle.amp.decorate(
        models=model, optimizers=optimizer, level="O2", dtype="bfloat16", master_grad=True
    )

    inp = paddle.randn([1, 32, 256, 256, 3], dtype="bfloat16")
    inp.stop_gradient = False
    mask = paddle.randn([1, 32, 256, 256, 1], dtype="bfloat16")

    with paddle.amp.auto_cast(enable=True, level="O2", dtype="bfloat16"):
        out = model(inp, mask)
    out.backward()

    new_event = lambda: paddle.device.Event(enable_timing=True)
    events = [(new_event(), new_event()) for _ in range(10)]

    paddle.device.reset_max_memory_allocated()
    nvtx_start()

    for e0, e1 in events:
        inp = inp.detach()
        inp.stop_gradient = False
        e0.record()

        nvtx_push("forward")
        with paddle.amp.auto_cast(enable=True, level="O2", dtype="bfloat16"):
            out = model(inp, mask)
        nvtx_pop()

        print(
            f"after fw: use={paddle.device.memory_allocated()/2**30:.3f}"
            f" max={paddle.device.max_memory_allocated()/2**30:.3f}"
        )
        paddle.device.reset_max_memory_allocated()

        nvtx_push("backward")
        out.backward()
        nvtx_pop()

        print(
            f"after bw: use={paddle.device.memory_allocated()/2**30:.3f}"
            f" max={paddle.device.max_memory_allocated()/2**30:.3f}"
        )
        paddle.device.reset_max_memory_allocated()
        e1.record()

        nvtx_push("optimizer")
        optimizer.step()
        optimizer.clear_grad()
        nvtx_pop()

    nvtx_stop()
    paddle.device.synchronize()

    for e0, e1 in events:
        print(f"duration: {e0.elapsed_time(e1):.3f}ms")
