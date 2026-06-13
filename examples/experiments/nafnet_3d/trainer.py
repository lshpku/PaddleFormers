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

"""
Trainer: 训练循环 + loss 计算 (Paddle 版)。

特性:
  - bf16 autocast（O2 + decorate）
  - AdamW 优化器
  - 梯度累加（grad_accum_steps）
  - grad clip
  - 步数驱动（不按 epoch），跑满 max_steps 个 optimizer.step 就停
  - save / load ckpt

Loss:
    w_recon_mask   * L1(pred, target)  on mask 区域
  + w_identity     * L1(pred, target)  on 非 dilated_mask 区域
  + w_recon_global * L1(pred, target)
"""
import os
import time
import numpy as np
from collections import defaultdict
from typing import Dict, Iterable, Iterator, Optional, Tuple

import paddle
from paddle import Tensor

from async_utils import to_device, to_host, wait_to_device
from config import Config
from vgg_loss import VGGLoss
from nvprof import nvtx_start, nvtx_stop, nvtx_range
from fusion import USE_TRITON_FUSION, clip_grad_norm_


class ShuffleBatchSampler:
    def __init__(self, size: int, batch_size: int = 1, seed: int = None):
        self.indices = np.arange(size)
        self.batch_size = batch_size
        rng = np.random if seed is None else np.random.RandomState(seed)
        rng.shuffle(self.indices)

    def __iter__(self) -> Iterator[list[int]]:
        for i in range(len(self)):
            yield list(self.indices[i * self.batch_size : (i + 1) * self.batch_size])

    def __len__(self) -> int:
        return len(self.indices) // self.batch_size


def _infinite(loader: Iterable):
    while True:
        for b in loader:
            yield b


def _masked_l1(pred: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    diff = (pred - target).abs() * mask
    denom = mask.sum().clip(min=1.0) * (pred.size / mask.size)
    return diff.sum() / denom


class Trainer:
    def __init__(
            self,
            cfg: Config,
            model: paddle.nn.Layer,
            train_dataset,
            optimizer,
            eval_dataset=None,
        ):
        self.cfg = cfg
        tcfg = cfg.train

        self.device = paddle.get_device(tcfg.device)

        # dtype: 固定 bf16（权重仍 fp32，autocast 里转 bf16）
        assert tcfg.dtype == "bf16", "this trainer is bf16-only by design"
        self.autocast_level = "O2"
        self.autocast_dtype = "bfloat16"

        self.model = model

        self.train_loader = paddle.io.DataLoader(
            train_dataset,
            num_workers=tcfg.num_workers,
            batch_sampler=ShuffleBatchSampler(
                len(train_dataset), tcfg.batch_size
            ),
        )
        self.eval_loader = None
        if eval_dataset is not None:
            self.eval_loader = paddle.io.DataLoader(
                eval_dataset,
                batch_size=7, # tcfg.batch_size,
                shuffle=False,
                num_workers=tcfg.num_workers,
                drop_last=False,
            )

        self.optimizer = optimizer

        self.vgg_loss = VGGLoss()

        self.global_step = 0

    def compute_loss(
        self, batch: Dict[str, Tensor], return_pred: bool = False
    ) -> Tuple[Tensor, Dict[str, float]]:
        lcfg = self.cfg.loss

        frames = to_device(batch["frames"])         # (B, T, H, W, 3)
        target = to_device(batch["target"])         # (B, T, H, W, 3)
        mask = to_device(batch["mask"])             # (B, H, W, 1)
        dilated = to_device(batch["dilated_mask"])  # (B, H, W, 1)

        frames = frames.cast(self.autocast_dtype) / 255.0
        frames = frames.unsqueeze(1)
        target = target.unsqueeze(1)

        if len(mask.shape) == 4:
            mask = mask.unsqueeze(1)  # (B, 1, H, W, 1)
        if len(dilated.shape) == 4:
            dilated = dilated.unsqueeze(1)

        with paddle.amp.auto_cast(
            enable=True,
            level=self.autocast_level,
            dtype=self.autocast_dtype,
        ):
            pred = self.model(frames, mask)  # (B, T, H, W, 3)

        pred = pred.cast("float32")
        dilated = dilated.cast("float32")
        target = target.cast("float32") / 255.0

        l_mask = _masked_l1(pred, target, dilated)
        l_global = (pred - target).abs().mean()
        l_vgg, _ = self.vgg_loss(pred.squeeze(1), target.squeeze(1))

        # l_mask == 2*l_global == 2*(l_vgg*0.025)
        loss = l_mask + l_global + l_vgg * 0.025

        wait_to_device()

        logs = {
            "loss": to_host(loss),
            "l_mask": to_host(l_mask),
            "l_global": to_host(l_global),
            "l_vgg": to_host(l_vgg),
        }
        if return_pred:
            return pred, logs
        return loss, logs

    def _forward_backward(self, batch: Dict[str, Tensor], scale: float) -> Dict[str, float]:
        loss, logs = self.compute_loss(batch)

        logs_event = paddle.device.Event()
        logs_event.record()

        loss = loss * scale
        loss.backward()

        logs_event.synchronize()
        return logs

    def _save_sample(self, batch, pred, batch_id: int = 0):
        from PIL import Image
        rows = []
        for inp, target, pred, mask, dilated in zip(
            batch["frames"].numpy(), batch["target"].numpy(),
            (pred.squeeze(1) * 255).clip(0, 255).cast("uint8").numpy(),
            batch["mask"].numpy(), batch["dilated_mask"].numpy(),
        ):
            mask = np.broadcast_to(mask * 100 + dilated * 100 + 50, inp.shape)
            row = np.concatenate([target, inp, pred, mask], axis=1)
            rows.append(row)
        grid = np.concatenate(rows, axis=0)
        image = Image.fromarray(grid)
        path = f"./pred/{self.global_step:06d}_{batch_id:03d}.png"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        image.save(path)
        print("saved sample:", path)

    @paddle.no_grad()
    def evaluate(self, max_batches: Optional[int] = None) -> Dict[str, float]:
        if self.eval_loader is None:
            return {}

        was_training = self.model.training
        self.model.eval()
        eval_logs = defaultdict(float)
        num_batches = 0

        for batch in self.eval_loader:
            pred, logs = self.compute_loss(batch, return_pred=True)
            for key, value in logs.items():
                eval_logs[key] += value
            num_batches += 1
            self._save_sample(batch, pred, num_batches)
            if max_batches is not None and num_batches >= max_batches:
                break

        if was_training:
            self.model.train()
        if num_batches == 0:
            return {}
        return {key: value / num_batches for key, value in eval_logs.items()}

    def _clip_grad(self, max_norm) -> float:
        parameters = self.model.parameters()
        if USE_TRITON_FUSION:
            return clip_grad_norm_(parameters, max_norm)
        return paddle.nn.utils.clip_grad_norm_(parameters, max_norm)

    def _optimizer_step(self) -> float:
        with nvtx_range("optimizer"):
            grad_clip = self.cfg.optim.grad_clip
            if grad_clip is not None:
                with nvtx_range("clip_grad"):
                    grad_norm = self._clip_grad(grad_clip)
                grad_norm = to_host(grad_norm)
                norm_event = paddle.device.Event()
                norm_event.record()
            else:
                grad_norm = -1.0

            with nvtx_range("step"):
                self.optimizer.step()
            with nvtx_range("clear_grad"):
                self.optimizer.clear_grad()

        if grad_clip is not None:
            norm_event.synchronize()
        return grad_norm

    def _maybe_save(self):
        tcfg = self.cfg.train
        if tcfg.ckpt_every <= 0 or tcfg.ckpt_dir is None:
            return
        if self.global_step % tcfg.ckpt_every != 0:
            return
        os.makedirs(tcfg.ckpt_dir, exist_ok=True)
        path = os.path.join(tcfg.ckpt_dir, f"step_{self.global_step:08d}.pdparams")
        paddle.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "global_step": self.global_step,
            },
            path,
        )
        print(f"[ckpt] saved -> {path}")

    def load_ckpt(self, path: str):
        state = paddle.load(path)
        self.model.set_state_dict(state["model"])
        self.optimizer.set_state_dict(state["optimizer"])
        self.global_step = state.get("global_step", 0)
        print(f"[ckpt] loaded from {path}, global_step={self.global_step}")

    def fit(self):
        tcfg = self.cfg.train
        self.model.train()
        self.optimizer.clear_grad()

        accum = tcfg.grad_accum_steps
        scale = 1.0 / accum

        data_iter = _infinite(self.train_loader)
        agg_logs = defaultdict(float)

        while self.global_step < tcfg.max_steps:
            t0 = time.time()
            for _ in range(accum):
                batch = next(data_iter)
                logs = self._forward_backward(batch, scale)
                for k, v in logs.items():
                    agg_logs[k] += v.item() / accum

            grad_norm = self._optimizer_step()
            self.global_step += 1
            dt = time.time() - t0

            if tcfg.log_every > 0 and self.global_step % tcfg.log_every == 0:
                fmt = " ".join(f"{k}={v:.4f}" for k, v in agg_logs.items())
                print(
                    f"[step {self.global_step:>5d}/{tcfg.max_steps}] {fmt} "
                    f"learning_rate={self.optimizer.get_lr()} "
                    f"grad_norm={grad_norm:.4f} step_interval={dt:.2f} "
                    f"mem_allocated_gb={paddle.device.memory_allocated()/2**30:.3f} "
                    f"max_mem_allocated_gb={paddle.device.max_memory_allocated()/2**30:.3f} "
                    f"max_mem_reserved_gb={paddle.device.max_memory_reserved()/2**30:.3f}"
                )
                agg_logs.clear()
                paddle.device.reset_max_memory_allocated()

            if self.global_step == 10:
                nvtx_start()
            if self.global_step == 15:
                nvtx_stop()

            if (
                self.eval_loader is not None
                and tcfg.eval_every > 0
                and self.global_step % tcfg.eval_every == 0
            ):
                eval_logs = self.evaluate()
                if eval_logs:
                    eval_fmt = " ".join(f"{k}={v:.4f}" for k, v in eval_logs.items())
                    print(f"[eval {self.global_step:>5d}/{tcfg.max_steps}] {eval_fmt}")

            self._maybe_save()
