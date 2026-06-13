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
训练入口

用法:
    # 1) smoke test（fake 数据，极小 step，仅验证通路）
    python train.py --smoke

    # 2) 从 YAML 配置文件训练
    python train.py --config config.yaml

    # 3) 恢复训练
    python train.py --config config.yaml --resume step_0100.pdparams
"""

import argparse
import numpy as np

import train_env as _

import paddle

from config import Config, DataConfig, load_config
from dataset import VideoDataset, ImageDataset, FakeDataset
from model import NAFNet3D
from trainer import Trainer


def build_dataset(data_cfg: DataConfig):
    train_dataset = ImageDataset("/root/autodl-tmp/DIV2K_Flicker2K")
    eval_dataset = ImageDataset(
        "/root/autodl-tmp/nafnet_eval",
        tile_args={"tile_w": 512, "tile_h": 512}, fix_mask=True,
    )
    return train_dataset, eval_dataset
    if data_cfg.data_root is None:
        return FakeDataset(
            data_cfg.max_length,
            data_cfg.num_frames,
            data_cfg.frame_h,
            data_cfg.frame_w,
        )
    raise NotImplementedError("Real dataset loader not ready yet")


def count_params(model: paddle.nn.Layer) -> int:
    return sum(p.size for p in model.parameters())


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true", help="fake 数据跑 2 个 step 验证框架")
    p.add_argument("--config", type=str, default=None, help="YAML 配置文件路径")
    p.add_argument("--resume", type=str, default=None, help="从 checkpoint 恢复训练")
    return p.parse_args()


def main():
    args = _parse_args()

    print("[build] config...")
    if args.smoke:
        # 极小配置，确保快速跑过
        cfg = Config()
        cfg.train.batch_size = 64
        cfg.train.grad_accum_steps = 1
        cfg.train.max_steps = 100_000
        cfg.train.eval_every = 100
        cfg.train.num_workers = 8
    elif args.config:
        cfg = load_config(args.config)
    else:
        raise SystemExit("请指定 --config <yaml> 或 --smoke")
    print(cfg)

    tcfg = cfg.train
    paddle.seed(tcfg.seed)
    np.random.seed(tcfg.seed)

    print("[build] dataset...")
    train_dataset, eval_dataset = build_dataset(cfg.data)
    print(f"  -> train: {type(train_dataset).__name__}, len={len(train_dataset)}")
    if eval_dataset is not None:
        print(f"  -> eval: {type(eval_dataset).__name__}, len={len(eval_dataset)}")

    print("[build] model...")
    model = NAFNet3D(cfg.model)
    n_params = count_params(model)
    print(f"  -> params = {n_params/1e6:.2f}M")

    # bf16 O2: decorate model + optimizer
    if tcfg.dtype == "bf16":
        print("[amp] decorate model/optimizer with O2 bf16")
        ocfg = cfg.optim
        grad_clip = (
            paddle.nn.ClipGradByGlobalNorm(ocfg.grad_clip)
            if ocfg.grad_clip is not None else None
        )
        optimizer = paddle.optimizer.AdamW(
            parameters=model.parameters(),
            learning_rate=ocfg.lr,
            beta1=ocfg.betas[0],
            beta2=ocfg.betas[1],
            weight_decay=ocfg.weight_decay,
            epsilon=ocfg.eps,
            grad_clip=grad_clip,
            multi_precision=True,
        )
        model, optimizer = paddle.amp.decorate(
            models=model,
            optimizers=optimizer,
            level="O2",
            dtype="bfloat16",
            master_grad=tcfg.master_grad,
        )
        trainer = Trainer(cfg, model, train_dataset, optimizer, eval_dataset)
    else:
        raise NotImplementedError(f"only supports bf16 training, got: {tcfg.dtype}")

    if args.resume:
        trainer.load_ckpt(args.resume)

    print(
        f"[run] device={trainer.device}, dtype={tcfg.dtype}, "
        f"max_steps={tcfg.max_steps}, accum={tcfg.grad_accum_steps}"
    )
    trainer.fit()
    print("[done]")


if __name__ == "__main__":
    main()
