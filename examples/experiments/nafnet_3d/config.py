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

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import yaml


@dataclass
class ModelConfig:
    img_channel: int = 3
    input_mask: bool = True
    width: int = 64
    middle_blk_num: int = 8
    enc_blk_nums: list = field(default_factory=lambda: [2, 4, 8, 16])
    dec_blk_nums: list = field(default_factory=lambda: [16, 8, 4, 4])
    dw_expand: int = 2
    ffn_expand: int = 2


@dataclass
class DataConfig:
    data_root: Optional[str] = None
    num_frames: int = 30
    frame_h: int = 256
    frame_w: int = 256
    max_length: int = 64


@dataclass
class OptimConfig:
    lr: float = 1e-3
    betas: Tuple[float, float] = (0.9, 0.9)
    weight_decay: float = 0.0
    eps: float = 1e-8
    grad_clip: float = 100.0


@dataclass
class LossConfig:
    w_recon_mask: float = 1.0  # mask 区域 L1
    w_identity: float = 0.5  # 非 mask 区域 identity
    w_recon_global: float = 0.1  # 全图 L1 兜底（避免前期发散）


@dataclass
class TrainConfig:
    # runtime
    device: str = "gpu"
    dtype: str = "bf16"  # 固定 bf16，此处只做记录
    seed: int = 42

    # batching
    batch_size: int = 1
    num_workers: int = 0  # smoke 先 0，避免 fork 问题
    grad_accum_steps: int = 4  # 梯度累加

    # scheduling
    max_steps: int = 100  # 总 optimizer.step 次数
    log_every: int = 1
    eval_every: int = 0  # 0 表示不 eval；>0 时在该 step 间隔 eval
    ckpt_every: int = 0  # 0 表示不保存；>0 时在该 step 间隔保存
    ckpt_dir: Optional[str] = None


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _set_field(obj: Any, key: str, value: Any) -> None:
    """递归地将 dict 合并到 dataclass 实例。"""
    if not hasattr(obj, key):
        return
    attr = getattr(obj, key)
    if hasattr(attr, "__dataclass_fields__") and isinstance(value, dict):
        for k, v in value.items():
            _set_field(attr, k, v)
    else:
        setattr(obj, key, value)


def load_config(path: str) -> Config:
    """从 YAML 文件加载配置。"""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = Config()
    if isinstance(raw, dict):
        for k, v in raw.items():
            _set_field(cfg, k, v)
    return cfg
