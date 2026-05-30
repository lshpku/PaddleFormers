# VideoWatermarkRemoval 训练框架 Demo

轻量视频水印去除模型的训练框架，实现方案 A（U-Net + 隐式时序融合）。

## 目录结构

```
VideoWatermarkRemoval/
├── config.py     # 所有超参（ModelConfig / DataConfig / OptimConfig / LossConfig / TrainConfig）
├── model.py      # WatermarkVideoUNet + UNetSpec + build_model()
├── dataset.py    # FakeWatermarkDataset（冒烟用）+ PtFileDataset（真实数据）
├── trainer.py    # forward + loss + 训练循环（bf16 / AdamW / grad accum / grad clip / ckpt）
└── train.py      # CLI 入口
```

## 设计约定

- **精度**：固定 bf16。权重和 AdamW 状态保持 fp32，仅在 `torch.autocast(dtype=bfloat16)` 里做 forward+loss。不使用 `GradScaler`（bf16 动态范围足够）。
- **优化器**：AdamW。
- **梯度累加**：每 `grad_accum_steps` 个 micro-batch 才调用一次 `optimizer.step()`，loss 自动按 `1/accum` 缩放。
- **步数驱动**：不按 epoch，跑满 `max_steps` 次 `optimizer.step()` 停止（dataloader 无限循环）。
- **数据接口**：`PtFileDataset` 读取目录下所有 `.pt`，每个文件约定为 dict：
  - `frames`: `(T, 4, H, W)` RGB(3) + mask(1)
  - `target`: `(3, H, W)` 中心帧 clean RGB
  - `mask`: `(1, H, W)` 可选，缺失则从 `frames[T//2, 3:4]` 自动回退
  - `dilated_mask`: `(1, H, W)` 可选

## 运行命令

### 1. 冒烟测试（fake 数据）

跑 2 个 step，64x64 小张量，用于验证训练通路：

```bash
cd /root/paddlejob/share-storage/gpfs/system-public/liangshuhao
python -m aigen.VideoWatermarkRemoval.train --smoke --device cpu
```

有 GPU 时改用：

```bash
python -m aigen.VideoWatermarkRemoval.train --smoke --device cuda
```

### 2. 真实数据训练

```bash
cd /root/paddlejob/share-storage/gpfs/system-public/liangshuhao
python -m aigen.VideoWatermarkRemoval.train \
    --data-root /path/to/pt_files \
    --num-frames 5 \
    --h 480 --w 832 \
    --batch-size 2 \
    --grad-accum 4 \
    --max-steps 50000 \
    --lr 2e-4 \
    --device cuda \
    --ckpt-dir ./ckpts \
    --ckpt-every 1000
```

### 3. 仅构建模型 + 打印参数量

```bash
cd /root/paddlejob/share-storage/gpfs/system-public/liangshuhao
python -c "
from aigen.VideoWatermarkRemoval.config import ModelConfig
from aigen.VideoWatermarkRemoval.model import build_model, count_params
m = build_model(ModelConfig())
print(f'{count_params(m)/1e6:.2f}M')
"
```

## CLI 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--smoke` | False | 开启即用 fake 数据 + 64x64 + 2 step |
| `--data-root` | None | 真实 .pt 数据目录；留空走 FakeWatermarkDataset |
| `--num-frames` | 5 | 时间窗（需与模型一致） |
| `--h` / `--w` | 128 / 128 | fake 数据分辨率 |
| `--batch-size` | 1 | |
| `--grad-accum` | 4 | 梯度累加步数 |
| `--max-steps` | 100 | 总 optimizer.step 次数 |
| `--lr` | 2e-4 | AdamW 学习率 |
| `--device` | cuda | cuda / cpu，cuda 不可用自动回退 cpu |
| `--ckpt-dir` | None | checkpoint 目录，None 则不保存 |
| `--ckpt-every` | 0 | 0 表示不保存；>0 时每 N 步存一次 |

## Loss 组成

```
L = w_recon_mask   * L1(pred, target)  on mask 区域
  + w_identity     * L1(pred, target)  on (1 - dilated_mask) 区域
  + w_recon_global * L1(pred, target)
```

权重在 `config.py::LossConfig` 里修改。后续要加感知 loss / 时序 loss，扩展 `Trainer.compute_loss` 即可。

## TODO

- [ ] 接入真实预处理好的 `.pt` 数据
- [ ] 加 VGG 感知 loss（mask 区域）
- [ ] 加时序一致性 loss（需要多帧输出分支）
- [ ] 支持 checkpoint resume
- [ ] 支持 DDP 多卡
