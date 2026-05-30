import random
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_PATH = "./Arial-Unicode-Bold.ttf"


def _get_font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(_FONT_PATH, size)


def _random_text(rng: random.Random, num_lines: int) -> list:
    """生成随机文本行，中文占比约 80%。"""
    lines = []
    for _ in range(num_lines):
        n = rng.randint(5, 10)
        chars = []
        for _ in range(n):
            if rng.random() < 0.8:
                # CJK Unified Ideographs (U+4E00 ~ U+9FFF)
                chars.append(chr(rng.randint(0x4E00, 0x9FFF)))
            else:
                chars.append(rng.choice(
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
                ))
        lines.append("".join(chars))
    return lines


def _measure_text(draw: ImageDraw.Draw, lines: list, font: ImageFont.FreeTypeFont):
    """返回 (max_width, line_heights, total_height_with_spacing, spacing)。"""
    max_w = 0
    heights = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        max_w = max(max_w, bbox[2] - bbox[0])
        heights.append(bbox[3] - bbox[1])
    spacing = max(1, font.size // 3)
    total_h = sum(heights) + (len(lines) - 1) * spacing
    return max_w, heights, total_h, spacing


def make_watermark(width: int, height: int, seed: int | None = None) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    生成随机文字水印及其 mask。

    Returns
    -------
    watermark : np.ndarray, uint8, shape (H, W, 4)
        RGBA 图像，文字区域带有透明度，背景完全透明。
    mask : np.ndarray, uint8, shape (H, W)
        二值 mask，255 表示水印区域（已包含 0~5 pix 的随机外扩）。
    log : str
        生成过程每一步的简要记录。
    """
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    rng = random.Random(seed)
    logs = [f"S={seed}"]

    # 1. 内容
    num_lines = rng.randint(1, 2)
    lines = _random_text(rng, num_lines)
    total_chars = sum(len(line) for line in lines)
    cn_count = sum(1 for line in lines for ch in line if '\u4e00' <= ch <= '\u9fff')
    logs.append(f"T({num_lines}L,{total_chars}C,{cn_count}Z)")

    # 2. 目标面积 5%~20%
    target_ratio = rng.uniform(0.05, 0.50)
    target_area = width * height * target_ratio
    logs.append(f"A({target_ratio:.1%},{width}x{height})")

    # 3. 搜索合适字号
    lo, hi = 8, max(8, min(width, height) // 2)
    best_font = _get_font(lo)
    best_size = lo

    tmp_img = Image.new("RGBA", (1, 1))
    tmp_draw = ImageDraw.Draw(tmp_img)

    for _ in range(10):
        if lo > hi:
            break
        mid = (lo + hi) // 2
        font = _get_font(mid)
        max_w, _, total_h, _ = _measure_text(tmp_draw, lines, font)
        area = max_w * total_h
        if area < target_area:
            best_size = mid
            best_font = font
            lo = mid + 1
        else:
            hi = mid - 1

    max_w, heights, total_h, spacing = _measure_text(tmp_draw, lines, best_font)
    actual_ratio = (max_w * total_h) / (width * height)
    logs.append(f"F({best_size}),B({max_w}x{total_h}),A'({actual_ratio:.1%})")

    # 4. 在大画布上绘制 → 旋转 → 放置到目标画布（允许溢出）
    # 颜色：浅灰 + 半透明
    v = rng.randint(128, 255)
    alpha = rng.randint(128, 248)
    fill = (v, v, v, alpha)

    # 计算旋转后需要的画布大小（外接矩形）
    angle = rng.uniform(-180.0, 180.0)
    angle = 0
    rad = abs(angle) * np.pi / 180.0
    diag = int(np.ceil(np.sqrt(max_w**2 + total_h**2)))
    # 旋转后的 bbox
    rot_w = int(np.ceil(max_w * abs(np.cos(rad)) + total_h * abs(np.sin(rad))))
    rot_h = int(np.ceil(max_w * abs(np.sin(rad)) + total_h * abs(np.cos(rad))))
    pad_w = max(rot_w, diag) + 4
    pad_h = max(rot_h, diag) + 4

    big = Image.new("RGBA", (pad_w, pad_h), (v, v, v, 0))
    draw = ImageDraw.Draw(big)

    # 文字在大画布中心绘制
    cx = (pad_w - max_w) // 2
    cy = (pad_h - total_h) // 2
    y = cy
    for line, lh in zip(lines, heights):
        bbox = draw.textbbox((0, 0), line, font=best_font)
        draw.text((cx - bbox[0], y - bbox[1]), line, font=best_font, fill=fill)
        y += lh + spacing

    # 旋转
    if abs(angle) > 0.1:
        big = big.rotate(angle, resample=Image.BICUBIC, expand=False)

    # 裁剪掉全透明的边界，得到紧凑的旋转后文字图
    arr = np.array(big)
    alpha_ch = arr[:, :, 3]
    ys, xs = np.where(alpha_ch > 0)
    if len(xs) == 0:
        # 退化情况：没有文字，返回全透明
        watermark_np = np.zeros((height, width, 4), dtype=np.uint8)
        mask = np.zeros((height, width), dtype=np.uint8)
        logs.append(f"D(0,0),C({v}),a={alpha},R({angle:.1f}),M(+0,0.0%)")
        log_str = " | ".join(logs)
        return watermark_np, mask, log_str

    x1, y1 = xs.min(), ys.min()
    x2, y2 = xs.max() + 1, ys.max() + 1
    cropped = big.crop((x1, y1, x2, y2))
    crop_w, crop_h = cropped.size

    # 放置到目标画布：75% 完全在内，25% 最多 30% 溢出
    if rng.random() < 0.75:
        tx = rng.randint(0, max(0, width - crop_w))
        ty = rng.randint(0, max(0, height - crop_h))
        overflow = False
    else:
        ox = max(1, int(crop_w * 0.3))
        oy = max(1, int(crop_h * 0.3))
        tx = rng.randint(-ox, width - 1)
        ty = rng.randint(-oy, height - 1)
        overflow = True

    canvas = Image.new("RGBA", (width, height), (v, v, v, 0))
    canvas.paste(cropped, (tx, ty), cropped)
    watermark_np = np.array(canvas)
    logs.append(f"D({tx},{ty}),C({v}),a={alpha},O={overflow},R({angle:.1f})")

    # 6. mask：从 alpha 提取并随机外扩 0~5 pix
    alpha_ch = watermark_np[:, :, 3]
    mask = (alpha_ch > 0).astype(np.uint8) * 255

    expand = rng.randint(0, 2)
    if expand > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (expand * 2 + 1, expand * 2 + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)
    logs.append(f"M(+{expand},{(mask > 0).sum() / (width * height):.1%})")

    log_str = " | ".join(logs)
    return watermark_np, mask, log_str


if __name__ == "__main__":
    import os
    import sys

    W, H = 256, 256
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else random.randint(0, 2**31 - 1)
    watermark, mask, log = make_watermark(W, H, seed=seed)
    print(log)

    # 绿色底图
    bg = np.full((H, W, 3), [0, 255, 0], dtype=np.uint8)

    # 合成 watermark 到绿色底图
    alpha = watermark[:, :, 3:4].astype(np.float32) / 255.0
    blended = (bg.astype(np.float32) * (1 - alpha) + watermark[:, :, :3].astype(np.float32) * alpha).astype(np.uint8)

    out_dir = "./"
    os.makedirs(out_dir, exist_ok=True)

    # 1. watermark 本身（去除透明背景，转 RGB）
    wm_rgb = watermark[:, :, :3].copy()
    wm_rgb = cv2.cvtColor(wm_rgb, cv2.COLOR_RGB2BGR)

    # 2. mask 转 3 通道以便拼接
    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # 3. blended 转 BGR
    blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)

    # 横向拼接
    combined = np.hstack([wm_rgb, mask_3ch, blended_bgr])

    fname = f"watermark_{H}x{W}_s{seed}.png"
    cv2.imwrite(os.path.join(out_dir, fname), combined)
    print(f"Saved to {out_dir}/{fname}")
