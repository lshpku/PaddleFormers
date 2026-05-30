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
import json
import zlib
import sqlite3
import subprocess
import numpy as np
from PIL import Image
from pathlib import Path
from fractions import Fraction

import paddle


def get_video_info(video_path):
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,nb_frames,avg_frame_rate,duration",
        "-of", "json",
        video_path,
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True
    )

    info = json.loads(result.stdout)
    stream = info["streams"][0]

    width = int(stream["width"])
    height = int(stream["height"])
    fps_str = stream.get("avg_frame_rate")

    nb_frames = stream.get("nb_frames", None)

    if nb_frames is not None and nb_frames != "N/A":
        frame_count = int(nb_frames)
    else:
        duration = stream.get("duration", None)
        if duration is not None and fps_str is not None:
            frame_count = round(float(duration) * eval(fps_str))
        else:
            frame_count = None

    return width, height, frame_count, fps_str


def read_frame_range_fast(
    video_path: str, width: int, height: int, fps: str,
    start_frame: int, num_frames: int, x_off: int, y_off: int,
):
    """
    假设视频是标准恒定帧率视频。
    帧号是 0-based。
    """
    start_time = float(start_frame / Fraction(fps))

    x_tile, y_tile = 256, 256
    frame_size = x_tile * y_tile * 3
    total_size = num_frames * frame_size

    cmd = [
        "ffmpeg",
        "-ss", str(start_time),
        "-i", video_path,
        "-loglevel", "error",
        "-frames:v", str(num_frames),
        "-vf", f"crop={x_tile}:{y_tile}:{x_off}:{y_off}",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "pipe:1",
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    raw = process.stdout.read(total_size)
    stderr = process.stderr.read()
    ret = process.wait()

    if ret != 0:
        raise RuntimeError(stderr.decode("utf-8", errors="ignore"))

    actual_frames = len(raw) // frame_size

    frames = np.frombuffer(raw, dtype=np.uint8)
    frames = frames[:actual_frames * frame_size]
    frames = frames.reshape((actual_frames, y_tile, x_tile, 3))

    return frames


def tile_video(
    width: int, height: int, frame_count: int,
    tile_w: int = 256, tile_h: int = 256, tile_t: int = 30
):
    """
    将视频拆成连续、不重叠的 (256, 256, 30) tiles。

    空间维度：居中取，边缘不能整除的部分丢弃。
    时间维度：从第 0 帧开始取，末尾不能整除的部分丢弃。

    返回：
        List[(x_offset, y_offset, t_offset)]
    """

    # 可以放下多少个完整 tile
    nx = width // tile_w
    ny = height // tile_h
    nt = frame_count // tile_t

    # 居中裁剪后的起始偏移
    used_w = nx * tile_w
    used_h = ny * tile_h

    x_start = (width - used_w) // 2
    y_start = (height - used_h) // 2

    tiles = []

    for t in range(nt):
        t_offset = t * tile_t

        for y in range(ny):
            y_offset = y_start + y * tile_h

            for x in range(nx):
                x_offset = x_start + x * tile_w
                tiles.append((x_offset, y_offset, t_offset))

    return tiles


def save_video_with_ffmpeg(
    frames: np.ndarray,
    output_path: str,
    fps: int = 10,
    crf: int = 18,
):
    """
    使用 ffmpeg pipe 将 numpy array 保存为视频。

    Parameters
    ----------
    frames : np.ndarray
        shape 为 [T, H, W, 3] 的视频帧，RGB 格式。
        支持 uint8 [0, 255] 或 float [0, 1] / [0, 255]。
    output_path : str
        输出视频路径，例如 "output.mp4"。
    fps : int
        视频帧率，默认 10。
    crf : int
        H.264 编码质量，越小质量越高，常用 18~28。
    """

    assert frames.ndim == 4, f"frames should have shape [T, H, W, 3], got {frames.shape}"
    assert frames.shape[-1] == 3, f"last dim should be 3, got {frames.shape[-1]}"
    assert frames.dtype == np.uint8, f"expected uint8 frames, got {frames.dtype}"

    T, H, W, C = frames.shape

    # 确保内存连续
    frames = np.ascontiguousarray(frames)

    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-y",  # 覆盖已有文件
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}",
        "-r", str(fps),
        "-i", "-",  # 从 stdin 读入

        # 输出编码设置
        "-an",  # 无音频
        "-vcodec", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", str(crf),
        "-preset", "slow",

        output_path,
    ]

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
    )

    try:
        process.stdin.write(frames.tobytes())
        process.stdin.close()
        process.wait()

    except Exception:
        process.kill()
        raise


def simple_mask(frames: np.ndarray, seed: int = None):
    # frames: (..., H, W, C)
    mask = np.zeros((256, 256, 1), np.uint8)
    dilated = np.zeros_like(mask)
    rng = np.random if seed is None else np.random.RandomState(seed)

    if seed is None:
        rot = rng.randint(4)
        frames = np.rot90(frames, k=rot, axes=(-3, -2))
        frames = np.ascontiguousarray(frames)

    def sr(off, size):
        return slice(max(0, off), max(0, off + size))

    for _ in range(rng.randint(1, 11)):
        x0 = rng.randint(-16, 256 - 16)
        y0 = rng.randint(-16, 256 - 16)
        xs = rng.randint(16, 64)
        ys = rng.randint(16, 64)
        mask[sr(x0, xs), sr(y0, ys)] = 1
        dilated[sr(x0 - 8, xs + 16), sr(y0 - 8, ys + 16)] = 1
        if np.sum(mask) >= mask.size * 0.3:
            break

    target = frames

    alpha = rng.uniform(0.05, 0.3)
    frames = frames.astype(np.float32)
    frames = frames * (1.0 - mask) + (frames * alpha + 260 * (1 - alpha)) * mask
    frames = frames.clip(0, 255).astype(np.uint8)

    return {
        "frames": frames,
        "target": target,
        "mask": mask,
        "dilated_mask": dilated,
    }


def save_image_grid(A, B, C, D, path: str):
    C = np.broadcast_to(C, A.shape)
    D = np.broadcast_to(D, A.shape)
    row1 = np.concatenate([A, B], axis=1)
    row2 = np.concatenate([C, D], axis=1)
    grid = np.concatenate([row1, row2], axis=0)
    Image.fromarray(grid).save(path)


class VideoMeta:
    def __init__(self, path):
        self.path = path
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self._init_table()

    def _init_table(self):
        cursor = self.conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS video_meta (
            name        TEXT PRIMARY KEY NOT NULL,
            width       INTEGER NOT NULL,
            height      INTEGER NOT NULL,
            frame_count INTEGER NOT NULL,
            fps         TEXT NOT NULL
        )
        """)

        self.conn.commit()

    def insert(self, name: str, width: int, height: int, frame_count: int, fps: str):
        if any(v is None for v in [name, width, height, frame_count, fps]):
            raise ValueError("所有字段都不能为空")

        cursor = self.conn.cursor()

        cursor.execute("""
        INSERT OR REPLACE INTO video_meta
        (name, width, height, frame_count, fps)
        VALUES (?, ?, ?, ?, ?)
        """, (name, width, height, frame_count, fps))

        self.conn.commit()

    def query(self, name: str):
        if name is None:
            raise ValueError("name 不能为空")

        cursor = self.conn.cursor()

        cursor.execute("""
        SELECT name, width, height, frame_count, fps
        FROM video_meta
        WHERE name = ?
        """, (name,))

        row = cursor.fetchone()

        if row is None:
            return None

        return row[1:5]

    def close(self):
        if self.conn is not None:
            self.conn.commit()
            self.conn.close()
            self.conn = None

    def __del__(self):
        try:
            self.close()
            print("VideoMeta closed")
        except Exception:
            pass


class VideoDataset(paddle.io.Dataset):
    """"Video tile dataset with fast ffmpeg loading."""

    def __init__(self, path: str, first_n: int = None):
        self._path = path
        self._meta = VideoMeta(os.path.join(path, "video_meta.sqlite"))
        self._index_mapping = []  # (name, (x_off, y_off, t_off))

        names = [name for name in os.listdir(path) if name.endswith(".mp4")]
        names.sort()

        if first_n is not None:
            names = names[:first_n]

        total_str = str(len(names))
        total_len = len(total_str)

        for i, name in enumerate(names, 1):
            print(f"reading [{i:>{total_len}}/{total_str}] {name}")
            if (meta := self._meta.query(name)) is None:
                video_path = os.path.join(path, name)
                meta = get_video_info(video_path)
                self._meta.insert(name, *meta)

            width, height, frame_count, _ = meta
            tiles = tile_video(width, height, frame_count)
            for tile in tiles:
                self._index_mapping.append((name, tile))

        print(f"read {len(self)} tiles")

    def __getitem__(self, idx):
        name, (x_off, y_off, t_off) = self._index_mapping[idx]
        width, height, _, fps = self._meta.query(name)
        frames = read_frame_range_fast(
            os.path.join(self._path, name), width, height, fps,
            t_off, 30, x_off, y_off,
        )
        return simple_mask(frames)

    def __len__(self):
        return len(self._index_mapping)


class FakeDataset(paddle.io.Dataset):
    """纯随机数据，用于 smoke test。"""

    def __init__(self, length: int, num_frames: int, h: int, w: int):
        self.length = length
        self.num_frames = num_frames
        self.h = h
        self.w = w

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 返回 numpy，DataLoader 会自动 collate 成 paddle.Tensor
        frame_size = (self.num_frames, self.h, self.w, 3)
        mask_size = (self.h, self.w, 1)
        return {
            "frames": np.random.randint(256, size=frame_size, dtype=np.uint8),
            "target": np.random.randint(256, size=frame_size, dtype=np.uint8),
            "mask": np.random.randint(2, size=mask_size, dtype=np.uint8),
            "dilated_mask": np.random.randint(2, size=mask_size, dtype=np.uint8),
        }


class ImageDataset(paddle.io.Dataset):
    IMAGE_SUFFIXES = {
        ".jpg", ".jpeg", ".png", ".gif", ".bmp",
        ".webp", ".tiff", ".tif", ".svg", ".ico"
    }
    TILE_ARGS = dict(frame_count=1, tile_w=256, tile_h=256, tile_t=1)

    def __init__(
        self, path: str, first_n: int = None, tile_args: dict = None, fix_mask: bool = False
    ):
        self._path = path
        self._meta = VideoMeta(os.path.join(path, "image_meta.sqlite"))
        self._index_mapping = []  # (fp, (x_off, y_off))
        self.tile_args = dict(self.TILE_ARGS)
        if tile_args is not None:
            self.tile_args.update(tile_args)
        self.fix_mask = fix_mask
        file_count = 0

        for fp in Path(path).rglob("*"):
            if not (fp.is_file() and fp.suffix.lower() in self.IMAGE_SUFFIXES):
                continue

            name = str(fp).removeprefix(path).lstrip("/")
            if (meta := self._meta.query(name)) is None:
                with Image.open(fp) as img:
                    width, height = img.size
                self._meta.insert(name, width, height, frame_count=1, fps=0)
            else:
                width, height, _, _ = meta

            for x_off, y_off, _ in tile_video(width, height, **self.tile_args):
                self._index_mapping.append((fp, (x_off, y_off)))

            file_count += 1
            if first_n is not None and file_count >= first_n:
                break
            if file_count % 1000 == 0:
                print(f"read {file_count} files...")

        print(f"read {file_count} files, {len(self)} tiles")

    def __getitem__(self, idx):
        fp, (x_off, y_off) = self._index_mapping[idx]

        W, H = self.TILE_ARGS["tile_w"], self.TILE_ARGS["tile_h"]
        w, h = self.tile_args["tile_w"], self.tile_args["tile_h"]

        image = Image.open(fp)
        image = image.crop((x_off, y_off, x_off + w, y_off + h))
        if image.size != (W, H):
            image = image.resize((W, H), Image.Resampling.LANCZOS)
        image = np.asarray(image)

        seed = zlib.crc32(f"{fp}\0{idx}".encode()) if self.fix_mask else None
        return simple_mask(image, seed)

    def __len__(self):
        return len(self._index_mapping)


if __name__ == "__main__":
    dataset = ImageDataset("/root/autodl-tmp/DIV2K_Flicker2K", 10)

    data = dataset[30]

    save_image_grid(
        data["frames"], data["target"], data["target"] // 2,
        data["mask"] * 100 + data["dilated_mask"] * 100 + 50,
        "/root/autodl-tmp/grid.png",
    )

    exit()

    dataset = VideoDataset("~/UltraVideo/clips_short_1920_1", 10)

    data = dataset[152]

    save_video_with_ffmpeg(data["frames"], "~/frames.mp4")
    save_video_with_ffmpeg(data["target"], "~/target.mp4")
