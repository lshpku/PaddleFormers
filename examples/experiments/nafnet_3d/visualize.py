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

import argparse
import os

import cv2
import numpy as np


def visualize_patch(patch_path, out_path=None, fps=30):
    arr = np.load(patch_path)
    print(f"loaded: {patch_path}, shape={arr.shape}, dtype={arr.dtype}")

    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"expect [T, H, W, 3], got {arr.shape}")

    T, H, W, C = arr.shape

    if out_path is None:
        base = os.path.splitext(patch_path)[0]
        out_path = base + "_vis.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open VideoWriter for {out_path}")

    for t in range(T):
        frame = arr[t]
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()
    print(f"saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("patch_path", help="path to .npy patch")
    parser.add_argument("--out_path", default=None, help="output video path")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    visualize_patch(args.patch_path, args.out_path, args.fps)
