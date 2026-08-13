"""Synchronized RGB video/image-sequence I/O."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from utils.image_io import tensor_to_rgb8


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int


def _rgb_array_to_tensor(rgb: np.ndarray, device: Optional[torch.device]) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return tensor.to(device=device) if device is not None else tensor


class RGBVideoReader:
    def __init__(self, path: str, device: Optional[torch.device] = None) -> None:
        self.path = Path(path).expanduser()
        self.capture = cv2.VideoCapture(str(self.path))
        if not self.capture.isOpened():
            raise FileNotFoundError("Could not open video: %s" % self.path)
        self.device = device
        self.info = VideoInfo(
            width=int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(self.capture.get(cv2.CAP_PROP_FPS)),
            frame_count=int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        )

    def read(self) -> Optional[torch.Tensor]:
        success, bgr = self.capture.read()
        if not success:
            return None
        return _rgb_array_to_tensor(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), self.device)

    def release(self) -> None:
        self.capture.release()

    def __enter__(self) -> "RGBVideoReader":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()


class PairedVideoReader:
    def __init__(self, wide_path: str, tele_path: str, device: Optional[torch.device] = None) -> None:
        self.wide = RGBVideoReader(wide_path, device)
        try:
            self.tele = RGBVideoReader(tele_path, device)
        except Exception:
            self.wide.release()
            raise
        if (self.wide.info.width, self.wide.info.height) != (self.tele.info.width, self.tele.info.height):
            self.release()
            raise ValueError("Wide/Tele video resolutions do not match")
        if abs(self.wide.info.fps - self.tele.info.fps) > 1.0e-3:
            self.release()
            raise ValueError(
                "Wide/Tele FPS mismatch: %.6f versus %.6f"
                % (self.wide.info.fps, self.tele.info.fps)
            )
        if (
            self.wide.info.frame_count > 0
            and self.tele.info.frame_count > 0
            and self.wide.info.frame_count != self.tele.info.frame_count
        ):
            self.release()
            raise ValueError(
                "Wide/Tele frame-count mismatch: %d versus %d"
                % (self.wide.info.frame_count, self.tele.info.frame_count)
            )
        self.info = self.wide.info

    def read(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        wide = self.wide.read()
        tele = self.tele.read()
        if (wide is None) != (tele is None):
            raise ValueError("Wide/Tele streams ended at different frames")
        return None if wide is None else (wide, tele)

    def release(self) -> None:
        self.wide.release()
        self.tele.release()

    def __enter__(self) -> "PairedVideoReader":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()


def _image_files(directory: str) -> List[Path]:
    root = Path(directory).expanduser()
    if not root.is_dir():
        raise FileNotFoundError("Image-sequence directory not found: %s" % root)
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = sorted(path for path in root.iterdir() if path.suffix.lower() in extensions)
    if not files:
        raise ValueError("No supported images found in %s" % root)
    return files


class PairedImageSequence:
    def __init__(self, wide_dir: str, tele_dir: str, device: Optional[torch.device] = None) -> None:
        self.wide_files = _image_files(wide_dir)
        self.tele_files = _image_files(tele_dir)
        if len(self.wide_files) != len(self.tele_files):
            raise ValueError("Wide/Tele image-sequence counts do not match")
        if [item.stem for item in self.wide_files] != [item.stem for item in self.tele_files]:
            raise ValueError("Wide/Tele image-sequence filenames do not align")
        self.device = device
        self.index = 0
        first_wide = cv2.imread(str(self.wide_files[0]), cv2.IMREAD_COLOR)
        first_tele = cv2.imread(str(self.tele_files[0]), cv2.IMREAD_COLOR)
        if first_wide is None or first_tele is None:
            raise ValueError("Could not read first image-sequence frame")
        if first_wide.shape != first_tele.shape:
            raise ValueError("Wide/Tele image-sequence resolutions do not match")
        self.info = VideoInfo(first_wide.shape[1], first_wide.shape[0], 0.0, len(self.wide_files))

    def read(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if self.index >= len(self.wide_files):
            return None
        wide = cv2.imread(str(self.wide_files[self.index]), cv2.IMREAD_COLOR)
        tele = cv2.imread(str(self.tele_files[self.index]), cv2.IMREAD_COLOR)
        self.index += 1
        if wide is None or tele is None:
            raise ValueError("Could not read image sequence at index %d" % (self.index - 1))
        if wide.shape != tele.shape:
            raise ValueError("Wide/Tele resolution mismatch at frame %d" % (self.index - 1))
        return (
            _rgb_array_to_tensor(cv2.cvtColor(wide, cv2.COLOR_BGR2RGB), self.device),
            _rgb_array_to_tensor(cv2.cvtColor(tele, cv2.COLOR_BGR2RGB), self.device),
        )

    def release(self) -> None:
        return None


class RGBVideoWriter:
    def __init__(self, path: str, width: int, height: int, fps: float, codec: str = "mp4v") -> None:
        if fps <= 0 or width <= 0 or height <= 0:
            raise ValueError("Video width, height, and FPS must be positive")
        if len(codec) != 4:
            raise ValueError("codec must be a four-character code")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width, self.height = int(width), int(height)
        self.writer = cv2.VideoWriter(
            str(self.path), cv2.VideoWriter_fourcc(*codec), float(fps), (self.width, self.height)
        )
        if not self.writer.isOpened():
            raise IOError("Could not create video writer: %s" % self.path)

    def write(self, image) -> None:
        rgb = tensor_to_rgb8(image) if isinstance(image, torch.Tensor) else np.asarray(image)
        if rgb.shape != (self.height, self.width, 3):
            raise ValueError(
                "Video frame shape %s does not match (%d,%d,3)"
                % (rgb.shape, self.height, self.width)
            )
        self.writer.write(cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR))

    def release(self) -> None:
        self.writer.release()

    def __enter__(self) -> "RGBVideoWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()


def load_zoom_ratios(path: str) -> List[float]:
    ratios = []
    with Path(path).expanduser().open("r", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        for row_index, row in enumerate(reader):
            if not row:
                continue
            try:
                value = float(row[-1])
            except ValueError:
                if row_index == 0:
                    continue
                raise ValueError("Invalid zoom schedule row %d: %s" % (row_index + 1, row))
            ratios.append(value)
    if not ratios:
        raise ValueError("Zoom schedule contains no numeric ratios")
    return ratios


__all__ = [
    "PairedImageSequence",
    "PairedVideoReader",
    "RGBVideoReader",
    "RGBVideoWriter",
    "VideoInfo",
    "load_zoom_ratios",
]
