"""Continuous zoom-to-viewpoint schedules and diagnostics."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np


@dataclass
class ZoomPoint:
    zoom: float
    progress: float
    alpha: float
    beta: float


class ZoomSchedule:
    """Map optical zoom ratios to smooth viewpoint progress.

    ``interpolation`` controls ratio normalization (linear or log); ``curve``
    controls easing. Endpoint beta is separate from paper-view alpha.
    """

    def __init__(
        self,
        wide_zoom: float,
        tele_zoom: float,
        interpolation: str = "log",
        curve: str = "smootherstep",
        endpoint_mode: str = "tele_endpoint",
        terminal_start: float = 0.8,
        custom_values: Optional[Sequence[float]] = None,
    ) -> None:
        if wide_zoom <= 0 or tele_zoom <= 0 or tele_zoom <= wide_zoom:
            raise ValueError("Require 0 < wide_zoom < tele_zoom")
        if not 0.0 <= terminal_start < 1.0:
            raise ValueError("terminal_start must be in [0,1)")
        self.wide_zoom = float(wide_zoom)
        self.tele_zoom = float(tele_zoom)
        self.interpolation = interpolation.lower()
        self.curve = curve.lower()
        self.endpoint_mode = endpoint_mode.lower()
        self.terminal_start = float(terminal_start)
        self.custom_values = None if custom_values is None else np.asarray(custom_values, dtype=np.float64)
        if self.interpolation not in ("linear", "log"):
            raise ValueError("interpolation must be linear or log")
        if self.curve not in ("linear", "smoothstep", "smootherstep", "cosine", "custom"):
            raise ValueError("Unsupported zoom curve: %s" % curve)
        if self.endpoint_mode not in ("paper_mixed", "tele_endpoint"):
            raise ValueError("endpoint_mode must be paper_mixed or tele_endpoint")
        if self.curve == "custom" and (self.custom_values is None or len(self.custom_values) < 2):
            raise ValueError("custom curve needs at least two samples")

    def normalized_progress(self, zoom: float) -> float:
        if zoom < self.wide_zoom - 1.0e-8 or zoom > self.tele_zoom + 1.0e-8:
            raise ValueError(
                "zoom ratio %.6f is outside [%.6f, %.6f]"
                % (zoom, self.wide_zoom, self.tele_zoom)
            )
        zoom = min(max(float(zoom), self.wide_zoom), self.tele_zoom)
        if self.interpolation == "log":
            return (math.log(zoom) - math.log(self.wide_zoom)) / (
                math.log(self.tele_zoom) - math.log(self.wide_zoom)
            )
        return (zoom - self.wide_zoom) / (self.tele_zoom - self.wide_zoom)

    def ease(self, progress: float) -> float:
        value = min(max(float(progress), 0.0), 1.0)
        if self.curve == "linear":
            return value
        if self.curve == "smoothstep":
            return value * value * (3.0 - 2.0 * value)
        if self.curve == "smootherstep":
            return value ** 3 * (value * (value * 6.0 - 15.0) + 10.0)
        if self.curve == "cosine":
            return 0.5 - 0.5 * math.cos(math.pi * value)
        positions = np.linspace(0.0, 1.0, len(self.custom_values))
        return float(np.interp(value, positions, self.custom_values))

    @staticmethod
    def _smoothstep(value: float) -> float:
        value = min(max(value, 0.0), 1.0)
        return value * value * (3.0 - 2.0 * value)

    def __call__(self, zoom: float) -> ZoomPoint:
        progress = self.normalized_progress(zoom)
        alpha = self.ease(progress)
        beta = 0.0
        if self.endpoint_mode == "tele_endpoint" and alpha > self.terminal_start:
            beta = self._smoothstep((alpha - self.terminal_start) / (1.0 - self.terminal_start))
        return ZoomPoint(float(zoom), progress, alpha, beta)

    def sample(self, start: float, end: float, frames: int) -> List[ZoomPoint]:
        if frames <= 0:
            raise ValueError("frames must be positive")
        ratios = np.linspace(float(start), float(end), int(frames))
        return [self(float(value)) for value in ratios]


def write_schedule_csv(path: str, points: Iterable[ZoomPoint], fps: float = 1.0) -> Path:
    rows = list(points)
    if not rows:
        raise ValueError("At least one zoom point is required")
    if fps <= 0:
        raise ValueError("fps must be positive")
    alphas = np.asarray([item.alpha for item in rows], dtype=np.float64)
    timestep = 1.0 / float(fps)
    first = np.gradient(alphas, timestep) if len(rows) > 1 else np.zeros_like(alphas)
    second = np.gradient(first, timestep) if len(rows) > 2 else np.zeros_like(alphas)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame", "zoom_ratio", "progress", "alpha", "beta", "d_alpha", "dd_alpha"])
        for index, item in enumerate(rows):
            writer.writerow([index, item.zoom, item.progress, item.alpha, item.beta, first[index], second[index]])
    return output


__all__ = ["ZoomPoint", "ZoomSchedule", "write_schedule_csv"]
