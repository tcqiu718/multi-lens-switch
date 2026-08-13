"""Per-frame geometry, usage, brightness, and temporal diagnostics."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from utils.flow_utils import flow_magnitude
from utils.warp import backward_warp
from zoom.zoom_pipeline import PreparedPair, ZoomFrameResult


class MetricsTracker:
    def __init__(self, rho: float, fps: float = 1.0) -> None:
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.rho = float(rho)
        self.fps = float(fps)
        self.rows: List[Dict[str, float]] = []
        self.previous_image = None
        self.previous_base_image = None
        self.previous_mask = None
        self.previous_brightness = None

    def update(
        self,
        frame: int,
        pair: PreparedPair,
        result: ZoomFrameResult,
        temporal_flow: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        image = result.result.detach()
        brightness = float(image.mean().cpu())
        temporal_difference = 0.0
        temporal_warping_error = 0.0
        mask_difference = 0.0
        brightness_difference = 0.0
        if self.previous_image is not None:
            temporal_difference = float(torch.mean(torch.abs(image - self.previous_image)).cpu())
            prediction = self.previous_base_image
            if temporal_flow is not None:
                prediction = backward_warp(prediction, temporal_flow)
            temporal_warping_error = float(
                torch.mean(torch.abs(result.fusion.full_result.detach() - prediction)).cpu()
            )
            previous_mask = self.previous_mask
            if temporal_flow is not None:
                previous_mask = backward_warp(previous_mask, temporal_flow)
            mask_difference = float(
                torch.mean(torch.abs(result.occlusion.soft_mask - previous_mask)).cpu()
            )
            brightness_difference = abs(brightness - float(self.previous_brightness))
        consistency = result.occlusion.consistency_error
        row = {
            "frame": int(frame),
            "zoom": result.zoom_point.zoom,
            "progress": result.zoom_point.progress,
            "alpha": result.zoom_point.alpha,
            "beta": result.zoom_point.beta,
            "rho": self.rho,
            "crop": result.fov.crop_scale,
            "crop_x0": result.fov.crop_box[0],
            "crop_y0": result.fov.crop_box[1],
            "crop_x1": result.fov.crop_box[2],
            "crop_y1": result.fov.crop_box[3],
            "mean_flow": float(flow_magnitude(pair.flow_t2w).mean().cpu()),
            "mean_delta": float(flow_magnitude(result.view.delta_flow).mean().cpu()),
            "occlusion_ratio": float(result.occlusion.hard_mask.mean().cpu()),
            "tele_usage_ratio": result.tele_usage_ratio,
            "brightness": brightness,
            "brightness_difference": brightness_difference,
            "temporal_difference": temporal_difference,
            "temporal_warping_error": temporal_warping_error,
            "flow_consistency": 0.0 if consistency is None else float(consistency.mean().cpu()),
            "mask_temporal_difference": mask_difference,
        }
        self.rows.append(row)
        self.previous_image = image
        self.previous_base_image = result.fusion.full_result.detach()
        self.previous_mask = result.occlusion.soft_mask.detach()
        self.previous_brightness = brightness
        return row

    def write(self, path: str) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        if not self.rows:
            raise ValueError("No metrics have been accumulated")
        alphas = np.asarray([row["alpha"] for row in self.rows], dtype=np.float64)
        timestep = 1.0 / self.fps
        first = np.gradient(alphas, timestep) if len(alphas) > 1 else np.zeros_like(alphas)
        second = np.gradient(first, timestep) if len(alphas) > 2 else np.zeros_like(alphas)
        rows = []
        for index, row in enumerate(self.rows):
            copied = dict(row)
            copied["d_alpha"] = float(first[index])
            copied["dd_alpha"] = float(second[index])
            rows.append(copied)
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return output


__all__ = ["MetricsTracker"]
