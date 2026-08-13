"""Paper-style and forward/backward-consistency occlusion masks.

Mask semantics are project-wide and intentionally explicit:

    M_occ = 1 -> select transformed Wide
    M_occ = 0 -> select transformed Tele
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
from scipy import ndimage

from utils.flow_utils import flow_magnitude, same_box_filter
from utils.warp import backward_warp


@dataclass
class OcclusionResult:
    hard_mask: torch.Tensor
    soft_mask: torch.Tensor
    consistency_error: Optional[torch.Tensor] = None


def soften_occlusion(mask: torch.Tensor, width: int = 15) -> torch.Tensor:
    """Linearly feather a hard Wide-selection mask out into Tele regions."""
    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError("mask must have shape [B,1,H,W]")
    if width <= 0:
        return (mask > 0.5).to(mask.dtype)
    output = np.zeros(tuple(mask.shape), dtype=np.float32)
    source = mask.detach().cpu().numpy() > 0.5
    for batch in range(mask.shape[0]):
        hard = source[batch, 0]
        if not np.any(hard):
            continue
        distance_outside = ndimage.distance_transform_edt(~hard)
        softened = np.maximum(1.0 - distance_outside / float(width), 0.0)
        softened[hard] = 1.0
        output[batch, 0] = softened
    return torch.from_numpy(output).to(device=mask.device, dtype=mask.dtype)


def _paper_foreground(flow: torch.Tensor, mode: str) -> torch.Tensor:
    mean = same_box_filter(flow, min(31, max(3, min(flow.shape[-2:]) // 2 * 2 + 1)))
    if mode == "component":
        return torch.any(torch.abs(flow) > torch.abs(mean), dim=1, keepdim=True).to(flow.dtype)
    if mode != "magnitude":
        raise ValueError("foreground mode must be magnitude or component")
    return (flow_magnitude(flow) > flow_magnitude(mean)).to(flow.dtype)


def paper_occlusion(
    flow_t2o: torch.Tensor,
    foreground_mask: Optional[torch.Tensor] = None,
    camera_x: str = "left",
    camera_y: str = "upper",
    foreground_mode: str = "magnitude",
    rectangle_scale: float = 1.0,
    max_rectangle: int = 256,
    max_boundary_points: int = 20000,
    tele_valid: Optional[torch.Tensor] = None,
    wide_valid: Optional[torch.Tensor] = None,
    soft_width: int = 15,
) -> OcclusionResult:
    """Approximate the paper's directional flow-discontinuity rectangles.

    PAPER_AMBIGUITY: the article gives a geometric description and illustration,
    but not executable boundary traversal/rasterization rules. This reference
    checks foreground pixels adjacent to background in the configured camera
    directions, then paints a rectangle sized by their flow-vector difference.
    """
    if flow_t2o.ndim != 4 or flow_t2o.shape[1] != 2:
        raise ValueError("flow_t2o must have shape [B,2,H,W]")
    if camera_x not in ("left", "right"):
        raise ValueError("camera_x must be left or right")
    if camera_y not in ("upper", "lower"):
        raise ValueError("camera_y must be upper or lower")
    foreground = foreground_mask if foreground_mask is not None else _paper_foreground(flow_t2o, foreground_mode)
    if foreground.shape != flow_t2o[:, :1].shape:
        raise ValueError("foreground_mask must have shape [B,1,H,W]")

    flow_np = flow_t2o.detach().cpu().numpy()
    foreground_np = foreground.detach().cpu().numpy()[:, 0] > 0.5
    masks = np.zeros((flow_t2o.shape[0], 1, flow_t2o.shape[2], flow_t2o.shape[3]), dtype=np.float32)
    height, width = flow_t2o.shape[-2:]
    directions = []
    directions.append((0, -1) if camera_x == "left" else (0, 1))
    directions.append((-1, 0) if camera_y == "upper" else (1, 0))

    for batch in range(flow_t2o.shape[0]):
        fg = foreground_np[batch]
        canvas = masks[batch, 0]
        for delta_y, delta_x in directions:
            shifted_background = np.zeros_like(fg)
            if delta_x == -1:
                shifted_background[:, 1:] = ~fg[:, :-1]
            elif delta_x == 1:
                shifted_background[:, :-1] = ~fg[:, 1:]
            elif delta_y == -1:
                shifted_background[1:, :] = ~fg[:-1, :]
            else:
                shifted_background[:-1, :] = ~fg[1:, :]
            candidates = np.argwhere(fg & shifted_background)
            if max_boundary_points > 0 and len(candidates) > max_boundary_points:
                sample_indices = np.linspace(
                    0, len(candidates) - 1, max_boundary_points, dtype=np.int64
                )
                candidates = candidates[sample_indices]
            for y, x in candidates:
                neighbor_y, neighbor_x = y + delta_y, x + delta_x
                difference = flow_np[batch, :, y, x] - flow_np[batch, :, neighbor_y, neighbor_x]
                extent_x = min(max_rectangle, max(1, int(np.ceil(abs(float(difference[0])) * rectangle_scale))))
                extent_y = min(max_rectangle, max(1, int(np.ceil(abs(float(difference[1])) * rectangle_scale))))
                x0, x1 = (x - extent_x, x) if camera_x == "left" else (x, x + extent_x)
                y0, y1 = (y - extent_y, y) if camera_y == "upper" else (y, y + extent_y)
                cv2.rectangle(
                    canvas,
                    (max(0, x0), max(0, y0)),
                    (min(width - 1, x1), min(height - 1, y1)),
                    1.0,
                    thickness=-1,
                )

    hard = torch.from_numpy(masks).to(device=flow_t2o.device, dtype=flow_t2o.dtype)
    if tele_valid is not None:
        hard = torch.maximum(hard, (tele_valid < 0.5).to(hard.dtype))
    if wide_valid is not None:
        hard = hard * (wide_valid > 0.5).to(hard.dtype)
    hard = (hard > 0.5).to(flow_t2o.dtype)
    return OcclusionResult(hard, soften_occlusion(hard, soft_width))


def fb_consistency_occlusion(
    flow_t2w: torch.Tensor,
    flow_w2t: torch.Tensor,
    absolute_threshold: float = 1.0,
    relative_threshold: float = 0.05,
    tele_valid: Optional[torch.Tensor] = None,
    wide_valid: Optional[torch.Tensor] = None,
    soft_width: int = 15,
) -> OcclusionResult:
    """Compute consistency in W coordinates: F_T2W + sample(F_W2T,p+F)."""
    if flow_t2w.shape != flow_w2t.shape:
        raise ValueError("forward and reverse flow must have identical [B,2,H,W] shapes")
    sampled_reverse, correspondence_valid = backward_warp(flow_w2t, flow_t2w, return_valid=True)
    residual = flow_t2w + sampled_reverse
    error = flow_magnitude(residual)
    scale = flow_magnitude(flow_t2w) + flow_magnitude(sampled_reverse)
    threshold = float(absolute_threshold) + float(relative_threshold) * scale
    hard = ((error > threshold) | (correspondence_valid < 0.5)).to(flow_t2w.dtype)
    if tele_valid is not None:
        hard = torch.maximum(hard, (tele_valid < 0.5).to(hard.dtype))
    if wide_valid is not None:
        hard = hard * (wide_valid > 0.5).to(hard.dtype)
    return OcclusionResult(hard, soften_occlusion(hard, soft_width), error)


__all__ = [
    "OcclusionResult",
    "fb_consistency_occlusion",
    "paper_occlusion",
    "soften_occlusion",
]
