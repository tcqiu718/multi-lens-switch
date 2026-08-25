"""Continuous virtual-view geometry between native Wide and Tele frames."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from utils.camera_geometry import interpolate_intrinsics, intrinsics_matrix
from utils.warp import coordinate_grid, forward_warp
from view_transition.flow_transform import interpolate_transformation
from view_transition.transform_tele import TeleTransformResult
from view_transition.transform_wide import WideTransformResult, transform_wide_to_output
from view_transition.view_transition import ViewTransition


@dataclass
class ContinuousViewResult:
    alpha: float
    transformed_flow: torch.Tensor
    delta_flow: torch.Tensor
    tele: TeleTransformResult
    wide: WideTransformResult
    wide_motion: Optional[torch.Tensor] = None
    tele_motion: Optional[torch.Tensor] = None
    overlap_mask: Optional[torch.Tensor] = None


def _center_scale_grid(grid: torch.Tensor, scale: float) -> torch.Tensor:
    height, width = grid.shape[-2:]
    center = grid.new_tensor([(width - 1) * 0.5, (height - 1) * 0.5]).view(1, 2, 1, 1)
    return center + float(scale) * (grid - center)


def _intrinsics_grid(
    grid: torch.Tensor,
    source_intrinsics: Sequence[Sequence[float]],
    target_intrinsics: torch.Tensor,
) -> torch.Tensor:
    source = intrinsics_matrix(source_intrinsics, grid.device, grid.dtype)
    homogeneous = torch.cat((grid, torch.ones_like(grid[:, :1])), dim=1).flatten(2)
    target = target_intrinsics @ torch.linalg.inv(source)
    mapped = torch.matmul(target.unsqueeze(0), homogeneous)
    mapped_xy = mapped[:, :2] / mapped[:, 2:3].clamp_min(1.0e-8)
    return mapped_xy.view_as(grid)


def _target_grids(
    base: torch.Tensor,
    wide_correspondence: torch.Tensor,
    tele_correspondence: torch.Tensor,
    target_zoom: float,
    wide_zoom: float,
    tele_zoom: float,
    fov_progress: float,
    fov_mode: str,
    wide_intrinsics: Optional[Sequence[Sequence[float]]],
    tele_intrinsics: Optional[Sequence[Sequence[float]]],
) -> tuple:
    normalized = str(fov_mode).lower()
    if normalized == "center_crop":
        wide_scale = float(target_zoom) / float(wide_zoom)
        tele_scale = float(target_zoom) / float(tele_zoom)
        return (
            _center_scale_grid(base, wide_scale),
            _center_scale_grid(tele_correspondence, tele_scale),
            _center_scale_grid(wide_correspondence, wide_scale),
            _center_scale_grid(base, tele_scale),
        )
    if normalized != "intrinsics":
        raise ValueError("zoom.fov_mode must be center_crop or intrinsics")
    if wide_intrinsics is None or tele_intrinsics is None:
        raise ValueError("intrinsics FOV mode requires wide_intrinsics and tele_intrinsics")
    wide_k = intrinsics_matrix(wide_intrinsics, base.device, base.dtype)
    tele_k = intrinsics_matrix(tele_intrinsics, base.device, base.dtype)
    target_k = interpolate_intrinsics(wide_k, tele_k, float(fov_progress))
    return (
        _intrinsics_grid(base, wide_intrinsics, target_k),
        _intrinsics_grid(tele_correspondence, tele_intrinsics, target_k),
        _intrinsics_grid(wide_correspondence, wide_intrinsics, target_k),
        _intrinsics_grid(base, tele_intrinsics, target_k),
    )


def bidirectional_zoom_transition(
    wide: torch.Tensor,
    tele: torch.Tensor,
    flow_t2w: torch.Tensor,
    flow_w2t: torch.Tensor,
    overlap_mask: torch.Tensor,
    alpha: float,
    target_zoom: float,
    wide_zoom: float,
    tele_zoom: float,
    fov_progress: float,
    fov_mode: str = "center_crop",
    wide_intrinsics: Optional[Sequence[Sequence[float]]] = None,
    tele_intrinsics: Optional[Sequence[Sequence[float]]] = None,
    splat_mode: str = "bilinear",
    multi_warp_average: bool = True,
    tele_multi_warp_average: bool = False,
    offset_min: float = -0.5,
    offset_max: float = 0.5,
    offset_step: float = 0.2,
) -> ContinuousViewResult:
    """Project both cameras to one target-FOV virtual view with exact endpoints.

    ``flow_t2w`` is the project backward-sampling field defined on Wide pixels:
    its correspondence in Tele is ``p_t = p_w + flow_t2w(p_w)``. The reverse
    field analogously supplies ``p_w`` for every Tele pixel.
    """
    value = float(alpha)
    if not 0.0 <= value <= 1.0:
        raise ValueError("alpha must be in [0,1]")
    if wide.shape != tele.shape or flow_t2w.shape != flow_w2t.shape:
        raise ValueError("Wide/Tele images and both flow fields must have matching shapes")
    if flow_t2w.shape != (wide.shape[0], 2, wide.shape[2], wide.shape[3]):
        raise ValueError("flow fields must have shape [B,2,H,W] matching the images")
    if overlap_mask.shape != wide[:, :1].shape:
        raise ValueError("overlap_mask must have shape [B,1,H,W]")

    base = coordinate_grid(wide.shape[0], wide.shape[2], wide.shape[3], wide.device, flow_t2w.dtype)
    tele_correspondence = base + flow_t2w
    wide_correspondence = base + flow_w2t
    wide_native, tele_from_wide, wide_from_tele, tele_native = _target_grids(
        base,
        wide_correspondence,
        tele_correspondence,
        target_zoom,
        wide_zoom,
        tele_zoom,
        fov_progress,
        fov_mode,
        wide_intrinsics,
        tele_intrinsics,
    )
    wide_target = torch.lerp(wide_native, tele_from_wide, value)
    tele_target = torch.lerp(wide_from_tele, tele_native, value)
    wide_motion = wide_target - base
    tele_motion = tele_target - base

    transform_kwargs = {
        "splat_mode": splat_mode,
        "multi_warp_average": multi_warp_average,
        "offset_min": offset_min,
        "offset_max": offset_max,
        "offset_step": offset_step,
    }
    wide_result = transform_wide_to_output(wide, wide_motion, **transform_kwargs)
    tele_transform_kwargs = dict(transform_kwargs)
    tele_transform_kwargs["multi_warp_average"] = tele_multi_warp_average
    tele_warp = transform_wide_to_output(tele, tele_motion, **tele_transform_kwargs)
    tele_result = TeleTransformResult(
        tele_warp.image,
        tele_motion,
        tele_motion,
        tele_warp.weight,
        tele_warp.valid_mask,
        1.0 - tele_warp.valid_mask,
    )
    moved_overlap, _, overlap_valid = forward_warp(overlap_mask, wide_motion, mode=splat_mode)
    moved_overlap = moved_overlap.clamp(0.0, 1.0) * overlap_valid
    return ContinuousViewResult(
        value,
        tele_motion,
        wide_motion,
        tele_result,
        wide_result,
        wide_motion,
        tele_motion,
        moved_overlap,
    )


def continuous_view_transition(
    engine: ViewTransition,
    wide: torch.Tensor,
    tele: torch.Tensor,
    original_flow: torch.Tensor,
    paper_delta: torch.Tensor,
    foreground_mask: torch.Tensor,
    alpha: float,
) -> ContinuousViewResult:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0,1]")
    interpolation = interpolate_transformation(original_flow, paper_delta, alpha)
    tele_result, wide_result = engine.transform(
        wide,
        tele,
        interpolation.transformed_flow,
        interpolation.delta_flow,
        foreground_mask,
    )
    return ContinuousViewResult(
        float(alpha),
        interpolation.transformed_flow,
        interpolation.delta_flow,
        tele_result,
        wide_result,
    )


__all__ = [
    "ContinuousViewResult",
    "bidirectional_zoom_transition",
    "continuous_view_transition",
]
