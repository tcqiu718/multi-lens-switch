"""Geometry-only paper View Transition pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from view_transition.flow_boundary import BoundaryResult, distance_to_boundary
from view_transition.flow_transform import ConstrainedFlowResult, constrain_target_flow
from view_transition.target_flow import TargetFlowResult, estimate_target_flow
from view_transition.transform_tele import TeleTransformResult, transform_tele_to_output
from view_transition.transform_wide import WideTransformResult, transform_wide_to_output


@dataclass
class ViewTransitionResult:
    target: TargetFlowResult
    boundary: BoundaryResult
    constrained: ConstrainedFlowResult
    tele: TeleTransformResult
    wide: WideTransformResult


class ViewTransition:
    """Reusable geometry stage; optical-flow estimation stays outside for caching."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config

    def prepare_flow(self, flow_t2w: torch.Tensor) -> tuple:
        target_cfg = self.config.get("target_flow", {})
        target = estimate_target_flow(
            flow_t2w,
            kernel_size=int(target_cfg.get("kernel_size", 600)),
            foreground_mode=str(target_cfg.get("foreground_mode", "magnitude")),
            eps=float(target_cfg.get("eps", 1.0e-6)),
        )
        boundary_cfg = self.config.get("boundary", {})
        boundary = distance_to_boundary(
            flow_t2w,
            mode=str(boundary_cfg.get("mode", "flow_aware")),
            gradient_threshold=boundary_cfg.get("gradient_threshold"),
            gradient_quantile=float(boundary_cfg.get("gradient_quantile", 0.90)),
            max_extra_distance=int(boundary_cfg.get("max_extra_distance", 256)),
        )
        ratio_cfg = self.config.get("view_transition", self.config.get("constraint", {}))
        ratio = float(ratio_cfg.get("ratio", 0.01))
        constrained = constrain_target_flow(flow_t2w, target.target_flow, boundary.distance, ratio)
        return target, boundary, constrained

    def transform(
        self,
        wide: torch.Tensor,
        tele: torch.Tensor,
        transformed_flow: torch.Tensor,
        delta_flow: torch.Tensor,
        foreground_mask: Optional[torch.Tensor] = None,
    ) -> tuple:
        cfg = self.config.get("transform", {})
        tele_result = transform_tele_to_output(
            tele,
            transformed_flow,
            delta_flow,
            foreground_mask=foreground_mask,
            splat_mode=str(cfg.get("splat_mode", "bilinear")),
            fill_mode=str(cfg.get("fill_mode", "background_nearest")),
        )
        wide_result = transform_wide_to_output(
            wide,
            delta_flow,
            splat_mode=str(cfg.get("splat_mode", "bilinear")),
            multi_warp_average=bool(cfg.get("multi_warp_average", True)),
            offset_min=float(cfg.get("wide_offset_min", -0.5)),
            offset_max=float(cfg.get("wide_offset_max", 0.5)),
            offset_step=float(cfg.get("wide_offset_step", 0.2)),
        )
        return tele_result, wide_result

    def __call__(self, wide: torch.Tensor, tele: torch.Tensor, flow_t2w: torch.Tensor) -> ViewTransitionResult:
        if wide.shape != tele.shape:
            raise ValueError("Wide and Tele tensors must have equal [B,3,H,W] shapes")
        if flow_t2w.shape != (wide.shape[0], 2, wide.shape[2], wide.shape[3]):
            raise ValueError("F_T2W must have shape [B,2,H,W] matching input images")
        target, boundary, constrained = self.prepare_flow(flow_t2w)
        tele_result, wide_result = self.transform(
            wide,
            tele,
            constrained.transformed_flow,
            constrained.delta_flow,
            target.foreground_mask,
        )
        return ViewTransitionResult(target, boundary, constrained, tele_result, wide_result)


__all__ = ["ViewTransition", "ViewTransitionResult"]
