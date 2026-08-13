"""Interpolate the paper deformation field for any zoom ratio."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from view_transition.flow_transform import interpolate_transformation
from view_transition.transform_tele import TeleTransformResult
from view_transition.transform_wide import WideTransformResult
from view_transition.view_transition import ViewTransition


@dataclass
class ContinuousViewResult:
    alpha: float
    transformed_flow: torch.Tensor
    delta_flow: torch.Tensor
    tele: TeleTransformResult
    wide: WideTransformResult


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


__all__ = ["ContinuousViewResult", "continuous_view_transition"]

