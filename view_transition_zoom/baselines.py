"""Five requested baselines sharing one cached W/T optical flow."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from fusion.full_view_fusion import full_view_fusion
from fusion.occlusion import OcclusionResult
from fusion.tone_matching import ToneMatcher
from utils.warp import backward_warp
from view_transition.flow_transform import interpolate_transformation
from zoom.continuous_view import ContinuousViewResult
from zoom.zoom_pipeline import ContinuousZoomPipeline, PreparedPair, ZoomFrameResult


@dataclass
class BaselineResult:
    wide_digital_zoom: torch.Tensor
    direct_warp: torch.Tensor
    direct_pyramid: torch.Tensor
    paper_view_transition: torch.Tensor
    continuous_view_transition: torch.Tensor


def render_baselines(
    pipeline: ContinuousZoomPipeline,
    pair: PreparedPair,
    continuous: ZoomFrameResult,
) -> BaselineResult:
    zoom = continuous.zoom_point.zoom
    levels = int(pipeline.config.get("blend", {}).get("pyramid_levels", 5))
    overlap_width = int(pipeline.config.get("blend", {}).get("overlap_soft_width", 100))
    wide_zoom = pipeline._apply_fov(pair.wide, zoom, pipeline.schedule.wide_zoom).image

    direct_tele, direct_valid = backward_warp(pair.tele, pair.flow_t2w, return_valid=True)
    hard_overlap = pair.overlap_mask * direct_valid
    direct_full = direct_tele * hard_overlap + pair.wide * (1.0 - hard_overlap)
    direct_zoom = pipeline._apply_fov(direct_full, zoom, pipeline.schedule.wide_zoom).image
    direct_multi = full_view_fusion(
        pair.wide,
        direct_tele,
        pair.wide,
        1.0 - direct_valid,
        pair.overlap_mask,
        levels,
        overlap_width,
    ).full_result
    direct_pyramid = pipeline._apply_fov(direct_multi, zoom, pipeline.schedule.wide_zoom).image

    paper_interpolation = interpolate_transformation(
        pair.flow_t2w, pair.constrained.delta_flow, 1.0
    )
    tele_result, wide_result = pipeline.view_engine.transform(
        pair.wide,
        pair.tele,
        paper_interpolation.transformed_flow,
        paper_interpolation.delta_flow,
        pair.target.foreground_mask,
    )
    paper_view = ContinuousViewResult(
        1.0,
        paper_interpolation.transformed_flow,
        paper_interpolation.delta_flow,
        tele_result,
        wide_result,
    )
    paper_occ = pipeline._occlusion(pair, paper_view)
    tone_cfg = pipeline.config.get("tone", {})
    paper_tone = ToneMatcher(
        mode=str(tone_cfg.get("mode", "local_affine")),
        block_size=int(tone_cfg.get("block_size", 200)),
        stride=int(tone_cfg.get("stride", 30)),
        histogram_bins=int(tone_cfg.get("histogram_bins", 256)),
        local_window=int(tone_cfg.get("local_window", 61)),
        ema=float(tone_cfg.get("ema", 0.85)),
    )(tele_result.image, wide_result.image, tele_result.valid_mask * wide_result.valid_mask)
    paper_full = full_view_fusion(
        wide_result.image,
        paper_tone.image,
        pair.wide,
        paper_occ.soft_mask,
        pair.overlap_mask,
        levels,
        overlap_width,
    ).full_result
    paper_zoom = pipeline._apply_fov(paper_full, zoom, pipeline.schedule.wide_zoom).image
    return BaselineResult(wide_zoom, direct_zoom, direct_pyramid, paper_zoom, continuous.result)


__all__ = ["BaselineResult", "render_baselines"]
