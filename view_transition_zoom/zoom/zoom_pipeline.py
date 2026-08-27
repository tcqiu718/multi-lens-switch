"""Continuous dual-camera zoom pipeline built on paper View Transition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from fusion.full_view_fusion import FusionResult, full_view_fusion
from fusion.occlusion import OcclusionResult, fb_consistency_occlusion, paper_occlusion, soften_occlusion
from fusion.tone_matching import ToneMatcher, ToneResult, interpolate_affine_tone_pair
from models.flow_estimator import FlowEstimator
from utils.image_io import validate_image_pair
from utils.flow_utils import load_flow
from utils.warp import forward_warp
from view_transition.empty_fill import fill_empty_regions
from view_transition.flow_transform import interpolate_transformation
from view_transition.view_transition import ViewTransition
from zoom.continuous_view import ContinuousViewResult, bidirectional_zoom_transition
from zoom.fov_transform import FovResult, center_crop_fov, intrinsics_fov
from zoom.temporal_stabilizer import TemporalStabilizer
from zoom.zoom_schedule import ZoomPoint, ZoomSchedule


@dataclass
class PreparedPair:
    wide: torch.Tensor
    tele: torch.Tensor
    flow_t2w: torch.Tensor
    reverse_flow: Optional[torch.Tensor]
    target: Any
    boundary: Any
    constrained: Any
    overlap_mask: torch.Tensor
    tone_calibration: ToneResult


@dataclass
class ZoomFrameResult:
    zoom_point: ZoomPoint
    view: ContinuousViewResult
    occlusion: OcclusionResult
    tone: ToneResult
    fusion: FusionResult
    fov: FovResult
    tele_fov: FovResult
    result: torch.Tensor
    tele_usage_ratio: float


class ContinuousZoomPipeline:
    """Cache pair flow/target once and render any number of zoom ratios."""

    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device,
        flow_estimator: Optional[FlowEstimator] = None,
    ) -> None:
        self.config = config
        self.device = device
        self.flow_estimator = flow_estimator
        self.cache_flow = bool(config.get("flow", {}).get("cache", True))
        self._prepared_cache_key = None
        self._prepared_cache_value = None
        self.view_engine = ViewTransition(config)
        zoom_cfg = config.get("zoom", {})
        camera_cfg = config.get("camera", {})
        geometry_mode = str(zoom_cfg.get("geometry_mode", "bidirectional_flow")).lower()
        geometry_aliases = {
            "bidirectional": "bidirectional_flow",
            "two_sided": "bidirectional_flow",
            "paper": "paper_delta",
        }
        self.geometry_mode = geometry_aliases.get(geometry_mode, geometry_mode)
        if self.geometry_mode not in ("bidirectional_flow", "paper_delta"):
            raise ValueError("zoom.geometry_mode must be bidirectional_flow or paper_delta")
        self.schedule = ZoomSchedule(
            wide_zoom=float(camera_cfg.get("wide_zoom", zoom_cfg.get("wide_ratio", 1.0))),
            tele_zoom=float(camera_cfg.get("tele_zoom", zoom_cfg.get("tele_ratio", 3.0))),
            interpolation=str(zoom_cfg.get("interpolation", "log")),
            curve=str(zoom_cfg.get("schedule", zoom_cfg.get("curve", "smootherstep"))),
            custom_values=zoom_cfg.get("custom_schedule"),
        )
        temporal_cfg = config.get("temporal", {})
        self.temporal = TemporalStabilizer(
            ema=float(temporal_cfg.get("ema", 0.82)),
            enabled=bool(temporal_cfg.get("enabled", True)),
        )
        self.delta_smoothing = bool(temporal_cfg.get("delta_smoothing", True))
        self.mask_smoothing = bool(temporal_cfg.get("mask_smoothing", True))
        tone_cfg = config.get("tone", {})
        tone_mode = str(tone_cfg.get("mode", "local_affine_temporal"))
        if not bool(temporal_cfg.get("tone_smoothing", True)) and tone_mode == "local_affine_temporal":
            tone_mode = "local_affine"
        self.tone_matcher = ToneMatcher(
            mode=tone_mode,
            block_size=int(tone_cfg.get("block_size", 200)),
            stride=int(tone_cfg.get("stride", 30)),
            histogram_bins=int(tone_cfg.get("histogram_bins", 256)),
            local_window=int(tone_cfg.get("local_window", 61)),
            ema=float(tone_cfg.get("ema", 0.85)),
        )

    def reset_temporal(self) -> None:
        self.temporal.reset()
        self.tone_matcher.reset()

    def _estimator(self) -> FlowEstimator:
        if self.flow_estimator is None:
            self.flow_estimator = FlowEstimator.from_config(self.config, self.device)
        return self.flow_estimator

    def estimate_temporal_flow(
        self,
        current_wide: torch.Tensor,
        previous_wide: torch.Tensor,
    ) -> torch.Tensor:
        """Return current->previous backward flow for propagating prior state."""
        validate_image_pair(current_wide, previous_wide)
        return self._estimator()(current_wide, previous_wide)

    def prepare_pair(
        self,
        wide: torch.Tensor,
        tele: torch.Tensor,
        flow_t2w: Optional[torch.Tensor] = None,
        reverse_flow: Optional[torch.Tensor] = None,
        overlap_mask: Optional[torch.Tensor] = None,
    ) -> PreparedPair:
        cacheable = self.cache_flow and flow_t2w is None and reverse_flow is None
        cache_key = (id(wide), id(tele), id(overlap_mask)) if cacheable else None
        if cacheable and cache_key == self._prepared_cache_key:
            return self._prepared_cache_value
        wide, tele = wide.to(self.device), tele.to(self.device)
        validate_image_pair(wide, tele)
        estimator = None
        if flow_t2w is None:
            estimator = self._estimator()
            flow_t2w = estimator(wide, tele)
        flow_t2w = flow_t2w.to(device=self.device, dtype=torch.float32)
        if not torch.isfinite(flow_t2w).all():
            raise ValueError("Optical flow contains NaN or Inf")
        occ_mode = str(self.config.get("occlusion", {}).get("mode", "paper")).lower()
        needs_reverse = self.geometry_mode == "bidirectional_flow" or occ_mode in ("fb", "fb_consistency")
        if needs_reverse and reverse_flow is None:
            reverse_path = self.config.get("flow", {}).get("reverse_precomputed_path")
            if reverse_path:
                reverse_flow = load_flow(reverse_path, tele)
            elif str(self.config.get("flow", {}).get("model", "")).lower() == "precomputed":
                raise ValueError(
                    "bidirectional geometry or fb_consistency with precomputed flow requires "
                    "flow.reverse_precomputed_path"
                )
            else:
                estimator = estimator or self._estimator()
                reverse_flow = estimator(tele, wide)
        if reverse_flow is not None:
            reverse_flow = reverse_flow.to(device=self.device, dtype=torch.float32)
            if reverse_flow.shape != flow_t2w.shape or not torch.isfinite(reverse_flow).all():
                raise ValueError("Reverse flow must be finite and match F_T2W shape")
        target, boundary, constrained = self.view_engine.prepare_flow(flow_t2w)
        overlap = torch.ones_like(wide[:, :1]) if overlap_mask is None else overlap_mask.to(self.device)
        if overlap.shape != wide[:, :1].shape or not torch.any(overlap > 0.5):
            raise ValueError("A non-empty overlap mask shaped [B,1,H,W] is required")
        zero_delta = torch.zeros_like(flow_t2w)
        calibration_tele, calibration_wide = self.view_engine.transform(
            wide,
            tele,
            flow_t2w,
            zero_delta,
            target.foreground_mask,
        )
        calibration_valid = calibration_tele.valid_mask * calibration_wide.valid_mask
        tone_calibration = self.tone_matcher(
            calibration_tele.image,
            calibration_wide.image,
            calibration_valid,
        )
        prepared = PreparedPair(
            wide,
            tele,
            flow_t2w,
            reverse_flow,
            target,
            boundary,
            constrained,
            overlap,
            tone_calibration,
        )
        if cacheable:
            self._prepared_cache_key = cache_key
            self._prepared_cache_value = prepared
        return prepared

    def _occlusion(self, pair: PreparedPair, view: ContinuousViewResult) -> OcclusionResult:
        cfg = self.config.get("occlusion", {})
        camera_position = cfg.get("camera_relative_position", {})
        mode = str(cfg.get("mode", "paper")).lower()
        if mode in ("fb", "fb_consistency"):
            if pair.reverse_flow is None:
                raise ValueError("fb_consistency requires reverse flow")
            base = fb_consistency_occlusion(
                pair.flow_t2w,
                pair.reverse_flow,
                float(cfg.get("fb_absolute_threshold", 1.0)),
                float(cfg.get("fb_relative_threshold", 0.05)),
                soft_width=int(cfg.get("soft_width", 15)),
            )
            # Move the W-coordinate consistency mask to O with the same delta.
            from utils.warp import forward_warp

            moved, _, valid = forward_warp(base.hard_mask, view.delta_flow, mode="bilinear")
            hard = torch.maximum((moved > 0.25).to(moved.dtype), 1.0 - view.tele.valid_mask)
            hard = hard * view.wide.valid_mask
            return OcclusionResult(hard, soften_occlusion(hard, int(cfg.get("soft_width", 15))), base.consistency_error)
        if mode not in ("paper", "paper_occlusion"):
            raise ValueError("occlusion.mode must be paper or fb_consistency")
        from utils.warp import forward_warp

        foreground_o, _, foreground_valid = forward_warp(
            pair.target.foreground_mask,
            view.delta_flow,
            mode=str(self.config.get("transform", {}).get("splat_mode", "bilinear")),
        )
        foreground_o = (foreground_o > 0.5).to(foreground_o.dtype) * foreground_valid
        return paper_occlusion(
            view.tele.flow_t2o,
            foreground_mask=foreground_o,
            camera_x=str(camera_position.get("x", cfg.get("camera_x", "left"))),
            camera_y=str(camera_position.get("y", cfg.get("camera_y", "upper"))),
            foreground_mode=str(cfg.get("foreground_mode", "magnitude")),
            rectangle_scale=float(cfg.get("rectangle_scale", 1.0)),
            max_rectangle=int(cfg.get("max_rectangle", 256)),
            max_boundary_points=int(cfg.get("max_boundary_points", 20000)),
            tele_valid=view.tele.valid_mask,
            wide_valid=view.wide.valid_mask,
            soft_width=int(cfg.get("soft_width", 15)),
        )

    def _apply_fov(self, image: torch.Tensor, zoom: float, source_zoom: float) -> FovResult:
        cfg = self.config.get("zoom", {})
        mode = str(cfg.get("fov_mode", "center_crop")).lower()
        if mode == "center_crop":
            return center_crop_fov(image, zoom, source_zoom)
        if mode != "intrinsics":
            raise ValueError("zoom.fov_mode must be center_crop or intrinsics")
        source = cfg.get("wide_intrinsics" if source_zoom == self.schedule.wide_zoom else "tele_intrinsics")
        wide_k, tele_k = cfg.get("wide_intrinsics"), cfg.get("tele_intrinsics")
        if source is None or wide_k is None or tele_k is None:
            raise ValueError("intrinsics FOV mode requires wide_intrinsics and tele_intrinsics")
        from utils.camera_geometry import interpolate_intrinsics, intrinsics_matrix

        point = self.schedule(zoom)
        wide_tensor = intrinsics_matrix(wide_k, image.device, image.dtype)
        tele_tensor = intrinsics_matrix(tele_k, image.device, image.dtype)
        target = interpolate_intrinsics(wide_tensor, tele_tensor, point.progress)
        return intrinsics_fov(image, source, target.detach().cpu().tolist())

    def _warp_calibration(
        self,
        gain: torch.Tensor,
        bias: torch.Tensor,
        motion: torch.Tensor,
    ) -> tuple:
        mode = str(self.config.get("transform", {}).get("splat_mode", "bilinear"))
        channels = gain.shape[1]
        moved, _, valid = forward_warp(torch.cat((gain, bias), dim=1), motion, mode=mode)
        if torch.any(valid < 0.5):
            moved = fill_empty_regions(moved, valid, mode="nearest")
        return moved[:, :channels], moved[:, channels:]

    def _bidirectional_view(
        self,
        pair: PreparedPair,
        point: ZoomPoint,
        temporal_flow: Optional[torch.Tensor],
    ) -> ContinuousViewResult:
        if pair.reverse_flow is None:
            raise ValueError("bidirectional_flow geometry requires reverse flow")
        forward, reverse = pair.flow_t2w, pair.reverse_flow
        if self.delta_smoothing:
            forward, reverse = self.temporal.smooth_correspondence_flows(
                forward,
                reverse,
                temporal_flow,
            )
        zoom_cfg = self.config.get("zoom", {})
        transform_cfg = self.config.get("transform", {})
        return bidirectional_zoom_transition(
            pair.wide,
            pair.tele,
            forward,
            reverse,
            pair.overlap_mask,
            point.alpha,
            point.zoom,
            self.schedule.wide_zoom,
            self.schedule.tele_zoom,
            point.progress,
            fov_mode=str(zoom_cfg.get("fov_mode", "center_crop")),
            wide_intrinsics=zoom_cfg.get("wide_intrinsics"),
            tele_intrinsics=zoom_cfg.get("tele_intrinsics"),
            splat_mode=str(transform_cfg.get("splat_mode", "bilinear")),
            multi_warp_average=bool(transform_cfg.get("multi_warp_average", True)),
            tele_multi_warp_average=bool(transform_cfg.get("tele_multi_warp_average", False)),
            offset_min=float(transform_cfg.get("wide_offset_min", -0.5)),
            offset_max=float(transform_cfg.get("wide_offset_max", 0.5)),
            offset_step=float(transform_cfg.get("wide_offset_step", 0.2)),
        )

    def render(
        self,
        pair: PreparedPair,
        zoom: float,
        temporal_flow: Optional[torch.Tensor] = None,
    ) -> ZoomFrameResult:
        point = self.schedule(zoom)
        if self.geometry_mode == "bidirectional_flow":
            view = self._bidirectional_view(pair, point, temporal_flow)
            current_overlap = view.overlap_mask
        else:
            # Legacy paper path ends at the constrained mixed view rather than
            # the native Tele geometry and remains available for ablations.
            smooth_paper_delta = (
                self.temporal.smooth_delta(pair.constrained.delta_flow, temporal_flow)
                if self.delta_smoothing
                else pair.constrained.delta_flow
            )
            delta_zoom = smooth_paper_delta * point.alpha
            interpolation = interpolate_transformation(pair.flow_t2w, delta_zoom, 1.0)
            tele_result, wide_result = self.view_engine.transform(
                pair.wide,
                pair.tele,
                interpolation.transformed_flow,
                interpolation.delta_flow,
                pair.target.foreground_mask,
            )
            view = ContinuousViewResult(
                point.alpha,
                interpolation.transformed_flow,
                interpolation.delta_flow,
                tele_result,
                wide_result,
            )
            current_overlap = pair.overlap_mask
        occlusion = self._occlusion(pair, view)
        if self.mask_smoothing:
            smooth_occ, smooth_overlap = self.temporal.smooth_masks(
                occlusion.soft_mask, current_overlap, temporal_flow
            )
        else:
            smooth_occ, smooth_overlap = occlusion.soft_mask, current_overlap
        blend_mask = smooth_occ
        if self.geometry_mode == "bidirectional_flow":
            blend_cfg = self.config.get("blend", {})
            occlusion_strength = float(blend_cfg.get("viewpoint_occlusion_strength", 2.0))
            endpoint_weight = point.alpha * (1.0 - point.alpha)
            blend_mask = (
                (1.0 - point.alpha) + occlusion_strength * endpoint_weight * smooth_occ
            ).clamp(0.0, 1.0)
            wide_valid = view.wide.valid_mask > 0.5
            tele_valid = view.tele.valid_mask > 0.5
            # Override smoothed weights only where a single camera is valid.
            blend_mask = torch.where(
                wide_valid & ~tele_valid, torch.ones_like(blend_mask), blend_mask
            )
            blend_mask = torch.where(
                ~wide_valid & tele_valid, torch.zeros_like(blend_mask), blend_mask
            )
        occlusion = OcclusionResult(occlusion.hard_mask, smooth_occ, occlusion.consistency_error)
        calibration = pair.tone_calibration
        if calibration.gain is not None and calibration.bias is not None:
            calibration_gain = calibration.gain
            calibration_bias = calibration.bias
            reference_full = pair.wide
            if self.geometry_mode == "bidirectional_flow":
                calibration_gain, calibration_bias = self._warp_calibration(
                    calibration_gain,
                    calibration_bias,
                    view.wide_motion,
                )
                reference_full = view.wide.image
            tele_tone, wide_tone, wide_full_tone, gain, bias = interpolate_affine_tone_pair(
                view.tele.image,
                view.wide.image,
                reference_full,
                calibration_gain,
                calibration_bias,
                point.tone_progress,
            )
            tone = ToneResult(tele_tone, gain, bias)
        else:
            current_tone = self.tone_matcher(
                view.tele.image,
                view.wide.image,
                view.tele.valid_mask * view.wide.valid_mask,
            )
            tele_tone = torch.lerp(current_tone.image, view.tele.image, point.tone_progress)
            wide_tone = view.wide.image
            wide_full_tone = view.wide.image if self.geometry_mode == "bidirectional_flow" else pair.wide
            tone = ToneResult(tele_tone)
        blend_cfg = self.config.get("blend", {})
        levels = int(blend_cfg.get("pyramid_levels", 5))
        geometric_valid = torch.maximum(view.wide.valid_mask, view.tele.valid_mask)
        effective_overlap = smooth_overlap * geometric_valid
        fusion = full_view_fusion(
            wide_tone,
            tone.image,
            wide_full_tone,
            blend_mask,
            effective_overlap,
            levels,
            int(blend_cfg.get("overlap_soft_width", 100)),
        )
        if self.geometry_mode == "bidirectional_flow":
            fov_metadata = self._apply_fov(pair.wide, zoom, self.schedule.wide_zoom)
            fov = FovResult(
                fusion.full_result,
                geometric_valid,
                fov_metadata.crop_scale,
                fov_metadata.crop_box,
            )
        else:
            fov = self._apply_fov(fusion.full_result, zoom, self.schedule.wide_zoom)
        tele_fov = self._apply_fov(pair.tele, zoom, self.schedule.tele_zoom)

        result = fov.image
        if point.is_wide_endpoint:
            result = pair.wide.clone()
        elif point.is_tele_endpoint:
            result = pair.tele.clone()
        mixed_tele_selection = (1.0 - blend_mask) * effective_overlap * view.tele.valid_mask
        if self.geometry_mode == "bidirectional_flow":
            mixed_selection_image = mixed_tele_selection
            overlap_image = effective_overlap
        else:
            mixed_selection_image = self._apply_fov(
                mixed_tele_selection, zoom, self.schedule.wide_zoom
            ).image
            overlap_image = self._apply_fov(effective_overlap, zoom, self.schedule.wide_zoom).image
        denominator = overlap_image.sum().clamp_min(1.0)
        tele_usage = float((mixed_selection_image.sum() / denominator).detach().cpu())
        if point.is_wide_endpoint:
            tele_usage = 0.0
        elif point.is_tele_endpoint:
            tele_usage = 1.0
        return ZoomFrameResult(point, view, occlusion, tone, fusion, fov, tele_fov, result, tele_usage)


__all__ = ["ContinuousZoomPipeline", "PreparedPair", "ZoomFrameResult"]
