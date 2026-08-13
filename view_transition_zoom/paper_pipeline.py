"""End-to-end paper reproduction pipeline and debug artifact writer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from fusion.full_view_fusion import FusionResult, full_view_fusion
from fusion.occlusion import OcclusionResult, fb_consistency_occlusion, paper_occlusion
from fusion.tone_matching import ToneMatcher, ToneResult
from models.flow_estimator import FlowEstimator
from utils.image_io import validate_image_pair, write_image, write_mask
from utils.flow_utils import load_flow
from utils.visualization import flow_to_color
from utils.warp import forward_warp
from view_transition.view_transition import ViewTransition, ViewTransitionResult


@dataclass
class PaperPipelineResult:
    flow_t2w: torch.Tensor
    reverse_flow: Optional[torch.Tensor]
    view: ViewTransitionResult
    occlusion: OcclusionResult
    tone: ToneResult
    fusion: FusionResult


class PaperReproPipeline:
    """Compose cached optical flow, paper geometry, and occlusion-aware fusion."""

    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device,
        flow_estimator: Optional[FlowEstimator] = None,
    ) -> None:
        self.config = config
        self.device = device
        self.flow_estimator = flow_estimator
        self.view_transition = ViewTransition(config)
        tone_cfg = config.get("tone", {})
        self.tone_matcher = ToneMatcher(
            mode=str(tone_cfg.get("mode", "paper_rhe")),
            block_size=int(tone_cfg.get("block_size", 200)),
            stride=int(tone_cfg.get("stride", 30)),
            histogram_bins=int(tone_cfg.get("histogram_bins", 256)),
            local_window=int(tone_cfg.get("local_window", 61)),
            ema=float(tone_cfg.get("ema", 0.85)),
        )

    def _estimator(self) -> FlowEstimator:
        if self.flow_estimator is None:
            self.flow_estimator = FlowEstimator.from_config(self.config, self.device)
        return self.flow_estimator

    def run(
        self,
        wide: torch.Tensor,
        tele: torch.Tensor,
        flow_t2w: Optional[torch.Tensor] = None,
        reverse_flow: Optional[torch.Tensor] = None,
        overlap_mask: Optional[torch.Tensor] = None,
    ) -> PaperPipelineResult:
        wide, tele = wide.to(self.device), tele.to(self.device)
        validate_image_pair(wide, tele)
        if flow_t2w is None:
            flow_t2w = self._estimator()(wide, tele)
        flow_t2w = flow_t2w.to(device=self.device, dtype=torch.float32)
        if flow_t2w.shape != (wide.shape[0], 2, wide.shape[2], wide.shape[3]):
            raise ValueError("F_T2W must be [B,2,H,W] and match the input pair")
        if not torch.isfinite(flow_t2w).all():
            raise ValueError("Optical flow contains NaN or Inf")
        if reverse_flow is not None:
            reverse_flow = reverse_flow.to(device=self.device, dtype=torch.float32)
            if reverse_flow.shape != flow_t2w.shape or not torch.isfinite(reverse_flow).all():
                raise ValueError("Reverse flow must be finite and match F_T2W shape")
        view = self.view_transition(wide, tele, flow_t2w)
        foreground_o, _, foreground_valid = forward_warp(
            view.target.foreground_mask,
            view.constrained.delta_flow,
            mode=str(self.config.get("transform", {}).get("splat_mode", "bilinear")),
        )
        foreground_o = (foreground_o > 0.5).to(flow_t2w.dtype) * foreground_valid

        occ_cfg = self.config.get("occlusion", {})
        camera_position = occ_cfg.get("camera_relative_position", {})
        occ_mode = str(occ_cfg.get("mode", "paper")).lower()
        if occ_mode in ("fb_consistency", "fb"):
            if reverse_flow is None:
                reverse_path = self.config.get("flow", {}).get("reverse_precomputed_path")
                if reverse_path:
                    reverse_flow = load_flow(reverse_path, tele)
                elif str(self.config.get("flow", {}).get("model", "")).lower() == "precomputed":
                    raise ValueError(
                        "fb_consistency with precomputed flow requires flow.reverse_precomputed_path"
                    )
                else:
                    reverse_flow = self._estimator()(tele, wide)
            base_occlusion = fb_consistency_occlusion(
                flow_t2w,
                reverse_flow,
                absolute_threshold=float(occ_cfg.get("fb_absolute_threshold", 1.0)),
                relative_threshold=float(occ_cfg.get("fb_relative_threshold", 0.05)),
                soft_width=int(occ_cfg.get("soft_width", 15)),
            )
            moved_occ, _, moved_valid = forward_warp(
                base_occlusion.hard_mask,
                view.constrained.delta_flow,
                mode=str(self.config.get("transform", {}).get("splat_mode", "bilinear")),
            )
            hard_occ = torch.maximum(
                ((moved_occ > 0.25) & (moved_valid > 0.5)).to(flow_t2w.dtype),
                (view.tele.valid_mask < 0.5).to(flow_t2w.dtype),
            )
            hard_occ = hard_occ * (view.wide.valid_mask > 0.5).to(flow_t2w.dtype)
            from fusion.occlusion import soften_occlusion

            occlusion = OcclusionResult(
                hard_occ,
                soften_occlusion(hard_occ, int(occ_cfg.get("soft_width", 15))),
                base_occlusion.consistency_error,
            )
        elif occ_mode in ("paper", "paper_occlusion"):
            occlusion = paper_occlusion(
                view.tele.flow_t2o,
                foreground_mask=foreground_o,
                camera_x=str(camera_position.get("x", occ_cfg.get("camera_x", "left"))),
                camera_y=str(camera_position.get("y", occ_cfg.get("camera_y", "upper"))),
                foreground_mode=str(occ_cfg.get("foreground_mode", "magnitude")),
                rectangle_scale=float(occ_cfg.get("rectangle_scale", 1.0)),
                max_rectangle=int(occ_cfg.get("max_rectangle", 256)),
                max_boundary_points=int(occ_cfg.get("max_boundary_points", 20000)),
                tele_valid=view.tele.valid_mask,
                wide_valid=view.wide.valid_mask,
                soft_width=int(occ_cfg.get("soft_width", 15)),
            )
        else:
            raise ValueError("occlusion.mode must be paper or fb_consistency")

        common_valid = view.tele.valid_mask * view.wide.valid_mask
        tone = self.tone_matcher(view.tele.image, view.wide.image, common_valid)
        if overlap_mask is None:
            overlap_mask = torch.ones_like(wide[:, :1])
        if overlap_mask.shape != wide[:, :1].shape:
            raise ValueError("overlap mask must have shape [B,1,H,W]")
        geometric_valid = torch.maximum(view.wide.valid_mask, view.tele.valid_mask)
        effective_overlap = overlap_mask.to(self.device) * geometric_valid
        blend_cfg = self.config.get("blend", {})
        fusion = full_view_fusion(
            view.wide.image,
            tone.image,
            wide,
            occlusion.soft_mask,
            effective_overlap,
            pyramid_levels=int(blend_cfg.get("pyramid_levels", 5)),
            overlap_soft_width=int(blend_cfg.get("overlap_soft_width", 100)),
        )
        return PaperPipelineResult(flow_t2w, reverse_flow, view, occlusion, tone, fusion)


def save_paper_debug(
    root: str,
    wide: torch.Tensor,
    tele: torch.Tensor,
    result: PaperPipelineResult,
) -> Path:
    output = Path(root)
    write_image(str(output / "input" / "wide.png"), wide)
    write_image(str(output / "input" / "tele.png"), tele)
    flows = {
        "original": result.flow_t2w,
        "mean": result.view.target.mean_flow,
        "foreground": result.view.target.foreground_flow,
        "background": result.view.target.background_flow,
        "target": result.view.target.target_flow,
        "transformed": result.view.constrained.transformed_flow,
        "delta": result.view.constrained.delta_flow,
    }
    for name, flow in flows.items():
        write_image(str(output / "flow" / (name + ".png")), flow_to_color(flow))
        path = output / "flow" / (name + ".npy")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(path), flow[0].detach().cpu().numpy())
    write_mask(str(output / "masks" / "foreground.png"), result.view.target.foreground_mask)
    write_mask(str(output / "masks" / "distance.png"), result.view.boundary.distance, normalize=True)
    write_mask(str(output / "masks" / "motion_boundary.png"), result.view.boundary.motion_boundary)
    write_mask(str(output / "masks" / "hole.png"), result.view.tele.hole_mask)
    write_mask(str(output / "masks" / "occlusion.png"), result.occlusion.soft_mask)
    write_mask(str(output / "masks" / "overlap.png"), result.fusion.overlap_mask)
    write_image(str(output / "transformed" / "tele_O.png"), result.view.tele.image)
    write_image(str(output / "transformed" / "wide_O.png"), result.view.wide.image)
    write_image(str(output / "transformed" / "tele_tone.png"), result.tone.image)
    write_image(str(output / "results" / "overlap_result.png"), result.fusion.overlap_result)
    write_image(str(output / "results" / "full_result.png"), result.fusion.full_result)
    return output


__all__ = ["PaperPipelineResult", "PaperReproPipeline", "save_paper_debug"]
