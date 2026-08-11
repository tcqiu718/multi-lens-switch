"""End-to-end hybrid zoom model assembled from reusable research modules."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from hybrid_zoom.config_utils import resolve_wide_crop_size
from hybrid_zoom.modules.adaptive_blending import AdaptiveBlending
from hybrid_zoom.modules.occlusion_mask import OcclusionMask
from hybrid_zoom.modules.preprocessing import extract_luminance, preprocess_pair, replace_luminance
from hybrid_zoom.modules.rejection_mask import RejectionMask
from hybrid_zoom.modules.warp import warp

from .flow_estimator import FlowEstimator
from .fusion_unet import FusionUNet


def _mapping_section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("config section {!r} must be a mapping".format(name))
    return value


def _positive_pair(values: Sequence[Any], name: str) -> Tuple[int, int]:
    if len(values) != 2:
        raise ValueError("{} must contain (height, width)".format(name))
    pair = (int(values[0]), int(values[1]))
    if pair[0] <= 0 or pair[1] <= 0:
        raise ValueError("{} values must be positive, got {}".format(name, pair))
    return pair


class HybridZoomModel(nn.Module):
    """Align Tele to Wide, fuse luminance, and reject unreliable regions.

    The model accepts synchronized RGB tensors shaped ``[B, 3, H, W]`` in
    ``[0, 1]``.  Inputs may have different spatial sizes: preprocessing makes
    them match the configured ``image.height`` and ``image.width`` (or the Wide
    input size if these fields are omitted).

    Flow convention:
        ``flow_w2t = flow_estimator(wide, tele)`` satisfies
        ``p_tele = p_wide + flow_w2t(p_wide)``.  Thus
        ``warp(tele, flow_w2t)`` samples Tele into Wide coordinates.  Reverse
        flow is estimated separately for forward-backward consistency.

    Mask convention is uniform: zero means reliable and one means unreliable.
    Optional flow-uncertainty and defocus masks are accepted now even though
    this first version does not estimate them internally.

    Args:
        config: Full nested project configuration.
        flow_estimator: Optional compatible module for experiments/tests.  Its
            ``forward(first, second)`` must return ``[B, 2, H, W]`` first-to-
            second flow.  If omitted, :class:`FlowEstimator` is built from config.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        flow_estimator: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if not isinstance(config, Mapping):
            raise TypeError("config must be a nested mapping")
        self.config = dict(config)

        image_config = _mapping_section(config, "image")
        image_height = image_config.get("height")
        image_width = image_config.get("width")
        if (image_height is None) != (image_width is None):
            raise ValueError("image.height and image.width must be specified together")
        self.output_size: Optional[Tuple[int, int]]
        if image_height is None:
            self.output_size = None
        else:
            self.output_size = _positive_pair((image_height, image_width), "image size")
        self.keep_aspect_ratio = bool(image_config.get("keep_aspect_ratio", False))
        self.aspect_mode = str(image_config.get("aspect_mode", "letterbox"))
        self.interpolation = str(image_config.get("interpolation", "bilinear"))
        self.wide_crop_size = self._parse_crop_config(image_config)

        if flow_estimator is not None and not isinstance(flow_estimator, nn.Module):
            raise TypeError("flow_estimator must be an nn.Module or None")
        self.flow_estimator = (
            flow_estimator if flow_estimator is not None else FlowEstimator.from_config(config)
        )

        fusion_config = _mapping_section(config, "fusion")
        self.use_rejection_input = bool(fusion_config.get("use_rejection_input", False))
        fusion_channels = 4 if self.use_rejection_input else 3
        self.fusion = FusionUNet(
            in_channels=fusion_channels,
            base_channels=int(fusion_config.get("base_channels", 32)),
            mode=str(fusion_config.get("mode", "residual")),
            activation=str(fusion_config.get("activation", "leaky_relu")),
        )

        occlusion_config = _mapping_section(config, "occlusion")
        self.occlusion = OcclusionMask(
            mode=str(occlusion_config.get("mode", "soft")),
            threshold=float(occlusion_config.get("threshold", 1.0)),
            temperature=float(occlusion_config.get("temperature", 0.2)),
        )

        rejection_config = _mapping_section(config, "rejection")
        self.rejection = RejectionMask(
            patch_size=int(rejection_config.get("patch_size", 16)),
            stride=int(rejection_config.get("stride", 8)),
            metric=str(rejection_config.get("metric", "normalized_l1")),
            threshold=float(rejection_config.get("threshold", 0.25)),
            temperature=float(rejection_config.get("temperature", 0.05)),
            eps=float(rejection_config.get("eps", 1e-6)),
        )

        blending_config = _mapping_section(config, "blending")
        raw_kernel_size = blending_config.get("kernel_size", None)
        kernel_size = None if raw_kernel_size in {None, 0} else int(raw_kernel_size)
        self.blending = AdaptiveBlending(
            smoothing=str(blending_config.get("smoothing", "gaussian")),
            sigma=float(blending_config.get("sigma", 2.0)),
            kernel_size=kernel_size,
        )

    @staticmethod
    def _parse_crop_config(image_config: Mapping[str, Any]) -> Optional[Tuple[int, int]]:
        return resolve_wide_crop_size(image_config)

    @staticmethod
    def _validate_rgb_pair(wide: Tensor, tele: Tensor) -> None:
        if not isinstance(wide, Tensor) or not isinstance(tele, Tensor):
            raise TypeError("wide and tele must be torch.Tensor instances")
        if wide.ndim != 4 or tele.ndim != 4:
            raise ValueError("wide and tele must have shape [B, 3, H, W]")
        if wide.shape[1] != 3 or tele.shape[1] != 3:
            raise ValueError("wide and tele must have exactly three RGB channels")
        if wide.shape[0] != tele.shape[0]:
            raise ValueError("wide and tele batch sizes must match")
        if wide.device != tele.device:
            raise ValueError("wide and tele must be on the same device")
        if wide.dtype != tele.dtype:
            raise ValueError("wide and tele must use the same dtype")
        valid_dtypes = {torch.uint8, torch.float16, torch.float32, torch.float64, torch.bfloat16}
        if wide.dtype not in valid_dtypes or tele.dtype not in valid_dtypes:
            raise TypeError("wide and tele must be uint8 or floating-point RGB tensors")
        if wide.shape[-2] <= 0 or wide.shape[-1] <= 0 or tele.shape[-2] <= 0 or tele.shape[-1] <= 0:
            raise ValueError("wide and tele must have non-empty spatial dimensions")

    @staticmethod
    def _validate_flow(flow: Tensor, reference: Tensor, name: str) -> None:
        expected = (reference.shape[0], 2, reference.shape[-2], reference.shape[-1])
        if not isinstance(flow, Tensor) or tuple(flow.shape) != expected:
            shape = getattr(flow, "shape", None)
            raise RuntimeError("{} must have shape {}, got {}".format(name, expected, shape))
        if not flow.is_floating_point():
            raise RuntimeError("{} must be floating point".format(name))
        if flow.device != reference.device:
            raise RuntimeError("{} and preprocessed images must be on the same device".format(name))

    @staticmethod
    def _prepare_optional_mask(mask: Optional[Tensor], reference: Tensor, name: str) -> Optional[Tensor]:
        if mask is None:
            return None
        if not isinstance(mask, Tensor):
            raise TypeError("{} must be a tensor or None".format(name))
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError("{} must have shape [B, 1, H, W] or [B, H, W]".format(name))
        if mask.shape[0] != reference.shape[0]:
            raise ValueError("{} batch size must match the images".format(name))
        if mask.device != reference.device:
            raise ValueError("{} must be on the same device as the images".format(name))
        prepared = mask.to(dtype=reference.dtype)
        if prepared.shape[-2:] != reference.shape[-2:]:
            prepared = F.interpolate(prepared, size=reference.shape[-2:], mode="bilinear", align_corners=False)
        return prepared.clamp(0.0, 1.0)

    @staticmethod
    def _resolve_mask_aliases(
        flow_uncertainty_mask: Optional[Tensor],
        defocus_mask: Optional[Tensor],
        optional_masks: Mapping[str, Any],
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        remaining: Dict[str, Any] = dict(optional_masks)
        nested = remaining.pop("masks", None)
        if nested is not None:
            if not isinstance(nested, Mapping):
                raise TypeError("masks must be a mapping")
            for key, value in nested.items():
                if key in remaining:
                    raise ValueError("mask {!r} was specified more than once".format(key))
                remaining[key] = value

        flow_alias = remaining.pop("flow_uncertainty", None)
        defocus_alias = remaining.pop("defocus", None)
        if flow_alias is not None:
            if flow_uncertainty_mask is not None:
                raise ValueError("flow uncertainty mask was specified more than once")
            flow_uncertainty_mask = flow_alias
        if defocus_alias is not None:
            if defocus_mask is not None:
                raise ValueError("defocus mask was specified more than once")
            defocus_mask = defocus_alias
        if remaining:
            raise TypeError("Unexpected optional mask arguments: {}".format(sorted(remaining.keys())))
        return flow_uncertainty_mask, defocus_mask

    def _preprocess(self, wide: Tensor, tele: Tensor) -> Tuple[Tensor, Tensor]:
        target_size = self.output_size or tuple(wide.shape[-2:])
        return preprocess_pair(
            wide,
            tele,
            output_size=target_size,
            wide_crop_size=self.wide_crop_size,
            keep_aspect_ratio=self.keep_aspect_ratio,
            aspect_mode=self.aspect_mode,
            interpolation=self.interpolation,
        )

    def forward(
        self,
        wide: Tensor,
        tele: Tensor,
        flow_uncertainty_mask: Optional[Tensor] = None,
        defocus_mask: Optional[Tensor] = None,
        preprocessed: bool = False,
        **optional_masks: Any
    ) -> Dict[str, Tensor]:
        """Run the complete differentiable fusion pipeline.

        The optional masks may be passed as ``flow_uncertainty_mask=...`` and
        ``defocus_mask=...``.  For convenient future integration, aliases
        ``flow_uncertainty`` / ``defocus`` and ``masks={...}`` are also accepted.
        ``preprocessed=True`` skips crop/resize and requires equal floating-point
        RGB shapes; Dataset-backed CLI programs use it to avoid applying geometry
        twice after their batch-safe preprocessing.

        Returns all required intermediate tensors for research visualization and
        debugging, not just the reconstructed RGB output.
        """
        self._validate_rgb_pair(wide, tele)
        flow_uncertainty_mask, defocus_mask = self._resolve_mask_aliases(
            flow_uncertainty_mask, defocus_mask, optional_masks
        )
        if preprocessed:
            if not wide.is_floating_point() or not tele.is_floating_point():
                raise TypeError("preprocessed Wide/Tele tensors must be floating point")
            if wide.shape != tele.shape:
                raise ValueError("preprocessed Wide and Tele tensors must have identical shapes")
            wide_processed, tele_processed = wide, tele
        else:
            wide_processed, tele_processed = self._preprocess(wide, tele)

        # flow_w2t is defined on Wide pixels and points to corresponding Tele
        # coordinates; flow_t2w uses the exact reverse call order.
        flow_w2t = self.flow_estimator(wide_processed, tele_processed)
        flow_t2w = self.flow_estimator(tele_processed, wide_processed)
        self._validate_flow(flow_w2t, wide_processed, "flow_w2t")
        self._validate_flow(flow_t2w, tele_processed, "flow_t2w")

        warped_result = warp(tele_processed, flow_w2t, return_mask=True)
        if not isinstance(warped_result, tuple) or len(warped_result) != 2:
            raise RuntimeError("warp(..., return_mask=True) must return (warped, valid_mask)")
        warped_tele, warp_valid_mask = warped_result

        wide_y = extract_luminance(wide_processed)
        warped_tele_y = extract_luminance(warped_tele)
        # OcclusionMask performs its own reverse-flow warp and marks the same
        # out-of-bounds Tele sampling region as unreliable.
        occlusion_mask = self.occlusion(flow_w2t, flow_t2w)
        rejection_mask = self.rejection(
            wide_y,
            warped_tele_y,
            valid_mask=warp_valid_mask,
        )

        fusion_inputs = [wide_y, warped_tele_y, occlusion_mask]
        if self.use_rejection_input:
            fusion_inputs.append(rejection_mask)
        fusion_y = self.fusion(torch.cat(fusion_inputs, dim=1))

        prepared_flow_mask = self._prepare_optional_mask(
            flow_uncertainty_mask, wide_processed, "flow_uncertainty_mask"
        )
        prepared_defocus_mask = self._prepare_optional_mask(
            defocus_mask, wide_processed, "defocus_mask"
        )
        reliability_masks = {
            "occlusion": occlusion_mask,
            "rejection": rejection_mask,
            "flow_uncertainty": prepared_flow_mask,
            "defocus": prepared_defocus_mask,
        }
        final_y, blend_mask = self.blending(fusion_y, wide_y, reliability_masks)
        fusion_rgb = replace_luminance(wide_processed, fusion_y, clamp=True)
        final_rgb = replace_luminance(wide_processed, final_y, clamp=True)

        result = {
            "output": final_rgb,
            "wide": wide_processed,
            "tele": tele_processed,
            "flow_w2t": flow_w2t,
            "flow_t2w": flow_t2w,
            "warped_tele": warped_tele,
            "occlusion_mask": occlusion_mask,
            "rejection_mask": rejection_mask,
            "blend_mask": blend_mask,
            "fusion_y": fusion_y,
            # Useful extra diagnostics that do not change the required API.
            "fusion_rgb": fusion_rgb,
            "warp_valid_mask": warp_valid_mask,
            "final_y": final_y,
        }
        if prepared_flow_mask is not None:
            result["flow_uncertainty_mask"] = prepared_flow_mask
        if prepared_defocus_mask is not None:
            result["defocus_mask"] = prepared_defocus_mask
        return result


def build_model(
    config: Mapping[str, Any],
    flow_estimator: Optional[nn.Module] = None,
) -> HybridZoomModel:
    """Config-friendly factory retained for training/test command-line tools."""
    return HybridZoomModel(config=config, flow_estimator=flow_estimator)


__all__ = ["HybridZoomModel", "build_model"]
