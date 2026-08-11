"""Optical-flow estimators used by the hybrid-zoom pipeline.

The project flow convention is deliberately explicit.  Calling
``FlowEstimator(wide, tele)`` returns ``flow_w2t`` such that, for every pixel
``p`` in the Wide (first-image) coordinate system::

    p_tele = p_wide + flow_w2t(p_wide)

Consequently ``warp(tele, flow_w2t)`` backward-samples Tele into the Wide
coordinate system.  Torchvision RAFT estimates flow from its first image to
its second image, so the wrapper calls it in exactly that order.
"""

from __future__ import annotations

import contextlib
import math
import warnings
from typing import Any, ContextManager, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from hybrid_zoom.modules.warp import resize_flow


SizeLike = Union[int, Sequence[int], torch.Size]


def _pair(value: Optional[SizeLike], name: str) -> Optional[Tuple[int, int]]:
    """Convert an optional size-like value to a positive ``(height, width)``."""
    if value is None:
        return None
    if isinstance(value, int):
        result = (value, value)
    else:
        if len(value) != 2:
            raise ValueError("{} must contain exactly two values, got {!r}".format(name, value))
        result = (int(value[0]), int(value[1]))
    if result[0] <= 0 or result[1] <= 0:
        raise ValueError("{} values must be positive, got {!r}".format(name, result))
    return result


def _resolve_weights(variant: str, weights: Any) -> Tuple[Any, bool]:
    """Resolve a user-friendly weight selection to torchvision's enum value.

    Returns the enum (or ``None``) and a legacy ``pretrained`` boolean used only
    as a compatibility fallback for older torchvision constructors.
    """
    normalized_variant = variant.lower().replace("-", "_")
    try:
        from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights
    except (ImportError, AttributeError) as exc:  # pragma: no cover - old torchvision
        if weights is None or weights is False or str(weights).lower() in {"none", "null", "false"}:
            return None, False
        if str(weights).lower() == "default" or weights is True:
            return None, True
        raise RuntimeError(
            "This torchvision version does not expose RAFT weight enums. "
            "Use weights='default' or weights=None, or install torchvision>=0.16."
        ) from exc

    enum_type = Raft_Large_Weights if normalized_variant == "large" else Raft_Small_Weights
    if weights is None or weights is False:
        return None, False
    if isinstance(weights, str):
        normalized = weights.strip()
        if normalized.lower() in {"", "none", "null", "false"}:
            return None, False
        if normalized.lower() == "default":
            return enum_type.DEFAULT, True
        # Accept documented enum member names, case-insensitively.
        member_name = normalized.upper()
        if hasattr(enum_type, member_name):
            return getattr(enum_type, member_name), True
        choices = ["default", "none"] + list(enum_type.__members__.keys())
        raise ValueError("Unknown RAFT weights {!r}; choose one of {}".format(weights, choices))
    if weights is True:
        return enum_type.DEFAULT, True
    # Advanced callers may pass an actual torchvision WeightsEnum instance.
    return weights, True


def _extract_final_flow(prediction: Any) -> Tensor:
    """Extract the final full-resolution flow from supported RAFT return forms."""
    if isinstance(prediction, Tensor):
        flow = prediction
    elif isinstance(prediction, (list, tuple)):
        if not prediction:
            raise RuntimeError("RAFT returned an empty prediction sequence")
        flow = prediction[-1]
    elif isinstance(prediction, Mapping):
        candidates = ("flow", "flows", "flow_preds", "predictions")
        value = None
        for key in candidates:
            if key in prediction:
                value = prediction[key]
                break
        if value is None:
            raise RuntimeError("Unsupported RAFT output mapping keys: {}".format(list(prediction.keys())))
        flow = value[-1] if isinstance(value, (list, tuple)) else value
    else:
        raise RuntimeError("Unsupported RAFT output type: {}".format(type(prediction).__name__))
    if not isinstance(flow, Tensor) or flow.ndim != 4 or flow.shape[1] != 2:
        shape = getattr(flow, "shape", None)
        raise RuntimeError("RAFT must return flow shaped [B, 2, H, W], got {}".format(shape))
    return flow


class FlowEstimator(nn.Module):
    """A robust wrapper around torchvision's official RAFT implementation.

    Args:
        variant: ``"large"`` or ``"small"``.
        weights: ``"default"`` for torchvision pretrained weights, ``None`` /
            ``"none"`` for random initialization, or a named weight enum member.
        freeze: Disable RAFT gradients when ``True`` (the research default).
        input_size: Optional lower-resolution ``(height, width)`` used only while
            estimating flow.  The flow is resized back with displacement scaling.
        input_height: Alternative config-friendly way to specify input height.
        input_width: Alternative config-friendly way to specify input width.
        train_mode_when_frozen: Keep RAFT in train mode despite frozen parameters.
            The default keeps the frozen model in evaluation mode.
        device: Optional initial device.  Unavailable CUDA falls back to CPU.
        pad_multiple: Torchvision RAFT requires spatial dimensions divisible by 8.
        minimum_size: Small inputs are padded because RAFT's four-level correlation
            pyramid requires feature maps of at least 16 pixels (128 input pixels).

    Inputs are RGB float tensors in ``[0, 1]`` shaped ``[B, 3, H, W]``.  The
    returned float tensor is shaped ``[B, 2, H, W]`` with ``dx`` then ``dy`` in
    pixel units of the original input resolution.
    """

    def __init__(
        self,
        variant: str = "large",
        weights: Any = "default",
        freeze: bool = True,
        input_size: Optional[SizeLike] = None,
        input_height: Optional[int] = None,
        input_width: Optional[int] = None,
        train_mode_when_frozen: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        pad_multiple: int = 8,
        minimum_size: int = 128,
        progress: bool = True,
    ) -> None:
        super().__init__()
        normalized_variant = variant.lower().replace("raft_", "").replace("-", "_")
        if normalized_variant not in {"large", "small"}:
            raise ValueError("variant must be 'large' or 'small', got {!r}".format(variant))
        if pad_multiple <= 0:
            raise ValueError("pad_multiple must be positive")
        if minimum_size <= 0:
            raise ValueError("minimum_size must be positive")
        if input_size is not None and (input_height is not None or input_width is not None):
            raise ValueError("Specify either input_size or input_height/input_width, not both")
        if (input_height is None) != (input_width is None):
            raise ValueError("input_height and input_width must be specified together")
        if input_size is None and input_height is not None:
            input_size = (input_height, int(input_width))

        self.variant = normalized_variant
        self.input_size = _pair(input_size, "input_size")
        self.freeze = bool(freeze)
        self.train_mode_when_frozen = bool(train_mode_when_frozen)
        self.pad_multiple = int(pad_multiple)
        self.minimum_size = int(minimum_size)

        try:
            from torchvision.models.optical_flow import raft_large, raft_small
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "FlowEstimator requires torchvision.models.optical_flow.RAFT; "
                "install a torchvision build compatible with the installed PyTorch."
            ) from exc

        constructor = raft_large if normalized_variant == "large" else raft_small
        resolved_weights, legacy_pretrained = _resolve_weights(normalized_variant, weights)
        try:
            self.raft = constructor(weights=resolved_weights, progress=progress)
        except TypeError:  # pragma: no cover - compatibility with older torchvision
            warnings.warn(
                "Falling back to torchvision's legacy pretrained= RAFT API. "
                "Upgrade torchvision for named weight selection.",
                RuntimeWarning,
            )
            self.raft = constructor(pretrained=legacy_pretrained, progress=progress)

        if self.freeze:
            self.requires_grad_(False)
        if self.freeze and not self.train_mode_when_frozen:
            self.raft.eval()

        if device is not None:
            resolved_device = torch.device(device)
            if resolved_device.type == "cuda" and not torch.cuda.is_available():
                warnings.warn("CUDA is unavailable; placing RAFT on CPU instead.", RuntimeWarning)
                resolved_device = torch.device("cpu")
            self.to(resolved_device)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        device: Optional[Union[str, torch.device]] = None,
    ) -> "FlowEstimator":
        """Build from either the full nested config or its ``flow`` section."""
        section = config.get("flow", config)
        if not isinstance(section, Mapping):
            raise TypeError("flow configuration must be a mapping")
        model_name = str(section.get("model", "raft")).lower()
        if model_name not in {"raft", "torchvision_raft", "raft_large", "raft_small"}:
            raise ValueError("Only torchvision RAFT is supported, got {!r}".format(model_name))
        variant = section.get("variant", "small" if model_name == "raft_small" else "large")
        height = section.get("input_height")
        width = section.get("input_width")
        kwargs: Dict[str, Any] = {
            "variant": variant,
            "weights": section.get("weights", "default"),
            "freeze": section.get("freeze", True),
            "train_mode_when_frozen": section.get("train_mode_when_frozen", False),
            "device": device,
        }
        if height is not None or width is not None:
            kwargs["input_height"] = height
            kwargs["input_width"] = width
        if "pad_multiple" in section:
            kwargs["pad_multiple"] = section["pad_multiple"]
        if "minimum_size" in section:
            kwargs["minimum_size"] = section["minimum_size"]
        if "progress" in section:
            kwargs["progress"] = section["progress"]
        return cls(**kwargs)

    def train(self, mode: bool = True) -> "FlowEstimator":
        """Preserve evaluation mode for a frozen RAFT unless explicitly requested."""
        super().train(mode)
        if self.freeze and not self.train_mode_when_frozen:
            self.raft.eval()
        return self

    @staticmethod
    def _validate_inputs(img1: Tensor, img2: Tensor) -> None:
        if not isinstance(img1, Tensor) or not isinstance(img2, Tensor):
            raise TypeError("img1 and img2 must be torch.Tensor instances")
        if img1.ndim != 4 or img2.ndim != 4:
            raise ValueError("RAFT inputs must be [B, 3, H, W] tensors")
        if img1.shape[1] != 3 or img2.shape[1] != 3:
            raise ValueError("RAFT inputs must have exactly three RGB channels")
        if img1.shape != img2.shape:
            raise ValueError("RAFT inputs must have identical shapes, got {} and {}".format(
                tuple(img1.shape), tuple(img2.shape)
            ))
        if not img1.is_floating_point() or not img2.is_floating_point():
            raise TypeError("RAFT inputs must be floating-point RGB tensors in [0, 1]")
        if img1.device != img2.device:
            raise ValueError("RAFT inputs must be on the same device")
        if img1.shape[-2] <= 0 or img1.shape[-1] <= 0:
            raise ValueError("RAFT inputs must have non-empty spatial dimensions")

    def _pad(self, image: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        height, width = image.shape[-2:]
        padded_height = int(
            math.ceil(max(height, self.minimum_size) / self.pad_multiple) * self.pad_multiple
        )
        padded_width = int(
            math.ceil(max(width, self.minimum_size) / self.pad_multiple) * self.pad_multiple
        )
        pad_height = padded_height - height
        pad_width = padded_width - width
        if pad_height == 0 and pad_width == 0:
            return image, (0, 0)
        # Symmetric padding limits artificial displacement near a single border.
        top = pad_height // 2
        bottom = pad_height - top
        left = pad_width // 2
        right = pad_width - left
        # Replication works for arbitrarily small images, unlike reflection padding.
        return F.pad(image, (left, right, top, bottom), mode="replicate"), (top, left)

    def forward(self, img1: Tensor, img2: Tensor) -> Tensor:
        """Estimate first-to-second flow and return it at the original resolution.

        ``forward(wide, tele)`` computes the desired ``flow_w2t``.  Each output
        vector ``(dx, dy)`` is expressed in pixels and locates the corresponding
        Tele sample at ``(x + dx, y + dy)`` for a Wide pixel ``(x, y)``.
        """
        self._validate_inputs(img1, img2)
        original_size = tuple(img1.shape[-2:])
        estimation_size = self.input_size or original_size

        # RAFT is kept in float32 for CPU compatibility and stable correlation.
        first = img1.float()
        second = img2.float()
        if tuple(estimation_size) != original_size:
            first = F.interpolate(first, size=estimation_size, mode="bilinear", align_corners=False)
            second = F.interpolate(second, size=estimation_size, mode="bilinear", align_corners=False)

        first, offset = self._pad(first)
        second, second_offset = self._pad(second)
        if offset != second_offset:  # Defensive; equal inputs must produce equal padding.
            raise RuntimeError("Internal RAFT padding mismatch")

        # Torchvision RAFT expects normalized RGB values in [-1, 1].  Its official
        # weight transforms perform this same affine mapping independently per image.
        first = first.mul(2.0).sub(1.0)
        second = second.mul(2.0).sub(1.0)
        grad_context: ContextManager[Any]
        grad_context = torch.no_grad() if self.freeze else contextlib.nullcontext()
        with grad_context:
            prediction = self.raft(first, second)
        flow = _extract_final_flow(prediction)

        estimate_height, estimate_width = estimation_size
        top, left = offset
        flow = flow[..., top : top + estimate_height, left : left + estimate_width]
        if tuple(estimation_size) != original_size:
            # resize_flow interpolates and also rescales dx by W/Wf and dy by H/Hf.
            flow = resize_flow(flow, original_size)
        return flow


__all__ = ["FlowEstimator"]
