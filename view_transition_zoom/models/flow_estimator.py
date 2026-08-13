"""Adapters for official FlowFormer, torchvision RAFT, and debug flow sources.

The public convention is fixed: ``FlowEstimator(wide, tele)`` returns
``F_T2W`` defined on Wide/output coordinates. For a Wide pixel ``p``, the
corresponding Tele sample is ``p + F_T2W(p)``. Thus
``backward_warp(tele, F_T2W)`` aligns Tele to the Wide viewpoint.
"""

from __future__ import annotations

import contextlib
import importlib
import math
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from utils.flow_utils import finite_flow, resize_flow


SizeLike = Union[int, Sequence[int], torch.Size]


def _raft_size(value: Optional[SizeLike]) -> Optional[Tuple[int, int]]:
    """Normalize a config-friendly RAFT inference size to ``(height, width)``."""
    if value is None:
        return None
    if isinstance(value, int):
        result = (value, value)
    else:
        if len(value) != 2:
            raise ValueError("raft_input_size must contain [height, width]")
        result = (int(value[0]), int(value[1]))
    if result[0] <= 0 or result[1] <= 0:
        raise ValueError("raft_input_size values must be positive")
    return result


def _resolve_raft_weights(variant: str, weights: Any) -> Any:
    """Resolve simple strings to the matching torchvision RAFT weight enum."""
    try:
        from torchvision.models.optical_flow import Raft_Large_Weights, Raft_Small_Weights
    except (ImportError, AttributeError) as exc:
        raise ImportError(
            "torchvision RAFT weight enums are unavailable; install a torchvision "
            "version compatible with the installed PyTorch (torchvision>=0.14)."
        ) from exc

    enum_type = Raft_Large_Weights if variant == "large" else Raft_Small_Weights
    if weights is None or weights is False:
        return None
    if not isinstance(weights, str):
        return weights
    name = weights.strip()
    if name.lower() in ("", "none", "null", "false"):
        return None
    if name.lower() == "default":
        return enum_type.DEFAULT
    member_name = name.upper()
    if hasattr(enum_type, member_name):
        return getattr(enum_type, member_name)
    choices = ["default", "none"] + list(enum_type.__members__.keys())
    raise ValueError("Unknown %s RAFT weights %r; choose one of %s" % (variant, weights, choices))


@contextmanager
def _official_import_scope(repo_path: Path):
    """Temporarily expose FlowFormer-Official's top-level import layout.

    The official repository imports ``utils`` as a top-level package, which
    collides with this project's own package. Imported model objects retain
    their direct helper references, so restoring our modules after construction
    lets both codebases coexist in one process.
    """
    roots = ("utils", "configs")
    saved_modules = {
        name: module
        for name, module in list(sys.modules.items())
        if name in roots or any(name.startswith(root + ".") for root in roots)
    }
    original_path = list(sys.path)
    for name in saved_modules:
        sys.modules.pop(name, None)
    try:
        # The official source expects core/utils to win for ``import utils``.
        sys.path.insert(0, str(repo_path))
        sys.path.insert(0, str(repo_path / "core"))
        yield
    finally:
        for name in list(sys.modules):
            if name in roots or any(name.startswith(root + ".") for root in roots):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
        sys.path[:] = original_path


def _extract_flow(prediction: Any) -> Tensor:
    if isinstance(prediction, Tensor):
        flow = prediction
    elif isinstance(prediction, (tuple, list)) and prediction:
        flow = prediction[-1]
        if isinstance(flow, (tuple, list)):
            flow = flow[-1]
    elif isinstance(prediction, Mapping):
        flow = None
        for key in ("flow", "flows", "flow_preds", "predictions"):
            if key in prediction:
                flow = prediction[key]
                break
        if isinstance(flow, (tuple, list)):
            flow = flow[-1]
    else:
        flow = None
    if not isinstance(flow, Tensor) or flow.ndim != 4 or flow.shape[1] != 2:
        raise RuntimeError("Flow model must return [B,2,H,W], got %r" % (getattr(flow, "shape", None),))
    return flow


def _extract_flowformer_flow(prediction: Any) -> Tensor:
    """Extract FlowFormer's full-resolution evaluation result.

    Official evaluation builds commonly return ``(flow_up, flow_low)`` while
    training-style builds return a prediction list. The first tuple element is
    therefore intentionally selected, unlike RAFT's final-list convention.
    """
    if isinstance(prediction, Tensor):
        flow = prediction
    elif isinstance(prediction, tuple) and prediction and isinstance(prediction[0], Tensor):
        flow = prediction[0]
    elif isinstance(prediction, list) and prediction:
        flow = prediction[-1]
    elif isinstance(prediction, Mapping):
        return _extract_flow(prediction)
    else:
        flow = None
    if not isinstance(flow, Tensor) or flow.ndim != 4 or flow.shape[1] != 2:
        raise RuntimeError(
            "FlowFormer must return full-resolution flow [B,2,H,W], got %r"
            % (getattr(flow, "shape", None),)
        )
    return flow


def _validate_pair(wide: Tensor, tele: Tensor) -> None:
    if wide.ndim != 4 or wide.shape[1] != 3:
        raise ValueError("wide must be RGB [B,3,H,W]")
    if wide.shape != tele.shape:
        raise ValueError("wide and tele must have identical [B,3,H,W] shapes")
    if not wide.is_floating_point() or not tele.is_floating_point():
        raise TypeError("wide and tele must be floating point tensors in [0,1]")
    if wide.device != tele.device:
        raise ValueError("wide and tele must be on the same device")


class _TorchvisionRaft(nn.Module):
    """Torchvision RAFT adapter using the project's Wide-to-Tele convention."""

    def __init__(
        self,
        weights: Any = "default",
        variant: str = "large",
        input_size: Optional[SizeLike] = None,
        progress: bool = True,
        minimum_size: int = 128,
        pad_multiple: int = 8,
    ) -> None:
        super().__init__()
        try:
            from torchvision.models.optical_flow import raft_large, raft_small
        except (ImportError, AttributeError) as exc:
            raise ImportError(
                "torchvision.models.optical_flow.RAFT is unavailable; install a "
                "torchvision build compatible with the installed PyTorch."
            ) from exc

        variant = variant.lower().replace("raft_", "").replace("-", "_")
        if variant not in ("large", "small"):
            raise ValueError("RAFT variant must be 'large' or 'small'")
        if minimum_size <= 0 or pad_multiple <= 0:
            raise ValueError("RAFT minimum_size and pad_multiple must be positive")

        self.variant = variant
        self.input_size = _raft_size(input_size)
        self.minimum_size = int(minimum_size)
        self.pad_multiple = int(pad_multiple)
        constructor = raft_large if variant == "large" else raft_small
        resolved = _resolve_raft_weights(variant, weights)
        self.model = constructor(weights=resolved, progress=bool(progress)).eval()
        self.model.requires_grad_(False)

    def _pad(self, image: Tensor) -> Tuple[Tensor, Tuple[int, int, int, int]]:
        height, width = image.shape[-2:]
        target_h = int(
            math.ceil(max(self.minimum_size, height) / float(self.pad_multiple))
            * self.pad_multiple
        )
        target_w = int(
            math.ceil(max(self.minimum_size, width) / float(self.pad_multiple))
            * self.pad_multiple
        )
        pad_h, pad_w = target_h - height, target_w - width
        top, left = pad_h // 2, pad_w // 2
        padding = (left, pad_w - left, top, pad_h - top)
        return F.pad(image, padding, mode="replicate"), padding

    def forward(self, wide: Tensor, tele: Tensor) -> Tensor:
        original_size = tuple(wide.shape[-2:])
        estimation_size = self.input_size or original_size
        first, second = wide.float(), tele.float()
        if estimation_size != original_size:
            first = F.interpolate(first, size=estimation_size, mode="bilinear", align_corners=False)
            second = F.interpolate(second, size=estimation_size, mode="bilinear", align_corners=False)

        # The official weight transform applies this same [0,1] -> [-1,1] map.
        first, padding = self._pad(first.mul(2.0).sub(1.0))
        second, second_padding = self._pad(second.mul(2.0).sub(1.0))
        if padding != second_padding:
            raise RuntimeError("Internal RAFT padding mismatch")
        with torch.no_grad():
            flow = _extract_flow(self.model(first, second))
        left, _, top, _ = padding
        estimate_h, estimate_w = estimation_size
        flow = flow[..., top : top + estimate_h, left : left + estimate_w]
        if estimation_size != original_size:
            flow = resize_flow(flow, original_size)
        return flow


class _OfficialFlowFormer(nn.Module):
    """Thin loader for the unmodified FlowFormer-Official repository.

    PAPER_AMBIGUITY: FlowFormer-Official does not publish one universal
    photography checkpoint. The configured checkpoint choice remains an
    experiment variable and is documented in README.
    """

    def __init__(self, repo: str, checkpoint: str, mixed_precision: bool = False) -> None:
        super().__init__()
        repo_path = Path(repo).expanduser().resolve()
        checkpoint_path = Path(checkpoint).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = repo_path / checkpoint_path
        if not repo_path.is_dir():
            raise FileNotFoundError(
                "FlowFormer-Official repository not found at %s. Clone "
                "https://github.com/drinkingcoder/FlowFormer-Official there." % repo_path
            )
        if not checkpoint_path.is_file():
            raise FileNotFoundError("FlowFormer checkpoint not found: %s" % checkpoint_path)

        try:
            with _official_import_scope(repo_path):
                get_cfg = importlib.import_module("configs.things_eval").get_cfg
                build_flowformer = importlib.import_module("FlowFormer").build_flowformer
                cfg = get_cfg()
                if hasattr(cfg, "defrost"):
                    cfg.defrost()
                cfg.model = str(checkpoint_path)
                if hasattr(cfg, "mixed_precision"):
                    cfg.mixed_precision = bool(mixed_precision)
                model = build_flowformer(cfg)
        except Exception as exc:
            raise ImportError(
                "Could not import the official FlowFormer code. Install its pinned "
                "dependencies (including yacs, loguru, einops, and timm==0.4.12)."
            ) from exc

        payload = torch.load(str(checkpoint_path), map_location="cpu")
        state = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        if any(str(key).startswith("module.") for key in state.keys()):
            state = {str(key)[7:] if str(key).startswith("module.") else key: value for key, value in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            warnings.warn(
                "FlowFormer checkpoint loaded non-strictly (missing=%d, unexpected=%d)."
                % (len(missing), len(unexpected)),
                RuntimeWarning,
            )
        self.model = model.eval()
        self.model.requires_grad_(False)

    def forward(self, wide: Tensor, tele: Tensor) -> Tensor:
        height, width = wide.shape[-2:]
        target_h = int(math.ceil(height / 8.0) * 8)
        target_w = int(math.ceil(width / 8.0) * 8)
        pad_h, pad_w = target_h - height, target_w - width
        first = F.pad(wide.float() * 255.0, (0, pad_w, 0, pad_h), mode="replicate")
        second = F.pad(tele.float() * 255.0, (0, pad_w, 0, pad_h), mode="replicate")
        with torch.no_grad():
            prediction = self.model(first, second)
        flow = _extract_flowformer_flow(prediction)
        return flow[..., :height, :width]


class FlowEstimator(nn.Module):
    """Unified flow estimator with explicit fallback and offline debug modes."""

    def __init__(
        self,
        model: str = "flowformer",
        fallback: Optional[str] = "raft",
        weights: Any = "default",
        flowformer_repo: str = "third_party/FlowFormer-Official",
        flowformer_checkpoint: str = "checkpoints/things.pth",
        precomputed_path: Optional[str] = None,
        raft_variant: str = "large",
        raft_weights: Any = None,
        raft_input_size: Optional[SizeLike] = None,
        raft_progress: bool = True,
        raft_minimum_size: int = 128,
        raft_pad_multiple: int = 8,
        mixed_precision: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.backend_name = model.lower().replace("-", "_")
        self.precomputed_path = precomputed_path
        effective_raft_weights = weights if raft_weights is None else raft_weights
        try:
            self.backend = self._build_backend(
                self.backend_name,
                effective_raft_weights,
                flowformer_repo,
                flowformer_checkpoint,
                raft_variant,
                raft_input_size,
                raft_progress,
                raft_minimum_size,
                raft_pad_multiple,
                mixed_precision,
            )
        except (ImportError, FileNotFoundError, RuntimeError) as exc:
            if fallback is None or fallback.lower() in ("none", self.backend_name):
                raise
            warnings.warn(
                "%s backend unavailable (%s). Falling back to %s."
                % (self.backend_name, exc, fallback),
                RuntimeWarning,
            )
            self.backend_name = fallback.lower()
            self.backend = self._build_backend(
                self.backend_name,
                effective_raft_weights,
                flowformer_repo,
                flowformer_checkpoint,
                raft_variant,
                raft_input_size,
                raft_progress,
                raft_minimum_size,
                raft_pad_multiple,
                mixed_precision,
            )
        if device is not None:
            self.to(device)

    def _build_backend(
        self,
        name: str,
        weights: Any,
        repo: str,
        checkpoint: str,
        raft_variant: str,
        raft_input_size: Optional[SizeLike],
        raft_progress: bool,
        raft_minimum_size: int,
        raft_pad_multiple: int,
        mixed_precision: bool,
    ) -> Optional[nn.Module]:
        if name == "flowformer":
            return _OfficialFlowFormer(repo, checkpoint, mixed_precision=mixed_precision)
        if name in ("raft_large", "torchvision_raft_large"):
            raft_variant = "large"
        elif name in ("raft_small", "torchvision_raft_small"):
            raft_variant = "small"
        if name in (
            "raft",
            "torchvision_raft",
            "raft_large",
            "raft_small",
            "torchvision_raft_large",
            "torchvision_raft_small",
        ):
            return _TorchvisionRaft(
                weights=weights,
                variant=raft_variant,
                input_size=raft_input_size,
                progress=raft_progress,
                minimum_size=raft_minimum_size,
                pad_multiple=raft_pad_multiple,
            )
        if name in ("precomputed", "farneback"):
            return None
        raise ValueError("Unknown flow model: %s" % name)

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        device: Optional[torch.device] = None,
    ) -> "FlowEstimator":
        section = config.get("flow", config)
        config_path = config.get("_config_path")
        config_root = Path(config_path).parent if config_path else Path.cwd()
        repo = Path(str(section.get("flowformer_repo", "third_party/FlowFormer-Official"))).expanduser()
        if not repo.is_absolute():
            repo = config_root / repo
        precomputed = section.get("precomputed_path")
        if precomputed:
            precomputed_path = Path(str(precomputed)).expanduser()
            if not precomputed_path.is_absolute():
                precomputed_path = config_root / precomputed_path
            precomputed = str(precomputed_path)
        model_name = str(section.get("model", "flowformer"))
        model_key = model_name.lower().replace("-", "_")
        default_variant = "small" if model_key.endswith("raft_small") else "large"
        return cls(
            model=model_name,
            fallback=section.get("fallback", "raft"),
            weights=section.get("weights", "default"),
            flowformer_repo=str(repo),
            flowformer_checkpoint=str(section.get("flowformer_checkpoint", "checkpoints/things.pth")),
            precomputed_path=precomputed,
            raft_variant=str(section.get("raft_variant", section.get("variant", default_variant))),
            raft_weights=section.get("raft_weights", section.get("weights", "default")),
            raft_input_size=section.get("raft_input_size"),
            raft_progress=bool(section.get("raft_progress", True)),
            raft_minimum_size=int(section.get("raft_minimum_size", 128)),
            raft_pad_multiple=int(section.get("raft_pad_multiple", 8)),
            mixed_precision=bool(section.get("mixed_precision", False)),
            device=device,
        )

    def _load_precomputed(self, target: Tensor) -> Tensor:
        if not self.precomputed_path:
            raise ValueError("flow.precomputed_path is required for model=precomputed")
        path = Path(self.precomputed_path).expanduser()
        if path.suffix.lower() == ".npy":
            array = np.load(str(path))
            flow = torch.from_numpy(array)
        else:
            payload = torch.load(str(path), map_location="cpu")
            flow = payload.get("flow", payload) if isinstance(payload, dict) else payload
        if flow.ndim == 3:
            if flow.shape[0] == 2:
                flow = flow.unsqueeze(0)
            elif flow.shape[-1] == 2:
                flow = flow.permute(2, 0, 1).unsqueeze(0)
        if flow.ndim != 4 or flow.shape[1] != 2:
            raise ValueError("Precomputed flow must be [B,2,H,W] or [H,W,2]")
        flow = flow.to(device=target.device, dtype=torch.float32)
        if flow.shape[0] == 1 and target.shape[0] > 1:
            flow = flow.expand(target.shape[0], -1, -1, -1)
        if flow.shape[-2:] != target.shape[-2:]:
            flow = resize_flow(flow, target.shape[-2:])
        return flow

    @staticmethod
    def _farneback(wide: Tensor, tele: Tensor) -> Tensor:
        results = []
        for first, second in zip(wide, tele):
            first_np = (first.detach().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
            second_np = (second.detach().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
            first_gray = cv2.cvtColor(first_np, cv2.COLOR_RGB2GRAY)
            second_gray = cv2.cvtColor(second_np, cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                first_gray, second_gray, None, 0.5, 5, 21, 5, 7, 1.5, 0
            )
            results.append(torch.from_numpy(flow).permute(2, 0, 1))
        return torch.stack(results).to(device=wide.device, dtype=torch.float32)

    def forward(self, wide: Tensor, tele: Tensor) -> Tensor:
        _validate_pair(wide, tele)
        if self.backend_name == "precomputed":
            flow = self._load_precomputed(wide)
        elif self.backend_name == "farneback":
            flow = self._farneback(wide, tele)
        else:
            with contextlib.nullcontext():
                flow = self.backend(wide, tele)
        flow = finite_flow(flow.float())
        if flow.shape != (wide.shape[0], 2, wide.shape[2], wide.shape[3]):
            raise RuntimeError("Estimator returned unexpected flow shape: %r" % (tuple(flow.shape),))
        return flow


__all__ = ["FlowEstimator"]
