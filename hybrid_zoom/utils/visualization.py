"""Optical-flow and full-pipeline visualization helpers."""

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from .image_utils import save_image

PathLike = Union[str, Path]


def _make_colorwheel(device: torch.device, dtype: torch.dtype) -> Tensor:
    """Create the standard Middlebury optical-flow color wheel."""

    transitions = (15, 6, 4, 11, 13, 6)
    wheel = torch.zeros(sum(transitions), 3, device=device, dtype=dtype)
    index = 0
    ry, yg, gc, cb, bm, mr = transitions
    wheel[index : index + ry, 0] = 1.0
    wheel[index : index + ry, 1] = torch.arange(ry, device=device, dtype=dtype) / ry
    index += ry
    wheel[index : index + yg, 0] = 1.0 - torch.arange(yg, device=device, dtype=dtype) / yg
    wheel[index : index + yg, 1] = 1.0
    index += yg
    wheel[index : index + gc, 1] = 1.0
    wheel[index : index + gc, 2] = torch.arange(gc, device=device, dtype=dtype) / gc
    index += gc
    wheel[index : index + cb, 1] = 1.0 - torch.arange(cb, device=device, dtype=dtype) / cb
    wheel[index : index + cb, 2] = 1.0
    index += cb
    wheel[index : index + bm, 2] = 1.0
    wheel[index : index + bm, 0] = torch.arange(bm, device=device, dtype=dtype) / bm
    index += bm
    wheel[index : index + mr, 2] = 1.0 - torch.arange(mr, device=device, dtype=dtype) / mr
    wheel[index : index + mr, 0] = 1.0
    return wheel


def make_colorwheel(
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return the standard Middlebury RGB color wheel as ``[55,3]``.

    This public wrapper keeps wheel construction available to diagnostics and
    tests without exposing the private implementation used by
    :func:`flow_to_color`.
    """

    if not dtype.is_floating_point:
        raise TypeError("dtype must be floating point")
    resolved_device = torch.device("cpu") if device is None else torch.device(device)
    return _make_colorwheel(resolved_device, dtype)


def flow_to_color(flow: Tensor, clip_flow: Optional[float] = None) -> Tensor:
    """Render CHW/BCHW ``(dx,dy)`` flow with the Middlebury color wheel.

    Returns RGB float values in [0, 1], retaining whether the input was batched.
    Invalid (NaN/Inf) vectors are rendered black.
    """

    single = flow.ndim == 3
    if single:
        flow = flow.unsqueeze(0)
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError("flow must have shape [2,H,W] or [B,2,H,W]")
    if clip_flow is not None and clip_flow <= 0:
        raise ValueError("clip_flow must be positive")

    working = flow.detach()
    if not working.is_floating_point():
        working = working.float()
    working = working.float()
    valid = torch.isfinite(working).all(dim=1)
    working = torch.nan_to_num(working, nan=0.0, posinf=0.0, neginf=0.0)
    if clip_flow is not None:
        working = working.clamp(-float(clip_flow), float(clip_flow))
    u, v = working[:, 0], working[:, 1]
    radius = torch.sqrt(u.square() + v.square())
    max_radius = radius.flatten(1).amax(dim=1).view(-1, 1, 1).clamp_min(1e-6)
    u, v, radius = u / max_radius, v / max_radius, radius / max_radius

    wheel = _make_colorwheel(working.device, working.dtype)
    angle = torch.atan2(-v, -u) / torch.pi
    position = (angle + 1.0) * 0.5 * (wheel.shape[0] - 1)
    lower = torch.floor(position).long()
    upper = (lower + 1) % wheel.shape[0]
    fraction = (position - lower.float()).unsqueeze(-1)
    color = wheel[lower] * (1.0 - fraction) + wheel[upper] * fraction
    inside = radius <= 1.0
    color = torch.where(
        inside.unsqueeze(-1),
        1.0 - radius.unsqueeze(-1) * (1.0 - color),
        color * 0.75,
    )
    color = color * valid.unsqueeze(-1)
    result = color.permute(0, 3, 1, 2).clamp(0.0, 1.0)
    return result[0] if single else result


def _as_batch(value: Tensor) -> Tensor:
    if value.ndim == 2:
        return value.unsqueeze(0).unsqueeze(0)
    if value.ndim == 3:
        return value.unsqueeze(0)
    if value.ndim != 4:
        raise ValueError("Pipeline image tensors must be HW, CHW, or BCHW")
    return value


def _names_for_batch(
    names: Optional[Union[PathLike, Sequence[PathLike]]], batch_size: int
) -> List[str]:
    if names is None:
        return ["{:06d}.png".format(index) for index in range(batch_size)]
    if isinstance(names, (str, Path)):
        result = [str(names)]
    else:
        result = [str(name) for name in names]
    if len(result) != batch_size:
        raise ValueError("Expected {} names, received {}".format(batch_size, len(result)))
    return result


def save_pipeline_outputs(
    outputs: Mapping[str, Tensor],
    output_dir: PathLike,
    names: Optional[Union[PathLike, Sequence[PathLike]]] = None,
    flat: bool = False,
) -> Dict[str, List[Path]]:
    """Save available model intermediates for test runs or a single demo.

    With ``flat=False`` results follow ``final/``, ``flow/`` and ``masks/...``
    directories.  With ``flat=True`` a singleton batch is written as the demo
    filenames ``final.png``, ``flow.png``, ``occlusion_mask.png``, etc.
    """

    root = Path(output_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    if not flat:
        # Always materialize the documented result layout, including when a
        # caller provides a partial diagnostic mapping.
        for folder in (
            "final",
            "warped_tele",
            "fusion",
            "flow",
            "masks/occlusion",
            "masks/rejection",
            "masks/blend",
        ):
            (root / folder).mkdir(parents=True, exist_ok=True)
    final_key = "output" if "output" in outputs else "final"
    fusion_key = "fusion_rgb" if "fusion_rgb" in outputs else "fusion_y"
    specs: List[Tuple[str, str, str]] = [
        (final_key, "final", "final.png"),
        ("wide", "wide", "wide.png"),
        ("tele", "tele", "tele.png"),
        ("warped_tele", "warped_tele", "warped_tele.png"),
        (fusion_key, "fusion", "fusion.png"),
        ("flow_w2t", "flow", "flow.png"),
        ("occlusion_mask", "masks/occlusion", "occlusion_mask.png"),
        ("rejection_mask", "masks/rejection", "rejection_mask.png"),
        ("blend_mask", "masks/blend", "blend_mask.png"),
    ]
    available = [(key, folder, filename) for key, folder, filename in specs if key in outputs]
    if not available:
        raise ValueError("outputs contains no recognized pipeline tensors")
    first = _as_batch(outputs[available[0][0]])
    batch_size = first.shape[0]
    sample_names = _names_for_batch(names, batch_size)
    saved: Dict[str, List[Path]] = {}

    for key, folder, demo_filename in available:
        batch = _as_batch(outputs[key])
        if batch.shape[0] != batch_size:
            raise ValueError("Inconsistent batch size for output {!r}".format(key))
        if key == "flow_w2t":
            batch = _as_batch(flow_to_color(batch))
        paths: List[Path] = []
        for index in range(batch_size):
            source_name = Path(sample_names[index]).name
            output_name = Path(source_name).stem + ".png"
            if flat:
                if batch_size == 1:
                    destination = root / demo_filename
                else:
                    destination = root / (Path(source_name).stem + "_" + demo_filename)
            else:
                destination = root / Path(folder) / output_name
            paths.append(save_image(batch[index], destination))
        saved[key] = paths
    return saved


flow_to_rgb = flow_to_color

__all__ = ["flow_to_color", "flow_to_rgb", "make_colorwheel", "save_pipeline_outputs"]
