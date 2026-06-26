import os
from pathlib import Path
from typing import Any, Optional, Union

import torch


PathLike = Union[str, os.PathLike]


def resolve_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Resolve the runtime device used by ZoomGS.

    ZoomGS still requires CUDA for its rasterizer and KNN extensions. This helper
    centralizes device parsing so CUDA_VISIBLE_DEVICES and --data_device behave
    consistently on newer PyTorch builds.
    """
    if device is None:
        device = os.environ.get("ZOOMGS_DEVICE", "cuda")

    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "ZoomGS was configured to use CUDA, but torch.cuda.is_available() is False. "
            "Install a CUDA-enabled PyTorch build and make sure the NVIDIA driver is visible."
        )
    return resolved


def set_runtime_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    resolved = resolve_device(device)
    if resolved.type == "cuda":
        torch.cuda.set_device(resolved if resolved.index is not None else torch.device("cuda:0"))
    return resolved


def maybe_empty_cache(device: Optional[Union[str, torch.device]] = None) -> None:
    resolved = resolve_device(device)
    if resolved.type == "cuda":
        torch.cuda.empty_cache()


def load_checkpoint(
    path: PathLike,
    map_location: Optional[Union[str, torch.device]] = None,
    *,
    weights_only: bool = False,
) -> Any:
    """Load old ZoomGS checkpoints across PyTorch 2.x releases.

    PyTorch 2.6 changed the default behavior for torch.load. Passing
    weights_only explicitly keeps full-module MLP checkpoints loadable.
    """
    path = Path(path)
    if map_location is None:
        map_location = resolve_device()

    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except TypeError:
        return torch.load(path, map_location=map_location)
