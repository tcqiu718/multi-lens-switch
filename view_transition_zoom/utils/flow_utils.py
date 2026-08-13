from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

import torch
import torch.nn.functional as F


def resize_flow(
    flow: torch.Tensor,
    size: Tuple[int, int],
    mode: str = "bilinear",
) -> torch.Tensor:
    """Resize flow and scale dx/dy into the new pixel coordinate system."""
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError("flow must have shape [B,2,H,W]")
    old_height, old_width = flow.shape[-2:]
    new_height, new_width = int(size[0]), int(size[1])
    if new_height <= 0 or new_width <= 0:
        raise ValueError("size values must be positive")
    if (old_height, old_width) == (new_height, new_width):
        return flow.clone()
    kwargs = {"size": (new_height, new_width), "mode": mode}
    if mode in ("linear", "bilinear", "bicubic", "trilinear"):
        kwargs["align_corners"] = True
    resized = F.interpolate(flow, **kwargs)
    resized[:, 0] *= float(new_width) / float(old_width)
    resized[:, 1] *= float(new_height) / float(old_height)
    return resized


def same_box_filter(tensor: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Replicate-padded box mean preserving size for odd or even kernels."""
    if tensor.ndim != 4:
        raise ValueError("tensor must have shape [B,C,H,W]")
    kernel_size = int(kernel_size)
    if kernel_size <= 1:
        return tensor
    left = (kernel_size - 1) // 2
    right = kernel_size // 2
    top = (kernel_size - 1) // 2
    bottom = kernel_size // 2
    padded = F.pad(tensor, (left, right, top, bottom), mode="replicate")
    # Integral-image evaluation keeps the paper's default k=600 practical.
    integral = torch.cumsum(torch.cumsum(padded, dim=-2), dim=-1)
    integral = F.pad(integral, (1, 0, 1, 0))
    window_sum = (
        integral[..., kernel_size:, kernel_size:]
        - integral[..., :-kernel_size, kernel_size:]
        - integral[..., kernel_size:, :-kernel_size]
        + integral[..., :-kernel_size, :-kernel_size]
    )
    return window_sum / float(kernel_size * kernel_size)


def flow_magnitude(flow: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError("flow must have shape [B,2,H,W]")
    return torch.sqrt(torch.sum(flow * flow, dim=1, keepdim=True) + eps)


def flow_spatial_gradient(flow: torch.Tensor) -> torch.Tensor:
    """Return a one-channel robust magnitude of horizontal/vertical flow jumps."""
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError("flow must have shape [B,2,H,W]")
    dx = torch.zeros_like(flow)
    dy = torch.zeros_like(flow)
    dx[:, :, :, 1:] = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    dy[:, :, 1:, :] = flow[:, :, 1:, :] - flow[:, :, :-1, :]
    return torch.sqrt(torch.sum(dx * dx + dy * dy, dim=1, keepdim=True).clamp_min(0.0))


def finite_flow(flow: torch.Tensor, replacement: float = 0.0) -> torch.Tensor:
    return torch.nan_to_num(flow, nan=replacement, posinf=replacement, neginf=replacement)


def load_flow(path: str, reference: torch.Tensor) -> torch.Tensor:
    """Load .npy/.pt flow, normalize layout, resize, and move to reference."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError("Precomputed flow not found: %s" % source)
    if source.suffix.lower() == ".npy":
        value = torch.from_numpy(np.load(str(source)))
    else:
        payload = torch.load(str(source), map_location="cpu")
        value = payload.get("flow", payload) if isinstance(payload, dict) else payload
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    if value.ndim == 3:
        if value.shape[0] == 2:
            value = value.unsqueeze(0)
        elif value.shape[-1] == 2:
            value = value.permute(2, 0, 1).unsqueeze(0)
    if value.ndim != 4 or value.shape[1] != 2:
        raise ValueError("Flow must have shape [B,2,H,W] or [H,W,2]")
    value = value.to(device=reference.device, dtype=torch.float32)
    if value.shape[0] == 1 and reference.shape[0] > 1:
        value = value.expand(reference.shape[0], -1, -1, -1)
    if value.shape[0] != reference.shape[0]:
        raise ValueError("Flow batch does not match reference image batch")
    if value.shape[-2:] != reference.shape[-2:]:
        value = resize_flow(value, reference.shape[-2:])
    if not torch.isfinite(value).all():
        raise ValueError("Precomputed flow contains NaN or Inf")
    return value
