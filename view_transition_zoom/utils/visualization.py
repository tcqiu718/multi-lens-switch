"""Flow visualization and labeled comparison frames."""

from __future__ import annotations

from typing import Iterable, List

import cv2
import numpy as np
import torch

from utils.image_io import tensor_to_rgb8


def flow_to_color(flow: torch.Tensor, max_magnitude: float = None) -> torch.Tensor:
    """Convert dx/dy flow to an HSV-wheel RGB tensor in [0,1]."""
    if flow.ndim == 3:
        flow = flow.unsqueeze(0)
    if flow.ndim != 4 or flow.shape[1] != 2:
        raise ValueError("flow must have shape [B,2,H,W]")
    outputs = []
    for item in flow.detach().float().cpu():
        dx, dy = item[0].numpy(), item[1].numpy()
        magnitude, angle = cv2.cartToPolar(dx, dy, angleInDegrees=True)
        scale = float(max_magnitude) if max_magnitude is not None else float(np.percentile(magnitude, 99.0))
        scale = max(scale, 1.0e-6)
        hsv = np.zeros((dx.shape[0], dx.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = np.round(angle / 2.0).astype(np.uint8)
        hsv[..., 1] = 255
        hsv[..., 2] = np.round(np.clip(magnitude / scale, 0.0, 1.0) * 255.0).astype(np.uint8)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        outputs.append(torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0)
    return torch.stack(outputs).to(device=flow.device, dtype=flow.dtype)


def comparison_frame(images: Iterable[torch.Tensor], labels: Iterable[str]) -> np.ndarray:
    arrays: List[np.ndarray] = [np.ascontiguousarray(tensor_to_rgb8(image)).copy() for image in images]
    names = list(labels)
    if len(arrays) != len(names) or not arrays:
        raise ValueError("images and labels must be non-empty and have equal lengths")
    height = arrays[0].shape[0]
    if any(item.shape[0] != height for item in arrays):
        raise ValueError("comparison images must have equal heights")
    for image, label in zip(arrays, names):
        cv2.rectangle(image, (8, 8), (18 + 12 * len(label), 38), (0, 0, 0), -1)
        cv2.putText(image, label, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate(arrays, axis=1)


__all__ = ["comparison_frame", "flow_to_color"]
