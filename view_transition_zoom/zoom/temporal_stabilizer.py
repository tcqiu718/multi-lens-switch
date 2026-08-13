"""Temporal smoothing for extra deformation and masks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import torch

from utils.warp import backward_warp


class TemporalFlowRefiner(ABC):
    """Reserved interface for future learned or motion-guided refiners."""

    @abstractmethod
    def __call__(self, previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TemporalStabilizer:
    """Stateful EMA with optional previous-to-current backward propagation flow."""

    def __init__(self, ema: float = 0.82, enabled: bool = True) -> None:
        if not 0.0 <= ema < 1.0:
            raise ValueError("ema must be in [0,1)")
        self.ema = float(ema)
        self.enabled = bool(enabled)
        self.previous_delta = None
        self.previous_occlusion = None
        self.previous_overlap = None

    def reset(self) -> None:
        self.previous_delta = None
        self.previous_occlusion = None
        self.previous_overlap = None

    def _smooth(
        self,
        current: torch.Tensor,
        previous: Optional[torch.Tensor],
        propagation_flow: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if not self.enabled or previous is None or previous.shape != current.shape:
            return current
        propagated = previous
        if propagation_flow is not None:
            propagated = backward_warp(previous, propagation_flow)
        return self.ema * propagated + (1.0 - self.ema) * current

    def smooth_delta(self, current: torch.Tensor, propagation_flow: Optional[torch.Tensor] = None) -> torch.Tensor:
        output = self._smooth(current, self.previous_delta, propagation_flow)
        self.previous_delta = output.detach()
        return output

    def smooth_masks(
        self,
        occlusion: torch.Tensor,
        overlap: torch.Tensor,
        propagation_flow: Optional[torch.Tensor] = None,
    ) -> tuple:
        smooth_occ = self._smooth(occlusion, self.previous_occlusion, propagation_flow).clamp(0.0, 1.0)
        smooth_overlap = self._smooth(overlap, self.previous_overlap, propagation_flow).clamp(0.0, 1.0)
        self.previous_occlusion = smooth_occ.detach()
        self.previous_overlap = smooth_overlap.detach()
        return smooth_occ, smooth_overlap


__all__ = ["TemporalFlowRefiner", "TemporalStabilizer"]

