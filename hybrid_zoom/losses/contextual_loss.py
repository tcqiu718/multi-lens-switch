"""Numerically stable Contextual Loss for image/feature comparison."""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ContextualLoss(nn.Module):
    """Compute the Contextual (CX) similarity loss between feature maps.

    The implementation follows the core CX construction rather than replacing
    it with a pointwise loss:

    1. center both feature sets using the target feature mean;
    2. L2-normalize feature vectors and form all pairwise cosine distances;
    3. divide by each query's nearest-target distance;
    4. turn relative distances into contextual similarities using softmax;
    5. maximize over candidate matches and minimize ``-log(mean(CX))``.

    Pairwise matching is quadratic in the number of spatial samples.  To bound
    memory, maps larger than ``max_samples`` are jointly downsampled with
    adaptive average pooling before matching.
    """

    def __init__(
        self,
        bandwidth: float = 0.5,
        max_samples: Optional[int] = 1024,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        if bandwidth <= 0:
            raise ValueError("bandwidth must be positive")
        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be positive or None")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.bandwidth = float(bandwidth)
        self.max_samples = int(max_samples) if max_samples is not None else None
        self.eps = float(eps)

    @staticmethod
    def _pooled_size(height: int, width: int, max_samples: int) -> Tuple[int, int]:
        if height * width <= max_samples:
            return height, width
        scale = math.sqrt(float(max_samples) / float(height * width))
        pooled_h = max(1, min(height, int(math.floor(height * scale))))
        pooled_w = max(1, min(width, int(math.floor(width * scale))))
        while pooled_h * pooled_w > max_samples:
            if pooled_w >= pooled_h and pooled_w > 1:
                pooled_w -= 1
            elif pooled_h > 1:
                pooled_h -= 1
            else:
                break
        return pooled_h, pooled_w

    def _limit_samples(self, prediction: Tensor, target: Tensor) -> Tuple[Tensor, Tensor]:
        if self.max_samples is None:
            return prediction, target
        pooled_size = self._pooled_size(prediction.shape[-2], prediction.shape[-1], self.max_samples)
        if pooled_size != prediction.shape[-2:]:
            prediction = F.adaptive_avg_pool2d(prediction, pooled_size)
            target = F.adaptive_avg_pool2d(target, pooled_size)
        return prediction, target

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Return scalar CX loss for equally batched BCHW feature maps."""

        if prediction.ndim != 4 or target.ndim != 4:
            raise ValueError("ContextualLoss expects BCHW feature maps")
        if prediction.shape[:2] != target.shape[:2]:
            raise ValueError(
                "Batch/channel dimensions must match, got {} and {}".format(
                    tuple(prediction.shape), tuple(target.shape)
                )
            )
        if prediction.shape[-2:] != target.shape[-2:]:
            target = F.interpolate(
                target, size=prediction.shape[-2:], mode="bilinear", align_corners=False
            )
        prediction, target = self._limit_samples(prediction, target)

        # Pairwise similarity is significantly more stable in float32 under AMP.
        compute_dtype = (
            torch.float32
            if prediction.dtype in {torch.float16, torch.bfloat16}
            else prediction.dtype
        )
        x = prediction.to(dtype=compute_dtype)
        y = target.to(dtype=compute_dtype)

        target_mean = y.mean(dim=(2, 3), keepdim=True)
        x = x - target_mean
        y = y - target_mean
        x = F.normalize(x.flatten(2), p=2, dim=1, eps=self.eps)
        y = F.normalize(y.flatten(2), p=2, dim=1, eps=self.eps)

        # [B, Nx, Ny], each row compares one prediction vector with all targets.
        cosine_similarity = torch.bmm(x.transpose(1, 2), y).clamp(-1.0, 1.0)
        distance = (1.0 - cosine_similarity).clamp_min(0.0)
        nearest = distance.amin(dim=2, keepdim=True)
        relative_distance = distance / (nearest + self.eps)

        logits = (1.0 - relative_distance) / self.bandwidth
        contextual_similarity = torch.softmax(logits, dim=2)
        # CX(X, Y) = (1 / N) * sum_j max_i CX_ij: every target feature j
        # must be covered by at least one prediction feature i.  Maximizing over
        # dim=2 instead would ask only whether each prediction can find some
        # target and would under-penalize duplicated/mode-collapsed features.
        best_similarity = contextual_similarity.amax(dim=1).mean(dim=1)
        return -torch.log(best_similarity.clamp_min(self.eps)).mean()


__all__ = ["ContextualLoss"]
