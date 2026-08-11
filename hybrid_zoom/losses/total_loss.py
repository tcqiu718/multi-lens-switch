"""Weighted training objective for hybrid-zoom reconstruction."""

from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .brightness_loss import BrightnessLoss
from .contextual_loss import ContextualLoss
from .perceptual_loss import VGG19FeatureExtractor


class TotalLoss(nn.Module):
    """Combine VGG perceptual, Contextual, and brightness losses.

    The first argument may be the ``loss`` mapping loaded from ``config.yaml``;
    explicit keyword arguments override mapping values.  ``forward`` returns a
    dictionary with differentiable scalar tensors named ``total``, ``vgg``,
    ``contextual`` and ``brightness``.
    """

    def __init__(self, config: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__()
        values: Dict[str, Any] = dict(config or {})
        values.update(kwargs)

        self.vgg_weight = float(values.pop("vgg_weight", values.pop("lambda_vgg", 1.0)))
        self.contextual_weight = float(
            values.pop("contextual_weight", values.pop("lambda_contextual", 0.05))
        )
        self.brightness_weight = float(
            values.pop("brightness_weight", values.pop("lambda_brightness", 1.0))
        )
        for name, weight in (
            ("vgg_weight", self.vgg_weight),
            ("contextual_weight", self.contextual_weight),
            ("brightness_weight", self.brightness_weight),
        ):
            if weight < 0:
                raise ValueError("{} must be non-negative".format(name))
        if self.vgg_weight + self.contextual_weight + self.brightness_weight <= 0:
            raise ValueError("At least one loss weight must be positive")

        vgg_layers = tuple(int(layer) for layer in values.pop("vgg_layers", (3, 8, 17, 26)))
        if not vgg_layers:
            raise ValueError("vgg_layers cannot be empty")
        self.contextual_layer = int(values.pop("contextual_layer", 17))
        vgg_weights = values.pop("vgg_weights", values.pop("weights", None))
        # Never silently train against random frozen VGG features when the
        # requested pretrained checkpoint is unavailable.  Offline smoke tests
        # can explicitly set vgg_weights=null, or opt into fallback themselves.
        offline_fallback = bool(values.pop("offline_fallback", False))

        required_layers = set(vgg_layers)
        if self.contextual_weight > 0:
            required_layers.add(self.contextual_layer)
        self.vgg_layers = tuple(vgg_layers)
        self.extractor: Optional[VGG19FeatureExtractor]
        if self.vgg_weight > 0 or self.contextual_weight > 0:
            self.extractor = VGG19FeatureExtractor(
                layers=sorted(required_layers),
                weights=vgg_weights,
                offline_fallback=offline_fallback,
            )
        else:
            self.extractor = None

        self.contextual = ContextualLoss(
            bandwidth=float(values.pop("contextual_bandwidth", values.pop("bandwidth", 0.5))),
            max_samples=values.pop("contextual_max_samples", values.pop("max_samples", 1024)),
            eps=float(values.pop("contextual_eps", 1e-5)),
        )
        self.brightness = BrightnessLoss(
            sigma=float(values.pop("brightness_sigma", 10.0)),
            kernel_size=values.pop("brightness_kernel_size", None),
        )
        if values:
            raise TypeError("Unknown TotalLoss options: {}".format(sorted(values)))

    def forward(self, prediction: Tensor, target: Tensor) -> Dict[str, Tensor]:
        if prediction.shape != target.shape:
            raise ValueError(
                "prediction and target shapes differ: {} versus {}".format(
                    tuple(prediction.shape), tuple(target.shape)
                )
            )
        if prediction.ndim != 4 or prediction.shape[1] != 3:
            raise ValueError("TotalLoss expects RGB BCHW tensors")

        zero = prediction.new_zeros(())
        vgg_loss = zero
        contextual_loss = zero
        if self.extractor is not None:
            prediction_features = self.extractor(prediction)
            with torch.no_grad():
                target_features = self.extractor(target)
            if self.vgg_weight > 0:
                vgg_loss = torch.stack(
                    [
                        F.l1_loss(prediction_features[layer], target_features[layer])
                        for layer in self.vgg_layers
                    ]
                ).sum()
            if self.contextual_weight > 0:
                contextual_loss = self.contextual(
                    prediction_features[self.contextual_layer],
                    target_features[self.contextual_layer],
                )

        brightness_loss = (
            self.brightness(prediction, target) if self.brightness_weight > 0 else zero
        )
        total = (
            self.vgg_weight * vgg_loss
            + self.contextual_weight * contextual_loss
            + self.brightness_weight * brightness_loss
        )
        return {
            "total": total,
            "vgg": vgg_loss,
            "contextual": contextual_loss,
            "brightness": brightness_loss,
        }


__all__ = ["TotalLoss"]
