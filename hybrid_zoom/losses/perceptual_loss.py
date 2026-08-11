"""Frozen VGG19 feature extraction and perceptual loss."""

import warnings
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class VGG19FeatureExtractor(nn.Module):
    """Extract selected activations from a frozen torchvision VGG19.

    ``weights`` can be ``None`` for a fully offline/randomly initialized model,
    ``"default"`` for torchvision's official default weights, an official
    ``VGG19_Weights`` enum value, or an enum member name.  If requested weights
    cannot be obtained (for example on an offline machine without a cached
    checkpoint).  Requested pretrained weights fail fast by default if they
    cannot be loaded; offline smoke tests must explicitly pass ``weights=None``.
    """

    def __init__(
        self,
        layers: Sequence[int] = (3, 8, 17, 26),
        weights: Optional[Any] = None,
        normalize_input: bool = True,
        offline_fallback: bool = False,
        progress: bool = False,
    ) -> None:
        super().__init__()
        unique_layers = sorted({int(layer) for layer in layers})
        if not unique_layers or unique_layers[0] < 0:
            raise ValueError("layers must contain non-negative feature indices")

        # Import lazily so unrelated utilities do not require a working
        # torchvision installation merely to import the losses package.
        try:
            from torchvision.models import VGG19_Weights, vgg19
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError(
                "VGG19FeatureExtractor requires a compatible torchvision installation"
            ) from exc

        resolved_weights = self._resolve_weights(weights, VGG19_Weights)
        try:
            backbone = vgg19(weights=resolved_weights, progress=progress)
        except Exception as exc:  # Download/cache failures differ by platform.
            if resolved_weights is None or not offline_fallback:
                raise
            warnings.warn(
                "Could not load requested VGG19 weights ({}); falling back to "
                "random frozen weights. Pass weights=None to explicitly select "
                "offline initialization.".format(exc),
                RuntimeWarning,
            )
            backbone = vgg19(weights=None, progress=False)

        if unique_layers[-1] >= len(backbone.features):
            raise ValueError(
                "VGG feature index {} is out of range [0, {})".format(
                    unique_layers[-1], len(backbone.features)
                )
            )
        self.layers = tuple(unique_layers)
        self.features = backbone.features[: unique_layers[-1] + 1]
        self.normalize_input = bool(normalize_input)
        self.register_buffer(
            "mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1), persistent=False
        )
        self.features.requires_grad_(False)
        self.features.eval()

    @staticmethod
    def _resolve_weights(weights: Optional[Any], enum_class: Any) -> Optional[Any]:
        if weights is None or weights is False:
            return None
        if weights is True:
            return enum_class.DEFAULT
        if isinstance(weights, str):
            value = weights.strip().lower()
            if value in {"", "none", "null", "false", "random"}:
                return None
            if value in {"default", "true", "imagenet", "pretrained"}:
                return enum_class.DEFAULT
            try:
                return enum_class[weights.strip().upper()]
            except KeyError as exc:
                valid = ", ".join(member.name for member in enum_class)
                raise ValueError(
                    "Unknown VGG19 weights {!r}; expected default/none or one of {}".format(
                        weights, valid
                    )
                ) from exc
        return weights

    def train(self, mode: bool = True) -> "VGG19FeatureExtractor":
        """Keep the frozen feature network in evaluation mode."""

        super().train(False)
        return self

    def forward(self, image: Tensor) -> Dict[int, Tensor]:
        """Return a mapping from requested torchvision layer index to feature."""

        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("VGG19 expects RGB BCHW input; got {}".format(tuple(image.shape)))
        feature = image
        if self.normalize_input:
            feature = (feature - self.mean.to(dtype=feature.dtype)) / self.std.to(
                dtype=feature.dtype
            )

        requested = set(self.layers)
        outputs: Dict[int, Tensor] = {}
        for index, layer in enumerate(self.features):
            feature = layer(feature)
            if index in requested:
                outputs[index] = feature
        return outputs


class VGGPerceptualLoss(nn.Module):
    """Mean L1 distance between matching frozen VGG19 activations."""

    def __init__(
        self,
        layers: Sequence[int] = (3, 8, 17, 26),
        weights: Optional[Any] = None,
        layer_weights: Optional[Mapping[int, float]] = None,
        extractor: Optional[VGG19FeatureExtractor] = None,
        offline_fallback: bool = False,
    ) -> None:
        super().__init__()
        self.extractor = extractor or VGG19FeatureExtractor(
            layers=layers,
            weights=weights,
            offline_fallback=offline_fallback,
        )
        if layer_weights is None:
            self.layer_weights = {layer: 1.0 for layer in self.extractor.layers}
        else:
            self.layer_weights = {
                int(layer): float(weight) for layer, weight in layer_weights.items()
            }
            unknown = set(self.layer_weights) - set(self.extractor.layers)
            if unknown:
                raise ValueError("layer_weights contains unextracted layers: {}".format(unknown))

    @staticmethod
    def from_features(
        prediction: Mapping[int, Tensor],
        target: Mapping[int, Tensor],
        layer_weights: Mapping[int, float],
    ) -> Tensor:
        """Compute perceptual loss from already extracted feature mappings."""

        losses = [
            float(weight) * F.l1_loss(prediction[layer], target[layer])
            for layer, weight in layer_weights.items()
        ]
        if not losses:
            raise ValueError("At least one perceptual layer is required")
        return torch.stack(losses).sum()

    def forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        prediction_features = self.extractor(prediction)
        with torch.no_grad():
            target_features = self.extractor(target)
        return self.from_features(prediction_features, target_features, self.layer_weights)


# Concise alias used by some training scripts.
PerceptualLoss = VGGPerceptualLoss


__all__ = ["PerceptualLoss", "VGG19FeatureExtractor", "VGGPerceptualLoss"]
