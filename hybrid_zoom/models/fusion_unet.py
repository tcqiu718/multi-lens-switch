"""Lightweight luminance-fusion U-Net."""

from __future__ import annotations

from typing import Callable, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _activation_factory(name: str) -> Callable[[], nn.Module]:
    normalized = name.lower().replace("-", "_")
    if normalized in {"leaky_relu", "leakyrelu"}:
        return lambda: nn.LeakyReLU(negative_slope=0.1, inplace=True)
    if normalized == "relu":
        return lambda: nn.ReLU(inplace=True)
    raise ValueError("activation must be 'leaky_relu' or 'relu', got {!r}".format(name))


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with a configurable pointwise activation."""

    def __init__(self, in_channels: int, out_channels: int, activation: str) -> None:
        super().__init__()
        make_activation = _activation_factory(activation)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            make_activation(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            make_activation(),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.block(inputs)


class UpBlock(nn.Module):
    """Bilinear upsampling, channel projection, skip concatenation and fusion."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        activation: str,
    ) -> None:
        super().__init__()
        self.project = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.fuse = ConvBlock(out_channels + skip_channels, out_channels, activation)

    def forward(self, inputs: Tensor, skip: Tensor) -> Tensor:
        inputs = F.interpolate(inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        inputs = self.project(inputs)
        return self.fuse(torch.cat((inputs, skip), dim=1))


class FusionUNet(nn.Module):
    """A compact five-level U-Net for Wide/Tele luminance fusion.

    The expected input is ``[Y_wide, Y_warped_tele, M_occ]`` (three channels),
    optionally followed by ``M_reject`` (four channels).  Mask values follow the
    project convention: zero is reliable and one is unreliable.

    ``mode='residual'`` predicts a residual and returns
    ``clamp(Y_wide + residual, 0, 1)``.  ``mode='direct'`` predicts luminance
    directly through a sigmoid.  Spatial dimensions are padded to a multiple of
    16 for four encoder downsamplings and cropped back exactly on return.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        mode: str = "residual",
        activation: str = "leaky_relu",
        residual_input_index: int = 0,
    ) -> None:
        super().__init__()
        if in_channels not in {3, 4}:
            raise ValueError("FusionUNet expects 3 or 4 input channels, got {}".format(in_channels))
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")
        normalized_mode = mode.lower()
        if normalized_mode not in {"residual", "direct"}:
            raise ValueError("mode must be 'residual' or 'direct', got {!r}".format(mode))
        if not 0 <= residual_input_index < in_channels:
            raise ValueError("residual_input_index is outside the input channel range")
        _activation_factory(activation)  # Validate eagerly.

        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)
        self.mode = normalized_mode
        self.residual_input_index = int(residual_input_index)

        channels = (
            self.base_channels,
            self.base_channels * 2,
            self.base_channels * 4,
            self.base_channels * 8,
            self.base_channels * 8,
        )
        self.encoder1 = ConvBlock(in_channels, channels[0], activation)
        self.encoder2 = ConvBlock(channels[0], channels[1], activation)
        self.encoder3 = ConvBlock(channels[1], channels[2], activation)
        self.encoder4 = ConvBlock(channels[2], channels[3], activation)
        self.bottleneck = ConvBlock(channels[3], channels[4], activation)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.decoder4 = UpBlock(channels[4], channels[3], channels[3], activation)
        self.decoder3 = UpBlock(channels[3], channels[2], channels[2], activation)
        self.decoder2 = UpBlock(channels[2], channels[1], channels[1], activation)
        self.decoder1 = UpBlock(channels[1], channels[0], channels[0], activation)
        self.output_conv = nn.Conv2d(channels[0], 1, kernel_size=3, padding=1)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, a=0.1, mode="fan_out", nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # With no trained fusion checkpoint, residual mode is a safe identity on Y.
        if self.mode == "residual":
            nn.init.zeros_(self.output_conv.weight)
            if self.output_conv.bias is not None:
                nn.init.zeros_(self.output_conv.bias)

    @staticmethod
    def _pad_to_multiple(inputs: Tensor, multiple: int = 16) -> Tuple[Tensor, Tuple[int, int]]:
        height, width = inputs.shape[-2:]
        pad_height = (multiple - height % multiple) % multiple
        pad_width = (multiple - width % multiple) % multiple
        if pad_height == 0 and pad_width == 0:
            return inputs, (height, width)
        padded = F.pad(inputs, (0, pad_width, 0, pad_height), mode="replicate")
        return padded, (height, width)

    def forward(self, inputs: Tensor) -> Tensor:
        """Fuse a ``[B, 3|4, H, W]`` tensor and return ``[B, 1, H, W]``."""
        if not isinstance(inputs, Tensor):
            raise TypeError("FusionUNet input must be a torch.Tensor")
        if inputs.ndim != 4:
            raise ValueError("FusionUNet input must be [B, C, H, W]")
        if inputs.shape[1] != self.in_channels:
            raise ValueError(
                "FusionUNet was built for {} channels but received {}".format(
                    self.in_channels, inputs.shape[1]
                )
            )
        if not inputs.is_floating_point():
            raise TypeError("FusionUNet input must be floating point")
        if inputs.shape[-2] <= 0 or inputs.shape[-1] <= 0:
            raise ValueError("FusionUNet input must have non-empty spatial dimensions")

        reference = inputs[:, self.residual_input_index : self.residual_input_index + 1]
        padded, original_size = self._pad_to_multiple(inputs)

        # Encoder feature shapes: H, H/2, H/4, H/8, H/16 (and likewise W).
        level1 = self.encoder1(padded)
        level2 = self.encoder2(self.pool(level1))
        level3 = self.encoder3(self.pool(level2))
        level4 = self.encoder4(self.pool(level3))
        level5 = self.bottleneck(self.pool(level4))

        decoded = self.decoder4(level5, level4)
        decoded = self.decoder3(decoded, level3)
        decoded = self.decoder2(decoded, level2)
        decoded = self.decoder1(decoded, level1)
        prediction = self.output_conv(decoded)

        height, width = original_size
        prediction = prediction[..., :height, :width]
        if self.mode == "residual":
            return (reference + prediction).clamp(0.0, 1.0)
        return torch.sigmoid(prediction)


__all__ = ["FusionUNet"]
