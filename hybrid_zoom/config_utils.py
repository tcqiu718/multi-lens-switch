"""Configuration and reproducibility helpers shared by command-line programs."""

from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import yaml


Config = Dict[str, Any]


def deep_update(base: MutableMapping[str, Any], updates: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Recursively merge ``updates`` into ``base`` and return ``base``."""
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_config(path: Union[str, Path], overrides: Optional[Mapping[str, Any]] = None) -> Config:
    """Load a YAML mapping and optionally merge programmatic overrides."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Top-level YAML value must be a mapping: {config_path}")
    config: Config = copy.deepcopy(loaded)
    if overrides:
        deep_update(config, overrides)
    config["_config_path"] = str(config_path)
    return config


def set_by_dotted_key(config: MutableMapping[str, Any], key: str, value: Any) -> None:
    """Set a nested value such as ``training.batch_size``."""
    parts = key.split(".")
    if not parts or any(not part for part in parts):
        raise ValueError(f"Invalid dotted configuration key: {key!r}")
    current: MutableMapping[str, Any] = config
    for part in parts[:-1]:
        child = current.setdefault(part, {})
        if not isinstance(child, MutableMapping):
            raise ValueError(f"Cannot descend through non-mapping key: {part}")
        current = child
    current[parts[-1]] = value


def apply_cli_overrides(config: Config, expressions: Sequence[str]) -> Config:
    """Apply ``key=value`` expressions, parsing values with YAML semantics."""
    for expression in expressions:
        if "=" not in expression:
            raise ValueError(f"Override must have key=value form: {expression!r}")
        key, raw_value = expression.split("=", 1)
        set_by_dotted_key(config, key.strip(), yaml.safe_load(raw_value))
    return config


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """Resolve a requested device, falling back from unavailable CUDA to CPU."""
    normalized = requested.strip().lower()
    if normalized.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA was requested but is unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def resolve_wide_crop_size(image_config: Mapping[str, Any]) -> Optional[Tuple[int, int]]:
    """Resolve an explicit pre-resize Wide centre crop from image config.

    A bare ``center_crop: true`` is rejected because inferring the output size
    would make it a no-op for pre-resized Dataset tensors and ambiguous for raw
    demo images.  Supply ``wide_crop_size: [H, W]``, ``center_crop: [H, W]``,
    or both ``crop_height`` and ``crop_width``.
    """
    explicit = image_config.get("wide_crop_size")
    setting = image_config.get("center_crop", False)
    candidate: Any = explicit
    if candidate is None:
        if not setting:
            return None
        if not isinstance(setting, bool):
            candidate = setting
        else:
            crop_height = image_config.get("crop_height")
            crop_width = image_config.get("crop_width")
            if crop_height is None or crop_width is None:
                raise ValueError(
                    "image.center_crop=true requires image.wide_crop_size or both "
                    "image.crop_height and image.crop_width"
                )
            candidate = (crop_height, crop_width)
    if isinstance(candidate, (str, bytes)) or not isinstance(candidate, Sequence):
        raise TypeError("image wide crop size must be a (height, width) sequence")
    if len(candidate) != 2:
        raise ValueError("image wide crop size must contain exactly two values")
    crop_size = (int(candidate[0]), int(candidate[1]))
    if crop_size[0] <= 0 or crop_size[1] <= 0:
        raise ValueError(f"image wide crop dimensions must be positive, got {crop_size}")
    return crop_size
