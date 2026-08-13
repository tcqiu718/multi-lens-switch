from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import yaml


def deep_merge(base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge dictionaries without mutating either input."""
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _set_dotted(config: Dict[str, Any], key: str, value: Any) -> None:
    cursor = config
    parts = key.split(".")
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            child = {}
            cursor[part] = child
        cursor = child
    cursor[parts[-1]] = value


def load_config(path: str, overrides: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping: %s" % config_path)

    for item in overrides or []:
        if "=" not in item:
            raise ValueError("Override must use key=value syntax: %s" % item)
        key, raw_value = item.split("=", 1)
        _set_dotted(config, key.strip(), yaml.safe_load(raw_value))
    config["_config_path"] = str(config_path)
    return config


def resolve_path(config: Dict[str, Any], value: Optional[str]) -> Optional[Path]:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    config_path = Path(config.get("_config_path", Path.cwd()))
    base = config_path.parent if config_path.suffix else config_path
    return (base / path).resolve()


def resolve_device(requested: str = "auto") -> torch.device:
    requested = str(requested).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return torch.device(requested)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

