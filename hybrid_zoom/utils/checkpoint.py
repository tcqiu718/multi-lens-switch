"""Checkpoint save/load helpers with optimizer and AMP resume support."""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import torch
from torch import nn

PathLike = Union[str, Path]


def save_checkpoint(
    path: PathLike,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    epoch: int = 0,
    best_metric: Optional[float] = None,
    scaler: Optional[Any] = None,
    config: Optional[Mapping[str, Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    """Atomically save model and optional training state."""

    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    state: Dict[str, Any] = {"model": model.state_dict(), "epoch": int(epoch)}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    if best_metric is not None:
        state["best_metric"] = float(best_metric)
    if config is not None:
        state["config"] = dict(config)
    if extra is not None:
        state["extra"] = dict(extra)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=destination.name + ".", suffix=".tmp", dir=str(destination.parent)
    )
    os.close(descriptor)
    try:
        torch.save(state, temporary_name)
        os.replace(temporary_name, destination)
    except Exception:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
        raise
    return destination


def _model_state(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("model", "model_state_dict", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value
    # Also support a plain state_dict checkpoint.
    if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return checkpoint
    raise KeyError("Checkpoint does not contain model parameters")


def load_checkpoint(
    path: PathLike,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    map_location: Optional[Union[str, torch.device]] = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """Load model/training state and return the full checkpoint dictionary."""

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError("Checkpoint does not exist: {}".format(source))
    checkpoint = torch.load(source, map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must contain a mapping")
    checkpoint_dict = dict(checkpoint)
    model.load_state_dict(_model_state(checkpoint_dict), strict=strict)
    if optimizer is not None and "optimizer" in checkpoint_dict:
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint_dict:
        scheduler.load_state_dict(checkpoint_dict["scheduler"])
    if scaler is not None and "scaler" in checkpoint_dict:
        scaler.load_state_dict(checkpoint_dict["scaler"])
    return checkpoint_dict


def resume_from_checkpoint(
    path: PathLike,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
    map_location: Optional[Union[str, torch.device]] = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """Alias emphasizing restoration of all available training state."""

    return load_checkpoint(
        path,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        map_location=map_location,
        strict=strict,
    )


__all__ = ["load_checkpoint", "resume_from_checkpoint", "save_checkpoint"]
