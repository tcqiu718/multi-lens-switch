"""Train the luminance Fusion UNet while keeping RAFT frozen by default."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ in {None, ""}:  # Make direct ``python hybrid_zoom/train.py`` package-safe.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hybrid_zoom.config_utils import (
    apply_cli_overrides,
    load_config,
    resolve_device,
    resolve_wide_crop_size,
    seed_everything,
)
from hybrid_zoom.datasets import HybridZoomDataset
from hybrid_zoom.losses import TotalLoss
from hybrid_zoom.models import HybridZoomModel
from hybrid_zoom.utils import (
    create_summary_writer,
    load_checkpoint,
    psnr,
    save_checkpoint,
    ssim,
)


def _make_writer(log_dir: Path) -> Any:
    return create_summary_writer(log_dir)


def _autocast(enabled: bool) -> Any:
    """Use the modern AMP API with a PyTorch 2.1-compatible fallback."""
    if not enabled:
        return contextlib.nullcontext()
    amp_namespace = getattr(torch, "amp", None)
    if amp_namespace is not None and hasattr(amp_namespace, "autocast"):
        return amp_namespace.autocast("cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)  # pragma: no cover - old PyTorch


def _make_grad_scaler(enabled: bool) -> Any:
    amp_namespace = getattr(torch, "amp", None)
    scaler_type = getattr(amp_namespace, "GradScaler", None)
    if scaler_type is not None:
        try:
            return scaler_type("cuda", enabled=enabled)
        except TypeError:  # pragma: no cover - transitional PyTorch signature.
            return scaler_type(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)  # pragma: no cover


def _image_size(config: Mapping[str, Any]) -> Tuple[int, int]:
    image_cfg = config.get("image", {})
    height = int(image_cfg.get("height", 512))
    width = int(image_cfg.get("width", 768))
    if height <= 0 or width <= 0:
        raise ValueError(f"image.height and image.width must be positive, got {(height, width)}")
    return height, width


def _as_float(value: Any) -> float:
    if isinstance(value, Tensor):
        return float(value.detach().mean().cpu())
    return float(value)


def _loss_dict(criterion: nn.Module, prediction: Tensor, target: Tensor) -> Dict[str, Tensor]:
    losses = criterion(prediction, target)
    if isinstance(losses, Tensor):
        return {"total": losses}
    if not isinstance(losses, Mapping) or "total" not in losses:
        raise TypeError("TotalLoss must return a tensor or a mapping containing the 'total' tensor")
    return dict(losses)


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def _set_frozen_flow_eval(model: nn.Module) -> None:
    """Keep a frozen flow backbone in evaluation mode after ``model.train()``."""
    for attribute in ("flow_estimator", "flow"):
        module = getattr(model, attribute, None)
        if isinstance(module, nn.Module) and not any(parameter.requires_grad for parameter in module.parameters()):
            module.eval()


def _log_images(writer: Any, prefix: str, batch: Mapping[str, Any], outputs: Mapping[str, Tensor], step: int) -> None:
    candidates: Dict[str, Optional[Tensor]] = {
        "wide": outputs.get("wide", batch.get("wide")),
        "tele": outputs.get("tele", batch.get("tele")),
        "warped_tele": outputs.get("warped_tele"),
        "gt": batch.get("gt"),
        "fusion": outputs.get("fusion_y"),
        "final": outputs.get("output"),
        "M_occ": outputs.get("occlusion_mask"),
        "M_reject": outputs.get("rejection_mask"),
        "M_blend": outputs.get("blend_mask"),
    }
    for name, tensor in candidates.items():
        if isinstance(tensor, Tensor) and tensor.ndim == 4:
            writer.add_images(f"{prefix}/{name}", tensor[:2].detach().float().clamp(0.0, 1.0), step)


def _save_training_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: Any,
    epoch: int,
    best_val_loss: float,
    config: Mapping[str, Any],
) -> None:
    save_checkpoint(
        path,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=epoch,
        best_metric=best_val_loss,
        config={key: value for key, value in config.items() if not key.startswith("_")},
        extra={"best_val_loss": best_val_loss},
    )


def _resume_training(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: Any,
    device: torch.device,
) -> Tuple[int, float]:
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
    checkpoint = load_checkpoint(
        path,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        map_location=device,
        strict=True,
    )
    extra = checkpoint.get("extra", {})
    extra_best = extra.get("best_val_loss", math.inf) if isinstance(extra, Mapping) else math.inf
    best = checkpoint.get("best_val_loss", checkpoint.get("best_metric", extra_best))
    return int(checkpoint.get("epoch", -1)) + 1, float(best)


def _checkpoint_contains_flow(path: Path) -> bool:
    """Return whether a checkpoint embeds the flow estimator state."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        return False
    state_dict = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    if not isinstance(state_dict, Mapping):
        return False
    return any(
        key.startswith("flow_estimator.") or ".flow_estimator." in key
        for key in state_dict.keys()
    )


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
) -> Tuple[Dict[str, float], Optional[Tuple[Mapping[str, Any], Mapping[str, Tensor]]]]:
    model.eval()
    totals = {"loss": 0.0, "psnr": 0.0, "ssim": 0.0}
    sample_count = 0
    preview = None
    progress = tqdm(loader, desc="validate", leave=False)
    for raw_batch in progress:
        batch = _move_batch(raw_batch, device)
        if "gt" not in batch:
            raise KeyError("Validation samples must contain 'gt'")
        batch_size = int(batch["wide"].shape[0])
        with _autocast(amp_enabled):
            outputs = model(batch["wide"], batch["tele"], preprocessed=True)
            losses = _loss_dict(criterion, outputs["output"], batch["gt"])
        totals["loss"] += _as_float(losses["total"]) * batch_size
        totals["psnr"] += _as_float(psnr(outputs["output"], batch["gt"])) * batch_size
        totals["ssim"] += _as_float(ssim(outputs["output"], batch["gt"])) * batch_size
        sample_count += batch_size
        if preview is None:
            preview = (batch, outputs)
        progress.set_postfix(loss=f"{_as_float(losses['total']):.4f}")
    if sample_count == 0:
        raise RuntimeError("Validation dataset is empty")
    return {key: value / sample_count for key, value in totals.items()}, preview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a YAML value; repeat for multiple values.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(load_config(args.config), args.overrides)
    if args.data_root is not None:
        config.setdefault("data", {})["root"] = str(args.data_root)
    if args.output_dir is not None:
        config.setdefault("training", {})["output_dir"] = str(args.output_dir)
    if args.resume is not None:
        config.setdefault("training", {})["resume"] = str(args.resume)

    seed_everything(int(config.get("seed", 42)))
    device = resolve_device(str(config.get("device", "cuda")))
    training_cfg = config.get("training", {})
    data_cfg = config.get("data", {})
    resume_value = training_cfg.get("resume")
    if resume_value:
        resume_candidate = Path(resume_value).expanduser().resolve()
        if not resume_candidate.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_candidate}")
        if _checkpoint_contains_flow(resume_candidate):
            # Our checkpoints contain RAFT.  Construct it without an external
            # download, then restore its exact weights from the checkpoint.
            config.setdefault("flow", {})["weights"] = None
    output_dir = Path(training_cfg.get("output_dir", "./runs/hybrid_zoom")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
        json.dump({k: v for k, v in config.items() if not k.startswith("_")}, handle, indent=2)

    image_size = _image_size(config)
    wide_crop_size = resolve_wide_crop_size(config.get("image", {}))
    data_root = Path(data_cfg.get("root", "./dataset")).expanduser()
    train_dataset = HybridZoomDataset(
        data_root,
        split=str(data_cfg.get("train_split", "train")),
        image_size=image_size,
        augment=bool(data_cfg.get("augment", True)),
        require_gt=True,
        keep_aspect_ratio=bool(config.get("image", {}).get("keep_aspect_ratio", False)),
        aspect_mode=str(config.get("image", {}).get("aspect_mode", "letterbox")),
        interpolation=str(config.get("image", {}).get("interpolation", "bilinear")),
        wide_crop_size=wide_crop_size,
        crop_gt_with_wide=bool(config.get("image", {}).get("crop_gt_with_wide", True)),
    )
    val_dataset = HybridZoomDataset(
        data_root,
        split=str(data_cfg.get("val_split", "val")),
        image_size=image_size,
        augment=False,
        require_gt=True,
        keep_aspect_ratio=bool(config.get("image", {}).get("keep_aspect_ratio", False)),
        aspect_mode=str(config.get("image", {}).get("aspect_mode", "letterbox")),
        interpolation=str(config.get("image", {}).get("interpolation", "bilinear")),
        wide_crop_size=wide_crop_size,
        crop_gt_with_wide=bool(config.get("image", {}).get("crop_gt_with_wide", True)),
    )
    batch_size = int(training_cfg.get("batch_size", 2))
    num_workers = int(training_cfg.get("num_workers", 4))
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, **loader_kwargs)

    model = HybridZoomModel(config).to(device)
    criterion = TotalLoss(config.get("loss", {})).to(device)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("The model has no trainable parameters; check fusion and flow freeze settings")
    optimizer = AdamW(
        trainable_parameters,
        lr=float(training_cfg.get("learning_rate", 1.0e-4)),
        weight_decay=float(training_cfg.get("weight_decay", 1.0e-4)),
    )
    epochs = int(training_cfg.get("epochs", 50))
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    amp_enabled = bool(training_cfg.get("amp", True)) and device.type == "cuda"
    scaler = _make_grad_scaler(amp_enabled)
    writer = _make_writer(output_dir / "tensorboard")

    start_epoch = 0
    best_val_loss = math.inf
    resume_path = resume_value
    if resume_path:
        start_epoch, best_val_loss = _resume_training(
            Path(resume_path).expanduser().resolve(), model, optimizer, scheduler, scaler, device
        )
        print(f"Resumed at epoch {start_epoch}; best validation loss={best_val_loss:.6f}")

    global_step = start_epoch * max(1, len(train_loader))
    validate_every = max(1, int(training_cfg.get("validate_every", 1)))
    save_every = max(1, int(training_cfg.get("save_every", 1)))
    grad_clip = float(training_cfg.get("grad_clip_norm", 0.0))
    started_at = time.time()
    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            _set_frozen_flow_eval(model)
            running_loss = 0.0
            sample_count = 0
            progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{epochs}")
            for raw_batch in progress:
                batch = _move_batch(raw_batch, device)
                if "gt" not in batch:
                    raise KeyError("Training samples must contain 'gt'")
                optimizer.zero_grad(set_to_none=True)
                with _autocast(amp_enabled):
                    outputs = model(batch["wide"], batch["tele"], preprocessed=True)
                    losses = _loss_dict(criterion, outputs["output"], batch["gt"])
                scaler.scale(losses["total"]).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_parameters, grad_clip)
                scaler.step(optimizer)
                scaler.update()

                current_batch_size = int(batch["wide"].shape[0])
                current_loss = _as_float(losses["total"])
                running_loss += current_loss * current_batch_size
                sample_count += current_batch_size
                writer.add_scalar("train/loss", current_loss, global_step)
                for loss_name, loss_value in losses.items():
                    if loss_name != "total":
                        writer.add_scalar(f"train/{loss_name}", _as_float(loss_value), global_step)
                if global_step % 100 == 0:
                    _log_images(writer, "train_images", batch, outputs, global_step)
                global_step += 1
                progress.set_postfix(loss=f"{current_loss:.4f}")

            scheduler.step()
            mean_train_loss = running_loss / max(1, sample_count)
            writer.add_scalar("epoch/train_loss", mean_train_loss, epoch + 1)
            writer.add_scalar("epoch/learning_rate", optimizer.param_groups[0]["lr"], epoch + 1)

            val_metrics = None
            if (epoch + 1) % validate_every == 0:
                val_metrics, preview = validate(model, val_loader, criterion, device, amp_enabled)
                for metric_name, metric_value in val_metrics.items():
                    writer.add_scalar(f"validation/{metric_name}", metric_value, epoch + 1)
                if preview is not None:
                    _log_images(writer, "validation_images", preview[0], preview[1], epoch + 1)
                print(
                    f"epoch={epoch + 1} train={mean_train_loss:.6f} "
                    f"val={val_metrics['loss']:.6f} PSNR={val_metrics['psnr']:.3f} "
                    f"SSIM={val_metrics['ssim']:.4f}"
                )
                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    _save_training_checkpoint(
                        output_dir / "best.pth", model, optimizer, scheduler, scaler,
                        epoch, best_val_loss, config,
                    )
            else:
                print(f"epoch={epoch + 1} train={mean_train_loss:.6f}")

            _save_training_checkpoint(
                output_dir / "last.pth", model, optimizer, scheduler, scaler,
                epoch, best_val_loss, config,
            )
            if (epoch + 1) % save_every == 0:
                _save_training_checkpoint(
                    output_dir / f"epoch_{epoch + 1:04d}.pth", model, optimizer, scheduler, scaler,
                    epoch, best_val_loss, config,
                )
    finally:
        writer.close()
    elapsed = time.time() - started_at
    print(f"Training finished in {elapsed / 60.0:.1f} minutes. Checkpoints: {output_dir}")


if __name__ == "__main__":
    main()
