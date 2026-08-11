"""Run hybrid-zoom inference over a paired dataset and save diagnostics."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hybrid_zoom.config_utils import (
    apply_cli_overrides,
    load_config,
    resolve_device,
    resolve_wide_crop_size,
    seed_everything,
)
from hybrid_zoom.datasets import HybridZoomDataset
from hybrid_zoom.models import HybridZoomModel
from hybrid_zoom.utils import psnr, save_pipeline_outputs, ssim


def _read_checkpoint(checkpoint_path: Path) -> Mapping[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint must contain a mapping, got {type(checkpoint).__name__}")
    return checkpoint


def _checkpoint_state(checkpoint: Mapping[str, Any]) -> Mapping[str, Tensor]:
    state_dict = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    if not isinstance(state_dict, Mapping):
        raise TypeError("Checkpoint model state must be a mapping")
    return state_dict


def _contains_flow_state(checkpoint: Mapping[str, Any]) -> bool:
    return any(
        key.startswith("flow_estimator.") or ".flow_estimator." in key
        for key in _checkpoint_state(checkpoint).keys()
    )


def _load_model_checkpoint(
    model: nn.Module,
    checkpoint: Mapping[str, Any],
    strict: bool,
) -> None:
    incompatible = model.load_state_dict(_checkpoint_state(checkpoint), strict=strict)
    if not strict and (incompatible.missing_keys or incompatible.unexpected_keys):
        print(f"Non-strict checkpoint load: missing={incompatible.missing_keys}, "
              f"unexpected={incompatible.unexpected_keys}")


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def _sample_names(value: Any, batch_size: int) -> List[str]:
    if isinstance(value, str):
        names = [value]
    elif isinstance(value, (tuple, list)):
        names = [str(item) for item in value]
    else:
        names = [f"{index:06d}.png" for index in range(batch_size)]
    if len(names) != batch_size:
        raise RuntimeError(f"Expected {batch_size} sample names, got {len(names)}")
    return names


def _metric_float(value: Any) -> float:
    if isinstance(value, Tensor):
        return float(value.detach().mean().cpu())
    return float(value)


def _autocast(enabled: bool) -> Any:
    if not enabled:
        return contextlib.nullcontext()
    amp_namespace = getattr(torch, "amp", None)
    if amp_namespace is not None and hasattr(amp_namespace, "autocast"):
        return amp_namespace.autocast("cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)  # pragma: no cover


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--non-strict", action="store_true", help="Allow missing/unexpected checkpoint keys.")
    parser.add_argument("--flow-weights", type=str, default=None, help="RAFT weights name or 'none'.")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(load_config(args.config), args.overrides)
    data_cfg = config.setdefault("data", {})
    test_cfg = config.setdefault("testing", {})
    if args.data_root is not None:
        data_cfg["root"] = str(args.data_root)
    if args.checkpoint is not None:
        test_cfg["checkpoint"] = str(args.checkpoint)
    if args.output is not None:
        test_cfg["output_dir"] = str(args.output)
    if args.device is not None:
        config["device"] = args.device
    if args.flow_weights is not None:
        config.setdefault("flow", {})["weights"] = args.flow_weights

    seed_everything(int(config.get("seed", 42)))
    device = resolve_device(str(config.get("device", "cuda")))
    image_cfg = config.get("image", {})
    image_size = (int(image_cfg.get("height", 512)), int(image_cfg.get("width", 768)))
    split = args.split or str(data_cfg.get("test_split", "test"))
    dataset = HybridZoomDataset(
        Path(data_cfg.get("root", "./dataset")).expanduser(),
        split=split,
        image_size=image_size,
        augment=False,
        require_gt=None,
        keep_aspect_ratio=bool(image_cfg.get("keep_aspect_ratio", False)),
        aspect_mode=str(image_cfg.get("aspect_mode", "letterbox")),
        interpolation=str(image_cfg.get("interpolation", "bilinear")),
        wide_crop_size=resolve_wide_crop_size(image_cfg),
        crop_gt_with_wide=bool(image_cfg.get("crop_gt_with_wide", True)),
    )
    batch_size = args.batch_size or int(test_cfg.get("batch_size", 1))
    num_workers = args.num_workers if args.num_workers is not None else int(test_cfg.get("num_workers", 2))
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers must be non-negative")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    checkpoint_value: Optional[str] = test_cfg.get("checkpoint")
    checkpoint_path: Optional[Path] = None
    checkpoint: Optional[Mapping[str, Any]] = None
    if checkpoint_value:
        checkpoint_path = Path(checkpoint_value).expanduser().resolve()
        checkpoint = _read_checkpoint(checkpoint_path)
        if _contains_flow_state(checkpoint):
            config.setdefault("flow", {})["weights"] = None

    model = HybridZoomModel(config).to(device)
    if checkpoint is not None and checkpoint_path is not None:
        _load_model_checkpoint(model, checkpoint, strict=not args.non_strict)
        print(f"Loaded checkpoint: {checkpoint_path}")
    else:
        print(
            "No fusion checkpoint was supplied. Residual FusionUNet starts as identity, "
            "so diagnostics are valid but no learned detail transfer is expected."
        )
    model.eval()

    output_dir = Path(test_cfg.get("output_dir", "./results")).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: List[Dict[str, Any]] = []
    amp_enabled = bool(test_cfg.get("amp", config.get("training", {}).get("amp", True))) and device.type == "cuda"
    for raw_batch in tqdm(loader, desc="test"):
        batch = _move_batch(raw_batch, device)
        with _autocast(amp_enabled):
            outputs = model(batch["wide"], batch["tele"], preprocessed=True)
        names = _sample_names(batch.get("name"), int(batch["wide"].shape[0]))
        save_pipeline_outputs(outputs, output_dir, names=names, flat=False)

        if "gt" in batch:
            for index, name in enumerate(names):
                prediction = outputs["output"][index : index + 1]
                target = batch["gt"][index : index + 1]
                metric_rows.append(
                    {
                        "name": name,
                        "psnr": _metric_float(psnr(prediction, target)),
                        "ssim": _metric_float(ssim(prediction, target)),
                    }
                )

    metrics_path = output_dir / "metrics.txt"
    with metrics_path.open("w", encoding="utf-8") as handle:
        handle.write(f"samples: {len(dataset)}\n")
        handle.write(f"checkpoint: {checkpoint_value or 'none'}\n")
        if metric_rows:
            mean_psnr = sum(row["psnr"] for row in metric_rows) / len(metric_rows)
            mean_ssim = sum(row["ssim"] for row in metric_rows) / len(metric_rows)
            handle.write(f"mean_psnr: {mean_psnr:.6f}\n")
            handle.write(f"mean_ssim: {mean_ssim:.6f}\n")
            handle.write("\nper_image:\n")
            for row in metric_rows:
                handle.write(f"{row['name']} PSNR={row['psnr']:.6f} SSIM={row['ssim']:.6f}\n")
        else:
            handle.write("GT unavailable; PSNR/SSIM were not computed.\n")
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump({key: value for key, value in config.items() if not key.startswith("_")}, handle, indent=2)
    print(f"Saved {len(dataset)} results and metrics to: {output_dir}")


if __name__ == "__main__":
    main()
