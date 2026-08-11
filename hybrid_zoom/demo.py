"""Run the hybrid-zoom pipeline on one Wide/Tele image pair."""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from torch import nn

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hybrid_zoom.config_utils import apply_cli_overrides, load_config, resolve_device, seed_everything
from hybrid_zoom.models import HybridZoomModel
from hybrid_zoom.utils import flow_to_color, read_image, save_image


def _read_checkpoint(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint must contain a mapping, got {type(checkpoint).__name__}")
    return checkpoint


def _checkpoint_state(checkpoint: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    state_dict = checkpoint.get("model", checkpoint.get("model_state_dict", checkpoint))
    if not isinstance(state_dict, Mapping):
        raise TypeError("Checkpoint model state must be a mapping")
    return state_dict


def _load_checkpoint(model: nn.Module, checkpoint: Mapping[str, Any], strict: bool) -> None:
    incompatible = model.load_state_dict(_checkpoint_state(checkpoint), strict=strict)
    if not strict and (incompatible.missing_keys or incompatible.unexpected_keys):
        print(f"Non-strict load: missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}")


def _autocast(enabled: bool) -> Any:
    if not enabled:
        return contextlib.nullcontext()
    amp_namespace = getattr(torch, "amp", None)
    if amp_namespace is not None and hasattr(amp_namespace, "autocast"):
        return amp_namespace.autocast("cuda", enabled=True)
    return torch.cuda.amp.autocast(enabled=True)  # pragma: no cover


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wide", type=Path, required=True)
    parser.add_argument("--tele", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("./demo_result"))
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--flow-weights", type=str, default=None, help="RAFT weights name or 'none'.")
    parser.add_argument("--flow-variant", choices=("large", "small"), default=None)
    parser.add_argument("--non-strict", action="store_true")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = apply_cli_overrides(load_config(args.config), args.overrides)
    if args.device is not None:
        config["device"] = args.device
    if (args.height is None) != (args.width is None):
        raise ValueError("--height and --width must be specified together")
    if args.height is not None:
        if args.height <= 0 or args.width <= 0:
            raise ValueError("--height and --width must be positive")
        config.setdefault("image", {}).update({"height": args.height, "width": args.width})
    if args.flow_weights is not None:
        config.setdefault("flow", {})["weights"] = args.flow_weights
    if args.flow_variant is not None:
        config.setdefault("flow", {})["variant"] = args.flow_variant

    seed_everything(int(config.get("seed", 42)))
    device = resolve_device(str(config.get("device", "cuda")))
    wide = read_image(args.wide, add_batch=True, device=device)
    tele = read_image(args.tele, add_batch=True, device=device)
    checkpoint_path: Optional[Path] = args.checkpoint
    if checkpoint_path is None:
        configured = config.get("testing", {}).get("checkpoint")
        checkpoint_path = Path(configured) if configured else None
    checkpoint: Optional[Mapping[str, Any]] = None
    resolved_checkpoint: Optional[Path] = None
    if checkpoint_path is not None:
        resolved_checkpoint = checkpoint_path.expanduser().resolve()
        checkpoint = _read_checkpoint(resolved_checkpoint)
        state_keys = _checkpoint_state(checkpoint).keys()
        if any(key.startswith("flow_estimator.") or ".flow_estimator." in key for key in state_keys):
            config.setdefault("flow", {})["weights"] = None

    model = HybridZoomModel(config).to(device)
    if checkpoint is not None and resolved_checkpoint is not None:
        _load_checkpoint(model, checkpoint, strict=not args.non_strict)
        print(f"Loaded checkpoint: {resolved_checkpoint}")
    else:
        print(
            "No fusion checkpoint supplied: the zero-initialized residual head preserves Wide. "
            "Flow, warp, occlusion, rejection, and blend diagnostics are still generated."
        )

    model.eval()
    amp_enabled = bool(config.get("training", {}).get("amp", True)) and device.type == "cuda"
    with _autocast(amp_enabled):
        outputs = model(wide, tele)

    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "wide.png": outputs["wide"],
        "tele.png": outputs["tele"],
        "warped_tele.png": outputs["warped_tele"],
        "fusion.png": outputs.get("fusion_rgb", outputs["fusion_y"]),
        "final.png": outputs["output"],
        "flow.png": flow_to_color(outputs["flow_w2t"]),
        "occlusion_mask.png": outputs["occlusion_mask"],
        "rejection_mask.png": outputs["rejection_mask"],
        "blend_mask.png": outputs["blend_mask"],
    }
    for filename, tensor in files.items():
        save_image(tensor, output_dir / filename)
    print(f"Saved {len(files)} demo images to: {output_dir}")


if __name__ == "__main__":
    main()
