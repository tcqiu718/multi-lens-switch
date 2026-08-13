"""Run paper_repro on one synchronized Wide/Tele image pair."""

from __future__ import annotations

import argparse
from pathlib import Path

from paper_pipeline import PaperReproPipeline, save_paper_debug
from utils.config import load_config, resolve_device, seed_everything
from utils.image_io import read_image, read_mask
from utils.flow_utils import load_flow


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper View Transition reproduction")
    parser.add_argument("--wide", required=True, help="Wide RGB image")
    parser.add_argument("--tele", required=True, help="Tele RGB image")
    parser.add_argument("--config", default="config_paper.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--flow", default=None, help="Optional precomputed .npy/.pt F_T2W")
    parser.add_argument("--reverse-flow", default=None)
    parser.add_argument("--overlap-mask", default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.set)
    seed_everything(int(config.get("seed", 7)))
    if args.flow:
        config.setdefault("flow", {})["model"] = "precomputed"
        config["flow"]["fallback"] = None
        config["flow"]["precomputed_path"] = args.flow
    device = resolve_device(config.get("device", "auto"))
    resize = config.get("input", {}).get("resize")
    resize = tuple(resize) if resize else None
    wide = read_image(args.wide, device=device, resize=resize)
    tele = read_image(args.tele, device=device, resize=resize)
    flow = load_flow(args.flow, wide) if args.flow else None
    reverse = load_flow(args.reverse_flow, tele) if args.reverse_flow else None
    overlap = read_mask(args.overlap_mask, wide.shape[-2:], device) if args.overlap_mask else None
    pipeline = PaperReproPipeline(config, device)
    result = pipeline.run(wide, tele, flow_t2w=flow, reverse_flow=reverse, overlap_mask=overlap)
    output = args.output or config.get("output", {}).get("root", "outputs/paper_repro")
    save_paper_debug(output, wide, tele, result)
    print("Paper reproduction complete: %s" % Path(output).resolve())


if __name__ == "__main__":
    main()
