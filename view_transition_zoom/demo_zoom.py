"""Generate continuous zoom from one synchronized Wide/Tele image pair."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from utils.config import load_config, resolve_device, seed_everything
from utils.flow_utils import load_flow
from utils.image_io import read_image, read_mask
from utils.video_io import load_zoom_ratios
from zoom.zoom_pipeline import ContinuousZoomPipeline
from zoom.zoom_schedule import write_schedule_csv
from zoom_runner import render_static_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuous dual-camera zoom from a static pair")
    parser.add_argument("--wide", required=True)
    parser.add_argument("--tele", required=True)
    parser.add_argument("--config", default="config_zoom.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--zoom-start", type=float, default=None)
    parser.add_argument("--zoom-end", type=float, default=None)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--zoom-schedule", default=None)
    parser.add_argument("--flow", default=None)
    parser.add_argument("--reverse-flow", default=None)
    parser.add_argument("--overlap-mask", default=None)
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument("--debug-every", type=int, default=30)
    parser.add_argument("--no-frames", action="store_true")
    parser.add_argument("--no-comparison", action="store_true")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.set)
    seed_everything(int(config.get("seed", 7)))
    device = resolve_device(config.get("device", "auto"))
    resize = config.get("input", {}).get("resize")
    resize = tuple(resize) if resize else None
    wide = read_image(args.wide, device=device, resize=resize)
    tele = read_image(args.tele, device=device, resize=resize)
    flow = load_flow(args.flow, wide) if args.flow else None
    reverse = load_flow(args.reverse_flow, tele) if args.reverse_flow else None
    overlap = read_mask(args.overlap_mask, wide.shape[-2:], device) if args.overlap_mask else None
    pipeline = ContinuousZoomPipeline(config, device)
    pair = pipeline.prepare_pair(wide, tele, flow, reverse, overlap)
    zoom_cfg = config.get("zoom", {})
    if args.zoom_schedule:
        ratios = load_zoom_ratios(args.zoom_schedule)
    else:
        start = args.zoom_start if args.zoom_start is not None else float(zoom_cfg.get("start", 1.0))
        end = args.zoom_end if args.zoom_end is not None else float(zoom_cfg.get("end", 3.0))
        frames = args.frames if args.frames is not None else int(zoom_cfg.get("frames", 90))
        ratios = np.linspace(start, end, frames).tolist()
    points = [pipeline.schedule(value) for value in ratios]
    output = args.output or config.get("output", {}).get("root", "outputs/continuous_zoom")
    output_path = Path(output)
    fps = args.fps if args.fps is not None else float(config.get("output", {}).get("fps", 30))
    write_schedule_csv(str(output_path / "zoom_schedule.csv"), points, fps=fps)
    render_static_sequence(
        pipeline,
        pair,
        ratios,
        output,
        fps,
        codec=str(config.get("output", {}).get("codec", "mp4v")),
        save_frames=not args.no_frames and bool(config.get("output", {}).get("save_frames", True)),
        save_debug=args.save_debug or bool(config.get("output", {}).get("save_debug", False)),
        debug_every=args.debug_every,
        comparison=not args.no_comparison and bool(config.get("output", {}).get("comparison_video", True)),
    )
    print("Continuous zoom complete: %s" % output_path.resolve())


if __name__ == "__main__":
    main()
