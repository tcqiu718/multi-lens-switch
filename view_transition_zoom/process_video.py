"""Process synchronized Wide/Tele videos or image sequences."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from utils.config import load_config, resolve_device, seed_everything
from utils.metrics import MetricsTracker
from utils.video_io import (
    PairedImageSequence,
    PairedVideoReader,
    RGBVideoWriter,
    load_zoom_ratios,
)
from utils.visualization import comparison_frame
from utils.image_io import read_mask
from baselines import render_baselines
from zoom.zoom_pipeline import ContinuousZoomPipeline
from zoom.zoom_schedule import write_schedule_csv
from zoom_runner import save_zoom_debug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuous zoom for synchronized W/T streams")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--wide-video")
    source.add_argument("--wide-dir")
    parser.add_argument("--tele-video")
    parser.add_argument("--tele-dir")
    parser.add_argument("--config", default="config_zoom.yaml")
    parser.add_argument("--output", required=True, help="Output .mp4 path")
    parser.add_argument("--zoom-start", type=float, default=None)
    parser.add_argument("--zoom-end", type=float, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--zoom-schedule", default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--save-debug", action="store_true")
    parser.add_argument("--debug-every", type=int, default=30)
    parser.add_argument("--comparison-output", default=None)
    parser.add_argument("--overlap-mask", default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if bool(args.wide_video) != bool(args.tele_video):
        raise ValueError("--wide-video and --tele-video must be supplied together")
    if bool(args.wide_dir) != bool(args.tele_dir):
        raise ValueError("--wide-dir and --tele-dir must be supplied together")
    config = load_config(args.config, args.set)
    seed_everything(int(config.get("seed", 7)))
    device = resolve_device(config.get("device", "auto"))
    reader = (
        PairedVideoReader(args.wide_video, args.tele_video, device)
        if args.wide_video
        else PairedImageSequence(args.wide_dir, args.tele_dir, device)
    )
    frame_count = reader.info.frame_count
    if args.num_frames is not None:
        frame_count = min(frame_count, args.num_frames) if frame_count > 0 else args.num_frames
    zoom_cfg = config.get("zoom", {})
    if args.zoom_schedule:
        ratios = load_zoom_ratios(args.zoom_schedule)
        if frame_count > 0 and len(ratios) != frame_count:
            raise ValueError("Zoom schedule length must match processed frame count")
        frame_count = len(ratios)
    else:
        if frame_count <= 0:
            raise ValueError("Unknown video length; provide --num-frames or --zoom-schedule")
        start = args.zoom_start if args.zoom_start is not None else float(zoom_cfg.get("start", 1.0))
        end = args.zoom_end if args.zoom_end is not None else float(zoom_cfg.get("end", 3.0))
        ratios = np.linspace(start, end, frame_count).tolist()
    fps = args.fps or reader.info.fps or float(config.get("output", {}).get("fps", 30))
    codec = str(config.get("output", {}).get("codec", "mp4v"))
    pipeline = ContinuousZoomPipeline(config, device)
    overlap_mask = (
        read_mask(args.overlap_mask, (reader.info.height, reader.info.width), device)
        if args.overlap_mask
        else None
    )
    temporal_cfg = config.get("temporal", {})
    temporal_enabled = bool(temporal_cfg.get("enabled", True))
    flow_guided = temporal_enabled and bool(temporal_cfg.get("flow_guided", False))
    scene_cut_threshold = float(temporal_cfg.get("scene_cut_threshold", 0.22))
    ratio_cfg = config.get("view_transition", config.get("constraint", {}))
    tracker = MetricsTracker(ratio_cfg.get("ratio", 0.01), fps=fps)
    output_path = Path(args.output)
    writer = RGBVideoWriter(str(output_path), reader.info.width, reader.info.height, fps, codec)
    comparison_writer = None
    if args.comparison_output:
        comparison_writer = RGBVideoWriter(
            args.comparison_output, reader.info.width * 3, reader.info.height, fps, codec
        )
    debug_root = output_path.parent / (output_path.stem + "_debug")
    previous_wide = None
    try:
        for index, zoom in enumerate(ratios):
            item = reader.read()
            if item is None:
                raise ValueError("Input streams ended before requested frame %d" % index)
            wide, tele = item
            temporal_flow = None
            if previous_wide is not None and temporal_enabled:
                scene_difference = float((wide - previous_wide).abs().mean().detach().cpu())
                if scene_difference > scene_cut_threshold:
                    pipeline.reset_temporal()
                elif flow_guided:
                    temporal_flow = pipeline.estimate_temporal_flow(wide, previous_wide)
            # Each synchronized video pair computes its dual-camera flow once.
            pair = pipeline.prepare_pair(wide, tele, overlap_mask=overlap_mask)
            result = pipeline.render(pair, float(zoom), temporal_flow=temporal_flow)
            writer.write(result.result)
            tracker.update(index, pair, result, temporal_flow=temporal_flow)
            if args.save_debug and index % max(1, args.debug_every) == 0:
                save_zoom_debug(debug_root, index, pair, result)
            if comparison_writer is not None:
                baseline = render_baselines(pipeline, pair, result)
                comparison_writer.write(
                    comparison_frame(
                        [baseline.wide_digital_zoom, baseline.direct_pyramid, result.result],
                        ["Wide Baseline", "Direct Warp", "View Transition"],
                    )
                )
            previous_wide = wide.detach()
    finally:
        reader.release()
        writer.release()
        if comparison_writer is not None:
            comparison_writer.release()
    tracker.write(str(output_path.with_name(output_path.stem + "_metrics.csv")))
    write_schedule_csv(
        str(output_path.with_name(output_path.stem + "_zoom_schedule.csv")),
        [pipeline.schedule(value) for value in ratios],
        fps=fps,
    )
    print("Video processing complete: %s" % output_path.resolve())


if __name__ == "__main__":
    main()
