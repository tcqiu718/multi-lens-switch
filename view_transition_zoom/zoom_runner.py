"""Shared rendering/export routines for static-pair and video CLIs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Optional

import torch

from baselines import BaselineResult, render_baselines
from utils.image_io import write_image, write_mask
from utils.metrics import MetricsTracker
from utils.video_io import RGBVideoWriter
from utils.visualization import comparison_frame, flow_to_color
from zoom.zoom_pipeline import ContinuousZoomPipeline, PreparedPair, ZoomFrameResult


def save_zoom_debug(
    root: Path,
    frame_index: int,
    pair: PreparedPair,
    result: ZoomFrameResult,
) -> Path:
    output = root / ("frame_%06d" % frame_index)
    write_image(str(output / "wide.png"), pair.wide)
    write_image(str(output / "tele.png"), pair.tele)
    write_image(str(output / "flow_original.png"), flow_to_color(pair.flow_t2w))
    write_image(str(output / "flow_target.png"), flow_to_color(pair.target.target_flow))
    write_image(str(output / "delta_paper.png"), flow_to_color(pair.constrained.delta_flow))
    write_image(str(output / "delta_zoom.png"), flow_to_color(result.view.delta_flow))
    if result.view.wide_motion is not None:
        write_image(str(output / "wide_motion.png"), flow_to_color(result.view.wide_motion))
    if result.view.tele_motion is not None:
        write_image(str(output / "tele_motion.png"), flow_to_color(result.view.tele_motion))
    write_mask(str(output / "distance_map.png"), pair.boundary.distance, normalize=True)
    write_mask(str(output / "occlusion.png"), result.occlusion.soft_mask)
    write_image(str(output / "tele_O.png"), result.view.tele.image)
    write_image(str(output / "wide_O.png"), result.view.wide.image)
    write_mask(str(output / "blend_mask.png"), result.fusion.occlusion_mask)
    write_mask(str(output / "overlap_mask.png"), result.fusion.overlap_mask)
    write_image(str(output / "result.png"), result.result)
    return output


def render_static_sequence(
    pipeline: ContinuousZoomPipeline,
    pair: PreparedPair,
    zoom_ratios: Iterable[float],
    output_root: str,
    fps: float,
    codec: str = "mp4v",
    save_frames: bool = True,
    save_debug: bool = False,
    debug_every: int = 30,
    comparison: bool = True,
) -> Path:
    ratios = list(zoom_ratios)
    if not ratios:
        raise ValueError("At least one zoom ratio is required")
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    height, width = pair.wide.shape[-2:]
    result_writer = RGBVideoWriter(str(output / "zoom_result.mp4"), width, height, fps, codec)
    comparison_writer = None
    if comparison:
        comparison_writer = RGBVideoWriter(
            str(output / "comparison.mp4"), width * 3, height, fps, codec
        )
    ratio_cfg = pipeline.config.get("view_transition", pipeline.config.get("constraint", {}))
    rho = ratio_cfg.get("ratio", 0.01)
    tracker = MetricsTracker(rho, fps=fps)
    parameter_rows = []
    try:
        for index, zoom in enumerate(ratios):
            result = pipeline.render(pair, float(zoom))
            result_writer.write(result.result)
            tracker.update(index, pair, result)
            parameter_rows.append(
                {
                    "frame": index,
                    "zoom": result.zoom_point.zoom,
                    "alpha": result.zoom_point.alpha,
                    "tone_progress": result.zoom_point.tone_progress,
                    "is_wide_endpoint": int(result.zoom_point.is_wide_endpoint),
                    "is_tele_endpoint": int(result.zoom_point.is_tele_endpoint),
                    "geometry_mode": pipeline.geometry_mode,
                    "rho": rho,
                    "crop": result.fov.crop_scale,
                    "crop_x0": result.fov.crop_box[0],
                    "crop_y0": result.fov.crop_box[1],
                    "crop_x1": result.fov.crop_box[2],
                    "crop_y1": result.fov.crop_box[3],
                    "tele_usage_ratio": result.tele_usage_ratio,
                }
            )
            if save_frames:
                write_image(str(output / "frames" / ("%06d.png" % index)), result.result)
            if save_debug and index % max(1, debug_every) == 0:
                save_zoom_debug(output / "debug", index, pair, result)
            if comparison_writer is not None:
                baselines = render_baselines(pipeline, pair, result)
                comparison_writer.write(
                    comparison_frame(
                        [
                            baselines.wide_digital_zoom,
                            baselines.direct_pyramid,
                            baselines.continuous_view_transition,
                        ],
                        ["Wide Baseline", "Direct Warp", "View Transition"],
                    )
                )
                if index in (0, len(ratios) // 2, len(ratios) - 1):
                    _save_baselines(output / "baselines" / ("frame_%06d" % index), baselines)
    finally:
        result_writer.release()
        if comparison_writer is not None:
            comparison_writer.release()
    tracker.write(str(output / "metrics.csv"))
    with (output / "zoom_parameters.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(parameter_rows[0].keys()))
        writer.writeheader()
        writer.writerows(parameter_rows)
    return output


def _save_baselines(root: Path, result: BaselineResult) -> None:
    write_image(str(root / "01_wide_digital_zoom.png"), result.wide_digital_zoom)
    write_image(str(root / "02_direct_warp.png"), result.direct_warp)
    write_image(str(root / "03_direct_pyramid.png"), result.direct_pyramid)
    write_image(str(root / "04_paper_view_transition.png"), result.paper_view_transition)
    write_image(str(root / "05_continuous_view_transition.png"), result.continuous_view_transition)


__all__ = ["render_static_sequence", "save_zoom_debug"]
