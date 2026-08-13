"""Synthetic checkerboard/rectangle geometry test and visual demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from fusion.occlusion import paper_occlusion
from utils.image_io import write_image, write_mask
from utils.visualization import flow_to_color
from utils.warp import backward_warp
from utils.warp import forward_warp
from view_transition.view_transition import ViewTransition


def make_synthetic_pair(height: int = 192, width: int = 320) -> tuple:
    """Create Tele, canonical T->W flow, and Wide via backward sampling.

    Background disparity is +10 px and foreground disparity is +30 px. The
    layer edge intentionally creates a discontinuous correspondence field.
    """
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    checker = (((x // 16 + y // 16) % 2).float() * 0.55 + 0.18)[None, None]
    tele = checker.repeat(1, 3, 1, 1)
    y0, y1 = height // 4, height * 3 // 4
    x0, x1 = width * 3 // 10, width * 7 // 10
    tele[:, 0, y0:y1, x0:x1] = 0.92
    tele[:, 1, y0:y1, x0:x1] = 0.20
    tele[:, 2, y0:y1, x0:x1] = 0.12
    flow = torch.zeros(1, 2, height, width)
    flow[:, 0] = 10.0
    flow[:, 0, y0:y1, x0:x1] = 30.0
    wide = backward_warp(tele, flow)
    return wide, tele, flow


def run(output: str) -> dict:
    wide, tele, flow = make_synthetic_pair()
    config = {
        "target_flow": {"kernel_size": 61, "foreground_mode": "magnitude"},
        "boundary": {"mode": "flow_aware", "gradient_threshold": 1.0, "max_extra_distance": 96},
        "constraint": {"ratio": 0.05},
        "transform": {
            "splat_mode": "bilinear",
            "fill_mode": "background_nearest",
            "multi_warp_average": True,
            "wide_offset_min": -0.5,
            "wide_offset_max": 0.5,
            "wide_offset_step": 0.5,
        },
    }
    engine = ViewTransition(config)
    result = engine(wide, tele, flow)
    direct = backward_warp(tele, flow)
    direct_occ = paper_occlusion(flow, result.target.foreground_mask, soft_width=0)
    foreground_o, _, foreground_valid = forward_warp(
        result.target.foreground_mask, result.constrained.delta_flow
    )
    foreground_o = (foreground_o > 0.5).float() * foreground_valid
    transition_occ = paper_occlusion(
        result.tele.flow_t2o, foreground_o, soft_width=0
    )
    output_root = Path(output)
    write_image(str(output_root / "wide.png"), wide)
    write_image(str(output_root / "tele.png"), tele)
    write_image(str(output_root / "direct_t2w.png"), direct)
    write_image(str(output_root / "tele_to_o.png"), result.tele.image)
    write_image(str(output_root / "wide_to_o.png"), result.wide.image)
    write_image(str(output_root / "flow_original.png"), flow_to_color(flow))
    write_image(str(output_root / "flow_t2o.png"), flow_to_color(result.tele.flow_t2o))
    write_mask(str(output_root / "direct_occlusion.png"), direct_occ.hard_mask)
    write_mask(str(output_root / "transition_occlusion.png"), transition_occ.hard_mask)
    metrics = {
        "direct_occlusion_ratio": float(direct_occ.hard_mask.mean()),
        "transition_occlusion_ratio": float(transition_occ.hard_mask.mean()),
        "tele_hole_ratio": float(result.tele.hole_mask.mean()),
        "mean_delta": float(result.constrained.delta_flow.abs().mean()),
        "finite": bool(torch.isfinite(result.tele.image).all() and torch.isfinite(result.wide.image).all()),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    import numpy as np

    np.save(str(output_root / "flow_t2w.npy"), flow[0].numpy())
    np.save(str(output_root / "flow_w2t_approx.npy"), (-flow[0]).numpy())
    with (output_root / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    if not metrics["finite"]:
        raise AssertionError("Synthetic View Transition produced NaN/Inf")
    if metrics["transition_occlusion_ratio"] > metrics["direct_occlusion_ratio"] + 1.0e-6:
        raise AssertionError("Synthetic View Transition increased the estimated occlusion ratio")
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic View Transition sanity check")
    parser.add_argument("--output", default="outputs/synthetic_test")
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
