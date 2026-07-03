"""View-transition style dual-camera image fusion demo.

This is an engineering demo inspired by "View Transition based Dual Camera
Image Fusion". It is not an official reproduction: the paper does not release
code, so several implementation choices are exposed as CLI parameters.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy import ndimage
from skimage.registration import optical_flow_tvl1
from skimage.transform import resize


Array = np.ndarray


@dataclass
class FusionConfig:
    flow_backend: str = "tvl1"
    flow_format: str = "yx"
    resize_long_edge: int = 0
    transition_box: int = 600
    transition_ratio: float = 0.01
    flow_smooth_strength: float = 0.65
    edge_quantile: float = 0.88
    flow_jump_quantile: float = 0.92
    edge_keep_distance: float = 8.0
    occlusion_soft_zone: float = 15.0
    residual_sigma: float = 0.12
    tone_block: int = 200
    tone_strength: float = 0.85
    pyramid_levels: int = 5
    debug: bool = True


def read_rgb(path: Path) -> Array:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def write_rgb(path: Path, image: Array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_u8 = np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)
    Image.fromarray(image_u8, mode="RGB").save(path)


def write_gray(path: Path, image: Array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(x)
    if finite.any():
        lo = float(np.percentile(x[finite], 1.0))
        hi = float(np.percentile(x[finite], 99.0))
        if hi > lo:
            x = (x - lo) / (hi - lo)
    x = np.clip(x, 0.0, 1.0)
    Image.fromarray(np.rint(x * 255.0).astype(np.uint8), mode="L").save(path)


def maybe_resize_long_edge(image: Array, long_edge: int) -> Array:
    if long_edge <= 0:
        return image
    h, w = image.shape[:2]
    scale = float(long_edge) / float(max(h, w))
    if scale >= 1.0:
        return image
    out_shape = (max(1, int(round(h * scale))), max(1, int(round(w * scale))))
    return resize(image, out_shape, order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)


def resize_like(image: Array, shape_hw: Tuple[int, int]) -> Array:
    if image.shape[:2] == shape_hw:
        return image
    return resize(image, shape_hw, order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)


def rgb_to_luma(image: Array) -> Array:
    return (0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]).astype(np.float32)


def odd_window(size: int, h: int, w: int) -> int:
    max_size = max(3, min(h, w))
    size = int(max(3, min(size, max_size)))
    if size % 2 == 0:
        size -= 1
    return max(3, size)


def box_smooth(image: Array, size: int) -> Array:
    if image.ndim == 2:
        return ndimage.uniform_filter(image, size=size, mode="reflect")
    channels = [ndimage.uniform_filter(image[..., c], size=size, mode="reflect") for c in range(image.shape[-1])]
    return np.stack(channels, axis=-1)


def estimate_flow_tvl1(wide: Array, tele: Array) -> Array:
    """Return flow in (dy, dx) order that samples tele into wide coordinates."""
    v, u = optical_flow_tvl1(
        rgb_to_luma(wide),
        rgb_to_luma(tele),
        attachment=15,
        tightness=0.3,
        num_warp=5,
        num_iter=30,
        prefilter=True,
        dtype=np.float32,
    )
    return np.stack([v, u], axis=-1).astype(np.float32)


def load_precomputed_flow(path: Path, shape_hw: Tuple[int, int], flow_format: str) -> Array:
    flow = np.load(path).astype(np.float32)
    if flow.ndim != 3 or flow.shape[-1] != 2:
        raise ValueError(f"Expected flow .npy with shape HxWx2, got {flow.shape}")
    if flow_format == "xy":
        flow = flow[..., ::-1]
    if flow.shape[:2] != shape_hw:
        old_h, old_w = flow.shape[:2]
        new_h, new_w = shape_hw
        flow = resize(flow, shape_hw, order=1, preserve_range=True, anti_aliasing=False).astype(np.float32)
        flow[..., 0] *= float(new_h) / float(old_h)
        flow[..., 1] *= float(new_w) / float(old_w)
    return flow


def flow_gradient_magnitude(flow: Array) -> Array:
    gy_y, gy_x = np.gradient(flow[..., 0])
    gx_y, gx_x = np.gradient(flow[..., 1])
    return np.sqrt(gy_y * gy_y + gy_x * gy_x + gx_y * gx_y + gx_x * gx_x).astype(np.float32)


def image_gradient_magnitude(image: Array) -> Array:
    gray = rgb_to_luma(image)
    gy, gx = np.gradient(gray)
    return np.sqrt(gy * gy + gx * gx).astype(np.float32)


def regularize_flow_for_view_transition(flow: Array, wide: Array, cfg: FusionConfig) -> Tuple[Array, Array, Array]:
    """Smooth flow in connected flat regions while preserving structure edges."""
    h, w = flow.shape[:2]
    box = odd_window(cfg.transition_box, h, w)
    smooth_flow = box_smooth(flow, box)

    image_edges = image_gradient_magnitude(wide)
    flow_jumps = flow_gradient_magnitude(flow)
    edge_thr = float(np.quantile(image_edges, cfg.edge_quantile))
    jump_thr = float(np.quantile(flow_jumps, cfg.flow_jump_quantile))
    barrier = (image_edges >= edge_thr) | (flow_jumps >= jump_thr)

    distance = ndimage.distance_transform_edt(~barrier)
    flat_weight = np.clip((distance - cfg.edge_keep_distance) / max(cfg.occlusion_soft_zone, 1e-6), 0.0, 1.0)
    smooth_blend = np.clip(cfg.flow_smooth_strength * flat_weight, 0.0, 1.0)[..., None]
    regularized = flow * (1.0 - smooth_blend) + smooth_flow * smooth_blend

    # Keep the paper's small transition ratio as an explicit, tunable step.
    transitioned = flow + cfg.transition_ratio * (regularized - flow)
    transition_delta = np.linalg.norm(transitioned - flow, axis=-1)
    return transitioned.astype(np.float32), flat_weight.astype(np.float32), transition_delta.astype(np.float32)


def warp_with_flow(image: Array, flow_yx: Array) -> Tuple[Array, Array]:
    h, w = flow_yx.shape[:2]
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    sample_y = yy + flow_yx[..., 0]
    sample_x = xx + flow_yx[..., 1]
    valid = (sample_y >= 0.0) & (sample_y <= image.shape[0] - 1) & (sample_x >= 0.0) & (sample_x <= image.shape[1] - 1)

    warped_channels = []
    for c in range(image.shape[-1]):
        warped = ndimage.map_coordinates(
            image[..., c],
            [sample_y, sample_x],
            order=1,
            mode="constant",
            cval=0.0,
        )
        warped_channels.append(warped)
    return np.stack(warped_channels, axis=-1).astype(np.float32), valid.astype(np.float32)


def build_tele_weight(wide: Array, tele_warped: Array, valid: Array, cfg: FusionConfig) -> Tuple[Array, Array]:
    residual = np.abs(rgb_to_luma(wide) - rgb_to_luma(tele_warped))
    residual_weight = np.exp(-residual / max(cfg.residual_sigma, 1e-6))
    weight = valid * residual_weight
    if cfg.occlusion_soft_zone > 0:
        weight = ndimage.gaussian_filter(weight, sigma=cfg.occlusion_soft_zone / 3.0, mode="nearest")
    weight = np.clip(weight, 0.0, 1.0)
    return weight.astype(np.float32), residual.astype(np.float32)


def weighted_local_stats(image: Array, weight: Array, window: int) -> Tuple[Array, Array]:
    eps = 1e-5
    h, w = image.shape[:2]
    window = odd_window(window, h, w)
    weight = weight.astype(np.float32)
    weight_sum = ndimage.uniform_filter(weight, size=window, mode="reflect")
    means = []
    stds = []
    for c in range(image.shape[-1]):
        x = image[..., c]
        mean = ndimage.uniform_filter(x * weight, size=window, mode="reflect") / (weight_sum + eps)
        second = ndimage.uniform_filter(x * x * weight, size=window, mode="reflect") / (weight_sum + eps)
        var = np.maximum(second - mean * mean, eps)
        means.append(mean)
        stds.append(np.sqrt(var))
    return np.stack(means, axis=-1), np.stack(stds, axis=-1)


def local_tone_match(tele_warped: Array, wide: Array, tele_weight: Array, cfg: FusionConfig) -> Array:
    valid_weight = np.clip(tele_weight, 0.0, 1.0)
    ref_mean, ref_std = weighted_local_stats(wide, np.ones_like(valid_weight), cfg.tone_block)
    src_mean, src_std = weighted_local_stats(tele_warped, valid_weight, cfg.tone_block)
    matched = (tele_warped - src_mean) * (ref_std / np.maximum(src_std, 1e-4)) + ref_mean
    matched = np.clip(matched, 0.0, 1.0)
    return (tele_warped * (1.0 - cfg.tone_strength) + matched * cfg.tone_strength).astype(np.float32)


def gaussian_pyramid(image: Array, levels: int) -> list[Array]:
    pyramid = [image.astype(np.float32)]
    cur = image.astype(np.float32)
    for _ in range(1, levels):
        if min(cur.shape[:2]) <= 8:
            break
        cur = blur_image(cur, sigma=1.0)
        next_shape = (max(1, cur.shape[0] // 2), max(1, cur.shape[1] // 2))
        cur = resize(cur, next_shape, order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
        pyramid.append(cur)
    return pyramid


def blur_image(image: Array, sigma: float) -> Array:
    if image.ndim == 2:
        return ndimage.gaussian_filter(image, sigma=sigma, mode="reflect")
    return np.stack(
        [ndimage.gaussian_filter(image[..., c], sigma=sigma, mode="reflect") for c in range(image.shape[-1])],
        axis=-1,
    ).astype(np.float32)


def resize_to(image: Array, shape_hw: Tuple[int, int]) -> Array:
    return resize(image, shape_hw, order=1, preserve_range=True, anti_aliasing=False).astype(np.float32)


def laplacian_pyramid(image: Array, levels: int) -> list[Array]:
    g = gaussian_pyramid(image, levels)
    lap = []
    for i in range(len(g) - 1):
        up = resize_to(g[i + 1], g[i].shape[:2])
        lap.append((g[i] - up).astype(np.float32))
    lap.append(g[-1])
    return lap


def multiband_blend(wide: Array, tele: Array, tele_weight: Array, levels: int) -> Array:
    weight = np.clip(tele_weight, 0.0, 1.0)
    weight3 = weight[..., None]
    wide_lap = laplacian_pyramid(wide, levels)
    tele_lap = laplacian_pyramid(tele, levels)
    mask_g = gaussian_pyramid(weight3, levels)
    n = min(len(wide_lap), len(tele_lap), len(mask_g))

    blended = tele_lap[-1] * mask_g[n - 1] + wide_lap[-1] * (1.0 - mask_g[n - 1])
    for i in range(n - 2, -1, -1):
        blended = resize_to(blended, wide_lap[i].shape[:2])
        blended = blended + tele_lap[i] * mask_g[i] + wide_lap[i] * (1.0 - mask_g[i])
    return np.clip(blended, 0.0, 1.0).astype(np.float32)


def synthetic_pair(size: Tuple[int, int] = (256, 384)) -> Tuple[Array, Array]:
    h, w = size
    base = Image.new("RGB", (w, h), (34, 42, 48))
    draw = ImageDraw.Draw(base)
    for y in range(h):
        color = (30 + y * 45 // h, 42 + y * 55 // h, 54 + y * 60 // h)
        draw.line([(0, y), (w, y)], fill=color)
    draw.rectangle([w * 0.10, h * 0.16, w * 0.42, h * 0.58], fill=(196, 69, 56))
    draw.ellipse([w * 0.47, h * 0.18, w * 0.82, h * 0.70], fill=(53, 145, 205))
    draw.rounded_rectangle([w * 0.21, h * 0.66, w * 0.90, h * 0.87], radius=8, fill=(229, 186, 76))
    draw.text((w * 0.12, h * 0.08), "WIDE / TELE", fill=(240, 240, 235))
    base_np = np.asarray(base, dtype=np.float32) / 255.0

    wide = np.asarray(base.filter(ImageFilter.GaussianBlur(radius=1.2)), dtype=np.float32) / 255.0
    wide = np.clip(wide * np.array([0.94, 1.00, 1.06], dtype=np.float32), 0.0, 1.0)

    yy, xx = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing="ij")
    cy, cx = (h - 1) * 0.5, (w - 1) * 0.5
    zoom = 1.08
    sample_y = (yy - cy) / zoom + cy + 3.5
    sample_x = (xx - cx) / zoom + cx - 10.0
    tele_ch = [
        ndimage.map_coordinates(base_np[..., c], [sample_y, sample_x], order=1, mode="reflect") for c in range(3)
    ]
    tele = np.stack(tele_ch, axis=-1)
    tele = np.clip(tele * np.array([1.06, 1.01, 0.93], dtype=np.float32) + 0.018, 0.0, 1.0)
    return wide.astype(np.float32), tele.astype(np.float32)


def fuse_pair(wide: Array, tele: Array, cfg: FusionConfig, flow_npy: Optional[Path] = None) -> dict[str, Array]:
    wide = maybe_resize_long_edge(wide, cfg.resize_long_edge)
    tele = maybe_resize_long_edge(tele, cfg.resize_long_edge)
    tele = resize_like(tele, wide.shape[:2])

    if cfg.flow_backend == "precomputed":
        if flow_npy is None:
            raise ValueError("--flow-npy is required when --flow-backend precomputed")
        raw_flow = load_precomputed_flow(flow_npy, wide.shape[:2], cfg.flow_format)
    elif cfg.flow_backend == "tvl1":
        raw_flow = estimate_flow_tvl1(wide, tele)
    else:
        raise ValueError(f"Unknown flow backend: {cfg.flow_backend}")

    transition_flow, flat_weight, transition_delta = regularize_flow_for_view_transition(raw_flow, wide, cfg)
    tele_warped, valid = warp_with_flow(tele, transition_flow)
    tele_weight, residual = build_tele_weight(wide, tele_warped, valid, cfg)
    tele_tone = local_tone_match(tele_warped, wide, tele_weight, cfg)
    fused = multiband_blend(wide, tele_tone, tele_weight, cfg.pyramid_levels)
    flow_mag = np.linalg.norm(raw_flow, axis=-1)

    return {
        "wide": wide,
        "tele_resized": tele,
        "raw_flow_mag": flow_mag,
        "transition_flat_weight": flat_weight,
        "transition_delta": transition_delta,
        "tele_warped": tele_warped,
        "valid_mask": valid,
        "photometric_residual": residual,
        "tele_weight": tele_weight,
        "tele_tone_matched": tele_tone,
        "fused": fused,
    }


def save_outputs(outputs: dict[str, Array], out_dir: Path, cfg: FusionConfig) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ["wide", "tele_resized", "tele_warped", "tele_tone_matched", "fused"]:
        write_rgb(out_dir / f"{name}.png", outputs[name])
    for name in [
        "raw_flow_mag",
        "transition_flat_weight",
        "transition_delta",
        "valid_mask",
        "photometric_residual",
        "tele_weight",
    ]:
        write_gray(out_dir / f"{name}.png", outputs[name])
    with (out_dir / "params.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="View-transition style dual-camera fusion demo.")
    parser.add_argument("--wide", type=Path, help="Wide-camera image path.")
    parser.add_argument("--tele", type=Path, help="Tele-camera image path.")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/view_transition_fusion_demo"))
    parser.add_argument("--self-test", action="store_true", help="Run on a synthetic pair generated by the script.")
    parser.add_argument("--flow-backend", choices=["tvl1", "precomputed"], default="tvl1")
    parser.add_argument("--flow-npy", type=Path, help="Precomputed HxWx2 optical flow .npy.")
    parser.add_argument("--flow-format", choices=["yx", "xy"], default="yx")
    parser.add_argument("--resize-long-edge", type=int, default=0)
    parser.add_argument("--transition-box", type=int, default=600)
    parser.add_argument("--transition-ratio", type=float, default=0.01)
    parser.add_argument("--flow-smooth-strength", type=float, default=0.65)
    parser.add_argument("--edge-quantile", type=float, default=0.88)
    parser.add_argument("--flow-jump-quantile", type=float, default=0.92)
    parser.add_argument("--edge-keep-distance", type=float, default=8.0)
    parser.add_argument("--occlusion-soft-zone", type=float, default=15.0)
    parser.add_argument("--residual-sigma", type=float, default=0.12)
    parser.add_argument("--tone-block", type=int, default=200)
    parser.add_argument("--tone-strength", type=float, default=0.85)
    parser.add_argument("--pyramid-levels", type=int, default=5)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = FusionConfig(
        flow_backend=args.flow_backend,
        flow_format=args.flow_format,
        resize_long_edge=args.resize_long_edge,
        transition_box=args.transition_box,
        transition_ratio=args.transition_ratio,
        flow_smooth_strength=args.flow_smooth_strength,
        edge_quantile=args.edge_quantile,
        flow_jump_quantile=args.flow_jump_quantile,
        edge_keep_distance=args.edge_keep_distance,
        occlusion_soft_zone=args.occlusion_soft_zone,
        residual_sigma=args.residual_sigma,
        tone_block=args.tone_block,
        tone_strength=args.tone_strength,
        pyramid_levels=args.pyramid_levels,
    )

    if args.self_test:
        wide, tele = synthetic_pair()
        args.out_dir.mkdir(parents=True, exist_ok=True)
        write_rgb(args.out_dir / "synthetic_wide_input.png", wide)
        write_rgb(args.out_dir / "synthetic_tele_input.png", tele)
    else:
        if args.wide is None or args.tele is None:
            raise SystemExit("Provide --wide and --tele, or use --self-test.")
        wide = read_rgb(args.wide)
        tele = read_rgb(args.tele)

    outputs = fuse_pair(wide, tele, cfg, args.flow_npy)
    save_outputs(outputs, args.out_dir, cfg)
    print(f"Saved fusion outputs to: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
