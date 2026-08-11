"""Fast CPU tests for coordinates, masks, color, and the integrated model."""

from __future__ import annotations

import unittest

import torch
from torch import Tensor, nn

from hybrid_zoom.models import FusionUNet, HybridZoomModel
from hybrid_zoom.modules import (
    adaptive_blend,
    compute_occlusion_mask,
    compute_rejection_mask,
    resize_flow,
    rgb_to_ycbcr,
    warp,
    ycbcr_to_rgb,
)


class ZeroFlow(nn.Module):
    """Deterministic first-to-second estimator for pipeline tests."""

    def forward(self, first: Tensor, second: Tensor) -> Tensor:
        return first.new_zeros((first.shape[0], 2, first.shape[-2], first.shape[-1]))


class WarpTests(unittest.TestCase):
    def test_positive_dx_samples_to_the_right(self) -> None:
        # A Tele marker at x=3 appears at Wide x=1 when flow_w2t.dx=+2.
        source = torch.zeros(1, 1, 5, 7)
        source[0, 0, 2, 3] = 1.0
        flow = torch.zeros(1, 2, 5, 7)
        flow[:, 0] = 2.0
        warped, valid = warp(source, flow, mode="nearest", return_mask=True)
        self.assertEqual(float(warped[0, 0, 2, 1]), 1.0)
        self.assertEqual(float(warped[0, 0, 2, 3]), 0.0)
        self.assertTrue(torch.all(valid[..., :5] == 1))
        self.assertTrue(torch.all(valid[..., 5:] == 0))

    def test_identity_for_both_align_corners_modes(self) -> None:
        image = torch.rand(2, 3, 9, 11)
        flow = torch.zeros(2, 2, 9, 11)
        for align_corners in (True, False):
            actual = warp(image, flow, align_corners=align_corners)
            self.assertTrue(torch.allclose(actual, image, atol=1e-6))

    def test_resize_flow_scales_displacements(self) -> None:
        flow = torch.ones(1, 2, 4, 5)
        resized = resize_flow(flow, (8, 15))
        self.assertEqual(tuple(resized.shape), (1, 2, 8, 15))
        self.assertTrue(torch.allclose(resized[:, 0], torch.full_like(resized[:, 0], 3.0)))
        self.assertTrue(torch.allclose(resized[:, 1], torch.full_like(resized[:, 1], 2.0)))


class ModuleTests(unittest.TestCase):
    def test_ycbcr_round_trip(self) -> None:
        rgb = torch.rand(2, 3, 13, 17)
        reconstructed = ycbcr_to_rgb(rgb_to_ycbcr(rgb))
        self.assertTrue(torch.allclose(reconstructed, rgb, atol=2e-5))

    def test_forward_backward_occlusion(self) -> None:
        forward = torch.zeros(1, 2, 12, 14)
        backward = torch.zeros_like(forward)
        reliable = compute_occlusion_mask(forward, backward, mode="hard", threshold=0.1)
        self.assertEqual(int(torch.count_nonzero(reliable)), 0)
        forward[:, 0] = 2.0
        inconsistent = compute_occlusion_mask(forward, backward, mode="hard", threshold=0.1)
        self.assertGreater(float(inconsistent.mean()), 0.9)

    def test_rejection_uses_patch_normalization(self) -> None:
        torch.manual_seed(3)
        wide = torch.rand(1, 1, 31, 37)
        same = compute_rejection_mask(wide, wide, patch_size=8, stride=4)
        shifted = torch.roll(wide, shifts=3, dims=-1)
        different = compute_rejection_mask(wide, shifted, patch_size=8, stride=4)
        self.assertLess(float(same.max()), 1e-6)
        self.assertGreater(float(different.mean()), 0.5)

    def test_dynamic_adaptive_blending(self) -> None:
        wide = torch.zeros(1, 1, 7, 9)
        fusion = torch.ones_like(wide)
        occ = torch.zeros_like(wide)
        reject = torch.full_like(wide, 0.25)
        final, blend = adaptive_blend(
            fusion,
            wide,
            {"occlusion": occ, "rejection": reject, "flow_uncertainty": None, "defocus": None},
            smoothing="none",
        )
        self.assertTrue(torch.allclose(blend, torch.full_like(blend, 0.75)))
        self.assertTrue(torch.allclose(final, blend))

    def test_unet_preserves_arbitrary_size_and_safe_identity(self) -> None:
        model = FusionUNet(in_channels=3, base_channels=4, mode="residual")
        inputs = torch.rand(1, 3, 35, 53)
        output = model(inputs)
        self.assertEqual(tuple(output.shape), (1, 1, 35, 53))
        self.assertTrue(torch.allclose(output, inputs[:, :1], atol=1e-7))


class PipelineTests(unittest.TestCase):
    def test_hybrid_model_returns_all_research_outputs(self) -> None:
        config = {
            "image": {"height": 35, "width": 53, "keep_aspect_ratio": False},
            "fusion": {"base_channels": 4, "mode": "residual", "use_rejection_input": True},
            "occlusion": {"mode": "hard", "threshold": 0.1, "temperature": 0.1},
            "rejection": {
                "patch_size": 8,
                "stride": 4,
                "metric": "normalized_l1",
                "threshold": 0.25,
                "temperature": 0.05,
            },
            "blending": {"smoothing": "none", "sigma": 1.0},
        }
        model = HybridZoomModel(config, flow_estimator=ZeroFlow()).eval()
        wide = torch.rand(2, 3, 31, 47)
        tele = wide.clone()
        outputs = model(wide, tele)
        required = {
            "output", "wide", "tele", "flow_w2t", "flow_t2w", "warped_tele",
            "occlusion_mask", "rejection_mask", "blend_mask", "fusion_y",
        }
        self.assertTrue(required.issubset(outputs))
        self.assertEqual(tuple(outputs["output"].shape), (2, 3, 35, 53))
        self.assertTrue(torch.allclose(outputs["output"], outputs["wide"], atol=2e-5))


if __name__ == "__main__":
    unittest.main()

