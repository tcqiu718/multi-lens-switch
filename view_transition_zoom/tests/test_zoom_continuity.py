import os
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusion.occlusion import OcclusionResult
from fusion.tone_matching import interpolate_affine_tone_pair
from view_transition.flow_transform import interpolate_transformation
from zoom.continuous_view import bidirectional_zoom_transition
from zoom.fov_transform import center_crop_fov
from zoom.zoom_schedule import ZoomSchedule
from zoom.zoom_pipeline import ContinuousZoomPipeline


class CountingEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, wide, tele):
        self.calls += 1
        return torch.zeros(wide.shape[0], 2, wide.shape[2], wide.shape[3])


class ZoomContinuityTest(unittest.TestCase):
    def test_static_pair_flow_cache(self):
        estimator = CountingEstimator()
        config = {
            "flow": {"cache": True},
            "target_flow": {"kernel_size": 5},
            "boundary": {"mode": "simple"},
            "view_transition": {"ratio": 0.01},
            "occlusion": {"mode": "paper"},
            "tone": {"mode": "none"},
            "blend": {"pyramid_levels": 3},
            "camera": {"wide_zoom": 1.0, "tele_zoom": 3.0},
            "zoom": {},
            "temporal": {"enabled": False},
        }
        pipeline = ContinuousZoomPipeline(config, torch.device("cpu"), estimator)
        wide = torch.rand(1, 3, 24, 32)
        tele = torch.rand_like(wide)
        first = pipeline.prepare_pair(wide, tele)
        second = pipeline.prepare_pair(wide, tele)
        self.assertIs(first, second)
        self.assertEqual(estimator.calls, 2)

    def test_schedule_endpoints_and_monotonicity(self):
        schedule = ZoomSchedule(1.0, 3.0, interpolation="log", curve="smootherstep")
        points = schedule.sample(1.0, 3.0, 101)
        alpha = np.asarray([point.alpha for point in points])
        self.assertAlmostEqual(points[0].alpha, 0.0, places=8)
        self.assertAlmostEqual(points[-1].alpha, 1.0, places=8)
        self.assertTrue(np.all(np.diff(alpha) >= -1.0e-10))
        self.assertLess(abs(alpha[1] - alpha[0]), 1.0e-4)
        self.assertLess(abs(alpha[-1] - alpha[-2]), 1.0e-4)

    def test_only_native_zoom_values_are_endpoints(self):
        schedule = ZoomSchedule(
            1.0,
            3.0,
            interpolation="linear",
            curve="linear",
        )
        self.assertTrue(schedule(1.0).is_wide_endpoint)
        self.assertFalse(schedule(1.0).is_tele_endpoint)
        self.assertFalse(schedule(2.9).is_wide_endpoint)
        self.assertFalse(schedule(2.9).is_tele_endpoint)
        self.assertTrue(schedule(3.0).is_tele_endpoint)
        self.assertAlmostEqual(schedule(2.0).tone_progress, 0.5, places=8)

    def test_affine_tone_transition_has_exact_tone_endpoints(self):
        tele = torch.full((1, 3, 4, 5), 0.2)
        wide = torch.full_like(tele, 0.5)
        gain = torch.full_like(tele, 2.0)
        bias = torch.full_like(tele, 0.1)
        start = interpolate_affine_tone_pair(tele, wide, wide, gain, bias, 0.0)
        middle = interpolate_affine_tone_pair(tele, wide, wide, gain, bias, 0.5)
        end = interpolate_affine_tone_pair(tele, wide, wide, gain, bias, 1.0)
        self.assertTrue(torch.allclose(start[0], wide))
        self.assertTrue(torch.allclose(start[1], wide))
        self.assertTrue(torch.allclose(middle[0], torch.full_like(tele, 0.35)))
        self.assertTrue(torch.allclose(middle[1], torch.full_like(tele, 0.35)))
        self.assertTrue(torch.allclose(end[0], tele))
        self.assertTrue(torch.allclose(end[1], tele))

    def test_render_uses_exact_camera_frames_only_at_endpoints(self):
        estimator = CountingEstimator()
        config = {
            "flow": {"cache": True},
            "target_flow": {"kernel_size": 5},
            "boundary": {"mode": "simple"},
            "view_transition": {"ratio": 0.01},
            "occlusion": {"mode": "paper", "soft_width": 2},
            "tone": {"mode": "none"},
            "blend": {"pyramid_levels": 2, "overlap_soft_width": 4},
            "camera": {"wide_zoom": 1.0, "tele_zoom": 3.0},
            "zoom": {"interpolation": "linear", "schedule": "linear"},
            "temporal": {"enabled": False},
        }
        pipeline = ContinuousZoomPipeline(config, torch.device("cpu"), estimator)
        wide = torch.full((1, 3, 24, 32), 0.2)
        tele = torch.full_like(wide, 0.8)
        pair = pipeline.prepare_pair(wide, tele)
        start = pipeline.render(pair, 1.0)
        middle = pipeline.render(pair, 2.9)
        end = pipeline.render(pair, 3.0)
        self.assertTrue(torch.equal(start.result, wide))
        self.assertFalse(torch.equal(middle.result, tele))
        self.assertTrue(torch.equal(end.result, tele))
        self.assertEqual(start.tele_usage_ratio, 0.0)
        self.assertEqual(end.tele_usage_ratio, 1.0)

    def test_bidirectional_blend_respects_both_camera_validity_masks(self):
        wide = torch.full((2, 3, 8, 16), 0.2)
        tele = torch.full_like(wide, 0.8)
        flow = torch.zeros(2, 2, 8, 16)
        overlap = torch.ones_like(wide[:, :1])
        # Columns cover Wide-only, Tele-only, both-valid, and both-invalid.
        wide_valid = overlap.clone()
        wide_valid[..., 4:8] = 0.0
        wide_valid[..., 12:] = 0.0
        tele_valid = overlap.clone()
        tele_valid[..., :4] = 0.0
        tele_valid[..., 12:] = 0.0
        view = bidirectional_zoom_transition(
            wide,
            tele,
            flow,
            flow,
            overlap,
            alpha=0.8,
            target_zoom=1.0,
            wide_zoom=1.0,
            tele_zoom=1.0,
            fov_progress=0.8,
            multi_warp_average=False,
        )
        view.wide.image = wide * wide_valid
        view.wide.valid_mask = wide_valid
        view.tele.image = tele * tele_valid
        view.tele.valid_mask = tele_valid
        occlusion = OcclusionResult(torch.zeros_like(overlap), overlap * 0.25)

        for mask_smoothing in (False, True):
            for levels in (1, 3):
                with self.subTest(mask_smoothing=mask_smoothing, levels=levels):
                    config = {
                        "target_flow": {"kernel_size": 5},
                        "boundary": {"mode": "simple"},
                        "transform": {"multi_warp_average": False},
                        "tone": {"mode": "none"},
                        "blend": {
                            "pyramid_levels": levels,
                            "overlap_soft_width": 0,
                            "viewpoint_occlusion_strength": 0.5,
                        },
                        "camera": {"wide_zoom": 1.0, "tele_zoom": 3.0},
                        "zoom": {"interpolation": "linear", "schedule": "linear"},
                        "temporal": {"enabled": True, "mask_smoothing": mask_smoothing},
                    }
                    pipeline = ContinuousZoomPipeline(config, torch.device("cpu"))
                    pair = pipeline.prepare_pair(wide, tele, flow, flow)
                    pipeline.temporal.previous_occlusion = overlap * 0.9
                    pipeline.temporal.previous_overlap = overlap.clone()
                    with patch.object(pipeline, "_bidirectional_view", return_value=view), \
                            patch.object(pipeline, "_occlusion", return_value=occlusion):
                        result = pipeline.render(pair, 2.6)

                    mask = result.fusion.occlusion_mask
                    alpha = result.zoom_point.alpha
                    expected = (1.0 - alpha) + 0.5 * alpha * (1.0 - alpha) * result.occlusion.soft_mask
                    self.assertTrue(torch.equal(mask[..., :4], torch.ones_like(mask[..., :4])))
                    self.assertTrue(torch.equal(mask[..., 4:8], torch.zeros_like(mask[..., 4:8])))
                    self.assertTrue(torch.allclose(mask[..., 8:], expected[..., 8:]))
                    self.assertTrue(torch.isfinite(result.result).all())
                    if levels == 1:
                        self.assertTrue(torch.equal(result.result[..., :4], wide[..., :4]))
                        self.assertTrue(torch.equal(result.result[..., 4:8], tele[..., 4:8]))
                        self.assertEqual(float(result.result[..., 12:].abs().max()), 0.0)

    def test_bidirectional_view_aligns_correspondences_at_every_progress(self):
        wide = torch.zeros(1, 1, 9, 24)
        tele = torch.zeros_like(wide)
        wide[..., 4, 5] = 1.0
        tele[..., 4, 9] = 1.0
        forward = torch.zeros(1, 2, 9, 24)
        reverse = torch.zeros_like(forward)
        forward[:, 0] = 4.0
        reverse[:, 0] = -4.0
        overlap = torch.ones_like(wide)
        expected_x = {0.0: 5, 0.25: 6, 0.5: 7, 0.75: 8, 1.0: 9}
        for alpha, x in expected_x.items():
            view = bidirectional_zoom_transition(
                wide,
                tele,
                forward,
                reverse,
                overlap,
                alpha,
                target_zoom=1.0,
                wide_zoom=1.0,
                tele_zoom=1.0,
                fov_progress=alpha,
                splat_mode="nearest",
                multi_warp_average=False,
            )
            wide_peak = torch.nonzero(view.wide.image[0, 0] > 0.5, as_tuple=False)
            tele_peak = torch.nonzero(view.tele.image[0, 0] > 0.5, as_tuple=False)
            self.assertEqual(wide_peak.tolist(), [[4, x]])
            self.assertEqual(tele_peak.tolist(), [[4, x]])

    def test_bidirectional_motion_has_exact_native_geometry_endpoints(self):
        image = torch.rand(1, 3, 12, 20)
        forward = torch.rand(1, 2, 12, 20) - 0.5
        reverse = torch.rand_like(forward) - 0.5
        overlap = torch.ones_like(image[:, :1])
        start = bidirectional_zoom_transition(
            image,
            image,
            forward,
            reverse,
            overlap,
            0.0,
            target_zoom=1.0,
            wide_zoom=1.0,
            tele_zoom=3.0,
            fov_progress=0.0,
            multi_warp_average=False,
        )
        end = bidirectional_zoom_transition(
            image,
            image,
            forward,
            reverse,
            overlap,
            1.0,
            target_zoom=3.0,
            wide_zoom=1.0,
            tele_zoom=3.0,
            fov_progress=1.0,
            multi_warp_average=False,
        )
        self.assertTrue(torch.count_nonzero(start.wide_motion) == 0)
        self.assertTrue(torch.count_nonzero(end.tele_motion) == 0)
        self.assertTrue(torch.equal(start.wide.image, image))
        self.assertTrue(torch.equal(end.tele.image, image))

    def test_target_fov_mapping_does_not_double_apply_optical_zoom(self):
        wide = torch.zeros(1, 1, 9, 21)
        tele = torch.zeros_like(wide)
        wide[..., 4, 12] = 1.0
        tele[..., 4, 16] = 1.0
        x = torch.arange(21, dtype=torch.float32).view(1, 1, 1, 21).expand(1, 1, 9, 21)
        forward = torch.zeros(1, 2, 9, 21)
        reverse = torch.zeros_like(forward)
        forward[:, 0:1] = 2.0 * (x - 10.0)
        reverse[:, 0:1] = -(2.0 / 3.0) * (x - 10.0)
        view = bidirectional_zoom_transition(
            wide,
            tele,
            forward,
            reverse,
            torch.ones_like(wide),
            alpha=0.5,
            target_zoom=2.0,
            wide_zoom=1.0,
            tele_zoom=3.0,
            fov_progress=0.5,
            splat_mode="nearest",
            multi_warp_average=False,
        )
        wide_peak = torch.nonzero(view.wide.image[0, 0] > 0.5, as_tuple=False)
        tele_peak = torch.nonzero(view.tele.image[0, 0] > 0.5, as_tuple=False)
        self.assertEqual(wide_peak.tolist(), [[4, 14]])
        self.assertEqual(tele_peak.tolist(), [[4, 14]])

    def test_alpha_zero_and_one_have_exact_flow_endpoints(self):
        original = torch.rand(1, 2, 12, 16)
        delta = torch.rand_like(original) - 0.5
        start = interpolate_transformation(original, delta, 0.0)
        end = interpolate_transformation(original, delta, 1.0)
        self.assertTrue(torch.equal(start.transformed_flow, original))
        self.assertTrue(torch.allclose(end.transformed_flow, original + delta))

    def test_native_fov_is_identity(self):
        image = torch.rand(1, 3, 31, 47)
        mapped = center_crop_fov(image, target_zoom=3.0, source_zoom=3.0)
        self.assertTrue(torch.equal(mapped.image, image))
        self.assertEqual(float(mapped.valid_mask.min()), 1.0)

    def test_larger_zoom_crops_toward_center(self):
        image = torch.zeros(1, 1, 9, 9)
        image[..., 3:6, 3:6] = 1.0
        mapped = center_crop_fov(image, target_zoom=3.0, source_zoom=1.0)
        self.assertAlmostEqual(float(mapped.image.mean()), 1.0, places=5)
        self.assertEqual(mapped.crop_box, (3, 3, 6, 6))


if __name__ == "__main__":
    unittest.main()
