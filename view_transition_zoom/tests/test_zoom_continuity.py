import os
import sys
import unittest

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fusion.tone_matching import interpolate_affine_tone_pair
from view_transition.flow_transform import interpolate_transformation
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
        self.assertEqual(estimator.calls, 1)

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
