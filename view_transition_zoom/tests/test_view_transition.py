import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from view_transition.flow_boundary import distance_to_boundary
from view_transition.flow_transform import constrain_target_flow
from view_transition.target_flow import estimate_target_flow
from view_transition.view_transition import ViewTransition
from fusion.full_view_fusion import full_view_fusion


def synthetic_flow(height=48, width=80):
    flow = torch.zeros(1, 2, height, width)
    flow[:, 0] = 10.0
    flow[:, 0, 12:36, 25:55] = 30.0
    return flow


class ViewTransitionTest(unittest.TestCase):
    def test_constant_flow_has_nonzero_interior_distance(self):
        flow = torch.zeros(1, 2, 32, 48)
        flow[:, 0] = 10.0
        result = distance_to_boundary(flow, mode="flow_aware", max_extra_distance=64)
        self.assertGreater(float(result.distance.max()), 10.0)
        self.assertEqual(float(result.motion_boundary.sum()), 0.0)

    def test_motion_boundary_limits_region_distance(self):
        flow = synthetic_flow()
        result = distance_to_boundary(
            flow,
            mode="flow_aware",
            gradient_threshold=1.0,
            max_extra_distance=64,
        )
        self.assertGreater(float(result.motion_boundary.sum()), 0.0)
        self.assertTrue(torch.all(result.distance <= result.simple_distance + 1.0e-6))
        self.assertLess(float(result.distance[0, 0, 24, 40]), float(result.simple_distance[0, 0, 24, 40]))

    def test_zero_ratio_exactly_degenerates_to_original_flow(self):
        flow = synthetic_flow()
        target = estimate_target_flow(flow, kernel_size=9)
        distance = distance_to_boundary(flow, mode="simple").distance
        result = constrain_target_flow(flow, target.target_flow, distance, ratio=0.0)
        self.assertTrue(torch.equal(result.transformed_flow, flow))
        self.assertEqual(float(result.delta_flow.abs().max()), 0.0)

    def test_zero_ratio_keeps_wide_transform_exact(self):
        config = {
            "target_flow": {"kernel_size": 9},
            "boundary": {"mode": "simple"},
            "view_transition": {"ratio": 0.0},
            "transform": {"multi_warp_average": True},
        }
        wide = torch.rand(1, 3, 32, 48)
        tele = torch.rand_like(wide)
        flow = torch.zeros(1, 2, 32, 48)
        flow[:, 0] = 10.0
        result = ViewTransition(config)(wide, tele, flow)
        self.assertTrue(torch.equal(result.wide.image, wide))

    def test_constraint_is_componentwise_bounded(self):
        flow = synthetic_flow()
        target = estimate_target_flow(flow, kernel_size=9)
        distance = distance_to_boundary(flow, mode="simple").distance
        ratio = 0.05
        result = constrain_target_flow(flow, target.target_flow, distance, ratio=ratio)
        bound = distance * ratio
        self.assertTrue(torch.all(torch.abs(result.delta_flow) <= bound + 1.0e-6))

    def test_geometry_pipeline_shapes_and_finite_values(self):
        config = {
            "target_flow": {"kernel_size": 9, "foreground_mode": "magnitude"},
            "boundary": {"mode": "flow_aware", "gradient_threshold": 1.0, "max_extra_distance": 32},
            "constraint": {"ratio": 0.05},
            "transform": {
                "splat_mode": "bilinear",
                "fill_mode": "background_nearest",
                "multi_warp_average": False,
            },
        }
        height, width = 48, 80
        wide = torch.rand(1, 3, height, width)
        tele = torch.rand_like(wide)
        result = ViewTransition(config)(wide, tele, synthetic_flow(height, width))
        self.assertEqual(result.tele.image.shape, wide.shape)
        self.assertEqual(result.wide.image.shape, wide.shape)
        self.assertEqual(result.tele.flow_t2o.shape, (1, 2, height, width))
        self.assertTrue(torch.isfinite(result.tele.image).all())
        self.assertTrue(torch.isfinite(result.wide.image).all())

    def test_full_view_falls_back_to_wide_outside_valid_overlap(self):
        wide = torch.rand(1, 3, 32, 48)
        transformed_wide = torch.zeros_like(wide)
        transformed_tele = torch.zeros_like(wide)
        overlap = torch.zeros_like(wide[:, :1])
        occlusion = torch.zeros_like(overlap)
        result = full_view_fusion(
            transformed_wide,
            transformed_tele,
            wide,
            occlusion,
            overlap,
            pyramid_levels=4,
            overlap_soft_width=8,
        )
        self.assertTrue(torch.allclose(result.full_result, wide, atol=1.0e-6))


if __name__ == "__main__":
    unittest.main()
