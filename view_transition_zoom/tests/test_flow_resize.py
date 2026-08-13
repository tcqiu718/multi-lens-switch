import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.flow_utils import resize_flow, same_box_filter


class FlowResizeTest(unittest.TestCase):
    def test_components_scale_with_axes(self):
        flow = torch.empty(1, 2, 10, 20)
        flow[:, 0] = 2.0
        flow[:, 1] = 3.0

        resized = resize_flow(flow, (20, 10))

        self.assertTrue(torch.allclose(resized[:, 0], torch.full_like(resized[:, 0], 1.0)))
        self.assertTrue(torch.allclose(resized[:, 1], torch.full_like(resized[:, 1], 6.0)))

    def test_even_box_filter_preserves_shape_and_constant(self):
        value = torch.full((1, 2, 17, 19), 4.0)
        result = same_box_filter(value, 6)
        self.assertEqual(result.shape, value.shape)
        self.assertTrue(torch.allclose(result, value))

    def test_integral_box_filter_matches_pooling_reference(self):
        import torch.nn.functional as functional

        value = torch.rand(1, 2, 11, 13)
        for kernel in (5, 6):
            left = (kernel - 1) // 2
            right = kernel // 2
            reference = functional.avg_pool2d(
                functional.pad(value, (left, right, left, right), mode="replicate"),
                kernel,
                stride=1,
            )
            self.assertTrue(torch.allclose(same_box_filter(value, kernel), reference, atol=2.0e-6))


if __name__ == "__main__":
    unittest.main()
