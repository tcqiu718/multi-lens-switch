import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.warp import forward_warp


class ForwardWarpTest(unittest.TestCase):
    def test_positive_dx_moves_content_right(self):
        image = torch.zeros(1, 1, 16, 64)
        image[:, :, 8, 10] = 1.0
        flow = torch.zeros(1, 2, 16, 64)
        flow[:, 0] = 20.0

        warped, weight, valid = forward_warp(image, flow, mode="nearest")

        self.assertEqual(float(warped[0, 0, 8, 30]), 1.0)
        self.assertEqual(float(weight[0, 0, 8, 30]), 1.0)
        self.assertEqual(float(valid[0, 0, 8, 30]), 1.0)

    def test_many_to_one_is_weighted_average(self):
        image = torch.zeros(1, 1, 1, 4)
        image[0, 0, 0, 0] = 1.0
        image[0, 0, 0, 1] = 3.0
        flow = torch.zeros(1, 2, 1, 4)
        flow[0, 0, 0, 0] = 1.0

        warped, weight, _ = forward_warp(image, flow, mode="nearest")

        self.assertAlmostEqual(float(warped[0, 0, 0, 1]), 2.0, places=6)
        self.assertAlmostEqual(float(weight[0, 0, 0, 1]), 2.0, places=6)

    def test_bilinear_half_pixel_splat(self):
        image = torch.zeros(1, 1, 3, 4)
        image[0, 0, 1, 1] = 1.0
        flow = torch.zeros(1, 2, 3, 4)
        flow[0, 0, 1, 1] = 0.5

        _, weight, _ = forward_warp(image, flow, mode="bilinear")

        self.assertAlmostEqual(float(weight[0, 0, 1, 1]), 0.5, places=6)
        self.assertAlmostEqual(float(weight[0, 0, 1, 2]), 1.5, places=6)


if __name__ == "__main__":
    unittest.main()

