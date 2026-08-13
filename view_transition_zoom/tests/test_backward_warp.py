import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.warp import backward_warp


class BackwardWarpTest(unittest.TestCase):
    def test_positive_dx_moves_visible_content_left(self):
        image = torch.zeros(1, 1, 16, 64)
        image[:, :, 8, 30] = 1.0
        flow = torch.zeros(1, 2, 16, 64)
        flow[:, 0] = 20.0

        warped, valid = backward_warp(image, flow, mode="nearest", return_valid=True)

        peak = torch.nonzero(warped[0, 0] > 0.5, as_tuple=False)
        self.assertEqual(peak.tolist(), [[8, 10]])
        self.assertEqual(float(valid[0, 0, 8, 10]), 1.0)
        self.assertEqual(float(valid[0, 0, 8, 50]), 0.0)

    def test_identity_is_exact(self):
        image = torch.rand(2, 3, 9, 11)
        flow = torch.zeros(2, 2, 9, 11)
        warped = backward_warp(image, flow)
        self.assertTrue(torch.allclose(image, warped, atol=1.0e-6))


if __name__ == "__main__":
    unittest.main()

