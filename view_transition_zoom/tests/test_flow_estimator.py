import os
import sys
import unittest
from unittest import mock

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.flow_estimator import FlowEstimator, _TorchvisionRaft, _extract_flow, _extract_flowformer_flow


class FlowFormerAdapterTest(unittest.TestCase):
    def test_eval_tuple_selects_full_resolution_first_item(self):
        full = torch.ones(1, 2, 32, 48)
        low = torch.zeros(1, 2, 4, 6)
        self.assertIs(_extract_flowformer_flow((full, low)), full)

    def test_training_prediction_list_selects_last_iteration(self):
        first = torch.zeros(1, 2, 32, 48)
        final = torch.ones(1, 2, 32, 48)
        self.assertIs(_extract_flowformer_flow([first, final]), final)


class _FakeRaft(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs = None

    def forward(self, first, second):
        self.inputs = (first, second)
        initial = torch.zeros(first.shape[0], 2, *first.shape[-2:], device=first.device)
        final = torch.ones_like(initial)
        return [initial, final]


class TorchvisionRaftAdapterTest(unittest.TestCase):
    def test_prediction_list_selects_final_iteration(self):
        first = torch.zeros(1, 2, 32, 48)
        final = torch.ones(1, 2, 32, 48)
        self.assertIs(_extract_flow([first, final]), final)

    def test_preprocessing_padding_and_output_resize(self):
        fake = _FakeRaft()
        with mock.patch("torchvision.models.optical_flow.raft_small", return_value=fake) as constructor:
            estimator = _TorchvisionRaft(
                weights="none",
                variant="small",
                input_size=[64, 96],
                progress=False,
            )
        wide = torch.rand(1, 3, 45, 79)
        tele = torch.rand_like(wide)
        flow = estimator(wide, tele)

        constructor.assert_called_once_with(weights=None, progress=False)
        self.assertEqual(tuple(fake.inputs[0].shape), (1, 3, 128, 128))
        self.assertGreaterEqual(float(fake.inputs[0].min()), -1.0)
        self.assertLessEqual(float(fake.inputs[0].max()), 1.0)
        self.assertEqual(tuple(flow.shape), (1, 2, 45, 79))
        self.assertAlmostEqual(float(flow[:, 0].mean()), 79.0 / 96.0, places=5)
        self.assertAlmostEqual(float(flow[:, 1].mean()), 45.0 / 64.0, places=5)

    @mock.patch("models.flow_estimator._TorchvisionRaft")
    def test_config_accepts_explicit_torchvision_backend(self, raft_adapter):
        raft_adapter.return_value = nn.Identity()
        estimator = FlowEstimator.from_config(
            {
                "flow": {
                    "model": "torchvision_raft_small",
                    "fallback": None,
                    "raft_weights": "none",
                    "raft_input_size": [256, 448],
                    "raft_progress": False,
                }
            },
            device=torch.device("cpu"),
        )

        self.assertEqual(estimator.backend_name, "torchvision_raft_small")
        raft_adapter.assert_called_once_with(
            weights="none",
            variant="small",
            input_size=[256, 448],
            progress=False,
            minimum_size=128,
            pad_multiple=8,
        )


if __name__ == "__main__":
    unittest.main()
