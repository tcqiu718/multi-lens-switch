"""Fast tests for Dataset, losses, image utilities, and checkpoints."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from hybrid_zoom.datasets import HybridZoomDataset
from hybrid_zoom.losses import ContextualLoss, TotalLoss
from hybrid_zoom.utils import (
    flow_to_color,
    load_checkpoint,
    make_colorwheel,
    psnr,
    read_image,
    save_checkpoint,
    save_image,
    save_pipeline_outputs,
    ssim,
)


class DatasetTests(unittest.TestCase):
    def test_pairing_native_wide_and_normalized_gt_crop(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for modality in ("wide", "tele", "gt"):
                (root / "train" / modality).mkdir(parents=True)
            wide = np.zeros((80, 100, 3), dtype=np.uint8)
            wide[20:60, 25:75] = (20, 180, 40)
            gt = np.zeros((160, 200, 3), dtype=np.uint8)
            gt[40:120, 50:150] = (20, 180, 40)
            tele = np.full((50, 60, 3), 100, dtype=np.uint8)
            Image.fromarray(wide).save(root / "train" / "wide" / "000001.png")
            Image.fromarray(tele).save(root / "train" / "tele" / "000001.png")
            Image.fromarray(gt).save(root / "train" / "gt" / "000001.png")

            dataset = HybridZoomDataset(
                root,
                split="train",
                image_size=(32, 48),
                augment=False,
                require_gt=True,
                wide_crop_size=(40, 50),
                crop_gt_with_wide=True,
            )
            sample = dataset[0]
            self.assertEqual(sample["name"], "000001.png")
            self.assertEqual(tuple(sample["wide"].shape), (3, 32, 48))
            self.assertEqual(tuple(sample["tele"].shape), (3, 32, 48))
            self.assertEqual(tuple(sample["gt"].shape), (3, 32, 48))
            self.assertLess(float((sample["wide"] - sample["gt"]).abs().mean()), 2e-3)


class LossTests(unittest.TestCase):
    def test_contextual_and_total_loss_backward(self) -> None:
        prediction_features = torch.rand(1, 8, 7, 9, requires_grad=True)
        target_features = torch.rand_like(prediction_features)
        contextual = ContextualLoss(max_samples=32)(prediction_features, target_features)
        self.assertTrue(torch.isfinite(contextual))
        contextual.backward()
        self.assertTrue(torch.isfinite(prediction_features.grad).all())

        prediction = torch.rand(1, 3, 24, 28, requires_grad=True)
        target = torch.rand_like(prediction)
        criterion = TotalLoss(
            {
                "vgg_weight": 0.0,
                "contextual_weight": 0.0,
                "brightness_weight": 1.0,
                "brightness_sigma": 2.0,
                "vgg_weights": None,
            }
        )
        losses = criterion(prediction, target)
        losses["total"].backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())


class UtilityTests(unittest.TestCase):
    def test_image_metrics_flow_and_pipeline_saving(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = torch.rand(3, 19, 23)
            image_path = save_image(image, root / "roundtrip.png")
            loaded = read_image(image_path)
            self.assertEqual(tuple(loaded.shape), (3, 19, 23))
            self.assertGreaterEqual(float(psnr(loaded, loaded)), 100.0)
            self.assertAlmostEqual(float(ssim(loaded, loaded)), 1.0, places=5)
            black = torch.zeros(1, 3, 19, 23)
            self.assertAlmostEqual(float(ssim(black, black)), 1.0, places=6)

            flow = torch.zeros(2, 2, 19, 23)
            color = flow_to_color(flow)
            self.assertEqual(tuple(color.shape), (2, 3, 19, 23))
            self.assertTrue(torch.allclose(color, torch.ones_like(color)))
            self.assertEqual(tuple(make_colorwheel().shape), (55, 3))

            outputs = {
                "output": torch.rand(2, 3, 19, 23),
                "warped_tele": torch.rand(2, 3, 19, 23),
                "fusion_y": torch.rand(2, 1, 19, 23),
                "fusion_rgb": torch.rand(2, 3, 19, 23),
                "flow_w2t": flow,
                "occlusion_mask": torch.rand(2, 1, 19, 23),
                "rejection_mask": torch.rand(2, 1, 19, 23),
                "blend_mask": torch.rand(2, 1, 19, 23),
            }
            saved = save_pipeline_outputs(
                outputs, root / "results", names=("../sample.one.jpg", "sample.two.png")
            )
            self.assertIn("fusion_rgb", saved)
            self.assertTrue((root / "results" / "final" / "sample.one.png").is_file())
            fusion_path = root / "results" / "fusion" / "sample.one.png"
            with Image.open(fusion_path) as fusion_image:
                self.assertEqual(fusion_image.mode, "RGB")
            for relative in (
                "warped_tele", "flow", "masks/occlusion", "masks/rejection", "masks/blend"
            ):
                self.assertTrue((root / "results" / relative).is_dir())

    def test_checkpoint_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.pth"
            torch.manual_seed(9)
            model = nn.Conv2d(3, 2, 1)
            expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            save_checkpoint(path, model, optimizer=optimizer, epoch=4, config={"name": "test"})
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
            restored = load_checkpoint(path, model, optimizer=optimizer)
            self.assertEqual(restored["epoch"], 4)
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, expected[key]))


if __name__ == "__main__":
    unittest.main()
