import os
import sys
import unittest
import tempfile
import json
import csv
import hashlib
import numpy as np
import cv2
import torch
from unittest.mock import patch, MagicMock

# Add root folder to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.external_benchmark import (
    calculate_classification_metrics,
    get_file_sha256,
    evaluate_classification,
    evaluate_segmentation,
    render_markdown_report
)

class TestI3ExternalBenchmark(unittest.TestCase):
    """Phase I3: External Dataset / Generalization Benchmark Verification Tests."""

    def test_calculate_classification_metrics(self):
        """Tests classification metrics accuracy (Specificity, F1, Accuracy, Sensitivity)."""
        # 4 classes, mapping: Glioma (0), Meningioma (1), Pituitary (2), No Tumor (3)
        y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        y_pred = np.array([0, 1, 1, 1, 2, 3, 3, 3])
        y_probs = np.array([
            [0.7, 0.1, 0.1, 0.1], # pred 0 (correct)
            [0.2, 0.6, 0.1, 0.1], # pred 1 (wrong)
            [0.1, 0.8, 0.05, 0.05], # pred 1 (correct)
            [0.1, 0.7, 0.1, 0.1], # pred 1 (correct)
            [0.05, 0.05, 0.8, 0.1], # pred 2 (correct)
            [0.1, 0.1, 0.1, 0.7], # pred 3 (wrong)
            [0.05, 0.05, 0.1, 0.8], # pred 3 (correct)
            [0.1, 0.1, 0.1, 0.7] # pred 3 (correct)
        ])

        metrics = calculate_classification_metrics(y_true, y_pred, y_probs)

        # Total accuracy check: 6 / 8 = 75%
        self.assertEqual(metrics["accuracy"], 0.75)

        # Per class specificity validation
        # Glioma (0): True [0, 0], Pred [0, 1]. TP=1, FP=0, FN=1, TN=6. Specificity = TN/(TN+FP) = 6/6 = 100%
        self.assertEqual(metrics["per_class"]["Glioma"]["specificity"], 1.0)
        self.assertEqual(metrics["per_class"]["Glioma"]["sensitivity"], 0.5)

        # Confusion matrix checks
        cm = np.array(metrics["confusion_matrix"])
        self.assertEqual(cm.shape, (4, 4))
        self.assertEqual(cm[0, 0], 1) # True Glioma predicted Glioma
        self.assertEqual(cm[0, 1], 1) # True Glioma predicted Meningioma

    def test_get_file_sha256(self):
        """Tests checksum function behaves correctly on existing and non-existing files."""
        self.assertEqual(get_file_sha256("non_existent_file_path.pth"), "N/A")

        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"model_weights_bytes_signature")
            temp_path = f.name

        try:
            expected_hash = hashlib.sha256(b"model_weights_bytes_signature").hexdigest()
            self.assertEqual(get_file_sha256(temp_path), expected_hash)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    @patch("scripts.external_benchmark.EfficientNetB0Model")
    @patch("scripts.external_benchmark.PyTorchModelAdapter")
    def test_evaluate_classification_run(self, mock_adapter_cls, mock_model_cls):
        """Tests the classification evaluation function using temp folders."""
        # Setup mock models
        mock_model = MagicMock()
        mock_model_cls.return_value = mock_model

        mock_adapter = MagicMock()
        mock_adapter_cls.return_value = mock_adapter
        mock_adapter.model = MagicMock()

        # Mock adapter forward pass output (8 samples, batch size 4)
        mock_logits = torch.zeros(4, 4)
        mock_logits[0, 0] = 5.0 # Class 0
        mock_logits[1, 1] = 5.0 # Class 1
        mock_logits[2, 2] = 5.0 # Class 2
        mock_logits[3, 3] = 5.0 # Class 3

        mock_adapter.model.side_effect = [mock_logits, mock_logits]

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create class directories
            for name in ["glioma", "meningioma", "pituitary", "no_tumor"]:
                sub = os.path.join(tmpdir, name)
                os.makedirs(sub)
                # Create a blank mock image
                img_path = os.path.join(sub, "sample.jpg")
                cv2.imwrite(img_path, np.zeros((10, 10, 3), dtype=np.uint8))

            metrics = evaluate_classification(
                class_dir=tmpdir,
                checkpoint_path="dummy_path.pt",
                device="cpu",
                output_dir=tmpdir
            )

            self.assertIsNotNone(metrics)
            self.assertEqual(metrics["total_samples"], 4)
            self.assertIn("accuracy", metrics)
            self.assertIn("confusion_matrix", metrics)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "external_confusion_matrix.png")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "external_roc_curves.png")))

    @patch("torch.load")
    @patch("scripts.external_benchmark.archs")
    def test_evaluate_segmentation_run(self, mock_archs, mock_load):
        """Tests the segmentation evaluation function using temp folders."""
        # Mock UNeXt model
        mock_model = MagicMock()
        mock_archs.__dict__ = {"UNext": MagicMock(return_value=mock_model)}

        # Mock model output: batch size 4, 1 class, 256x256 shape
        mock_output = torch.zeros(1, 1, 256, 256)
        # Add simulated positive pixels
        mock_output[0, 0, 10:20, 10:20] = 5.0
        mock_model.return_value = mock_output
        mock_model.to.return_value = mock_model

        with tempfile.TemporaryDirectory() as tmpdir:
            images_dir = os.path.join(tmpdir, "images")
            masks_dir = os.path.join(tmpdir, "masks", "0")
            os.makedirs(images_dir)
            os.makedirs(masks_dir)

            # Create a blank image and a matching mask
            img_path = os.path.join(images_dir, "slice_1.tif")
            mask_path = os.path.join(masks_dir, "slice_1.tif")

            cv2.imwrite(img_path, np.zeros((256, 256, 3), dtype=np.uint8))
            cv2.imwrite(mask_path, np.zeros((256, 256), dtype=np.uint8))

            # Create a mock config.yml
            config_path = os.path.join(tmpdir, "config.yml")
            with open(config_path, "w") as f:
                f.write("arch: UNext\nnum_classes: 1\ninput_channels: 3\ndeep_supervision: false\nimg_ext: .tif\nmask_ext: .tif\ninput_h: 256\ninput_w: 256\n")

            metrics = evaluate_segmentation(
                images_dir=images_dir,
                masks_dir=os.path.join(tmpdir, "masks"),
                checkpoint_path="dummy_seg.pth",
                config_path=config_path,
                device="cpu",
                custom_threshold=0.5
            )

            self.assertIsNotNone(metrics)
            self.assertEqual(metrics["total_samples"], 1)
            self.assertIn("mean_iou", metrics)
            self.assertIn("mean_dice", metrics)

    def test_render_markdown_report(self):
        """Tests that markdown reports render formatted summaries correctly."""
        cls_metrics = {
            "total_samples": 4,
            "accuracy": 1.0,
            "roc_auc_macro": 1.0,
            "roc_auc_weighted": 1.0,
            "per_class": {
                "Glioma": {"sensitivity": 1.0, "specificity": 1.0, "precision": 1.0, "f1_score": 1.0, "support": 1},
                "Meningioma": {"sensitivity": 1.0, "specificity": 1.0, "precision": 1.0, "f1_score": 1.0, "support": 1},
                "Pituitary": {"sensitivity": 1.0, "specificity": 1.0, "precision": 1.0, "f1_score": 1.0, "support": 1},
                "No Tumor": {"sensitivity": 1.0, "specificity": 1.0, "precision": 1.0, "f1_score": 1.0, "support": 1}
            },
            "macro_sensitivity": 1.0,
            "macro_specificity": 1.0,
            "macro_precision": 1.0,
            "macro_f1": 1.0,
            "weighted_sensitivity": 1.0,
            "weighted_specificity": 1.0,
            "weighted_precision": 1.0,
            "weighted_f1": 1.0,
            "confusion_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        }

        seg_metrics = {
            "total_samples": 2,
            "threshold": 0.5,
            "mean_iou": 0.90,
            "mean_dice": 0.95
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = os.path.join(tmpdir, "report.md")
            render_markdown_report(cls_metrics, seg_metrics, "cls_sha256", "seg_sha256", report_path)

            self.assertTrue(os.path.exists(report_path))
            with open(report_path, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn("Clinical AI Generalization Benchmarking Report", content)
            self.assertIn("cls_sha256", content)
            self.assertIn("seg_sha256", content)
            self.assertIn("Overall Classification Accuracy", content)
            self.assertIn("Mean IoU (Jaccard)", content)
            self.assertIn("Generalization Gap", content)

if __name__ == "__main__":
    unittest.main()
