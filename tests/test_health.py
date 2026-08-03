import os
import unittest
import torch
import numpy as np
from unittest.mock import MagicMock
from monitoring.infrastructure.health_monitor import PipelineHealthMonitor


class TestPipelineHealthMonitor(unittest.TestCase):

    def setUp(self):
        # Create a mock database path
        self.db_path = "test_fake.db"
        self.monitor = PipelineHealthMonitor(db_path=self.db_path)

    def test_get_system_metrics(self):
        """Verifies system resource query functions return expected hardware properties."""
        metrics = self.monitor.get_system_metrics()
        self.assertIn("cpu_cores", metrics)
        self.assertIn("cpu_threads", metrics)
        self.assertIn("ram_total_gb", metrics)
        self.assertIn("ram_usage_percent", metrics)
        self.assertIn("disk_usage_percent", metrics)
        self.assertIn("cuda_available", metrics)

    def test_verify_models_with_none(self):
        """Checks model validator flags when references are not loaded."""
        cls_h, seg_h = self.monitor.verify_models(None, None, "cpu")
        self.assertIn("Not Loaded", cls_h)
        self.assertIn("Not Loaded", seg_h)

    def test_verify_models_forward_pass_mocked(self):
        """Tests model verification forward passes using lightweight mock objects."""
        mock_cls = MagicMock()
        mock_cls.named_parameters.return_value = [("weights", torch.tensor([0.5, -0.2]))]
        mock_cls.return_value = torch.zeros(1, 4)

        mock_seg = MagicMock()
        mock_seg.named_parameters.return_value = [("weights", torch.tensor([0.1, 0.8]))]
        mock_seg.return_value = torch.zeros(1, 1, 256, 256)

        cls_h, seg_h = self.monitor.verify_models(mock_cls, mock_seg, "cpu")
        self.assertEqual(cls_h, "HEALTHY")
        self.assertEqual(seg_h, "HEALTHY")

    def test_verify_components_dependencies(self):
        """Verifies imports and configurations check succeeds."""
        statuses = self.monitor.verify_components(None, "cpu")
        self.assertIn("gradcam", statuses)
        self.assertIn("pdf_gen", statuses)
        self.assertIn("sqlite", statuses)
        self.assertEqual(statuses["pdf_gen"], "HEALTHY")  # Should pass if ReportLab is installed


if __name__ == "__main__":
    unittest.main()
