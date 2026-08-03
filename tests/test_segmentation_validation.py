import unittest
import numpy as np
import cv2
from monitoring.infrastructure.segmentation_validator import SegmentationValidator


class TestSegmentationValidator(unittest.TestCase):

    def setUp(self):
        self.validator = SegmentationValidator(tiny_threshold_px=15, huge_brain_ratio=0.45)

    def test_empty_mask_warning(self):
        """Verifies completely blank masks generate an empty warning."""
        mask = np.zeros((256, 256), dtype=np.uint8)
        report = self.validator.validate_mask(mask, (256, 256))
        self.assertTrue(report.is_valid)
        self.assertEqual(report.pixel_count, 0)
        self.assertEqual(report.num_components, 0)
        self.assertTrue(any("empty" in w.lower() for w in report.warnings))

    def test_tiny_mask_warning(self):
        """Checks that tiny clusters of pixels raise warning alerts."""
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[100:103, 100:103] = 255  # 9 pixels (<15 px)
        report = self.validator.validate_mask(mask, (256, 256))
        self.assertTrue(report.is_valid)
        self.assertEqual(report.pixel_count, 9)
        self.assertTrue(any("extremely small" in w.lower() for w in report.warnings))

    def test_huge_mask_leakage_warning(self):
        """Checks that large masks triggering leak thresholds generate warnings."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[10:90, 10:90] = 255  # 6400 pixels out of 10000 (64% brain area)
        report = self.validator.validate_mask(mask, (100, 100), brain_pixels=10000)
        self.assertTrue(report.is_valid)
        self.assertTrue(any("occupancy" in w.lower() or "leakage" in w.lower() for w in report.warnings))

    def test_invalid_dimensions(self):
        """Verifies that dimension mismatch flags mask as invalid."""
        mask = np.zeros((128, 128), dtype=np.uint8)
        report = self.validator.validate_mask(mask, (256, 256))
        self.assertFalse(report.is_valid)
        self.assertTrue(any("dimension mismatch" in w.lower() for w in report.warnings))

    def test_invalid_pixel_values(self):
        """Verifies that non-binary pixel values (e.g. grayscale values) are flagged."""
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[10, 10] = 128  # Grey value
        report = self.validator.validate_mask(mask, (256, 256))
        self.assertFalse(report.is_valid)
        self.assertTrue(any("non-binary" in w.lower() for w in report.warnings))

    def test_disconnected_components_warning(self):
        """Verifies that multiple disconnected blobs trigger multi-focal warnings."""
        mask = np.zeros((256, 256), dtype=np.uint8)
        # Draw 3 separate circles
        cv2.circle(mask, (50, 50), 10, 255, -1)
        cv2.circle(mask, (150, 150), 10, 255, -1)
        cv2.circle(mask, (200, 200), 10, 255, -1)
        
        report = self.validator.validate_mask(mask, (256, 256))
        self.assertTrue(report.is_valid)
        self.assertEqual(report.num_components, 3)
        self.assertTrue(any("disconnected" in w.lower() for w in report.warnings))


if __name__ == "__main__":
    unittest.main()
