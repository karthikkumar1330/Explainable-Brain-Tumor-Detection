import unittest
import numpy as np
from monitoring.infrastructure.explainability_validator import ExplainabilityValidator


class TestExplainabilityValidator(unittest.TestCase):

    def setUp(self):
        self.validator = ExplainabilityValidator(min_overlap_ratio=0.10, max_distance_mm=50.0)

    def test_heatmap_nan_inf_warning(self):
        """Verifies that NaNs or Infs inside heatmaps are flagged invalid."""
        heatmap = np.zeros((10, 10))
        heatmap[3, 3] = np.nan
        is_valid, warnings = self.validator.validate_explanation(heatmap, None, None)
        self.assertFalse(is_valid)
        self.assertTrue(any("invalid numerical" in w.lower() for w in warnings))

    def test_low_overlap_warning(self):
        """Checks that overlap ratios below threshold trigger warnings."""
        heatmap = np.ones((10, 10))
        tumor_mask = np.ones((10, 10))
        is_valid, warnings = self.validator.validate_explanation(
            heatmap=heatmap,
            tumor_mask=tumor_mask,
            overlap_percentage=0.05  # < 0.10 threshold
        )
        self.assertTrue(is_valid)  # Low overlap is warning-only, not structural failure
        self.assertTrue(any("low explainability" in w.lower() for w in warnings))

    def test_centroid_distance_warning(self):
        """Checks spatial gap between tumor centroid and activation centroid triggers warnings."""
        heatmap = np.zeros((100, 100))
        heatmap[10, 10] = 1.0  # Activation is top-left
        
        tumor_mask = np.zeros((100, 100))
        tumor_mask[80:90, 80:90] = 1  # Tumor is bottom-right (center is 85,85)
        
        # Distance between (10,10) and (85,85) is sqrt(75^2 + 75^2) = 106 pixels
        # 106 px * 1.0 spacing = 106 mm (> 50.0 mm max limit)
        is_valid, warnings = self.validator.validate_explanation(
            heatmap=heatmap,
            tumor_mask=tumor_mask,
            overlap_percentage=0.20,
            pixel_spacing_mm=1.0
        )
        self.assertTrue(is_valid)
        self.assertTrue(any("centroid is" in w.lower() and "away from" in w.lower() for w in warnings))


if __name__ == "__main__":
    unittest.main()
