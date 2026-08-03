import unittest
import numpy as np
from unittest.mock import MagicMock
from monitoring.infrastructure.segmentation_validator import SegmentationValidator
from monitoring.infrastructure.consistency_checker import ConfidenceConsistencyChecker
from monitoring.infrastructure.explainability_validator import ExplainabilityValidator
from monitoring.application.warning_engine import CentralWarningEngine


class TestCentralWarningEngine(unittest.TestCase):

    def setUp(self):
        self.seg_validator = SegmentationValidator(tiny_threshold_px=10, huge_brain_ratio=0.5)
        self.consistency_checker = ConfidenceConsistencyChecker(high_conf_threshold=0.8, min_tumor_area_mm2=5.0)
        self.explain_validator = ExplainabilityValidator(min_overlap_ratio=0.1, max_distance_mm=50.0)
        
        self.engine = CentralWarningEngine(
            seg_validator=self.seg_validator,
            consistency_checker=self.consistency_checker,
            explain_validator=self.explain_validator
        )

    def test_collect_warnings_success_empty(self):
        """Tests standard healthy prediction run aggregates zero warnings."""
        heatmap = np.zeros((10, 10))
        heatmap[2:7, 2:7] = 1.0
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:7, 2:7] = 255
        
        result = self.engine.collect_warnings(
            input_errors=[],
            predicted_class="Glioma",
            confidence_score=0.95,
            is_calibrated=True,
            uncalibrated_confidence=0.95,
            probabilities={"Glioma": 0.95, "No Tumor": 0.05},
            mask=mask,
            expected_shape=(10, 10),
            tumor_area_mm2=10.0,
            heatmap=heatmap,
            overlap_percentage=0.80
        )
        self.assertTrue(result["is_safe"])
        self.assertEqual(len(result["warnings"]), 0)

    def test_collect_warnings_aggregates_multiple(self):
        """Tests that warnings from multiple subsystems accumulate correctly in engine output."""
        heatmap = np.zeros((10, 10))
        # 1. Invalid mask value to trigger segmentation validator error
        mask = np.zeros((10, 10))
        mask[5, 5] = 127 # Grey value
        
        # 2. Ambiguous probabilities to trigger consistency checker warning
        probs = {"Glioma": 0.46, "Meningioma": 0.44}
        
        result = self.engine.collect_warnings(
            input_errors=["Magic number check failed"],
            predicted_class="Glioma",
            confidence_score=0.46,
            is_calibrated=False,
            uncalibrated_confidence=None,
            probabilities=probs,
            mask=mask,
            expected_shape=(10, 10),
            tumor_area_mm2=10.0,
            heatmap=heatmap,
            overlap_percentage=0.20
        )
        
        self.assertFalse(result["is_safe"])
        warnings = result["warnings"]
        self.assertTrue(any("magic number" in w.lower() for w in warnings)) # Input
        self.assertTrue(any("non-binary" in w.lower() for w in warnings))   # Segmentation
        self.assertTrue(any("ambiguity" in w.lower() for w in warnings))    # Consistency


if __name__ == "__main__":
    unittest.main()
