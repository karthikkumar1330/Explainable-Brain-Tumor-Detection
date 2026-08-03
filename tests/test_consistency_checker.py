import unittest
from monitoring.infrastructure.consistency_checker import ConfidenceConsistencyChecker


class TestConfidenceConsistencyChecker(unittest.TestCase):

    def setUp(self):
        self.checker = ConfidenceConsistencyChecker(
            high_conf_threshold=0.85,
            min_tumor_area_mm2=5.0,
            false_neg_pixels=100
        )

    def test_consistent_report(self):
        """Verifies that matching tumor prediction and segmentation area returns consistent status."""
        report = self.checker.check_consistency(
            predicted_class="Glioma",
            confidence_score=0.90,
            is_calibrated=True,
            uncalibrated_confidence=0.90,
            tumor_area_mm2=50.0,
            pixel_count=1000
        )
        self.assertTrue(report.is_consistent)
        self.assertEqual(len(report.warnings), 0)

    def test_false_positive_risk_inconsistent(self):
        """Verifies warning is raised for high confidence of tumor but zero mask pixels."""
        report = self.checker.check_consistency(
            predicted_class="Glioma",
            confidence_score=0.95,
            is_calibrated=True,
            uncalibrated_confidence=0.95,
            tumor_area_mm2=0.0,
            pixel_count=0
        )
        self.assertFalse(report.is_consistent)
        self.assertTrue(any("false alarm" in w.lower() or "detection failure" in w.lower() for w in report.warnings))

    def test_false_negative_risk_inconsistent(self):
        """Verifies warning is raised when 'No Tumor' is predicted but mask outlines a large tumor."""
        report = self.checker.check_consistency(
            predicted_class="No Tumor",
            confidence_score=0.92,
            is_calibrated=True,
            uncalibrated_confidence=0.92,
            tumor_area_mm2=60.0,
            pixel_count=200
        )
        self.assertFalse(report.is_consistent)
        self.assertTrue(any("false negative" in w.lower() for w in report.warnings))

    def test_calibration_discrepancy_warning(self):
        """Checks warning triggers when platt calibration shifts raw model score significantly."""
        report = self.checker.check_consistency(
            predicted_class="Glioma",
            confidence_score=0.60,
            is_calibrated=True,
            uncalibrated_confidence=0.90,  # 0.30 difference (>0.25 threshold)
            tumor_area_mm2=40.0,
            pixel_count=800
        )
        self.assertTrue(report.is_consistent)  # Calib scaling doesn't invalidate clinical consensus
        self.assertTrue(any("calibration scaling" in w.lower() for w in report.warnings))

    def test_narrow_margin_clinical_ambiguity(self):
        """Verifies clinical warning is raised when margin between top classes is < 10%."""
        probs = {"Glioma": 0.46, "Meningioma": 0.43, "Pituitary": 0.08, "No Tumor": 0.03}
        report = self.checker.check_consistency(
            predicted_class="Glioma",
            confidence_score=0.46,
            is_calibrated=False,
            uncalibrated_confidence=None,
            tumor_area_mm2=30.0,
            pixel_count=600,
            probabilities=probs
        )
        self.assertTrue(report.is_consistent)
        self.assertTrue(any("ambiguity" in w.lower() or "margin" in w.lower() for w in report.warnings))


if __name__ == "__main__":
    unittest.main()
