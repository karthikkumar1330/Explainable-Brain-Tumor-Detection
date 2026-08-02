import unittest
from severity_assessment.infrastructure.classifier import RuleBasedSeverityClassifier
from severity_assessment.domain.entities import SeverityCategory


class TestSeverityClassifier(unittest.TestCase):
    """Unit tests for rule-based risk classification and clinical disclaimers."""

    def setUp(self):
        self.classifier = RuleBasedSeverityClassifier()

    def test_no_tumor_severity(self):
        """Verify No Tumor returns Low/Normal risk status."""
        res = self.classifier.assess(tumor_type="No Tumor", tumor_area_mm2=0.0, tumor_percentage=0.0)
        self.assertEqual(res.category, SeverityCategory.LOW)
        self.assertIn("No active tumor mass detected", res.rule_description)

    def test_glioma_severity_thresholds(self):
        """Verify risk categorizations adapt to tumor size boundaries."""
        # Low risk: slow-growing Meningioma with small parameters
        res_low = self.classifier.assess(tumor_type="Meningioma", tumor_area_mm2=0.8, tumor_percentage=0.02)
        self.assertEqual(res_low.category, SeverityCategory.LOW)

        # Medium risk: Glioma with small parameters defaults to Medium for safety
        res_med = self.classifier.assess(tumor_type="Glioma", tumor_area_mm2=3.5, tumor_percentage=0.15)
        self.assertEqual(res_med.category, SeverityCategory.MEDIUM)

        # High risk: size > 5.0 mm^2 (or occupancy > 3%)
        res_high = self.classifier.assess(tumor_type="Glioma", tumor_area_mm2=550.0, tumor_percentage=3.5)
        self.assertEqual(res_high.category, SeverityCategory.HIGH)

    def test_educational_disclaimer(self):
        """Verify the educational label is attached to all risk profiles."""
        res = self.classifier.assess(tumor_type="Meningioma", tumor_area_mm2=10.0, tumor_percentage=0.5)
        self.assertIsNotNone(res.educational_disclaimer)
        self.assertIn("educational", res.educational_disclaimer.lower())
        self.assertIn("demonstration", res.educational_disclaimer.lower())
        self.assertIn("not constitute professional medical advice", res.educational_disclaimer.lower())


if __name__ == "__main__":
    unittest.main()
