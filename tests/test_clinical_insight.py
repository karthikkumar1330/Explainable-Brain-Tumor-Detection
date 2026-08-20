import unittest
from clinical_insight.domain.entities import ClinicalInsight
from clinical_insight.application.use_cases import GenerateClinicalInsightUseCase


class TestClinicalInsightGenerator(unittest.TestCase):
    def setUp(self) -> None:
        self.use_case = GenerateClinicalInsightUseCase()

    def test_generate_clinical_insight_no_tumor(self):
        """Verifies insight generator handles healthy brain scan diagnostics correctly."""
        insight = self.use_case.execute(
            predicted_class="No Tumor",
            confidence_score=0.985,
            is_calibrated=True,
            probabilities={"No Tumor": 0.985, "Glioma": 0.015},
            tumor_area_mm2=0.0,
            pixel_count=0,
            solidity=None,
            circularity=None,
            xai_method="gradcam",
            xai_overlap_percentage=0.0
        )
        self.assertIsInstance(insight, ClinicalInsight)
        self.assertTrue("no focal" in insight.summary_narrative.lower())
        self.assertTrue("calibrated" in insight.summary_narrative.lower())
        self.assertEqual(len(insight.key_findings), 3)
        self.assertTrue("no focal" in insight.key_findings[0].lower())
        self.assertEqual(len(insight.recommendations), 2)
        self.assertTrue("routine" in insight.recommendations[1].lower())

    def test_generate_clinical_insight_with_tumor(self):
        """Verifies insight generator synthesizes correct morphology border details for positive scan."""
        insight = self.use_case.execute(
            predicted_class="Glioma",
            confidence_score=0.924,
            is_calibrated=True,
            probabilities={"Glioma": 0.924, "No Tumor": 0.076},
            tumor_area_mm2=34.20,
            pixel_count=342,
            solidity=0.74,     # irregular
            circularity=0.35,  # elongated
            xai_method="gradcam_plus_plus",
            xai_overlap_percentage=0.885
        )
        self.assertIsInstance(insight, ClinicalInsight)
        self.assertTrue("glioma" in insight.summary_narrative.lower())
        self.assertTrue("irregular" in insight.summary_narrative.lower())
        self.assertTrue("elongated" in insight.summary_narrative.lower())
        self.assertTrue(any("irregular" in f.lower() for f in insight.key_findings))
        self.assertTrue(any("elongated" in f.lower() for f in insight.key_findings))
        self.assertTrue(any("gradcam_plus_plus" in f.lower() for f in insight.key_findings))

    def test_generate_clinical_insight_tumor_but_zero_pixels(self):
        """Verifies insight generator handles classification predicting tumor but segmentation having 0 pixels correctly."""
        insight = self.use_case.execute(
            predicted_class="Pituitary",
            confidence_score=0.4297,
            is_calibrated=False,
            probabilities={"Pituitary": 0.4297, "No Tumor": 0.5703},
            tumor_area_mm2=0.0,
            pixel_count=0,
            solidity=None,
            circularity=None,
            xai_method="gradcam",
            xai_overlap_percentage=0.0
        )
        self.assertIsInstance(insight, ClinicalInsight)
        self.assertIn("Pituitary", insight.summary_narrative)
        self.assertNotIn("active space-occupying lesion", insight.summary_narrative.lower())
        self.assertIn("No segmented tumor region was identified in the analyzed image.", insight.summary_narrative)
        self.assertTrue(any("No segmented tumor region was identified" in f for f in insight.key_findings))
        for r in insight.recommendations:
            self.assertNotIn("staging", r.lower())
            self.assertNotIn("neurosurg", r.lower())
            self.assertNotIn("oncolog", r.lower())
