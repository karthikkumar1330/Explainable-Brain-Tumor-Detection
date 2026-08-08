import unittest
import os
import tempfile
from reportlab.pdfgen import canvas
from clinical_reporting.domain.entities import PatientInfo, ProcessingSummary, ClinicalReport
from clinical_reporting.infrastructure.pdf_generator import ReportLabPDFGenerator
from classification.domain.entities import PredictionResult
from tumor_analysis.domain.entities import TumorAnalysisResult, SeverityLevel, TumorStats
from severity_assessment.domain.entities import SeverityAssessment, SeverityCategory
from clinical_insight.domain.entities import ClinicalInsight


class TestPDFReportGenerator(unittest.TestCase):
    """Unit test suite for validating the PDF Report Generation pipeline."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.output_pdf_path = os.path.join(self.temp_dir.name, "test_report.pdf")

        # Create dummy image files to satisfy exists checks in generator
        self.dummy_orig_path = os.path.join(self.temp_dir.name, "dummy_orig.png")
        self.dummy_overlay_path = os.path.join(self.temp_dir.name, "dummy_overlay.png")
        self.dummy_mask_path = os.path.join(self.temp_dir.name, "dummy_mask.png")

        # Create actual dummy files using Pillow
        from PIL import Image as PILImage
        for path in [self.dummy_orig_path, self.dummy_overlay_path, self.dummy_mask_path]:
            img = PILImage.new("RGB", (150, 150), color="gray")
            img.save(path, "PNG")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_pdf_report_compilation(self) -> None:
        """Verify that ReportLabPDFGenerator compiles a complete PDF document with visuals and signature blocks."""
        # 1. Arrange mock entities
        patient = PatientInfo(
            patient_id="PATIENT_XYZ_999",
            name="John Doe",
            age=52,
            gender="Male",
            scan_date="2026-08-07",
            ref_physician="Dr. Sarah Johnson, MD"
        )
        
        proc_summary = ProcessingSummary(
            device="CPU",
            execution_time_sec=1.2345,
            classification_model_path="models/efficientnet.pth",
            segmentation_model_path="models/unext.pth",
            classification_latency_sec=0.25,
            segmentation_latency_sec=0.75,
            explainability_latency_sec=0.23
        )
        
        classification = PredictionResult(
            label=0,
            class_name="Glioma",
            confidence_score=0.9856,
            probabilities={"Glioma": 0.9856, "Meningioma": 0.01, "Pituitary": 0.003, "No Tumor": 0.0014}
        )
        
        shape_stats = TumorStats(
            area_pixels=1250,
            area_mm2=1250.0,
            perimeter_pixels=180.0,
            perimeter_mm=180.0,
            bbox_x_px=50,
            bbox_y_px=50,
            bbox_w_px=40,
            bbox_h_px=35,
            bbox_w_mm=40.0,
            bbox_h_mm=35.0,
            major_axis_mm=41.2,
            minor_axis_mm=34.1,
            eccentricity=0.55,
            orientation_deg=45.0,
            solidity=0.94,
            circularity=0.82
        )

        segmentation = TumorAnalysisResult(
            pixel_count=1250,
            tumor_area_mm2=1250.0,
            tumor_percentage_brain=2.45,
            tumor_percentage_image=0.85,
            estimated_brain_pixel_count=51000,
            severity_level=SeverityLevel.HIGH,
            metadata={"post_processing_applied": True},
            rule_based_severity="HIGH",
            severity_rule_description="Lesion area exceeds threshold.",
            stats=shape_stats,
            quality_score=0.975,
            quality_category="Excellent",
            post_processing_applied=True,
            post_processing_metadata={"steps_applied": ["median_filter", "contour_cleaning"]}
        )
        
        severity = SeverityAssessment(
            category=SeverityCategory.HIGH,
            rule_description="Tumor area > 1000 mm2 triggers high severity.",
            educational_disclaimer="For educational and research review only."
        )

        insight = ClinicalInsight(
            summary_narrative="AI findings indicate a significant mass in the left parietal lobe.",
            key_findings=["Mass size is 1250.00 mm2", "High severity rule matched"],
            recommendations=["Immediate neurological consultation", "Surgical planning scan"],
            disclaimer="Research use only."
        )

        report = ClinicalReport(
            patient_info=patient,
            processing_summary=proc_summary,
            classification=classification,
            segmentation_metrics=segmentation,
            severity_assessment=severity,
            original_image_path=self.dummy_orig_path,
            heatmap_image_path=None,
            overlay_image_path=self.dummy_overlay_path,
            segmentation_mask_path=self.dummy_mask_path,
            xai_method="gradcam",
            xai_explanation_text="Focus is concentrated on lesion area.",
            xai_overlap_percentage=0.92,
            quality_warnings=["Inconsistent boundary detected"],
            clinical_insight=insight
        )

        # 2. Act
        generator = ReportLabPDFGenerator()
        generator.generate_pdf(report, self.output_pdf_path)

        # 3. Assert
        self.assertTrue(os.path.exists(self.output_pdf_path))
        self.assertTrue(os.path.getsize(self.output_pdf_path) > 0)


if __name__ == "__main__":
    unittest.main()
