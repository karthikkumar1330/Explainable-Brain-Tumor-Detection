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

    def test_report_builder_validation(self) -> None:
        """Verify that ReportDataBuilder raises ValueError when critical details are absent."""
        from clinical_reporting.application.builder import ReportDataBuilder
        builder = ReportDataBuilder()

        # Test empty report
        with self.assertRaises(ValueError):
            builder.build(None)

        # Test missing classification
        patient = PatientInfo("PAT_1", "Name", 30, "Male", "2026-08-08", "Phys")
        proc = ProcessingSummary("cpu", 1.0, "cls", "seg", 0.5, 0.5, 0.0)
        report = ClinicalReport(
            patient_info=patient,
            processing_summary=proc,
            classification=None,
            segmentation_metrics=None,
            severity_assessment=None,
            original_image_path="",
            heatmap_image_path=None,
            overlay_image_path=None,
            segmentation_mask_path=None
        )
        with self.assertRaises(ValueError):
            builder.build(report)

    def test_report_builder_no_tumor_case(self) -> None:
        """Verify report generation adapts gracefully when no tumor is classified."""
        from clinical_reporting.application.builder import ReportDataBuilder
        from clinical_reporting.domain.entities import ReportData

        patient = PatientInfo("PAT_NT", "Normal Subject", 45, "Female", "2026-08-08", "Phys")
        proc = ProcessingSummary("cpu", 0.5, "cls", "seg", 0.1, 0.1, 0.1)
        classification = PredictionResult(
            label=3,
            class_name="No Tumor",
            confidence_score=0.9992,
            probabilities={"Glioma": 0.0001, "Meningioma": 0.0001, "Pituitary": 0.0001, "No Tumor": 0.9997}
        )

        report = ClinicalReport(
            patient_info=patient,
            processing_summary=proc,
            classification=classification,
            segmentation_metrics=None,
            severity_assessment=None,
            original_image_path="",
            heatmap_image_path=None,
            overlay_image_path=None,
            segmentation_mask_path=None
        )

        # Build and compile
        builder = ReportDataBuilder()
        data = builder.build(report)

        self.assertEqual(data.classification_result.predicted_class, "No Tumor")
        self.assertEqual(data.system_status.segmentation, "Failed") # none is failed

        generator = ReportLabPDFGenerator()
        generator.generate_pdf(data, self.output_pdf_path)
        self.assertTrue(os.path.exists(self.output_pdf_path))
        self.assertTrue(os.path.getsize(self.output_pdf_path) > 0)

    def test_renderer_safe_image_fallback(self) -> None:
        """Verify that safe_image handles missing and invalid files without raising unhandled errors."""
        from clinical_reporting.infrastructure.renderer import EnterpriseReportRenderer
        from reportlab.platypus import Paragraph

        renderer = EnterpriseReportRenderer()

        # Test None path
        none_flowables = renderer._safe_image(None, 100, 100, "Test None")
        self.assertTrue(any(isinstance(f, Paragraph) and "Unavailable" in f.text for f in none_flowables))

        # Test non-existent file path
        missing_flowables = renderer._safe_image("non_existent_file.png", 100, 100, "Test Missing")
        self.assertTrue(any(isinstance(f, Paragraph) and "Missing" in f.text for f in missing_flowables))

    def test_xai_status_gradcam_consistency(self) -> None:
        """Verify Grad-CAM available vs unavailable dynamically maps to actual files."""
        from clinical_reporting.application.builder import ReportDataBuilder

        patient = PatientInfo("PAT_NT", "Normal Subject", 45, "Female", "2026-08-08", "Phys")
        proc = ProcessingSummary("cpu", 0.5, "cls", "seg", 0.1, 0.1, 0.1)
        classification = PredictionResult(
            label=2,
            class_name="Pituitary",
            confidence_score=0.80,
            probabilities={"Glioma": 0.1, "Meningioma": 0.1, "Pituitary": 0.8, "No Tumor": 0.0}
        )

        # Test Case 1: Grad-CAM overlay exists
        import tempfile
        with tempfile.NamedTemporaryFile(suffix="_overlay.png", delete=False) as tmp_overlay:
            overlay_path = tmp_overlay.name
        try:
            # write raw image bytes to make PIL happy
            from PIL import Image
            img = Image.new('RGB', (100, 100), color = 'red')
            img.save(overlay_path)

            report_avail = ClinicalReport(
                patient_info=patient,
                processing_summary=proc,
                classification=classification,
                segmentation_metrics=None,
                severity_assessment=None,
                original_image_path="",
                heatmap_image_path=None,
                overlay_image_path=overlay_path,
                segmentation_mask_path=None
            )
            builder = ReportDataBuilder()
            data_avail = builder.build(report_avail)
            self.assertEqual(data_avail.explainability_result.xai_status, "Available")
            self.assertEqual(data_avail.system_status.explainability, "Available")
        finally:
            if os.path.exists(overlay_path):
                os.remove(overlay_path)

        # Test Case 2: Grad-CAM files missing
        report_unavail = ClinicalReport(
            patient_info=patient,
            processing_summary=proc,
            classification=classification,
            segmentation_metrics=None,
            severity_assessment=None,
            original_image_path="",
            heatmap_image_path="non_existent_heatmap.png",
            overlay_image_path="non_existent_overlay.png",
            segmentation_mask_path=None
        )
        data_unavail = builder.build(report_unavail)
        self.assertEqual(data_unavail.explainability_result.xai_status, "Unavailable")
        self.assertEqual(data_unavail.system_status.explainability, "Unavailable")

    def test_clinical_insight_rendering_states(self) -> None:
        """Verify clinical insight maps and displays correct unavailable alerts in renderer."""
        from clinical_reporting.application.builder import ReportDataBuilder
        from clinical_reporting.infrastructure.renderer import EnterpriseReportRenderer
        from clinical_insight.domain.entities import ClinicalInsight

        patient = PatientInfo("PAT_NT", "Normal Subject", 45, "Female", "2026-08-08", "Phys")
        proc = ProcessingSummary("cpu", 0.5, "cls", "seg", 0.1, 0.1, 0.1)
        classification = PredictionResult(
            label=2,
            class_name="Pituitary",
            confidence_score=0.80,
            probabilities={"Glioma": 0.1, "Meningioma": 0.1, "Pituitary": 0.8, "No Tumor": 0.0}
        )

        # Case 1: insight exists
        insight = ClinicalInsight(
            summary_narrative="Calibrated Pituitary adenoma detected.",
            key_findings=["Lesion in sella turcica"],
            recommendations=["MRI endocrinology protocol"],
            disclaimer="Research only"
        )
        report_avail = ClinicalReport(
            patient_info=patient,
            processing_summary=proc,
            classification=classification,
            segmentation_metrics=None,
            severity_assessment=None,
            original_image_path="",
            heatmap_image_path=None,
            overlay_image_path=None,
            segmentation_mask_path=None,
            clinical_insight=insight
        )
        builder = ReportDataBuilder()
        data_avail = builder.build(report_avail)

        renderer = EnterpriseReportRenderer()
        story_avail = renderer.render(data_avail)

        def get_all_texts(flowables):
            t = ""
            for f in flowables:
                if hasattr(f, "text"):
                    t += f.text + " "
                elif hasattr(f, "_cellvalues"):
                    for row in f._cellvalues:
                        for cell in row:
                            if isinstance(cell, list):
                                t += get_all_texts(cell) + " "
                            elif hasattr(cell, "text"):
                                t += cell.text + " "
                elif hasattr(f, "_content"):
                    t += get_all_texts(f._content) + " "
                elif hasattr(f, "_flowables"):
                    t += get_all_texts(f._flowables) + " "
            return t

        all_text_avail = get_all_texts(story_avail)
        self.assertIn("Calibrated Pituitary adenoma detected.", all_text_avail)

        # Case 2: insight missing
        report_unavail = ClinicalReport(
            patient_info=patient,
            processing_summary=proc,
            classification=classification,
            segmentation_metrics=None,
            severity_assessment=None,
            original_image_path="",
            heatmap_image_path=None,
            overlay_image_path=None,
            segmentation_mask_path=None,
            clinical_insight=None
        )
        data_unavail = builder.build(report_unavail)
        story_unavail = renderer.render(data_unavail)
        all_text_unavail = get_all_texts(story_unavail)
        self.assertIn("Clinical insight summary is unavailable", all_text_unavail)

    def test_device_metrics_cpu_gpu(self) -> None:
        """Verify processing device propagates CPU/GPU settings properly."""
        from clinical_reporting.application.builder import ReportDataBuilder

        patient = PatientInfo("PAT_NT", "Normal Subject", 45, "Female", "2026-08-08", "Phys")
        proc_cpu = ProcessingSummary("cpu", 0.5, "cls", "seg", 0.1, 0.1, 0.1)
        proc_cuda = ProcessingSummary("cuda", 0.5, "cls", "seg", 0.1, 0.1, 0.1)
        classification = PredictionResult(
            label=2,
            class_name="Pituitary",
            confidence_score=0.80,
            probabilities={"Glioma": 0.1, "Meningioma": 0.1, "Pituitary": 0.8, "No Tumor": 0.0}
        )

        builder = ReportDataBuilder()
        data_cpu = builder.build(ClinicalReport(patient, proc_cpu, classification, None, None, "", None, None, None))
        data_cuda = builder.build(ClinicalReport(patient, proc_cuda, classification, None, None, "", None, None, None))

        self.assertEqual(data_cpu.processing_metrics.device, "CPU")
        self.assertEqual(data_cuda.processing_metrics.device, "CUDA")

    def test_wording_certification_safety(self) -> None:
        """Ensure no regulatory release status wording exists in PDF renderer."""
        from clinical_reporting.application.builder import ReportDataBuilder
        from clinical_reporting.infrastructure.renderer import EnterpriseReportRenderer

        patient = PatientInfo("PAT_NT", "Normal Subject", 45, "Female", "2026-08-08", "Phys")
        proc = ProcessingSummary("cpu", 0.5, "cls", "seg", 0.1, 0.1, 0.1)
        classification = PredictionResult(
            label=2,
            class_name="Pituitary",
            confidence_score=0.80,
            probabilities={"Glioma": 0.1, "Meningioma": 0.1, "Pituitary": 0.8, "No Tumor": 0.0}
        )
        builder = ReportDataBuilder()
        data = builder.build(ClinicalReport(patient, proc, classification, None, None, "", None, None, None))

        renderer = EnterpriseReportRenderer()
        story = renderer.render(data)

        def get_all_texts(flowables):
            t = ""
            for f in flowables:
                if hasattr(f, "text"):
                    t += f.text + " "
                elif hasattr(f, "_cellvalues"):
                    for row in f._cellvalues:
                        for cell in row:
                            if isinstance(cell, list):
                                t += get_all_texts(cell) + " "
                            elif hasattr(cell, "text"):
                                t += cell.text + " "
                elif hasattr(f, "_content"):
                    t += get_all_texts(f._content) + " "
                elif hasattr(f, "_flowables"):
                    t += get_all_texts(f._flowables) + " "
            return t

        all_text = get_all_texts(story)

        self.assertNotIn("Clinical Validation Stamp", all_text)
        self.assertNotIn("Diagnostic System Certified", all_text)
        self.assertNotIn("VERIFIED RELEASE", all_text)

        self.assertIn("Software Verification Status", all_text)
        self.assertIn("Research / Educational Prototype", all_text)
        self.assertIn("Not independently clinically validated", all_text)

    def test_eccentricity_numerical_safety(self) -> None:
        """Verify stats engine eccentricity safeguards edge cases: minor>major, major=0, nan/inf."""
        from tumor_analysis.infrastructure.stats_engine import OpenCVTumorStatsEngine
        import numpy as np

        analyzer = OpenCVTumorStatsEngine()

        contour = np.array([[[10, 10]], [[10, 10]], [[10, 10]], [[10, 10]]])
        major, minor, ecc, orient = analyzer._compute_moments_fallback(contour, 1.0)
        self.assertEqual(major, 0.0)
        self.assertEqual(minor, 0.0)
        self.assertEqual(ecc, 0.0)


if __name__ == "__main__":
    unittest.main()
