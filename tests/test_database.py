import unittest
import sqlite3
import os
from persistence.infrastructure.repository import SQLitePersistenceRepository
from prediction_history.infrastructure.repository import SQLitePredictionHistoryRepository
from prediction_history.domain.entities import HistorySearchCriteria
from clinical_reporting.domain.entities import PatientInfo, ProcessingSummary, ClinicalReport
from classification.domain.entities import PredictionResult
from severity_assessment.domain.entities import SeverityAssessment, SeverityCategory
from tumor_analysis.domain.entities import TumorAnalysisResult, SeverityLevel


class TestDatabasePersistence(unittest.TestCase):
    """Database and integration checks utilizing local isolated database setups."""

    def setUp(self):
        self.db_path = "outputs/test_clinical_reports.db"
        os.makedirs("outputs", exist_ok=True)
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass
        self.persistence_repo = SQLitePersistenceRepository(db_path=self.db_path)
        self.persistence_repo.initialize_db()
        self.history_repo = SQLitePredictionHistoryRepository(db_path=self.db_path)

    def tearDown(self):
        # Clean up database file after test run
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def test_save_and_retrieve_patient_records(self):
        """Verify patient details and report output paths log dynamically."""
        # 1. Create Mock Clinical Report
        patient_info = PatientInfo(
            patient_id="PATIENT_TEST_999",
            name="Bob Jones",
            age=45,
            gender="Male",
            scan_date="2026-08-02",
            ref_physician="Dr. Sarah"
        )
        summary = ProcessingSummary(
            device="cpu",
            execution_time_sec=0.15,
            classification_model_path="models/cls.pth",
            segmentation_model_path="models/seg.pth",
            classification_latency_sec=0.05,
            segmentation_latency_sec=0.08,
            explainability_latency_sec=0.02
        )
        classification = PredictionResult(
            label=1,
            class_name="Meningioma",
            confidence_score=0.95,
            probabilities={"Meningioma": 0.95}
        )
        metrics = TumorAnalysisResult(
            pixel_count=200,
            tumor_area_mm2=20.0,
            tumor_percentage_brain=0.5,
            tumor_percentage_image=0.1,
            estimated_brain_pixel_count=40000,
            severity_level=SeverityLevel.HIGH,
            metadata={}
        )
        severity = SeverityAssessment(
            category=SeverityCategory.HIGH,
            rule_description="Meningioma high risk",
            educational_disclaimer="Disclaimer"
        )
        
        report = ClinicalReport(
            patient_info=patient_info,
            processing_summary=summary,
            classification=classification,
            segmentation_metrics=metrics,
            severity_assessment=severity,
            original_image_path="test_in.png",
            heatmap_image_path="test_h.png",
            overlay_image_path="test_o.png",
            segmentation_mask_path="test_m.png"
        )

        # 2. Save report to DB
        report_id = self.persistence_repo.save_report(report, output_dir="outputs")
        self.assertEqual(report_id, 1)

        # 3. Query report summary using History repository wildcards
        criteria = HistorySearchCriteria(patient_id="PATIENT_TEST%")
        summaries = self.history_repo.search_history(criteria)
        
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0].patient_name, "Bob Jones")
        self.assertEqual(summaries[0].predicted_class, "Meningioma")
        self.assertEqual(summaries[0].rule_based_severity, "High")

        # 4. Search report PDF paths
        paths = self.history_repo.get_report_paths(report_id)
        self.assertIsNotNone(paths)
        # Should return markdown_path, json_path, pdf_path
        self.assertEqual(len(paths), 3)


if __name__ == "__main__":
    unittest.main()
