import os
import unittest
import sqlite3
import datetime
import json
import numpy as np
import torch
import cv2

# Set the test DB path env var
os.environ["DB_PATH"] = os.path.abspath("outputs/test_provenance_comparison.db")

from persistence.infrastructure.repository import SQLitePersistenceRepository
from clinical_reporting.application.services import ReportService
from clinical_reporting.domain.entities import PatientInfo, ProcessingSummary, PredictionResult, ClinicalReport
from explainable_ai.infrastructure.services import PyTorchXAIEngine

class TestPipelineProvenanceAndComparison(unittest.TestCase):
    def setUp(self) -> None:
        self.db_path = os.environ.get("DB_PATH", "outputs/test_provenance_comparison.db")
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

        self.repo = SQLitePersistenceRepository(db_path=self.db_path)
        self.repo.initialize_db()

        self.output_dir = "outputs/clinical_reports"
        os.makedirs(self.output_dir, exist_ok=True)

    def tearDown(self) -> None:
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def test_visualization_provenance(self):
        """1. Verify that when a report is saved, visualization paths are correctly stored and linked."""
        patient_info = PatientInfo(
            patient_id="pat_prov_test",
            name="Alice Doe",
            age="45",
            gender="Female",
            ref_physician="Dr. Gregory",
            scan_date="2026-08-19"
        )
        proc_summary = ProcessingSummary(
            device="cpu",
            execution_time_sec=1.5,
            classification_model_path="dummy_cls.pth",
            segmentation_model_path="dummy_seg.pth",
            classification_latency_sec=0.5,
            segmentation_latency_sec=0.5,
            explainability_latency_sec=0.5
        )
        pred_res = PredictionResult(
            label=0,
            class_name="Glioma",
            confidence_score=0.92,
            probabilities={"Glioma": 0.92, "No Tumor": 0.08}
        )

        # Create dummy visual files
        heatmap_file = os.path.join(self.output_dir, "pat_prov_test_dummy_heatmap.png")
        overlay_file = os.path.join(self.output_dir, "pat_prov_test_dummy_overlay.png")
        mask_file = os.path.join(self.output_dir, "pat_prov_test_dummy_mask.png")

        cv2.imwrite(heatmap_file, np.zeros((100, 100, 3), dtype=np.uint8))
        cv2.imwrite(overlay_file, np.zeros((100, 100, 3), dtype=np.uint8))
        cv2.imwrite(mask_file, np.zeros((100, 100), dtype=np.uint8))

        report = ClinicalReport(
            patient_info=patient_info,
            processing_summary=proc_summary,
            classification=pred_res,
            segmentation_metrics=None,
            severity_assessment=None,
            original_image_path="dummy_mri.png",
            heatmap_image_path=heatmap_file,
            overlay_image_path=overlay_file,
            segmentation_mask_path=mask_file
        )

        # Precreate old path files to simulate generation
        old_md = os.path.join(self.output_dir, "pat_prov_test_clinical_report.md")
        old_js = os.path.join(self.output_dir, "pat_prov_test_clinical_report.json")
        old_pdf = os.path.join(self.output_dir, "pat_prov_test_clinical_report.pdf")

        with open(old_md, "w") as f: f.write("dummy md")
        with open(old_js, "w") as f: json.dump({"dummy": "json"}, f)
        with open(old_pdf, "w") as f: f.write("dummy pdf")

        # Save to DB (this should rename files on disk)
        report_id = self.repo.save_report(report, output_dir=self.output_dir)

        # Retrieve paths from DB
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM clinical_reports WHERE id = ?;", (report_id,)).fetchone()
        conn.close()

        self.assertIsNotNone(row)
        self.assertIn("pat_prov_test_dummy_heatmap.png", row["heatmap_path"])
        self.assertIn("pat_prov_test_dummy_mask.png", row["mask_path"])
        self.assertTrue(os.path.exists(row["json_path"]))
        self.assertTrue(os.path.exists(row["pdf_path"]))
        self.assertTrue(os.path.exists(row["markdown_path"]))
        self.assertIn("RPT-", row["json_path"]) # Check report number is in path

        # Cleanup created files
        for f in [heatmap_file, overlay_file, mask_file, row["json_path"], row["pdf_path"], row["markdown_path"]]:
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass

    def test_gradcam_non_empty_output(self):
        """2. Verify that Grad-CAM generated heatmaps for non-flat input are non-empty and non-flat."""
        class MockNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1)
                class Backbone:
                    def __init__(self, conv):
                        self.features = [None, None, None, None, None, None, None, None, conv]
                self.backbone = Backbone(self.conv)
                self.fc = torch.nn.Linear(8 * 7 * 7, 4)

            def forward(self, x):
                x = self.conv(x)
                x = torch.nn.functional.relu(x)
                x = torch.nn.functional.adaptive_avg_pool2d(x, (7, 7))
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x

        model = MockNet()
        engine = PyTorchXAIEngine(
            model=model,
            target_layer=model.conv,
            device=torch.device("cpu")
        )

        # Prepare random input tensor to produce non-flat activations/gradients
        input_tensor = torch.randn(1, 3, 224, 224)
        heatmap = engine.generate_explanation(input_tensor, target_class=2, method="gradcam")

        self.assertEqual(heatmap.shape, (224, 224))
        self.assertTrue(np.max(heatmap) > np.min(heatmap), "Heatmap must not be flat/empty.")
        self.assertTrue(np.std(heatmap) > 0.0, "Heatmap standard deviation must be positive.")

    def test_segmentation_mask_provenance(self):
        """3. Verify that the segmentation mask is correctly written and matches target dimensions."""
        mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.circle(mask, (128, 128), 30, 255, -1) # Draw tumor circle

        mask_file = os.path.join(self.output_dir, "pat_test_mask.png")
        cv2.imwrite(mask_file, mask)

        loaded_mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        self.assertEqual(loaded_mask.shape, (256, 256))
        self.assertEqual(np.max(loaded_mask), 255)
        self.assertEqual(np.min(loaded_mask), 0)

        # Cleanup
        if os.path.exists(mask_file):
            os.remove(mask_file)

    def test_report_id_comparison_correctness(self):
        """4. Verify that two reports with distinct JSON paths return correct comparison metrics."""
        patient_id = "pat_compare_test"

        # Insert patient
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
                     (patient_id, "Patient C", 50, "Male", "2026-08-19 12:00:00"))

        # Insert MRI Scans
        conn.execute("INSERT INTO mri_scans (id, patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                     (10, patient_id, "scan1.png", 1.0, "Dr. A", "2026-08-01", "2026-08-19 12:00:00"))
        conn.execute("INSERT INTO mri_scans (id, patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                     (20, patient_id, "scan2.png", 1.0, "Dr. A", "2026-08-15", "2026-08-19 12:00:00"))

        # Insert Predictions (include all NOT NULL columns)
        conn.execute("""INSERT INTO predictions (
            id, scan_id, predicted_class, confidence_score, prob_glioma, prob_meningioma, prob_pituitary, prob_no_tumor,
            tumor_pixel_count, tumor_area_mm2, tumor_percentage_brain, tumor_percentage_image, estimated_brain_pixel_count,
            rule_based_severity, severity_rule_description, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
        (101, 10, "Glioma", 0.90, 0.90, 0.05, 0.03, 0.02, 100, 10.0, 0.05, 0.02, 5000, "LOW", "Mild tumor", "2026-08-19 12:00:00"))

        conn.execute("""INSERT INTO predictions (
            id, scan_id, predicted_class, confidence_score, prob_glioma, prob_meningioma, prob_pituitary, prob_no_tumor,
            tumor_pixel_count, tumor_area_mm2, tumor_percentage_brain, tumor_percentage_image, estimated_brain_pixel_count,
            rule_based_severity, severity_rule_description, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
        (202, 20, "Meningioma", 0.80, 0.05, 0.80, 0.10, 0.05, 200, 20.0, 0.10, 0.04, 5000, "HIGH", "Severe tumor", "2026-08-19 12:00:00"))

        # Generate distinct JSON reports on disk
        json_a = os.path.join(self.output_dir, "pat_compare_test_RPT-2026-000001_clinical_report.json")
        json_b = os.path.join(self.output_dir, "pat_compare_test_RPT-2026-000002_clinical_report.json")

        json_data_a = {
            "patient": {"patient_id": patient_id, "scan_date": "2026-08-01"},
            "classification": {"predicted_class": "Glioma", "confidence_score": 0.90},
            "segmentation": {"tumor_area_mm2": 10.0, "tumor_percentage_brain": 0.05},
            "severity": {"category": "LOW"}
        }
        json_data_b = {
            "patient": {"patient_id": patient_id, "scan_date": "2026-08-15"},
            "classification": {"predicted_class": "Meningioma", "confidence_score": 0.80},
            "segmentation": {"tumor_area_mm2": 20.0, "tumor_percentage_brain": 0.10},
            "severity": {"category": "HIGH"}
        }

        with open(json_a, "w") as f: json.dump(json_data_a, f)
        with open(json_b, "w") as f: json.dump(json_data_b, f)

        # Insert Reports
        conn.execute("""INSERT INTO reports (
            report_id, report_number, patient_id, created_by, report_type, current_version, status, created_at, updated_at, pdf_path, json_path, checksum
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
        (1, "RPT-2026-000001", patient_id, "Dr. A", "MRI Brain Scan", 1, "GENERATED", "2026-08-19 12:00:00", "2026-08-19 12:00:00", "pdf1", json_a, "check1"))

        conn.execute("""INSERT INTO reports (
            report_id, report_number, patient_id, created_by, report_type, current_version, status, created_at, updated_at, pdf_path, json_path, checksum
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
        (2, "RPT-2026-000002", patient_id, "Dr. A", "MRI Brain Scan", 1, "GENERATED", "2026-08-19 12:00:00", "2026-08-19 12:00:00", "pdf2", json_b, "check2"))

        # Insert Report Versions
        conn.execute("""INSERT INTO report_versions (
            report_id, version_number, created_at, created_by, reason, pdf_path, json_path, checksum, status, prediction_id
        ) VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?, ?);""",
        (1, "2026-08-19 12:00:00", "Dr. A", "Initial", "pdf1", json_a, "check1", "GENERATED", 101))

        conn.execute("""INSERT INTO report_versions (
            report_id, version_number, created_at, created_by, reason, pdf_path, json_path, checksum, status, prediction_id
        ) VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?, ?);""",
        (2, "2026-08-19 12:00:00", "Dr. A", "Initial", "pdf2", json_b, "check2", "GENERATED", 202))

        conn.commit()
        conn.close()

        # Run comparison service
        service = ReportService(db_path=self.db_path)
        service.check_report_access = lambda report_id, user: "AUTHORIZED"

        comparison = service.compare_reports(previous_report_id=1, current_report_id=2)

        # Get specific metrics by name from metrics list
        cls_metric = next(m for m in comparison["metrics"] if m["name"] == "classification")
        conf_metric = next(m for m in comparison["metrics"] if m["name"] == "confidence")
        area_metric = next(m for m in comparison["metrics"] if m["name"] == "tumor_area_mm2")

        self.assertEqual(cls_metric["previous_value"], "Glioma")
        self.assertEqual(cls_metric["current_value"], "Meningioma")
        self.assertAlmostEqual(conf_metric["previous_value"], 0.90)
        self.assertAlmostEqual(conf_metric["current_value"], 0.80)
        self.assertAlmostEqual(area_metric["previous_value"], 10.0)
        self.assertAlmostEqual(area_metric["current_value"], 20.0)

        # Cleanup JSON files
        if os.path.exists(json_a): os.remove(json_a)
        if os.path.exists(json_b): os.remove(json_b)

if __name__ == "__main__":
    unittest.main()
