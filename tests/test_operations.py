import unittest
import torch
import numpy as np
import os
from monitoring.domain.version_manager import ModelVersionManager
from monitoring.application.pipeline_recovery import PipelineExecutionRecovery
from persistence.infrastructure.repository import SQLitePersistenceRepository


class TestModelOperations(unittest.TestCase):
    def setUp(self):
        self.db_path = "test_ops.db"
        self.repo = SQLitePersistenceRepository(db_path=self.db_path)
        self.repo.initialize_db()
        self.recovery = PipelineExecutionRecovery()

    def tearDown(self):
        import gc
        gc.collect()
        try:
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
        except Exception:
            pass

    def test_model_version_manager(self):
        """Verifies ModelVersionManager outputs the expected format."""
        manager = ModelVersionManager()
        details = manager.get_version_details()
        self.assertIn("classification_version", details)
        self.assertIn("segmentation_version", details)
        self.assertIn("calibration_version", details)
        self.assertIn("classification_training_date", details)
        self.assertIn("checkpoint_version", details)

    def test_pipeline_recovery_graceful_success(self):
        """Tests that graceful auxiliary stages execute successfully when healthy."""
        def dummy_action():
            return "healthy_output"
            
        result, warnings = self.recovery.execute_graceful_stage("Test Stage", dummy_action, "fallback")
        self.assertEqual(result, "healthy_output")
        self.assertEqual(len(warnings), 0)

    def test_pipeline_recovery_graceful_fallback(self):
        """Tests that graceful auxiliary stages catch exceptions and return fallbacks."""
        def failing_action():
            raise ValueError("Computation failed")
            
        result, warnings = self.recovery.execute_graceful_stage("Test Stage", failing_action, "fallback_value")
        self.assertEqual(result, "fallback_value")
        self.assertTrue(any("failed" in w.lower() for w in warnings))

    def test_timeline_persistence(self):
        """Tests that step timelines can be persisted and loaded from SQLite."""
        pred_id = 998
        timeline_data = {
            "Upload": 0.05,
            "Validation": 0.12,
            "Classification": 0.35,
            "Completed": 1.25
        }
        # Insert a dummy patient, scan, and prediction record to satisfy database constraints
        conn = self.repo._get_connection()
        with conn:
            conn.execute("INSERT OR IGNORE INTO patients (patient_id, name, age, gender, created_at) VALUES ('P1', 'Test Patient', 45, 'M', '2026-08-01');")
            conn.execute("INSERT OR IGNORE INTO mri_scans (id, patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at) VALUES (1, 'P1', 'path', 1.0, 'Dr', '2026-08-01', '2026-08-01');")
            conn.execute(f"""
                INSERT OR IGNORE INTO predictions (
                    id, scan_id, predicted_class, confidence_score, 
                    prob_glioma, prob_meningioma, prob_pituitary, prob_no_tumor, 
                    tumor_pixel_count, tumor_area_mm2, tumor_percentage_brain, tumor_percentage_image, 
                    estimated_brain_pixel_count, rule_based_severity, severity_rule_description, created_at
                ) VALUES (
                    {pred_id}, 1, 'Glioma', 0.95, 
                    0.95, 0.0, 0.0, 0.0, 
                    100, 100.0, 1.0, 1.0, 
                    1000, 'Low', 'Desc', '2026-08-01'
                );
            """)
        conn.close()

        self.repo.save_timeline_trace(pred_id, timeline_data)
        loaded = self.repo.get_timeline_trace(pred_id)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["Validation"], 0.12)
        self.assertEqual(loaded["Classification"], 0.35)
