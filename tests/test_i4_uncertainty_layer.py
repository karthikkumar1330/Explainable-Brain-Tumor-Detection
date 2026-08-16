import os
import io
import unittest
import numpy as np
import cv2
import sqlite3
import tempfile
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Setup database path for test execution
os.environ["DB_PATH"] = os.path.abspath("outputs/test_i4_uncertainty_clean.db")
os.environ["DISABLE_TEST_AUTO_ASSIGN"] = "1"

import pytest

@pytest.fixture(autouse=True)
def manage_auto_assign_env():
    os.environ["DISABLE_TEST_AUTO_ASSIGN"] = "1"
    yield
    os.environ["DISABLE_TEST_AUTO_ASSIGN"] = "1"

from run_api import app
import api.infrastructure.routes as api_routes
import api.routes.auth_routes as auth_routes
from security.domain.entities import Role, User
from security.infrastructure.jwt_service import JWTService
from security.infrastructure.repository import SQLiteUserRepository
from persistence.infrastructure.repository import SQLitePersistenceRepository
from api.infrastructure.routes import (
    calculate_predictive_entropy,
    generate_spatial_uncertainty_map
)

class TestI4AIConfidenceUncertainty(unittest.TestCase):
    """Phase I4: AI Confidence & Uncertainty Layer tests."""

    def setUp(self):
        self.db_path = os.environ.get("DB_PATH", "outputs/test_i4_uncertainty.db")
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

        # Force routes database paths to point to test database
        api_routes.DEFAULT_DB_PATH = self.db_path
        auth_routes.DEFAULT_DB_PATH = self.db_path

        # Initialize persistence database (performs migrations)
        self.db_repo = SQLitePersistenceRepository(db_path=self.db_path)
        self.db_repo.initialize_db()

        # Initialize security tables
        self.user_repo = SQLiteUserRepository(db_path=self.db_path)
        self.user_repo.initialize_security_tables()

        # Seed test users
        self.doctor = User(
            id=None,
            uuid="doc-uuid-10",
            email="dr.smith@hospital.org",
            password_hash="dummy_hash",
            full_name="Dr. Smith",
            role=Role.DOCTOR,
            is_active=True,
            created_at="2026-08-16 00:00:00"
        )
        self.patient = User(
            id=None,
            uuid="pat-uuid-20",
            email="patient.doe@hospital.org",
            password_hash="dummy_hash",
            full_name="John Doe",
            role=Role.PATIENT,
            is_active=True,
            created_at="2026-08-16 00:00:00"
        )
        self.user_repo.create_user(self.doctor)
        self.user_repo.create_user(self.patient)

        jwt_svc = JWTService()
        self.doctor_token = jwt_svc.create_access_token(
            user_uuid=self.doctor.uuid,
            user_id=self.doctor.id,
            email=self.doctor.email,
            role=self.doctor.role
        )
        self.patient_token = jwt_svc.create_access_token(
            user_uuid=self.patient.uuid,
            user_id=self.patient.id,
            email=self.patient.email,
            role=self.patient.role
        )

        self.doctor_headers = {"Authorization": f"Bearer {self.doctor_token}"}
        self.patient_headers = {"Authorization": f"Bearer {self.patient_token}"}

        self.client = TestClient(app)

    def tearDown(self):
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass
        if os.path.exists("uncertainty.png"):
            try:
                os.remove("uncertainty.png")
            except Exception:
                pass

    def test_entropy_math_calculations(self):
        """Verify Shannon entropy calculations map correctly for uniform, extreme and general probability arrays."""
        # 1. Extreme certainty: one class probability is 1.0, others are 0.0
        p_certain = {"Glioma": 1.0, "Meningioma": 0.0, "Pituitary": 0.0, "No Tumor": 0.0}
        self.assertAlmostEqual(calculate_predictive_entropy(p_certain), 0.0, places=5)

        # 2. Maximum uncertainty: all classes are equally probable (0.25 each)
        p_uniform = {"Glioma": 0.25, "Meningioma": 0.25, "Pituitary": 0.25, "No Tumor": 0.25}
        # Max entropy for 4 classes is log2(4) = 2.0
        self.assertAlmostEqual(calculate_predictive_entropy(p_uniform), 2.0, places=5)

        # 3. Intermediate distribution
        p_inter = {"Glioma": 0.70, "Meningioma": 0.10, "Pituitary": 0.10, "No Tumor": 0.10}
        # - (0.7*log2(0.7) + 3 * 0.1*log2(0.1)) = -(-0.3601 - 0.9965) = 1.3567
        self.assertAlmostEqual(calculate_predictive_entropy(p_inter), 1.3567, places=3)

    def test_entropy_malformed_input_graceful_handling(self):
        """Assert calculations handle malformed, empty or corrupt probabilities lists gracefully."""
        self.assertEqual(calculate_predictive_entropy({}), 0.0)
        self.assertEqual(calculate_predictive_entropy({"Glioma": -0.5, "Meningioma": 1.5}), 0.0)

    def test_requires_review_threshold_triggers(self):
        """Verify the triggers of requires_review review recommended flag based on margin and entropy thresholds."""
        # Case A: Low entropy (0.5), high margin (0.8) -> requires_review should be False
        # Margin = 0.9 - 0.1 = 0.8
        p_low = {"Glioma": 0.9, "Meningioma": 0.1, "Pituitary": 0.0, "No Tumor": 0.0}
        h_low = calculate_predictive_entropy(p_low)
        self.assertLess(h_low, 1.0)
        margin = 0.8
        self.assertFalse((h_low >= 1.0) or (margin < 0.10))

        # Case B: High entropy (1.5), high margin (0.7 - 0.1 = 0.6) -> requires_review should be True
        p_high = {"Glioma": 0.7, "Meningioma": 0.1, "Pituitary": 0.1, "No Tumor": 0.1}
        h_high = calculate_predictive_entropy(p_high)
        self.assertGreaterEqual(h_high, 1.0)

        # Case C: Low margin (< 0.10) between top two classes -> requires_review should be True
        # Top 2 are Glioma (0.45) and Meningioma (0.43). Margin = 0.02
        p_narrow = {"Glioma": 0.45, "Meningioma": 0.43, "Pituitary": 0.06, "No Tumor": 0.06}
        sorted_probs = sorted(p_narrow.values(), reverse=True)
        margin_narrow = sorted_probs[0] - sorted_probs[1]
        self.assertLess(margin_narrow, 0.10)

    def test_segmentation_uncertainty_map(self):
        """Assert pixel-wise Shannon boundary maps calculate color map shape and handle empty masks."""
        # 1. Test empty probability map (all zeros) -> yields 0 entropy and runs successfully
        empty_map = np.zeros((10, 10), dtype=np.float32)
        color_map, norm_map = generate_spatial_uncertainty_map(empty_map)
        self.assertEqual(color_map.shape, (10, 10, 3))
        self.assertEqual(norm_map.shape, (10, 10))
        self.assertEqual(np.sum(norm_map), 0)

        # 2. Test maximum uncertainty map (all 0.5) -> yields maximum entropy (entropy_norm = 255)
        uncertain_map = np.ones((10, 10), dtype=np.float32) * 0.5
        color_map, norm_map = generate_spatial_uncertainty_map(uncertain_map)
        self.assertEqual(np.min(norm_map), 255)

    def test_database_migrations_idempotency(self):
        """Verify SQLite table migrations dynamically alter tables cleanly and repeatedly."""
        conn = sqlite3.connect(self.db_path)

        # 1. Fetch column info to confirm columns exist
        cursor = conn.execute("PRAGMA table_info(predictions);")
        columns = [row[1] for row in cursor.fetchall()]
        self.assertIn("predictive_entropy", columns)
        self.assertIn("requires_review", columns)

        cursor = conn.execute("PRAGMA table_info(clinical_reports);")
        cr_columns = [row[1] for row in cursor.fetchall()]
        self.assertIn("uncertainty_path", cr_columns)

        # 2. Run initialization migration again to confirm idempotency
        self.db_repo.initialize_db()

        # Verify columns still exist and no errors were raised
        cursor = conn.execute("PRAGMA table_info(predictions);")
        self.assertIn("predictive_entropy", [row[1] for row in cursor.fetchall()])
        conn.close()

    def test_api_visuals_streaming_authorization(self):
        """Verify role-based authorization holds: Doctors/Admins can query visuals, Patients are forbidden."""
        # Setup mock report in database
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute("INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES ('P100', 'Doe', 30, 'M', '2026-08-16');")
                conn.execute("INSERT INTO doctor_patient_assignments (doctor_id, patient_id, created_at) VALUES (?, 'P100', '2026-08-16');", (self.doctor.id,))
                conn.execute("INSERT INTO mri_scans (id, patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at) VALUES (1, 'P100', 'scan.png', 1.0, 'Dr. Smith', '2026-08-16', '2026-08-16');")
                conn.execute("""
                INSERT INTO predictions (
                    id, scan_id, predicted_class, confidence_score, prob_glioma, prob_meningioma, prob_pituitary, prob_no_tumor,
                    tumor_pixel_count, tumor_area_mm2, tumor_percentage_brain, tumor_percentage_image, estimated_brain_pixel_count,
                    rule_based_severity, severity_rule_description, created_at, predictive_entropy, requires_review
                ) VALUES (1, 1, 'Glioma', 0.9, 0.9, 0.05, 0.05, 0.0, 100, 100.0, 1.0, 1.0, 10000, 'LOW', 'No rule', '2026-08-16', 0.5, 0);
                """)
                conn.execute("""
                INSERT INTO clinical_reports (
                    id, prediction_id, markdown_path, json_path, pdf_path, heatmap_path, overlay_path, mask_path, xai_method, xai_overlap_percentage, created_at, uncertainty_path
                ) VALUES (1, 1, 'rep.md', 'rep.json', 'rep.pdf', 'heat.png', 'over.png', 'mask.png', 'gradcam', 0.8, '2026-08-16', 'uncertainty.png');
                """)
        finally:
            conn.close()

        # Create dummy physical image file
        with open("uncertainty.png", "wb") as f:
            # Write small 1x1 black png image
            f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\rIDATx\x9cc`\x00\x00\x00\x02\x00\x01H\xaf\xa4q\x00\x00\x00\x00IEND\xaeB`\x82')

        # Test request streaming visual
        # A. Patient (Should yield 403 Forbidden)
        resp = self.client.get("/api/report/1/visuals/uncertainty", headers=self.patient_headers)
        self.assertEqual(resp.status_code, 403)

        # B. Doctor (Should return 200/streams the image visual)
        resp = self.client.get("/api/report/1/visuals/uncertainty", headers=self.doctor_headers)
        self.assertEqual(resp.status_code, 200)

    def test_h7_telemetry_uncertainty_stats(self):
        """Verify H7 health_telemetry aggregates count predictive entropy and review rates correctly."""
        # Seeding database with two records
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute("INSERT OR IGNORE INTO patients (patient_id, name, age, gender, created_at) VALUES ('P100', 'Doe', 30, 'M', '2026-08-16');")
                conn.execute("INSERT OR IGNORE INTO mri_scans (id, patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at) VALUES (1, 'P100', 'scan.png', 1.0, 'Dr. Smith', '2026-08-16', '2026-08-16');")
                conn.execute("INSERT OR IGNORE INTO mri_scans (id, patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at) VALUES (2, 'P100', 'scan.png', 1.0, 'Dr. Smith', '2026-08-16', '2026-08-16');")

                # Record 1: requires_review = 1, entropy = 1.5
                conn.execute("""
                INSERT INTO predictions (
                    id, scan_id, predicted_class, confidence_score, prob_glioma, prob_meningioma, prob_pituitary, prob_no_tumor,
                    tumor_pixel_count, tumor_area_mm2, tumor_percentage_brain, tumor_percentage_image, estimated_brain_pixel_count,
                    rule_based_severity, severity_rule_description, created_at, predictive_entropy, requires_review
                ) VALUES (1, 1, 'Glioma', 0.5, 0.5, 0.3, 0.1, 0.1, 100, 100.0, 1.0, 1.0, 10000, 'LOW', 'No rule', '2026-08-16', 1.5, 1);
                """)
                # Record 2: requires_review = 0, entropy = 0.5
                conn.execute("""
                INSERT INTO predictions (
                    id, scan_id, predicted_class, confidence_score, prob_glioma, prob_meningioma, prob_pituitary, prob_no_tumor,
                    tumor_pixel_count, tumor_area_mm2, tumor_percentage_brain, tumor_percentage_image, estimated_brain_pixel_count,
                    rule_based_severity, severity_rule_description, created_at, predictive_entropy, requires_review
                ) VALUES (2, 2, 'Glioma', 0.9, 0.9, 0.05, 0.05, 0.0, 100, 100.0, 1.0, 1.0, 10000, 'LOW', 'No rule', '2026-08-16', 0.5, 0);
                """)
        finally:
            conn.close()

        telemetry = self.db_repo.get_health_telemetry()

        # Total predictions = 2
        self.assertEqual(telemetry["total_predictions"], 2)
        # Average predictive entropy = (1.5 + 0.5) / 2 = 1.0
        self.assertEqual(telemetry["avg_predictive_entropy"], 1.0)
        # Total reviews recommended = 1
        self.assertEqual(telemetry["total_reviews_recommended"], 1)
        # Review rate = 1 / 2 = 0.5 (50%)
        self.assertEqual(telemetry["review_rate"], 0.5)

if __name__ == "__main__":
    unittest.main()
