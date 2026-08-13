import os
import sqlite3
import datetime
import json
import unittest
from fastapi.testclient import TestClient

os.environ["DB_PATH"] = os.path.abspath("outputs/test_h44_export.db")
os.environ["DISABLE_TEST_AUTO_ASSIGN"] = "1"

from run_api import app
from dashboard.infrastructure.web_server import create_app
from security.domain.entities import Role, User
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.jwt_service import JWTService
from persistence.infrastructure.repository import SQLitePersistenceRepository
from security.infrastructure.encryption_service import PIIEncryptionService


class TestH44AIResultsExport(unittest.TestCase):
    """Focused integration tests for Phase H4.4 Patient AI Results Sanitization and Export."""

    def setUp(self):
        test_method_name = self.id().split('.')[-1]
        self.db_path = os.path.abspath(f"outputs/test_h44_export_{test_method_name}.db")
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

        os.environ["DB_PATH"] = self.db_path
        os.environ["DISABLE_TEST_AUTO_ASSIGN"] = "1"

        # Update route db paths
        import api.infrastructure.routes as api_routes
        import api.routes.auth_routes as auth_routes
        api_routes.DEFAULT_DB_PATH = self.db_path
        auth_routes.DEFAULT_DB_PATH = self.db_path

        # Flask App setup
        self.flask_app = create_app(db_path=self.db_path)
        self.flask_app.config["TESTING"] = True
        self.flask_app.config["DISABLE_CSRF"] = True
        self.flask_client = self.flask_app.test_client()

        # DB Setup
        self.persistence_repo = SQLitePersistenceRepository(db_path=self.db_path)
        self.persistence_repo.initialize_db()
        self.user_repo = SQLiteUserRepository(db_path=self.db_path)
        self.user_repo.initialize_security_tables()

        # Drop auto-assignment triggers
        conn_drop = sqlite3.connect(self.db_path)
        try:
            conn_drop.execute("DROP TRIGGER IF EXISTS auto_assign_doctor_on_insert;")
            conn_drop.execute("DROP TRIGGER IF EXISTS auto_assign_patient_on_insert;")
            conn_drop.execute("DELETE FROM doctor_patient_assignments;")
            conn_drop.commit()
        finally:
            conn_drop.close()

        # Create mock JSON directories and files
        os.makedirs("outputs/clinical_reports", exist_ok=True)
        self.mock_json_a = os.path.abspath("outputs/clinical_reports/pat_a_clinical_report.json")
        self.mock_json_b = os.path.abspath("outputs/clinical_reports/pat_b_clinical_report.json")

        self.payload_a = {
            "patient": {
                "patient_id": "pat-uuid-a",
                "name": "Patient A Name",
                "age": "45",
                "gender": "Male",
                "scan_date": "2026-08-01"
            },
            "processing": {
                "device": "cpu",
                "total_execution_time_sec": 0.5,
                "classification_model": "models/classification/efficientnet_b0_brain_tumor.pth",
                "segmentation_model": "models/segmentation/unext.pth"
            },
            "classification": {
                "predicted_class": "Glioma",
                "confidence_score": 0.95
            },
            "files": {
                "original_image": "outputs/scans/pat_a.png",
                "heatmap_image": "outputs/reports/pat_a_hm.png",
                "overlay_image": "outputs/reports/pat_a_ov.png"
            }
        }

        self.payload_b = {
            "patient": {
                "patient_id": "pat-uuid-b",
                "name": "Patient B Name",
                "age": "30",
                "gender": "Female",
                "scan_date": "2026-08-02"
            },
            "processing": {
                "device": "cpu",
                "total_execution_time_sec": 0.3,
                "classification_model": "models/classification/efficientnet_b0_brain_tumor.pth",
                "segmentation_model": "models/segmentation/unext.pth"
            },
            "classification": {
                "predicted_class": "No Tumor",
                "confidence_score": 0.99
            },
            "files": {
                "original_image": "outputs/scans/pat_b.png"
            }
        }

        with open(self.mock_json_a, "w", encoding="utf-8") as f:
            json.dump(self.payload_a, f)
        with open(self.mock_json_b, "w", encoding="utf-8") as f:
            json.dump(self.payload_b, f)

        # Setup passwords
        from security.infrastructure.password import PasswordHasher
        pass_hash = PasswordHasher.hash_password("Password@123")

        # Bootstrap Database Users
        conn = sqlite3.connect(self.db_path)
        try:
            # 1. Doctors
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (200, "doc-uuid-a", "doctor_a@aurascan.ai", pass_hash, "Doctor A", Role.DOCTOR.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            # 2. Patients
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (300, "pat-uuid-a", "patient_a@aurascan.ai", pass_hash, "Patient A", Role.PATIENT.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (301, "pat-uuid-b", "patient_b@aurascan.ai", pass_hash, "Patient B", Role.PATIENT.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            # 3. Patient Demographics (Encrypted)
            enc_svc = PIIEncryptionService()
            conn.execute(
                "INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
                ("pat-uuid-a", enc_svc.encrypt("Patient A Name"), enc_svc.encrypt("45"), enc_svc.encrypt("Male"), datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
                ("pat-uuid-b", enc_svc.encrypt("Patient B Name"), enc_svc.encrypt("30"), enc_svc.encrypt("Female"), datetime.datetime.utcnow().isoformat())
            )
            # 4. Assignments: Doctor A is assigned to Patient A
            conn.execute(
                "INSERT INTO doctor_patient_assignments (doctor_id, patient_id, created_at) VALUES (?, ?, ?);",
                (200, "pat-uuid-a", datetime.datetime.utcnow().isoformat())
            )

            # 5. Seed Patient A Scans, Predictions, and Reports
            conn.execute(
                "INSERT INTO mri_scans (id, patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                (1, "pat-uuid-a", "outputs/scans/pat_a.png", 1.0, "Dr. Sarah", "2026-08-01", datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO predictions (id, scan_id, predicted_class, confidence_score, prob_glioma, prob_meningioma, prob_pituitary, prob_no_tumor, tumor_pixel_count, tumor_area_mm2, tumor_percentage_brain, tumor_percentage_image, estimated_brain_pixel_count, rule_based_severity, severity_rule_description, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
                (1, 1, "Glioma", 0.95, 0.95, 0.02, 0.02, 0.01, 500, 50.0, 1.2, 0.8, 40000, "HIGH", "High severity detected", datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO clinical_reports (id, prediction_id, markdown_path, json_path, pdf_path, heatmap_path, overlay_path, mask_path, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);",
                (100, 1, "outputs/reports/pat_a.md", self.mock_json_a, "outputs/reports/pat_a.pdf", "outputs/reports/pat_a_hm.png", "outputs/reports/pat_a_ov.png", "outputs/reports/pat_a_mk.png", datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at, pdf_path, json_path, checksum) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
                (100, "REP-2026-100", "pat-uuid-a", 1, "FINAL", datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat(), "outputs/reports/pat_a.pdf", self.mock_json_a, "dummychecksum")
            )
            conn.execute(
                "INSERT INTO report_versions (version_id, report_id, version_number, created_at, created_by, reason, pdf_path, json_path, checksum, status, prediction_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
                (1001, 100, 1, datetime.datetime.utcnow().isoformat(), "doctor_a@aurascan.ai", "Initial baseline", "outputs/reports/pat_a.pdf", self.mock_json_a, "dummychecksum", "FINAL", 1)
            )

            # 6. Seed Patient B Scans, Predictions, and Reports
            conn.execute(
                "INSERT INTO mri_scans (id, patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                (2, "pat-uuid-b", "outputs/scans/pat_b.png", 1.0, "Dr. Sarah", "2026-08-02", datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO predictions (id, scan_id, predicted_class, confidence_score, prob_glioma, prob_meningioma, prob_pituitary, prob_no_tumor, tumor_pixel_count, tumor_area_mm2, tumor_percentage_brain, tumor_percentage_image, estimated_brain_pixel_count, rule_based_severity, severity_rule_description, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
                (2, 2, "No Tumor", 0.99, 0.01, 0.01, 0.01, 0.97, 0, 0.0, 0.0, 0.0, 42000, "LOW", "No tumor detected", datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO clinical_reports (id, prediction_id, markdown_path, json_path, pdf_path, heatmap_path, overlay_path, mask_path, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);",
                (200, 2, "outputs/reports/pat_b.md", self.mock_json_b, "outputs/reports/pat_b.pdf", None, None, None, datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at, pdf_path, json_path, checksum) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
                (200, "REP-2026-200", "pat-uuid-b", 1, "FINAL", datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat(), "outputs/reports/pat_b.pdf", self.mock_json_b, "dummychecksum")
            )
            conn.execute(
                "INSERT INTO report_versions (version_id, report_id, version_number, created_at, created_by, reason, pdf_path, json_path, checksum, status, prediction_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
                (2001, 200, 1, datetime.datetime.utcnow().isoformat(), "doctor_b@aurascan.ai", "Initial baseline", "outputs/reports/pat_b.pdf", self.mock_json_b, "dummychecksum", "FINAL", 2)
            )

            conn.commit()
        finally:
            conn.close()

        # Initialize JWT Tokens
        self.jwt_svc = JWTService()
        self.token_doc_a = self.jwt_svc.create_access_token("doc-uuid-a", 200, "doctor_a@aurascan.ai", Role.DOCTOR)
        self.token_pat_a = self.jwt_svc.create_access_token("pat-uuid-a", 300, "patient_a@aurascan.ai", Role.PATIENT)
        self.token_pat_b = self.jwt_svc.create_access_token("pat-uuid-b", 301, "patient_b@aurascan.ai", Role.PATIENT)

        # Setup FastAPI Client
        self.fastapi_ctx = TestClient(app)
        self.fastapi_client = self.fastapi_ctx.__enter__()

    def tearDown(self):
        self.fastapi_ctx.__exit__(None, None, None)
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

        if os.path.exists(self.mock_json_a):
            os.remove(self.mock_json_a)
        if os.path.exists(self.mock_json_b):
            os.remove(self.mock_json_b)

        # Restore global state to avoid contaminating other tests (H4.4)
        if "DISABLE_TEST_AUTO_ASSIGN" in os.environ:
            del os.environ["DISABLE_TEST_AUTO_ASSIGN"]
        os.environ["DB_PATH"] = os.path.abspath("outputs/clinical_reports.db")

        import api.infrastructure.routes as api_routes
        import api.routes.auth_routes as auth_routes
        api_routes.DEFAULT_DB_PATH = os.path.abspath("outputs/clinical_reports.db")
        auth_routes.DEFAULT_DB_PATH = os.path.abspath("outputs/clinical_reports.db")

    def test_patient_json_export_is_sanitized(self):
        """Verify that Patient A own JSON export strips 'files' and internal model paths."""
        self.flask_client.set_cookie("access_token", self.token_pat_a)
        res = self.flask_client.get("/api/report/100/json")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        # Files list must be completely stripped
        self.assertNotIn("files", data)
        # Model path details must be popped from processing
        self.assertIn("processing", data)
        self.assertNotIn("classification_model", data["processing"])
        self.assertNotIn("segmentation_model", data["processing"])
        # Basic processing metrics are kept
        self.assertEqual(data["processing"]["device"], "cpu")
        self.assertEqual(data["processing"]["total_execution_time_sec"], 0.5)

    def test_doctor_json_export_is_fully_enriched(self):
        """Verify that Doctor A (clinician) receives full JSON details including file paths and model checkpoint paths."""
        self.flask_client.set_cookie("access_token", self.token_doc_a)
        res = self.flask_client.get("/api/report/100/json")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        # Clinician gets files paths
        self.assertIn("files", data)
        self.assertEqual(data["files"]["original_image"], "outputs/scans/pat_a.png")
        self.assertEqual(data["files"]["heatmap_image"], "outputs/reports/pat_a_hm.png")
        # Clinician gets model checkpoint paths
        self.assertIn("processing", data)
        self.assertEqual(data["processing"]["classification_model"], "models/classification/efficientnet_b0_brain_tumor.pth")
        self.assertEqual(data["processing"]["segmentation_model"], "models/segmentation/unext.pth")

    def test_patient_unauthorized_json_export_is_blocked(self):
        """Verify that Patient A is blocked from exporting Patient B's report JSON (IDOR boundary check)."""
        self.flask_client.set_cookie("access_token", self.token_pat_a)
        res = self.flask_client.get("/api/report/200/json")
        self.assertEqual(res.status_code, 403)  # Access Denied

    def test_unauthenticated_export_is_blocked(self):
        """Verify that requests without access token are rejected (401)."""
        res = self.flask_client.get("/api/report/100/json")
        self.assertEqual(res.status_code, 401)
