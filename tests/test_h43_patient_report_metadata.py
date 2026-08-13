import os
import sqlite3
import datetime
import unittest
from fastapi.testclient import TestClient

os.environ["DB_PATH"] = os.path.abspath("outputs/test_h43_metadata.db")
os.environ["DISABLE_TEST_AUTO_ASSIGN"] = "1"

from run_api import app
from dashboard.infrastructure.web_server import create_app
from security.domain.entities import Role, User
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.jwt_service import JWTService
from persistence.infrastructure.repository import SQLitePersistenceRepository
from security.infrastructure.encryption_service import PIIEncryptionService


class TestH43PatientReportMetadata(unittest.TestCase):
    """Focused integration tests for Phase H4.3-S Patient Report Metadata Hardening."""

    def setUp(self):
        test_method_name = self.id().split('.')[-1]
        self.db_path = os.path.abspath(f"outputs/test_h43_metadata_{test_method_name}.db")
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
                (100, 1, "outputs/reports/pat_a.md", "outputs/reports/pat_a.json", "outputs/reports/pat_a.pdf", "outputs/reports/pat_a_hm.png", "outputs/reports/pat_a_ov.png", "outputs/reports/pat_a_mk.png", datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at, pdf_path, json_path, checksum) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
                (100, "REP-2026-100", "pat-uuid-a", 1, "FINAL", datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat(), "outputs/reports/pat_a.pdf", "outputs/reports/pat_a.json", "dummychecksum")
            )
            conn.execute(
                "INSERT INTO report_versions (version_id, report_id, version_number, created_at, created_by, reason, pdf_path, json_path, checksum, status, prediction_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
                (1001, 100, 1, datetime.datetime.utcnow().isoformat(), "doctor_a@aurascan.ai", "Initial baseline", "outputs/reports/pat_a.pdf", "outputs/reports/pat_a.json", "dummychecksum", "FINAL", 1)
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
                (200, 2, "outputs/reports/pat_b.md", "outputs/reports/pat_b.json", "outputs/reports/pat_b.pdf", None, None, None, datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at, pdf_path, json_path, checksum) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
                (200, "REP-2026-200", "pat-uuid-b", 1, "FINAL", datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat(), "outputs/reports/pat_b.pdf", "outputs/reports/pat_b.json", "dummychecksum")
            )
            conn.execute(
                "INSERT INTO report_versions (version_id, report_id, version_number, created_at, created_by, reason, pdf_path, json_path, checksum, status, prediction_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
                (2001, 200, 1, datetime.datetime.utcnow().isoformat(), "doctor_b@aurascan.ai", "Initial baseline", "outputs/reports/pat_b.pdf", "outputs/reports/pat_b.json", "dummychecksum", "FINAL", 2)
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

        # Restore global state to avoid contaminating other tests (H4.3-S)
        if "DISABLE_TEST_AUTO_ASSIGN" in os.environ:
            del os.environ["DISABLE_TEST_AUTO_ASSIGN"]
        os.environ["DB_PATH"] = os.path.abspath("outputs/clinical_reports.db")

        import api.infrastructure.routes as api_routes
        import api.routes.auth_routes as auth_routes
        api_routes.DEFAULT_DB_PATH = os.path.abspath("outputs/clinical_reports.db")
        auth_routes.DEFAULT_DB_PATH = os.path.abspath("outputs/clinical_reports.db")

    def test_patient_can_access_own_report_metadata(self):
        """Verify that Patient A can retrieve metadata of their own report (Flask & FastAPI)."""
        # Flask: /api/report/100
        self.flask_client.set_cookie("access_token", self.token_pat_a)
        res = self.flask_client.get("/api/report/100")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["report_id"], 100)
        self.assertEqual(data["patient_id"], "pat-uuid-a")

        # FastAPI: /api/reports/100
        headers = {"Authorization": f"Bearer {self.token_pat_a}"}
        res_fast = self.fastapi_client.get("/api/reports/100", headers=headers)
        self.assertEqual(res_fast.status_code, 200)
        data_fast = res_fast.json()
        self.assertEqual(data_fast["report_id"], 100)
        self.assertEqual(data_fast["patient_id"], "pat-uuid-a")

    def test_patient_cannot_access_another_patients_report(self):
        """Verify IDOR protection: Patient A is blocked from Patient B's report metadata (Flask & FastAPI)."""
        # Flask: /api/report/200
        self.flask_client.set_cookie("access_token", self.token_pat_a)
        res = self.flask_client.get("/api/report/200")
        self.assertEqual(res.status_code, 403)  # Access Denied

        # FastAPI: /api/reports/200
        headers = {"Authorization": f"Bearer {self.token_pat_a}"}
        res_fast = self.fastapi_client.get("/api/reports/200", headers=headers)
        self.assertEqual(res_fast.status_code, 403)  # Access Denied

    def test_patient_response_contains_no_internal_paths(self):
        """Verify that patient response excludes pdf_path and json_path from metadata and versions list."""
        # --- Flask Details ---
        self.flask_client.set_cookie("access_token", self.token_pat_a)
        res = self.flask_client.get("/api/report/100")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        self.assertNotIn("pdf_path", data)
        self.assertNotIn("json_path", data)
        self.assertTrue(len(data["versions"]) > 0)
        for v in data["versions"]:
            self.assertNotIn("pdf_path", v)
            self.assertNotIn("json_path", v)

        # --- FastAPI Details ---
        headers = {"Authorization": f"Bearer {self.token_pat_a}"}
        res_fast = self.fastapi_client.get("/api/reports/100", headers=headers)
        self.assertEqual(res_fast.status_code, 200)
        data_fast = res_fast.json()

        self.assertNotIn("pdf_path", data_fast)
        self.assertNotIn("json_path", data_fast)
        self.assertTrue(len(data_fast["versions"]) > 0)
        for v in data_fast["versions"]:
            self.assertNotIn("pdf_path", v)
            self.assertNotIn("json_path", v)

        # --- FastAPI Versions List ---
        res_fast_v = self.fastapi_client.get("/api/reports/100/versions", headers=headers)
        self.assertEqual(res_fast_v.status_code, 200)
        data_v = res_fast_v.json()
        self.assertTrue(len(data_v) > 0)
        for v in data_v:
            self.assertNotIn("pdf_path", v)
            self.assertNotIn("json_path", v)

    def test_doctor_access_remains_functional_with_internal_paths(self):
        """Verify that Doctor A (clinician) can view Patient A's report metadata, including pdf_path and json_path."""
        # Flask: /api/report/100
        self.flask_client.set_cookie("access_token", self.token_doc_a)
        res = self.flask_client.get("/api/report/100")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(len(data["versions"]) > 0)
        # Doctor gets versions with internal filesystem paths intact
        for v in data["versions"]:
            self.assertIn("pdf_path", v)
            self.assertIn("json_path", v)
            self.assertEqual(v["pdf_path"], "outputs/reports/pat_a.pdf")
            self.assertEqual(v["json_path"], "outputs/reports/pat_a.json")

        # FastAPI Details: /api/reports/100
        headers = {"Authorization": f"Bearer {self.token_doc_a}"}
        res_fast = self.fastapi_client.get("/api/reports/100", headers=headers)
        self.assertEqual(res_fast.status_code, 200)
        data_fast = res_fast.json()
        self.assertIn("pdf_path", data_fast)
        self.assertIn("json_path", data_fast)
        self.assertEqual(data_fast["pdf_path"], "outputs/reports/pat_a.pdf")
        self.assertEqual(data_fast["json_path"], "outputs/reports/pat_a.json")

        # FastAPI Versions List: /api/reports/100/versions
        res_fast_v = self.fastapi_client.get("/api/reports/100/versions", headers=headers)
        self.assertEqual(res_fast_v.status_code, 200)
        data_v = res_fast_v.json()
        for v in data_v:
            self.assertIn("pdf_path", v)
            self.assertIn("json_path", v)

    def test_unauthenticated_access_remains_blocked(self):
        """Verify that request without authentication token is blocked (401)."""
        # Flask
        res = self.flask_client.get("/api/report/100")
        self.assertEqual(res.status_code, 401)

        # FastAPI
        res_fast = self.fastapi_client.get("/api/reports/100")
        self.assertEqual(res_fast.status_code, 401)

        res_fast_v = self.fastapi_client.get("/api/reports/100/versions")
        self.assertEqual(res_fast_v.status_code, 401)
