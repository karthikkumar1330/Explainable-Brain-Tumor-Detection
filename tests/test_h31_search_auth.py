import os
import sqlite3
import datetime
import unittest
from fastapi.testclient import TestClient

# Configure test database environment
os.environ["DB_PATH"] = os.path.abspath("outputs/test_h31_search_auth.db")
os.environ["DISABLE_TEST_AUTO_ASSIGN"] = "1"

from run_api import app
from dashboard.infrastructure.web_server import create_app
from security.domain.entities import Role, User
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.jwt_service import JWTService
from security.application.authorization_service import AuthorizationService
from persistence.infrastructure.repository import SQLitePersistenceRepository
from clinical_reporting.application.services import ReportService

class TestH31SearchAuth(unittest.TestCase):
    """Focused integration tests for Phase H3.1 search authorization hardening."""

    def setUp(self):
        self.db_path = os.environ.get("DB_PATH", "outputs/test_h31_search_auth.db")
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

        # Make sure FastAPI and routes default paths point to our test database
        import api.infrastructure.routes as api_routes
        import api.routes.auth_routes as auth_routes
        api_routes.DEFAULT_DB_PATH = self.db_path
        auth_routes.DEFAULT_DB_PATH = self.db_path

        # Configure Flask App
        self.flask_app = create_app(db_path=self.db_path)
        self.flask_app.config["TESTING"] = True
        self.flask_app.config["DISABLE_CSRF"] = True
        self.flask_client = self.flask_app.test_client()

        # Initialize DB
        self.persistence_repo = SQLitePersistenceRepository(db_path=self.db_path)
        self.persistence_repo.initialize_db()

        self.user_repo = SQLiteUserRepository(db_path=self.db_path)
        self.user_repo.initialize_security_tables()

        # Drop the auto-assignment triggers to ensure strict clean testing
        conn_drop = sqlite3.connect(self.db_path)
        try:
            conn_drop.execute("DROP TRIGGER IF EXISTS auto_assign_doctor_on_insert;")
            conn_drop.execute("DROP TRIGGER IF EXISTS auto_assign_patient_on_insert;")
            conn_drop.execute("DELETE FROM doctor_patient_assignments;")
            conn_drop.commit()
        finally:
            conn_drop.close()

        # Bootstrap users
        self.admin_user = self.user_repo.bootstrap_admin()

        from security.infrastructure.password import PasswordHasher
        pass_hash = PasswordHasher.hash_password("Password@123")

        conn = sqlite3.connect(self.db_path)
        try:
            # Users
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (10, "doc-uuid-a", "doctor_a@aurascan.ai", pass_hash, "Dr. Alice", Role.DOCTOR.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (11, "doc-uuid-b", "doctor_b@aurascan.ai", pass_hash, "Dr. Bob", Role.DOCTOR.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (20, "pat-uuid-a", "patient_a@aurascan.ai", pass_hash, "Patient A", Role.PATIENT.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (21, "pat-uuid-b", "patient_b@aurascan.ai", pass_hash, "Patient B", Role.PATIENT.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )

            # Patients
            conn.execute(
                "INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
                ("pat-uuid-a", "Patient A", 45, "Male", datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
                ("pat-uuid-b", "Patient B", 30, "Female", datetime.datetime.utcnow().isoformat())
            )

            # MRI Scans: Scan A (100) -> Patient A, Scan B (101) -> Patient B
            conn.execute(
                "INSERT INTO mri_scans (id, patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                (100, "pat-uuid-a", "uploads/scan_a.png", 1.0, "Dr. Alice", "2026-08-08", datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO mri_scans (id, patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                (101, "pat-uuid-b", "uploads/scan_b.png", 1.0, "Dr. Bob", "2026-08-08", datetime.datetime.utcnow().isoformat())
            )

            # Predictions: Prediction A (200) -> Scan A, Prediction B (201) -> Scan B
            conn.execute(
                """INSERT INTO predictions (
                    id, scan_id, predicted_class, confidence_score, prob_glioma, prob_meningioma,
                    prob_pituitary, prob_no_tumor, tumor_pixel_count, tumor_area_mm2,
                    tumor_percentage_brain, tumor_percentage_image, estimated_brain_pixel_count,
                    rule_based_severity, severity_rule_description, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
                (200, 100, "Glioma", 0.95, 0.95, 0.02, 0.02, 0.01, 500, 50.0, 5.0, 2.0, 10000, "High", "High severity", datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                """INSERT INTO predictions (
                    id, scan_id, predicted_class, confidence_score, prob_glioma, prob_meningioma,
                    prob_pituitary, prob_no_tumor, tumor_pixel_count, tumor_area_mm2,
                    tumor_percentage_brain, tumor_percentage_image, estimated_brain_pixel_count,
                    rule_based_severity, severity_rule_description, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
                (201, 101, "Meningioma", 0.90, 0.05, 0.90, 0.03, 0.02, 300, 30.0, 3.0, 1.0, 10000, "Medium", "Medium severity", datetime.datetime.utcnow().isoformat())
            )

            # Clinical Reports: Report A (300) -> Prediction A, Report B (301) -> Prediction B
            conn.execute(
                "INSERT INTO clinical_reports (id, prediction_id, markdown_path, json_path, pdf_path, created_at) VALUES (?, ?, ?, ?, ?, ?);",
                (300, 200, "outputs/clinical_reports/rep_bob.md", "outputs/clinical_reports/rep_bob.json", "outputs/clinical_reports/rep_bob.pdf", datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO clinical_reports (id, prediction_id, markdown_path, json_path, pdf_path, created_at) VALUES (?, ?, ?, ?, ?, ?);",
                (301, 201, "outputs/clinical_reports/rep_alice.md", "outputs/clinical_reports/rep_alice.json", "outputs/clinical_reports/rep_alice.pdf", datetime.datetime.utcnow().isoformat())
            )

            # Reports Table
            conn.execute(
                "INSERT INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                (300, "REP-2026-0001", "pat-uuid-a", 1, "FINAL", "2026-08-08", "2026-08-08")
            )
            conn.execute(
                "INSERT INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                (301, "REP-2026-0002", "pat-uuid-b", 1, "FINAL", "2026-08-08", "2026-08-08")
            )

            # Assignments: Doctor A (10) -> Patient A, Doctor B (11) -> Patient B
            conn.execute(
                "INSERT INTO doctor_patient_assignments (doctor_id, patient_id, created_at) VALUES (?, ?, ?);",
                (10, "pat-uuid-a", datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO doctor_patient_assignments (doctor_id, patient_id, created_at) VALUES (?, ?, ?);",
                (11, "pat-uuid-b", datetime.datetime.utcnow().isoformat())
            )

            conn.commit()
        finally:
            conn.close()

        # Load users
        self.doc_a = self.user_repo.get_by_email("doctor_a@aurascan.ai")
        self.doc_b = self.user_repo.get_by_email("doctor_b@aurascan.ai")
        self.pat_a = self.user_repo.get_by_email("patient_a@aurascan.ai")
        self.pat_b = self.user_repo.get_by_email("patient_b@aurascan.ai")

        # JWT Tokens
        self.jwt_svc = JWTService()
        self.token_doc_a = self.jwt_svc.create_access_token(self.doc_a.uuid, self.doc_a.id, self.doc_a.email, Role.DOCTOR)
        self.token_doc_b = self.jwt_svc.create_access_token(self.doc_b.uuid, self.doc_b.id, self.doc_b.email, Role.DOCTOR)
        self.token_pat_a = self.jwt_svc.create_access_token(self.pat_a.uuid, self.pat_a.id, self.pat_a.email, Role.PATIENT)
        self.token_pat_b = self.jwt_svc.create_access_token(self.pat_b.uuid, self.pat_b.id, self.pat_b.email, Role.PATIENT)

        self.fastapi_ctx = TestClient(app)
        self.fastapi_client = self.fastapi_ctx.__enter__()

    def tearDown(self):
        self.fastapi_ctx.__exit__(None, None, None)
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def _get_headers(self, token):
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

    def test_01_assigned_doctor_can_search_assigned_patient(self):
        """1. Assigned doctor can search an assigned patient."""
        # Flask search
        headers = self._get_headers(self.token_doc_a)
        resp = self.flask_client.get("/api/search", headers=headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(len(data) > 0)
        patient_ids = [item["patient_id"] for item in data]
        self.assertIn("pat-uuid-a", patient_ids)

    def test_02_unassigned_patient_metadata_does_not_appear_in_search(self):
        """2. Unassigned patient's name/ID/diagnosis/report/scan information does NOT appear in /api/search."""
        headers = self._get_headers(self.token_doc_a)
        resp = self.flask_client.get("/api/search", headers=headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()

        # Verify no mention of Patient B or related fields
        for item in data:
            self.assertNotEqual(item["patient_id"], "pat-uuid-b")
            self.assertNotEqual(item["patient_name"], "Patient B")
            self.assertNotEqual(item["predicted_class"], "Meningioma")
            self.assertNotEqual(item["report_id"], 301)

    def test_03_unassigned_patient_cannot_be_retrieved_directly(self):
        """3. Unassigned patient cannot be retrieved through direct patient/history access."""
        # Direct Flask report details
        headers = self._get_headers(self.token_doc_a)
        resp = self.flask_client.get("/api/report/301", headers=headers)
        self.assertEqual(resp.status_code, 403)

    def test_04_doctor_to_doctor_isolation(self):
        """4. Doctor A cannot see Doctor B's assigned patients."""
        # Doctor A query
        headers_a = self._get_headers(self.token_doc_a)
        resp_a = self.flask_client.get("/api/search", headers=headers_a)
        p_ids_a = [item["patient_id"] for item in resp_a.get_json()]
        self.assertIn("pat-uuid-a", p_ids_a)
        self.assertNotIn("pat-uuid-b", p_ids_a)

        # Doctor B query
        headers_b = self._get_headers(self.token_doc_b)
        resp_b = self.flask_client.get("/api/search", headers=headers_b)
        p_ids_b = [item["patient_id"] for item in resp_b.get_json()]
        self.assertIn("pat-uuid-b", p_ids_b)
        self.assertNotIn("pat-uuid-a", p_ids_b)

    def test_05_patient_accounts_cannot_access_doctor_only_search(self):
        """5. Patient accounts cannot access other patient data through search/history."""
        # Patient A query
        headers = self._get_headers(self.token_pat_a)
        resp = self.flask_client.get("/api/search", headers=headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()

        # Verify response is strictly scoped to the authenticated patient's own records and contains no other patient's data
        self.assertTrue(len(data) > 0, "Patient A should see their own records.")
        for item in data:
            self.assertEqual(item["patient_id"], "pat-uuid-a", "Search response contains records belonging to another patient.")
            self.assertEqual(item["patient_name"], "Patient A", "Search response contains name belonging to another patient.")
            self.assertNotEqual(item["patient_id"], "pat-uuid-b", "Search response leaks Patient B's ID.")
            self.assertNotEqual(item["patient_name"], "Patient B", "Search response leaks Patient B's name.")
            self.assertNotEqual(item["predicted_class"], "Meningioma", "Search response leaks Patient B's predicted class.")
            self.assertNotEqual(item["report_id"], 301, "Search response leaks Patient B's report ID.")

    def test_06_api_history_respects_doctor_assignment_filtering(self):
        """6. /api/history respects doctor assignment filtering."""
        headers = self._get_headers(self.token_doc_a)
        resp = self.flask_client.get("/api/history", headers=headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        patient_ids = [item["patient_id"] for item in data]
        self.assertIn("pat-uuid-a", patient_ids)
        self.assertNotIn("pat-uuid-b", patient_ids)

    def test_06b_api_history_explicit_patient_filtering(self):
        """6b. /api/history returns empty response when Doctor A requests an unassigned or nonexistent patient."""
        headers = self._get_headers(self.token_doc_a)

        # Doctor A queries for an explicitly requested unassigned patient (pat-uuid-b) -> empty result
        resp_unassigned = self.flask_client.get("/api/history?patient_id=pat-uuid-b", headers=headers)
        self.assertEqual(resp_unassigned.status_code, 200)
        data_unassigned = resp_unassigned.get_json()
        self.assertEqual(len(data_unassigned), 0, "Querying an unassigned patient should return empty history list.")

        # Doctor A queries for a nonexistent patient (pat-uuid-none) -> empty result
        resp_nonexistent = self.flask_client.get("/api/history?patient_id=pat-uuid-none", headers=headers)
        self.assertEqual(resp_nonexistent.status_code, 200)
        data_nonexistent = resp_nonexistent.get_json()
        self.assertEqual(len(data_nonexistent), 0, "Querying a nonexistent patient should return empty history list.")

        # Both responses should be identical (empty list)
        self.assertEqual(data_unassigned, data_nonexistent)

    def test_07_api_database_history_respects_doctor_assignment_filtering(self):
        """7. /api/database/history (FastAPI) respects doctor assignment filtering."""
        headers = self._get_headers(self.token_doc_a)
        resp = self.fastapi_client.get("/api/database/history", headers=headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        patient_ids = [item["patient_id"] for item in data]
        self.assertIn("pat-uuid-a", patient_ids)
        self.assertNotIn("pat-uuid-b", patient_ids)

    def test_07b_api_database_history_explicit_patient_filtering(self):
        """7b. /api/database/history (FastAPI) returns 403 for unauthorized/nonexistent patient queries."""
        headers = self._get_headers(self.token_doc_a)

        # Doctor A queries for an unassigned patient (pat-uuid-b) -> 403 Forbidden
        resp_unassigned = self.fastapi_client.get("/api/database/history?patient_id=pat-uuid-b", headers=headers)
        self.assertEqual(resp_unassigned.status_code, 403)
        self.assertEqual(resp_unassigned.json()["detail"], "Access denied to patient records.")

        # Doctor A queries for a nonexistent patient (pat-uuid-none) -> 403 Forbidden
        resp_nonexistent = self.fastapi_client.get("/api/database/history?patient_id=pat-uuid-none", headers=headers)
        self.assertEqual(resp_nonexistent.status_code, 403)
        self.assertEqual(resp_nonexistent.json()["detail"], "Access denied to patient records.")

        # Patient A queries for another patient (pat-uuid-b) -> should be overridden to pat-uuid-a and return 200 with patient A records
        headers_pat = self._get_headers(self.token_pat_a)
        resp_pat = self.fastapi_client.get("/api/database/history?patient_id=pat-uuid-b", headers=headers_pat)
        self.assertEqual(resp_pat.status_code, 200)
        data_pat = resp_pat.json()
        self.assertTrue(len(data_pat) > 0)
        for item in data_pat:
            self.assertEqual(item["patient_id"], "pat-uuid-a")

    def test_08_existing_assigned_patient_functionality_passes(self):
        """8. Existing assigned-patient functionality still passes."""
        # Get notes for assigned scan
        headers = self._get_headers(self.token_doc_a)
        resp = self.flask_client.get("/api/doctor/scans/100/clinical-notes", headers=headers)
        self.assertEqual(resp.status_code, 200)

    def test_09_empty_search_results_do_not_leak_existence(self):
        """9. Empty search results do not leak whether an unassigned patient exists."""
        headers = self._get_headers(self.token_doc_a)

        # Query unassigned patient ID explicitly
        resp_unassigned = self.flask_client.get("/api/search?patient_id=pat-uuid-b", headers=headers)
        self.assertEqual(resp_unassigned.status_code, 200)
        data_unassigned = resp_unassigned.get_json()
        self.assertEqual(len(data_unassigned), 0)

        # Query nonexistent patient ID explicitly
        resp_nonexistent = self.flask_client.get("/api/search?patient_id=pat-uuid-none", headers=headers)
        self.assertEqual(resp_nonexistent.status_code, 200)
        data_nonexistent = resp_nonexistent.get_json()
        self.assertEqual(len(data_nonexistent), 0)

        # Confirm the response format is identical (no side-channels or error differences)
        self.assertEqual(data_unassigned, data_nonexistent)

    def test_10_nonexistent_patient_behavior(self):
        """10. Nonexistent patient behavior remains correct."""
        headers = self._get_headers(self.token_doc_a)
        # Direct lookup of nonexistent report ID
        resp = self.flask_client.get("/api/report/9999", headers=headers)
        self.assertEqual(resp.status_code, 404)

if __name__ == "__main__":
    unittest.main()
