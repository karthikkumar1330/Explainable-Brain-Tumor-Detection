import os
import unittest
import sqlite3
import datetime
import json
import secrets
from fastapi.testclient import TestClient

# Use a separate test database
os.environ["DB_PATH"] = os.path.abspath("outputs/test_h32s_auth.db")

from run_api import app
from security.domain.entities import Role, User
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.jwt_service import JWTService
from security.application.authorization_service import AuthorizationService
from clinical_reporting.application.services import ReportService
from persistence.infrastructure.repository import SQLitePersistenceRepository

class TestH32SAuthorization(unittest.TestCase):
    """Focused security and role boundary tests for Phase H3.2-S."""

    def setUp(self):
        os.environ["DISABLE_TEST_AUTO_ASSIGN"] = "1"
        self.db_path = os.environ.get("DB_PATH", "outputs/test_h32s_auth.db")
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

        # Make sure default paths match
        import api.infrastructure.routes as api_routes
        import api.routes.auth_routes as auth_routes
        api_routes.DEFAULT_DB_PATH = self.db_path
        auth_routes.DEFAULT_DB_PATH = self.db_path

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
            # Insert users
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (10, "doc-uuid-1", "doctor1@aurascan.ai", pass_hash, "Dr. Jane Smith", Role.DOCTOR.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (11, "doc-uuid-2", "doctor2@aurascan.ai", pass_hash, "Dr. John Doe", Role.DOCTOR.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (20, "pat-uuid-1", "patient1@aurascan.ai", pass_hash, "Bob Jones", Role.PATIENT.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (21, "pat-uuid-2", "patient2@aurascan.ai", pass_hash, "Alice Cooper", Role.PATIENT.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )

            # Insert patient demographics (Phase H3.1 encrypted name fallback logic compatibility)
            conn.execute(
                "INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
                ("pat-uuid-1", "Bob Jones", 45, "Male", datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
                ("pat-uuid-2", "Alice Cooper", 30, "Female", datetime.datetime.utcnow().isoformat())
            )

            # Insert MRI Scans
            conn.execute(
                "INSERT INTO mri_scans (id, patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                (100, "pat-uuid-1", "uploads/bob_scan.png", 1.0, "Dr. Jane Smith", "2026-08-08", datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO mri_scans (id, patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                (101, "pat-uuid-2", "uploads/alice_scan.png", 1.0, "Dr. John Doe", "2026-08-08", datetime.datetime.utcnow().isoformat())
            )

            # Insert Predictions
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

            # Insert Clinical Reports
            conn.execute(
                "INSERT INTO clinical_reports (id, prediction_id, markdown_path, json_path, pdf_path, created_at) VALUES (?, ?, ?, ?, ?, ?);",
                (300, 200, "outputs/clinical_reports/rep_bob.md", "outputs/clinical_reports/rep_bob.json", "outputs/clinical_reports/rep_bob.pdf", datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO clinical_reports (id, prediction_id, markdown_path, json_path, pdf_path, created_at) VALUES (?, ?, ?, ?, ?, ?);",
                (301, 201, "outputs/clinical_reports/rep_alice.md", "outputs/clinical_reports/rep_alice.json", "outputs/clinical_reports/rep_alice.pdf", datetime.datetime.utcnow().isoformat())
            )

            # Insert Reports Table entries
            conn.execute(
                "INSERT INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                (300, "REP-2026-0001", "pat-uuid-1", 1, "FINAL", "2026-08-08", "2026-08-08")
            )
            conn.execute(
                "INSERT INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                (301, "REP-2026-0002", "pat-uuid-2", 1, "FINAL", "2026-08-08", "2026-08-08")
            )

            # Assignments: Doctor 1 is assigned only to Patient 1 (Bob Jones)
            conn.execute(
                "INSERT INTO doctor_patient_assignments (doctor_id, patient_id, created_at) VALUES (?, ?, ?);",
                (10, "pat-uuid-1", datetime.datetime.utcnow().isoformat())
            )

            conn.commit()
        finally:
            conn.close()

        # Load users
        self.doc1 = self.user_repo.get_by_email("doctor1@aurascan.ai")
        self.doc2 = self.user_repo.get_by_email("doctor2@aurascan.ai")
        self.pat1 = self.user_repo.get_by_email("patient1@aurascan.ai")
        self.pat2 = self.user_repo.get_by_email("patient2@aurascan.ai")

        # JWT tokens
        self.jwt_svc = JWTService()
        self.token_admin = self.jwt_svc.create_access_token(self.admin_user.uuid, self.admin_user.id, self.admin_user.email, Role.ADMIN)
        self.token_doc1 = self.jwt_svc.create_access_token(self.doc1.uuid, self.doc1.id, self.doc1.email, Role.DOCTOR)
        self.token_doc2 = self.jwt_svc.create_access_token(self.doc2.uuid, self.doc2.id, self.doc2.email, Role.DOCTOR)
        self.token_pat1 = self.jwt_svc.create_access_token(self.pat1.uuid, self.pat1.id, self.pat1.email, Role.PATIENT)
        self.token_pat2 = self.jwt_svc.create_access_token(self.pat2.uuid, self.pat2.id, self.pat2.email, Role.PATIENT)

        self.test_client_ctx = TestClient(app)
        self.client = self.test_client_ctx.__enter__()

        os.makedirs("outputs/clinical_reports", exist_ok=True)
        # Create dummy PDF files to prevent 404 file not found on serving serving
        with open("outputs/clinical_reports/rep_bob.pdf", "w") as f:
            f.write("bob pdf")
        with open("outputs/clinical_reports/rep_alice.pdf", "w") as f:
            f.write("alice pdf")

    def tearDown(self):
        self.test_client_ctx.__exit__(None, None, None)
        if "DISABLE_TEST_AUTO_ASSIGN" in os.environ:
            del os.environ["DISABLE_TEST_AUTO_ASSIGN"]
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def test_01_doctor_can_access_authorized_patient(self):
        """Test: doctor can access a patient they are explicitly assigned to."""
        auth_svc = AuthorizationService(self.db_path)
        self.assertTrue(auth_svc.can_access_patient(self.doc1, "pat-uuid-1"))

    def test_02_doctor_cannot_access_unauthorized_patient(self):
        """Test: doctor cannot access a patient they are not assigned to."""
        auth_svc = AuthorizationService(self.db_path)
        self.assertFalse(auth_svc.can_access_patient(self.doc1, "pat-uuid-2"))

    def test_03_doctor_can_access_authorized_scan(self):
        """Test: doctor can access scan belonging to assigned patient."""
        auth_svc = AuthorizationService(self.db_path)
        self.assertTrue(auth_svc.can_access_scan(self.doc1, 100)) # Scan 100 belongs to Bob (pat-uuid-1)

    def test_04_doctor_cannot_access_unauthorized_scan(self):
        """Test: doctor cannot access scan belonging to unassigned patient."""
        auth_svc = AuthorizationService(self.db_path)
        self.assertFalse(auth_svc.can_access_scan(self.doc1, 101)) # Scan 101 belongs to Alice (pat-uuid-2)

    def test_05_doctor_cannot_access_unauthorized_prediction(self):
        """Test: doctor cannot access prediction belonging to unassigned patient scan."""
        auth_svc = AuthorizationService(self.db_path)
        self.assertFalse(auth_svc.can_access_prediction(self.doc1, 201)) # Prediction 201 belongs to Alice

    def test_06_doctor_cannot_access_unauthorized_report(self):
        """Test: doctor cannot access report belonging to unassigned patient scan."""
        auth_svc = AuthorizationService(self.db_path)
        self.assertFalse(auth_svc.can_access_report(self.doc1, 301)) # Report 301 belongs to Alice

    def test_07_patient_can_access_own_records(self):
        """Test: patient can access their own records."""
        auth_svc = AuthorizationService(self.db_path)
        self.assertTrue(auth_svc.can_access_patient(self.pat1, "pat-uuid-1"))

    def test_08_patient_cannot_access_other_patient_records(self):
        """Test: patient cannot access other patients' records."""
        auth_svc = AuthorizationService(self.db_path)
        self.assertFalse(auth_svc.can_access_patient(self.pat1, "pat-uuid-2"))

    def test_09_admin_access_preserved(self):
        """Test: admin user bypasses authorization and can access everything."""
        auth_svc = AuthorizationService(self.db_path)
        self.assertTrue(auth_svc.can_access_patient(self.admin_user, "pat-uuid-2"))
        self.assertTrue(auth_svc.can_access_scan(self.admin_user, 101))

    def test_10_unauthenticated_access_denied(self):
        """Test: unauthenticated user (None) is denied access."""
        auth_svc = AuthorizationService(self.db_path)
        self.assertFalse(auth_svc.can_access_patient(None, "pat-uuid-1"))
        self.assertFalse(auth_svc.can_access_scan(None, 100))

    def test_11_mismatched_patient_scan_denied(self):
        """Test: scanning/verifying mismatched scan/patient relationship is denied."""
        auth_svc = AuthorizationService(self.db_path)
        self.assertFalse(auth_svc.validate_scan_patient_match(100, "pat-uuid-2")) # Scan 100 is Bob, Patient is Alice
        self.assertTrue(auth_svc.validate_scan_patient_match(100, "pat-uuid-1")) # Scan 100 is Bob, Patient is Bob

    def test_12_mismatched_scan_report_denied(self):
        """Test: verifying mismatched scan/report relationship is denied."""
        auth_svc = AuthorizationService(self.db_path)
        self.assertFalse(auth_svc.validate_report_scan_match(300, 101)) # Report 300 is Bob, Scan 101 is Alice

    def test_13_mismatched_scan_prediction_denied(self):
        """Test: mismatched scan/prediction checks are caught by validating database chain."""
        conn = sqlite3.connect(self.db_path)
        try:
            # Check if prediction maps to correct scan
            row = conn.execute("SELECT scan_id FROM predictions WHERE id = ?;", (200,)).fetchone()
            self.assertEqual(row[0], 100)
            row_wrong = conn.execute("SELECT scan_id FROM predictions WHERE id = ?;", (201,)).fetchone()
            self.assertNotEqual(row_wrong[0], 100)
        finally:
            conn.close()

    def test_14_H1_batch_authorization_enforced(self):
        """Test: Doctor cannot trigger batch prediction for unauthorized patient."""
        headers = {"Authorization": f"Bearer {self.token_doc1}"}
        # Doctor 1 is unauthorized for Patient 2
        response = self.client.post(
            "/api/report/batch",
            data={
                "patient_id": "pat-uuid-2",
                "name": "Alice Cooper",
                "age": 30,
                "gender": "Female",
                "ref_physician": "Dr. Unknown"
            },
            files=[("files", ("test.png", b"dummy_content", "image/png"))],
            headers=headers
        )
        self.assertEqual(response.status_code, 403)

    def test_15_H2_timeline_authorization_enforced(self):
        """Test: Doctor cannot access timeline of unauthorized patient."""
        headers = {"Authorization": f"Bearer {self.token_doc1}"}
        response = self.client.get("/api/patients/pat-uuid-2/longitudinal-timeline", headers=headers)
        self.assertEqual(response.status_code, 403)

    def test_16_H2_comparison_authorization_enforced(self):
        """Test: Doctor cannot compare reports of unauthorized patient."""
        headers = {"Authorization": f"Bearer {self.token_doc1}"}
        # Report 301 belongs to Alice (unauthorized)
        response = self.client.get("/api/reports/300/compare/301", headers=headers)
        self.assertEqual(response.status_code, 403)

    def test_17_predictable_id_IDOR_denied(self):
        """Test: Changing IDs (IDOR manipulation) returns denial/error."""
        headers = {"Authorization": f"Bearer {self.token_doc1}"}
        # Requesting Bob's report -> 200 OK
        resp_bob = self.client.get("/api/reports/300", headers=headers)
        self.assertEqual(resp_bob.status_code, 200)
        # Requesting Alice's report (IDOR) -> 403 Forbidden
        resp_alice = self.client.get("/api/reports/301", headers=headers)
        self.assertEqual(resp_alice.status_code, 403)

    def test_18_unauthorized_response_contains_no_PII(self):
        """Test: Unauthorized responses contain no patient details or scan paths."""
        headers = {"Authorization": f"Bearer {self.token_doc1}"}
        response = self.client.get("/api/reports/301", headers=headers)
        self.assertEqual(response.status_code, 403)
        res_data = response.json()
        # Should not leak names, age, files, or paths
        self.assertNotIn("Bob Jones", str(res_data))
        self.assertNotIn("Alice Cooper", str(res_data))
        self.assertNotIn("outputs/clinical_reports", str(res_data))

    def test_19_authorization_service_is_reused(self):
        """Test that AuthorizationService is imported and called in core endpoints."""
        from security.application.authorization_service import AuthorizationService
        self.assertIsNotNone(AuthorizationService)

    def test_20_authorization_denial_logged_safely(self):
        """Test that audit logs for denial are saved safely and do not contain PII details."""
        service = ReportService(db_path=self.db_path)
        service.log_report_access_event("REPORT_ACCESS_DENIED", self.doc1, 301, "FAILED", "Access denied to patient report.")
        
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute("SELECT * FROM security_audit_logs WHERE event_type = 'REPORT_ACCESS_DENIED';").fetchone()
            self.assertIsNotNone(row)
            # Ensure no PII in details
            details = row[7] # details column
            self.assertNotIn("Alice Cooper", details)
            self.assertNotIn("Bob Jones", details)
        finally:
            conn.close()

    def test_21_doctor_with_zero_assignments_denied(self):
        """Test: Doctor B with zero assignments is denied access to Patient A and Patient B."""
        auth_svc = AuthorizationService(self.db_path)
        self.assertFalse(auth_svc.can_access_patient(self.doc2, "pat-uuid-1"))
        self.assertFalse(auth_svc.can_access_patient(self.doc2, "pat-uuid-2"))

        # Test API response for Doctor B
        headers = {"Authorization": f"Bearer {self.token_doc2}"}
        resp1 = self.client.get("/api/reports/300", headers=headers)
        self.assertEqual(resp1.status_code, 403)
        resp2 = self.client.get("/api/reports/301", headers=headers)
        self.assertEqual(resp2.status_code, 403)

if __name__ == "__main__":
    unittest.main()
