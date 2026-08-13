import os
import sqlite3
import datetime
import unittest
from unittest.mock import patch
from flask import json
from dashboard.infrastructure.web_server import create_app
from security.domain.entities import Role, User
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.jwt_service import JWTService
from persistence.infrastructure.repository import SQLitePersistenceRepository

class TestH3ExceptionDisclosureRegression(unittest.TestCase):
    """Focused regression tests proving exception disclosure prevention in Phase H3 Flask routes."""

    def setUp(self):
        self.db_path = os.path.abspath("outputs/test_h3_exception_regression.db")
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

        # Configure Flask App
        self.app = create_app(db_path=self.db_path)
        self.app.config["TESTING"] = True
        self.app.config["DISABLE_CSRF"] = True
        self.client = self.app.test_client()

        # Initialize DB
        self.persistence_repo = SQLitePersistenceRepository(db_path=self.db_path)
        self.persistence_repo.initialize_db()

        self.user_repo = SQLiteUserRepository(db_path=self.db_path)
        self.user_repo.initialize_security_tables()

        # Drop auto triggers if they exist to keep it simple and clean
        conn_drop = sqlite3.connect(self.db_path)
        try:
            conn_drop.execute("DROP TRIGGER IF EXISTS auto_assign_doctor_on_insert;")
            conn_drop.execute("DROP TRIGGER IF EXISTS auto_assign_patient_on_insert;")
            conn_drop.execute("DELETE FROM doctor_patient_assignments;")
            conn_drop.commit()
        finally:
            conn_drop.close()

        # Bootstrap admin
        self.admin_user = self.user_repo.bootstrap_admin()

        from security.infrastructure.password import PasswordHasher
        pass_hash = PasswordHasher.hash_password("Password@123")

        conn = sqlite3.connect(self.db_path)
        try:
            # Doctor A (ID: 10), Patient A (ID: 20)
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (10, "doc-uuid-a", "doctor_a@aurascan.ai", pass_hash, "Dr. Alice", Role.DOCTOR.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (20, "pat-uuid-a", "patient_a@aurascan.ai", pass_hash, "Patient A", Role.PATIENT.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )

            # Insert patient demographics
            conn.execute(
                "INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
                ("pat-uuid-a", "Patient A", 45, "Male", datetime.datetime.utcnow().isoformat())
            )

            # Insert MRI Scan: Scan A (100) -> Patient A
            conn.execute(
                "INSERT INTO mri_scans (id, patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                (100, "pat-uuid-a", "uploads/scan_a.png", 1.0, "Dr. Alice", "2026-08-08", datetime.datetime.utcnow().isoformat())
            )

            # Assignments: Doctor A -> Patient A
            conn.execute(
                "INSERT INTO doctor_patient_assignments (doctor_id, patient_id, created_at) VALUES (?, ?, ?);",
                (10, "pat-uuid-a", datetime.datetime.utcnow().isoformat())
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

            # Insert Reports Table
            conn.execute(
                "INSERT INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                (300, "REP-2026-0001", "pat-uuid-a", 1, "FINAL", "2026-08-08", "2026-08-08")
            )
            # Insert clinical_reports Table
            conn.execute(
                "INSERT INTO clinical_reports (id, prediction_id, markdown_path, json_path, pdf_path, created_at) VALUES (?, ?, ?, ?, ?, ?);",
                (300, 200, "outputs/clinical_reports/rep_bob.md", "outputs/clinical_reports/rep_bob.json", "outputs/clinical_reports/rep_bob.pdf", datetime.datetime.utcnow().isoformat())
            )

            conn.commit()
        finally:
            conn.close()

        # Load users
        self.doc_a = self.user_repo.get_by_email("doctor_a@aurascan.ai")
        self.pat_a = self.user_repo.get_by_email("patient_a@aurascan.ai")

        # JWT tokens
        self.jwt_svc = JWTService()
        self.token_doc_a = self.jwt_svc.create_access_token(self.doc_a.uuid, self.doc_a.id, self.doc_a.email, Role.DOCTOR)
        self.token_pat_a = self.jwt_svc.create_access_token(self.pat_a.uuid, self.pat_a.id, self.pat_a.email, Role.PATIENT)

    def tearDown(self):
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

    @patch("clinical_reporting.application.services.ReportService.get_patient_profile")
    def test_01_unexpected_exceptions_do_not_leak_details(self, mock_get_profile):
        """1. Unexpected exceptions do not leak raw exception text, paths, or DB details."""
        # Setup mock to raise a sensitive unexpected exception
        sensitive_msg = "FATAL SQLITE_CORRUPT: Database disk image is malformed at d:/Workspace/Project/outputs/clinical_reports.db"
        mock_get_profile.side_effect = RuntimeError(sensitive_msg)

        headers = self._get_headers(self.token_doc_a)
        resp = self.client.get("/api/doctor/patients/pat-uuid-a", headers=headers)

        # Verify 500 status code
        self.assertEqual(resp.status_code, 500)

        # Verify safe client error message
        data = resp.get_json()
        self.assertEqual(data["error"], "Internal patient profile fetch error")

        # Verify absolutely no leakage of the sensitive text
        self.assertNotIn("SQLITE_CORRUPT", str(data))
        self.assertNotIn("Workspace", str(data))
        self.assertNotIn("db", str(data))
        self.assertNotIn("RuntimeError", str(data))

    def test_02_authorization_remains_unchanged(self):
        """2. Authorization checks still enforce boundaries correctly."""
        # Unauthenticated request (no token) -> 401
        resp = self.client.get("/api/doctor/patients/pat-uuid-a")
        self.assertEqual(resp.status_code, 401)

        # Patient token accessing doctor endpoint -> 403
        headers = self._get_headers(self.token_pat_a)
        resp2 = self.client.get("/api/doctor/patients/pat-uuid-a", headers=headers)
        self.assertEqual(resp2.status_code, 403)

    def test_03_404_remains_unchanged(self):
        """3. Requests for nonexistent entities still return correct 404 code and standard messages."""
        headers = self._get_headers(self.token_doc_a)

        # Nonexistent patient -> 403 Forbidden (preventing enumeration of existence for unassigned doctor)
        resp = self.client.get("/api/doctor/patients/nonexistent-uuid", headers=headers)
        self.assertEqual(resp.status_code, 403)
        self.assertIn("access denied", resp.get_json()["error"].lower())

        # Nonexistent report details -> 404
        resp2 = self.client.get("/api/report/9999", headers=headers)
        self.assertEqual(resp2.status_code, 404)
        self.assertEqual(resp2.get_json()["error"], "Report not found")

    def test_04_successful_requests_remain_unchanged(self):
        """4. Successful requests return correct 200 code and payload."""
        headers = self._get_headers(self.token_doc_a)

        # Fetch patient profile details -> 200 OK
        resp = self.client.get("/api/doctor/patients/pat-uuid-a", headers=headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["patient_id"], "pat-uuid-a")
        self.assertEqual(data["name"], "Patient A")

    @patch("clinical_reporting.application.mri_annotation_service.MriAnnotationService.create_rectangle_annotation")
    def test_05_unexpected_annotation_exceptions_do_not_leak_details(self, mock_create_rect):
        """5. Unexpected exceptions on annotations return safe error message."""
        mock_create_rect.side_effect = Exception("System error: /var/lib/sqlite/disk_space_low")

        headers = self._get_headers(self.token_doc_a)
        payload = {
            "x": 0.1,
            "y": 0.2,
            "width": 0.3,
            "height": 0.4,
            "label": "Test",
            "comment": "Nice annotation"
        }
        resp = self.client.post("/api/doctor/scans/100/rectangle-annotations", json=payload, headers=headers)

        # Verify 500 and safe message
        self.assertEqual(resp.status_code, 500)
        self.assertEqual(resp.get_json()["error"], "Internal rectangle annotation operation failure")
        self.assertNotIn("disk_space_low", str(resp.get_json()))
        self.assertNotIn("sqlite", str(resp.get_json()))

    @patch("security.application.use_cases.AuthUseCases.refresh_token")
    def test_06_refresh_token_invalid_response(self, mock_refresh):
        """6. Invalid refresh token ValueError returns generic error and code."""
        mock_refresh.side_effect = ValueError("Invalid refresh token: signature verification failed in /root/jwt.py")

        resp = self.client.post("/api/auth/refresh", json={"refresh_token": "some-token"})
        self.assertEqual(resp.status_code, 401)
        data = resp.get_json()
        self.assertEqual(data["error"], "Invalid or expired refresh token.")
        self.assertEqual(data["code"], "REFRESH_TOKEN_INVALID")
        self.assertNotIn("signature verification", str(data))
        self.assertNotIn("jwt.py", str(data))

    @patch("security.application.use_cases.AuthUseCases.refresh_token")
    def test_07_refresh_token_unexpected_exception(self, mock_refresh):
        """7. Unexpected exception in refresh token returns generic 500 error."""
        mock_refresh.side_effect = RuntimeError("sqlite3.OperationalError: no such table: user_tokens at /usr/lib/sqlite")

        resp = self.client.post("/api/auth/refresh", json={"refresh_token": "some-token"})
        self.assertEqual(resp.status_code, 500)
        data = resp.get_json()
        self.assertEqual(data["error"], "An unexpected error occurred.")
        self.assertEqual(data["code"], "INTERNAL_SERVER_ERROR")
        self.assertNotIn("sqlite3", str(data))
        self.assertNotIn("user_tokens", str(data))

    @patch("security.infrastructure.repository.SQLiteUserRepository.delete_user")
    def test_08_profile_deletion_unexpected_exception(self, mock_delete):
        """8. Unexpected exception in profile deletion returns generic 500 error without leak."""
        mock_delete.side_effect = Exception("Failed filesystem sync on /home/user/app/data.db")

        headers = self._get_headers(self.token_doc_a)
        resp = self.client.delete("/api/auth/profile", headers=headers)
        self.assertEqual(resp.status_code, 500)
        data = resp.get_json()
        self.assertEqual(data["error"], "Internal profile deletion error")
        self.assertNotIn("filesystem sync", str(data))
        self.assertNotIn("data.db", str(data))

if __name__ == "__main__":
    unittest.main()
