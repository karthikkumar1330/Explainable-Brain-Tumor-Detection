import os
import unittest
import sqlite3
import datetime
import json
import secrets
from fastapi.testclient import TestClient

os.environ["DB_PATH"] = os.path.abspath("outputs/test_report_access.db")

from run_api import app
from clinical_reporting.application.services import (
    ReportService, ReportNotFoundException, VersionNotFoundException,
    PathTraversalException, IntegrityFailureException
)
from clinical_reporting.domain.entities import ReportStatus, generate_integrity_hash
from persistence.infrastructure.repository import SQLitePersistenceRepository
from security.domain.entities import Role, User
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.jwt_service import JWTService


class TestReportAccessSecurity(unittest.TestCase):
    """Phase F2.3 Automated unit, integration, and security compliance tests."""

    def _create_test_hierarchy(self, conn, patient_id="pat-uuid-999", prediction_id=1, scan_id=1):
        conn.execute(
            "INSERT OR IGNORE INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
            (patient_id, "Bob Jones", 45, "Male", datetime.datetime.utcnow().isoformat())
        )
        conn.execute(
            "INSERT OR IGNORE INTO mri_scans (id, patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
            (scan_id, patient_id, "dummy_scan.png", 1.0, "Dr. Smith", "2026-08-08", datetime.datetime.utcnow().isoformat())
        )
        conn.execute(
            """INSERT OR IGNORE INTO predictions (
                id, scan_id, predicted_class, confidence_score, prob_glioma, prob_meningioma,
                prob_pituitary, prob_no_tumor, tumor_pixel_count, tumor_area_mm2,
                tumor_percentage_brain, tumor_percentage_image, estimated_brain_pixel_count,
                rule_based_severity, severity_rule_description, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
            (prediction_id, scan_id, "Glioma", 0.95, 0.95, 0.02, 0.02, 0.01, 500, 50.0, 5.0, 2.0, 10000, "High", "High severity rule", datetime.datetime.utcnow().isoformat())
        )

    def setUp(self):
        self.db_path = os.environ.get("DB_PATH", "outputs/test_report_access.db")
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

        import api.infrastructure.routes as api_routes
        import api.routes.auth_routes as auth_routes
        api_routes.DEFAULT_DB_PATH = self.db_path
        auth_routes.DEFAULT_DB_PATH = self.db_path

        # Initialize DB
        self.persistence_repo = SQLitePersistenceRepository(db_path=self.db_path)
        self.persistence_repo.initialize_db()

        # Set up fastapi test client
        self.test_client_ctx = TestClient(app)
        self.client = self.test_client_ctx.__enter__()

        # Bootstrap user repo & doctor / admin / patient users
        self.user_repo = SQLiteUserRepository(db_path=self.db_path)
        self.user_repo.initialize_security_tables()
        self.admin_user = self.user_repo.bootstrap_admin()

        from security.infrastructure.password import PasswordHasher
        pass_hash = PasswordHasher.hash_password("Doctor@123")

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO users (uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                ("doc-uuid-1", "doctor@aurascan.ai", pass_hash, "Dr. Jane Smith", Role.DOCTOR.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO users (uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                ("pat-uuid-999", "patient@aurascan.ai", pass_hash, "Bob Jones", Role.PATIENT.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            # Create another patient for boundary check
            conn.execute(
                "INSERT INTO users (uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                ("pat-uuid-888", "unauthorized_patient@aurascan.ai", pass_hash, "Alice Cooper", Role.PATIENT.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            self._create_test_hierarchy(conn)
            conn.commit()
        finally:
            conn.close()

        self.doctor_user = self.user_repo.get_by_email("doctor@aurascan.ai")
        self.patient_user = self.user_repo.get_by_email("patient@aurascan.ai")
        self.wrong_patient_user = self.user_repo.get_by_email("unauthorized_patient@aurascan.ai")

        self.jwt_svc = JWTService()
        self.admin_token = self.jwt_svc.create_access_token(
            user_uuid=self.admin_user.uuid, user_id=self.admin_user.id, email=self.admin_user.email, role=Role.ADMIN
        )
        self.doctor_token = self.jwt_svc.create_access_token(
            user_uuid=self.doctor_user.uuid, user_id=self.doctor_user.id, email=self.doctor_user.email, role=Role.DOCTOR
        )
        self.patient_token = self.jwt_svc.create_access_token(
            user_uuid=self.patient_user.uuid, user_id=self.patient_user.id, email=self.patient_user.email, role=Role.PATIENT
        )
        self.wrong_patient_token = self.jwt_svc.create_access_token(
            user_uuid=self.wrong_patient_user.uuid, user_id=self.wrong_patient_user.id, email=self.wrong_patient_user.email, role=Role.PATIENT
        )

        # Create dummy directories
        os.makedirs("outputs/clinical_reports", exist_ok=True)

    def tearDown(self):
        self.test_client_ctx.__exit__(None, None, None)
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def test_check_report_access(self):
        """Tests check_report_access service layer logic."""
        service = ReportService(db_path=self.db_path)
        # Create a report in DB
        report_id = 123
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                (report_id, "REP-2026-0001", "pat-uuid-999", 1, "FINALIZED", "2026-08-08", "2026-08-08")
            )
            conn.commit()
        finally:
            conn.close()

        # Doctor access -> AUTHORIZED
        status = service.check_report_access(report_id, self.doctor_user)
        self.assertEqual(status, "AUTHORIZED")

        # Admin access -> AUTHORIZED
        status = service.check_report_access(report_id, self.admin_user)
        self.assertEqual(status, "AUTHORIZED")

        # Bob Jones (correct patient) -> AUTHORIZED
        status = service.check_report_access(report_id, self.patient_user)
        self.assertEqual(status, "AUTHORIZED")

        # Alice Cooper (wrong patient) -> FORBIDDEN
        status = service.check_report_access(report_id, self.wrong_patient_user)
        self.assertEqual(status, "FORBIDDEN")

        # Nonexistent report -> NOT_FOUND
        status = service.check_report_access(99999, self.doctor_user)
        self.assertEqual(status, "NOT_FOUND")

    def test_path_traversal_detection(self):
        """Tests resolve_secure_pdf_path directory boundary check."""
        service = ReportService(db_path=self.db_path)
        report_id = 124
        # Add report with illegal traversing path
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                (report_id, "REP-2026-0002", "pat-uuid-999", 1, "FINALIZED", "2026-08-08", "2026-08-08")
            )
            conn.execute(
                """INSERT INTO report_versions (
                    report_id, version_number, pdf_path, json_path, checksum, status, integrity_hash, verification_token, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);""",
                (report_id, 1, "outputs/clinical_reports/../../windows/system32/cmd.exe", "outputs/clinical_reports/some.json", "dummy_checksum", "FINALIZED", "somehash", "token123", "2026-08-08")
            )
            conn.commit()
        finally:
            conn.close()

        with self.assertRaises(PathTraversalException):
            service.resolve_secure_pdf_path(report_id, 1)

    def test_integrity_hash_verification(self):
        """Tests resolve_secure_pdf_path integrity validation."""
        service = ReportService(db_path=self.db_path)
        report_id = 125

        payload = {"dummy_field": "val"}
        h = generate_integrity_hash(payload)

        json_path = os.path.abspath("outputs/clinical_reports/test_report.json")
        pdf_path = os.path.abspath("outputs/clinical_reports/test_report.pdf")

        # Write dummy files
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        with open(pdf_path, "w", encoding="utf-8") as f:
            f.write("dummy pdf content")

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                (report_id, "REP-2026-0003", "pat-uuid-999", 1, "FINALIZED", "2026-08-08", "2026-08-08")
            )
            conn.execute(
                """INSERT INTO report_versions (
                    report_id, version_number, pdf_path, json_path, checksum, status, integrity_hash, verification_token, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);""",
                (report_id, 1, pdf_path, json_path, "dummy_checksum", "FINALIZED", h, "token125", "2026-08-08")
            )
            conn.commit()
        finally:
            conn.close()

        # Check with correct hash -> Should pass and return path
        resolved = service.resolve_secure_pdf_path(report_id, 1)
        self.assertEqual(resolved, pdf_path)

        # Modifying json content -> Should raise IntegrityFailureException
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"dummy_field": "tampered_val"}, f)

        with self.assertRaises(IntegrityFailureException):
            service.resolve_secure_pdf_path(report_id, 1)

        # Clean up files
        for p in [json_path, pdf_path]:
            if os.path.exists(p):
                os.remove(p)

    def test_audit_logs_rbac_and_filtering(self):
        """Tests that get_report_audit_history filters logs correctly by role."""
        service = ReportService(db_path=self.db_path)
        conn = sqlite3.connect(self.db_path)
        try:
            # Insert corresponding reports to establish patient links
            conn.execute(
                "INSERT INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                (10, "REP-10", self.patient_user.uuid, 1, "FINALIZED", "2026-08-08", "2026-08-08")
            )
            conn.execute(
                "INSERT INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                (20, "REP-20", self.wrong_patient_user.uuid, 1, "FINALIZED", "2026-08-08", "2026-08-08")
            )
            conn.execute(
                "INSERT OR IGNORE INTO doctor_patient_assignments (doctor_id, patient_id, created_at) VALUES (?, ?, '2026-08-08');",
                (self.doctor_user.id, self.patient_user.uuid)
            )
            conn.execute(
                "INSERT OR IGNORE INTO doctor_patient_assignments (doctor_id, patient_id, created_at) VALUES (?, ?, '2026-08-08');",
                (self.doctor_user.id, self.wrong_patient_user.uuid)
            )

            # 1. Access denied log for Bob Jones's report (ID 10)
            service._log_security_audit_event(
                conn, "REPORT_ACCESS_DENIED", self.wrong_patient_user, 10, "FAILED", "Access denied to patient report."
            )
            # 2. Success log for Bob Jones's report (ID 10)
            service._log_security_audit_event(
                conn, "REPORT_VIEWED", self.patient_user, 10, "SUCCESS", "Viewed report details."
            )
            # 3. Success log for Alice Cooper's report (ID 20)
            service._log_security_audit_event(
                conn, "REPORT_VIEWED", self.wrong_patient_user, 20, "SUCCESS", "Viewed report details."
            )
            conn.commit()
        finally:
            conn.close()

        # Doctor user -> Should see all logs
        doc_logs = service.get_report_audit_history(self.doctor_user)
        self.assertEqual(len(doc_logs), 3)

        # Patient user Bob Jones -> Should only see logs relating to Report ID 10
        bob_logs = service.get_report_audit_history(self.patient_user)
        self.assertEqual(len(bob_logs), 2)
        for log in bob_logs:
            self.assertIn("Report ID: 10", log["details"])

    def test_secure_pdf_download_endpoint(self):
        """Integration test for secure PDF serving and authorization."""
        report_id = 200
        pdf_path = os.path.abspath("outputs/clinical_reports/integ_test.pdf")
        with open(pdf_path, "w", encoding="utf-8") as f:
            f.write("pdf data")

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                (report_id, "REP-2026-0200", "pat-uuid-999", 1, "FINALIZED", "2026-08-08", "2026-08-08")
            )
            conn.execute(
                """INSERT INTO report_versions (
                    report_id, version_number, pdf_path, json_path, checksum, status, integrity_hash, verification_token, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);""",
                (report_id, 1, pdf_path, "outputs/clinical_reports/dummy.json", "dummy_checksum", "FINALIZED", None, "token200", "2026-08-08")
            )
            conn.commit()
        finally:
            conn.close()

        # 1. Unauthorized request (no token) -> 401
        res = self.client.get(f"/api/report/{report_id}/pdf")
        self.assertEqual(res.status_code, 401)

        # 2. Correct patient download request -> 200 with security headers
        headers = {"Authorization": f"Bearer {self.patient_token}"}
        res = self.client.get(f"/api/report/{report_id}/pdf", headers=headers)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get("X-Content-Type-Options"), "nosniff")
        self.assertEqual(res.headers.get("Content-Disposition"), "attachment; filename=integ_test.pdf")

        # 3. Wrong patient request -> 403 Forbidden
        headers = {"Authorization": f"Bearer {self.wrong_patient_token}"}
        res = self.client.get(f"/api/report/{report_id}/pdf", headers=headers)
        self.assertEqual(res.status_code, 403)

        # Clean up
        if os.path.exists(pdf_path):
            os.remove(pdf_path)


if __name__ == "__main__":
    unittest.main()
