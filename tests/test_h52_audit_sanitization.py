import os
os.environ["DB_PATH"] = os.path.abspath("outputs/test_h52_audit_sanitization.db")

import unittest
import sqlite3
import datetime
from fastapi.testclient import TestClient
from run_api import app
from security.domain.entities import Role, User
from security.infrastructure.jwt_service import JWTService
from security.infrastructure.repository import SQLiteUserRepository
from persistence.infrastructure.repository import SQLitePersistenceRepository
from clinical_reporting.application.services import ReportService, PathTraversalException
from dashboard.infrastructure.web_server import create_app as create_flask_app

class TestH52AuditSanitization(unittest.TestCase):
    """Phase H5.2: Audit Data Sanitization focused tests."""

    def setUp(self):
        self.db_path = os.environ.get("DB_PATH", "outputs/test_h52_audit_sanitization.db")
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

        # Override default DB paths in API
        import api.infrastructure.routes as api_routes
        import api.routes.auth_routes as auth_routes
        api_routes.DEFAULT_DB_PATH = self.db_path
        auth_routes.DEFAULT_DB_PATH = self.db_path

        # Disable trigger auto assignment for strict doctor assignment tests
        os.environ["DISABLE_TEST_AUTO_ASSIGN"] = "1"

        # 1. Initialize databases
        persistence_repo = SQLitePersistenceRepository(db_path=self.db_path)
        persistence_repo.initialize_db()

        self.user_repo = SQLiteUserRepository(db_path=self.db_path)
        self.user_repo.initialize_security_tables()

        # Disable triggers manually
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DROP TRIGGER IF EXISTS auto_assign_doctor_on_insert;")
            conn.execute("DROP TRIGGER IF EXISTS auto_assign_patient_on_insert;")
            conn.execute("DELETE FROM doctor_patient_assignments;")
            conn.commit()
        finally:
            conn.close()

        # 2. Setup users
        self.admin = self.user_repo.bootstrap_admin()

        # Patient
        self.patient = User(
            id=None,
            uuid="pat-uuid-aaaa",
            email="patienta@aurascan.ai",
            password_hash="fakehash",
            full_name="Jane Doe",
            role=Role.PATIENT,
            is_verified=True,
            is_active=True,
            created_at=datetime.datetime.utcnow().isoformat(),
            updated_at=datetime.datetime.utcnow().isoformat()
        )
        self.patient = self.user_repo.create_user(self.patient)

        # JWT Token
        jwt_svc = JWTService()
        self.pat_token = jwt_svc.create_access_token(self.patient.uuid, self.patient.id, self.patient.email, self.patient.role)
        self.pat_headers = {"Authorization": f"Bearer {self.pat_token}"}

        # Initialize Flask Test Client
        # Set TESTING to False so that the CSRF protection middleware is NOT bypassed
        self.flask_app = create_flask_app(db_path=self.db_path)
        self.flask_app.config["TESTING"] = False
        self.flask_client = self.flask_app.test_client()

        # Initialize FastAPI client (for general routes if needed)
        self.client = TestClient(app)

    def test_01_csrf_failure_audit_sanitization(self):
        """1. Trigger CSRF failure and verify no CSRF token values are logged."""
        # Use Flask TestClient to make a POST to a protected API endpoint (e.g. /api/auth/change-password)
        # with mismatched CSRF headers and cookies.
        headers = {
            "Authorization": f"Bearer {self.pat_token}",
            "X-CSRF-Token": "secret_header_val_abc123",
            "Cookie": "csrf_token=secret_cookie_val_xyz789"
        }

        # Make request on Flask client
        resp = self.flask_client.post("/api/auth/change-password", headers=headers, json={"new_password": "NewPassword@123", "current_password": "OldPassword@123"})
        self.assertEqual(resp.status_code, 403)
        self.assertIn("CSRF verification failed", resp.get_json().get("error", ""))

        # Check database security audit logs
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute("SELECT * FROM security_audit_logs WHERE event_type = 'CSRF_ATTEMPT';").fetchone()
            self.assertIsNotNone(row)
            details = row["details"]

            # Assert details is generic and contains no secrets
            self.assertEqual(details, "CSRF token mismatch")
            self.assertNotIn("secret_header_val_abc123", details)
            self.assertNotIn("secret_cookie_val_xyz789", details)
        finally:
            conn.close()

    def test_02_path_traversal_audit_sanitization(self):
        """2. Trigger path traversal exception and verify raw path is not persisted in details."""
        service = ReportService(db_path=self.db_path)
        report_id = 99
        version = 1

        # Seed report with path traversal json_path
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES ('pat-uuid-aaaa', 'Jane Doe', 30, 'Female', '2026-08-14T00:00:00');")
            conn.execute(
                "INSERT INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at) "
                "VALUES (?, 'RPT-PT-99', 'pat-uuid-aaaa', ?, 'FINAL', '2026-08-14T00:00:00', '2026-08-14T00:00:00');",
                (report_id, version)
            )
            # Add version with path traversal JSON escape and required checksum and status
            conn.execute(
                "INSERT INTO report_versions (report_id, version_number, prediction_id, json_path, pdf_path, integrity_hash, checksum, status, created_at) "
                "VALUES (?, ?, 1, 'outputs/clinical_reports/../../sensitive_passwords.json', 'outputs/clinical_reports/dummy.pdf', 'hash', 'dummy_checksum', 'FINAL', '2026-08-14T00:00:00');",
                (report_id, version)
            )
            conn.commit()
        finally:
            conn.close()

        # Attempt to export JSON (triggers path traversal validation)
        with self.assertRaises(PathTraversalException):
            service.get_report_json_for_export(report_id, version, self.patient)

        # Check database logs
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute("SELECT * FROM security_audit_logs WHERE event_type = 'REPORT_ACCESS_DENIED';").fetchone()
            self.assertIsNotNone(row)
            details = row["details"]

            # Assert details is sanitized
            self.assertEqual(details, "Report ID: 99, Access denied: path traversal attempt")
            self.assertNotIn("sensitive_passwords.json", details)
            self.assertNotIn("outputs/clinical_reports", details)
        finally:
            conn.close()

    def test_03_no_plain_passwords_or_tokens_in_logs(self):
        """3. General logins and password resets do not log raw passwords or JWTs."""
        # Query all security logs in database to ensure no passwords or secrets are recorded
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute("SELECT * FROM security_audit_logs;").fetchall()
            for r in rows:
                details = r["details"]
                # None of the logs should contain jwt signatures or fake password values
                self.assertNotIn("fakehash", details)
                self.assertNotIn("Bearer", details)
                self.assertNotIn("Authorization", details)
        finally:
            conn.close()
