import os
os.environ["DB_PATH"] = os.path.abspath("outputs/test_email_delivery.db")

import unittest
import sqlite3
import datetime
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from run_api import app
from security.domain.entities import Role, User
from security.infrastructure.jwt_service import JWTService
from security.infrastructure.repository import SQLiteUserRepository
from persistence.infrastructure.repository import SQLitePersistenceRepository
from clinical_reporting.application.services import ReportService, ReportNotFoundException


class TestEmailDelivery(unittest.TestCase):
    """Phase G4 Clinical Report Email Delivery and authorization tests."""

    def setUp(self):
        self.db_path = os.environ.get("DB_PATH", "outputs/test_email_delivery.db")
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

        # Override route default DB paths
        import api.infrastructure.routes as api_routes
        import api.routes.auth_routes as auth_routes
        api_routes.DEFAULT_DB_PATH = self.db_path
        auth_routes.DEFAULT_DB_PATH = self.db_path

        # 1. Initialize databases
        persistence_repo = SQLitePersistenceRepository(db_path=self.db_path)
        persistence_repo.initialize_db()

        self.user_repo = SQLiteUserRepository(db_path=self.db_path)
        self.user_repo.initialize_security_tables()

        # 2. Bootstrap default users
        self.admin = self.user_repo.bootstrap_admin()
        
        # Create extra doctor user
        self.doctor = User(
            id=None,
            uuid="doc-uuid-1111",
            email="doctor@aurascan.ai",
            password_hash="fakehash",
            full_name="Dr. House",
            role=Role.DOCTOR,
            is_verified=True,
            is_active=True,
            created_at=datetime.datetime.utcnow().isoformat(),
            updated_at=datetime.datetime.utcnow().isoformat()
        )
        self.doctor = self.user_repo.create_user(self.doctor)

        # Create patient 1 user
        self.patient_user1 = User(
            id=None,
            uuid="pat-uuid-2222",
            email="patient1@aurascan.ai",
            password_hash="fakehash",
            full_name="Alice Patient",
            role=Role.PATIENT,
            is_verified=True,
            is_active=True,
            created_at=datetime.datetime.utcnow().isoformat(),
            updated_at=datetime.datetime.utcnow().isoformat()
        )
        self.patient_user1 = self.user_repo.create_user(self.patient_user1)

        # Create patient 2 user
        self.patient_user2 = User(
            id=None,
            uuid="pat-uuid-3333",
            email="patient2@aurascan.ai",
            password_hash="fakehash",
            full_name="Bob Patient",
            role=Role.PATIENT,
            is_verified=True,
            is_active=True,
            created_at=datetime.datetime.utcnow().isoformat(),
            updated_at=datetime.datetime.utcnow().isoformat()
        )
        self.patient_user2 = self.user_repo.create_user(self.patient_user2)

        # 3. Create tokens
        jwt_svc = JWTService()
        self.doc_token = jwt_svc.create_access_token(
            user_uuid=self.doctor.uuid,
            user_id=self.doctor.id,
            email=self.doctor.email,
            role=self.doctor.role
        )
        self.pat1_token = jwt_svc.create_access_token(
            user_uuid=self.patient_user1.uuid,
            user_id=self.patient_user1.id,
            email=self.patient_user1.email,
            role=self.patient_user1.role
        )
        self.pat2_token = jwt_svc.create_access_token(
            user_uuid=self.patient_user2.uuid,
            user_id=self.patient_user2.id,
            email=self.patient_user2.email,
            role=self.patient_user2.role
        )

        self.doc_headers = {"Authorization": f"Bearer {self.doc_token}"}
        self.pat1_headers = {"Authorization": f"Bearer {self.pat1_token}"}
        self.pat2_headers = {"Authorization": f"Bearer {self.pat2_token}"}

        # 4. Bootstrap report data directly in database
        self.pdf_dir = "outputs/clinical_reports"
        os.makedirs(self.pdf_dir, exist_ok=True)
        self.pdf_file_path = os.path.join(self.pdf_dir, "clinical_report_1.pdf")
        
        # Write mock PDF content
        with open(self.pdf_file_path, "wb") as f:
            f.write(b"%PDF-1.4 Mock PDF Content")

        conn = sqlite3.connect(self.db_path)
        try:
            # Insert patient
            conn.execute("""
                INSERT INTO patients (patient_id, name, age, gender, created_at)
                VALUES (?, ?, 25, 'Female', '2026-08-09T00:00:00');
            """, ("pat-uuid-2222", "Alice Patient"))

            conn.execute("""
                INSERT INTO patients (patient_id, name, age, gender, created_at)
                VALUES (?, ?, 30, 'Male', '2026-08-09T00:00:00');
            """, ("pat-uuid-3333", "Bob Patient"))

            # Insert report 1 (belongs to Patient 1)
            conn.execute("""
                INSERT INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at)
                VALUES (1, 'RPT-2026-000001', 'pat-uuid-2222', 1, 'FINAL', '2026-08-09T00:00:00', '2026-08-09T00:00:00');
            """)
            conn.execute("""
                INSERT INTO report_versions (version_id, report_id, version_number, created_at, pdf_path, json_path, checksum, status)
                VALUES (1, 1, 1, '2026-08-09T00:00:00', ?, 'outputs/clinical_reports/report_1.json', 'hash123', 'FINAL');
            """, (self.pdf_file_path,))

            # Insert report 2 (belongs to Patient 2)
            conn.execute("""
                INSERT INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at)
                VALUES (2, 'RPT-2026-000002', 'pat-uuid-3333', 1, 'FINAL', '2026-08-09T00:00:00', '2026-08-09T00:00:00');
            """)
            conn.execute("""
                INSERT INTO report_versions (version_id, report_id, version_number, created_at, pdf_path, json_path, checksum, status)
                VALUES (2, 2, 1, '2026-08-09T00:00:00', ?, 'outputs/clinical_reports/report_2.json', 'hash456', 'FINAL');
            """, (self.pdf_file_path,))

            conn.commit()
        finally:
            conn.close()

        # 5. Fastapi TestClient
        self.test_client_ctx = TestClient(app)
        self.client = self.test_client_ctx.__enter__()

    def tearDown(self):
        self.test_client_ctx.__exit__()
        if os.path.exists(self.pdf_file_path):
            try:
                os.remove(self.pdf_file_path)
            except Exception:
                pass
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    @patch("clinical_reporting.infrastructure.email_service.EmailService.send")
    def test_01_authorized_user_can_email_own_report(self, mock_send) -> None:
        """Verify that Patient 1 can successfully email their own report."""
        response = self.client.post(
            "/api/reports/1/email",
            json={},
            headers=self.pat1_headers
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["success"])
        
        # Verify EmailService was called with Patient 1's email
        mock_send.assert_called_once()
        kwargs = mock_send.call_args[1]
        self.assertEqual(kwargs["to_email"], "patient1@aurascan.ai")
        self.assertIn("RPT-2026-000001", kwargs["subject"])
        self.assertEqual(kwargs["attachment_path"], os.path.abspath(self.pdf_file_path))

    def test_02_unauthenticated_user_rejected(self) -> None:
        """Verify that requests without authorization headers are blocked with 401."""
        response = self.client.post("/api/reports/1/email", json={})
        self.assertEqual(response.status_code, 401)

    @patch("clinical_reporting.infrastructure.email_service.EmailService.send")
    def test_03_unauthorized_user_cannot_email_other_report_idor(self, mock_send) -> None:
        """Verify that Patient 1 cannot request report 2 (owned by Patient 2) - IDOR protection."""
        response = self.client.post(
            "/api/reports/2/email",
            json={},
            headers=self.pat1_headers
        )
        self.assertEqual(response.status_code, 403)
        # Verify absolutely no email was sent
        mock_send.assert_not_called()

    @patch("clinical_reporting.infrastructure.email_service.EmailService.send")
    def test_04_invalid_report_id_returns_404(self, mock_send) -> None:
        """Verify nonexistent report IDs return a 404 error."""
        response = self.client.post(
            "/api/reports/999/email",
            json={},
            headers=self.doc_headers
        )
        self.assertEqual(response.status_code, 404)
        mock_send.assert_not_called()

    @patch("clinical_reporting.infrastructure.email_service.EmailService.send")
    def test_05_unauthorized_recipient_rejected(self, mock_send) -> None:
        """Verify that patients cannot email reports to unauthorized third parties."""
        response = self.client.post(
            "/api/reports/1/email",
            json={"recipient_email": "stranger@external.com"},
            headers=self.pat1_headers
        )
        self.assertEqual(response.status_code, 403)
        mock_send.assert_not_called()

    @patch("clinical_reporting.infrastructure.email_service.EmailService.send")
    def test_06_malformed_recipient_rejected(self, mock_send) -> None:
        """Verify that malformed email addresses are rejected with 400."""
        response = self.client.post(
            "/api/reports/1/email",
            json={"recipient_email": "invalid-email-format"},
            headers=self.doc_headers
        )
        self.assertEqual(response.status_code, 400)
        mock_send.assert_not_called()

    @patch("clinical_reporting.infrastructure.email_service.EmailService.send")
    def test_07_recipient_header_injection_rejected(self, mock_send) -> None:
        """Verify that header injection characters in emails are blocked."""
        response = self.client.post(
            "/api/reports/1/email",
            json={"recipient_email": "recipient@aurascan.ai\r\nBcc: spy@external.com"},
            headers=self.doc_headers
        )
        self.assertEqual(response.status_code, 400)
        mock_send.assert_not_called()

    @patch("clinical_reporting.infrastructure.email_service.EmailService.send")
    def test_08_missing_pdf_file_returns_404(self, mock_send) -> None:
        """Verify that if the PDF is deleted/missing from disk, a 404 is returned."""
        # Temporarily delete PDF
        if os.path.exists(self.pdf_file_path):
            os.remove(self.pdf_file_path)
            
        response = self.client.post(
            "/api/reports/1/email",
            json={},
            headers=self.doc_headers
        )
        self.assertEqual(response.status_code, 404)
        mock_send.assert_not_called()

    @patch("clinical_reporting.infrastructure.email_service.EmailService.send")
    def test_09_smtp_transmission_failure_handled_gracefully(self, mock_send) -> None:
        """Verify that connection/SMTP send failures return a server error without details leakage."""
        mock_send.side_effect = Exception("SMTP Connection Timeout")
        
        response = self.client.post(
            "/api/reports/1/email",
            json={},
            headers=self.doc_headers
        )
        self.assertEqual(response.status_code, 500)
        # Ensure filesystem path or internal details are not exposed in payload
        self.assertNotIn("outputs/clinical_reports", response.text)
        self.assertNotIn("SMTP Connection Timeout", response.text)

    @patch("clinical_reporting.infrastructure.email_service.EmailService.send")
    def test_10_safe_pdf_attachment_metadata(self, mock_send) -> None:
        """Verify that the resolved attachment path matches the correct file reference."""
        response = self.client.post(
            "/api/reports/1/email",
            json={},
            headers=self.doc_headers
        )
        self.assertEqual(response.status_code, 200)
        kwargs = mock_send.call_args[1]
        
        # Attachment path must match exactly and filename resolved safely
        self.assertEqual(kwargs["attachment_path"], os.path.abspath(self.pdf_file_path))
        self.assertTrue(kwargs["attachment_path"].endswith("clinical_report_1.pdf"))

    @patch("clinical_reporting.infrastructure.email_service.EmailService.send")
    def test_11_response_does_not_leak_secrets(self, mock_send) -> None:
        """Verify that the response does not expose server keys or paths."""
        response = self.client.post(
            "/api/reports/1/email",
            json={},
            headers=self.doc_headers
        )
        data = response.json()
        self.assertNotIn("password", str(data).lower())
        self.assertNotIn("smtp", str(data).lower())
        self.assertNotIn("outputs/clinical_reports", str(data).lower())

    @patch("clinical_reporting.infrastructure.email_service.EmailService.send")
    def test_12_email_history_record_created_on_success(self, mock_send) -> None:
        """Verify that an email delivery record with status 'SENT' is logged in the database on success."""
        response = self.client.post(
            "/api/reports/1/email",
            json={"recipient_email": "patient1@aurascan.ai"},
            headers=self.doc_headers
        )
        self.assertEqual(response.status_code, 200)

        # Inspect the email_deliveries table in the database
        conn = sqlite3.connect(self.db_path)
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM email_deliveries WHERE report_id = 1;").fetchall()
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["status"], "SENT")
            self.assertEqual(row["recipient_email"], "patient1@aurascan.ai")
            self.assertIsNotNone(row["sent_at"])
            self.assertIsNone(row["failure_reason"])
        finally:
            conn.close()

    @patch.dict(os.environ, {"EMAIL_RETRY_ENABLED": "False"})
    @patch("clinical_reporting.infrastructure.email_service.EmailService.send")
    def test_13_email_history_record_created_on_failure(self, mock_send) -> None:
        """Verify that a failed send attempt logs status 'FAILED' and normalizes failure reason."""
        mock_send.side_effect = Exception("SMTP Connection Refused")

        response = self.client.post(
            "/api/reports/1/email",
            json={"recipient_email": "patient1@aurascan.ai"},
            headers=self.doc_headers
        )
        self.assertEqual(response.status_code, 500)

        conn = sqlite3.connect(self.db_path)
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM email_deliveries WHERE report_id = 1;").fetchall()
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["status"], "FAILED")
            self.assertEqual(row["failure_reason"], "SMTP_CONNECTION_FAILED")
            self.assertIsNone(row["sent_at"])
        finally:
            conn.close()

    @patch("clinical_reporting.infrastructure.email_service.EmailService.send")
    def test_14_email_history_api_authorization_and_masking(self, mock_send) -> None:
        """Verify that the email history API is secured and applies masking to recipient emails."""
        # Insert a successful delivery record manually to ensure history exists
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT INTO email_deliveries (report_id, actor_user_id, recipient_email, status, attempted_at, created_at, updated_at)
                VALUES (1, ?, 'patient1@aurascan.ai', 'SENT', '2026-08-09T12:00:00', '2026-08-09T12:00:00', '2026-08-09T12:00:00');
            """, (self.doctor.id,))
            conn.commit()
        finally:
            conn.close()

        # Doctor user should see the record, but with masked recipient email
        response = self.client.get(
            "/api/reports/email/history",
            headers=self.doc_headers
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("items", data)
        self.assertTrue(len(data["items"]) >= 1)

        item = data["items"][0]
        # Masking should mask the email (e.g. 'patient1@aurascan.ai' to 'pa****@aurascan.ai')
        self.assertEqual(item["recipient_email"], "pa****@aurascan.ai")
        self.assertEqual(item["report_number"], "RPT-2026-000001")

        # Unauthenticated user should be rejected
        response_anon = self.client.get("/api/reports/email/history")
        self.assertEqual(response_anon.status_code, 401)

    @patch("clinical_reporting.infrastructure.email_service.EmailService.send")
    def test_15_email_history_patient_idor_filtering(self, mock_send) -> None:
        """Verify that patients can only retrieve history for reports they own."""
        # Insert two delivery records: one for report 1 (owned by patient 1), one for report 2 (owned by patient 2)
        conn = sqlite3.connect(self.db_path)
        try:
            # Report 1
            conn.execute("""
                INSERT INTO email_deliveries (report_id, actor_user_id, recipient_email, status, attempted_at, created_at, updated_at)
                VALUES (1, ?, 'patient1@aurascan.ai', 'SENT', '2026-08-09T12:00:00', '2026-08-09T12:00:00', '2026-08-09T12:00:00');
            """, (self.patient_user1.id,))
            # Report 2
            conn.execute("""
                INSERT INTO email_deliveries (report_id, actor_user_id, recipient_email, status, attempted_at, created_at, updated_at)
                VALUES (2, ?, 'patient2@aurascan.ai', 'SENT', '2026-08-09T12:05:00', '2026-08-09T12:05:00', '2026-08-09T12:05:00');
            """, (self.patient_user2.id,))
            conn.commit()
        finally:
            conn.close()

        # Alice (Patient 1) queries the email history API
        response = self.client.get(
            "/api/reports/email/history",
            headers=self.pat1_headers
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Verify Alice ONLY sees report 1's history
        for item in data["items"]:
            self.assertEqual(item["report_id"], 1)
            self.assertNotEqual(item["report_id"], 2)

    @patch("clinical_reporting.infrastructure.email_service.EmailService.send")
    def test_16_email_history_search_and_filters(self, mock_send) -> None:
        """Verify searching and status filtering on the history endpoint."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT INTO email_deliveries (report_id, actor_user_id, recipient_email, status, attempted_at, created_at, updated_at)
                VALUES (1, ?, 'doctor@aurascan.ai', 'SENT', '2026-08-09T12:00:00', '2026-08-09T12:00:00', '2026-08-09T12:00:00');
            """, (self.doctor.id,))
            conn.execute("""
                INSERT INTO email_deliveries (report_id, actor_user_id, recipient_email, status, attempted_at, created_at, updated_at, failure_reason)
                VALUES (1, ?, 'recipient@aurascan.ai', 'FAILED', '2026-08-09T12:10:00', '2026-08-09T12:10:00', '2026-08-09T12:10:00', 'SMTP_TIMEOUT');
            """, (self.doctor.id,))
            conn.commit()
        finally:
            conn.close()

        # Filter by status = FAILED
        response_failed = self.client.get(
            "/api/reports/email/history?status=FAILED",
            headers=self.doc_headers
        )
        self.assertEqual(response_failed.status_code, 200)
        items = response_failed.json()["items"]
        self.assertTrue(len(items) >= 1)
        for item in items:
            self.assertEqual(item["status"], "FAILED")

        # Search by recipient_email (searched by raw input, but displayed masked)
        response_search = self.client.get(
            "/api/reports/email/history?search=recipient",
            headers=self.doc_headers
        )
        self.assertEqual(response_search.status_code, 200)
        items_search = response_search.json()["items"]
        self.assertEqual(len(items_search), 1)
        self.assertEqual(items_search[0]["recipient_email"], "re****@aurascan.ai")


if __name__ == "__main__":
    unittest.main()
