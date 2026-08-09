import os
os.environ["DB_PATH"] = os.path.abspath("outputs/test_email_retry.db")

import unittest
import sqlite3
import datetime
from unittest.mock import patch, MagicMock

from clinical_reporting.application.services import ReportService, ReportServiceException, VersionNotFoundException, ReportNotFoundException
from clinical_reporting.infrastructure.email_config import EmailConfig, EmailConfigException
from clinical_reporting.infrastructure.email_service import (
    ConnectionException,
    AuthenticationException,
    AttachmentException,
    InvalidAddressException,
    ConfigurationException
)
from clinical_reporting.application.scheduler import EmailRetryScheduler
from persistence.infrastructure.repository import SQLitePersistenceRepository
from security.infrastructure.repository import SQLiteUserRepository
from security.domain.entities import User, Role


class TestEmailRetry(unittest.TestCase):
    """Phase G6 Email Delivery Retry & Failure Handling Tests."""

    def setUp(self):
        self.db_path = os.environ["DB_PATH"]
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

        # Bootstrap DB
        self.persistence_repo = SQLitePersistenceRepository(db_path=self.db_path)
        self.persistence_repo.initialize_db()

        self.user_repo = SQLiteUserRepository(db_path=self.db_path)
        self.user_repo.initialize_security_tables()

        # Create doctor actor user
        self.doctor = User(
            id=None,
            uuid="doc-uuid-9999",
            email="doctor@aurascan.ai",
            password_hash="fakehash",
            full_name="Dr. Retry",
            role=Role.DOCTOR,
            is_verified=True,
            is_active=True,
            created_at=datetime.datetime.utcnow().isoformat(),
            updated_at=datetime.datetime.utcnow().isoformat()
        )
        self.doctor = self.user_repo.create_user(self.doctor)

        # Setup reports data directly in DB
        self.pdf_dir = "outputs/clinical_reports"
        os.makedirs(self.pdf_dir, exist_ok=True)
        self.pdf_file_path = os.path.join(self.pdf_dir, "clinical_report_retry.pdf")
        with open(self.pdf_file_path, "wb") as f:
            f.write(b"%PDF-1.4 Mock PDF")

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT INTO patients (patient_id, name, age, gender, created_at)
                VALUES (?, ?, 30, 'Male', '2026-08-09T00:00:00');
            """, ("pat-retry", "Bob Retry"))
            conn.execute("""
                INSERT INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at)
                VALUES (1, 'RPT-RETRY-001', 'pat-retry', 1, 'FINAL', '2026-08-09T00:00:00', '2026-08-09T00:00:00');
            """)
            conn.execute("""
                INSERT INTO report_versions (version_id, report_id, version_number, created_at, pdf_path, json_path, checksum, status)
                VALUES (1, 1, 1, '2026-08-09T00:00:00', ?, 'outputs/clinical_reports/report_retry.json', 'hash123', 'FINAL');
            """, (self.pdf_file_path,))
            conn.commit()
        finally:
            conn.close()

        self.service = ReportService(db_path=self.db_path)

    def tearDown(self):
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

    def test_01_failure_classification(self):
        """Verify that classify_email_error maps exceptions to the correct transient/permanent categories."""
        # Transient Exceptions
        code, retryable = self.service.classify_email_error(ConnectionException("timeout occurred"))
        self.assertEqual(code, "SMTP_TIMEOUT")
        self.assertTrue(retryable)

        code, retryable = self.service.classify_email_error(ConnectionException("connection refused"))
        self.assertEqual(code, "SMTP_CONNECTION_FAILED")
        self.assertTrue(retryable)

        code, retryable = self.service.classify_email_error(Exception("421 temporary server error"))
        self.assertEqual(code, "SMTP_TEMPORARY_UNAVAILABLE")
        self.assertTrue(retryable)

        # Permanent Exceptions
        code, retryable = self.service.classify_email_error(AuthenticationException("invalid login credentials"))
        self.assertEqual(code, "SMTP_AUTH_FAILED")
        self.assertFalse(retryable)

        code, retryable = self.service.classify_email_error(InvalidAddressException("consecutive dots"))
        self.assertEqual(code, "INVALID_RECIPIENT")
        self.assertFalse(retryable)

        code, retryable = self.service.classify_email_error(AttachmentException("does not exist"))
        self.assertEqual(code, "MISSING_PDF")
        self.assertFalse(retryable)

        code, retryable = self.service.classify_email_error(ConfigurationException("host missing"))
        self.assertEqual(code, "INVALID_CONFIGURATION")
        self.assertFalse(retryable)

    @patch.dict(os.environ, {
        "EMAIL_MAX_ATTEMPTS": "-1",
        "EMAIL_RETRY_BASE_DELAY": "10",
        "EMAIL_RETRY_MAX_DELAY": "100"
    })
    def test_02_retry_policy_validation_negative_attempts(self):
        """Verify that negative attempts are rejected by validation."""
        config = EmailConfig()
        with self.assertRaises(EmailConfigException):
            config.validate(active=True)

    @patch.dict(os.environ, {
        "EMAIL_MAX_ATTEMPTS": "3",
        "EMAIL_RETRY_BASE_DELAY": "100",
        "EMAIL_RETRY_MAX_DELAY": "10"
    })
    def test_03_retry_policy_validation_invalid_delay_bounds(self):
        """Verify that max delay less than base delay is rejected."""
        config = EmailConfig()
        with self.assertRaises(EmailConfigException):
            config.validate(active=True)

    @patch.dict(os.environ, {
        "EMAIL_RETRY_ENABLED": "True",
        "EMAIL_MAX_ATTEMPTS": "5",
        "EMAIL_RETRY_BASE_DELAY": "2",
        "EMAIL_RETRY_MAX_DELAY": "10"
    })
    @patch("clinical_reporting.infrastructure.email_service.EmailService.send")
    def test_04_backoff_calculation(self, mock_send):
        """Verify backoff exponentially increases and caps at max delay."""
        mock_send.side_effect = ConnectionException("timeout")

        # 1. Attempt 1 (attempt_count=0 when starting, increments to 1)
        with self.assertRaises(ReportServiceException):
            self.service.send_report_email(
                report_id=1,
                actor=self.doctor,
                recipient_email="doctor@aurascan.ai",
                version=1
            )

        # Inspect next_retry_at and backoff
        conn = sqlite3.connect(self.db_path)
        try:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM email_deliveries;").fetchone()
            self.assertEqual(row["status"], "RETRY_PENDING")
            self.assertEqual(row["attempt_count"], 1)
            self.assertEqual(row["last_failure_code"], "SMTP_TIMEOUT")
            self.assertTrue(row["retryable"])
            
            # Since attempt_count = 1, delay = 2 * 2^(1 - 1) = 2s
            next_retry = datetime.datetime.fromisoformat(row["next_retry_at"])
            attempted = datetime.datetime.fromisoformat(row["attempted_at"])
            diff = (next_retry - attempted).total_seconds()
            self.assertTrue(2.0 <= diff <= 3.0)  # base delay 2s + small random jitter
            delivery_id = row["id"]
        finally:
            conn.close()

        # 2. Attempt 2 (attempt_count=1)
        with self.assertRaises(ReportServiceException):
            self.service.execute_email_delivery(
                delivery_id=delivery_id,
                report_id=1,
                actor_user_id=self.doctor.id,
                recipient_email="doctor@aurascan.ai",
                version=1,
                attempt_count=1,
                max_attempts=5
            )

        # Inspect next_retry_at after second attempt
        conn = sqlite3.connect(self.db_path)
        try:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM email_deliveries;").fetchone()
            self.assertEqual(row["attempt_count"], 2)
            # delay = 2 * 2^(2 - 1) = 4s
            next_retry = datetime.datetime.fromisoformat(row["next_retry_at"])
            last_attempt = datetime.datetime.fromisoformat(row["last_attempt_at"])
            diff = (next_retry - last_attempt).total_seconds()
            self.assertTrue(4.0 <= diff <= 5.0)
        finally:
            conn.close()

        # 3. Attempt 3 (attempt_count=3 -> 4th attempt)
        with self.assertRaises(ReportServiceException):
            self.service.execute_email_delivery(
                delivery_id=delivery_id,
                report_id=1,
                actor_user_id=self.doctor.id,
                recipient_email="doctor@aurascan.ai",
                version=1,
                attempt_count=3,
                max_attempts=5
            )

        conn = sqlite3.connect(self.db_path)
        try:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM email_deliveries;").fetchone()
            self.assertEqual(row["attempt_count"], 4)
            # delay = 2 * 2^(4 - 1) = 16s. Capped at max_delay=10.
            next_retry = datetime.datetime.fromisoformat(row["next_retry_at"])
            last_attempt = datetime.datetime.fromisoformat(row["last_attempt_at"])
            diff = (next_retry - last_attempt).total_seconds()
            self.assertTrue(10.0 <= diff <= 11.5)
        finally:
            conn.close()

    def test_05_idempotency_protection(self):
        """Verify that optimistic claiming guarantees only one worker can process a retry."""
        # Insert a retryable record
        conn = sqlite3.connect(self.db_path)
        try:
            now = datetime.datetime.utcnow().isoformat()
            conn.execute("""
                INSERT INTO email_deliveries (
                    report_id, actor_user_id, recipient_email, status, attempted_at, created_at, updated_at,
                    attempt_count, max_attempts, next_retry_at, retryable
                )
                VALUES (1, ?, 'doctor@aurascan.ai', 'RETRY_PENDING', ?, ?, ?, 1, 3, ?, 1);
            """, (self.doctor.id, now, now, now, now))
            conn.commit()
            cursor = conn.execute("SELECT id FROM email_deliveries LIMIT 1;")
            delivery_id = cursor.fetchone()[0]
        finally:
            conn.close()

        # Concurrent claim simulation
        # Thread A claims
        conn_a = sqlite3.connect(self.db_path)
        cursor_a = conn_a.cursor()
        now_a = datetime.datetime.utcnow().isoformat()
        cursor_a.execute("""
            UPDATE email_deliveries
            SET status = 'SENDING', updated_at = ?
            WHERE id = ? AND status = 'RETRY_PENDING';
        """, (now_a, delivery_id))
        conn_a.commit()
        claimed_a = cursor_a.rowcount > 0
        conn_a.close()

        # Thread B claims (since status is now SENDING, this must fail)
        conn_b = sqlite3.connect(self.db_path)
        cursor_b = conn_b.cursor()
        now_b = datetime.datetime.utcnow().isoformat()
        cursor_b.execute("""
            UPDATE email_deliveries
            SET status = 'SENDING', updated_at = ?
            WHERE id = ? AND status = 'RETRY_PENDING';
        """, (now_b, delivery_id))
        conn_b.commit()
        claimed_b = cursor_b.rowcount > 0
        conn_b.close()

        self.assertTrue(claimed_a)
        self.assertFalse(claimed_b)

    @patch.dict(os.environ, {
        "EMAIL_RETRY_ENABLED": "True",
        "EMAIL_MAX_ATTEMPTS": "3",
        "EMAIL_RETRY_BASE_DELAY": "1",
        "EMAIL_RETRY_MAX_DELAY": "5"
    })
    @patch("clinical_reporting.infrastructure.email_service.EmailService.send")
    def test_06_scheduler_execution_and_lifecycle(self, mock_send):
        """Verify that the scheduler picks up and runs eligible retry records, transitioning states properly."""
        mock_send.side_effect = ConnectionException("SMTP network loss")

        # 1. Send triggers transient fail -> RETRY_PENDING
        with self.assertRaises(ReportServiceException):
            self.service.send_report_email(
                report_id=1,
                actor=self.doctor,
                recipient_email="doctor@aurascan.ai",
                version=1
            )

        # Force next_retry_at to be in the past to trigger immediately
        past_time = (datetime.datetime.utcnow() - datetime.timedelta(seconds=10)).isoformat()
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("UPDATE email_deliveries SET next_retry_at = ?;", (past_time,))
            conn.commit()
        finally:
            conn.close()

        # Mock success for the retry
        mock_send.side_effect = None

        # Start scheduler manually and process pending retries
        scheduler = EmailRetryScheduler(db_path=self.db_path)
        scheduler._process_pending_retries()

        # Inspect database: status should be SENT, attempt_count=2
        conn = sqlite3.connect(self.db_path)
        try:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM email_deliveries;").fetchone()
            self.assertEqual(row["status"], "SENT")
            self.assertEqual(row["attempt_count"], 2)
            self.assertIsNotNone(row["sent_at"])
            self.assertIsNone(row["next_retry_at"])
        finally:
            conn.close()

    @patch.dict(os.environ, {
        "EMAIL_RETRY_ENABLED": "True",
        "EMAIL_MAX_ATTEMPTS": "3",
        "EMAIL_RETRY_BASE_DELAY": "1",
        "EMAIL_RETRY_MAX_DELAY": "5"
    })
    @patch("clinical_reporting.infrastructure.email_service.EmailService.send")
    def test_07_observability_and_audit_logging(self, mock_send):
        """Verify that security audit log registers all delivery success/failure/retry transitions."""
        mock_send.side_effect = ConnectionException("timeout")

        with self.assertRaises(ReportServiceException):
            self.service.send_report_email(
                report_id=1,
                actor=self.doctor,
                recipient_email="doctor@aurascan.ai",
                version=1
            )

        # Inspect the security audit logs in the DB
        conn = sqlite3.connect(self.db_path)
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM security_audit_logs ORDER BY timestamp DESC;").fetchall()
            event_types = [r["event_type"] for r in rows]
            self.assertIn("REPORT_EMAIL_RETRY_SCHEDULED", event_types)
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
