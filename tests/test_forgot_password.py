import unittest
import os
import tempfile
import datetime
import sqlite3
import hashlib
from unittest.mock import patch, MagicMock

from security.domain.entities import Role, SecurityAuditLog, User
from security.infrastructure.repository import SQLiteUserRepository
from security.application.use_cases import AuthUseCases
from security.infrastructure.password import PasswordHasher


class TestForgotPasswordIntegration(unittest.TestCase):
    def setUp(self):
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        from persistence.infrastructure.repository import SQLitePersistenceRepository
        persistence_repo = SQLitePersistenceRepository(db_path=self.db_path)
        persistence_repo.initialize_db()
        self.repo = SQLiteUserRepository(db_path=self.db_path)
        self.repo.initialize_security_tables()
        self.auth_cases = AuthUseCases(user_repo=self.repo)

        # Register a test user
        self.auth_cases.register(
            email="clinician@aurascan.ai",
            password="StrongClinician@123",
            full_name="Dr. Jane Doe",
            role_str="doctor"
        )

        # Patch environment with missing SMTP settings to verify fail-fast by default
        self.env_patcher = patch.dict(os.environ, {
            "SMTP_HOST": "",
            "SMTP_PORT": "",
            "EMAIL_FROM": ""
        })
        self.env_patcher.start()

    def tearDown(self):
        self.env_patcher.stop()
        os.close(self.db_fd)
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    @patch("security.application.use_cases.is_testing_env", return_value=False)
    def test_forgot_password_fail_fast_on_missing_config(self, mock_is_test):
        """Verify that forgot_password fails fast and raises ValueError if SMTP is not configured."""
        with self.assertRaises(ValueError) as context:
            self.auth_cases.forgot_password("clinician@aurascan.ai")
        self.assertIn("Email delivery service is currently unavailable", str(context.exception))

    @patch.dict(os.environ, {
        "SMTP_HOST": "smtp.example.com",
        "SMTP_PORT": "587",
        "EMAIL_FROM": "noreply@example.com"
    })
    @patch("smtplib.SMTP")
    def test_forgot_password_success_flow(self, mock_smtp):
        """Verify the full recovery initiation flow when email config is valid."""
        # Setup mock connection
        mock_conn = MagicMock()
        mock_smtp.return_value = mock_conn

        # Call usecase
        res = self.auth_cases.forgot_password("clinician@aurascan.ai")
        self.assertIn("message", res)
        self.assertIn("reset_token", res) # Exposed in testing environment

        # Verify password_reset_tokens record exists
        token_hash = hashlib.sha256(res["reset_token"].encode("utf-8")).hexdigest()
        token_rec = self.repo.get_password_reset_token(token_hash)
        self.assertIsNotNone(token_rec)
        self.assertEqual(token_rec["used_at"], None)

        # Verify audit log recorded
        logs = self.repo.get_security_audit_logs(limit=10)
        reset_logs = [l for l in logs if l["event_type"] == "PASSWORD_RESET_REQUESTED"]
        self.assertEqual(len(reset_logs), 1)

    @patch.dict(os.environ, {
        "SMTP_HOST": "smtp.example.com",
        "SMTP_PORT": "587",
        "EMAIL_FROM": "noreply@example.com"
    })
    def test_forgot_password_nonexistent_account(self):
        """Verify anti-enumeration behavior: request returns success message but saves no token."""
        res = self.auth_cases.forgot_password("nonexistent@aurascan.ai")
        self.assertIn("message", res)
        self.assertNotIn("reset_token", res)

        # Verify no token was created in DB
        conn = sqlite3.connect(self.db_path)
        count = conn.execute("SELECT COUNT(*) FROM password_reset_tokens").fetchone()[0]
        conn.close()
        self.assertEqual(count, 0)

    @patch.dict(os.environ, {
        "SMTP_HOST": "smtp.example.com",
        "SMTP_PORT": "587",
        "EMAIL_FROM": "noreply@example.com"
    })
    @patch("smtplib.SMTP")
    def test_password_reset_success(self, mock_smtp):
        """Verify resetting password with a valid token."""
        mock_conn = MagicMock()
        mock_smtp.return_value = mock_conn

        # 1. Generate reset token
        res = self.auth_cases.forgot_password("clinician@aurascan.ai")
        token = res["reset_token"]

        # 2. Reset password
        reset_res = self.auth_cases.reset_password(
            reset_token_or_otp=token,
            email="clinician@aurascan.ai",
            new_password="NewSecurePassword@999"
        )
        self.assertEqual(reset_res["message"], "Password successfully reset.")

        # 3. Verify old password no longer works
        user_in_db = self.repo.get_by_email("clinician@aurascan.ai")
        self.assertFalse(PasswordHasher.verify_password("StrongClinician@123", user_in_db.password_hash))

        # 4. Verify new password works
        self.assertTrue(PasswordHasher.verify_password("NewSecurePassword@999", user_in_db.password_hash))

    @patch.dict(os.environ, {
        "SMTP_HOST": "smtp.example.com",
        "SMTP_PORT": "587",
        "EMAIL_FROM": "noreply@example.com"
    })
    @patch("smtplib.SMTP")
    def test_password_reset_expired_token(self, mock_smtp):
        """Verify resetting password with an expired token is rejected."""
        mock_conn = MagicMock()
        mock_smtp.return_value = mock_conn

        res = self.auth_cases.forgot_password("clinician@aurascan.ai")
        token = res["reset_token"]

        # Manually expire the token in the DB
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        expired_time = (datetime.datetime.utcnow() - datetime.timedelta(minutes=10)).isoformat()

        conn = sqlite3.connect(self.db_path)
        conn.execute("UPDATE password_reset_tokens SET expires_at = ? WHERE token_hash = ?", (expired_time, token_hash))
        conn.commit()
        conn.close()

        with self.assertRaises(ValueError) as context:
            self.auth_cases.reset_password(
                reset_token_or_otp=token,
                email="clinician@aurascan.ai",
                new_password="NewSecurePassword@999"
            )
        self.assertIn("expired", str(context.exception))

    def test_password_reset_invalid_token(self):
        """Verify resetting password with an invalid token is rejected."""
        with self.assertRaises(ValueError) as context:
            self.auth_cases.reset_password(
                reset_token_or_otp="invalid-token-1234",
                email="clinician@aurascan.ai",
                new_password="NewSecurePassword@999"
            )
        self.assertIn("Invalid or expired", str(context.exception))


if __name__ == "__main__":
    unittest.main()
