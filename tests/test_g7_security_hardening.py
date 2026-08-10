import unittest
import os
import tempfile
import datetime
import hashlib
import io
from unittest.mock import patch

from security.domain.entities import Role, SecurityAuditLog, User
from security.infrastructure.repository import SQLiteUserRepository
from security.application.use_cases import AuthUseCases
from security.application.tfa_service import TFAService
from dashboard.infrastructure.web_server import create_app


class TestG7SecurityHardening(unittest.TestCase):
    def setUp(self):
        self.orig_db_path = os.environ.get("DB_PATH")
        os.environ["G7_TESTING"] = "True"
        os.environ["TESTING"] = "True"
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.environ["DB_PATH"] = self.db_path

        self.app = create_app(db_path=self.db_path)
        self.app.config["TESTING"] = True
        self.app.config["DISABLE_CSRF"] = True
        self.client = self.app.test_client()

        self.repo = SQLiteUserRepository(db_path=self.db_path)
        self.repo.initialize_security_tables()
        self.auth_cases = AuthUseCases(user_repo=self.repo)

        # Register a test doctor user
        self.doctor_email = "doctor@aurascan.ai"
        self.doctor_pass = "DoctorPass@123"
        self.doctor_name = "Dr. Jane Doe"
        self.reg_res = self.auth_cases.register(
            email=self.doctor_email,
            password=self.doctor_pass,
            full_name=self.doctor_name,
            role_str="doctor"
        )

    def tearDown(self):
        os.close(self.db_fd)
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass
        os.environ.pop("G7_TESTING", None)
        os.environ.pop("TESTING", None)
        if self.orig_db_path is not None:
            os.environ["DB_PATH"] = self.orig_db_path
        else:
            os.environ.pop("DB_PATH", None)

    @patch("security.application.tfa_service.logger")
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_01_otp_logging_production_mode(self, mock_stdout, mock_logger):
        """Verify raw OTP is absent from logs and stdout when not in test environment."""
        with patch("security.application.use_cases.is_testing_env", return_value=False):
            TFAService.send_otp_via_email("doctor@aurascan.ai", "123456")

            # Verify stdout does not contain the OTP code
            stdout_val = mock_stdout.getvalue()
            self.assertNotIn("123456", stdout_val)

            # Verify logger.info was called, but did not contain the OTP code
            info_calls = [call[0][0] for call in mock_logger.info.call_args_list if call[0]]
            for msg in info_calls:
                self.assertNotIn("123456", msg)

    @patch("clinical_reporting.infrastructure.email_service.EmailService.send")
    def test_02_forgot_password_async_delivery(self, mock_send):
        """Verify forgot-password email delivery occurs asynchronously and returns generic message."""
        email = "doctor@aurascan.ai"

        # Call forgot_password (runs async send in background thread)
        res = self.auth_cases.forgot_password(email)

        # HTTP response should be immediate and generic
        self.assertEqual(res["message"], "If the account exists, password reset instructions have been sent.")

        # Check that mock_send is called. Since it runs in a background thread, we poll/wait for a bit.
        import time
        start_time = time.time()
        called = False
        while time.time() - start_time < 2.0:
            if mock_send.called:
                called = True
                break
            time.sleep(0.05)

        self.assertTrue(called, "EmailService.send was not called asynchronously")

        # Verify database email history record is created (G5 compatibility)
        conn = self.repo._get_connection()
        try:
            row = conn.execute("SELECT status, email_type FROM email_deliveries WHERE recipient_email = ? AND email_type = 'PASSWORD_RESET';", (email,)).fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["email_type"], "PASSWORD_RESET")
        finally:
            conn.close()

    def test_03_forgot_password_equivalence(self):
        """Verify forgot-password response is identical for existing and non-existing accounts."""
        # 1. Existing account
        res_existing = self.auth_cases.forgot_password(self.doctor_email)
        # 2. Non-existing account
        res_nonexistent = self.auth_cases.forgot_password("ghost@aurascan.ai")

        self.assertEqual(res_existing["message"], res_nonexistent["message"])
        self.assertEqual(res_existing["message"], "If the account exists, password reset instructions have been sent.")

    def test_04_resend_verification_generic_responses(self):
        """Verify verification resend outputs identical message regardless of account presence or state."""
        # 1. Existing unverified user
        res_unverified = self.auth_cases.resend_verification(self.doctor_email)
        self.assertEqual(res_unverified["message"], "If the account exists, verification instructions have been sent.")

        # 2. Existing verified user (verify doctor user first)
        user = self.repo.get_by_email(self.doctor_email)
        user.is_verified = True
        self.repo.update_user(user)

        res_verified = self.auth_cases.resend_verification(self.doctor_email)
        self.assertEqual(res_verified["message"], "If the account exists, verification instructions have been sent.")

        # 3. Unknown account
        res_unknown = self.auth_cases.resend_verification("ghost@aurascan.ai")
        self.assertEqual(res_unknown["message"], "If the account exists, verification instructions have been sent.")

    def test_05_otp_resend_cooldown(self):
        """Verify OTP cooldown restricts rapid generation on the server side."""
        # Set doctor user as verified and enable 2FA
        user = self.repo.get_by_email(self.doctor_email)
        user.is_verified = True
        user.two_factor_enabled = True
        self.repo.update_user(user)

        # Enable OTP cooldown explicitly in environment
        with patch.dict(os.environ, {"OTP_RESEND_COOLDOWN_SECONDS": "60"}):
            # 1. First request login -> succeeds (starts 2FA, sends OTP)
            res1 = self.auth_cases.login(self.doctor_email, self.doctor_pass)
            self.assertTrue(res1["requires_2fa"])
            self.assertIn("otp_code", res1)

            # 2. Second request login immediately -> rejected with ValueError (cooldown active)
            with self.assertRaises(ValueError) as ctx:
                self.auth_cases.login(self.doctor_email, self.doctor_pass)
            self.assertIn("cooldown active", str(ctx.exception).lower())

            # 3. Fast-forward past cooldown (modifying database created_at timestamp to 70s ago)
            conn = self.repo._get_connection()
            try:
                past_time = (datetime.datetime.utcnow() - datetime.timedelta(seconds=70)).isoformat()
                conn.execute("UPDATE otp_codes SET created_at = ? WHERE user_id = ?;", (past_time, user.id))
                conn.commit()
            finally:
                conn.close()

            # 4. Third request login -> succeeds again
            res3 = self.auth_cases.login(self.doctor_email, self.doctor_pass)
            self.assertTrue(res3["requires_2fa"])

    @patch("security.application.use_cases.is_testing_env", return_value=False)
    def test_06_production_mode_does_not_expose_otp(self, mock_test):
        """Verify production mode never returns OTP or tokens in responses."""
        user = self.repo.get_by_email(self.doctor_email)
        user.is_verified = True
        user.two_factor_enabled = True
        self.repo.update_user(user)

        res = self.auth_cases.login(self.doctor_email, self.doctor_pass)
        self.assertTrue(res["requires_2fa"])
        self.assertNotIn("otp_code", res)

    @patch("security.application.use_cases.is_testing_env", return_value=False)
    def test_07_production_mode_does_not_autoverify(self, mock_test):
        """Verify production mode registers users as unverified."""
        self.auth_cases.register(
            email="new_prod@aurascan.ai",
            password="StrongPassword@123",
            full_name="Prod User",
            role_str="patient"
        )
        user = self.repo.get_by_email("new_prod@aurascan.ai")
        self.assertFalse(user.is_verified)

    @patch("security.application.use_cases.is_testing_env", return_value=False)
    def test_08_production_mode_enforces_rate_limiting(self, mock_test):
        """Verify production mode does not bypass rate limiting."""
        from api.routes.auth_routes import enforce_rate_limit
        from fastapi import HTTPException

        for _ in range(5):
            enforce_rate_limit("test_limit_key", max_requests=5, window_seconds=60)

        with self.assertRaises(HTTPException) as ctx:
            enforce_rate_limit("test_limit_key", max_requests=5, window_seconds=60)
        self.assertEqual(ctx.exception.status_code, 429)


if __name__ == "__main__":
    unittest.main()
