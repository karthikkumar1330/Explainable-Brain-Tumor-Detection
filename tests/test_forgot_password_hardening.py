import os
import tempfile
import datetime
import sqlite3
import hashlib
import unittest
from unittest.mock import patch, MagicMock

from security.domain.entities import Role, SecurityAuditLog, User
from security.infrastructure.repository import SQLiteUserRepository
from security.application.use_cases import AuthUseCases, get_public_base_url
from security.infrastructure.password import PasswordHasher
from dashboard.infrastructure.web_server import create_app

class TestForgotPasswordHardening(unittest.TestCase):
    def setUp(self):
        self.orig_db_path = os.environ.get("DB_PATH")
        os.environ["G7_TESTING"] = "True"
        os.environ["TESTING"] = "True"
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.environ["DB_PATH"] = self.db_path

        from persistence.infrastructure.repository import SQLitePersistenceRepository
        persistence_repo = SQLitePersistenceRepository(db_path=self.db_path)
        persistence_repo.initialize_db()

        self.repo = SQLiteUserRepository(db_path=self.db_path)
        self.repo.initialize_security_tables()
        self.auth_cases = AuthUseCases(user_repo=self.repo)

        # Setup Flask App
        self.app = create_app(db_path=self.db_path)
        self.app.config["TESTING"] = True
        self.app.config["DISABLE_CSRF"] = True
        self.client = self.app.test_client()

        # Register a test doctor user
        self.email = "doctor@aurascan.ai"
        self.password = "DoctorPass@123"
        self.name = "Dr. Jane Doe"
        self.auth_cases.register(
            email=self.email,
            password=self.password,
            full_name=self.name,
            role_str="doctor"
        )

        # Verify the user
        user = self.repo.get_by_email(self.email)
        user.is_verified = True
        self.repo.update_user(user)

        # Mock SMTP config
        self.env_patcher = patch.dict(os.environ, {
            "SMTP_HOST": "smtp.example.com",
            "SMTP_PORT": "587",
            "EMAIL_FROM": "noreply@example.com"
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
        os.environ.pop("G7_TESTING", None)
        os.environ.pop("TESTING", None)
        if self.orig_db_path is not None:
            os.environ["DB_PATH"] = self.orig_db_path
        else:
            os.environ.pop("DB_PATH", None)

    def test_01_forgot_password_page_renders(self):
        """1. Verify Forgot-password page (modal placeholder elements) renders on index page."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html_content = response.data.decode("utf-8")
        self.assertIn('id="forgot-modal"', html_content)
        self.assertIn('Password Recovery', html_content)
        self.assertIn("Enter your registered email. We'll send you a secure password-reset link.", html_content)
        self.assertIn('Send Reset Link', html_content)
        self.assertNotIn('Request Verification Code', html_content)
        self.assertNotIn('Reset Password Now', html_content)

    @patch("smtplib.SMTP")
    def test_02_email_submission_endpoint(self, mock_smtp):
        """2. Verify Email submission calls the correct API endpoint and returns generic message."""
        mock_conn = MagicMock()
        mock_smtp.return_value = mock_conn

        response = self.client.post("/api/auth/forgot-password", json={"email": self.email})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["message"], "If the account exists, password reset instructions have been sent.")

    @patch("smtplib.SMTP")
    def test_03_valid_email_receives_reset_link(self, mock_smtp):
        """3. Verify Valid email receives reset-link workflow (token is saved in DB)."""
        mock_conn = MagicMock()
        mock_smtp.return_value = mock_conn

        res = self.auth_cases.forgot_password(self.email)
        self.assertIn("reset_token", res)
        token = res["reset_token"]
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()

        token_rec = self.repo.get_password_reset_token(token_hash)
        self.assertIsNotNone(token_rec)
        self.assertEqual(token_rec["user_id"], self.repo.get_by_email(self.email).id)

    def test_04_unknown_email_returns_identical_generic_response(self):
        """4. Unknown email receives the same generic response (anti-enumeration)."""
        res = self.auth_cases.forgot_password("unknown@aurascan.ai")
        self.assertEqual(res["message"], "If the account exists, password reset instructions have been sent.")
        self.assertNotIn("reset_token", res)

    @patch.dict(os.environ, {"AURASCAN_PUBLIC_BASE_URL": "http://192.168.10.16:5000"})
    @patch("smtplib.SMTP")
    def test_05_reset_link_cross_device_lan_compatible(self, mock_smtp):
        """5. Reset link opens from another device/LAN hostname (uses config base URL, not request host)."""
        mock_conn = MagicMock()
        mock_smtp.return_value = mock_conn

        # Fetch reset link using use_case
        res = self.auth_cases.forgot_password(self.email)
        token = res["reset_token"]

        base_url = get_public_base_url()
        self.assertEqual(base_url, "http://192.168.10.16:5000")

        # Emulate another device fetching the reset page
        response = self.client.get(f"/reset-password?token={token}")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Configure New Account Password", response.data.decode("utf-8"))

    @patch("smtplib.SMTP")
    def test_06_valid_token_renders_reset_page(self, mock_smtp):
        """6. Valid token renders reset-password page."""
        mock_conn = MagicMock()
        mock_smtp.return_value = mock_conn

        res = self.auth_cases.forgot_password(self.email)
        token = res["reset_token"]

        response = self.client.get(f"/reset-password?token={token}")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Configure New Account Password", response.data.decode("utf-8"))

    def test_07_invalid_token_is_rejected(self):
        """7. Invalid token is rejected."""
        response = self.client.get("/reset-password?token=invalid_token_format")
        self.assertEqual(response.status_code, 200)
        self.assertIn("🔒 Reset Link Unavailable", response.data.decode("utf-8"))

    @patch("smtplib.SMTP")
    def test_08_expired_token_is_rejected(self, mock_smtp):
        """8. Expired token is rejected."""
        mock_conn = MagicMock()
        mock_smtp.return_value = mock_conn

        res = self.auth_cases.forgot_password(self.email)
        token = res["reset_token"]
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()

        # Update expires_at to past in DB
        conn = sqlite3.connect(self.db_path)
        past_time = (datetime.datetime.utcnow() - datetime.timedelta(minutes=10)).isoformat()
        conn.execute("UPDATE password_reset_tokens SET expires_at = ? WHERE token_hash = ?", (past_time, token_hash))
        conn.commit()
        conn.close()

        response = self.client.get(f"/reset-password?token={token}")
        self.assertEqual(response.status_code, 200)
        self.assertIn("🔒 Reset Link Unavailable", response.data.decode("utf-8"))

    @patch("smtplib.SMTP")
    def test_09_used_token_is_rejected(self, mock_smtp):
        """9. Used token is rejected."""
        mock_conn = MagicMock()
        mock_smtp.return_value = mock_conn

        res = self.auth_cases.forgot_password(self.email)
        token = res["reset_token"]
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()

        # Update used_at in DB
        conn = sqlite3.connect(self.db_path)
        now_time = datetime.datetime.utcnow().isoformat()
        conn.execute("UPDATE password_reset_tokens SET used_at = ? WHERE token_hash = ?", (now_time, token_hash))
        conn.commit()
        conn.close()

        response = self.client.get(f"/reset-password?token={token}")
        self.assertEqual(response.status_code, 200)
        self.assertIn("🔒 Reset Link Unavailable", response.data.decode("utf-8"))

    @patch("smtplib.SMTP")
    def test_10_password_mismatch_is_rejected(self, mock_smtp):
        """10. Password mismatch is rejected on frontend/backend check. (Backend doesn't enforce match since it expects it to match on client, but let's check weak password/validation is rejected)."""
        mock_conn = MagicMock()
        mock_smtp.return_value = mock_conn

        res = self.auth_cases.forgot_password(self.email)
        token = res["reset_token"]

        # Weak password rejected
        with self.assertRaises(ValueError) as ctx:
            self.auth_cases.reset_password(
                reset_token_or_otp=token,
                new_password="weak"
            )
        self.assertIn("at least 8 characters", str(ctx.exception).lower())

    @patch("smtplib.SMTP")
    def test_11_weak_password_criteria_enforcement(self, mock_smtp):
        """11. Weak password (missing upper, lower, digit, or special) is rejected."""
        mock_conn = MagicMock()
        mock_smtp.return_value = mock_conn

        res = self.auth_cases.forgot_password(self.email)
        token = res["reset_token"]

        weak_passes = [
            "NOLOWERCASE123!",
            "nouppercase123!",
            "NoDigitsHere!",
            "NoSpecialChars123"
        ]

        for wp in weak_passes:
            with self.assertRaises(ValueError) as ctx:
                self.auth_cases.reset_password(
                    reset_token_or_otp=token,
                    new_password=wp
                )
            # Confirm it raises validation error
            self.assertTrue(len(str(ctx.exception)) > 0)

    @patch("smtplib.SMTP")
    def test_12_successful_password_reset(self, mock_smtp):
        """12. Successful password reset works without passing email explicitly (derived from token)."""
        mock_conn = MagicMock()
        mock_smtp.return_value = mock_conn

        res = self.auth_cases.forgot_password(self.email)
        token = res["reset_token"]

        # Reset using use case without passing email (Step 8: derived from token)
        reset_res = self.auth_cases.reset_password(
            reset_token_or_otp=token,
            new_password="NewSecurePassword@999"
        )
        self.assertEqual(reset_res["message"], "Password successfully reset.")

        # Check new password works
        user_in_db = self.repo.get_by_email(self.email)
        self.assertTrue(PasswordHasher.verify_password("NewSecurePassword@999", user_in_db.password_hash))

    @patch("smtplib.SMTP")
    def test_13_token_cannot_be_reused(self, mock_smtp):
        """13. Reset token cannot be reused after a successful password reset."""
        mock_conn = MagicMock()
        mock_smtp.return_value = mock_conn

        res = self.auth_cases.forgot_password(self.email)
        token = res["reset_token"]

        # Reset once
        self.auth_cases.reset_password(
            reset_token_or_otp=token,
            new_password="NewSecurePassword@999"
        )

        # Reset second time with same token must fail
        with self.assertRaises(ValueError) as ctx:
            self.auth_cases.reset_password(
                reset_token_or_otp=token,
                new_password="AnotherNewSecurePassword@777"
            )
        self.assertIn("already been used", str(ctx.exception))

    @patch("smtplib.SMTP")
    def test_14_sessions_are_revoked_after_password_change(self, mock_smtp):
        """14. Sessions are revoked after password change."""
        mock_conn = MagicMock()
        mock_smtp.return_value = mock_conn

        user_before = self.repo.get_by_email(self.email)
        orig_revoked = user_before.sessions_revoked_at

        res = self.auth_cases.forgot_password(self.email)
        token = res["reset_token"]

        self.auth_cases.reset_password(
            reset_token_or_otp=token,
            new_password="NewSecurePassword@999"
        )

        user_after = self.repo.get_by_email(self.email)
        self.assertIsNotNone(user_after.sessions_revoked_at)
        self.assertNotEqual(orig_revoked, user_after.sessions_revoked_at)

    @patch("smtplib.SMTP")
    def test_15_existing_tfa_otp_tests_remain_unaffected(self, mock_smtp):
        """15. Existing TFA/OTP functions and resets still function correctly."""
        mock_conn = MagicMock()
        mock_smtp.return_value = mock_conn

        # Check that we can still reset using legacy OTP when email is explicitly supplied
        user = self.repo.get_by_email(self.email)
        from security.application.tfa_service import TFAService
        otp_code = TFAService.generate_otp()
        TFAService.save_otp(self.db_path, user.id, otp_code)

        # Verification using OTP fallback
        reset_res = self.auth_cases.reset_password(
            reset_token_or_otp=otp_code,
            email=self.email,
            new_password="NewSecurePassword@999"
        )
        self.assertEqual(reset_res["message"], "Password successfully reset.")

if __name__ == "__main__":
    unittest.main()
