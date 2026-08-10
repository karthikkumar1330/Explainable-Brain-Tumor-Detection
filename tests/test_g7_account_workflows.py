import unittest
import os
import tempfile
import datetime
import hashlib
import json
from security.domain.entities import Role, SecurityAuditLog, User
from security.infrastructure.repository import SQLiteUserRepository
from security.application.use_cases import AuthUseCases
from security.application.tfa_service import TFAService
from dashboard.infrastructure.web_server import create_app


class TestG7AccountWorkflows(unittest.TestCase):
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

    def test_registration_unverified_by_default(self):
        # Assert user is marked unverified on registration
        user = self.repo.get_by_email(self.doctor_email)
        self.assertIsNotNone(user)
        self.assertFalse(user.is_verified)
        self.assertIn("verification_token", self.reg_res)

        # Verify a record was inserted in email_deliveries
        conn = self.repo._get_connection()
        try:
            row = conn.execute("SELECT recipient_email, status, email_type FROM email_deliveries WHERE recipient_email = ?;", (self.doctor_email,)).fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["email_type"], "ACCOUNT_VERIFICATION")
        finally:
            conn.close()

    def test_block_unverified_login(self):
        # Logging in prior to verification should raise ValueError
        with self.assertRaises(ValueError) as ctx:
            self.auth_cases.login(self.doctor_email, self.doctor_pass)
        self.assertIn("Please verify your email address", str(ctx.exception))

    def test_verify_email_success_and_consumption(self):
        token = self.reg_res["verification_token"]

        # Verify email API endpoint works
        resp = self.client.post("/api/auth/verify-email", json={"token": token})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("verified", data["message"])

        # Check DB that user is verified
        user = self.repo.get_by_email(self.doctor_email)
        self.assertTrue(user.is_verified)

        # Token must be consumed
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        token_rec = self.repo.get_verification_token(token_hash)
        self.assertIsNotNone(token_rec["used_at"])

        # Second verification attempt must fail
        with self.assertRaises(ValueError) as ctx:
            self.auth_cases.verify_email(token)
        self.assertIn("already been used", str(ctx.exception))

    def test_resend_verification_anti_enumeration(self):
        # Active domain resend verification
        res = self.auth_cases.resend_verification(self.doctor_email)
        self.assertIn("verification_token", res)
        self.assertIn("instructions have been sent", res["message"])

        # Non-existing domain resend verification
        res_non = self.auth_cases.resend_verification("ghost@aurascan.ai")
        # Assert same generic response message returned
        self.assertEqual(res_non["message"], res["message"])
        # Should not expose token for non-existing email
        self.assertNotIn("verification_token", res_non)

    def test_forgot_password_anti_enumeration(self):
        # Recovery request on registered user
        res = self.auth_cases.forgot_password(self.doctor_email)
        self.assertIn("reset_token", res)
        self.assertIn("instructions have been sent", res["message"])

        # Recovery request on non-registered user
        res_non = self.auth_cases.forgot_password("ghost@aurascan.ai")
        # Assert response message matches registered user's exactly
        self.assertEqual(res_non["message"], res["message"])
        self.assertNotIn("reset_token", res_non)

    def test_password_strength_and_reset(self):
        token = self.auth_cases.forgot_password(self.doctor_email)["reset_token"]

        # Test password policy check (too short)
        with self.assertRaises(ValueError) as ctx:
            self.auth_cases.reset_password(token, self.doctor_email, "Weak@1")
        self.assertIn("at least 8 characters", str(ctx.exception))

        # Test password policy check (no special characters)
        with self.assertRaises(ValueError) as ctx:
            self.auth_cases.reset_password(token, self.doctor_email, "WeakPassword123")
        self.assertIn("at least one special character", str(ctx.exception))

        # Successful password reset
        new_pass = "StrongRecovered@123"
        res = self.auth_cases.reset_password(token, self.doctor_email, new_pass)
        self.assertIn("successfully reset", res["message"])

        # Reset token must be marked used
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        token_rec = self.repo.get_password_reset_token(token_hash)
        self.assertIsNotNone(token_rec["used_at"])

        # Check user database record: sessions_revoked_at is updated
        user = self.repo.get_by_email(self.doctor_email)
        self.assertIsNotNone(user.sessions_revoked_at)

        # Allow logging in now that user is verified and reset has been performed
        user.is_verified = True
        self.repo.update_user(user)
        login_res = self.auth_cases.login(self.doctor_email, new_pass)
        self.assertIsNotNone(login_res)

    def test_otp_attempt_count_locking(self):
        # Seed an OTP
        otp_code = "123456"
        TFAService.save_otp(self.db_path, 1, otp_code)

        # Attempt 1: Fail
        res1 = TFAService.verify_otp(self.db_path, 1, "000000")
        self.assertFalse(res1)

        # Attempt 2: Fail
        res2 = TFAService.verify_otp(self.db_path, 1, "999999")
        self.assertFalse(res2)

        # Check DB states
        conn = self.repo._get_connection()
        try:
            row = conn.execute("SELECT attempt_count, max_attempts FROM otp_codes WHERE user_id = 1;").fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["attempt_count"], 2)
        finally:
            conn.close()

        # Attempt 3: Fail (Max reached, deletes record)
        res3 = TFAService.verify_otp(self.db_path, 1, "888888")
        self.assertFalse(res3)

        # Check DB states: OTP should be deleted completely
        conn = self.repo._get_connection()
        try:
            row = conn.execute("SELECT otp_code FROM otp_codes WHERE user_id = 1;").fetchone()
            self.assertIsNone(row)
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
