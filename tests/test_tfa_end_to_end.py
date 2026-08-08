import unittest
import os
import tempfile
import io
import json
from dashboard.infrastructure.web_server import create_app
from security.domain.entities import Role, User
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.password import PasswordHasher


class TestTwoFactorAuthentication(unittest.TestCase):
    """Integration test suite to verify Two-Factor Authentication (2FA) functionality."""

    def setUp(self):
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        self.app = create_app(db_path=self.db_path)
        self.app.config["TESTING"] = True
        self.app.config["DISABLE_CSRF"] = True  # Bypass CSRF checks for ease of testing API
        self.client = self.app.test_client()

        # Initialize tables
        self.repo = SQLiteUserRepository(db_path=self.db_path)
        self.repo.initialize_security_tables()

        # Create user
        self.password = "SecurePassword@123"
        self.user = User(
            id=None,
            uuid="test-doctor-uuid",
            email="doctor@aurascan.ai",
            password_hash=PasswordHasher.hash_password(self.password),
            full_name="Doctor User",
            role=Role.DOCTOR,
            is_verified=True,
            is_active=True
        )
        self.repo.create_user(self.user)

    def tearDown(self):
        os.close(self.db_fd)
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def _login(self):
        resp = self.client.post("/api/auth/login", json={
            "email": "doctor@aurascan.ai",
            "password": self.password
        })
        self.assertEqual(resp.status_code, 200)
        return resp.get_json()

    def test_2fa_enable_and_disable_flow(self):
        """PUT /api/auth/profile with enable_2fa should toggle 2FA state and generate recovery codes."""
        # Establish session via login
        login_data = self._login()
        self.assertFalse(login_data["requires_2fa"])

        # Enable 2FA
        resp = self.client.put("/api/auth/profile", json={
            "enable_2fa": True
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["user"]["two_factor_enabled"])
        self.assertIn("recovery_codes", data)
        self.assertEqual(len(data["recovery_codes"]), 5)

        # Verify database record updated
        db_user = self.repo.get_by_id(self.user.id)
        self.assertTrue(db_user.two_factor_enabled)
        self.assertIsNotNone(db_user.two_factor_secret)
        self.assertIsNotNone(db_user.two_factor_recovery_codes)

        # Disable 2FA
        resp = self.client.put("/api/auth/profile", json={
            "enable_2fa": False
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertFalse(data["user"]["two_factor_enabled"])

        # Verify database record cleared
        db_user = self.repo.get_by_id(self.user.id)
        self.assertFalse(db_user.two_factor_enabled)
        self.assertIsNone(db_user.two_factor_secret)
        self.assertIsNone(db_user.two_factor_recovery_codes)

    def test_2fa_login_and_otp_verification(self):
        """Flow: Enable 2FA -> Login -> returns requires_2fa -> Verify OTP succeeds."""
        # Enable 2FA
        self._login()
        resp_enable = self.client.put("/api/auth/profile", json={"enable_2fa": True})
        self.assertEqual(resp_enable.status_code, 200)

        # Attempt Login again (should require 2FA)
        resp_login = self.client.post("/api/auth/login", json={
            "email": "doctor@aurascan.ai",
            "password": self.password
        })
        self.assertEqual(resp_login.status_code, 200)
        login_data = resp_login.get_json()
        self.assertTrue(login_data["requires_2fa"])
        self.assertIn("otp_code", login_data)
        otp_code = login_data["otp_code"]
        user_id = login_data["user_id"]

        # Attempt verification with wrong OTP code
        resp_verify_fail = self.client.post("/api/auth/verify-otp", json={
            "user_id": user_id,
            "otp_code": "999999"
        })
        self.assertEqual(resp_verify_fail.status_code, 400)
        self.assertIn("Invalid or expired 2FA code", resp_verify_fail.get_json()["error"])

        # Attempt verification with correct OTP code
        resp_verify_success = self.client.post("/api/auth/verify-otp", json={
            "user_id": user_id,
            "otp_code": otp_code
        })
        self.assertEqual(resp_verify_success.status_code, 200)
        verify_data = resp_verify_success.get_json()
        self.assertIn("access_token", verify_data)
        self.assertTrue(verify_data["user"]["two_factor_enabled"]) # two_factor_enabled is returned in User dict (mapped as bool)

    def test_2fa_login_via_recovery_code(self):
        """Flow: Enable 2FA -> Login -> Verify via single-use recovery code -> code is consumed."""
        # Enable 2FA and get recovery codes
        self._login()
        resp_enable = self.client.put("/api/auth/profile", json={"enable_2fa": True})
        self.assertEqual(resp_enable.status_code, 200)
        recovery_codes = resp_enable.get_json()["recovery_codes"]
        recovery_code = recovery_codes[0]

        # Login to trigger requires_2fa
        resp_login = self.client.post("/api/auth/login", json={
            "email": "doctor@aurascan.ai",
            "password": self.password
        })
        login_data = resp_login.get_json()
        user_id = login_data["user_id"]

        # Verify using valid recovery code
        resp_verify = self.client.post("/api/auth/verify-otp", json={
            "user_id": user_id,
            "otp_code": recovery_code
        })
        self.assertEqual(resp_verify.status_code, 200)
        self.assertIn("access_token", resp_verify.get_json())

        # Attempt to use the same recovery code again (should fail because it's consumed)
        resp_login_again = self.client.post("/api/auth/login", json={
            "email": "doctor@aurascan.ai",
            "password": self.password
        })
        user_id = resp_login_again.get_json()["user_id"]

        resp_verify_reuse = self.client.post("/api/auth/verify-otp", json={
            "user_id": user_id,
            "otp_code": recovery_code
        })
        self.assertEqual(resp_verify_reuse.status_code, 400)
        self.assertIn("Invalid or expired 2FA code", resp_verify_reuse.get_json()["error"])


if __name__ == "__main__":
    unittest.main()
