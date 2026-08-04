import unittest
import os
import tempfile
import sqlite3
import time
from security.domain.entities import User, Role, TokenType, OtpPurpose, SecurityAuditLog
from security.infrastructure.password import PasswordHasher
from security.infrastructure.jwt_service import JWTService
from security.infrastructure.otp_service import OTPService
from security.infrastructure.rate_limiter import RateLimiter
from security.infrastructure.repository import SQLiteUserRepository
from security.application.use_cases import AuthUseCases


class TestSecuritySystem(unittest.TestCase):
    def setUp(self):
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        self.repo = SQLiteUserRepository(db_path=self.db_path)
        self.repo.initialize_security_tables()
        self.admin = self.repo.bootstrap_admin(admin_email="testadmin@aurascan.ai", admin_pass="Admin@123456")
        self.jwt_svc = JWTService(secret_key="test_secret_key_123456789_minimum_32bytes_required")
        self.auth_cases = AuthUseCases(user_repo=self.repo, jwt_service=self.jwt_svc)

    def tearDown(self):
        os.close(self.db_fd)
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    # 1. Password Hashing & OWASP Strength Validation
    def test_password_strength_and_hashing(self):
        # Weak password
        valid, msg = PasswordHasher.validate_password_strength("weak")
        self.assertFalse(valid)
        self.assertIn("at least 8 characters", msg)

        # Missing special char
        valid, msg = PasswordHasher.validate_password_strength("NoSpecial123")
        self.assertFalse(valid)
        self.assertIn("special character", msg)

        # Strong password
        valid, msg = PasswordHasher.validate_password_strength("StrongPass@123")
        self.assertTrue(valid)

        # Hash and Verify
        hashed = PasswordHasher.hash_password("StrongPass@123")
        self.assertTrue(PasswordHasher.verify_password("StrongPass@123", hashed))
        self.assertFalse(PasswordHasher.verify_password("WrongPass@123", hashed))

    # 2. User Registration & Email Verification
    def test_user_registration_and_verification(self):
        reg_res = self.auth_cases.register(
            email="doctor@hospital.org",
            password="DoctorPass@123",
            full_name="Dr. Gregory House",
            role_str="doctor"
        )
        self.assertIn("user", reg_res)
        self.assertEqual(reg_res["user"]["email"], "doctor@hospital.org")
        self.assertEqual(reg_res["user"]["role"], "doctor")
        self.assertFalse(reg_res["user"]["is_verified"])

        # Try duplicate registration
        with self.assertRaises(ValueError):
            self.auth_cases.register("doctor@hospital.org", "DoctorPass@123", "Duplicate", "doctor")

        # Verify email using OTP
        otp_code = reg_res["verification_otp"]
        ver_res = self.auth_cases.verify_email(token_or_otp=otp_code, email="doctor@hospital.org")
        self.assertTrue(ver_res["user"]["is_verified"])

    # 3. User Login & 2FA Flow
    def test_login_and_2fa(self):
        # Register user
        reg = self.auth_cases.register("patient@test.com", "PatientPass@123", "John Patient", "patient")
        
        # Successful login
        login_res = self.auth_cases.login("patient@test.com", "PatientPass@123")
        self.assertFalse(login_res["requires_2fa"])
        self.assertIn("access_token", login_res)

        # Decode access token
        payload = self.jwt_svc.decode_token(login_res["access_token"])
        self.assertIsNotNone(payload)
        self.assertEqual(payload.email, "patient@test.com")
        self.assertEqual(payload.role, Role.PATIENT)

        # Invalid password login attempt
        with self.assertRaises(ValueError):
            self.auth_cases.login("patient@test.com", "WrongPassword@123")

        # Enable 2FA for user
        user = self.repo.get_by_email("patient@test.com")
        self.auth_cases.update_profile(user.id, enable_2fa=True)

        # Login with 2FA enabled
        two_fa_login = self.auth_cases.login("patient@test.com", "PatientPass@123")
        self.assertTrue(two_fa_login["requires_2fa"])
        self.assertIn("otp_code", two_fa_login)

        # Verify 2FA OTP
        otp_res = self.auth_cases.verify_login_otp(user.id, two_fa_login["otp_code"])
        self.assertIn("access_token", otp_res)

    # 4. Token Revocation (Logout)
    def test_token_revocation(self):
        login_res = self.auth_cases.login("testadmin@aurascan.ai", "Admin@123456")
        token = login_res["access_token"]
        payload = self.jwt_svc.decode_token(token)

        self.assertFalse(self.repo.is_token_revoked(payload.jti))

        # Logout
        self.auth_cases.logout(token)
        self.assertTrue(self.repo.is_token_revoked(payload.jti))

    # 5. Forgot & Reset Password
    def test_forgot_and_reset_password(self):
        self.auth_cases.register("user@test.com", "OldPassword@123", "User Reset", "patient")
        forgot_res = self.auth_cases.forgot_password("user@test.com")
        reset_otp = forgot_res["reset_otp"]

        # Reset password
        reset_res = self.auth_cases.reset_password(
            reset_token_or_otp=reset_otp,
            new_password="NewPassword@123",
            email="user@test.com"
        )
        self.assertIn("successfully", reset_res["message"])

        # Verify login with new password
        login_res = self.auth_cases.login("user@test.com", "NewPassword@123")
        self.assertIn("access_token", login_res)

    # 6. Rate Limiting
    def test_rate_limiter(self):
        limiter = RateLimiter()
        key = "login:192.168.1.1"
        for i in range(5):
            limited, wait = limiter.is_rate_limited(key, max_requests=5, window_seconds=60)
            self.assertFalse(limited)

        # 6th attempt should be blocked
        limited, wait = limiter.is_rate_limited(key, max_requests=5, window_seconds=60)
        self.assertTrue(limited)
        self.assertGreater(wait, 0)

    # 7. Audit Logging
    def test_audit_logs(self):
        self.repo.log_security_event(SecurityAuditLog(
            id=None,
            timestamp="2026-08-04T09:00:00",
            event_type="TEST_EVENT",
            user_id=1,
            email="admin@aurascan.ai",
            ip_address="127.0.0.1",
            status="SUCCESS",
            details="Test audit log entry"
        ))
        logs = self.repo.get_security_audit_logs(limit=10)
        self.assertGreaterEqual(len(logs), 1)
        self.assertEqual(logs[0]["event_type"], "TEST_EVENT")


if __name__ == "__main__":
    unittest.main()
