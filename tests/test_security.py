import unittest
import os
import tempfile
import sqlite3
import time
from security.domain.entities import User, Role, TokenType, SecurityAuditLog
from security.infrastructure.password import PasswordHasher
from security.infrastructure.jwt_service import JWTService
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

    # 2. User Registration
    def test_user_registration(self):
        reg_res = self.auth_cases.register(
            email="doctor@hospital.org",
            password="DoctorPass@123",
            full_name="Dr. Gregory House",
            role_str="doctor"
        )
        self.assertIn("user", reg_res)
        self.assertEqual(reg_res["user"]["email"], "doctor@hospital.org")
        self.assertEqual(reg_res["user"]["role"], "doctor")
        self.assertTrue(reg_res["user"]["is_verified"])

        # Try duplicate registration
        with self.assertRaises(ValueError):
            self.auth_cases.register("doctor@hospital.org", "DoctorPass@123", "Duplicate", "doctor")

    # 3. User Login (Simplified)
    def test_login(self):
        # Register user
        reg = self.auth_cases.register("patient@test.com", "PatientPass@123", "John Patient", "patient")

        # Successful login immediately after registration
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

    # 4. Token Revocation (Logout)
    def test_token_revocation(self):
        login_res = self.auth_cases.login("testadmin@aurascan.ai", "Admin@123456")
        token = login_res["access_token"]
        payload = self.jwt_svc.decode_token(token)

        self.assertFalse(self.repo.is_token_revoked(payload.jti))

        # Logout
        self.auth_cases.logout(token)
        self.assertTrue(self.repo.is_token_revoked(payload.jti))

    # 5. Rate Limiting
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

    # 6. Audit Logging
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

    # 7. Token Refresh, Rotation & Replay Attack Protection
    def test_token_refresh_and_rotation(self):
        # Register user
        reg = self.auth_cases.register("refresh_test@test.com", "RefreshPass@123", "Test Refresh", "patient")

        # Login
        login_res = self.auth_cases.login("refresh_test@test.com", "RefreshPass@123")
        refresh_token = login_res["refresh_token"]

        # Decode refresh token
        from security.infrastructure.jwt_service import TokenType
        payload = self.jwt_svc.decode_token(refresh_token, expected_type=TokenType.REFRESH)
        self.assertEqual(payload.type, TokenType.REFRESH)

        # Perform Refresh
        refresh_res = self.auth_cases.refresh_token(refresh_token)
        self.assertIn("access_token", refresh_res)
        self.assertIn("refresh_token", refresh_res)

        # Verify old refresh token JTI is now revoked in database
        self.assertTrue(self.repo.is_token_revoked(payload.jti))

        # Attempt to reuse the old refresh token (Token Replay Attack)
        with self.assertRaises(ValueError):
            self.auth_cases.refresh_token(refresh_token)

    # 8. Remember Me Custom Lifetimes
    def test_remember_me_lifetimes(self):
        reg = self.auth_cases.register("remember_test@test.com", "RememberPass@123", "Test Remember", "patient")

        # Login with remember_me=True
        login_true = self.auth_cases.login("remember_test@test.com", "RememberPass@123", remember_me=True)
        from security.infrastructure.jwt_service import TokenType
        payload_true = self.jwt_svc.decode_token(login_true["refresh_token"], expected_type=TokenType.REFRESH)
        duration_days_true = (payload_true.exp - payload_true.iat) / 86400.0
        self.assertAlmostEqual(duration_days_true, 30.0, places=1)

        # Login with remember_me=False
        login_false = self.auth_cases.login("remember_test@test.com", "RememberPass@123", remember_me=False)
        payload_false = self.jwt_svc.decode_token(login_false["refresh_token"], expected_type=TokenType.REFRESH)
        duration_days_false = (payload_false.exp - payload_false.iat) / 86400.0
        self.assertAlmostEqual(duration_days_false, 7.0, places=1)

    # 9. Verbose Exceptions
    def test_verbose_exceptions(self):
        from security.infrastructure.jwt_service import TokenExpiredError, TokenInvalidError
        
        # Invalid Token
        with self.assertRaises(TokenInvalidError):
            self.jwt_svc.decode_token("this_is_an_invalid_token_signature")

        # Expired Token
        import time
        from security.domain.entities import Role
        now = time.time()
        # Create a token that expired 10 seconds ago
        payload = {
            "sub": "user-uuid-123",
            "user_id": 1,
            "email": "user@test.com",
            "role": Role.PATIENT.value,
            "jti": "jti-123",
            "type": TokenType.ACCESS.value,
            "iat": int(now - 100),
            "exp": int(now - 10),
        }
        import jwt
        from security.infrastructure.jwt_service import ALGORITHM
        expired_token = jwt.encode(payload, self.jwt_svc.secret_key, algorithm=ALGORITHM)

        with self.assertRaises(TokenExpiredError):
            self.jwt_svc.decode_token(expired_token)


if __name__ == "__main__":
    unittest.main()
