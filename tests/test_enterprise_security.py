import unittest
import os
import tempfile
import time
import datetime
from flask import request
from dashboard.infrastructure.web_server import create_app
from security.domain.entities import Role, User
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.password import PasswordHasher


class TestEnterpriseSecurity(unittest.TestCase):
    """Test suite to verify CSRF protections, account lockout rules, secure headers, and audit user agent logging."""

    def setUp(self):
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        self.app = create_app(db_path=self.db_path)
        self.app.config["TESTING"] = True
        self.app.config["DISABLE_CSRF"] = False
        self.client = self.app.test_client()

        # Initialize tables
        self.repo = SQLiteUserRepository(db_path=self.db_path)
        self.repo.initialize_security_tables()

        # Create Patient
        self.patient_pass = "SecurePatient@123"
        self.patient = User(
            id=None,
            uuid="patient-uuid",
            email="patient@aurascan.ai",
            password_hash=PasswordHasher.hash_password(self.patient_pass),
            full_name="Patient User",
            role=Role.PATIENT,
            is_verified=True,
            is_active=True
        )
        self.repo.create_user(self.patient)

    def tearDown(self):
        os.close(self.db_fd)
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def _login(self):
        resp = self.client.post("/api/auth/login", json={
            "email": "patient@aurascan.ai",
            "password": self.patient_pass
        }, headers={"User-Agent": "Mozilla/5.0 TestBrowser"})
        self.assertEqual(resp.status_code, 200)
        return resp.get_json()

    def test_secure_headers_present(self):
        """Web responses should enforce OWASP recommended secure headers."""
        resp = self.client.get("/")
        self.assertEqual(resp.headers.get("X-Frame-Options"), "DENY")
        self.assertEqual(resp.headers.get("X-Content-Type-Options"), "nosniff")
        self.assertEqual(resp.headers.get("X-XSS-Protection"), "1; mode=block")
        self.assertIn("Content-Security-Policy", resp.headers)
        self.assertIn("Strict-Transport-Security", resp.headers)

    def test_csrf_exempt_routes(self):
        """State-changing authentication initiation endpoints should skip CSRF checks."""
        # /api/auth/login is exempt from CSRF
        resp = self.client.post("/api/auth/login", json={
            "email": "patient@aurascan.ai",
            "password": "wrong_password"
        })
        # Should return 400 invalid credentials instead of 403 CSRF failure
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Invalid email or password", resp.get_json()["error"])

    def test_csrf_protection_failures(self):
        """Protected state-changing routes should reject requests lacking proper CSRF cookies/headers."""
        self._login()  # Establish cookies/sessions

        # Attempt profile update with no CSRF header
        resp_no_header = self.client.put("/api/auth/profile", json={
            "full_name": "New Name",
            "email": "patient@aurascan.ai"
        })
        self.assertEqual(resp_no_header.status_code, 403)
        self.assertIn("CSRF verification failed", resp_no_header.get_json()["error"])

        # Attempt update with mismatched header
        self.client.set_cookie("csrf_token", "matchingcookiecsrf")
        resp_mismatch = self.client.put(
            "/api/auth/profile",
            json={"full_name": "New Name", "email": "patient@aurascan.ai"},
            headers={"X-CSRF-Token": "differentheadercsrf"}
        )
        self.assertEqual(resp_mismatch.status_code, 403)

    def test_csrf_protection_success(self):
        """Protected state-changing routes should accept matching CSRF credentials."""
        self._login()

        csrf_val = "1234abcd5678efgh"
        self.client.set_cookie("csrf_token", csrf_val)
        
        resp = self.client.put(
            "/api/auth/profile",
            json={"full_name": "Updated Unique Name", "email": "patient@aurascan.ai"},
            headers={"X-CSRF-Token": csrf_val}
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["user"]["full_name"], "Updated Unique Name")

    def test_brute_force_account_lockout(self):
        """Account lock out should trigger for 15 minutes after 5 consecutive login failures."""
        # Execute 5 incorrect logins
        for i in range(5):
            resp = self.client.post("/api/auth/login", json={
                "email": "patient@aurascan.ai",
                "password": "WrongPassword@123"
            })
            if i < 4:
                self.assertEqual(resp.status_code, 400)
                self.assertIn(f"Attempt {i+1}/5", resp.get_json()["error"])
            else:
                self.assertEqual(resp.status_code, 400)
                self.assertIn("temporarily locked", resp.get_json()["error"])

        # 6th attempt with CORRECT credentials should still fail due to lockout!
        resp_lockout = self.client.post("/api/auth/login", json={
            "email": "patient@aurascan.ai",
            "password": self.patient_pass
        })
        self.assertEqual(resp_lockout.status_code, 400)
        self.assertIn("temporarily locked", resp_lockout.get_json()["error"])

    def test_audit_logs_user_agent(self):
        """Audit logs must record the browser User Agent header for tracking devices."""
        self._login()
        
        # Verify database logs
        logs = self.repo.get_security_audit_logs(limit=10)
        success_logs = [l for l in logs if l["event_type"] == "LOGIN_SUCCESS"]
        self.assertTrue(len(success_logs) > 0)
        self.assertEqual(success_logs[0]["user_agent"], "Mozilla/5.0 TestBrowser")


    def test_timezone_aware_session_revocation(self):
        """Tokens issued before a session revocation event must be immediately invalidated timezone-safely."""
        login_data = self._login()
        token = login_data["access_token"]

        # Call profile with active token to verify it works initially
        self.client.set_cookie("csrf_token", "matching_csrf")
        resp_ok = self.client.put(
            "/api/auth/profile",
            json={"full_name": "Check Session", "email": "patient@aurascan.ai"},
            headers={"X-CSRF-Token": "matching_csrf"}
        )
        self.assertEqual(resp_ok.status_code, 200)

        # Trigger logout-other-devices to invalidate token sessions
        resp_revoke = self.client.post(
            "/api/auth/profile/logout-other-devices",
            headers={"X-CSRF-Token": "matching_csrf"}
        )
        self.assertEqual(resp_revoke.status_code, 200)

        # Now attempting to use the old token must return 401 TOKEN_REVOKED
        self.client.set_cookie("access_token", token)
        resp_blocked = self.client.put(
            "/api/auth/profile",
            json={"full_name": "Try Blocked Update", "email": "patient@aurascan.ai"},
            headers={"X-CSRF-Token": "matching_csrf"}
        )
        self.assertEqual(resp_blocked.status_code, 401)
        self.assertEqual(resp_blocked.get_json()["code"], "TOKEN_REVOKED")

    def test_account_lockout_expiry(self):
        """Once the lockout countdown expires, a user should be allowed fresh attempts and should not be locked out on the first failed attempt."""
        # 1. Execute 5 incorrect logins to lock the account
        for i in range(5):
            resp = self.client.post("/api/auth/login", json={
                "email": "patient@aurascan.ai",
                "password": "WrongPassword@123"
            })
            self.assertEqual(resp.status_code, 400)

        # 2. Verify account is locked
        resp_lockout = self.client.post("/api/auth/login", json={
            "email": "patient@aurascan.ai",
            "password": "WrongPassword@123"
        })
        self.assertIn("temporarily locked", resp_lockout.get_json()["error"])

        # 3. Manually edit database to set lockout_until to 20 minutes in the past
        import sqlite3
        import datetime
        past_time = (datetime.datetime.utcnow() - datetime.timedelta(minutes=20)).isoformat() + "Z"
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("UPDATE users SET lockout_until = ? WHERE email = ?", (past_time, "patient@aurascan.ai"))
            conn.commit()
        finally:
            conn.close()

        # 4. Attempt login with INCORRECT password. It should not be locked out; instead it should return a fresh attempt message (Attempt 1/5)
        resp_fail_after_expiry = self.client.post("/api/auth/login", json={
            "email": "patient@aurascan.ai",
            "password": "WrongPassword@1234"
        })
        self.assertEqual(resp_fail_after_expiry.status_code, 400)
        self.assertIn("Attempt 1/5", resp_fail_after_expiry.get_json()["error"])

        # 5. Attempt login with CORRECT password. It should succeed!
        resp_success = self.client.post("/api/auth/login", json={
            "email": "patient@aurascan.ai",
            "password": self.patient_pass
        })
        self.assertEqual(resp_success.status_code, 200)
        self.assertIn("access_token", resp_success.get_json())

    def test_legacy_pbkdf2_login(self):
        """Verifies that a legacy user with PBKDF2 hash can successfully authenticate via API endpoint."""
        import hashlib
        import secrets

        # 1. Generate PBKDF2 hash for legacy user
        pwd = "LegacyPassword@123"
        salt = secrets.token_bytes(16)
        key = hashlib.pbkdf2_hmac("sha256", pwd.encode("utf-8"), salt, 600000)
        legacy_hash = f"pbkdf2_sha256$600000${salt.hex()}${key.hex()}"

        # 2. Insert into repository
        legacy_user = User(
            id=None,
            uuid="legacy-doctor-uuid",
            email="legacy_doc@aurascan.ai",
            password_hash=legacy_hash,
            full_name="Legacy Doctor",
            role=Role.DOCTOR,
            is_verified=True,
            is_active=True
        )
        self.repo.create_user(legacy_user)

        # 3. Authenticate via login endpoint
        resp = self.client.post("/api/auth/login", json={
            "email": "legacy_doc@aurascan.ai",
            "password": pwd
        })
        self.assertEqual(resp.status_code, 200)
        self.assertIn("access_token", resp.get_json())
        self.assertFalse(resp.get_json()["requires_2fa"])


if __name__ == "__main__":
    unittest.main()
