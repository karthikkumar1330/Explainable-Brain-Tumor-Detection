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


if __name__ == "__main__":
    unittest.main()
