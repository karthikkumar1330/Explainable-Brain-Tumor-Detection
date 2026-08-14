import unittest
import os
import tempfile
import json
import time
from dashboard.infrastructure.web_server import create_app
from security.domain.entities import Role, User
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.password import PasswordHasher
from security.application.tfa_service import TFAService


class TestSessionLoggingAndRevocation(unittest.TestCase):
    """Integration test suite to verify session logging and session management."""

    def setUp(self):
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        self.app = create_app(db_path=self.db_path)
        self.app.config["TESTING"] = True
        self.app.config["DISABLE_CSRF"] = True
        self.client = self.app.test_client()

        # Initialize tables
        self.repo = SQLiteUserRepository(db_path=self.db_path)
        self.repo.initialize_security_tables()

        # Create test doctor user
        self.password = "DoctorSecure@123"
        self.user = User(
            id=None,
            uuid="test-doctor-uuid",
            email="doctor@aurascan.ai",
            password_hash=PasswordHasher.hash_password(self.password),
            full_name="Doctor House",
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

    def test_user_agent_parser(self):
        """Verify TFAService.parse_user_agent correctly extracts browser and device info."""
        chrome_desktop = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        safari_iphone = "Mozilla/5.0 (iPhone; CPU iPhone OS 16_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Mobile/15E148 Safari/604.1"
        firefox_android = "Mozilla/5.0 (Android 13; Mobile; rv:109.0) Gecko/114.0 Firefox/114.0"

        browser, device = TFAService.parse_user_agent(chrome_desktop)
        self.assertEqual(browser, "Google Chrome")
        self.assertEqual(device, "Desktop")

        browser, device = TFAService.parse_user_agent(safari_iphone)
        self.assertEqual(browser, "Safari")
        self.assertEqual(device, "Mobile (iPhone)")

        browser, device = TFAService.parse_user_agent(firefox_android)
        self.assertEqual(browser, "Mozilla Firefox")
        self.assertEqual(device, "Mobile (Android)")

    def test_ip_geolocation_fallback(self):
        """Verify geolocator correctly handles local networks and fallbacks gracefully."""
        self.assertEqual(TFAService.get_location_from_ip("127.0.0.1"), "Localhost")
        self.assertEqual(TFAService.get_location_from_ip("192.168.1.1"), "Local Network")
        self.assertEqual(TFAService.get_location_from_ip("10.0.0.1"), "Local Network")
        self.assertTrue(len(TFAService.get_location_from_ip("8.8.8.8")) > 0)

    def test_login_and_logout_logging(self):
        """Verify that login (success/failure) and logout operations are logged with session details."""
        user_agent = "Mozilla/5.0 Chrome/115.0 Desktop"
        
        # 1. Failed login logging
        resp_fail = self.client.post("/api/auth/login", json={
            "email": "doctor@aurascan.ai",
            "password": "WrongPassword"
        }, headers={"User-Agent": user_agent})
        self.assertEqual(resp_fail.status_code, 400)

        # Assert failure log exists
        logs = self.repo.get_security_audit_logs(limit=5)
        fail_log = next(log for log in logs if log["event_type"] == "LOGIN_FAILURE")
        self.assertEqual(fail_log["status"], "FAILURE")
        details = json.loads(fail_log["details"])
        self.assertEqual(details["browser"], "Google Chrome")
        self.assertEqual(details["device"], "Desktop")
        self.assertIn("Invalid credentials", details["reason"])

        # 2. Successful login logging
        resp_success = self.client.post("/api/auth/login", json={
            "email": "doctor@aurascan.ai",
            "password": self.password
        }, headers={"User-Agent": user_agent})
        self.assertEqual(resp_success.status_code, 200)

        # Assert success log exists
        logs = self.repo.get_security_audit_logs(limit=5)
        success_log = next(log for log in logs if log["event_type"] == "LOGIN_SUCCESS")
        self.assertEqual(success_log["status"], "SUCCESS")
        details = json.loads(success_log["details"])
        self.assertEqual(details["browser"], "Google Chrome")
        self.assertEqual(details["device"], "Desktop")
        self.assertIsNone(details["logout_time"])

        # 3. GET sessions list
        resp_sessions = self.client.get("/api/auth/profile/sessions")
        self.assertEqual(resp_sessions.status_code, 200)
        sessions = resp_sessions.get_json()["sessions"]
        self.assertTrue(any(s["event_type"] == "LOGIN_SUCCESS" for s in sessions))
        self.assertTrue(any(s["event_type"] == "LOGIN_FAILURE" for s in sessions))

        # 4. Logout creates a separate LOGOUT event and does not mutate LOGIN_SUCCESS row
        resp_logout = self.client.post("/api/auth/logout")
        self.assertEqual(resp_logout.status_code, 200)

        # Assert LOGIN_SUCCESS details remain unchanged (append-only)
        logs = self.repo.get_security_audit_logs(limit=10)
        success_log = next(log for log in logs if log["event_type"] == "LOGIN_SUCCESS")
        details_success = json.loads(success_log["details"])
        self.assertIsNone(details_success["logout_time"])

        # Assert a separate LOGOUT event exists
        def has_jti(details_str):
            if not details_str:
                return False
            try:
                d = json.loads(details_str)
                return isinstance(d, dict) and "jti" in d
            except Exception:
                return False
        logout_log = next(log for log in logs if log["event_type"] == "LOGOUT" and has_jti(log["details"]))
        self.assertIsNotNone(logout_log)

        # 5. Log in again to verify GET sessions endpoint returns dynamically resolved logout_time for the previous session
        resp_login_again = self.client.post("/api/auth/login", json={
            "email": "doctor@aurascan.ai",
            "password": self.password
        }, headers={"User-Agent": user_agent})
        self.assertEqual(resp_login_again.status_code, 200)

        resp_sessions = self.client.get("/api/auth/profile/sessions")
        self.assertEqual(resp_sessions.status_code, 200)
        sessions = resp_sessions.get_json()["sessions"]

        # There should be two LOGIN_SUCCESS entries now.
        # The older one should have logout_time populated, and the new one should have logout_time None!
        login_success_sessions = [s for s in sessions if s["event_type"] == "LOGIN_SUCCESS"]
        self.assertEqual(len(login_success_sessions), 2)

        self.assertIsNone(login_success_sessions[0]["logout_time"])
        self.assertIsNotNone(login_success_sessions[1]["logout_time"])

    def test_logout_other_devices_revocation(self):
        """Verify that revoking all other sessions updates database logout times and revokes JTIs."""
        user_agent = "Mozilla/5.0 Chrome/115.0 Desktop"

        # Log in on device 1
        resp_dev1 = self.client.post("/api/auth/login", json={
            "email": "doctor@aurascan.ai",
            "password": self.password
        }, headers={"User-Agent": user_agent})
        self.assertEqual(resp_dev1.status_code, 200)
        token_dev1 = resp_dev1.get_json()["access_token"]

        # Sleep to separate timestamps
        time.sleep(1.1)

        # Log in on device 2
        resp_dev2 = self.client.post("/api/auth/login", json={
            "email": "doctor@aurascan.ai",
            "password": self.password
        }, headers={"User-Agent": user_agent})
        self.assertEqual(resp_dev2.status_code, 200)
        token_dev2 = resp_dev2.get_json()["access_token"]

        # Call logout other devices from device 2 session
        # Clear cookies in client or set Authorization header manually
        resp_logout_others = self.client.post(
            "/api/auth/profile/logout-other-devices",
            headers={"Authorization": f"Bearer {token_dev2}"}
        )
        self.assertEqual(resp_logout_others.status_code, 200)

        # Device 1 token should now be rejected (revoked)
        resp_dev1_protected = self.client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {token_dev1}"}
        )
        self.assertEqual(resp_dev1_protected.status_code, 200)
        self.assertFalse(resp_dev1_protected.get_json()["authenticated"])

        # Device 2 token should still work (since it's not revoked)
        # Note: the endpoint issues a fresh token in return, but the new token is in new_data
        new_token_dev2 = resp_logout_others.get_json()["access_token"]
        resp_dev2_protected = self.client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {new_token_dev2}"}
        )
        self.assertTrue(resp_dev2_protected.get_json()["authenticated"])


if __name__ == "__main__":
    unittest.main()
