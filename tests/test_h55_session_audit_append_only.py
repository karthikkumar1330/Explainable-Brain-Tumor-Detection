import unittest
import os
import tempfile
import json
import time
import datetime
from dashboard.infrastructure.web_server import create_app
from security.domain.entities import Role, User, SecurityAuditLog
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.password import PasswordHasher

class TestH55SessionAuditAppendOnly(unittest.TestCase):
    """Focused tests for Phase H5.5-A: Convert Mutable Session Audit Updates to Append-Only Events."""

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

    def test_login_logout_and_revocation_are_append_only(self):
        user_agent = "Mozilla/5.0 Chrome/115.0 Desktop"

        # 1. Login success -> creates LOGIN_SUCCESS audit row
        resp_login = self.client.post("/api/auth/login", json={
            "email": "doctor@aurascan.ai",
            "password": self.password
        }, headers={"User-Agent": user_agent})
        self.assertEqual(resp_login.status_code, 200)

        # Retrieve the original LOGIN_SUCCESS log
        logs = self.repo.get_security_audit_logs(limit=10)
        login_log = next(log for log in logs if log["event_type"] == "LOGIN_SUCCESS")
        self.assertEqual(login_log["status"], "SUCCESS")
        details_before = json.loads(login_log["details"])
        self.assertIsNone(details_before["logout_time"])
        self.assertIsNotNone(details_before["jti"])
        login_jti = details_before["jti"]

        # 2. Logout -> creates a separate LOGOUT audit row, original LOGIN_SUCCESS is untouched
        resp_logout = self.client.post("/api/auth/logout")
        self.assertEqual(resp_logout.status_code, 200)

        logs_after_logout = self.repo.get_security_audit_logs(limit=10)

        # Verify original LOGIN_SUCCESS row is unchanged
        login_log_after = next(log for log in logs_after_logout if log["id"] == login_log["id"])
        details_after = json.loads(login_log_after["details"])
        self.assertIsNone(details_after["logout_time"])  # Unchanged!

        # Verify LOGOUT event was inserted with JTI
        logout_log = next(log for log in logs_after_logout if log["event_type"] == "LOGOUT" and login_jti in log["details"])
        self.assertIsNotNone(logout_log)

        # 3. Logout invalidates session correctly (authenticated request fails)
        resp_me = self.client.get("/api/auth/me")
        self.assertEqual(resp_me.status_code, 200)
        self.assertFalse(resp_me.get_json()["authenticated"])

        # 4. Revocation (logout-other-devices) -> creates separate SESSION_REVOKED audit row
        # Log in on device 1 (which will be revoked)
        resp_dev1 = self.client.post("/api/auth/login", json={
            "email": "doctor@aurascan.ai",
            "password": self.password
        }, headers={"User-Agent": "Device 1"})
        self.assertEqual(resp_dev1.status_code, 200)
        token_dev1 = resp_dev1.get_json()["access_token"]
        jti_dev1 = json.loads(next(log for log in self.repo.get_security_audit_logs(limit=5) if log["event_type"] == "LOGIN_SUCCESS")["details"])["jti"]

        # Log in on device 2 (current session)
        resp_dev2 = self.client.post("/api/auth/login", json={
            "email": "doctor@aurascan.ai",
            "password": self.password
        }, headers={"User-Agent": "Device 2"})
        self.assertEqual(resp_dev2.status_code, 200)
        token_dev2 = resp_dev2.get_json()["access_token"]

        # Call logout other devices from Device 2 session
        resp_revoke = self.client.post(
            "/api/auth/profile/logout-other-devices",
            headers={"Authorization": f"Bearer {token_dev2}"}
        )
        self.assertEqual(resp_revoke.status_code, 200)

        # Verify Device 1 session is invalidated (fails authentication)
        resp_dev1_me = self.client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {token_dev1}"}
        )
        self.assertEqual(resp_dev1_me.status_code, 200)
        self.assertFalse(resp_dev1_me.get_json()["authenticated"])

        # Verify Device 2 session remains active
        new_token_dev2 = resp_revoke.get_json()["access_token"]
        resp_dev2_me = self.client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {new_token_dev2}"}
        )
        self.assertEqual(resp_dev2_me.status_code, 200)
        self.assertTrue(resp_dev2_me.get_json()["authenticated"])

        # Verify audit logs for revocation:
        # A separate SESSION_REVOKED event was logged for Device 1 JTI
        logs_after_revocation = self.repo.get_security_audit_logs(limit=20)
        revoked_log = next(log for log in logs_after_revocation if log["event_type"] == "SESSION_REVOKED" and jti_dev1 in log["details"])
        self.assertIsNotNone(revoked_log)

        # Original LOGIN_SUCCESS row for Device 1 remains unchanged
        dev1_login_log = next(log for log in logs_after_revocation if log["event_type"] == "LOGIN_SUCCESS" and jti_dev1 in log["details"])
        dev1_details = json.loads(dev1_login_log["details"])
        self.assertIsNone(dev1_details["logout_time"]) # Unchanged!

    def test_no_sensitive_data_in_audit_details(self):
        user_agent = "Mozilla/5.0 Chrome/115.0 Desktop"
        resp_login = self.client.post("/api/auth/login", json={
            "email": "doctor@aurascan.ai",
            "password": self.password
        }, headers={"User-Agent": user_agent})
        self.assertEqual(resp_login.status_code, 200)

        token = resp_login.get_json()["access_token"]

        resp_logout = self.client.post("/api/auth/logout")
        self.assertEqual(resp_logout.status_code, 200)

        logs = self.repo.get_security_audit_logs(limit=50)
        for log in logs:
            details_str = log["details"] or ""
            # Verify no raw secrets (plaintext password or JWT token value) exist in details
            self.assertNotIn(self.password, details_str)
            self.assertNotIn(token, details_str)

    def test_repeated_logout_and_revocation_no_duplicates(self):
        user_agent = "Mozilla/5.0 Chrome/115.0 Desktop"
        # 1. Repeated logout
        resp_login = self.client.post("/api/auth/login", json={
            "email": "doctor@aurascan.ai",
            "password": self.password
        }, headers={"User-Agent": user_agent})
        self.assertEqual(resp_login.status_code, 200)

        jti = json.loads(next(log for log in self.repo.get_security_audit_logs(limit=5) if log["event_type"] == "LOGIN_SUCCESS")["details"])["jti"]

        now = datetime.datetime.utcnow().isoformat()
        self.repo.update_session_logout(jti, now)
        self.repo.update_session_logout(jti, now)

        logs = self.repo.get_security_audit_logs(limit=20)
        logout_logs = [log for log in logs if log["event_type"] == "LOGOUT" and jti in (log["details"] or "")]
        self.assertEqual(len(logout_logs), 1)

        # 2. Repeated revocation on a NEW active session
        resp_login_2 = self.client.post("/api/auth/login", json={
            "email": "doctor@aurascan.ai",
            "password": self.password
        }, headers={"User-Agent": user_agent})
        self.assertEqual(resp_login_2.status_code, 200)

        jti_2 = json.loads(next(log for log in self.repo.get_security_audit_logs(limit=5) if log["event_type"] == "LOGIN_SUCCESS")["details"])["jti"]

        self.repo.revoke_other_sessions(self.user.id, "some_current_jti", now)
        self.repo.revoke_other_sessions(self.user.id, "some_current_jti", now)

        logs_rev = self.repo.get_security_audit_logs(limit=20)
        rev_logs = [log for log in logs_rev if log["event_type"] == "SESSION_REVOKED" and jti_2 in (log["details"] or "")]
        self.assertEqual(len(rev_logs), 1)

    def test_100_event_limit_correctness(self):
        user_agent = "Mozilla/5.0 Chrome/115.0 Desktop"
        # 1. Login success -> creates LOGIN_SUCCESS audit row
        resp_login = self.client.post("/api/auth/login", json={
            "email": "doctor@aurascan.ai",
            "password": self.password
        }, headers={"User-Agent": user_agent})
        self.assertEqual(resp_login.status_code, 200)

        logs = self.repo.get_security_audit_logs(limit=5)
        jti = json.loads(next(log for log in logs if log["event_type"] == "LOGIN_SUCCESS")["details"])["jti"]

        # 2. Logout -> creates LOGOUT event
        resp_logout = self.client.post("/api/auth/logout")
        self.assertEqual(resp_logout.status_code, 200)

        # 3. Insert 105 unrelated audit events
        now = datetime.datetime.utcnow().isoformat()
        for i in range(105):
            self.repo.log_security_event(SecurityAuditLog(
                id=None,
                timestamp=now,
                event_type="UNRELATED_EVENT",
                user_id=self.user.id,
                email=self.user.email,
                ip_address="127.0.0.1",
                status="SUCCESS",
                details=f"Unrelated event {i}",
                user_agent="System"
            ))

        # 4. Log in again to query the sessions endpoint
        resp_login_2 = self.client.post("/api/auth/login", json={
            "email": "doctor@aurascan.ai",
            "password": self.password
        }, headers={"User-Agent": user_agent})
        self.assertEqual(resp_login_2.status_code, 200)

        resp_sessions = self.client.get("/api/auth/profile/sessions")
        self.assertEqual(resp_sessions.status_code, 200)
        sessions = resp_sessions.get_json()["sessions"]

        login_success_sessions = [s for s in sessions if s["event_type"] == "LOGIN_SUCCESS"]
        self.assertIsNone(login_success_sessions[0]["logout_time"])
        self.assertIsNotNone(login_success_sessions[1]["logout_time"])

    def test_h5_regressions(self):
        # H5.1 Audit Authorization Regression
        resp = self.client.get("/api/admin/audit-logs")
        self.assertIn(resp.status_code, [401, 302])

        # H5.2 Sanitization Regression
        now = datetime.datetime.utcnow().isoformat()
        self.repo.log_security_event(SecurityAuditLog(
            id=None, timestamp=now, event_type="CSRF_ATTEMPT", user_id=None, email=None,
            ip_address="127.0.0.1", status="FAILURE", details="CSRF token mismatch", user_agent="System"
        ))
        csrf_log = next(log for log in self.repo.get_security_audit_logs(limit=5) if log["event_type"] == "CSRF_ATTEMPT")
        self.assertEqual(csrf_log["details"], "CSRF token mismatch")

        # H5.3 MRI Upload Audit Regression
        self.repo.log_security_event(SecurityAuditLog(
            id=None, timestamp=now, event_type="MRI_UPLOAD", user_id=self.user.id, email=self.user.email,
            ip_address="127.0.0.1", status="SUCCESS", details="MRI uploaded successfully: mri.png", user_agent="System"
        ))
        mri_log = next(log for log in self.repo.get_security_audit_logs(limit=5) if log["event_type"] == "MRI_UPLOAD")
        self.assertEqual(mri_log["status"], "SUCCESS")

        # H5.4 Telemetry Regression
        self.repo.log_security_event(SecurityAuditLog(
            id=None, timestamp=now, event_type="REPORT_VIEWED", user_id=self.user.id, email=self.user.email,
            ip_address="127.0.0.1", status="SUCCESS", details="Report viewed", user_agent="System"
        ))
        telemetry_log = next(log for log in self.repo.get_security_audit_logs(limit=5) if log["event_type"] == "REPORT_VIEWED")
        self.assertEqual(telemetry_log["status"], "SUCCESS")

if __name__ == "__main__":
    unittest.main()
