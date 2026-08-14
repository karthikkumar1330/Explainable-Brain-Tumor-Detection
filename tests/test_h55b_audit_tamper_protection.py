import unittest
import os
import tempfile
import sqlite3
import json
import datetime
from dashboard.infrastructure.web_server import create_app
from security.domain.entities import Role, User, SecurityAuditLog
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.password import PasswordHasher

class TestH55BAuditTamperProtection(unittest.TestCase):
    """Focused tests for Phase H5.5-B: SQLite Audit Log Tamper Protection."""

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

    def test_01_insert_and_select_succeed(self):
        """Verify that legitimate INSERT and SELECT operations function correctly."""
        now = datetime.datetime.utcnow().isoformat()
        log_event = SecurityAuditLog(
            id=None, timestamp=now, event_type="TEST_INSERT", user_id=self.user.id, email=self.user.email,
            ip_address="127.0.0.1", status="SUCCESS", details="Test insertion event", user_agent="System"
        )
        self.repo.log_security_event(log_event)

        logs = self.repo.get_security_audit_logs(limit=10)
        self.assertTrue(len(logs) > 0)
        inserted = next(l for l in logs if l["event_type"] == "TEST_INSERT")
        self.assertEqual(inserted["details"], "Test insertion event")
        self.assertEqual(inserted["email"], self.user.email)

    def test_02_trigger_metadata_exists(self):
        """Verify that the no-update and no-delete triggers are registered in SQLite metadata."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name, tbl_name FROM sqlite_master WHERE type = 'trigger';")
            triggers = cursor.fetchall()

            trigger_names = [t[0] for t in triggers]
            self.assertIn("security_audit_logs_no_update", trigger_names)
            self.assertIn("security_audit_logs_no_delete", trigger_names)
        finally:
            conn.close()

    def test_03_update_is_blocked(self):
        """Verify that direct SQL updates on security_audit_logs are blocked and raise IntegrityError."""
        # Insert a log row
        now = datetime.datetime.utcnow().isoformat()
        self.repo.log_security_event(SecurityAuditLog(
            id=None, timestamp=now, event_type="TEST_UPDATE_BLOCK", user_id=self.user.id, email=self.user.email,
            ip_address="127.0.0.1", status="SUCCESS", details="Original details", user_agent="System"
        ))

        logs = self.repo.get_security_audit_logs(limit=5)
        log_row = next(l for l in logs if l["event_type"] == "TEST_UPDATE_BLOCK")

        # Attempt UPDATE on security_audit_logs
        conn = sqlite3.connect(self.db_path)
        try:
            with self.assertRaises(sqlite3.IntegrityError) as context:
                conn.execute(
                    "UPDATE security_audit_logs SET details = 'tampered' WHERE id = ?;",
                    (log_row["id"],)
                )
            self.assertIn("Updates to security_audit_logs are not allowed.", str(context.exception))

            # Verify the details remained unchanged
            cursor = conn.cursor()
            cursor.execute("SELECT details FROM security_audit_logs WHERE id = ?;", (log_row["id"],))
            db_row = cursor.fetchone()
            self.assertEqual(db_row[0], "Original details")
        finally:
            conn.close()

    def test_04_delete_is_blocked(self):
        """Verify that direct SQL deletions on security_audit_logs are blocked and raise IntegrityError."""
        # Insert a log row
        now = datetime.datetime.utcnow().isoformat()
        self.repo.log_security_event(SecurityAuditLog(
            id=None, timestamp=now, event_type="TEST_DELETE_BLOCK", user_id=self.user.id, email=self.user.email,
            ip_address="127.0.0.1", status="SUCCESS", details="Original details", user_agent="System"
        ))

        logs = self.repo.get_security_audit_logs(limit=5)
        log_row = next(l for l in logs if l["event_type"] == "TEST_DELETE_BLOCK")

        # Attempt DELETE on security_audit_logs
        conn = sqlite3.connect(self.db_path)
        try:
            with self.assertRaises(sqlite3.IntegrityError) as context:
                conn.execute(
                    "DELETE FROM security_audit_logs WHERE id = ?;",
                    (log_row["id"],)
                )
            self.assertIn("Deletions from security_audit_logs are not allowed.", str(context.exception))

            # Verify the row still exists in the database
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM security_audit_logs WHERE id = ?;", (log_row["id"],))
            count = cursor.fetchone()[0]
            self.assertEqual(count, 1)
        finally:
            conn.close()

    def test_05_initialization_idempotency(self):
        """Verify that re-running schema initialization is completely idempotent and does not create duplicate trigger errors."""
        # Call initialization multiple times
        self.repo.initialize_security_tables()
        self.repo.initialize_security_tables()

        # Check metadata trigger count is exactly 1 per type
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type = 'trigger' AND name LIKE 'security_audit_logs_no_%';")
            triggers = [t[0] for t in cursor.fetchall()]
            self.assertEqual(triggers.count("security_audit_logs_no_update"), 1)
            self.assertEqual(triggers.count("security_audit_logs_no_delete"), 1)
        finally:
            conn.close()

    def test_06_regressions(self):
        # Verification of dynamic session append-only integration with triggers active
        user_agent = "Mozilla/5.0 Chrome/115.0 Desktop"

        # Login success -> creates LOGIN_SUCCESS audit row
        resp_login = self.client.post("/api/auth/login", json={
            "email": "doctor@aurascan.ai",
            "password": self.password
        }, headers={"User-Agent": user_agent})
        self.assertEqual(resp_login.status_code, 200)

        # Logout -> creates LOGOUT event, doesn't update LOGIN_SUCCESS (which would fail with triggers active!)
        resp_logout = self.client.post("/api/auth/logout")
        self.assertEqual(resp_logout.status_code, 200)

        # Retrieve logs and assert LOGIN_SUCCESS has no logout_time, and a separate LOGOUT event exists
        logs = self.repo.get_security_audit_logs(limit=10)
        success_log = next(log for log in logs if log["event_type"] == "LOGIN_SUCCESS")
        self.assertIsNone(json.loads(success_log["details"])["logout_time"])

        logout_log = next(log for log in logs if log["event_type"] == "LOGOUT" and "jti" in log["details"])
        self.assertIsNotNone(logout_log)

if __name__ == "__main__":
    unittest.main()
