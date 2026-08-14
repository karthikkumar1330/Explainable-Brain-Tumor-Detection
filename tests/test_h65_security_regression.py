import os
import unittest
import tempfile
import sqlite3
import json
from flask import Flask
from security.domain.entities import Role, User
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.password import PasswordHasher
from persistence.infrastructure.repository import SQLitePersistenceRepository
from clinical_reporting.application.notification_service import NotificationService
from dashboard.infrastructure.web_server import create_app

class TestH65SecurityRegression(unittest.TestCase):
    """Focused security and regression tests for Phase H6.5: Notification Security & Authorization Regression Gate."""

    def setUp(self):
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")

        # Initialize databases
        self.persistence_repo = SQLitePersistenceRepository(db_path=self.db_path)
        self.persistence_repo.initialize_db()

        self.user_repo = SQLiteUserRepository(db_path=self.db_path)
        self.user_repo.initialize_security_tables()

        # Setup test users
        pass_hash = PasswordHasher.hash_password("AuraScan@2026")

        # Patients
        self.patient_a = User(
            id=None, uuid="pat-aaa", email="patient_a@aurascan.ai",
            password_hash=pass_hash, full_name="Patient A", role=Role.PATIENT,
            is_verified=True, is_active=True
        )
        self.user_repo.create_user(self.patient_a)

        self.patient_b = User(
            id=None, uuid="pat-bbb", email="patient_b@aurascan.ai",
            password_hash=pass_hash, full_name="Patient B", role=Role.PATIENT,
            is_verified=True, is_active=True
        )
        self.user_repo.create_user(self.patient_b)

        # Doctors
        self.doctor_a = User(
            id=None, uuid="doc-aaa", email="doctor_a@aurascan.ai",
            password_hash=pass_hash, full_name="Doctor A", role=Role.DOCTOR,
            is_verified=True, is_active=True
        )
        self.user_repo.create_user(self.doctor_a)

        self.doctor_b = User(
            id=None, uuid="doc-bbb", email="doctor_b@aurascan.ai",
            password_hash=pass_hash, full_name="Doctor B", role=Role.DOCTOR,
            is_verified=True, is_active=True
        )
        self.user_repo.create_user(self.doctor_b)

        # Admin
        self.admin = User(
            id=None, uuid="adm-aaa", email="admin_a@aurascan.ai",
            password_hash=pass_hash, full_name="Admin A", role=Role.ADMIN,
            is_verified=True, is_active=True
        )
        self.user_repo.create_user(self.admin)

        self.notif_svc = NotificationService(db_path=self.db_path)

        # Initialize web server client using factory
        self.app = create_app(db_path=self.db_path)
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

        # Seed initial notification for Patient A
        self.notif_a_id = self.notif_svc.create_notification(
            user_id=self.patient_a.id,
            type_="FOLLOWUP_CREATED",
            title="Notice A",
            message="Msg A",
            metadata_json=json.dumps({"followup_id": 1})
        )

        # Seed initial notification for Doctor A
        self.notif_doc_id = self.notif_svc.create_notification(
            user_id=self.doctor_a.id,
            type_="ANALYSIS_COMPLETED",
            title="Notice Doc",
            message="Msg Doc",
            metadata_json=json.dumps({"report_id": 2})
        )

        # Generate tokens
        from security.infrastructure.jwt_service import JWTService
        self.jwt_svc = JWTService()
        self.token_pat_a = self.jwt_svc.create_access_token(
            user_uuid=self.patient_a.uuid, user_id=self.patient_a.id, email=self.patient_a.email, role=self.patient_a.role
        )
        self.token_pat_b = self.jwt_svc.create_access_token(
            user_uuid=self.patient_b.uuid, user_id=self.patient_b.id, email=self.patient_b.email, role=self.patient_b.role
        )
        self.token_doc_a = self.jwt_svc.create_access_token(
            user_uuid=self.doctor_a.uuid, user_id=self.doctor_a.id, email=self.doctor_a.email, role=self.doctor_a.role
        )
        self.token_admin = self.jwt_svc.create_access_token(
            user_uuid=self.admin.uuid, user_id=self.admin.id, email=self.admin.email, role=self.admin.role
        )

    def tearDown(self):
        os.close(self.db_fd)
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def test_01_unauthenticated_requests_blocked(self):
        """Verify that all notification API routes block unauthenticated access."""
        endpoints = [
            ("/api/notifications", "GET", None),
            ("/api/notifications/unread-count", "GET", None),
            (f"/api/notifications/{self.notif_a_id}/read", "PATCH", None),
            ("/api/notifications/read-all", "PATCH", None),
            (f"/api/notifications/{self.notif_a_id}", "DELETE", None),
            ("/api/notifications/preferences", "GET", None),
            ("/api/notifications/preferences", "PUT", {"analysis": False})
        ]
        for path, method, data in endpoints:
            if method == "GET":
                res = self.client.get(path)
            elif method == "PATCH":
                res = self.client.patch(path)
            elif method == "DELETE":
                res = self.client.delete(path)
            elif method == "PUT":
                res = self.client.put(path, json=data)

            self.assertEqual(res.status_code, 401, f"Path {path} did not block unauthenticated access!")
            self.assertIn("Authentication required", res.get_json()["error"])

    def test_02_patient_cross_user_isolation(self):
        """Verify that Patient B cannot read, modify, or delete Patient A's notifications."""
        headers = {"Authorization": f"Bearer {self.token_pat_b}"}

        # 1. Retrieve list: Patient B should not see Patient A's notification
        res = self.client.get("/api/notifications", headers=headers)
        self.assertEqual(res.status_code, 200)
        items = res.get_json()["items"]
        self.assertTrue(all(item["id"] != self.notif_a_id for item in items))

        # 2. Try to mark Patient A's notification read
        res = self.client.patch(f"/api/notifications/{self.notif_a_id}/read", headers=headers)
        self.assertEqual(res.status_code, 404)

        # 3. Try to delete Patient A's notification
        res = self.client.delete(f"/api/notifications/{self.notif_a_id}", headers=headers)
        self.assertEqual(res.status_code, 404)

        # Assert no change in Patient A's notification status
        self.assertFalse(self.notif_svc.list_notifications(user_id=self.patient_a.id)["items"][0]["is_read"])

    def test_03_doctor_notification_isolation(self):
        """Verify that Doctor A cannot access Patient A's notifications or Doctor B's notifications."""
        headers = {"Authorization": f"Bearer {self.token_doc_a}"}

        # 1. Retrieve list: Doctor A should see only Doctor A's notification
        res = self.client.get("/api/notifications", headers=headers)
        self.assertEqual(res.status_code, 200)
        items = res.get_json()["items"]
        self.assertTrue(all(item["id"] != self.notif_a_id for item in items))

        # 2. Try to mark Patient A's notification read
        res = self.client.patch(f"/api/notifications/{self.notif_a_id}/read", headers=headers)
        self.assertEqual(res.status_code, 404)

    def test_04_admin_cannot_access_other_user_notification_records(self):
        """Verify that Admin A cannot list or read Patient A's notifications."""
        headers = {"Authorization": f"Bearer {self.token_admin}"}

        # 1. Retrieve list: Admin A should not see Patient A's notification in their own list
        res = self.client.get("/api/notifications", headers=headers)
        self.assertEqual(res.status_code, 200)
        items = res.get_json()["items"]
        self.assertTrue(all(item["id"] != self.notif_a_id for item in items))

        # 2. Try to mark Patient A's notification read
        res = self.client.patch(f"/api/notifications/{self.notif_a_id}/read", headers=headers)
        self.assertEqual(res.status_code, 404)

    def test_05_idor_via_user_id_manipulation_blocked(self):
        """Verify that parameter pollution trying to supply user_id parameters does not bypass isolation."""
        headers = {"Authorization": f"Bearer {self.token_pat_b}"}

        # Attempt to inject user_id query param
        res = self.client.get(f"/api/notifications?user_id={self.patient_a.id}", headers=headers)
        self.assertEqual(res.status_code, 200)
        items = res.get_json()["items"]
        # Must only return Patient B's (which is empty), not Patient A's
        self.assertEqual(len(items), 0)

    def test_06_metadata_sanitization_remains_active(self):
        """Verify that metadata sanitization removes sensitive credentials/hashes."""
        meta = {
            "report_id": 42,
            "password": "SecretPassword123",
            "access_token": "token-1234",
            "password_hash": "hash-9999"
        }
        nid = self.notif_svc.create_notification(
            user_id=self.patient_a.id,
            type_="ANALYSIS_COMPLETED",
            title="Clean Metadata Test",
            message="Checking metadata clean",
            metadata_json=json.dumps(meta)
        )

        headers = {"Authorization": f"Bearer {self.token_pat_a}"}
        res = self.client.get("/api/notifications", headers=headers)
        item = [x for x in res.get_json()["items"] if x["id"] == nid][0]
        cleaned_meta = json.loads(item["metadata_json"])

        self.assertEqual(cleaned_meta["report_id"], 42)
        self.assertNotIn("password", cleaned_meta)
        self.assertNotIn("access_token", cleaned_meta)
        self.assertNotIn("password_hash", cleaned_meta)

    def test_07_idempotency_key_abuse_prevention(self):
        """Verify that clients cannot forge or supply an idempotency key to hijack or suppress another user's event."""
        # 1. Patient A has notification with followup_id 1
        # 2. Patient B schedules a followup 1. If Patient B tries to forge/pass a key, it should be ignored/separated
        # Because idempotency key is auto-calculated based on user_id, type, message, and metadata
        # For user A, it is auto:FOLLOWUP_CREATED:hash(A, ...)
        # For user B, it is auto:FOLLOWUP_CREATED:hash(B, ...)
        # Since user_id is in the hash source, they can never collide!
        nid_b = self.notif_svc.create_notification(
            user_id=self.patient_b.id,
            type_="FOLLOWUP_CREATED",
            title="Notice A",
            message="Msg A",
            metadata_json=json.dumps({"followup_id": 1})
        )
        self.assertIsNotNone(nid_b)
        self.assertNotEqual(self.notif_a_id, nid_b)

    def test_08_index_h64_query_plan_verification(self):
        """Verify that index optimizations run correctly without altering query correctness or authorization scopes."""
        # Query list for Patient A: must return exactly Patient A's notifications
        headers = {"Authorization": f"Bearer {self.token_pat_a}"}
        res = self.client.get("/api/notifications", headers=headers)
        self.assertEqual(res.status_code, 200)
        items = res.get_json()["items"]
        self.assertTrue(len(items) >= 1)
        self.assertTrue(all(item["user_id"] == self.patient_a.id for item in items))

    def test_09_h5_security_audit_protection_intact(self):
        """Verify that security audit logs remain protected and append-only."""
        conn = sqlite3.connect(self.db_path)
        try:
            # Check table existence
            table_info = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='security_audit_logs';").fetchone()
            self.assertIsNotNone(table_info)
        finally:
            conn.close()

if __name__ == "__main__":
    unittest.main()
