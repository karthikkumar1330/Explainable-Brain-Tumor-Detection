import os
import unittest
import tempfile
import sqlite3
import datetime
import json
from dashboard.infrastructure.web_server import create_app
from security.domain.entities import Role, User
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.jwt_service import JWTService
from security.infrastructure.password import PasswordHasher
from persistence.infrastructure.repository import SQLitePersistenceRepository
from clinical_reporting.application.notification_service import NotificationService

class TestH61AdminNotifications(unittest.TestCase):
    """Focused integration and security tests for Phase H6.1: Admin Notification Center."""

    def setUp(self):
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        self.app = create_app(db_path=self.db_path)
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

        # Initialize databases
        self.persistence_repo = SQLitePersistenceRepository(db_path=self.db_path)
        self.persistence_repo.initialize_db()

        self.user_repo = SQLiteUserRepository(db_path=self.db_path)
        self.user_repo.initialize_security_tables()

        # Setup roles/accounts
        pass_hash = PasswordHasher.hash_password("AuraScan@2026")

        self.patient = User(
            id=None, uuid="pat-111", email="patient@aurascan.ai",
            password_hash=pass_hash, full_name="John Patient", role=Role.PATIENT,
            is_verified=True, is_active=True
        )
        self.user_repo.create_user(self.patient)

        self.doctor = User(
            id=None, uuid="doc-222", email="doctor@aurascan.ai",
            password_hash=pass_hash, full_name="Dr. Smith", role=Role.DOCTOR,
            is_verified=True, is_active=True
        )
        self.user_repo.create_user(self.doctor)

        # Retrieve bootstrapped Admin user
        self.admin = self.user_repo.get_by_email("admin@aurascan.ai")

        # Generate tokens
        self.jwt_svc = JWTService()
        self.admin_token = self.jwt_svc.create_access_token(
            user_uuid=self.admin.uuid, user_id=self.admin.id, email=self.admin.email, role=self.admin.role
        )
        self.patient_token = self.jwt_svc.create_access_token(
            user_uuid=self.patient.uuid, user_id=self.patient.id, email=self.patient.email, role=self.patient.role
        )
        self.doctor_token = self.jwt_svc.create_access_token(
            user_uuid=self.doctor.uuid, user_id=self.doctor.id, email=self.doctor.email, role=self.doctor.role
        )

        self.notif_svc = NotificationService(db_path=self.db_path)

    def tearDown(self):
        os.close(self.db_fd)
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def test_01_admin_retrieves_own_notifications_and_unread_count(self):
        """Verify that an authenticated Admin can retrieve their own notifications and unread count."""
        # Create some notifications for Admin
        self.notif_svc.create_notification(self.admin.id, "SECURITY_WARNING", "Admin Notice 1", "Message 1")
        self.notif_svc.create_notification(self.admin.id, "SYSTEM_ALERT", "Admin Notice 2", "Message 2")

        # Check list API
        res = self.client.get(
            "/api/notifications",
            headers={"Authorization": f"Bearer {self.admin_token}"}
        )
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(data["total_count"], 2)
        self.assertEqual(len(data["items"]), 2)

        # Check unread count API
        res_count = self.client.get(
            "/api/notifications/unread-count",
            headers={"Authorization": f"Bearer {self.admin_token}"}
        )
        self.assertEqual(res_count.status_code, 200)
        data_count = json.loads(res_count.data)
        self.assertEqual(data_count["unread_count"], 2)

    def test_02_admin_marks_single_and_bulk_read(self):
        """Verify that an Admin can mark a specific notification read, and mark all read."""
        id1 = self.notif_svc.create_notification(self.admin.id, "SECURITY_WARNING", "Notice 1", "Msg 1")
        id2 = self.notif_svc.create_notification(self.admin.id, "SYSTEM_ALERT", "Notice 2", "Msg 2")

        # Mark single read
        res_single = self.client.patch(
            f"/api/notifications/{id1}/read",
            headers={"Authorization": f"Bearer {self.admin_token}"}
        )
        self.assertEqual(res_single.status_code, 200)

        # Verify unread count is 1
        res_count = self.client.get(
            "/api/notifications/unread-count",
            headers={"Authorization": f"Bearer {self.admin_token}"}
        )
        self.assertEqual(json.loads(res_count.data)["unread_count"], 1)

        # Mark all read
        res_all = self.client.patch(
            "/api/notifications/read-all",
            headers={"Authorization": f"Bearer {self.admin_token}"}
        )
        self.assertEqual(res_all.status_code, 200)

        # Verify unread count is 0
        res_count_2 = self.client.get(
            "/api/notifications/unread-count",
            headers={"Authorization": f"Bearer {self.admin_token}"}
        )
        self.assertEqual(json.loads(res_count_2.data)["unread_count"], 0)

    def test_03_patient_cannot_access_admin_notifications_idor(self):
        """Verify that a Patient user cannot view, read-mark, or delete an Admin's notification (IDOR protection)."""
        admin_notif_id = self.notif_svc.create_notification(self.admin.id, "SECURITY_WARNING", "Secret Admin Alert", "Confidential")

        # Patient tries to mark read -> must return 404 (Not Found or Unauthorized)
        res_read = self.client.patch(
            f"/api/notifications/{admin_notif_id}/read",
            headers={"Authorization": f"Bearer {self.patient_token}"}
        )
        self.assertEqual(res_read.status_code, 404)

        # Patient tries to delete -> must return 404
        res_del = self.client.delete(
            f"/api/notifications/{admin_notif_id}",
            headers={"Authorization": f"Bearer {self.patient_token}"}
        )
        self.assertEqual(res_del.status_code, 404)

    def test_04_doctor_cannot_access_admin_notifications_idor(self):
        """Verify that a Doctor user cannot view, read-mark, or delete an Admin's notification (IDOR protection)."""
        admin_notif_id = self.notif_svc.create_notification(self.admin.id, "SYSTEM_ALERT", "Secret System Warning", "Confidential")

        # Doctor tries to mark read -> must return 404
        res_read = self.client.patch(
            f"/api/notifications/{admin_notif_id}/read",
            headers={"Authorization": f"Bearer {self.doctor_token}"}
        )
        self.assertEqual(res_read.status_code, 404)

        # Doctor tries to delete -> must return 404
        res_del = self.client.delete(
            f"/api/notifications/{admin_notif_id}",
            headers={"Authorization": f"Bearer {self.doctor_token}"}
        )
        self.assertEqual(res_del.status_code, 404)

    def test_05_unauthenticated_requests_are_rejected(self):
        """Verify that unauthenticated notification queries are rejected with 401."""
        res_list = self.client.get("/api/notifications")
        self.assertEqual(res_list.status_code, 401)

        res_count = self.client.get("/api/notifications/unread-count")
        self.assertEqual(res_count.status_code, 401)

        res_read = self.client.patch("/api/notifications/1/read")
        self.assertEqual(res_read.status_code, 401)

        res_del = self.client.delete("/api/notifications/1")
        self.assertEqual(res_del.status_code, 401)

    def test_06_error_disclosure_prevention(self):
        """Verify that internal error details/tracebacks are not returned in error responses (F1.3 compliance)."""
        # Request with an invalid page number parameter (e.g. page = 0)
        res = self.client.get(
            "/api/notifications?page=0",
            headers={"Authorization": f"Bearer {self.admin_token}"}
        )
        # Should return a client/bad request error code (400)
        self.assertEqual(res.status_code, 400)

        # Verify body doesn't contain tracebacks or raw exception formats
        body = res.data.decode("utf-8")
        self.assertNotIn("Traceback", body)
        self.assertNotIn("sqlite3.OperationalError", body)
        self.assertNotIn("Exception", body)

if __name__ == "__main__":
    unittest.main()
