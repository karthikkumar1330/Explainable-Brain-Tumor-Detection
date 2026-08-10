import os
import unittest
import tempfile
import sqlite3
import datetime
import json
from flask import json as flask_json
from dashboard.infrastructure.web_server import create_app
from security.domain.entities import Role, User
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.jwt_service import JWTService
from security.infrastructure.password import PasswordHasher
from persistence.infrastructure.repository import SQLitePersistenceRepository
from clinical_reporting.application.notification_service import NotificationService


class TestNotificationsIntegration(unittest.TestCase):
    """Phase G8.2 Automated unit, integration, and security compliance tests for notifications."""

    def setUp(self):
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        self.app = create_app(db_path=self.db_path)
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

        # Initialize persistence database schema
        self.persistence_repo = SQLitePersistenceRepository(db_path=self.db_path)
        self.persistence_repo.initialize_db()

        # Initialize security tables (creates users, notifications, preferences, etc.)
        self.user_repo = SQLiteUserRepository(db_path=self.db_path)
        self.user_repo.initialize_security_tables()

        # Create test users
        pass_hash = PasswordHasher.hash_password("AuraScan@2026")

        self.doctor = User(
            id=None,
            uuid="doc-111",
            email="doctor@aurascan.ai",
            password_hash=pass_hash,
            full_name="Dr. House",
            role=Role.DOCTOR,
            is_verified=True,
            is_active=True
        )
        self.user_repo.create_user(self.doctor)

        self.patient = User(
            id=None,
            uuid="pat-222",
            email="patient@aurascan.ai",
            password_hash=pass_hash,
            full_name="John Doe",
            role=Role.PATIENT,
            is_verified=True,
            is_active=True
        )
        self.user_repo.create_user(self.patient)

        self.other_patient = User(
            id=None,
            uuid="pat-333",
            email="other@aurascan.ai",
            password_hash=pass_hash,
            full_name="Jane Doe",
            role=Role.PATIENT,
            is_verified=True,
            is_active=True
        )
        self.user_repo.create_user(self.other_patient)

        self.admin = self.user_repo.get_by_email("admin@aurascan.ai")

        # Generate access tokens
        self.jwt_svc = JWTService()
        self.doctor_token = self.jwt_svc.create_access_token(
            user_uuid=self.doctor.uuid, user_id=self.doctor.id, email=self.doctor.email, role=self.doctor.role
        )
        self.patient_token = self.jwt_svc.create_access_token(
            user_uuid=self.patient.uuid, user_id=self.patient.id, email=self.patient.email, role=self.patient.role
        )
        self.other_patient_token = self.jwt_svc.create_access_token(
            user_uuid=self.other_patient.uuid, user_id=self.other_patient.id, email=self.other_patient.email, role=self.other_patient.role
        )
        self.admin_token = self.jwt_svc.create_access_token(
            user_uuid=self.admin.uuid, user_id=self.admin.id, email=self.admin.email, role=self.admin.role
        )

        self.notif_svc = NotificationService(db_path=self.db_path)

    def tearDown(self):
        os.close(self.db_fd)
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def test_notification_creation_retrieval_and_unread_count(self):
        """Test notification creation, listing/retrieval, and unread counting (items 1, 2, 3)."""
        # 1. notification creation
        notif_id = self.notif_svc.create_notification(
            user_id=self.patient.id,
            type_="ANALYSIS_COMPLETED",
            title="Analysis Done",
            message="Your MRI analysis is complete.",
            metadata_json=json.dumps({"scan_id": 42})
        )
        self.assertIsNotNone(notif_id)

        # 2. notification retrieval
        res = self.notif_svc.list_notifications(user_id=self.patient.id)
        self.assertEqual(res["total_count"], 1)
        self.assertEqual(res["items"][0]["id"], notif_id)
        self.assertEqual(res["items"][0]["title"], "Analysis Done")
        self.assertEqual(res["items"][0]["type"], "ANALYSIS_COMPLETED")
        self.assertEqual(json.loads(res["items"][0]["metadata_json"])["scan_id"], 42)

        # 3. unread count
        unread = self.notif_svc.get_unread_count(user_id=self.patient.id)
        self.assertEqual(unread, 1)

    def test_mark_read_and_mark_all_read(self):
        """Test marking a specific notification read and marking all read (items 4, 5)."""
        id1 = self.notif_svc.create_notification(self.patient.id, "ANALYSIS_COMPLETED", "Title 1", "Message 1")
        id2 = self.notif_svc.create_notification(self.patient.id, "REPORT_READY", "Title 2", "Message 2")

        # Verify initial unread count
        self.assertEqual(self.notif_svc.get_unread_count(self.patient.id), 2)

        # 4. mark notification read
        success = self.notif_svc.mark_notification_read(self.patient.id, id1)
        self.assertTrue(success)
        self.assertEqual(self.notif_svc.get_unread_count(self.patient.id), 1)

        # Verify only id2 is returned as unread
        res = self.notif_svc.list_notifications(self.patient.id, unread_only=True)
        self.assertEqual(res["total_count"], 1)
        self.assertEqual(res["items"][0]["id"], id2)

        # 5. mark all read
        self.notif_svc.mark_all_notifications_read(self.patient.id)
        self.assertEqual(self.notif_svc.get_unread_count(self.patient.id), 0)

    def test_ownership_and_idor_protection(self):
        """Test notification ownership constraints, IDOR protection (patient, doctor, admin) (items 6, 7, 8, 9)."""
        # Create private notification for patient
        pat_notif_id = self.notif_svc.create_notification(
            user_id=self.patient.id,
            type_="ANALYSIS_COMPLETED",
            title="Patient Notif",
            message="Private message"
        )

        # 6. notification ownership (Service layer checks)
        # Other user tries to mark read
        self.assertFalse(self.notif_svc.mark_notification_read(self.other_patient.id, pat_notif_id))
        # Other user tries to delete
        self.assertFalse(self.notif_svc.delete_notification(self.other_patient.id, pat_notif_id))

        # 7. patient IDOR protection (API layer)
        res = self.client.patch(
            f"/api/notifications/{pat_notif_id}/read",
            headers={"Authorization": f"Bearer {self.other_patient_token}"}
        )
        self.assertEqual(res.status_code, 404)

        res = self.client.delete(
            f"/api/notifications/{pat_notif_id}",
            headers={"Authorization": f"Bearer {self.other_patient_token}"}
        )
        self.assertEqual(res.status_code, 404)

        # 8. doctor IDOR protection (API layer)
        res = self.client.patch(
            f"/api/notifications/{pat_notif_id}/read",
            headers={"Authorization": f"Bearer {self.doctor_token}"}
        )
        self.assertEqual(res.status_code, 404)

        res = self.client.delete(
            f"/api/notifications/{pat_notif_id}",
            headers={"Authorization": f"Bearer {self.doctor_token}"}
        )
        self.assertEqual(res.status_code, 404)

        # 9. admin authorization (API layer / admin cannot read other's notifications either)
        res = self.client.patch(
            f"/api/notifications/{pat_notif_id}/read",
            headers={"Authorization": f"Bearer {self.admin_token}"}
        )
        self.assertEqual(res.status_code, 404)

        res = self.client.delete(
            f"/api/notifications/{pat_notif_id}",
            headers={"Authorization": f"Bearer {self.admin_token}"}
        )
        self.assertEqual(res.status_code, 404)

    def test_pagination_and_ordering(self):
        """Test pagination limits and descending order of notifications (items 10, 11)."""
        # Create 15 notifications with distinct sequential times
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            for i in range(1, 16):
                dt = (datetime.datetime.utcnow() - datetime.timedelta(minutes=20 - i)).isoformat()
                cursor.execute("""
                    INSERT INTO notifications (user_id, type, title, message, is_read, created_at)
                    VALUES (?, ?, ?, ?, 0, ?);
                """, (self.patient.id, "ANALYSIS_COMPLETED", f"Title {i}", f"Msg {i}", dt))
            conn.commit()
        finally:
            conn.close()

        # 10. pagination
        res = self.notif_svc.list_notifications(user_id=self.patient.id, page=1, page_size=10)
        self.assertEqual(res["total_count"], 15)
        self.assertEqual(res["page"], 1)
        self.assertEqual(res["page_size"], 10)
        self.assertEqual(res["total_pages"], 2)
        self.assertEqual(len(res["items"]), 10)

        # 11. ordering (latest first)
        self.assertEqual(res["items"][0]["title"], "Title 15")
        self.assertEqual(res["items"][9]["title"], "Title 6")

        # Page 2
        res2 = self.notif_svc.list_notifications(user_id=self.patient.id, page=2, page_size=10)
        self.assertEqual(len(res2["items"]), 5)
        self.assertEqual(res2["items"][0]["title"], "Title 5")
        self.assertEqual(res2["items"][4]["title"], "Title 1")

        # Parameter validation checks
        with self.assertRaises(ValueError):
            self.notif_svc.list_notifications(user_id=self.patient.id, page=0)
        with self.assertRaises(ValueError):
            self.notif_svc.list_notifications(user_id=self.patient.id, page_size=0)
        with self.assertRaises(ValueError):
            self.notif_svc.list_notifications(user_id=self.patient.id, page_size=101)

        # REST API pagination checks
        api_res = self.client.get(
            "/api/notifications?page=2&page_size=10",
            headers={"Authorization": f"Bearer {self.patient_token}"}
        )
        self.assertEqual(api_res.status_code, 200)
        data = flask_json.loads(api_res.data)
        self.assertEqual(data["page"], 2)
        self.assertEqual(len(data["items"]), 5)

    def test_notification_type_handling(self):
        """Test validation of notification types (item 12)."""
        valid_types = [
            "ANALYSIS_COMPLETED", "REPORT_READY", "EMAIL_DELIVERY_FAILED",
            "EMAIL_RETRY_PENDING", "SECURITY_LOGIN", "SECURITY_WARNING",
            "ACCOUNT_UPDATE", "SYSTEM_ALERT"
        ]
        for vt in valid_types:
            notif_id = self.notif_svc.create_notification(self.patient.id, vt, "Title", "Message")
            self.assertIsNotNone(notif_id)

        # Invalid type must raise ValueError
        with self.assertRaises(ValueError):
            self.notif_svc.create_notification(self.patient.id, "INVALID_TYPE", "Title", "Msg")

    def test_preferences_handling_and_enforcement(self):
        """Test preference retrieval, preference update, and preference enforcement (items 13, 14, 15)."""
        # 13. preference retrieval defaults
        prefs = self.notif_svc.get_preferences(self.patient.id)
        self.assertTrue(prefs["analysis"])
        self.assertTrue(prefs["report"])
        self.assertTrue(prefs["security"])
        self.assertTrue(prefs["account"])

        # 14. preference update (Service layer)
        self.notif_svc.update_preferences(
            user_id=self.patient.id,
            analysis=False,
            report=True,
            security=False,
            account=True
        )
        new_prefs = self.notif_svc.get_preferences(self.patient.id)
        self.assertFalse(new_prefs["analysis"])
        self.assertTrue(new_prefs["report"])
        self.assertFalse(new_prefs["security"])
        self.assertTrue(new_prefs["account"])

        # 15. preference enforcement (Service layer)
        # ANALYSIS_COMPLETED should be suppressed (returns None, not created)
        notif_id = self.notif_svc.create_notification(self.patient.id, "ANALYSIS_COMPLETED", "Suppressed", "Msg")
        self.assertIsNone(notif_id)

        # SECURITY_WARNING must bypass suppression checks and always be created
        warning_id = self.notif_svc.create_notification(self.patient.id, "SECURITY_WARNING", "Warning", "Msg")
        self.assertIsNotNone(warning_id)

        # SYSTEM_ALERT must bypass suppression checks and always be created
        system_id = self.notif_svc.create_notification(self.patient.id, "SYSTEM_ALERT", "System Alert", "Msg")
        self.assertIsNotNone(system_id)

        # API PUT endpoint overrides security = True regardless of user payload
        api_res = self.client.put(
            "/api/notifications/preferences",
            headers={"Authorization": f"Bearer {self.patient_token}"},
            json={
                "analysis": True,
                "report": False,
                "security": False,
                "account": False
            }
        )
        self.assertEqual(api_res.status_code, 200)

        # Verify preferences override is enforced in database
        updated_prefs = self.notif_svc.get_preferences(self.patient.id)
        self.assertTrue(updated_prefs["analysis"])
        self.assertFalse(updated_prefs["report"])
        self.assertTrue(updated_prefs["security"])  # Overridden to True by REST handler
        self.assertFalse(updated_prefs["account"])

    def test_security_and_boundary_cases(self):
        """Test duplicate handling, XSS-safety, invalid IDs, unauthorized access, and malicious metadata (items 16-20)."""
        # 16. duplicate-event protection (graceful multiple writes allowed without DB index/integrity errors)
        id1 = self.notif_svc.create_notification(self.patient.id, "ANALYSIS_COMPLETED", "Duplicate", "Msg")
        id2 = self.notif_svc.create_notification(self.patient.id, "ANALYSIS_COMPLETED", "Duplicate", "Msg")
        self.assertIsNotNone(id1)
        self.assertIsNotNone(id2)
        self.assertNotEqual(id1, id2)

        # 17. XSS-safe rendering/API behavior (API preserves raw HTML/JS input as strings in JSON)
        xss_title = "<script>alert('XSS')</script>"
        xss_msg = "<img src=x onerror=alert(1)>"
        xss_id = self.notif_svc.create_notification(self.patient.id, "ANALYSIS_COMPLETED", xss_title, xss_msg)

        api_res = self.client.get(
            "/api/notifications",
            headers={"Authorization": f"Bearer {self.patient_token}"}
        )
        self.assertEqual(api_res.status_code, 200)
        data = flask_json.loads(api_res.data)
        item = [x for x in data["items"] if x["id"] == xss_id][0]
        self.assertEqual(item["title"], xss_title)
        self.assertEqual(item["message"], xss_msg)

        # 18. invalid notification ID
        res = self.client.patch(
            "/api/notifications/99999/read",
            headers={"Authorization": f"Bearer {self.patient_token}"}
        )
        self.assertEqual(res.status_code, 404)

        res = self.client.delete(
            "/api/notifications/99999",
            headers={"Authorization": f"Bearer {self.patient_token}"}
        )
        self.assertEqual(res.status_code, 404)

        # 19. unauthorized access
        res = self.client.get("/api/notifications")
        self.assertEqual(res.status_code, 401)

        res = self.client.patch("/api/notifications/1/read")
        self.assertEqual(res.status_code, 401)

        # 20. malicious metadata handling
        # Invalid JSON string in metadata must fallback to None
        notif_invalid_json = self.notif_svc.create_notification(
            user_id=self.patient.id,
            type_="ANALYSIS_COMPLETED",
            title="Invalid JSON",
            message="Msg",
            metadata_json="{bad_json"
        )
        self.assertIsNotNone(notif_invalid_json)
        res_invalid = self.notif_svc.list_notifications(user_id=self.patient.id)
        item_invalid = [x for x in res_invalid["items"] if x["id"] == notif_invalid_json][0]
        self.assertIsNone(item_invalid["metadata_json"])

        # Sensitive keys in metadata must be stripped
        sensitive_meta = {
            "scan_id": 100,
            "password": "secretPassword",
            "token": "apiToken",
            "otp": "654321",
            "jwt": "eyJhbGciOi...",
            "otp_code": "0000",
            "password_hash": "argon2..."
        }
        notif_sensitive = self.notif_svc.create_notification(
            user_id=self.patient.id,
            type_="ANALYSIS_COMPLETED",
            title="Sensitive Data",
            message="Msg",
            metadata_json=json.dumps(sensitive_meta)
        )
        self.assertIsNotNone(notif_sensitive)
        res_sensitive = self.notif_svc.list_notifications(user_id=self.patient.id)
        item_sensitive = [x for x in res_sensitive["items"] if x["id"] == notif_sensitive][0]
        cleaned_meta = json.loads(item_sensitive["metadata_json"])

        self.assertEqual(cleaned_meta["scan_id"], 100)
        self.assertNotIn("password", cleaned_meta)
        self.assertNotIn("token", cleaned_meta)
        self.assertNotIn("otp", cleaned_meta)
        self.assertNotIn("jwt", cleaned_meta)
        self.assertNotIn("otp_code", cleaned_meta)
        self.assertNotIn("password_hash", cleaned_meta)


if __name__ == "__main__":
    unittest.main()
