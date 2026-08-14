import os
import unittest
import tempfile
import sqlite3
import datetime
import json
from security.domain.entities import Role, User
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.jwt_service import JWTService
from security.infrastructure.password import PasswordHasher
from persistence.infrastructure.repository import SQLitePersistenceRepository
from clinical_reporting.application.notification_service import NotificationService
from clinical_reporting.application.followup_service import FollowupScheduleService, FollowupScheduleServiceException

class TestH62FollowupNotifications(unittest.TestCase):
    """Focused integration tests for Phase H6.2: Follow-up Notification Events."""

    def setUp(self):
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")

        # Initialize databases
        self.persistence_repo = SQLitePersistenceRepository(db_path=self.db_path)
        self.persistence_repo.initialize_db()

        self.user_repo = SQLiteUserRepository(db_path=self.db_path)
        self.user_repo.initialize_security_tables()

        # Setup test data
        pass_hash = PasswordHasher.hash_password("AuraScan@2026")

        # Patients
        self.patient = User(
            id=None, uuid="pat-111", email="patient@aurascan.ai",
            password_hash=pass_hash, full_name="John Patient", role=Role.PATIENT,
            is_verified=True, is_active=True
        )
        self.user_repo.create_user(self.patient)

        self.other_patient = User(
            id=None, uuid="pat-999", email="other@aurascan.ai",
            password_hash=pass_hash, full_name="Other Patient", role=Role.PATIENT,
            is_verified=True, is_active=True
        )
        self.user_repo.create_user(self.other_patient)

        # Doctors
        self.doctor = User(
            id=None, uuid="doc-222", email="doctor@aurascan.ai",
            password_hash=pass_hash, full_name="Dr. Smith", role=Role.DOCTOR,
            is_verified=True, is_active=True
        )
        self.user_repo.create_user(self.doctor)

        self.other_doctor = User(
            id=None, uuid="doc-888", email="otherdoc@aurascan.ai",
            password_hash=pass_hash, full_name="Dr. Anonymous", role=Role.DOCTOR,
            is_verified=True, is_active=True
        )
        self.user_repo.create_user(self.other_doctor)

        # Admin
        self.admin = self.user_repo.get_by_email("admin@aurascan.ai")

        # Setup patient/doctor records and assignments
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute("INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
                             ("pat-111", "John Patient", 30, "M", datetime.datetime.utcnow().isoformat()))
                conn.execute("INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
                             ("pat-999", "Other Patient", 40, "F", datetime.datetime.utcnow().isoformat()))
                conn.execute("INSERT OR IGNORE INTO doctor_patient_assignments (doctor_id, patient_id, created_at) VALUES (?, ?, ?);",
                             (self.doctor.id, "pat-111", datetime.datetime.utcnow().isoformat()))
        finally:
            conn.close()

        self.followup_svc = FollowupScheduleService(db_path=self.db_path)
        self.notif_svc = NotificationService(db_path=self.db_path)

    def tearDown(self):
        os.close(self.db_fd)
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def test_01_authorized_patient_receives_creation_notification(self):
        """Verify that scheduling a follow-up triggers a notification to the patient, and no self-notification to doctor."""
        res = self.followup_svc.create_followup(
            actor=self.doctor,
            patient_id="pat-111",
            scheduled_date="2026-12-01",
            reason="Checkup",
            notes="Routine follow-up"
        )
        self.assertIsNotNone(res["followup_id"])

        # Check patient's notifications
        p_notifs = self.notif_svc.list_notifications(user_id=self.patient.id)
        self.assertEqual(p_notifs["total_count"], 1)
        self.assertEqual(p_notifs["items"][0]["type"], "FOLLOWUP_CREATED")
        self.assertIn("2026-12-01", p_notifs["items"][0]["message"])

        # Doctor (actor) must NOT receive a duplicate self-notification
        d_notifs = self.notif_svc.list_notifications(user_id=self.doctor.id)
        self.assertEqual(d_notifs["total_count"], 0)

    def test_02_assigned_doctor_receives_creation_notification_when_scheduled_by_admin(self):
        """Verify that when an admin schedules a follow-up, both the patient and the assigned doctor receive it."""
        res = self.followup_svc.create_followup(
            actor=self.admin,
            patient_id="pat-111",
            scheduled_date="2026-12-05",
            reason="MRI review"
        )

        # Patient receives notification
        p_notifs = self.notif_svc.list_notifications(user_id=self.patient.id)
        self.assertEqual(p_notifs["total_count"], 1)

        # Assigned doctor receives notification (since they are not the actor)
        d_notifs = self.notif_svc.list_notifications(user_id=self.doctor.id)
        self.assertEqual(d_notifs["total_count"], 1)
        self.assertEqual(d_notifs["items"][0]["type"], "FOLLOWUP_CREATED")

    def test_03_followup_update_reschedule_notification(self):
        """Verify that updating scheduled_date, reason, or notes triggers FOLLOWUP_UPDATED."""
        res = self.followup_svc.create_followup(
            actor=self.doctor,
            patient_id="pat-111",
            scheduled_date="2026-12-01",
            reason="Checkup"
        )
        fid = res["followup_id"]

        # Clear patient's create notification to isolate update tests
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute("DELETE FROM notifications WHERE user_id = ?;", (self.patient.id,))
        finally:
            conn.close()

        # Update reschedule
        self.followup_svc.update_followup(
            actor=self.doctor,
            followup_id=fid,
            scheduled_date="2026-12-10"
        )

        # Patient receives update notification
        p_notifs = self.notif_svc.list_notifications(user_id=self.patient.id)
        self.assertEqual(p_notifs["total_count"], 1)
        self.assertEqual(p_notifs["items"][0]["type"], "FOLLOWUP_UPDATED")
        self.assertIn("2026-12-10", p_notifs["items"][0]["message"])

    def test_04_cancellation_notification(self):
        """Verify that cancelling a follow-up triggers a FOLLOWUP_CANCELLED notification."""
        res = self.followup_svc.create_followup(
            actor=self.doctor,
            patient_id="pat-111",
            scheduled_date="2026-12-01"
        )
        fid = res["followup_id"]

        # Clear notifications
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute("DELETE FROM notifications;")
        finally:
            conn.close()

        # Cancel follow-up
        self.followup_svc.cancel_followup(actor=self.doctor, followup_id=fid)

        # Patient gets cancellation notification
        p_notifs = self.notif_svc.list_notifications(user_id=self.patient.id)
        self.assertEqual(p_notifs["total_count"], 1)
        self.assertEqual(p_notifs["items"][0]["type"], "FOLLOWUP_CANCELLED")

    def test_05_unauthorized_patient_cannot_receive_another_patients_followup_notification(self):
        """Verify that a patient user is strictly isolated and cannot retrieve follow-up notifications of other patients."""
        # Create followup for patient A
        self.followup_svc.create_followup(
            actor=self.doctor,
            patient_id="pat-111",
            scheduled_date="2026-12-01"
        )

        # Other patient queries own notifications -> should be empty
        other_notifs = self.notif_svc.list_notifications(user_id=self.other_patient.id)
        self.assertEqual(other_notifs["total_count"], 0)

    def test_06_unauthorized_doctor_cannot_receive_notifications_for_unassigned_patient(self):
        """Verify that unassigned doctors do not receive creation/update notifications for a patient."""
        # Unassign other_doctor manually from pat-111
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute("DELETE FROM doctor_patient_assignments WHERE doctor_id = ? AND patient_id = ?;",
                             (self.other_doctor.id, "pat-111"))
        finally:
            conn.close()

        # Admin schedules a follow-up for patient A (pat-111)
        self.followup_svc.create_followup(
            actor=self.admin,
            patient_id="pat-111",
            scheduled_date="2026-12-01"
        )

        # Assigned doctor receives it
        d1_notifs = self.notif_svc.list_notifications(user_id=self.doctor.id)
        self.assertEqual(d1_notifs["total_count"], 1)

        # Unassigned doctor does NOT receive it
        d2_notifs = self.notif_svc.list_notifications(user_id=self.other_doctor.id)
        self.assertEqual(d2_notifs["total_count"], 0)

    def test_07_notification_failure_does_not_corrupt_followup_state(self):
        """Verify that a notification system failure (e.g. invalid database connection) does not crash or roll back the follow-up schedule creation/update."""
        # Mock notif_svc to raise an exception
        original_create = self.followup_svc.notif_svc.create_notification

        def mock_create(*args, **kwargs):
            raise Exception("Database write failed for notification log")

        self.followup_svc.notif_svc.create_notification = mock_create

        try:
            # Creation should complete successfully despite notification crash
            res = self.followup_svc.create_followup(
                actor=self.doctor,
                patient_id="pat-111",
                scheduled_date="2026-12-01",
                reason="Stress test"
            )
            self.assertIsNotNone(res["followup_id"])

            # Verify database record actually exists
            conn = sqlite3.connect(self.db_path)
            try:
                row = conn.execute("SELECT status FROM followup_schedules WHERE followup_id = ?;", (res["followup_id"],)).fetchone()
                self.assertIsNotNone(row)
                self.assertEqual(row[0], "scheduled")
            finally:
                conn.close()
        finally:
            # Restore original method
            self.followup_svc.notif_svc.create_notification = original_create

    def test_08_notification_metadata_is_sanitized(self):
        """Verify that the notification system strips out sensitive metadata payload properties."""
        # Trigger notification using metadata that contains sensitive keys
        meta = {
            "followup_id": 42,
            "patient_id": "pat-111",
            "password": "SecretPassword",
            "token": "SuperSecretTokenValue"
        }

        self.notif_svc.create_notification(
            user_id=self.patient.id,
            type_="FOLLOWUP_CREATED",
            title="Sanitize Test",
            message="Check details",
            metadata_json=json.dumps(meta)
        )

        notifs = self.notif_svc.list_notifications(user_id=self.patient.id)
        self.assertEqual(notifs["total_count"], 1)

        stored_meta = json.loads(notifs["items"][0]["metadata_json"])
        self.assertIn("followup_id", stored_meta)
        self.assertIn("patient_id", stored_meta)
        # Sensitive properties must be stripped
        self.assertNotIn("password", stored_meta)
        self.assertNotIn("token", stored_meta)

if __name__ == "__main__":
    unittest.main()
