import os
import unittest
import tempfile
import sqlite3
import datetime
import json
import concurrent.futures
from security.domain.entities import Role, User
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.password import PasswordHasher
from persistence.infrastructure.repository import SQLitePersistenceRepository
from clinical_reporting.application.notification_service import NotificationService

class TestH63Deduplication(unittest.TestCase):
    """Focused integration, concurrency, and security tests for Phase H6.3: Notification Deduplication."""

    def setUp(self):
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")

        # Initialize databases
        self.persistence_repo = SQLitePersistenceRepository(db_path=self.db_path)
        self.persistence_repo.initialize_db()

        self.user_repo = SQLiteUserRepository(db_path=self.db_path)
        self.user_repo.initialize_security_tables()

        # Setup test users
        pass_hash = PasswordHasher.hash_password("AuraScan@2026")

        self.user_a = User(
            id=None, uuid="user-aaa", email="user_a@aurascan.ai",
            password_hash=pass_hash, full_name="User A", role=Role.PATIENT,
            is_verified=True, is_active=True
        )
        self.user_repo.create_user(self.user_a)

        self.user_b = User(
            id=None, uuid="user-bbb", email="user_b@aurascan.ai",
            password_hash=pass_hash, full_name="User B", role=Role.PATIENT,
            is_verified=True, is_active=True
        )
        self.user_repo.create_user(self.user_b)

        self.notif_svc = NotificationService(db_path=self.db_path)

    def tearDown(self):
        os.close(self.db_fd)
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def test_01_sequential_duplicates_same_logical_event(self):
        """Verify that sequential duplicate inserts of the same logical event result in only one notification row."""
        id1 = self.notif_svc.create_notification(
            user_id=self.user_a.id,
            type_="FOLLOWUP_CREATED",
            title="Follow-up Scheduled",
            message="Your follow-up is scheduled for 2026-12-01.",
            metadata_json=json.dumps({"followup_id": 10})
        )
        self.assertIsNotNone(id1)

        id2 = self.notif_svc.create_notification(
            user_id=self.user_a.id,
            type_="FOLLOWUP_CREATED",
            title="Follow-up Scheduled",
            message="Your follow-up is scheduled for 2026-12-01.",
            metadata_json=json.dumps({"followup_id": 10})
        )
        # Should return the same ID and not insert a new row
        self.assertEqual(id1, id2)

        # Assert database has exactly 1 row
        conn = sqlite3.connect(self.db_path)
        try:
            row_count = conn.execute("SELECT COUNT(*) FROM notifications;").fetchone()[0]
            self.assertEqual(row_count, 1)
        finally:
            conn.close()

    def test_02_concurrent_duplicates_same_logical_event(self):
        """Verify that concurrent threads attempting to insert the same logical notification are deduplicated safely without bubbling IntegrityErrors."""
        num_threads = 10
        metadata = json.dumps({"followup_id": 12})

        def run_insert():
            # Return result of the insert
            return self.notif_svc.create_notification(
                user_id=self.user_a.id,
                type_="FOLLOWUP_CREATED",
                title="Follow-up Concurrent Test",
                message="Concurrent test message",
                metadata_json=metadata
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(run_insert) for _ in range(num_threads)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All threads must receive a valid non-None ID
        for res in results:
            self.assertIsNotNone(res)

        # All returned IDs must be identical
        first_id = results[0]
        for res in results:
            self.assertEqual(first_id, res)

        # Exactly 1 row must be created in the database
        conn = sqlite3.connect(self.db_path)
        try:
            row_count = conn.execute("SELECT COUNT(*) FROM notifications;").fetchone()[0]
            self.assertEqual(row_count, 1)
        finally:
            conn.close()

    def test_03_different_followup_update_events_create_separate_notifications(self):
        """Verify that legitimate repeated follow-up updates (with different dates or details) create separate notifications."""
        # Initial creation
        id1 = self.notif_svc.create_notification(
            user_id=self.user_a.id,
            type_="FOLLOWUP_UPDATED",
            title="Follow-up Rescheduled",
            message="Your follow-up is rescheduled for 2026-12-10.",
            metadata_json=json.dumps({"followup_id": 15, "scheduled_date": "2026-12-10"})
        )

        # Subsequent rescheduled update to a new date
        id2 = self.notif_svc.create_notification(
            user_id=self.user_a.id,
            type_="FOLLOWUP_UPDATED",
            title="Follow-up Rescheduled",
            message="Your follow-up is rescheduled for 2026-12-12.",
            metadata_json=json.dumps({"followup_id": 15, "scheduled_date": "2026-12-12"})
        )

        self.assertIsNotNone(id1)
        self.assertIsNotNone(id2)
        self.assertNotEqual(id1, id2)

        # Database should have exactly 2 rows
        conn = sqlite3.connect(self.db_path)
        try:
            row_count = conn.execute("SELECT COUNT(*) FROM notifications;").fetchone()[0]
            self.assertEqual(row_count, 2)
        finally:
            conn.close()

    def test_04_different_recipients_create_separate_notifications(self):
        """Verify that identical logical events sent to different recipients create separate notifications."""
        id1 = self.notif_svc.create_notification(
            user_id=self.user_a.id,
            type_="ANALYSIS_COMPLETED",
            title="MRI Ready",
            message="Scan processed.",
            metadata_json=json.dumps({"report_id": 5})
        )

        id2 = self.notif_svc.create_notification(
            user_id=self.user_b.id,
            type_="ANALYSIS_COMPLETED",
            title="MRI Ready",
            message="Scan processed.",
            metadata_json=json.dumps({"report_id": 5})
        )

        self.assertNotEqual(id1, id2)

    def test_05_different_event_types_create_separate_notifications(self):
        """Verify that different event types for the same resource create separate notifications."""
        id1 = self.notif_svc.create_notification(
            user_id=self.user_a.id,
            type_="ANALYSIS_COMPLETED",
            title="MRI Processed",
            message="Processed.",
            metadata_json=json.dumps({"report_id": 6})
        )

        id2 = self.notif_svc.create_notification(
            user_id=self.user_a.id,
            type_="REPORT_READY",
            title="Report Ready",
            message="Processed.",
            metadata_json=json.dumps({"report_id": 6})
        )

        self.assertNotEqual(id1, id2)

    def test_06_different_resource_ids_create_separate_notifications(self):
        """Verify that notifications for different resource IDs create separate records."""
        id1 = self.notif_svc.create_notification(
            user_id=self.user_a.id,
            type_="ANALYSIS_COMPLETED",
            title="MRI Ready",
            message="Scan processed.",
            metadata_json=json.dumps({"report_id": 10})
        )

        id2 = self.notif_svc.create_notification(
            user_id=self.user_a.id,
            type_="ANALYSIS_COMPLETED",
            title="MRI Ready",
            message="Scan processed.",
            metadata_json=json.dumps({"report_id": 11})
        )

        self.assertNotEqual(id1, id2)

    def test_07_existing_notifications_remain_intact_and_migration_is_idempotent(self):
        """Verify that database migration handles existing tables, including duplicate rows, safely without deletion, and is safe to run repeatedly."""
        # 1. Simulate an old database by dropping columns or creating the old schema
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute("DROP TABLE IF EXISTS notifications;")
                conn.execute("""
                CREATE TABLE notifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    is_read INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    read_at TEXT,
                    metadata_json TEXT
                );
                """)
                # Insert duplicates in the old schema
                conn.execute("INSERT INTO notifications (user_id, type, title, message, created_at) VALUES (1, 'T', 'Title', 'Msg', '2026-08-14');")
                conn.execute("INSERT INTO notifications (user_id, type, title, message, created_at) VALUES (1, 'T', 'Title', 'Msg', '2026-08-14');")
        finally:
            conn.close()

        # 2. Run initialize_security_tables to execute the migration
        self.user_repo.initialize_security_tables()

        # 3. Verify both historical rows are still present and have unique idempotency keys
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute("SELECT id, idempotency_key FROM notifications ORDER BY id ASC;").fetchall()
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0][1], "migrated:1")
            self.assertEqual(rows[1][1], "migrated:2")
        finally:
            conn.close()

        # 4. Run migration again to verify it is completely idempotent (no exception)
        try:
            self.user_repo.initialize_security_tables()
        except Exception as e:
            self.fail(f"Re-running migration failed: {e}")

    def test_08_no_secrets_or_sensitive_pii_in_idempotency_key(self):
        """Verify that the generated idempotency key contains no passwords, tokens, or PII names."""
        meta = json.dumps({
            "followup_id": 30,
            "patient_id": "pat-111",
            "password": "Password123!",
            "token": "sensitive-token-xyz"
        })
        id1 = self.notif_svc.create_notification(
            user_id=self.user_a.id,
            type_="FOLLOWUP_CREATED",
            title="Secure Test",
            message="Check follow-up",
            metadata_json=meta
        )

        conn = sqlite3.connect(self.db_path)
        try:
            key = conn.execute("SELECT idempotency_key FROM notifications WHERE id = ?;", (id1,)).fetchone()[0]
            # Verify key has only type and safe hash, no plaintext secrets
            self.assertTrue(key.startswith("auto:FOLLOWUP_CREATED:"))
            self.assertNotIn("Password123!", key)
            self.assertNotIn("sensitive-token-xyz", key)
            self.assertNotIn("pat-111", key)
        finally:
            conn.close()

if __name__ == "__main__":
    unittest.main()
