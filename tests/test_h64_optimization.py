import os
import unittest
import tempfile
import sqlite3
import datetime
import json
from security.domain.entities import Role, User
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.password import PasswordHasher
from persistence.infrastructure.repository import SQLitePersistenceRepository
from clinical_reporting.application.notification_service import NotificationService

class TestH64Optimization(unittest.TestCase):
    """Focused performance optimization and query plan tests for Phase H6.4: Notification Database Query Optimization."""

    def setUp(self):
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")

        # Initialize databases
        self.persistence_repo = SQLitePersistenceRepository(db_path=self.db_path)
        self.persistence_repo.initialize_db()

        self.user_repo = SQLiteUserRepository(db_path=self.db_path)
        self.user_repo.initialize_security_tables()

        # Setup test users
        pass_hash = PasswordHasher.hash_password("AuraScan@2026")

        self.patient = User(
            id=None, uuid="pat-111", email="patient@aurascan.ai",
            password_hash=pass_hash, full_name="John Patient", role=Role.PATIENT,
            is_verified=True, is_active=True
        )
        self.user_repo.create_user(self.patient)

        self.other_patient = User(
            id=None, uuid="pat-222", email="other@aurascan.ai",
            password_hash=pass_hash, full_name="Other Patient", role=Role.PATIENT,
            is_verified=True, is_active=True
        )
        self.user_repo.create_user(self.other_patient)

        self.notif_svc = NotificationService(db_path=self.db_path)

    def tearDown(self):
        os.close(self.db_fd)
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def test_01_fresh_database_index_creation(self):
        """Verify that a fresh database setup includes the composite index."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("PRAGMA index_list(notifications);")
            indexes = [row[1] for row in cursor.fetchall()]
            self.assertIn("idx_notifications_user_created", indexes)

            # Check columns in index
            cursor.execute("PRAGMA index_info(idx_notifications_user_created);")
            cols = [row[2] for row in cursor.fetchall()]
            self.assertEqual(cols, ["user_id", "created_at"])
        finally:
            conn.close()

    def test_02_existing_database_migration_and_repeated_runs(self):
        """Verify that migrations are idempotent, preserve existing notifications, and do not fail on multiple runs."""
        # 1. Simulate an old database without the idx_notifications_user_created index
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute("DROP INDEX IF EXISTS idx_notifications_user_created;")
                conn.execute("INSERT INTO notifications (user_id, type, title, message, created_at) VALUES (?, 'T', 'Title', 'Msg', '2026-08-14');", (self.patient.id,))
        finally:
            conn.close()

        # 2. Run migration again
        self.user_repo.initialize_security_tables()

        # 3. Verify index exists and historical notification is preserved
        conn = sqlite3.connect(self.db_path)
        try:
            # Check index list
            indexes = [r[1] for r in conn.execute("PRAGMA index_list(notifications);").fetchall()]
            self.assertIn("idx_notifications_user_created", indexes)

            # Check row count
            count = conn.execute("SELECT COUNT(*) FROM notifications;").fetchone()[0]
            self.assertEqual(count, 1)
        finally:
            conn.close()

        # 4. Verify repeated migration runs do not fail
        try:
            self.user_repo.initialize_security_tables()
            self.user_repo.initialize_security_tables()
        except Exception as e:
            self.fail(f"Repeated migrations failed with exception: {e}")

    def test_03_preservation_of_h63_idempotency_index(self):
        """Verify that the composite query index does not alter or conflict with the unique idempotency index."""
        conn = sqlite3.connect(self.db_path)
        try:
            indexes = conn.execute("PRAGMA index_list(notifications);").fetchall()
            idempotency_index = [idx for idx in indexes if idx[1] == "idx_notifications_idempotency"][0]

            # Ensure it is unique (unique = 1)
            self.assertEqual(idempotency_index[2], 1)
        finally:
            conn.close()

    def test_04_listing_pagination_and_unread_count_correctness(self):
        """Verify that notifications listing, pagination, sorting, and unread counts behave correctly with the index."""
        # Insert 5 notifications with different timestamps
        for i in range(5):
            self.notif_svc.create_notification(
                user_id=self.patient.id,
                type_="ANALYSIS_COMPLETED",
                title=f"Notification {i}",
                message=f"Msg {i}",
                metadata_json=json.dumps({"report_id": 10 + i})
            )

        # Get unread count
        self.assertEqual(self.notif_svc.get_unread_count(self.patient.id), 5)

        # List first page (size 3)
        res1 = self.notif_svc.list_notifications(user_id=self.patient.id, page=1, page_size=3)
        self.assertEqual(res1["total_count"], 5)
        self.assertEqual(len(res1["items"]), 3)
        self.assertEqual(res1["total_pages"], 2)

        # Confirm sorting (latest first)
        timestamps = [item["created_at"] for item in res1["items"]]
        self.assertEqual(timestamps, sorted(timestamps, reverse=True))

        # List second page (size 3)
        res2 = self.notif_svc.list_notifications(user_id=self.patient.id, page=2, page_size=3)
        self.assertEqual(len(res2["items"]), 2)

    def test_05_authorization_and_idor_correctness(self):
        """Verify that listing notifications is strictly bound to the requesting user_id."""
        self.notif_svc.create_notification(
            user_id=self.patient.id,
            type_="ANALYSIS_COMPLETED",
            title="Patient A Notice",
            message="Msg A",
            metadata_json=json.dumps({"report_id": 99})
        )

        # Patient B queries their own notifications -> should be empty
        res_b = self.notif_svc.list_notifications(user_id=self.other_patient.id)
        self.assertEqual(res_b["total_count"], 0)

    def test_06_explain_query_plan_index_usage(self):
        """Verify using EXPLAIN QUERY PLAN that the index is utilized for paginated queries and eliminates temporary sorting."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            # Query plan for listing notifications
            sql = """
                SELECT id, user_id, type, title, message, is_read, created_at, read_at, metadata_json
                FROM notifications
                WHERE user_id = 1
                ORDER BY created_at DESC
                LIMIT 5 OFFSET 0;
            """
            cursor.execute(f"EXPLAIN QUERY PLAN {sql}")
            plans = cursor.fetchall()

            # Print for visual review
            plan_str = "\n".join([row[3] for row in plans])
            print(f"\nExplain Output:\n{plan_str}")

            # Verify index is used
            index_used = any("idx_notifications_user_created" in row[3] for row in plans)
            self.assertTrue(index_used, "Index idx_notifications_user_created was not used!")

            # Verify no temporary B-tree sorting is used
            sorting_used = any("USE TEMP B-TREE FOR ORDER BY" in row[3] for row in plans)
            self.assertFalse(sorting_used, "Temporary B-tree sort was detected!")
        finally:
            conn.close()

if __name__ == "__main__":
    unittest.main()
