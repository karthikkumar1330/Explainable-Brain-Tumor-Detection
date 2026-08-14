import os
os.environ["DB_PATH"] = os.path.abspath("outputs/test_h48_email_history_security.db")

import unittest
import sqlite3
import datetime
from unittest.mock import patch
from fastapi.testclient import TestClient
from run_api import app
from security.domain.entities import Role, User
from security.infrastructure.jwt_service import JWTService
from security.infrastructure.repository import SQLiteUserRepository
from persistence.infrastructure.repository import SQLitePersistenceRepository
from clinical_reporting.application.services import ReportService

class TestEmailHistorySecurity(unittest.TestCase):
    """Phase H4 Final Security Gate: Email History Access Control Tests."""

    def setUp(self):
        self.db_path = os.environ.get("DB_PATH", "outputs/test_h48_email_history_security.db")
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

        # Override default DB paths in API
        import api.infrastructure.routes as api_routes
        import api.routes.auth_routes as auth_routes
        api_routes.DEFAULT_DB_PATH = self.db_path
        auth_routes.DEFAULT_DB_PATH = self.db_path

        # Disable trigger auto assignment for strict doctor assignment tests
        os.environ["DISABLE_TEST_AUTO_ASSIGN"] = "1"

        # 1. Initialize databases
        persistence_repo = SQLitePersistenceRepository(db_path=self.db_path)
        persistence_repo.initialize_db()

        self.user_repo = SQLiteUserRepository(db_path=self.db_path)
        self.user_repo.initialize_security_tables()

        # Disable trigger auto assignment for strict doctor assignment tests
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DROP TRIGGER IF EXISTS auto_assign_doctor_on_insert;")
            conn.execute("DROP TRIGGER IF EXISTS auto_assign_patient_on_insert;")
            conn.execute("DELETE FROM doctor_patient_assignments;")
            conn.commit()
        finally:
            conn.close()

        # 2. Setup users
        self.admin = self.user_repo.bootstrap_admin()

        # Same-name patients
        # Patient A (Jane Doe, UUID = pat-uuid-aaaa)
        self.patient_a = User(
            id=None,
            uuid="pat-uuid-aaaa",
            email="patienta@aurascan.ai",
            password_hash="fakehash",
            full_name="Jane Doe",
            role=Role.PATIENT,
            is_verified=True,
            is_active=True,
            created_at=datetime.datetime.utcnow().isoformat(),
            updated_at=datetime.datetime.utcnow().isoformat()
        )
        self.patient_a = self.user_repo.create_user(self.patient_a)

        # Patient B (Jane Doe, UUID = pat-uuid-bbbb)
        self.patient_b = User(
            id=None,
            uuid="pat-uuid-bbbb",
            email="patientb@aurascan.ai",
            password_hash="fakehash",
            full_name="Jane Doe",
            role=Role.PATIENT,
            is_verified=True,
            is_active=True,
            created_at=datetime.datetime.utcnow().isoformat(),
            updated_at=datetime.datetime.utcnow().isoformat()
        )
        self.patient_b = self.user_repo.create_user(self.patient_b)

        # Doctor 1 (assigned to Patient A only)
        self.doc_assigned = User(
            id=None,
            uuid="doc-uuid-assigned",
            email="doc_assigned@aurascan.ai",
            password_hash="fakehash",
            full_name="Dr. House",
            role=Role.DOCTOR,
            is_verified=True,
            is_active=True,
            created_at=datetime.datetime.utcnow().isoformat(),
            updated_at=datetime.datetime.utcnow().isoformat()
        )
        self.doc_assigned = self.user_repo.create_user(self.doc_assigned)

        # Doctor 2 (unassigned)
        self.doc_unassigned = User(
            id=None,
            uuid="doc-uuid-unassigned",
            email="doc_unassigned@aurascan.ai",
            password_hash="fakehash",
            full_name="Dr. Stranger",
            role=Role.DOCTOR,
            is_verified=True,
            is_active=True,
            created_at=datetime.datetime.utcnow().isoformat(),
            updated_at=datetime.datetime.utcnow().isoformat()
        )
        self.doc_unassigned = self.user_repo.create_user(self.doc_unassigned)

        # 3. Create tokens
        jwt_svc = JWTService()
        self.admin_token = jwt_svc.create_access_token(self.admin.uuid, self.admin.id, self.admin.email, self.admin.role)
        self.pat_a_token = jwt_svc.create_access_token(self.patient_a.uuid, self.patient_a.id, self.patient_a.email, self.patient_a.role)
        self.pat_b_token = jwt_svc.create_access_token(self.patient_b.uuid, self.patient_b.id, self.patient_b.email, self.patient_b.role)
        self.doc_ass_token = jwt_svc.create_access_token(self.doc_assigned.uuid, self.doc_assigned.id, self.doc_assigned.email, self.doc_assigned.role)
        self.doc_unass_token = jwt_svc.create_access_token(self.doc_unassigned.uuid, self.doc_unassigned.id, self.doc_unassigned.email, self.doc_unassigned.role)

        self.pat_a_headers = {"Authorization": f"Bearer {self.pat_a_token}"}
        self.pat_b_headers = {"Authorization": f"Bearer {self.pat_b_token}"}
        self.doc_ass_headers = {"Authorization": f"Bearer {self.doc_ass_token}"}
        self.doc_unass_headers = {"Authorization": f"Bearer {self.doc_unass_token}"}
        self.admin_headers = {"Authorization": f"Bearer {self.admin_token}"}

        # 4. Seeding data
        conn = sqlite3.connect(self.db_path)
        try:
            # Patients metadata
            conn.execute("INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES ('pat-uuid-aaaa', 'Jane Doe', 30, 'Female', '2026-08-14T00:00:00');")
            conn.execute("INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES ('pat-uuid-bbbb', 'Jane Doe', 40, 'Female', '2026-08-14T00:00:00');")

            # Report 1 owned by Patient A
            conn.execute("INSERT INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at) VALUES (1, 'RPT-A-0001', 'pat-uuid-aaaa', 1, 'FINAL', '2026-08-14T00:00:00', '2026-08-14T00:00:00');")
            # Report 2 owned by Patient B
            conn.execute("INSERT INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at) VALUES (2, 'RPT-B-0002', 'pat-uuid-bbbb', 1, 'FINAL', '2026-08-14T00:00:00', '2026-08-14T00:00:00');")

            # Doctor 1 assigned to Patient A
            conn.execute("INSERT INTO doctor_patient_assignments (doctor_id, patient_id, created_at) VALUES (?, 'pat-uuid-aaaa', '2026-08-14T00:00:00');", (self.doc_assigned.id,))

            # Email delivery for Report 1 (Patient A)
            conn.execute("""
                INSERT INTO email_deliveries (report_id, actor_user_id, recipient_email, status, attempted_at, created_at, updated_at)
                VALUES (1, ?, 'patienta@aurascan.ai', 'SENT', '2026-08-14T10:00:00', '2026-08-14T10:00:00', '2026-08-14T10:00:00');
            """, (self.patient_a.id,))

            # Email delivery for Report 2 (Patient B)
            conn.execute("""
                INSERT INTO email_deliveries (report_id, actor_user_id, recipient_email, status, attempted_at, created_at, updated_at)
                VALUES (2, ?, 'patientb@aurascan.ai', 'SENT', '2026-08-14T10:05:00', '2026-08-14T10:05:00', '2026-08-14T10:05:00');
            """, (self.patient_b.id,))

            conn.commit()
        finally:
            conn.close()

        # 5. FastAPI client
        self.test_client_ctx = TestClient(app)
        self.client = self.test_client_ctx.__enter__()

    def tearDown(self):
        self.test_client_ctx.__exit__()
        if "DISABLE_TEST_AUTO_ASSIGN" in os.environ:
            del os.environ["DISABLE_TEST_AUTO_ASSIGN"]
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def test_01_authenticated_patient_can_access_own_email_history(self) -> None:
        """1. Verify that Patient A can access their own email history."""
        response = self.client.get("/api/reports/email/history", headers=self.pat_a_headers)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("items", data)
        self.assertTrue(len(data["items"]) >= 1)

        # Verify Patient A only sees Report 1's history
        for item in data["items"]:
            self.assertEqual(item["report_id"], 1)

    def test_02_patient_cannot_access_another_patient_history(self) -> None:
        """2. Verify that Patient A cannot access Patient B's history."""
        response = self.client.get("/api/reports/email/history", headers=self.pat_a_headers)
        data = response.json()
        for item in data["items"]:
            self.assertNotEqual(item["report_id"], 2)

    def test_03_same_name_patients_remain_completely_isolated(self) -> None:
        """3. Verify that Patient A and Patient B, sharing the same name 'Jane Doe', remain completely isolated."""
        # Patient A queries history
        res_a = self.client.get("/api/reports/email/history", headers=self.pat_a_headers)
        self.assertEqual(res_a.status_code, 200)
        items_a = res_a.json()["items"]
        self.assertEqual(len(items_a), 1)
        self.assertEqual(items_a[0]["report_id"], 1)

        # Patient B queries history
        res_b = self.client.get("/api/reports/email/history", headers=self.pat_b_headers)
        self.assertEqual(res_b.status_code, 200)
        items_b = res_b.json()["items"]
        self.assertEqual(len(items_b), 1)
        self.assertEqual(items_b[0]["report_id"], 2)

    def test_04_patient_id_tampering_cannot_bypass_ownership(self) -> None:
        """4. Verify that trying to tamper or supply another patient's ID has no impact on patient query isolation."""
        res = self.client.get("/api/reports/email/history?patient_id=pat-uuid-bbbb", headers=self.pat_a_headers)
        self.assertEqual(res.status_code, 200)
        items = res.json()["items"]
        for item in items:
            self.assertEqual(item["report_id"], 1)
            self.assertNotEqual(item["report_id"], 2)

    def test_05_report_id_tampering_cannot_bypass_ownership(self) -> None:
        """5. Verify that Patient A cannot query Report 2 (Patient B's report) directly using report_id parameter."""
        response = self.client.get("/api/reports/email/history?report_id=2", headers=self.pat_a_headers)
        self.assertEqual(response.status_code, 403)

    def test_06_search_cannot_bypass_ownership(self) -> None:
        """6. Verify search filter cannot bypass ownership (e.g. querying history with search terms for other reports)."""
        response = self.client.get("/api/reports/email/history?search=patientb", headers=self.pat_a_headers)
        self.assertEqual(response.status_code, 200)
        items = response.json()["items"]
        self.assertEqual(len(items), 0)

    def test_07_unauthenticated_access_is_rejected(self) -> None:
        """7. Verify that unauthenticated requests to the history endpoint are rejected."""
        response = self.client.get("/api/reports/email/history")
        self.assertEqual(response.status_code, 401)

    def test_08_assigned_doctor_behavior_remains_correct(self) -> None:
        """8. Verify that a doctor assigned to Patient A can query the list or specific report 1 history."""
        # Get history overall
        response = self.client.get("/api/reports/email/history", headers=self.doc_ass_headers)
        self.assertEqual(response.status_code, 200)
        items = response.json()["items"]
        self.assertTrue(len(items) >= 1)
        self.assertEqual(items[0]["report_id"], 1)

        # Get specific report_id 1
        res_rep = self.client.get("/api/reports/email/history?report_id=1", headers=self.doc_ass_headers)
        self.assertEqual(res_rep.status_code, 200)
        self.assertEqual(res_rep.json()["items"][0]["report_id"], 1)

    def test_09_unassigned_doctor_cannot_access_protected_patient_data(self) -> None:
        """9. Verify that an unassigned doctor is restricted from fetching unassigned patient's email history."""
        # 1. Accessing overall list should filter out Patient A's and Patient B's history (unassigned to both)
        response = self.client.get("/api/reports/email/history", headers=self.doc_unass_headers)
        self.assertEqual(response.status_code, 200)
        items = response.json()["items"]
        self.assertEqual(len(items), 0)

        # 2. Querying specific report_id 1 directly must be rejected with 403
        res_rep = self.client.get("/api/reports/email/history?report_id=1", headers=self.doc_unass_headers)
        self.assertEqual(res_rep.status_code, 403)

    def test_10_admin_behavior_remains_correct(self) -> None:
        """10. Verify that Admin can fetch all history including both report 1 and report 2."""
        response = self.client.get("/api/reports/email/history", headers=self.admin_headers)
        self.assertEqual(response.status_code, 200)
        items = response.json()["items"]
        self.assertTrue(len(items) >= 2)

        report_ids = {item["report_id"] for item in items}
        self.assertIn(1, report_ids)
        self.assertIn(2, report_ids)

    def test_11_no_patient_response_contains_another_patient_email_history(self) -> None:
        """11. Verify that Patient A's response contains absolutely zero records of Patient B's history."""
        response = self.client.get("/api/reports/email/history", headers=self.pat_a_headers)
        data = response.json()
        for item in data["items"]:
            self.assertNotEqual(item["report_id"], 2)
            self.assertNotIn("patientb@aurascan.ai", str(item))

    def test_12_no_sql_database_exception_details_are_leaked(self) -> None:
        """12. Verify that invalid requests (e.g. malformed parameters) return error code but do not leak SQL database exception details."""
        response = self.client.get("/api/reports/email/history?report_id=invalid", headers=self.pat_a_headers)
        self.assertIn(response.status_code, [400, 422, 500])
        self.assertNotIn("sqlite3.OperationalError", response.text)
        self.assertNotIn("near \"SELECT\"", response.text)
        self.assertNotIn("outputs/test_h48_email_history_security.db", response.text)
