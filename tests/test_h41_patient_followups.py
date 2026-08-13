import os
import sqlite3
import datetime
import unittest
from fastapi.testclient import TestClient

os.environ["DB_PATH"] = os.path.abspath("outputs/test_h41_followup.db")
os.environ["DISABLE_TEST_AUTO_ASSIGN"] = "1"

from run_api import app
from dashboard.infrastructure.web_server import create_app
from security.domain.entities import Role, User
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.jwt_service import JWTService
from persistence.infrastructure.repository import SQLitePersistenceRepository
from security.infrastructure.encryption_service import PIIEncryptionService


class TestH41PatientFollowups(unittest.TestCase):
    """Focused integration tests for Phase H4.1 Patient Follow-up View & Authorization."""

    def setUp(self):
        test_method_name = self.id().split('.')[-1]
        self.db_path = os.path.abspath(f"outputs/test_h41_followup_{test_method_name}.db")
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

        os.environ["DB_PATH"] = self.db_path
        os.environ["DISABLE_TEST_AUTO_ASSIGN"] = "1"

        # Update route db paths
        import api.infrastructure.routes as api_routes
        import api.routes.auth_routes as auth_routes
        api_routes.DEFAULT_DB_PATH = self.db_path
        auth_routes.DEFAULT_DB_PATH = self.db_path

        # Flask App setup
        self.flask_app = create_app(db_path=self.db_path)
        self.flask_app.config["TESTING"] = True
        self.flask_app.config["DISABLE_CSRF"] = True
        self.flask_client = self.flask_app.test_client()

        # DB Setup
        self.persistence_repo = SQLitePersistenceRepository(db_path=self.db_path)
        self.persistence_repo.initialize_db()
        self.user_repo = SQLiteUserRepository(db_path=self.db_path)
        self.user_repo.initialize_security_tables()

        # Drop auto-assignment triggers
        conn_drop = sqlite3.connect(self.db_path)
        try:
            conn_drop.execute("DROP TRIGGER IF EXISTS auto_assign_doctor_on_insert;")
            conn_drop.execute("DROP TRIGGER IF EXISTS auto_assign_patient_on_insert;")
            conn_drop.execute("DELETE FROM doctor_patient_assignments;")
            conn_drop.commit()
        finally:
            conn_drop.close()

        # Setup passwords and decrypt keys
        from security.infrastructure.password import PasswordHasher
        pass_hash = PasswordHasher.hash_password("Password@123")

        # Bootstrap Database Users
        conn = sqlite3.connect(self.db_path)
        try:
            # 1. Doctors
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (200, "doc-uuid-a", "doctor_a@aurascan.ai", pass_hash, "Doctor A", Role.DOCTOR.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (201, "doc-uuid-b", "doctor_b@aurascan.ai", pass_hash, "Doctor B", Role.DOCTOR.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            # 2. Patients
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (300, "pat-uuid-a", "patient_a@aurascan.ai", pass_hash, "Patient A", Role.PATIENT.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (301, "pat-uuid-b", "patient_b@aurascan.ai", pass_hash, "Patient B", Role.PATIENT.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            # 3. Patient Demographics (Encrypted)
            enc_svc = PIIEncryptionService()
            conn.execute(
                "INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
                ("pat-uuid-a", enc_svc.encrypt("Patient A Name"), enc_svc.encrypt("45"), enc_svc.encrypt("Male"), datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
                ("pat-uuid-b", enc_svc.encrypt("Patient B Name"), enc_svc.encrypt("30"), enc_svc.encrypt("Female"), datetime.datetime.utcnow().isoformat())
            )
            # 4. Assignments: Doctor A is assigned to Patient A
            conn.execute(
                "INSERT INTO doctor_patient_assignments (doctor_id, patient_id, created_at) VALUES (?, ?, ?);",
                (200, "pat-uuid-a", datetime.datetime.utcnow().isoformat())
            )
            # 5. Seed a follow-up for Patient A
            conn.execute(
                "INSERT INTO followup_schedules (followup_id, patient_id, doctor_id, scheduled_date, status, reason, notes, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);",
                (1, "pat-uuid-a", 200, "2026-09-15T10:00:00", "scheduled", "Diagnostic checkup", "Please bring historical scans.", datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            conn.commit()
        finally:
            conn.close()

        # Initialize JWT Tokens
        self.jwt_svc = JWTService()
        self.token_doc_a = self.jwt_svc.create_access_token("doc-uuid-a", 200, "doctor_a@aurascan.ai", Role.DOCTOR)
        self.token_doc_b = self.jwt_svc.create_access_token("doc-uuid-b", 201, "doctor_b@aurascan.ai", Role.DOCTOR)
        self.token_pat_a = self.jwt_svc.create_access_token("pat-uuid-a", 300, "patient_a@aurascan.ai", Role.PATIENT)
        self.token_pat_b = self.jwt_svc.create_access_token("pat-uuid-b", 301, "patient_b@aurascan.ai", Role.PATIENT)

        # Setup FastAPI Client
        self.fastapi_ctx = TestClient(app)
        self.fastapi_client = self.fastapi_ctx.__enter__()

    def tearDown(self):
        self.fastapi_ctx.__exit__(None, None, None)
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def test_01_authorized_patient_can_retrieve_own_followups(self):
        """Verify that an authorized patient can successfully retrieve their own follow-ups in Flask."""
        self.flask_client.set_cookie("access_token", self.token_pat_a)
        res = self.flask_client.get("/api/patient/followups")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["followup_id"], 1)
        self.assertEqual(data[0]["reason"], "Diagnostic checkup")
        self.assertEqual(data[0]["status"], "scheduled")

    def test_02_patient_with_no_followups_receives_empty_result_safely(self):
        """Verify that a patient with no follow-ups receives an empty list safely."""
        self.flask_client.set_cookie("access_token", self.token_pat_b)
        res = self.flask_client.get("/api/patient/followups")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data, [])

    def test_03_patient_cannot_retrieve_another_patients_followups(self):
        """Verify that a patient cannot access another patient's follow-up records directly or indirectly."""
        # Flask: Patient A tries to hit the Doctor route for Patient B -> denied
        self.flask_client.set_cookie("access_token", self.token_pat_a)
        res = self.flask_client.get("/api/doctor/patients/pat-uuid-b/followups")
        self.assertEqual(res.status_code, 403)

        # FastAPI: Patient A tries to retrieve Patient B's follow-ups -> denied via service auth
        headers = {"Authorization": f"Bearer {self.token_pat_a}"}
        res_fast = self.fastapi_client.get("/api/patients/pat-uuid-b/followups", headers=headers)
        self.assertEqual(res_fast.status_code, 403)

    def test_04_client_cannot_bypass_authorization_by_changing_patient_id(self):
        """Verify that the client cannot bypass authorization by manipulating the parameter values."""
        # Patient B tries to get Patient A's follow-ups on FastAPI by supplying Patient A's ID
        headers = {"Authorization": f"Bearer {self.token_pat_b}"}
        res_fast = self.fastapi_client.get("/api/patients/pat-uuid-a/followups", headers=headers)
        self.assertEqual(res_fast.status_code, 403)

        # The Flask patient route /api/patient/followups has no parameter, so they cannot manipulate it
        self.flask_client.set_cookie("access_token", self.token_pat_b)
        res = self.flask_client.get("/api/patient/followups?patient_id=pat-uuid-a")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json(), [])  # Returns Patient B's empty list, ignores query param

    def test_05_unauthenticated_user_is_denied(self):
        """Verify that requests without valid authentication tokens are rejected with 401."""
        res = self.flask_client.get("/api/patient/followups")
        self.assertEqual(res.status_code, 401)

        res_fast = self.fastapi_client.get("/api/patients/pat-uuid-a/followups")
        self.assertEqual(res_fast.status_code, 401)

    def test_06_doctor_admin_access_follows_existing_role_rules(self):
        """Verify that clinician/doctor access continues to enforce doctor assignment rules."""
        # Assigned Doctor A can fetch Patient A
        self.flask_client.set_cookie("access_token", self.token_doc_a)
        res = self.flask_client.get("/api/doctor/patients/pat-uuid-a/followups")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(len(res.get_json()), 1)

        # Unassigned Doctor B is denied
        self.flask_client.set_cookie("access_token", self.token_doc_b)
        res_denied = self.flask_client.get("/api/doctor/patients/pat-uuid-a/followups")
        self.assertEqual(res_denied.status_code, 403)

    def test_07_patient_blocked_from_doctor_write_endpoints(self):
        """Verify that patient roles are blocked from accessing write follow-up endpoints."""
        self.flask_client.set_cookie("access_token", self.token_pat_a)

        # Try to create follow-up
        res1 = self.flask_client.post(
            "/api/doctor/patients/pat-uuid-a/followups",
            json={"scheduled_date": "2026-09-20", "reason": "Self-scheduling"}
        )
        self.assertEqual(res1.status_code, 403)

        # Try to update follow-up
        res2 = self.flask_client.put(
            "/api/doctor/followups/1",
            json={"status": "completed"}
        )
        self.assertEqual(res2.status_code, 403)

    def test_08_followup_records_persisted_safely(self):
        """Verify follow-up schedule records remain persisted correctly in SQLite."""
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute("SELECT reason, status FROM followup_schedules WHERE followup_id = 1;").fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row[0], "Diagnostic checkup")
            self.assertEqual(row[1], "scheduled")
        finally:
            conn.close()

    def test_09_error_responses_do_not_disclose_sensitive_information(self):
        """Verify that error messages are generic and do not leak database/internal exceptions."""
        self.flask_client.set_cookie("access_token", self.token_pat_a)

        original_db = self.flask_app.config["DB_PATH"]
        self.flask_app.config["DB_PATH"] = "invalid_directory/nonexistent_file.db"
        try:
            res = self.flask_client.get("/api/patient/followups")
            self.assertEqual(res.status_code, 500)
            data = res.get_json()
            self.assertEqual(data["error"], "Internal server error")
        finally:
            self.flask_app.config["DB_PATH"] = original_db
