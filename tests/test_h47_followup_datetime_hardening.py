import os
import sqlite3
import datetime
import unittest
from fastapi.testclient import TestClient

os.environ["DB_PATH"] = os.path.abspath("outputs/test_h47_datetime.db")
os.environ["DISABLE_TEST_AUTO_ASSIGN"] = "1"

from run_api import app
from dashboard.infrastructure.web_server import create_app
from security.domain.entities import Role, User
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.jwt_service import JWTService
from persistence.infrastructure.repository import SQLitePersistenceRepository
from security.infrastructure.encryption_service import PIIEncryptionService


class TestH47FollowupDatetimeHardening(unittest.TestCase):
    """Focused integration tests for Phase H4.7.1 Follow-up Date/Time Hardening."""

    def setUp(self):
        test_method_name = self.id().split('.')[-1]
        self.db_path = os.path.abspath(f"outputs/test_h47_datetime_{test_method_name}.db")
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

        # Setup passwords
        from security.infrastructure.password import PasswordHasher
        pass_hash = PasswordHasher.hash_password("Password@123")

        # Bootstrap Users
        conn = sqlite3.connect(self.db_path)
        try:
            # Doctors
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (200, "doc-uuid-a", "doctor_a@aurascan.ai", pass_hash, "Doctor A", Role.DOCTOR.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (201, "doc-uuid-b", "doctor_b@aurascan.ai", pass_hash, "Doctor B", Role.DOCTOR.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            # Patients
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (300, "pat-uuid-a", "patient_a@aurascan.ai", pass_hash, "Patient A", Role.PATIENT.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            # Patient Demographics
            enc_svc = PIIEncryptionService()
            conn.execute(
                "INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
                ("pat-uuid-a", enc_svc.encrypt("Patient A Name"), enc_svc.encrypt("45"), enc_svc.encrypt("Male"), datetime.datetime.utcnow().isoformat())
            )
            # Assignments: Doctor A is assigned to Patient A
            conn.execute(
                "INSERT INTO doctor_patient_assignments (doctor_id, patient_id, created_at) VALUES (?, ?, ?);",
                (200, "pat-uuid-a", datetime.datetime.utcnow().isoformat())
            )
            conn.commit()
        finally:
            conn.close()

        # Initialize JWT Tokens
        self.jwt_svc = JWTService()
        self.token_doc_a = self.jwt_svc.create_access_token("doc-uuid-a", 200, "doctor_a@aurascan.ai", Role.DOCTOR)
        self.token_doc_b = self.jwt_svc.create_access_token("doc-uuid-b", 201, "doctor_b@aurascan.ai", Role.DOCTOR)
        self.token_pat_a = self.jwt_svc.create_access_token("pat-uuid-a", 300, "patient_a@aurascan.ai", Role.PATIENT)

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

    def test_01_date_only_format_accepted_and_preserved(self):
        """Verify that YYYY-MM-DD is accepted and stored as YYYY-MM-DD."""
        self.flask_client.set_cookie("access_token", self.token_doc_a)
        res = self.flask_client.post(
            "/api/doctor/patients/pat-uuid-a/followups",
            json={"scheduled_date": "2026-12-25", "reason": "Date-only Check"}
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["scheduled_date"], "2026-12-25")

        # Double check database row
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute("SELECT scheduled_date FROM followup_schedules WHERE followup_id = ?;", (data["followup_id"],)).fetchone()
            self.assertEqual(row[0], "2026-12-25")
        finally:
            conn.close()

    def test_02_iso_datetime_naive_accepted_and_normalized(self):
        """Verify that ISO naive datetime is accepted and formatted as YYYY-MM-DDTHH:MM:SS."""
        self.flask_client.set_cookie("access_token", self.token_doc_a)
        res = self.flask_client.post(
            "/api/doctor/patients/pat-uuid-a/followups",
            json={"scheduled_date": "2026-12-25T15:30:00", "reason": "Naive Datetime Check"}
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["scheduled_date"], "2026-12-25T15:30:00")

        # Check with space separator instead of T
        res_space = self.flask_client.post(
            "/api/doctor/patients/pat-uuid-a/followups",
            json={"scheduled_date": "2026-12-25 15:30:00", "reason": "Space Separator Check"}
        )
        self.assertEqual(res_space.status_code, 200)
        data_space = res_space.get_json()
        self.assertEqual(data_space["scheduled_date"], "2026-12-25T15:30:00")

    def test_03_timezone_aware_datetime_converted_to_utc(self):
        """Verify timezone-aware inputs (Z, offsets) are converted to UTC and standardized with Z suffix."""
        self.flask_client.set_cookie("access_token", self.token_doc_a)

        # Test UTC Z format
        res_z = self.flask_client.post(
            "/api/doctor/patients/pat-uuid-a/followups",
            json={"scheduled_date": "2026-12-25T10:00:00Z", "reason": "UTC Z Check"}
        )
        self.assertEqual(res_z.status_code, 200)
        self.assertEqual(res_z.get_json()["scheduled_date"], "2026-12-25T10:00:00Z")

        # Test positive offset (+05:30) -> should subtract 5h30m to get UTC
        res_offset_pos = self.flask_client.post(
            "/api/doctor/patients/pat-uuid-a/followups",
            json={"scheduled_date": "2026-12-25T15:30:00+05:30", "reason": "Positive Offset Check"}
        )
        self.assertEqual(res_offset_pos.status_code, 200)
        self.assertEqual(res_offset_pos.get_json()["scheduled_date"], "2026-12-25T10:00:00Z")

        # Test negative offset (-04:00) -> should add 4h to get UTC
        res_offset_neg = self.flask_client.post(
            "/api/doctor/patients/pat-uuid-a/followups",
            json={"scheduled_date": "2026-12-25T06:00:00-04:00", "reason": "Negative Offset Check"}
        )
        self.assertEqual(res_offset_neg.status_code, 200)
        self.assertEqual(res_offset_neg.get_json()["scheduled_date"], "2026-12-25T10:00:00Z")

    def test_04_invalid_and_malformed_values_rejected(self):
        """Verify that malformed dates, invalid values, empty formats, and slashes are rejected with 400."""
        self.flask_client.set_cookie("access_token", self.token_doc_a)

        bad_formats = [
            "2026/09/15",          # Slash separators (must be rejected)
            "2026-09-15T25:00:00", # Invalid hour (25)
            "2026-13-15",          # Invalid month (13)
            "2026-09-32",          # Invalid day (32)
            "not-a-date",          # Completely malformed
            "",                    # Empty string
            "   ",                 # Whitespace only
        ]

        for val in bad_formats:
            res = self.flask_client.post(
                "/api/doctor/patients/pat-uuid-a/followups",
                json={"scheduled_date": val, "reason": "Should Fail"}
            )
            self.assertEqual(res.status_code, 400, f"Expected value '{val}' to be rejected with 400.")

    def test_05_update_followup_date_hardening(self):
        """Verify that updates also validate and normalize date formats correctly."""
        self.flask_client.set_cookie("access_token", self.token_doc_a)

        # Create initial appointment
        res_create = self.flask_client.post(
            "/api/doctor/patients/pat-uuid-a/followups",
            json={"scheduled_date": "2026-12-25"}
        )
        fid = res_create.get_json()["followup_id"]

        # Update with positive offset
        res_up = self.flask_client.put(
            f"/api/doctor/followups/{fid}",
            json={"scheduled_date": "2026-12-26T12:00:00+02:00"}
        )
        self.assertEqual(res_up.status_code, 200)
        self.assertEqual(res_up.get_json()["scheduled_date"], "2026-12-26T10:00:00Z")

        # Update with invalid format
        res_up_bad = self.flask_client.put(
            f"/api/doctor/followups/{fid}",
            json={"scheduled_date": "2026/12/26"}
        )
        self.assertEqual(res_up_bad.status_code, 400)

    def test_06_unauthorized_rbac_and_idor_barriers_preserved(self):
        """Verify that patients and unassigned doctors cannot create or update follow-ups using the new endpoint logic."""
        # Unassigned doctor B
        self.flask_client.set_cookie("access_token", self.token_doc_b)
        res_unassigned = self.flask_client.post(
            "/api/doctor/patients/pat-uuid-a/followups",
            json={"scheduled_date": "2026-12-25T10:00:00Z"}
        )
        self.assertEqual(res_unassigned.status_code, 403)

        # Patient roles
        self.flask_client.set_cookie("access_token", self.token_pat_a)
        res_patient = self.flask_client.post(
            "/api/doctor/patients/pat-uuid-a/followups",
            json={"scheduled_date": "2026-12-25T10:00:00Z"}
        )
        self.assertEqual(res_patient.status_code, 403)
