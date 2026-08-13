import os
import sqlite3
import datetime
import unittest
from fastapi.testclient import TestClient

os.environ["DB_PATH"] = os.path.abspath("outputs/test_h42_profile.db")
os.environ["DISABLE_TEST_AUTO_ASSIGN"] = "1"

from run_api import app
from dashboard.infrastructure.web_server import create_app
from security.domain.entities import Role, User
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.jwt_service import JWTService
from persistence.infrastructure.repository import SQLitePersistenceRepository
from security.infrastructure.encryption_service import PIIEncryptionService


class TestH42PatientProfile(unittest.TestCase):
    """Focused integration tests for Phase H4.2 Patient Profile Implementation."""

    def setUp(self):
        test_method_name = self.id().split('.')[-1]
        self.db_path = os.path.abspath(f"outputs/test_h42_profile_{test_method_name}.db")
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
            conn.commit()
        finally:
            conn.close()

        # Initialize JWT Tokens
        self.jwt_svc = JWTService()
        self.token_doc_a = self.jwt_svc.create_access_token("doc-uuid-a", 200, "doctor_a@aurascan.ai", Role.DOCTOR)
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

    def test_01_patient_can_retrieve_own_profile_with_decrypted_demographics(self):
        """Verify that a patient can retrieve their own profile with decrypted demographics."""
        self.flask_client.set_cookie("access_token", self.token_pat_a)
        res = self.flask_client.get("/api/patient/profile")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["patient_id"], "pat-uuid-a")
        self.assertEqual(data["name"], "Patient A Name")
        self.assertEqual(data["age"], 45)
        self.assertEqual(data["gender"], "Male")
        self.assertEqual(data["email"], "patient_a@aurascan.ai")

    def test_02_patient_cannot_retrieve_another_patients_profile(self):
        """Verify that Patient A cannot retrieve Patient B's profile demographics."""
        # Querying /api/patient/profile as Patient A returns Patient A's profile since it's derived from session
        self.flask_client.set_cookie("access_token", self.token_pat_a)
        res = self.flask_client.get("/api/patient/profile")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json()["patient_id"], "pat-uuid-a")

        # Patient A is blocked from doctor-only patient profile details for Patient B
        res_doc = self.flask_client.get("/api/doctor/patients/pat-uuid-b")
        self.assertEqual(res_doc.status_code, 403)

    def test_03_client_supplied_patient_id_cannot_bypass_authorization(self):
        """Verify that manipulating the query parameter values does not bypass auth checks."""
        self.flask_client.set_cookie("access_token", self.token_pat_a)
        res = self.flask_client.get("/api/patient/profile?patient_id=pat-uuid-b")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json()["patient_id"], "pat-uuid-a")  # Still returns Patient A's own profile

    def test_04_patient_cannot_modify_another_patients_profile(self):
        """Verify that Patient A cannot modify Patient B's profile details."""
        # Patient A makes profile update which is scoped strictly to self.id
        self.flask_client.set_cookie("access_token", self.token_pat_a)
        res = self.flask_client.put("/api/auth/profile", json={"full_name": "Updated Pat A"})
        self.assertEqual(res.status_code, 200)

        # Patient B remains unchanged
        user_b = self.user_repo.get_by_id(301)
        self.assertEqual(user_b.full_name, "Patient B")

    def test_05_patient_cannot_change_role(self):
        """Verify that role escalation attempts via profile update are rejected/ignored."""
        self.flask_client.set_cookie("access_token", self.token_pat_a)
        res = self.flask_client.put("/api/auth/profile", json={"role": "admin", "role_str": "doctor"})
        self.assertEqual(res.status_code, 200)

        # Role remains PATIENT
        user_a = self.user_repo.get_by_id(300)
        self.assertEqual(user_a.role, Role.PATIENT)

    def test_06_patient_cannot_change_doctor_assignment(self):
        """Verify that patients cannot mutate doctor assignments."""
        self.flask_client.set_cookie("access_token", self.token_pat_a)
        res = self.flask_client.put("/api/auth/profile", json={"doctor_id": 200, "assigned_doctor": 200})
        self.assertEqual(res.status_code, 200)

        # Verify assignments table is empty
        conn = sqlite3.connect(self.db_path)
        try:
            cnt = conn.execute("SELECT COUNT(*) FROM doctor_patient_assignments;").fetchone()[0]
            self.assertEqual(cnt, 0)
        finally:
            conn.close()

    def test_07_patient_cannot_change_patient_id(self):
        """Verify that patient ID / patient_id cannot be changed."""
        self.flask_client.set_cookie("access_token", self.token_pat_a)
        res = self.flask_client.put("/api/auth/profile", json={"patient_id": "new-pat-uuid", "patient_uuid": "new-pat-uuid"})
        self.assertEqual(res.status_code, 200)

        user_a = self.user_repo.get_by_id(300)
        self.assertEqual(user_a.uuid, "pat-uuid-a")

    def test_08_patient_cannot_change_internal_uuid(self):
        """Verify that internal authentication UUID is immutable for the profile update route."""
        self.flask_client.set_cookie("access_token", self.token_pat_a)
        res = self.flask_client.put("/api/auth/profile", json={"uuid": "hacked-uuid"})
        self.assertEqual(res.status_code, 200)

        user_a = self.user_repo.get_by_id(300)
        self.assertEqual(user_a.uuid, "pat-uuid-a")

    def test_09_valid_profile_update_succeeds_where_allowed(self):
        """Verify that updating full_name works and synchronizes the encrypted name."""
        self.flask_client.set_cookie("access_token", self.token_pat_a)
        res = self.flask_client.put("/api/auth/profile", json={"full_name": "New Valid Name"})
        self.assertEqual(res.status_code, 200)

        # Check demographics database sync
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute("SELECT name FROM patients WHERE patient_id = ?;", ("pat-uuid-a",)).fetchone()
            self.assertIsNotNone(row)
            enc_svc = PIIEncryptionService()
            self.assertEqual(enc_svc.decrypt(row[0]), "New Valid Name")
        finally:
            conn.close()

    def test_10_invalid_email_is_rejected(self):
        """Verify that malformed or markdown/mailto links inside updated email are rejected."""
        self.flask_client.set_cookie("access_token", self.token_pat_a)
        res1 = self.flask_client.put("/api/auth/profile", json={"email": "invalid_email_no_at.com"})
        self.assertEqual(res1.status_code, 400)

        res2 = self.flask_client.put("/api/auth/profile", json={"email": "mailto:test@aurascan.ai"})
        self.assertEqual(res2.status_code, 400)

        res3 = self.flask_client.put("/api/auth/profile", json={"email": "[markdown](mailto:test@aurascan.ai)"})
        self.assertEqual(res3.status_code, 400)

    def test_11_email_normalization_works(self):
        """Verify that email whitespace and uppercase inputs are normalized correctly."""
        self.flask_client.set_cookie("access_token", self.token_pat_a)
        res = self.flask_client.put("/api/auth/profile", json={"email": "  NORMALIZED@AuraScan.AI  "})
        self.assertEqual(res.status_code, 200)

        user_a = self.user_repo.get_by_id(300)
        self.assertEqual(user_a.email, "normalized@aurascan.ai")

    def test_12_email_change_resets_verification_state(self):
        """Verify that changing the email address resets the user verification status to False."""
        # Initially verified
        user_a = self.user_repo.get_by_id(300)
        self.assertTrue(user_a.is_verified)

        self.flask_client.set_cookie("access_token", self.token_pat_a)
        res = self.flask_client.put("/api/auth/profile", json={"email": "new_email@aurascan.ai"})
        self.assertEqual(res.status_code, 200)

        user_a_updated = self.user_repo.get_by_id(300)
        self.assertFalse(user_a_updated.is_verified)

    def test_13_duplicate_email_is_handled_safely(self):
        """Verify that updating the email to one that is already registered is rejected."""
        self.flask_client.set_cookie("access_token", self.token_pat_a)
        res = self.flask_client.put("/api/auth/profile", json={"email": "patient_b@aurascan.ai"})
        self.assertEqual(res.status_code, 400)
        self.assertIn("already in use", res.get_json()["error"])

    def test_14_encrypted_demographics_are_decrypted_only_for_authorized_response(self):
        """Verify demographic details are decrypted for authorized response, not exposed raw."""
        self.flask_client.set_cookie("access_token", self.token_pat_a)
        res = self.flask_client.get("/api/patient/profile")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["name"], "Patient A Name")
        self.assertEqual(data["gender"], "Male")

    def test_15_raw_ciphertext_is_not_returned(self):
        """Verify that raw database ciphertext is never leaked in the response."""
        self.flask_client.set_cookie("access_token", self.token_pat_a)
        res = self.flask_client.get("/api/patient/profile")
        self.assertEqual(res.status_code, 200)

        # None of the values should be encrypted prefixes
        for val in res.get_json().values():
            self.assertFalse(str(val).startswith("enc:v1:"))

    def test_16_unauthenticated_access_is_denied(self):
        """Verify that requests without access token are rejected with 401."""
        res = self.flask_client.get("/api/patient/profile")
        self.assertEqual(res.status_code, 401)

    def test_17_error_responses_do_not_expose_internal_details(self):
        """Verify that error responses contain only generic safe error messages."""
        self.flask_client.set_cookie("access_token", self.token_pat_a)
        original_db = self.flask_app.config["DB_PATH"]
        self.flask_app.config["DB_PATH"] = "invalid_directory/nonexistent_file.db"
        try:
            res = self.flask_client.get("/api/patient/profile")
            self.assertEqual(res.status_code, 500)
            self.assertEqual(res.get_json()["error"], "Internal server error")
        finally:
            self.flask_app.config["DB_PATH"] = original_db
