import os
import unittest
import sqlite3
import datetime
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Isolate database path for regression test
TEST_DB_PATH = os.path.abspath("outputs/test_doctor_assigned_patients.db")
os.environ["DB_PATH"] = TEST_DB_PATH

import api.infrastructure.routes as api_routes
import api.routes.auth_routes as auth_routes
api_routes.DEFAULT_DB_PATH = TEST_DB_PATH
auth_routes.DEFAULT_DB_PATH = TEST_DB_PATH

from run_api import app
from dashboard.infrastructure.web_server import create_app as create_flask_app
from persistence.infrastructure.repository import SQLitePersistenceRepository
from security.domain.entities import Role, User
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.jwt_service import JWTService
from security.infrastructure.password import PasswordHasher
from security.infrastructure.encryption_service import PIIEncryptionService
from security.application.config import app_config


class TestDoctorAssignedPatientsRegression(unittest.TestCase):
    """Full regression suite covering doctor assigned patients retrieval, isolation, encryption, and proxying."""

    @classmethod
    def setUpClass(cls):
        cls.db_path = TEST_DB_PATH
        if os.path.exists(cls.db_path):
            try:
                os.remove(cls.db_path)
            except Exception:
                pass

        api_routes.DEFAULT_DB_PATH = cls.db_path
        auth_routes.DEFAULT_DB_PATH = cls.db_path

        cls.persistence_repo = SQLitePersistenceRepository(db_path=cls.db_path)
        cls.persistence_repo.initialize_db()

        cls.user_repo = SQLiteUserRepository(db_path=cls.db_path)
        cls.user_repo.initialize_security_tables()

        # Disable auto-assign triggers for isolated testing
        conn = sqlite3.connect(cls.db_path)
        try:
            conn.execute("DROP TRIGGER IF EXISTS auto_assign_doctor_on_insert;")
            conn.execute("DROP TRIGGER IF EXISTS auto_assign_doctor_on_patient_insert;")
            conn.execute("DELETE FROM doctor_patient_assignments;")
            conn.commit()
        finally:
            conn.close()

        cls.hasher = PasswordHasher()
        cls.jwt_service = JWTService()
        cls.enc_service = PIIEncryptionService()

        now_iso = datetime.datetime.utcnow().isoformat()
        pass_hash = cls.hasher.hash_password("DocPass123!")

        cls.pat_1_id = "pat-uuid-001"
        cls.pat_1_name = "Patient Alpha"
        cls.pat_1_age = "42"
        cls.pat_1_gender = "Male"

        cls.pat_2_id = "pat-uuid-002"
        cls.pat_2_name = "Patient Beta"
        cls.pat_2_age = "55"
        cls.pat_2_gender = "Female"

        cls.pat_unassigned_id = "pat-uuid-unassigned"

        conn = sqlite3.connect(cls.db_path)
        try:
            # Seed Doctor A (ID 10)
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (10, "doc-a-uuid", "doctor_a@hospital.org", pass_hash, "Dr. Alice Smith", Role.DOCTOR.value, now_iso, now_iso)
            )
            # Seed Doctor B (ID 20)
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (20, "doc-b-uuid", "doctor_b@hospital.org", pass_hash, "Dr. Bob Jones", Role.DOCTOR.value, now_iso, now_iso)
            )
            # Seed Patient User (ID 30)
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (30, "pat-user-uuid", "patient_c@hospital.org", pass_hash, "Charlie Brown", Role.PATIENT.value, now_iso, now_iso)
            )
            # Seed Patient Alpha (ID 40)
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (40, cls.pat_1_id, "alpha@example.com", pass_hash, cls.pat_1_name, Role.PATIENT.value, now_iso, now_iso)
            )
            # Seed Patient Beta (ID 41)
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (41, cls.pat_2_id, "beta@example.com", pass_hash, cls.pat_2_name, Role.PATIENT.value, now_iso, now_iso)
            )
            # Seed Unassigned Patient (ID 42)
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (42, cls.pat_unassigned_id, "unassigned@example.com", pass_hash, "Unassigned Patient", Role.PATIENT.value, now_iso, now_iso)
            )
            conn.commit()
        finally:
            conn.close()

        cls.doc_a = cls.user_repo.get_by_id(10)
        cls.doc_b = cls.user_repo.get_by_id(20)
        cls.pat_user = cls.user_repo.get_by_id(30)

        # Insert encrypted records into patients table
        now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        conn = sqlite3.connect(cls.db_path)
        try:
            conn.execute(
                "INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?)",
                (cls.pat_1_id, cls.enc_service.encrypt(cls.pat_1_name), cls.enc_service.encrypt(cls.pat_1_age), cls.enc_service.encrypt(cls.pat_1_gender), now)
            )
            conn.execute(
                "INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?)",
                (cls.pat_2_id, cls.enc_service.encrypt(cls.pat_2_name), cls.enc_service.encrypt(cls.pat_2_age), cls.enc_service.encrypt(cls.pat_2_gender), now)
            )
            conn.execute(
                "INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?)",
                (cls.pat_unassigned_id, cls.enc_service.encrypt("Unassigned Patient"), cls.enc_service.encrypt("60"), cls.enc_service.encrypt("Other"), now)
            )

            # Assign pat_1 and pat_2 to Doctor A
            conn.execute(
                "INSERT INTO doctor_patient_assignments (doctor_id, patient_id, created_at) VALUES (?, ?, ?)",
                (cls.doc_a.id, cls.pat_1_id, now)
            )
            conn.execute(
                "INSERT INTO doctor_patient_assignments (doctor_id, patient_id, created_at) VALUES (?, ?, ?)",
                (cls.doc_a.id, cls.pat_2_id, now)
            )
            conn.commit()
        finally:
            conn.close()

        cls.client = TestClient(app)
        cls.flask_app = create_flask_app(db_path=cls.db_path)
        cls.flask_client = cls.flask_app.test_client()

    def _token_for(self, user):
        return self.jwt_service.create_access_token(
            user_uuid=user.uuid,
            user_id=user.id,
            email=user.email,
            role=user.role
        )

    def test_01_doctor_with_two_assignments_receives_exactly_those(self):
        """TEST 1: Doctor with two assignments receives exactly those assigned patients."""
        token = self._token_for(self.doc_a)
        resp = self.client.get("/api/doctor/assigned-patients", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data), 2)

    def test_02_doctor_receives_correct_patient_ids(self):
        """TEST 2: Doctor receives correct patient IDs."""
        token = self._token_for(self.doc_a)
        resp = self.client.get("/api/doctor/assigned-patients", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        patient_ids = {p["patient_id"] for p in data}
        self.assertEqual(patient_ids, {self.pat_1_id, self.pat_2_id})

    def test_03_doctor_cannot_receive_unassigned_patients(self):
        """TEST 3: Doctor cannot receive unassigned patients."""
        token = self._token_for(self.doc_a)
        resp = self.client.get("/api/doctor/assigned-patients", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        patient_ids = [p["patient_id"] for p in data]
        self.assertNotIn(self.pat_unassigned_id, patient_ids)

    def test_04_second_doctor_cannot_see_first_doctors_assigned_patients(self):
        """TEST 4: Second doctor cannot see first doctor's assigned patients."""
        token = self._token_for(self.doc_b)
        resp = self.client.get("/api/doctor/assigned-patients", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data), 0, "Doctor B has no assignments and must receive empty list")

    def test_05_patient_cannot_call_doctor_assigned_patients_endpoint(self):
        """TEST 5: Patient cannot call doctor assigned-patients endpoint (must receive 403)."""
        token = self._token_for(self.pat_user)
        resp = self.client.get("/api/doctor/assigned-patients", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(resp.status_code, 403)

    def test_06_unauthenticated_request_is_rejected(self):
        """TEST 6: Unauthenticated request is rejected (must receive 401)."""
        resp = self.client.get("/api/doctor/assigned-patients")
        self.assertEqual(resp.status_code, 401)

    def test_07_flask_proxy_correctly_forwards_authentication(self):
        """TEST 7: Flask proxy correctly forwards authentication to FastAPI backend."""
        token = self._token_for(self.doc_a)

        # Mock the requests.get inside proxy to verify headers and response pass-through
        with patch("requests.get") as mock_get:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.content = b'[{"patient_id":"pat-uuid-001","name":"Patient Alpha","email":"alpha@example.com"}]'
            mock_get.return_value = mock_resp

            resp = self.flask_client.get(
                "/api/doctor/assigned-patients",
                headers={"Authorization": f"Bearer {token}"}
            )
            self.assertEqual(resp.status_code, 200)
            mock_get.assert_called_once()
            called_headers = mock_get.call_args[1].get("headers", {})
            self.assertIn("Authorization", called_headers)
            self.assertEqual(called_headers["Authorization"], f"Bearer {token}")

    def test_08_frontend_response_parsing_matches_backend_response_structure(self):
        """TEST 8: Frontend response parsing matches backend response structure (contract test)."""
        token = self._token_for(self.doc_a)
        resp = self.client.get("/api/doctor/assigned-patients", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIsInstance(data, list)
        for item in data:
            self.assertIn("patient_id", item)
            self.assertIn("name", item)
            self.assertIn("age", item)
            self.assertIn("gender", item)
            self.assertIn("email", item)
            # Verify JS template string: `${p.name} (${p.email})`
            label = f"{item['name']} ({item['email']})"
            self.assertTrue(len(label) > 5)

    def test_09_encrypted_patient_fields_are_decrypted_properly(self):
        """TEST 9: Encrypted patient fields are handled correctly according to existing architecture."""
        token = self._token_for(self.doc_a)
        resp = self.client.get("/api/doctor/assigned-patients", headers={"Authorization": f"Bearer {token}"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        alpha = next(p for p in data if p["patient_id"] == self.pat_1_id)
        # Database stores 'enc:v1:...', endpoint must return decrypted cleartext
        self.assertEqual(alpha["name"], self.pat_1_name)
        self.assertEqual(alpha["age"], self.pat_1_age)
        self.assertEqual(alpha["gender"], self.pat_1_gender)
        self.assertFalse(alpha["name"].startswith("enc:v1:"))

    def test_10_existing_report_generation_patient_selection_authorization(self):
        """TEST 10: Existing report-generation patient selection still respects authorization."""
        from security.application.authorization_service import AuthorizationService
        auth_service = AuthorizationService(db_path=self.db_path)

        # Doctor A CAN access assigned patients pat_1 and pat_2
        self.assertTrue(auth_service.can_access_patient(self.doc_a, self.pat_1_id))
        self.assertTrue(auth_service.can_access_patient(self.doc_a, self.pat_2_id))

        # Doctor A CANNOT access unassigned patient
        self.assertFalse(auth_service.can_access_patient(self.doc_a, self.pat_unassigned_id))

        # Doctor B CANNOT access Doctor A's patients
        self.assertFalse(auth_service.can_access_patient(self.doc_b, self.pat_1_id))
        self.assertFalse(auth_service.can_access_patient(self.doc_b, self.pat_2_id))


if __name__ == "__main__":
    unittest.main()
