import os
import unittest
import sqlite3
import datetime
import tempfile
from fastapi.testclient import TestClient

# Isolate database path
db_fd, db_path = tempfile.mkstemp(suffix=".db")
os.close(db_fd)
os.environ["DB_PATH"] = db_path

from run_api import app
from persistence.infrastructure.repository import SQLitePersistenceRepository
from security.domain.entities import Role, User
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.jwt_service import JWTService
from security.infrastructure.password import PasswordHasher
from security.application.authorization_service import AuthorizationService

class TestPatientOnboardingHardening(unittest.TestCase):
    """Hardening test cases to prove explicit onboarding, demographic creation, and patient isolation."""

    def setUp(self):
        self.db_path = db_path
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

        # Make sure default paths match in routes
        import api.infrastructure.routes as api_routes
        import api.routes.auth_routes as auth_routes
        api_routes.DEFAULT_DB_PATH = self.db_path
        auth_routes.DEFAULT_DB_PATH = self.db_path

        # Initialize DB schemas
        self.persistence_repo = SQLitePersistenceRepository(db_path=self.db_path)
        self.persistence_repo.initialize_db()

        self.user_repo = SQLiteUserRepository(db_path=self.db_path)
        self.user_repo.initialize_security_tables()

        # Seed doctors and patient users
        pass_hash = PasswordHasher.hash_password("Password@123")
        conn = sqlite3.connect(self.db_path)
        try:
            # Drop any triggers
            conn.execute("DROP TRIGGER IF EXISTS auto_assign_doctor_on_insert;")
            conn.execute("DROP TRIGGER IF EXISTS auto_assign_patient_on_insert;")
            conn.execute("DELETE FROM doctor_patient_assignments;")

            # Insert doctors
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (401, "doc-a-uuid", "doctor_a@hospital.org", pass_hash, "Dr. Alice", Role.DOCTOR.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (402, "doc-b-uuid", "doctor_b@hospital.org", pass_hash, "Dr. Bob", Role.DOCTOR.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            # Insert registered patient user who doesn't have demographics in patients table yet
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (501, "pat-user-uuid", "patient_user@hospital.org", pass_hash, "Jane Doe", Role.PATIENT.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            # Insert Karthik Kumar's account specifically
            self.karthik_uuid = "67d5923b-9680-4af5-8e02-70af8c09b1de"
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (2, self.karthik_uuid, "karthikkemisatty@gmail.com", pass_hash, "Karthik Kumar", Role.PATIENT.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            conn.commit()
        finally:
            conn.close()

        # Generate tokens
        self.doc_a = self.user_repo.get_by_email("doctor_a@hospital.org")
        self.doc_b = self.user_repo.get_by_email("doctor_b@hospital.org")

        self.jwt_svc = JWTService()
        self.token_doc_a = self.jwt_svc.create_access_token(self.doc_a.uuid, self.doc_a.id, self.doc_a.email, Role.DOCTOR)
        self.token_doc_b = self.jwt_svc.create_access_token(self.doc_b.uuid, self.doc_b.id, self.doc_b.email, Role.DOCTOR)

        self.client = TestClient(app)

    def tearDown(self):
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def test_01_onboarding_creates_correct_demographics(self):
        """Onboard a new patient user and verify they have correct decrypted demographic values."""
        headers = {"Authorization": f"Bearer {self.token_doc_a}"}
        # Explicit onboarding assign patient to Doctor A with custom age/gender
        res = self.client.post(
            "/api/doctor/assign-patient",
            params={"patient_id": "pat-user-uuid", "age": "45", "gender": "Female"},
            headers=headers
        )
        self.assertEqual(res.status_code, 200)

        # Retrieve and verify demographics in database
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute("SELECT name, age, gender FROM patients WHERE patient_id = 'pat-user-uuid';").fetchone()
            self.assertIsNotNone(row)

            # Decrypt fields
            from security.infrastructure.encryption_service import PIIEncryptionService
            encryption_service = PIIEncryptionService()
            dec_name = encryption_service.decrypt(row[0])
            dec_age = encryption_service.decrypt(row[1])
            dec_gender = encryption_service.decrypt(row[2])

            self.assertEqual(dec_name, "Jane Doe")
            self.assertEqual(dec_age, "45")
            self.assertEqual(dec_gender, "Female")
        finally:
            conn.close()

    def test_02_explicit_assignment_idempotent(self):
        """Repeated onboarding/assignment requests should be idempotent and not create duplicate assignment rows."""
        headers = {"Authorization": f"Bearer {self.token_doc_a}"}

        # Onboard first time
        res1 = self.client.post("/api/doctor/assign-patient", params={"patient_id": "pat-user-uuid", "age": "32", "gender": "Male"}, headers=headers)
        self.assertEqual(res1.status_code, 200)

        # Onboard second time
        res2 = self.client.post("/api/doctor/assign-patient", params={"patient_id": "pat-user-uuid", "age": "32", "gender": "Male"}, headers=headers)
        self.assertEqual(res2.status_code, 200)

        # Verify exactly one assignment row exists
        conn = sqlite3.connect(self.db_path)
        try:
            count = conn.execute("SELECT COUNT(*) FROM doctor_patient_assignments WHERE doctor_id = 401 AND patient_id = 'pat-user-uuid';").fetchone()[0]
            self.assertEqual(count, 1)
        finally:
            conn.close()

    def test_03_unassigned_doctor_denied_access(self):
        """Unassigned doctor must receive HTTP 403 when trying to retrieve patient profile or generate report."""
        headers = {"Authorization": f"Bearer {self.token_doc_b}"}

        # Attempt to access profile
        res_profile = self.client.get("/api/doctor/patients/pat-user-uuid", headers=headers)
        self.assertEqual(res_profile.status_code, 403)

        # Attempt report generation
        payload = {
            "ref_physician": "Dr. Bob",
            "name": "Jane Doe",
            "patient_id": "pat-user-uuid",
            "age": 30,
            "gender": "Female",
            "pixel_spacing_mm": 1.0
        }
        res_report = self.client.post("/api/report", params={"filepath": "scan.png"}, json=payload, headers=headers)
        self.assertEqual(res_report.status_code, 403)

    def test_04_assigned_doctor_allowed_access(self):
        """Assigned doctor must receive HTTP 200 and report generation must succeed."""
        headers = {"Authorization": f"Bearer {self.token_doc_a}"}

        # Onboard/assign
        res_assign = self.client.post("/api/doctor/assign-patient", params={"patient_id": "pat-user-uuid", "age": "28", "gender": "Male"}, headers=headers)
        self.assertEqual(res_assign.status_code, 200)

        # Check profile
        res_profile = self.client.get("/api/doctor/patients/pat-user-uuid", headers=headers)
        self.assertEqual(res_profile.status_code, 200)

    def test_05_karthik_onboarding_and_isolation(self):
        """Verify that onboarding Karthik Kumar allows Doctor A access, but Doctor B remains blocked with 403."""
        # 1. Initially Doctor A and B are blocked
        headers_a = {"Authorization": f"Bearer {self.token_doc_a}"}
        headers_b = {"Authorization": f"Bearer {self.token_doc_b}"}

        self.assertEqual(self.client.get(f"/api/doctor/patients/{self.karthik_uuid}", headers=headers_a).status_code, 403)
        self.assertEqual(self.client.get(f"/api/doctor/patients/{self.karthik_uuid}", headers=headers_b).status_code, 403)

        # 2. Onboard Karthik to Doctor A
        res_assign = self.client.post(
            "/api/doctor/assign-patient",
            params={"patient_id": self.karthik_uuid, "age": "50", "gender": "Male"},
            headers=headers_a
        )
        self.assertEqual(res_assign.status_code, 200)

        # 3. Doctor A gets 200, Doctor B still gets 403
        self.assertEqual(self.client.get(f"/api/doctor/patients/{self.karthik_uuid}", headers=headers_a).status_code, 200)
        self.assertEqual(self.client.get(f"/api/doctor/patients/{self.karthik_uuid}", headers=headers_b).status_code, 403)

        # 4. Assert demographics decrypted correctly
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute("SELECT name, age, gender FROM patients WHERE patient_id = ?;", (self.karthik_uuid,)).fetchone()
            self.assertIsNotNone(row)
            from security.infrastructure.encryption_service import PIIEncryptionService
            encryption_service = PIIEncryptionService()
            self.assertEqual(encryption_service.decrypt(row[0]), "Karthik Kumar")
            self.assertEqual(encryption_service.decrypt(row[1]), "50")
            self.assertEqual(encryption_service.decrypt(row[2]), "Male")
        finally:
            conn.close()

if __name__ == "__main__":
    unittest.main()
