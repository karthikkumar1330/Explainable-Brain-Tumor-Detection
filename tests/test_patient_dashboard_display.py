import unittest
import os
import tempfile
import sqlite3
import datetime
from flask import url_for
from dashboard.infrastructure.web_server import create_app
from security.domain.entities import Role, User
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.jwt_service import JWTService
from security.infrastructure.password import PasswordHasher
from persistence.infrastructure.repository import SQLitePersistenceRepository

class TestPatientDashboardDisplay(unittest.TestCase):
    """Integration test suite to verify patient Referring Physician display logic on the dashboard."""

    def setUp(self):
        # Create a temporary isolated database
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        self.app = create_app(db_path=self.db_path)
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

        # Initialize tables
        self.persistence_repo = SQLitePersistenceRepository(db_path=self.db_path)
        self.persistence_repo.initialize_db()

        self.user_repo = SQLiteUserRepository(db_path=self.db_path)
        self.user_repo.initialize_security_tables()

        # Bootstrap doctor, assigned patient, and unassigned patient
        pass_hash = PasswordHasher.hash_password("Password@123")

        self.doctor = User(
            id=None,
            uuid="doc-uuid-alice",
            email="doctor_alice@hospital.org",
            password_hash=pass_hash,
            full_name="Dr. Alice Smith",
            role=Role.DOCTOR,
            is_verified=True,
            is_active=True
        )
        self.user_repo.create_user(self.doctor)
        # Retrieve initialized auto-increment ID
        self.doctor = self.user_repo.get_by_email("doctor_alice@hospital.org")

        self.patient_assigned = User(
            id=None,
            uuid="patient-assigned-uuid",
            email="karthik@health.org",
            password_hash=pass_hash,
            full_name="Karthik Kumar",
            role=Role.PATIENT,
            is_verified=True,
            is_active=True
        )
        self.user_repo.create_user(self.patient_assigned)
        self.patient_assigned = self.user_repo.get_by_email("karthik@health.org")

        self.patient_unassigned = User(
            id=None,
            uuid="patient-unassigned-uuid",
            email="unassigned@health.org",
            password_hash=pass_hash,
            full_name="John Doe",
            role=Role.PATIENT,
            is_verified=True,
            is_active=True
        )
        self.user_repo.create_user(self.patient_unassigned)
        self.patient_unassigned = self.user_repo.get_by_email("unassigned@health.org")

        # Set up demographics and assignment in database
        conn = sqlite3.connect(self.db_path)
        now_str = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        try:
            from security.infrastructure.encryption_service import PIIEncryptionService
            enc = PIIEncryptionService()

            # Demographics for assigned patient
            conn.execute(
                "INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
                (self.patient_assigned.uuid, enc.encrypt(self.patient_assigned.full_name), enc.encrypt("42"), enc.encrypt("Male"), now_str)
            )
            # Demographics for unassigned patient
            conn.execute(
                "INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
                (self.patient_unassigned.uuid, enc.encrypt(self.patient_unassigned.full_name), enc.encrypt("30"), enc.encrypt("Female"), now_str)
            )
            # Explicit Assignment
            conn.execute(
                "INSERT INTO doctor_patient_assignments (doctor_id, patient_id, created_at) VALUES (?, ?, ?);",
                (self.doctor.id, self.patient_assigned.uuid, now_str)
            )
            conn.commit()
        finally:
            conn.close()

        # Set up JWT tokens
        self.jwt_svc = JWTService()
        self.token_assigned = self.jwt_svc.create_access_token(
            user_uuid=self.patient_assigned.uuid,
            user_id=self.patient_assigned.id,
            email=self.patient_assigned.email,
            role=self.patient_assigned.role
        )
        self.token_unassigned = self.jwt_svc.create_access_token(
            user_uuid=self.patient_unassigned.uuid,
            user_id=self.patient_unassigned.id,
            email=self.patient_unassigned.email,
            role=self.patient_unassigned.role
        )

    def tearDown(self):
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def test_assigned_patient_receives_doctor_details(self):
        """Verify that an assigned patient gets the correct assigned doctor name and email."""
        self.client.set_cookie("access_token", self.token_assigned)
        res = self.client.get("/api/patient/profile")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        self.assertIn("assigned_doctor", data)
        self.assertIsNotNone(data["assigned_doctor"])
        self.assertEqual(data["assigned_doctor"]["name"], "Dr. Alice Smith")
        self.assertEqual(data["assigned_doctor"]["email"], "doctor_alice@hospital.org")

    def test_unassigned_patient_receives_none_doctor(self):
        """Verify that an unassigned patient receives None (null) for the doctor."""
        self.client.set_cookie("access_token", self.token_unassigned)
        res = self.client.get("/api/patient/profile")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        self.assertIn("assigned_doctor", data)
        self.assertIsNone(data["assigned_doctor"])

    def test_patient_profile_isolation_no_idor(self):
        """Verify that the profile endpoint strictly returns current authenticated patient's profile."""
        # When unassigned patient calls profile, they get their own details, not the assigned patient's
        self.client.set_cookie("access_token", self.token_unassigned)
        res = self.client.get("/api/patient/profile")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        self.assertEqual(data["patient_id"], self.patient_unassigned.uuid)
        self.assertEqual(data["name"], "John Doe")
        self.assertIsNone(data["assigned_doctor"])

if __name__ == "__main__":
    unittest.main()
