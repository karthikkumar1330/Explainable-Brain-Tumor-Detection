import os
import unittest
import sqlite3
import datetime
from fastapi.testclient import TestClient

# Use a separate test database
os.environ["DB_PATH"] = os.path.abspath("outputs/test_new_patient_auth_regression.db")

from run_api import app
from security.domain.entities import Role, User
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.jwt_service import JWTService
from security.infrastructure.password import PasswordHasher
from persistence.infrastructure.repository import SQLitePersistenceRepository

class TestNewPatientAuthRegression(unittest.TestCase):
    """Regression tests verifying doctor access control and auto-assignment for new patient records."""

    def setUp(self):
        os.environ["DISABLE_TEST_AUTO_ASSIGN"] = "1"
        self.db_path = os.environ.get("DB_PATH", "outputs/test_new_patient_auth_regression.db")
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

        # Make sure default paths match
        import api.infrastructure.routes as api_routes
        import api.routes.auth_routes as auth_routes
        api_routes.DEFAULT_DB_PATH = self.db_path
        auth_routes.DEFAULT_DB_PATH = self.db_path

        # Initialize DB
        self.persistence_repo = SQLitePersistenceRepository(db_path=self.db_path)
        self.persistence_repo.initialize_db()

        # Drop the auto-assignment triggers to ensure clean testing of our application logic
        conn_drop = sqlite3.connect(self.db_path)
        try:
            conn_drop.execute("DROP TRIGGER IF EXISTS auto_assign_doctor_on_insert;")
            conn_drop.execute("DROP TRIGGER IF EXISTS auto_assign_patient_on_insert;")
            conn_drop.execute("DELETE FROM doctor_patient_assignments;")
            conn_drop.commit()
        finally:
            conn_drop.close()

        self.fastapi_client_ctx = TestClient(app)
        self.client = self.fastapi_client_ctx.__enter__()

        # Bootstrap users
        self.user_repo = SQLiteUserRepository(db_path=self.db_path)
        self.user_repo.initialize_security_tables()
        self.admin_user = self.user_repo.bootstrap_admin()

        pass_hash = PasswordHasher.hash_password("Doctor@123")
        conn = sqlite3.connect(self.db_path)
        try:
            # Doctor A
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, is_verified, is_active, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, 1, 1, ?, ?);",
                (301, "doc-uuid-a", "doctor_a@aurascan.ai", pass_hash, "Dr. Alice", Role.DOCTOR.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            # Doctor B
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, is_verified, is_active, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, 1, 1, ?, ?);",
                (302, "doc-uuid-b", "doctor_b@aurascan.ai", pass_hash, "Dr. Bob", Role.DOCTOR.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            # Patient User C
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, is_verified, is_active, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, 1, 1, ?, ?);",
                (303, "pat-new-999", "patient_c@aurascan.ai", pass_hash, "Patient C", Role.PATIENT.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            conn.commit()
        finally:
            conn.close()

        self.doc_a = self.user_repo.get_by_id(301)
        self.doc_b = self.user_repo.get_by_id(302)

        jwt_svc = JWTService()
        self.token_doc_a = jwt_svc.create_access_token(
            user_uuid=self.doc_a.uuid, user_id=self.doc_a.id, email=self.doc_a.email, role=Role.DOCTOR
        )
        self.token_doc_b = jwt_svc.create_access_token(
            user_uuid=self.doc_b.uuid, user_id=self.doc_b.id, email=self.doc_b.email, role=Role.DOCTOR
        )

        # Copy sample scan to uploads
        self.scan_file = "uploads/regression_bob_scan.png"
        os.makedirs("uploads", exist_ok=True)
        import shutil
        if os.path.exists("imgs/unext.png"):
            shutil.copy("imgs/unext.png", self.scan_file)
        else:
            with open(self.scan_file, "wb") as f:
                f.write(b"fake image bytes")

    def tearDown(self):
        self.fastapi_client_ctx.__exit__(None, None, None)
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass
        if os.path.exists(self.scan_file):
            try:
                os.remove(self.scan_file)
            except Exception:
                pass

    def test_new_patient_report_generation_and_authorization_flow(self):
        """Verify that Doctor A can generate a report for a brand new patient,
        gets auto-assigned to that patient, and Doctor B is forbidden from accessing it.
        """
        new_patient_id = "pat-new-999"
        headers_a = {"Authorization": f"Bearer {self.token_doc_a}"}

        # 0. Doctor A explicitly assigns/onboards the new patient
        res_assign = self.client.post(
            f"/api/doctor/assign-patient?patient_id={new_patient_id}",
            headers=headers_a
        )
        self.assertEqual(res_assign.status_code, 200)

        payload = {
            "ref_physician": "Dr. Alice",
            "name": "Brand New Patient",
            "patient_id": new_patient_id,
            "age": 42,
            "gender": "Male",
            "pixel_spacing_mm": 1.0
        }

        # 1. Doctor A initiates diagnostic pipeline for new patient
        res = self.client.post(
            f"/api/report?filepath={self.scan_file}",
            json=payload,
            headers=headers_a
        )
        self.assertEqual(res.status_code, 200, f"Failed to generate report for new patient: {res.text}")
        report_data = res.json()
        report_db_id = report_data["report_id"]

        # 2. Check that the patient is assigned to Doctor A in database
        conn = sqlite3.connect(self.db_path)
        try:
            assignment = conn.execute(
                "SELECT 1 FROM doctor_patient_assignments WHERE doctor_id = 301 AND patient_id = ?;",
                (new_patient_id,)
            ).fetchone()
            self.assertIsNotNone(assignment, "New patient was not assigned to Doctor A.")

            # Check that Doctor B is NOT assigned
            assignment_b = conn.execute(
                "SELECT 1 FROM doctor_patient_assignments WHERE doctor_id = 302 AND patient_id = ?;",
                (new_patient_id,)
            ).fetchone()
            self.assertIsNone(assignment_b, "New patient was incorrectly assigned to Doctor B.")
        finally:
            conn.close()

        # 3. Doctor A (assigned) requesting report detail -> 200
        res_detail_a = self.client.get(
            f"/api/report/{report_db_id}/json",
            headers=headers_a
        )
        self.assertEqual(res_detail_a.status_code, 200)

        # 4. Doctor B (unassigned) requesting report detail -> 403 Forbidden
        headers_b = {"Authorization": f"Bearer {self.token_doc_b}"}
        res_detail_b = self.client.get(
            f"/api/report/{report_db_id}/json",
            headers=headers_b
        )
        self.assertEqual(res_detail_b.status_code, 403)
        self.assertEqual(res_detail_b.json()["detail"], "Access denied to patient report.")
