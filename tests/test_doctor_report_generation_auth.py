import os
import unittest
import sqlite3
import datetime
import json
import numpy as np
import cv2
from fastapi.testclient import TestClient

os.environ["DB_PATH"] = os.path.abspath("outputs/test_doctor_report_auth.db")

from run_api import app
from security.domain.entities import Role, User
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.jwt_service import JWTService
from security.infrastructure.password import PasswordHasher
from persistence.infrastructure.repository import SQLitePersistenceRepository
from security.application.authorization_service import AuthorizationService


class TestDoctorReportGenerationAuth(unittest.TestCase):
    """Focused regression tests for Doctor Report Generation authorization boundaries (H3 reinforcement)."""

    def _create_mock_brain_mri_slice(self):
        img = np.zeros((224, 224), dtype=np.uint8)
        # Draw a brain-like shape
        cv2.circle(img, (112, 112), 80, 200, -1)
        # Add a tumor
        cv2.circle(img, (130, 130), 20, 255, -1)
        _, buf = cv2.imencode(".png", img)
        return buf.tobytes()

    def setUp(self):
        os.environ["DISABLE_TEST_AUTO_ASSIGN"] = "1"
        env_db = os.environ.get("DB_PATH")
        if env_db and "clinical_reports.db" not in env_db:
            self.db_path = os.path.abspath(env_db)
        else:
            self.db_path = os.path.abspath("outputs/test_doctor_report_auth.db")

        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

        import api.infrastructure.routes as api_routes
        import api.routes.auth_routes as auth_routes
        api_routes.DEFAULT_DB_PATH = self.db_path
        auth_routes.DEFAULT_DB_PATH = self.db_path

        # Initialize DB
        self.persistence_repo = SQLitePersistenceRepository(db_path=self.db_path)
        self.persistence_repo.initialize_db()

        # Drop the auto-assignment triggers to ensure strict clean testing
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
                "INSERT INTO users (uuid, email, password_hash, full_name, role, is_verified, is_active, created_at, updated_at) VALUES (?, ?, ?, ?, ?, 1, 1, ?, ?);",
                ("doc-uuid-a", "doctor_a@aurascan.ai", pass_hash, "Dr. Alice", Role.DOCTOR.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            # Doctor B
            conn.execute(
                "INSERT INTO users (uuid, email, password_hash, full_name, role, is_verified, is_active, created_at, updated_at) VALUES (?, ?, ?, ?, ?, 1, 1, ?, ?);",
                ("doc-uuid-b", "doctor_b@aurascan.ai", pass_hash, "Dr. Bob", Role.DOCTOR.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            # Patient User 1
            conn.execute(
                "INSERT INTO users (uuid, email, password_hash, full_name, role, is_verified, is_active, created_at, updated_at) VALUES (?, ?, ?, ?, ?, 1, 1, ?, ?);",
                ("pat-uuid-1", "patient1@aurascan.ai", pass_hash, "John Doe", Role.PATIENT.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            # Seed Patient A in patients table
            conn.execute(
                "INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
                ("pat-uuid-1", "John Doe", 45, "Male", datetime.datetime.utcnow().isoformat())
            )
            # Assign Doctor A to Patient A
            conn.execute(
                "INSERT INTO doctor_patient_assignments (doctor_id, patient_id, created_at) VALUES ((SELECT id FROM users WHERE uuid = 'doc-uuid-a'), 'pat-uuid-1', ?);",
                (datetime.datetime.utcnow().isoformat(),)
            )

            # Patient User 2
            conn.execute(
                "INSERT INTO users (uuid, email, password_hash, full_name, role, is_verified, is_active, created_at, updated_at) VALUES (?, ?, ?, ?, ?, 1, 1, ?, ?);",
                ("pat-uuid-2", "patient2@aurascan.ai", pass_hash, "Jane Smith", Role.PATIENT.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            # Seed Patient B in patients table
            conn.execute(
                "INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
                ("pat-uuid-2", "Jane Smith", 30, "Female", datetime.datetime.utcnow().isoformat())
            )
            # Assign Doctor B to Patient B
            conn.execute(
                "INSERT INTO doctor_patient_assignments (doctor_id, patient_id, created_at) VALUES ((SELECT id FROM users WHERE uuid = 'doc-uuid-b'), 'pat-uuid-2', ?);",
                (datetime.datetime.utcnow().isoformat(),)
            )
            conn.commit()
        finally:
            conn.close()

        self.doc_a = self.user_repo.get_by_email("doctor_a@aurascan.ai")
        self.doc_b = self.user_repo.get_by_email("doctor_b@aurascan.ai")
        self.pat_1 = self.user_repo.get_by_email("patient1@aurascan.ai")
        self.pat_2 = self.user_repo.get_by_email("patient2@aurascan.ai")

        self.jwt_svc = JWTService()
        self.token_doc_a = self.jwt_svc.create_access_token(
            user_uuid=self.doc_a.uuid, user_id=self.doc_a.id, email=self.doc_a.email, role=Role.DOCTOR
        )
        self.token_doc_b = self.jwt_svc.create_access_token(
            user_uuid=self.doc_b.uuid, user_id=self.doc_b.id, email=self.doc_b.email, role=Role.DOCTOR
        )
        self.token_pat_1 = self.jwt_svc.create_access_token(
            user_uuid=self.pat_1.uuid, user_id=self.pat_1.id, email=self.pat_1.email, role=Role.PATIENT
        )
        self.token_pat_2 = self.jwt_svc.create_access_token(
            user_uuid=self.pat_2.uuid, user_id=self.pat_2.id, email=self.pat_2.email, role=Role.PATIENT
        )

        os.makedirs("outputs/clinical_reports", exist_ok=True)
        self.scan_file = os.path.abspath("outputs/clinical_reports/test_scan.png")
        with open(self.scan_file, "wb") as f:
            f.write(self._create_mock_brain_mri_slice())

    def tearDown(self):
        self.fastapi_client_ctx.__exit__(None, None, None)
        if "DISABLE_TEST_AUTO_ASSIGN" in os.environ:
            del os.environ["DISABLE_TEST_AUTO_ASSIGN"]
        if os.path.exists(self.scan_file):
            try:
                os.remove(self.scan_file)
            except Exception:
                pass
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def test_assigned_doctor_can_generate_report(self):
        """Case A: Assigned doctor A can successfully generate a report for Patient A."""
        headers = {"Authorization": f"Bearer {self.token_doc_a}"}
        payload = {
            "ref_physician": "Dr. Alice",
            "name": "John Doe",
            "patient_id": "pat-uuid-1",
            "age": 45,
            "gender": "Male",
            "pixel_spacing_mm": 1.0
        }
        res = self.client.post(
            f"/api/report?filepath={self.scan_file}",
            json=payload,
            headers=headers
        )
        self.assertEqual(res.status_code, 200)
        self.assertIn("report_id", res.json())

    def test_unassigned_doctor_cannot_generate_report_for_existing_patient(self):
        """Case B & C: Unassigned doctor cannot generate report for existing patient.
        Doctor A requesting existing Patient B -> 403.
        Doctor B requesting existing Patient A -> 403.
        """
        # Case B: Doctor A requesting existing Patient B (Jane Smith)
        headers_a = {"Authorization": f"Bearer {self.token_doc_a}"}
        payload_b = {
            "ref_physician": "Dr. Alice",
            "name": "Jane Smith",
            "patient_id": "pat-uuid-2",
            "age": 30,
            "gender": "Female",
            "pixel_spacing_mm": 1.0
        }
        res_b = self.client.post(
            f"/api/report?filepath={self.scan_file}",
            json=payload_b,
            headers=headers_a
        )
        self.assertEqual(res_b.status_code, 403)
        self.assertEqual(res_b.json()["detail"], "Access denied to patient records.")

        # Case C: Doctor B requesting existing Patient A (John Doe)
        headers_b = {"Authorization": f"Bearer {self.token_doc_b}"}
        payload_a = {
            "ref_physician": "Dr. Bob",
            "name": "John Doe",
            "patient_id": "pat-uuid-1",
            "age": 45,
            "gender": "Male",
            "pixel_spacing_mm": 1.0
        }
        res_a = self.client.post(
            f"/api/report?filepath={self.scan_file}",
            json=payload_a,
            headers=headers_b
        )
        self.assertEqual(res_a.status_code, 403)
        self.assertEqual(res_a.json()["detail"], "Access denied to patient records.")

    def test_patient_cannot_use_report_endpoint(self):
        """Case E: Patients requesting doctor report endpoint -> 403."""
        headers = {"Authorization": f"Bearer {self.token_pat_1}"}
        payload = {
            "ref_physician": "Dr. Self",
            "name": "John Doe",
            "patient_id": "pat-uuid-1",
            "age": 45,
            "gender": "Male",
            "pixel_spacing_mm": 1.0
        }
        res = self.client.post(
            f"/api/report?filepath={self.scan_file}",
            json=payload,
            headers=headers
        )
        self.assertEqual(res.status_code, 403)

    def test_generate_report_for_nonexistent_patient_fails(self):
        """Case D, F, G, H: Doctor generating report for nonexistent/malformed patient ID fails.
        - Reject; MUST NOT create or assign a patient.
        - Existing doctor-patient assignments remain unchanged.
        - Existing patient count does not increase.
        - Existing assignment count does not increase.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            initial_patient_count = conn.execute("SELECT COUNT(*) FROM patients;").fetchone()[0]
            initial_assignment_count = conn.execute("SELECT COUNT(*) FROM doctor_patient_assignments;").fetchone()[0]
            initial_assignments = conn.execute("SELECT doctor_id, patient_id FROM doctor_patient_assignments;").fetchall()
        finally:
            conn.close()

        headers = {"Authorization": f"Bearer {self.token_doc_a}"}
        malformed_patient_id = "pat-nonexistent-999-typo"
        payload = {
            "ref_physician": "Dr. Alice",
            "name": "Nonexistent Patient",
            "patient_id": malformed_patient_id,
            "age": 30,
            "gender": "Female",
            "pixel_spacing_mm": 1.0
        }

        # Request report generation for nonexistent patient
        res = self.client.post(
            f"/api/report?filepath={self.scan_file}",
            json=payload,
            headers=headers
        )
        self.assertEqual(res.status_code, 403)
        self.assertEqual(res.json()["detail"], "Access denied to patient records.")

        # Verify database metrics remain unchanged
        conn = sqlite3.connect(self.db_path)
        try:
            final_patient_count = conn.execute("SELECT COUNT(*) FROM patients;").fetchone()[0]
            final_assignment_count = conn.execute("SELECT COUNT(*) FROM doctor_patient_assignments;").fetchone()[0]
            final_assignments = conn.execute("SELECT doctor_id, patient_id FROM doctor_patient_assignments;").fetchall()
        finally:
            conn.close()

        # Invariant checks:
        self.assertEqual(final_patient_count, initial_patient_count, "Patient count must not increase after invalid request (Case G)")
        self.assertEqual(final_assignment_count, initial_assignment_count, "Assignment count must not increase after invalid request (Case H)")
        self.assertEqual(final_assignments, initial_assignments, "Existing doctor-patient assignments must remain unchanged (Case F)")

        # Verify doctor still does not have access
        auth_svc = AuthorizationService(self.db_path)
        self.assertFalse(auth_svc.can_access_patient(self.doc_a, malformed_patient_id))


if __name__ == "__main__":
    unittest.main()
