import os
import sqlite3
import datetime
import unittest
from fastapi.testclient import TestClient

os.environ["DB_PATH"] = os.path.abspath("outputs/test_h32b_followup.db")
os.environ["DISABLE_TEST_AUTO_ASSIGN"] = "1"

from run_api import app
from dashboard.infrastructure.web_server import create_app
from security.domain.entities import Role, User
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.jwt_service import JWTService
from persistence.infrastructure.repository import SQLitePersistenceRepository
from security.infrastructure.encryption_service import PIIEncryptionService

class TestH32BFollowupWorkflow(unittest.TestCase):
    """Focused integration tests for Phase H3.2-B Patient Profile + Follow-up Workflow."""

    def setUp(self):
        test_method_name = self.id().split('.')[-1]
        self.db_path = os.path.abspath(f"outputs/test_h32b_followup_{test_method_name}.db")
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
            # 5. MRI scan for Patient A (to verify profile loading)
            conn.execute(
                "INSERT INTO mri_scans (id, patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                (500, "pat-uuid-a", "scans/scan_500.png", 1.0, "Doctor A", "2026-08-01", datetime.datetime.utcnow().isoformat())
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

    def test_01_assigned_doctor_can_create_followup(self):
        """1. Verify that an assigned doctor can successfully schedule a follow-up."""
        self.flask_client.set_cookie("access_token", self.token_doc_a)
        res = self.flask_client.post(
            "/api/doctor/patients/pat-uuid-a/followups",
            json={"scheduled_date": "2026-09-15", "reason": "Checkup", "notes": "Post-treatment check."}
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIsNotNone(data.get("followup_id"))
        self.assertEqual(data["status"], "scheduled")
        self.assertEqual(data["scheduled_date"], "2026-09-15")

        # FastAPI check
        headers = {"Authorization": f"Bearer {self.token_doc_a}"}
        res_fast = self.fastapi_client.post(
            "/api/patients/pat-uuid-a/followups",
            json={"scheduled_date": "2026-10-15", "reason": "MRI checkup", "notes": "Monitor progress."},
            headers=headers
        )
        self.assertEqual(res_fast.status_code, 200)

    def test_02_assigned_doctor_can_retrieve_followup(self):
        """2. Verify that an assigned doctor can retrieve follow-up schedules."""
        # Create one first
        self.flask_client.set_cookie("access_token", self.token_doc_a)
        self.flask_client.post(
            "/api/doctor/patients/pat-uuid-a/followups",
            json={"scheduled_date": "2026-09-15", "reason": "Checkup"}
        )

        res = self.flask_client.get("/api/doctor/patients/pat-uuid-a/followups")
        self.assertEqual(res.status_code, 200)
        items = res.get_json()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["reason"], "Checkup")

        # FastAPI check
        headers = {"Authorization": f"Bearer {self.token_doc_a}"}
        res_fast = self.fastapi_client.get("/api/patients/pat-uuid-a/followups", headers=headers)
        self.assertEqual(res_fast.status_code, 200)
        self.assertEqual(len(res_fast.json()), 1)

    def test_03_assigned_doctor_can_update_followup(self):
        """3. Verify that an assigned doctor can update follow-up schedules."""
        # Create
        self.flask_client.set_cookie("access_token", self.token_doc_a)
        res_create = self.flask_client.post(
            "/api/doctor/patients/pat-uuid-a/followups",
            json={"scheduled_date": "2026-09-15", "reason": "Checkup"}
        )
        fid = res_create.get_json()["followup_id"]

        # Update
        res_up = self.flask_client.put(
            f"/api/doctor/followups/{fid}",
            json={"scheduled_date": "2026-09-20", "notes": "Date modified."}
        )
        self.assertEqual(res_up.status_code, 200)
        data = res_up.get_json()
        self.assertEqual(data["scheduled_date"], "2026-09-20")
        self.assertEqual(data["notes"], "Date modified.")

        # FastAPI check
        headers = {"Authorization": f"Bearer {self.token_doc_a}"}
        res_fast_up = self.fastapi_client.put(
            f"/api/doctor/followups/{fid}",
            json={"scheduled_date": "2026-09-25", "notes": "Date modified via FastAPI."},
            headers=headers
        )
        self.assertEqual(res_fast_up.status_code, 200)

    def test_04_assigned_doctor_can_complete_followup(self):
        """4. Verify that an assigned doctor can mark a follow-up as completed."""
        self.flask_client.set_cookie("access_token", self.token_doc_a)
        res_create = self.flask_client.post(
            "/api/doctor/patients/pat-uuid-a/followups",
            json={"scheduled_date": "2026-09-15"}
        )
        fid = res_create.get_json()["followup_id"]

        # Complete
        res_comp = self.flask_client.put(f"/api/doctor/followups/{fid}", json={"status": "completed"})
        self.assertEqual(res_comp.status_code, 200)
        data = res_comp.get_json()
        self.assertEqual(data["status"], "completed")
        self.assertIsNotNone(data["completed_at"])

    def test_05_assigned_doctor_can_cancel_followup(self):
        """5. Verify that an assigned doctor can cancel a follow-up."""
        self.flask_client.set_cookie("access_token", self.token_doc_a)
        res_create = self.flask_client.post(
            "/api/doctor/patients/pat-uuid-a/followups",
            json={"scheduled_date": "2026-09-15"}
        )
        fid = res_create.get_json()["followup_id"]

        # Cancel
        res_cancel = self.flask_client.put(f"/api/doctor/followups/{fid}", json={"status": "cancelled"})
        self.assertEqual(res_cancel.status_code, 200)
        data = res_cancel.get_json()
        self.assertEqual(data["status"], "cancelled")
        self.assertIsNotNone(data["completed_at"])

    def test_06_unassigned_doctor_gets_403(self):
        """6. Verify that an unassigned doctor receives a 403 error."""
        # Doctor B is unassigned to Patient A
        self.flask_client.set_cookie("access_token", self.token_doc_b)
        res = self.flask_client.post(
            "/api/doctor/patients/pat-uuid-a/followups",
            json={"scheduled_date": "2026-09-15", "reason": "Illegal checkup"}
        )
        self.assertEqual(res.status_code, 403)

        # GET check
        res_get = self.flask_client.get("/api/doctor/patients/pat-uuid-a/followups")
        self.assertEqual(res_get.status_code, 403)

        # FastAPI check
        headers = {"Authorization": f"Bearer {self.token_doc_b}"}
        res_fast = self.fastapi_client.post(
            "/api/patients/pat-uuid-a/followups",
            json={"scheduled_date": "2026-09-15"},
            headers=headers
        )
        self.assertEqual(res_fast.status_code, 403)

    def test_07_patient_cannot_access_other_patient_followup(self):
        """7. Verify that patients cannot access another patient's follow-up schedule."""
        self.flask_client.set_cookie("access_token", self.token_pat_b)
        res = self.flask_client.post(
            "/api/doctor/patients/pat-uuid-a/followups",
            json={"scheduled_date": "2026-09-15"}
        )
        self.assertEqual(res.status_code, 403)

        # Retrieve
        res_get = self.flask_client.get("/api/doctor/patients/pat-uuid-a/followups")
        self.assertEqual(res_get.status_code, 403)

    def test_08_nonexistent_patient_returns_404(self):
        """8. Verify that queries for nonexistent patients return 404."""
        self.flask_client.set_cookie("access_token", self.token_doc_a)
        res = self.flask_client.post(
            "/api/doctor/patients/nonexistent-pat-uuid/followups",
            json={"scheduled_date": "2026-09-15"}
        )
        self.assertEqual(res.status_code, 404)

        res_get = self.flask_client.get("/api/doctor/patients/nonexistent-pat-uuid/followups")
        self.assertEqual(res_get.status_code, 404)

    def test_09_invalid_inputs_rejected(self):
        """9. Verify that invalid dates, statuses, or inputs are strictly rejected."""
        self.flask_client.set_cookie("access_token", self.token_doc_a)

        # Invalid date format
        res1 = self.flask_client.post(
            "/api/doctor/patients/pat-uuid-a/followups",
            json={"scheduled_date": "2026/09/15"}
        )
        self.assertEqual(res1.status_code, 400)

        # Invalid status
        res_create = self.flask_client.post(
            "/api/doctor/patients/pat-uuid-a/followups",
            json={"scheduled_date": "2026-09-15"}
        )
        fid = res_create.get_json()["followup_id"]
        res2 = self.flask_client.put(
            f"/api/doctor/followups/{fid}",
            json={"status": "invalid_status_value"}
        )
        self.assertEqual(res2.status_code, 400)

    def test_10_followup_survives_restart(self):
        """10. Verify that follow-up schedules survive database reloads/restarts."""
        self.flask_client.set_cookie("access_token", self.token_doc_a)
        res_create = self.flask_client.post(
            "/api/doctor/patients/pat-uuid-a/followups",
            json={"scheduled_date": "2026-09-15", "reason": "Checkup Persistence"}
        )
        self.assertEqual(res_create.status_code, 200)

        # Force reconnect database query directly
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute("SELECT reason FROM followup_schedules WHERE patient_id = ?;", ("pat-uuid-a",)).fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row[0], "Checkup Persistence")
        finally:
            conn.close()

    def test_11_audit_logging_works(self):
        """11. Verify that security/audit logging captures follow-up operations."""
        self.flask_client.set_cookie("access_token", self.token_doc_a)
        res_create = self.flask_client.post(
            "/api/doctor/patients/pat-uuid-a/followups",
            json={"scheduled_date": "2026-09-15", "reason": "Audit Test"}
        )
        self.assertEqual(res_create.status_code, 200)

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute("SELECT * FROM security_audit_logs WHERE event_type = 'FOLLOWUP_CREATED' ORDER BY timestamp DESC LIMIT 1;").fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["status"], "SUCCESS")
            self.assertIn("pat-uuid-a", row["details"])
        finally:
            conn.close()

    def test_12_doctor_patient_profile_endpoint(self):
        """12. Verify the patient demographics profile endpoint returns decrypted metrics and history."""
        self.flask_client.set_cookie("access_token", self.token_doc_a)
        res = self.flask_client.get("/api/doctor/patients/pat-uuid-a")
        self.assertEqual(res.status_code, 200)
        profile = res.get_json()

        self.assertEqual(profile["patient_id"], "pat-uuid-a")
        self.assertEqual(profile["name"], "Patient A Name")
        self.assertEqual(profile["age"], 45)
        self.assertEqual(profile["gender"], "Male")
        self.assertIsNotNone(profile["latest_mri"])
        self.assertEqual(profile["latest_mri"]["id"], 500)

        # Unassigned doctor get denied
        self.flask_client.set_cookie("access_token", self.token_doc_b)
        res_forbidden = self.flask_client.get("/api/doctor/patients/pat-uuid-a")
        self.assertEqual(res_forbidden.status_code, 403)

    def test_13_nonexistent_followup_id_returns_404(self):
        """13. Verify that updating a nonexistent follow-up ID returns 404."""
        self.flask_client.set_cookie("access_token", self.token_doc_a)
        res = self.flask_client.put(
            "/api/doctor/followups/999999",
            json={"scheduled_date": "2026-09-30", "status": "completed"}
        )
        self.assertEqual(res.status_code, 404)

    def test_14_patient_profile_role_boundary(self):
        """14. Verify that patient users are blocked from doctor workspace patient profile endpoint."""
        self.flask_client.set_cookie("access_token", self.token_pat_a)
        res = self.flask_client.get("/api/doctor/patients/pat-uuid-a")
        self.assertEqual(res.status_code, 403)

    def test_15_shared_followup_workspace(self):
        """15. Verify that multiple doctors assigned to the same patient can view and modify follow-ups."""
        # Assign Doctor B (201) to Patient A (pat-uuid-a) as well
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO doctor_patient_assignments (doctor_id, patient_id, created_at) VALUES (?, ?, ?);",
                (201, "pat-uuid-a", datetime.datetime.utcnow().isoformat())
            )
            conn.commit()
        finally:
            conn.close()

        # Doctor A creates follow-up
        self.flask_client.set_cookie("access_token", self.token_doc_a)
        res_create = self.flask_client.post(
            "/api/doctor/patients/pat-uuid-a/followups",
            json={"scheduled_date": "2026-09-15", "reason": "Doc A follow-up"}
        )
        self.assertEqual(res_create.status_code, 200)
        fid = res_create.get_json()["followup_id"]

        # Doctor B retrieves it
        self.flask_client.set_cookie("access_token", self.token_doc_b)
        res_get = self.flask_client.get("/api/doctor/patients/pat-uuid-a/followups")
        self.assertEqual(res_get.status_code, 200)
        items = res_get.get_json()
        self.assertTrue(any(i["followup_id"] == fid for i in items))

        # Doctor B completes it
        res_up = self.flask_client.put(
            f"/api/doctor/followups/{fid}",
            json={"status": "completed", "notes": "Completed by Doc B"}
        )
        self.assertEqual(res_up.status_code, 200)
        self.assertEqual(res_up.get_json()["status"], "completed")
