import os
import unittest
import sqlite3
import datetime
import cv2
import numpy as np
from fastapi.testclient import TestClient

# Isolate database path
os.environ["DB_PATH"] = os.path.abspath("outputs/test_patient_isolation.db")

from run_api import app
from dashboard.infrastructure.web_server import create_app as create_flask_app
from clinical_reporting.application.services import ReportService
from persistence.infrastructure.repository import SQLitePersistenceRepository
from security.domain.entities import Role, User
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.jwt_service import JWTService
from security.infrastructure.password import PasswordHasher
from security.application.authorization_service import AuthorizationService

class TestPatientIsolation(unittest.TestCase):
    """End-to-end regression tests to verify medical-grade patient/doctor isolation."""

    def _create_mock_brain_mri_slice(self) -> bytes:
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.ellipse(mask, (128, 128), (60, 80), 0, 0, 360, 255, -1)
        np.random.seed(42)
        half_width = 128
        left_texture = np.random.randint(80, 200, size=(256, half_width), dtype=np.uint8)
        right_texture = np.fliplr(left_texture)
        texture = np.hstack([left_texture, right_texture])
        brain_tissue = cv2.bitwise_and(texture, texture, mask=mask)
        brain_tissue_blurred = cv2.GaussianBlur(brain_tissue, (5, 5), 0)
        brain_tissue = np.where(mask > 0, brain_tissue_blurred, 0)
        img[:, :, 0] = brain_tissue
        img[:, :, 1] = brain_tissue
        img[:, :, 2] = brain_tissue
        cv2.ellipse(img, (128, 110), (15, 8), 0, 0, 360, (40, 40, 40), -1)
        cv2.ellipse(img, (128, 140), (12, 6), 0, 0, 360, (40, 40, 40), -1)
        _, buf = cv2.imencode(".png", img)
        return buf.tobytes()

    def setUp(self):
        os.environ["DISABLE_TEST_AUTO_ASSIGN"] = "1"
        self.db_path = os.environ.get("DB_PATH", "outputs/test_patient_isolation.db")
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

        self.user_repo = SQLiteUserRepository(db_path=self.db_path)
        self.user_repo.initialize_security_tables()

        # Drop the auto-assignment triggers to ensure clean testing
        conn_drop = sqlite3.connect(self.db_path)
        try:
            conn_drop.execute("DROP TRIGGER IF EXISTS auto_assign_doctor_on_insert;")
            conn_drop.execute("DROP TRIGGER IF EXISTS auto_assign_patient_on_insert;")
            conn_drop.execute("DELETE FROM doctor_patient_assignments;")
            conn_drop.commit()
        finally:
            conn_drop.close()

        # Bootstrap users
        self.admin_user = self.user_repo.bootstrap_admin()
        pass_hash = PasswordHasher.hash_password("Password@123")

        conn = sqlite3.connect(self.db_path)
        try:
            # Insert doctors
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (30, "doc-a-uuid", "doctor_a@hospital.org", pass_hash, "Dr. Doctor A", Role.DOCTOR.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (31, "doc-b-uuid", "doctor_b@hospital.org", pass_hash, "Dr. Doctor B", Role.DOCTOR.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            # Insert registered patient users (X, Y, Z)
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (40, "pat-x-uuid", "patient_x@hospital.org", pass_hash, "Patient X", Role.PATIENT.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (41, "pat-y-uuid", "patient_y@hospital.org", pass_hash, "Patient Y", Role.PATIENT.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (42, "pat-z-uuid", "patient_z@hospital.org", pass_hash, "Patient Z", Role.PATIENT.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )

            # Insert patient demographics for Patient X and Y (encrypted names/etc.)
            from security.infrastructure.encryption_service import PIIEncryptionService
            encryption_service = PIIEncryptionService()
            enc_x = encryption_service.encrypt("Patient X")
            enc_y = encryption_service.encrypt("Patient Y")

            conn.execute(
                "INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, 30, 'Male', ?);",
                ("pat-x-uuid", enc_x, datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, 25, 'Female', ?);",
                ("pat-y-uuid", enc_y, datetime.datetime.utcnow().isoformat())
            )

            # Insert MRI Scans
            conn.execute(
                "INSERT INTO mri_scans (id, patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                (500, "pat-x-uuid", "uploads/scan_x.png", 1.0, "Dr. Doctor A", "2026-08-08", datetime.datetime.utcnow().isoformat())
            )

            # Insert Predictions
            conn.execute(
                """INSERT INTO predictions (
                    id, scan_id, predicted_class, confidence_score, prob_glioma, prob_meningioma,
                    prob_pituitary, prob_no_tumor, tumor_pixel_count, tumor_area_mm2,
                    tumor_percentage_brain, tumor_percentage_image, estimated_brain_pixel_count,
                    rule_based_severity, severity_rule_description, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
                (600, 500, "Glioma", 0.95, 0.95, 0.02, 0.02, 0.01, 500, 50.0, 5.0, 2.0, 10000, "High", "High severity", datetime.datetime.utcnow().isoformat())
            )

            # Insert Clinical Reports
            conn.execute(
                "INSERT INTO clinical_reports (id, prediction_id, markdown_path, json_path, pdf_path, created_at) VALUES (?, ?, ?, ?, ?, ?);",
                (700, 600, "outputs/clinical_reports/rep_x.md", "outputs/clinical_reports/rep_x.json", "outputs/clinical_reports/rep_x.pdf", datetime.datetime.utcnow().isoformat())
            )

            # Insert Reports Table entry
            conn.execute(
                "INSERT INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                (700, "REP-2026-0001", "pat-x-uuid", 1, "FINAL", "2026-08-08", "2026-08-08")
            )

            # Setup initial assignment: Doctor A is assigned to Patient X
            conn.execute(
                "INSERT INTO doctor_patient_assignments (doctor_id, patient_id, created_at) VALUES (?, ?, ?);",
                (30, "pat-x-uuid", datetime.datetime.utcnow().isoformat())
            )

            conn.commit()
        finally:
            conn.close()

        # Load users
        self.doc_a = self.user_repo.get_by_email("doctor_a@hospital.org")
        self.doc_b = self.user_repo.get_by_email("doctor_b@hospital.org")
        self.pat_x = self.user_repo.get_by_email("patient_x@hospital.org")
        self.pat_y = self.user_repo.get_by_email("patient_y@hospital.org")

        # Create JWT tokens
        self.jwt_svc = JWTService()
        self.token_admin = self.jwt_svc.create_access_token(self.admin_user.uuid, self.admin_user.id, self.admin_user.email, Role.ADMIN)
        self.token_doc_a = self.jwt_svc.create_access_token(self.doc_a.uuid, self.doc_a.id, self.doc_a.email, Role.DOCTOR)
        self.token_doc_b = self.jwt_svc.create_access_token(self.doc_b.uuid, self.doc_b.id, self.doc_b.email, Role.DOCTOR)
        self.token_pat_x = self.jwt_svc.create_access_token(self.pat_x.uuid, self.pat_x.id, self.pat_x.email, Role.PATIENT)
        self.token_pat_y = self.jwt_svc.create_access_token(self.pat_y.uuid, self.pat_y.id, self.pat_y.email, Role.PATIENT)

        # Clients
        self.fastapi_client_ctx = TestClient(app)
        self.client = self.fastapi_client_ctx.__enter__()

        self.flask_app = create_flask_app(db_path=self.db_path)
        self.flask_app.config["TESTING"] = True
        self.flask_client = self.flask_app.test_client()

        # Mock PDF file
        os.makedirs("outputs/clinical_reports", exist_ok=True)
        with open("outputs/clinical_reports/rep_x.pdf", "w") as f:
            f.write("mock pdf contents")

        self.scan_file = os.path.abspath("outputs/clinical_reports/temp_scan.png")
        with open(self.scan_file, "wb") as f:
            f.write(self._create_mock_brain_mri_slice())

    def tearDown(self):
        self.fastapi_client_ctx.__exit__(None, None, None)
        if "DISABLE_TEST_AUTO_ASSIGN" in os.environ:
            del os.environ["DISABLE_TEST_AUTO_ASSIGN"]
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

    def test_01_doctor_a_assigned_patient_x_allowed(self):
        """TEST 1: Doctor A assigned Patient X -> 200/True"""
        auth_svc = AuthorizationService(self.db_path)
        self.assertTrue(auth_svc.can_access_patient(self.doc_a, "pat-x-uuid"))

    def test_02_doctor_b_not_assigned_patient_x_denied(self):
        """TEST 2: Doctor B not assigned Patient X -> 403/False"""
        auth_svc = AuthorizationService(self.db_path)
        self.assertFalse(auth_svc.can_access_patient(self.doc_b, "pat-x-uuid"))

    def test_03_patient_x_accesses_own_data_allowed(self):
        """TEST 3: Patient X accesses own data -> 200/True"""
        auth_svc = AuthorizationService(self.db_path)
        self.assertTrue(auth_svc.can_access_patient(self.pat_x, "pat-x-uuid"))

    def test_04_patient_y_accesses_patient_x_data_denied(self):
        """TEST 4: Patient Y accesses Patient X -> 403/False"""
        auth_svc = AuthorizationService(self.db_path)
        self.assertFalse(auth_svc.can_access_patient(self.pat_y, "pat-x-uuid"))

    def test_05_admin_accesses_patient_x_allowed(self):
        """TEST 5: Admin accesses Patient X -> True"""
        auth_svc = AuthorizationService(self.db_path)
        self.assertTrue(auth_svc.can_access_patient(self.admin_user, "pat-x-uuid"))

    def test_06_unauthenticated_request_fails(self):
        """TEST 6: Unauthenticated request -> 401"""
        res = self.client.get("/api/reports/700")
        self.assertEqual(res.status_code, 401)

    def test_07_registered_patient_but_no_doctor_assignment_denied(self):
        """TEST 7: Registered patient (Z) but NO doctor assignment -> Doctor B gets 403"""
        auth_svc = AuthorizationService(self.db_path)
        # Patient Z demographics do not exist, and no assignment exists. Doctor B should be denied.
        self.assertFalse(auth_svc.can_access_patient(self.doc_b, "pat-z-uuid"))

    def test_08_new_patient_created_by_doctor_a_only_assigns_doctor_a(self):
        """TEST 8: New patient created by Doctor A -> assignment created for Doctor A only"""
        headers = {"Authorization": f"Bearer {self.token_doc_a}"}

        # Explicit onboarding assign patient to Doctor A
        res_assign = self.client.post("/api/doctor/assign-patient", params={"patient_id": "pat-z-uuid"}, headers=headers)
        self.assertEqual(res_assign.status_code, 200)

        payload = {
            "ref_physician": "Dr. Doctor A",
            "name": "Patient Z",
            "patient_id": "pat-z-uuid",
            "age": 28,
            "gender": "Female",
            "pixel_spacing_mm": 1.0
        }
        res = self.client.post("/api/report", params={"filepath": self.scan_file}, json=payload, headers=headers)
        self.assertEqual(res.status_code, 200)

        # Check that assignment was created for Doctor A only
        conn = sqlite3.connect(self.db_path)
        try:
            doc_a_assign = conn.execute("SELECT 1 FROM doctor_patient_assignments WHERE doctor_id = 30 AND patient_id = 'pat-z-uuid';").fetchone()
            doc_b_assign = conn.execute("SELECT 1 FROM doctor_patient_assignments WHERE doctor_id = 31 AND patient_id = 'pat-z-uuid';").fetchone()
            self.assertIsNotNone(doc_a_assign)
            self.assertIsNone(doc_b_assign)
        finally:
            conn.close()

    def test_09_after_doctor_a_creates_patient_z_doctor_a_200_doctor_b_403(self):
        """TEST 9: After Doctor A creates Patient Z: Doctor A -> 200, Doctor B -> 403"""
        headers_a = {"Authorization": f"Bearer {self.token_doc_a}"}

        # Explicit onboarding assign patient to Doctor A
        res_assign = self.client.post("/api/doctor/assign-patient", params={"patient_id": "pat-z-uuid"}, headers=headers_a)
        self.assertEqual(res_assign.status_code, 200)

        payload = {
            "ref_physician": "Dr. Doctor A",
            "name": "Patient Z",
            "patient_id": "pat-z-uuid",
            "age": 28,
            "gender": "Female",
            "pixel_spacing_mm": 1.0
        }
        res = self.client.post("/api/report", params={"filepath": self.scan_file}, json=payload, headers=headers_a)
        self.assertEqual(res.status_code, 200)
        report_id = res.json()["report_id"]

        # Doctor A accesses report details -> 200
        res_a = self.client.get(f"/api/reports/{report_id}", headers=headers_a)
        self.assertEqual(res_a.status_code, 200)

        # Doctor B accesses report details -> 403
        headers_b = {"Authorization": f"Bearer {self.token_doc_b}"}
        res_b = self.client.get(f"/api/reports/{report_id}", headers=headers_b)
        self.assertEqual(res_b.status_code, 403)

    def test_10_duplicate_assignment_creation_prevented(self):
        """TEST 10: Duplicate assignment creation -> only one assignment exists"""
        conn = sqlite3.connect(self.db_path)
        try:
            # Try to assign Doctor A to Patient X again (already assigned in setUp)
            conn.execute("INSERT OR IGNORE INTO doctor_patient_assignments (doctor_id, patient_id, created_at) VALUES (30, 'pat-x-uuid', '2026');")
            conn.commit()

            # Count the assignments for Doctor A and Patient X
            count = conn.execute("SELECT COUNT(*) FROM doctor_patient_assignments WHERE doctor_id = 30 AND patient_id = 'pat-x-uuid';").fetchone()[0]
            self.assertEqual(count, 1)
        finally:
            conn.close()

    def test_11_doctor_b_cannot_access_patient_x_scan(self):
        """TEST 11: Doctor B cannot access Patient X scan -> 403"""
        auth_svc = AuthorizationService(self.db_path)
        self.assertFalse(auth_svc.can_access_scan(self.doc_b, 500))

    def test_12_doctor_b_cannot_access_patient_x_prediction(self):
        """TEST 12: Doctor B cannot access Patient X prediction -> 403"""
        auth_svc = AuthorizationService(self.db_path)
        self.assertFalse(auth_svc.can_access_prediction(self.doc_b, 600))

    def test_13_doctor_b_cannot_access_patient_x_report(self):
        """TEST 13: Doctor B cannot access Patient X report -> 403"""
        auth_svc = AuthorizationService(self.db_path)
        self.assertFalse(auth_svc.can_access_report(self.doc_b, 700))

    def test_14_doctor_b_cannot_download_patient_x_pdf(self):
        """TEST 14: Doctor B cannot download Patient X PDF -> 403"""
        headers = {"Authorization": f"Bearer {self.token_doc_b}"}
        res = self.client.get("/api/report/700/pdf", headers=headers)
        self.assertEqual(res.status_code, 403)

    def test_15_doctor_dashboard_contains_only_assigned_patients(self):
        """TEST 15: Doctor dashboard contains only assigned patients"""
        # Query doctor A history -> should include Patient X (REP-2026-0001)
        headers_a = {"Authorization": f"Bearer {self.token_doc_a}"}
        res_a = self.client.get("/api/database/history", headers=headers_a)
        self.assertEqual(res_a.status_code, 200)
        items_a = res_a.json()
        patient_ids_a = [item["patient_id"] for item in items_a]
        self.assertIn("pat-x-uuid", patient_ids_a)

        # Query doctor B history -> should NOT include Patient X
        headers_b = {"Authorization": f"Bearer {self.token_doc_b}"}
        res_b = self.client.get("/api/database/history", headers=headers_b)
        self.assertEqual(res_b.status_code, 200)
        items_b = res_b.json()
        patient_ids_b = [item["patient_id"] for item in items_b]
        self.assertNotIn("pat-x-uuid", patient_ids_b)

    def test_16_doctor_b_attempts_mri_upload_for_patient_x_denied(self):
        """TEST 16: Doctor B attempts MRI upload for Patient X -> 403"""
        headers = {"Authorization": f"Bearer {self.token_doc_b}"}
        # Doctor B attempts upload for Patient X
        with open(self.scan_file, "rb") as f:
            res = self.client.post(
                "/api/upload",
                files={"file": ("scan.png", f, "image/png")},
                data={"patient_id": "pat-x-uuid"},
                headers=headers
            )
        self.assertEqual(res.status_code, 403)

    def test_17_doctor_b_attempts_report_generation_for_patient_x_denied(self):
        """TEST 17: Doctor B attempts report generation for Patient X -> 403"""
        headers = {"Authorization": f"Bearer {self.token_doc_b}"}
        payload = {
            "ref_physician": "Dr. Doctor B",
            "name": "Patient X",
            "patient_id": "pat-x-uuid",
            "age": 30,
            "gender": "Male",
            "pixel_spacing_mm": 1.0
        }
        res = self.client.post("/api/report", params={"filepath": self.scan_file}, json=payload, headers=headers)
        self.assertEqual(res.status_code, 403)

    def test_18_restart_api_assignments_unchanged(self):
        """TEST 18: Restart API -> assignments remain unchanged and no triggers are recreated"""
        conn1 = sqlite3.connect(self.db_path)
        try:
            count_before = conn1.execute("SELECT COUNT(*) FROM doctor_patient_assignments;").fetchone()[0]
        finally:
            conn1.close()

        # Re-initialize the repository (simulating API reboot)
        db_repo = SQLitePersistenceRepository(db_path=self.db_path)
        db_repo.initialize_db()

        conn2 = sqlite3.connect(self.db_path)
        try:
            count_after = conn2.execute("SELECT COUNT(*) FROM doctor_patient_assignments;").fetchone()[0]
            self.assertEqual(count_before, count_after)

            # Assert triggers do not exist
            trigger_doc = conn2.execute("SELECT 1 FROM sqlite_master WHERE type='trigger' AND name='auto_assign_doctor_on_insert';").fetchone()
            trigger_pat = conn2.execute("SELECT 1 FROM sqlite_master WHERE type='trigger' AND name='auto_assign_patient_on_insert';").fetchone()
            self.assertIsNone(trigger_doc)
            self.assertIsNone(trigger_pat)
        finally:
            conn2.close()

    def test_19_generate_report_does_not_create_assignments_row(self):
        """TEST 19: Report generation does not create new doctor_patient_assignments row"""
        conn = sqlite3.connect(self.db_path)
        try:
            before_count = conn.execute(
                "SELECT COUNT(*) FROM doctor_patient_assignments WHERE doctor_id = 30 AND patient_id = 'pat-x-uuid';"
            ).fetchone()[0]
        finally:
            conn.close()

        headers = {"Authorization": f"Bearer {self.token_doc_a}"}
        payload = {
            "ref_physician": "Dr. Doctor A",
            "name": "Patient X",
            "patient_id": "pat-x-uuid",
            "age": 30,
            "gender": "Male",
            "pixel_spacing_mm": 1.0
        }
        res = self.client.post("/api/report", params={"filepath": self.scan_file}, json=payload, headers=headers)
        self.assertEqual(res.status_code, 200)

        conn = sqlite3.connect(self.db_path)
        try:
            after_count = conn.execute(
                "SELECT COUNT(*) FROM doctor_patient_assignments WHERE doctor_id = 30 AND patient_id = 'pat-x-uuid';"
            ).fetchone()[0]
            self.assertEqual(before_count, after_count)
        finally:
            conn.close()

if __name__ == "__main__":
    unittest.main()
