import os
import sqlite3
import datetime
import json
import unittest
from fastapi.testclient import TestClient

os.environ["DB_PATH"] = os.path.abspath("outputs/test_h45_reports.db")
os.environ["DISABLE_TEST_AUTO_ASSIGN"] = "1"

from run_api import app
from dashboard.infrastructure.web_server import create_app
from security.domain.entities import Role, User
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.jwt_service import JWTService
from persistence.infrastructure.repository import SQLitePersistenceRepository
from security.infrastructure.encryption_service import PIIEncryptionService
from security.application.authorization_service import AuthorizationService


class TestH45PatientReportsSecurity(unittest.TestCase):
    """Focused integration and security tests for Phase H4.5-S Patient Reports name-collision IDOR hardening."""

    def setUp(self):
        test_method_name = self.id().split('.')[-1]
        self.db_path = os.path.abspath(f"outputs/test_h45_reports_{test_method_name}.db")
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

        # Create mock directories
        os.makedirs("outputs/clinical_reports", exist_ok=True)
        self.mock_json_a = os.path.abspath(f"outputs/clinical_reports/pat_a_report_{test_method_name}.json")
        self.mock_json_b = os.path.abspath(f"outputs/clinical_reports/pat_b_report_{test_method_name}.json")
        self.dummy_pdf_a = os.path.abspath(f"outputs/clinical_reports/pat_a_{test_method_name}.pdf")
        self.dummy_pdf_b = os.path.abspath(f"outputs/clinical_reports/pat_b_{test_method_name}.pdf")

        # Create physical dummy files
        with open(self.dummy_pdf_a, "wb") as f:
            f.write(b"%PDF-1.4 dummy data")
        with open(self.dummy_pdf_b, "wb") as f:
            f.write(b"%PDF-1.4 dummy data")

        self.payload_a = {
            "patient": {"patient_id": "pat-uuid-a", "name": "Jane Doe", "age": "45", "gender": "Male", "scan_date": "2026-08-01"},
            "processing": {"device": "cpu", "total_execution_time_sec": 0.5},
            "classification": {"predicted_class": "Glioma", "confidence_score": 0.95},
            "files": {"original_image": "outputs/scans/pat_a.png"}
        }

        self.payload_b = {
            "patient": {"patient_id": "pat-uuid-b", "name": "Jane Doe", "age": "30", "gender": "Female", "scan_date": "2026-08-02"},
            "processing": {"device": "cpu", "total_execution_time_sec": 0.3},
            "classification": {"predicted_class": "No Tumor", "confidence_score": 0.99},
            "files": {"original_image": "outputs/scans/pat_b.png"}
        }

        with open(self.mock_json_a, "w", encoding="utf-8") as f:
            json.dump(self.payload_a, f)
        with open(self.mock_json_b, "w", encoding="utf-8") as f:
            json.dump(self.payload_b, f)

        # Setup passwords
        from security.infrastructure.password import PasswordHasher
        pass_hash = PasswordHasher.hash_password("Password@123")

        # Bootstrap Database Users (Both Patients share the full name "Jane Doe")
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
            # 2. Patients with name-collision ("Jane Doe")
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (300, "pat-uuid-a", "patient_a@aurascan.ai", pass_hash, "Jane Doe", Role.PATIENT.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (301, "pat-uuid-b", "patient_b@aurascan.ai", pass_hash, "Jane Doe", Role.PATIENT.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            # 3. Patient Demographics (Encrypted)
            enc_svc = PIIEncryptionService()
            conn.execute(
                "INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
                ("pat-uuid-a", enc_svc.encrypt("Jane Doe"), enc_svc.encrypt("45"), enc_svc.encrypt("Male"), datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
                ("pat-uuid-b", enc_svc.encrypt("Jane Doe"), enc_svc.encrypt("30"), enc_svc.encrypt("Female"), datetime.datetime.utcnow().isoformat())
            )
            # 4. Assignments: Doctor A is assigned to Patient A, Doctor B is unassigned
            conn.execute(
                "INSERT INTO doctor_patient_assignments (doctor_id, patient_id, created_at) VALUES (?, ?, ?);",
                (200, "pat-uuid-a", datetime.datetime.utcnow().isoformat())
            )

            # 5. Seed Patient A Scans, Predictions, and Reports
            conn.execute(
                "INSERT INTO mri_scans (id, patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                (1, "pat-uuid-a", "outputs/scans/pat_a.png", 1.0, "Dr. Sarah", "2026-08-01", datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO predictions (id, scan_id, predicted_class, confidence_score, prob_glioma, prob_meningioma, prob_pituitary, prob_no_tumor, tumor_pixel_count, tumor_area_mm2, tumor_percentage_brain, tumor_percentage_image, estimated_brain_pixel_count, rule_based_severity, severity_rule_description, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
                (1, 1, "Glioma", 0.95, 0.95, 0.02, 0.02, 0.01, 500, 50.0, 1.2, 0.8, 40000, "HIGH", "High severity detected", datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO clinical_reports (id, prediction_id, markdown_path, json_path, pdf_path, heatmap_path, overlay_path, mask_path, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);",
                (100, 1, "outputs/reports/pat_a.md", self.mock_json_a, self.dummy_pdf_a, "outputs/reports/pat_a_hm.png", "outputs/reports/pat_a_ov.png", "outputs/reports/pat_a_mk.png", datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at, pdf_path, json_path, checksum) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
                (100, "REP-2026-100", "pat-uuid-a", 1, "FINAL", datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat(), self.dummy_pdf_a, self.mock_json_a, "dummychecksum")
            )
            conn.execute(
                "INSERT INTO report_versions (version_id, report_id, version_number, created_at, created_by, reason, pdf_path, json_path, checksum, status, prediction_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
                (1001, 100, 1, datetime.datetime.utcnow().isoformat(), "doctor_a@aurascan.ai", "Initial baseline", self.dummy_pdf_a, self.mock_json_a, "dummychecksum", "FINAL", 1)
            )

            # 6. Seed Patient B Scans, Predictions, and Reports
            conn.execute(
                "INSERT INTO mri_scans (id, patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                (2, "pat-uuid-b", "outputs/scans/pat_b.png", 1.0, "Dr. Sarah", "2026-08-02", datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO predictions (id, scan_id, predicted_class, confidence_score, prob_glioma, prob_meningioma, prob_pituitary, prob_no_tumor, tumor_pixel_count, tumor_area_mm2, tumor_percentage_brain, tumor_percentage_image, estimated_brain_pixel_count, rule_based_severity, severity_rule_description, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
                (2, 2, "No Tumor", 0.99, 0.01, 0.01, 0.01, 0.97, 0, 0.0, 0.0, 0.0, 42000, "LOW", "No tumor detected", datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO clinical_reports (id, prediction_id, markdown_path, json_path, pdf_path, heatmap_path, overlay_path, mask_path, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);",
                (200, 2, "outputs/reports/pat_b.md", self.mock_json_b, self.dummy_pdf_b, None, None, None, datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at, pdf_path, json_path, checksum) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
                (200, "REP-2026-200", "pat-uuid-b", 1, "FINAL", datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat(), self.dummy_pdf_b, self.mock_json_b, "dummychecksum")
            )
            conn.execute(
                "INSERT INTO report_versions (version_id, report_id, version_number, created_at, created_by, reason, pdf_path, json_path, checksum, status, prediction_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
                (2001, 200, 1, datetime.datetime.utcnow().isoformat(), "doctor_b@aurascan.ai", "Initial baseline", self.dummy_pdf_b, self.mock_json_b, "dummychecksum", "FINAL", 2)
            )

            conn.commit()
        finally:
            conn.close()

        # Initialize JWT Tokens
        self.jwt_svc = JWTService()
        self.token_admin = self.jwt_svc.create_access_token("admin-uuid", 1, "admin@aurascan.ai", Role.ADMIN)
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

        for p in [self.mock_json_a, self.mock_json_b, self.dummy_pdf_a, self.dummy_pdf_b]:
            if os.path.exists(p):
                os.remove(p)

        # Restore global state to avoid contaminating other tests
        if "DISABLE_TEST_AUTO_ASSIGN" in os.environ:
            del os.environ["DISABLE_TEST_AUTO_ASSIGN"]
        os.environ["DB_PATH"] = os.path.abspath("outputs/clinical_reports.db")

        import api.infrastructure.routes as api_routes
        import api.routes.auth_routes as auth_routes
        api_routes.DEFAULT_DB_PATH = os.path.abspath("outputs/clinical_reports.db")
        auth_routes.DEFAULT_DB_PATH = os.path.abspath("outputs/clinical_reports.db")

    # 1. Patient's own history
    def test_patient_own_history(self):
        self.flask_client.set_cookie("access_token", self.token_pat_a)
        res = self.flask_client.get("/api/history")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["report_id"], 100)
        self.assertEqual(data[0]["patient_id"], "pat-uuid-a")

    # 2. Patient cross-patient history
    def test_patient_cross_patient_history(self):
        self.flask_client.set_cookie("access_token", self.token_pat_a)
        res = self.flask_client.get("/api/history?patient_id=pat-uuid-b")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["patient_id"], "pat-uuid-a")

    # 3. Same-name collision
    def test_same_name_collision_history(self):
        self.flask_client.set_cookie("access_token", self.token_pat_a)
        res = self.flask_client.get("/api/history")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        for record in data:
            self.assertNotEqual(record["report_id"], 200)
            self.assertEqual(record["patient_id"], "pat-uuid-a")

    # 4. Patient UUID override
    def test_patient_uuid_override_in_query(self):
        self.flask_client.set_cookie("access_token", self.token_pat_a)
        res = self.flask_client.get("/api/history?patient_id=pat-uuid-b")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["patient_id"], "pat-uuid-a")

    # 5. Patient name override
    def test_patient_name_override_in_query(self):
        self.flask_client.set_cookie("access_token", self.token_pat_a)
        res = self.flask_client.get("/api/history?patient_name=Jane+Doe&patient_id=pat-uuid-b")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["patient_id"], "pat-uuid-a")

    # 6. Own PDF access
    def test_own_pdf_access(self):
        self.flask_client.set_cookie("access_token", self.token_pat_a)
        res = self.flask_client.get("/api/report/100/pdf")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get("Content-Type"), "application/pdf")

    # 7. Other patient's PDF access
    def test_other_patient_pdf_access_blocked(self):
        self.flask_client.set_cookie("access_token", self.token_pat_a)
        res = self.flask_client.get("/api/report/200/pdf")
        self.assertEqual(res.status_code, 403)

    # 8. Own JSON access
    def test_own_json_access(self):
        self.flask_client.set_cookie("access_token", self.token_pat_a)
        res = self.flask_client.get("/api/report/100/json")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["patient"]["patient_id"], "pat-uuid-a")

    # 9. Other patient's JSON access
    def test_other_patient_json_access_blocked(self):
        self.flask_client.set_cookie("access_token", self.token_pat_a)
        res = self.flask_client.get("/api/report/200/json")
        self.assertEqual(res.status_code, 403)

    # 10. Own CSV access
    def test_own_csv_access(self):
        self.flask_client.set_cookie("access_token", self.token_pat_a)
        res = self.flask_client.get("/api/report/100/csv")
        self.assertEqual(res.status_code, 200)
        self.assertIn("Report Number", res.get_data(as_text=True))

    # 11. Other patient's CSV access
    def test_other_patient_csv_access_blocked(self):
        self.flask_client.set_cookie("access_token", self.token_pat_a)
        res = self.flask_client.get("/api/report/200/csv")
        self.assertEqual(res.status_code, 403)

    # 12. Unauthenticated access
    def test_unauthenticated_access_blocked(self):
        res = self.flask_client.get("/api/history")
        self.assertEqual(res.status_code, 401)
        res = self.flask_client.get("/api/report/100/pdf")
        self.assertEqual(res.status_code, 401)
        res = self.flask_client.get("/api/report/100/json")
        self.assertEqual(res.status_code, 401)
        res = self.flask_client.get("/api/report/100/csv")
        self.assertEqual(res.status_code, 401)

    # 13. Doctor assigned patient
    def test_doctor_assigned_patient_access(self):
        self.flask_client.set_cookie("access_token", self.token_doc_a)
        res = self.flask_client.get("/api/report/100")
        self.assertEqual(res.status_code, 200)

    # 14. Doctor unassigned patient
    def test_doctor_unassigned_patient_access_blocked(self):
        self.flask_client.set_cookie("access_token", self.token_doc_b)
        res = self.flask_client.get("/api/report/100")
        self.assertEqual(res.status_code, 403)

    # 15. Admin access
    def test_admin_access_unrestricted(self):
        self.flask_client.set_cookie("access_token", self.token_admin)
        res = self.flask_client.get("/api/report/100")
        self.assertEqual(res.status_code, 200)
        res = self.flask_client.get("/api/report/200")
        self.assertEqual(res.status_code, 200)

    # 16. Authorization service direct tests
    def test_authorization_service_direct_checks(self):
        auth_svc = AuthorizationService(db_path=self.db_path)
        user_pat_a = User(id=300, uuid="pat-uuid-a", email="patient_a@aurascan.ai", password_hash="dummy", full_name="Jane Doe", role=Role.PATIENT)
        user_pat_b = User(id=301, uuid="pat-uuid-b", email="patient_b@aurascan.ai", password_hash="dummy", full_name="Jane Doe", role=Role.PATIENT)
        user_doc_a = User(id=200, uuid="doc-uuid-a", email="doctor_a@aurascan.ai", password_hash="dummy", full_name="Doctor A", role=Role.DOCTOR)
        user_doc_b = User(id=201, uuid="doc-uuid-b", email="doctor_b@aurascan.ai", password_hash="dummy", full_name="Doctor B", role=Role.DOCTOR)
        user_admin = User(id=1, uuid="admin-uuid", email="admin@aurascan.ai", password_hash="dummy", full_name="Admin", role=Role.ADMIN)

        # Patient own UUID -> ALLOWED
        self.assertTrue(auth_svc.can_access_patient(user_pat_a, "pat-uuid-a"))
        # Same-name collision UUID -> BLOCKED
        self.assertFalse(auth_svc.can_access_patient(user_pat_a, "pat-uuid-b"))
        # Unassigned doctor -> BLOCKED
        self.assertFalse(auth_svc.can_access_patient(user_doc_b, "pat-uuid-a"))
        # Assigned doctor -> ALLOWED
        self.assertTrue(auth_svc.can_access_patient(user_doc_a, "pat-uuid-a"))
        # Admin -> ALLOWED
        self.assertTrue(auth_svc.can_access_patient(user_admin, "pat-uuid-a"))
        self.assertTrue(auth_svc.can_access_patient(user_admin, "pat-uuid-b"))

    # Security check: verify no stack traces or database info in exceptions
    def test_no_exception_information_leakage(self):
        self.flask_client.set_cookie("access_token", self.token_pat_a)
        res = self.flask_client.get("/api/report/99999/pdf")
        self.assertEqual(res.status_code, 404)
        data = res.get_json()
        self.assertIn("error", data)
        # Verify no database schema terms or python traceback lines are in the error message
        err_msg = str(data["error"]).lower()
        self.assertNotIn("traceback", err_msg)
        self.assertNotIn("sqlite", err_msg)
        self.assertNotIn("select", err_msg)
        self.assertNotIn("table", err_msg)
