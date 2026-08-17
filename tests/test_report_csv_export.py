import os
import unittest
import sqlite3
import datetime
import csv
import io
import math
import json
from fastapi.testclient import TestClient
from flask import url_for

# Ensure we use a test database
os.environ["DB_PATH"] = os.path.abspath("outputs/test_csv_export.db")

from run_api import app as fastapi_app
from dashboard.infrastructure.web_server import create_app as create_flask_app
from clinical_reporting.application.services import ReportService, PathTraversalException
from persistence.infrastructure.repository import SQLitePersistenceRepository
from security.domain.entities import Role, User
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.jwt_service import JWTService
from security.infrastructure.password import PasswordHasher


class TestReportCSVExportBase(unittest.TestCase):
    def setUp(self):
        self.db_path = os.environ.get("DB_PATH") or os.path.abspath("outputs/test_csv_export.db")
        os.environ["DB_PATH"] = self.db_path

        # Override database path on imported FastAPI routers to avoid cached connections
        from api.infrastructure import routes as api_routes
        from api.routes import auth_routes
        api_routes.DEFAULT_DB_PATH = self.db_path
        auth_routes.DEFAULT_DB_PATH = self.db_path

        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

        os.makedirs("outputs/clinical_reports", exist_ok=True)
        self.persistence_repo = SQLitePersistenceRepository(db_path=self.db_path)
        self.persistence_repo.initialize_db()

        self.user_repo = SQLiteUserRepository(db_path=self.db_path)
        self.user_repo.initialize_security_tables()
        self.admin_user = self.user_repo.bootstrap_admin()

        pass_hash = PasswordHasher.hash_password("Doctor@123")
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO users (uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                ("doc-uuid-1", "doctor@aurascan.ai", pass_hash, "Dr. Jane Smith", Role.DOCTOR.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO users (uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                ("pat-uuid-999", "patient@aurascan.ai", pass_hash, "Bob Jones", Role.PATIENT.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO users (uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                ("pat-uuid-888", "unauthorized_patient@aurascan.ai", pass_hash, "Alice Cooper", Role.PATIENT.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            self._create_test_hierarchy(conn)
            conn.commit()
        finally:
            conn.close()

        self.doctor_user = self.user_repo.get_by_email("doctor@aurascan.ai")
        self.patient_user = self.user_repo.get_by_email("patient@aurascan.ai")
        self.wrong_patient_user = self.user_repo.get_by_email("unauthorized_patient@aurascan.ai")

        self.jwt_svc = JWTService()
        self.admin_token = self.jwt_svc.create_access_token(
            user_uuid=self.admin_user.uuid, user_id=self.admin_user.id, email=self.admin_user.email, role=Role.ADMIN
        )
        self.doctor_token = self.jwt_svc.create_access_token(
            user_uuid=self.doctor_user.uuid, user_id=self.doctor_user.id, email=self.doctor_user.email, role=Role.DOCTOR
        )
        self.patient_token = self.jwt_svc.create_access_token(
            user_uuid=self.patient_user.uuid, user_id=self.patient_user.id, email=self.patient_user.email, role=Role.PATIENT
        )
        self.wrong_patient_token = self.jwt_svc.create_access_token(
            user_uuid=self.wrong_patient_user.uuid, user_id=self.wrong_patient_user.id, email=self.wrong_patient_user.email, role=Role.PATIENT
        )

        # Write dummy valid JSON report
        self.dummy_json_path = os.path.abspath("outputs/clinical_reports/dummy.json")
        self.dummy_data = {
            "patient": {"patient_id": "pat-uuid-999", "name": "Bob Jones", "age": 45, "gender": "Male", "scan_date": "2026-08-08"},
            "classification": {"predicted_class": "Glioma", "confidence_score": 0.95},
            "segmentation": {"tumor_area_mm2": 50.0, "tumor_percentage_brain": 5.0},
            "severity": {"category": "High"},
            "metrics": {"nan_value": float("nan"), "inf_value": float("inf")},
            "xai": {"injection_field": "=1+2"}
        }
        with open(self.dummy_json_path, "w", encoding="utf-8") as f:
            json.dump(self.dummy_data, f)

    def tearDown(self):
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass
        if os.path.exists("outputs/clinical_reports/dummy.json"):
            try:
                os.remove("outputs/clinical_reports/dummy.json")
            except Exception:
                pass

    def _create_test_hierarchy(self, conn, patient_id="pat-uuid-999", prediction_id=1, scan_id=1, report_id=1):
        conn.execute(
            "INSERT OR IGNORE INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
            (patient_id, "Bob Jones", 45, "Male", datetime.datetime.utcnow().isoformat())
        )
        conn.execute(
            "INSERT OR IGNORE INTO mri_scans (id, patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
            (scan_id, patient_id, "dummy_scan.png", 1.0, "Dr. Smith", "2026-08-08", datetime.datetime.utcnow().isoformat())
        )
        conn.execute(
            """INSERT OR IGNORE INTO predictions (
                id, scan_id, predicted_class, confidence_score, prob_glioma, prob_meningioma,
                prob_pituitary, prob_no_tumor, tumor_pixel_count, tumor_area_mm2,
                tumor_percentage_brain, tumor_percentage_image, estimated_brain_pixel_count,
                rule_based_severity, severity_rule_description, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
            (prediction_id, scan_id, "Glioma", 0.95, 0.95, 0.02, 0.02, 0.01, 500, 50.0, 5.0, 2.0, 10000, "High", "High severity rule", datetime.datetime.utcnow().isoformat())
        )
        conn.execute(
            """INSERT OR IGNORE INTO reports (
                report_id, report_number, patient_id, current_version, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?);""",
            (report_id, "REP-2026-0001", patient_id, 1, "FINALIZED", "2026-08-08", "2026-08-08")
        )
        conn.execute(
            """INSERT OR IGNORE INTO report_versions (
                report_id, version_number, pdf_path, json_path, checksum, status, integrity_hash, verification_token, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);""",
            (report_id, 1, "outputs/clinical_reports/dummy.pdf", os.path.abspath("outputs/clinical_reports/dummy.json"), "dummy_checksum", "FINALIZED", None, "token1", "2026-08-08")
        )


class TestFastAPICSVExport(TestReportCSVExportBase):
    def setUp(self):
        super().setUp()
        self.client_ctx = TestClient(fastapi_app)
        self.client = self.client_ctx.__enter__()

    def tearDown(self):
        self.client_ctx.__exit__()
        super().tearDown()

    def test_authenticated_admin_export(self):
        headers = {"Authorization": f"Bearer {self.admin_token}"}
        res = self.client.get("/api/report/1/csv", headers=headers)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get("content-type"), "text/csv; charset=utf-8")

    def test_authenticated_doctor_export(self):
        headers = {"Authorization": f"Bearer {self.doctor_token}"}
        res = self.client.get("/api/report/1/csv", headers=headers)
        self.assertEqual(res.status_code, 200)

    def test_authenticated_patient_own_report_export(self):
        headers = {"Authorization": f"Bearer {self.patient_token}"}
        res = self.client.get("/api/report/1/csv", headers=headers)
        self.assertEqual(res.status_code, 200)

    def test_patient_cross_patient_denial(self):
        headers = {"Authorization": f"Bearer {self.wrong_patient_token}"}
        res = self.client.get("/api/report/1/csv", headers=headers)
        self.assertEqual(res.status_code, 403)

    def test_unauthenticated_denial(self):
        res = self.client.get("/api/report/1/csv")
        self.assertEqual(res.status_code, 401)

    def test_invalid_report_id(self):
        headers = {"Authorization": f"Bearer {self.doctor_token}"}
        res = self.client.get("/api/report/9999/csv", headers=headers)
        self.assertEqual(res.status_code, 404)

    def test_nonexistent_version(self):
        headers = {"Authorization": f"Bearer {self.doctor_token}"}
        res = self.client.get("/api/report/1/csv?version=99", headers=headers)
        self.assertEqual(res.status_code, 404)

    def test_malformed_version_str(self):
        headers = {"Authorization": f"Bearer {self.doctor_token}"}
        res = self.client.get("/api/report/1/csv?version=abc", headers=headers)
        self.assertEqual(res.status_code, 400)

    def test_malformed_version_decimal(self):
        headers = {"Authorization": f"Bearer {self.doctor_token}"}
        res = self.client.get("/api/report/1/csv?version=1.5", headers=headers)
        self.assertEqual(res.status_code, 400)

    def test_malformed_version_negative(self):
        headers = {"Authorization": f"Bearer {self.doctor_token}"}
        res = self.client.get("/api/report/1/csv?version=-1", headers=headers)
        self.assertEqual(res.status_code, 400)

    def test_malformed_version_zero(self):
        headers = {"Authorization": f"Bearer {self.doctor_token}"}
        res = self.client.get("/api/report/1/csv?version=0", headers=headers)
        self.assertEqual(res.status_code, 400)

    def test_malformed_version_empty(self):
        headers = {"Authorization": f"Bearer {self.doctor_token}"}
        res = self.client.get("/api/report/1/csv?version=", headers=headers)
        self.assertEqual(res.status_code, 400)


class TestFlaskCSVExport(TestReportCSVExportBase):
    def setUp(self):
        super().setUp()
        self.app = create_flask_app(db_path=self.db_path)
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def test_authenticated_doctor_export_flask(self):
        headers = {"Authorization": f"Bearer {self.doctor_token}"}
        res = self.client.get("/api/report/1/csv", headers=headers)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get("Content-Disposition"), "attachment; filename=report_1_v1.csv")
        self.assertEqual(res.headers.get("Content-Type"), "text/csv; charset=utf-8")

        # Parse CSV to check columns
        csv_data = res.data.decode("utf-8")
        reader = csv.DictReader(io.StringIO(csv_data))
        rows = list(reader)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Report Number"], "REP-2026-0001")
        self.assertEqual(rows[0]["AI Classification"], "Glioma")
        self.assertEqual(rows[0]["Tumor Area (mm2)"], "50")

    def test_csv_injection_and_nan_handling(self):
        # Override dummy JSON file with injection payload and nan value
        injection_data = {
            "patient": {"patient_id": "pat-uuid-999", "name": "=1+2", "age": 45, "gender": "Male"},
            "classification": {"predicted_class": "Glioma", "confidence_score": float("nan")},
            "segmentation": {"tumor_area_mm2": float("inf"), "tumor_percentage_brain": 5.0}
        }
        with open(self.dummy_json_path, "w", encoding="utf-8") as f:
            json.dump(injection_data, f)

        # Update database integrity hash for the overridden file content
        from clinical_reporting.domain.entities import generate_integrity_hash
        new_hash = generate_integrity_hash(injection_data)
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("UPDATE report_versions SET integrity_hash = ? WHERE report_id = 1;", (new_hash,))
            conn.commit()
        finally:
            conn.close()

        headers = {"Authorization": f"Bearer {self.doctor_token}"}
        res = self.client.get("/api/report/1/csv", headers=headers)
        self.assertEqual(res.status_code, 200)

        csv_data = res.data.decode("utf-8")
        reader = csv.DictReader(io.StringIO(csv_data))
        rows = list(reader)
        self.assertEqual(len(rows), 1)
        # Check that CSV formula injection is escaped with single quote '
        self.assertEqual(rows[0]["Patient Name"], "'=1+2")
        # Check that NaN and Infinity are serialized as empty strings
        self.assertEqual(rows[0]["Confidence Score"], "")
        self.assertEqual(rows[0]["Tumor Area (mm2)"], "")

    def test_audit_success_event_flask(self):
        headers = {"Authorization": f"Bearer {self.doctor_token}"}
        res = self.client.get("/api/report/1/csv", headers=headers)
        self.assertEqual(res.status_code, 200)

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute("SELECT event_type, status, details FROM security_audit_logs WHERE event_type='REPORT_CSV_EXPORTED';")
            row = cursor.fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row[1], "SUCCESS")
        finally:
            conn.close()

    def test_audit_failure_event_flask(self):
        headers = {"Authorization": f"Bearer {self.wrong_patient_token}"}
        res = self.client.get("/api/report/1/csv", headers=headers)
        self.assertEqual(res.status_code, 403)

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute("SELECT event_type, status FROM security_audit_logs WHERE event_type='REPORT_ACCESS_DENIED';")
            row = cursor.fetchone()
            self.assertIsNotNone(row)
        finally:
            conn.close()

    def test_malformed_version_string_flask(self):
        headers = {"Authorization": f"Bearer {self.doctor_token}"}
        res = self.client.get("/api/report/1/csv?version=abc", headers=headers)
        self.assertEqual(res.status_code, 400)

    def test_malformed_version_float_flask(self):
        headers = {"Authorization": f"Bearer {self.doctor_token}"}
        res = self.client.get("/api/report/1/csv?version=1.5", headers=headers)
        self.assertEqual(res.status_code, 400)


class TestFrontendCSVExportIntegration(unittest.TestCase):
    def test_frontend_integration(self):
        templates_dir = "dashboard/presentation/templates"
        for template in ["dashboard_doctor.html", "dashboard_patient.html"]:
            path = os.path.join(templates_dir, template)
            self.assertTrue(os.path.exists(path))
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn("modal-csv-link", content)
            self.assertIn("/csv", content)


if __name__ == "__main__":
    unittest.main()
