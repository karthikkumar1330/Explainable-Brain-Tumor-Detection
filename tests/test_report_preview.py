import os
import unittest
import sqlite3
import datetime
import json
from fastapi.testclient import TestClient

os.environ["DB_PATH"] = os.path.abspath("outputs/test_report_preview.db")

from run_api import app
from clinical_reporting.application.services import ReportService
from persistence.infrastructure.repository import SQLitePersistenceRepository
from security.domain.entities import Role, User
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.jwt_service import JWTService


class TestReportPreview(unittest.TestCase):
    """Phase F2 Automated unit, integration, and template verification tests."""

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

        pdf_path = os.path.abspath("outputs/clinical_reports/dummy_report.pdf")

        conn.execute(
            """INSERT OR IGNORE INTO reports (
                report_id, report_number, patient_id, current_version, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?);""",
            (report_id, f"REP-2026-0001", patient_id, 1, "FINALIZED", "2026-08-08", "2026-08-08")
        )
        conn.execute(
            """INSERT OR IGNORE INTO report_versions (
                report_id, version_number, pdf_path, json_path, checksum, status, integrity_hash, verification_token, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);""",
            (report_id, 1, pdf_path, "outputs/clinical_reports/dummy.json", "dummy_checksum", "FINALIZED", None, "token1", "2026-08-08")
        )

    def setUp(self):
        self.db_path = os.environ.get("DB_PATH", "outputs/test_report_preview.db")
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

        os.makedirs("outputs/clinical_reports", exist_ok=True)
        with open("outputs/clinical_reports/dummy_report.pdf", "wb") as f:
            f.write(b"%PDF-1.4 ... dummy pdf content ...")

        self.persistence_repo = SQLitePersistenceRepository(db_path=self.db_path)
        self.persistence_repo.initialize_db()

        self.test_client_ctx = TestClient(app)
        self.client = self.test_client_ctx.__enter__()

        self.user_repo = SQLiteUserRepository(db_path=self.db_path)
        self.user_repo.initialize_security_tables()
        self.admin_user = self.user_repo.bootstrap_admin()

        from security.infrastructure.password import PasswordHasher
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
            from tests.helpers.authorization_fixtures import assign_doctor_to_patient
            assign_doctor_to_patient(conn, "doctor@aurascan.ai", "pat-uuid-999")
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

    def tearDown(self):
        self.test_client_ctx.__exit__()
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass
        if os.path.exists("outputs/clinical_reports/dummy_report.pdf"):
            try:
                os.remove("outputs/clinical_reports/dummy_report.pdf")
            except Exception:
                pass

    def test_authenticated_report_preview_access(self):
        """1. Verify authorized access gets PDF content successfully."""
        headers = {"Authorization": f"Bearer {self.doctor_token}"}
        res = self.client.get("/api/report/1/pdf", headers=headers)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get("content-type"), "application/pdf")

    def test_unauthenticated_access(self):
        """2. Verify unauthenticated requests return 401."""
        res = self.client.get("/api/report/1/pdf")
        self.assertEqual(res.status_code, 401)

    def test_unauthorized_patient_access(self):
        """3. Verify patient cannot fetch other patient's report PDF."""
        headers = {"Authorization": f"Bearer {self.wrong_patient_token}"}
        res = self.client.get("/api/report/1/pdf", headers=headers)
        self.assertEqual(res.status_code, 403)

    def test_authorized_doctor_access(self):
        """4. Verify doctor can view report PDF."""
        headers = {"Authorization": f"Bearer {self.doctor_token}"}
        res = self.client.get("/api/report/1/pdf", headers=headers)
        self.assertEqual(res.status_code, 200)

    def test_authorized_admin_access(self):
        """5. Verify admin can view report PDF."""
        headers = {"Authorization": f"Bearer {self.admin_token}"}
        res = self.client.get("/api/report/1/pdf", headers=headers)
        self.assertEqual(res.status_code, 200)

    def test_invalid_report_id(self):
        """6. Verify querying invalid report ID returns 404."""
        headers = {"Authorization": f"Bearer {self.doctor_token}"}
        res = self.client.get("/api/report/9999/pdf", headers=headers)
        self.assertEqual(res.status_code, 404)

    def test_invalid_version(self):
        """7. Verify querying invalid version returns 404."""
        headers = {"Authorization": f"Bearer {self.doctor_token}"}
        res = self.client.get("/api/report/1/pdf?version=99", headers=headers)
        self.assertEqual(res.status_code, 404)

    def test_access_logging(self):
        """8. Verify that access requests write to audit history table."""
        headers = {"Authorization": f"Bearer {self.doctor_token}"}
        self.client.get("/api/report/1/pdf", headers=headers)

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute("SELECT event_type, status FROM security_audit_logs;")
            logs = cursor.fetchall()
            self.assertTrue(len(logs) > 0)
            self.assertIn(("REPORT_DOWNLOADED", "SUCCESS"), logs)
        finally:
            conn.close()

    def test_content_type_and_safety(self):
        """9 & 10. Verify response content type is application/pdf and contains no stack traces."""
        headers = {"Authorization": f"Bearer {self.doctor_token}"}
        res = self.client.get("/api/report/9999/pdf", headers=headers)
        self.assertEqual(res.status_code, 404)
        data = res.json()
        self.assertIn("detail", data)
        self.assertNotIn("Traceback", json.dumps(data))
        self.assertNotIn("SELECT", json.dumps(data))

    def test_frontend_templates_integration(self):
        """Verify presence of preview controls and modal containers in HTML templates."""
        templates_dir = "dashboard/presentation/templates"
        for template in ["dashboard_doctor.html", "dashboard_patient.html"]:
            path = os.path.join(templates_dir, template)
            self.assertTrue(os.path.exists(path))
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            self.assertIn('id="report-preview-modal"', content)
            self.assertIn('id="preview-canvas"', content)
            self.assertIn('id="preview-loading"', content)
            self.assertIn('id="preview-error"', content)
            self.assertIn('id="preview-controls"', content)

            self.assertIn("openReportPreview", content)
            self.assertIn("renderPreviewPage", content)
            self.assertIn("prevPreviewPage", content)
            self.assertIn("nextPreviewPage", content)
            self.assertIn("zoomPreview", content)
            self.assertIn("resetPreviewZoom", content)
            self.assertIn("closeReportPreview", content)
            self.assertIn("printPreviewReport", content)

            self.assertIn("AbortController", content)
            self.assertIn("pdfjsLib", content)
            self.assertIn("aria-label", content)
            self.assertIn("Escape", content)
            self.assertIn("currentPdfDoc.destroy()", content)
