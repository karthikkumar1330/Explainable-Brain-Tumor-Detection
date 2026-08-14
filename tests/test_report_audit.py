import os
import unittest
import sqlite3
import datetime
import json
import cv2
import numpy as np
from fastapi.testclient import TestClient

# Isolate database path
os.environ["DB_PATH"] = os.path.abspath("outputs/test_report_audit.db")

from run_api import app
from dashboard.infrastructure.web_server import create_app as create_flask_app
from clinical_reporting.application.services import ReportService
from clinical_reporting.domain.entities import ReportStatus, generate_integrity_hash
from persistence.infrastructure.repository import SQLitePersistenceRepository
from security.domain.entities import Role, User
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.jwt_service import JWTService
from security.infrastructure.password import PasswordHasher

class TestReportAuditTrail(unittest.TestCase):
    """Phase F2.4 automated audit trail and access history integration tests."""

    def _create_mock_brain_mri_slice(self) -> bytes:
        """Generates a mock brain MRI slice programmatically (centered ellipse with tissue texture)."""
        # Create black canvas
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Draw brain-like centered ellipse
        # Center = (128, 128), Axes = (60, 80)
        mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.ellipse(mask, (128, 128), (60, 80), 0, 0, 360, 255, -1)
        
        # Generate some synthetic symmetric texture inside the brain region
        np.random.seed(42)
        half_width = 128
        left_texture = np.random.randint(80, 200, size=(256, half_width), dtype=np.uint8)
        right_texture = np.fliplr(left_texture)
        texture = np.hstack([left_texture, right_texture])
        
        # Apply texture inside mask
        brain_tissue = cv2.bitwise_and(texture, texture, mask=mask)
        
        # Blur the interior slightly to match soft tissues
        brain_tissue_blurred = cv2.GaussianBlur(brain_tissue, (5, 5), 0)
        brain_tissue = np.where(mask > 0, brain_tissue_blurred, 0)
        
        # Convert to BGR color image (standard MRI encoding)
        img[:, :, 0] = brain_tissue  # B
        img[:, :, 1] = brain_tissue  # G
        img[:, :, 2] = brain_tissue  # R
        
        # Add high frequency details inside the brain to pass contrast & blur check
        # e.g., draw ventricles
        cv2.ellipse(img, (128, 110), (15, 8), 0, 0, 360, (40, 40, 40), -1)
        cv2.ellipse(img, (128, 140), (12, 6), 0, 0, 360, (40, 40, 40), -1)
        
        # Encode to PNG bytes
        _, buf = cv2.imencode(".png", img)
        return buf.tobytes()

    def _create_test_hierarchy(self, conn, patient_id="pat-uuid-999", prediction_id=1, scan_id=1):
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

    def setUp(self):
        self.db_path = os.environ.get("DB_PATH", "outputs/test_report_audit.db")
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

        # FastAPI Routes configuration
        import api.infrastructure.routes as api_routes
        import api.routes.auth_routes as auth_routes
        api_routes.DEFAULT_DB_PATH = self.db_path
        auth_routes.DEFAULT_DB_PATH = self.db_path

        # Initialize DB
        self.persistence_repo = SQLitePersistenceRepository(db_path=self.db_path)
        self.persistence_repo.initialize_db()

        # Set up fastapi test client
        self.fastapi_client_ctx = TestClient(app)
        self.client = self.fastapi_client_ctx.__enter__()

        # Set up flask app client
        self.flask_app = create_flask_app(db_path=self.db_path)
        self.flask_app.config["TESTING"] = True
        self.flask_client = self.flask_app.test_client()

        # Bootstrap users
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
            # Second patient for boundary check
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

        os.makedirs("outputs/clinical_reports", exist_ok=True)

    def tearDown(self):
        self.fastapi_client_ctx.__exit__(None, None, None)
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def test_report_generation_logs_create_report(self):
        """Verifies report creation via the pipeline API endpoint triggers audit logging."""
        # Create a mock brain mri scan that passes magic number and other checks
        scan_file = os.path.abspath("outputs/clinical_reports/mock_scan.png")
        with open(scan_file, "wb") as f:
            f.write(self._create_mock_brain_mri_slice())

        headers = {"Authorization": f"Bearer {self.doctor_token}"}
        payload = {
            "ref_physician": "Dr. Jane Smith",
            "name": "Bob Jones",
            "patient_id": "pat-uuid-999",
            "age": 45,
            "gender": "Male",
            "pixel_spacing_mm": 1.0
        }

        res = self.client.post("/api/report", params={"filepath": scan_file}, json=payload, headers=headers)
        if os.path.exists(scan_file):
            os.remove(scan_file)

        self.assertEqual(res.status_code, 200)
        report_id = res.json()["report_id"]

        # Assert audit trail logs REPORT_LIFECYCLE_CHANGE with action CREATE_REPORT
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT * FROM security_audit_logs WHERE event_type = ? AND details LIKE ?;",
                ("REPORT_LIFECYCLE_CHANGE", f"%Report ID: {report_id}%")
            ).fetchone()
            self.assertIsNotNone(row)
            self.assertIn("CREATE_REPORT", row["details"])
            self.assertEqual(row["status"], "SUCCESS")
        finally:
            conn.close()

    def test_view_report_metadata_logs_access(self):
        """Verifies viewing report metadata logs REPORT_VIEWED."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at) VALUES (1, 'REP-1', 'pat-uuid-999', 1, 'FINAL', '2026-08-08', '2026-08-08')"
            )
            conn.commit()
        finally:
            conn.close()

        headers = {"Authorization": f"Bearer {self.patient_token}"}
        res = self.client.get("/api/reports/1", headers=headers)
        self.assertEqual(res.status_code, 200)

        # Check logs
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT * FROM security_audit_logs WHERE event_type = ? AND details LIKE ?;",
                ("REPORT_VIEWED", "%Report ID: 1%")
            ).fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["email"], "patient@aurascan.ai")
        finally:
            conn.close()

    def test_view_nonexistent_report_logs_not_found(self):
        """Verifies accessing a nonexistent report logs REPORT_NOT_FOUND."""
        headers = {"Authorization": f"Bearer {self.doctor_token}"}
        res = self.client.get("/api/reports/9999", headers=headers)
        self.assertEqual(res.status_code, 404)

        # Check logs
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT * FROM security_audit_logs WHERE event_type = ? AND details LIKE ?;",
                ("REPORT_NOT_FOUND", "%Report ID: 9999%")
            ).fetchone()
            self.assertIsNotNone(row)
        finally:
            conn.close()

    def test_unauthorized_report_access_logs_denied(self):
        """Verifies unauthorized patient accessing someone else's report logs REPORT_ACCESS_DENIED."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at) VALUES (1, 'REP-1', 'pat-uuid-999', 1, 'FINAL', '2026-08-08', '2026-08-08')"
            )
            conn.commit()
        finally:
            conn.close()

        # Alice Cooper tries to view Bob Jones's report (ID 1)
        headers = {"Authorization": f"Bearer {self.wrong_patient_token}"}
        res = self.client.get("/api/reports/1", headers=headers)
        self.assertEqual(res.status_code, 403)

        # Check logs
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT * FROM security_audit_logs WHERE event_type = ? AND details LIKE ?;",
                ("REPORT_ACCESS_DENIED", "%Report ID: 1%")
            ).fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["email"], "unauthorized_patient@aurascan.ai")
        finally:
            conn.close()

    def test_serve_report_visuals_logs_audit(self):
        """Verifies requesting report visuals logs REPORT_VIEWED in FastAPI."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at) VALUES (5, 'REP-5', 'pat-uuid-999', 1, 'FINAL', '2026-08-08', '2026-08-08')"
            )
            conn.execute(
                "INSERT INTO clinical_reports (id, prediction_id, markdown_path, json_path, pdf_path, overlay_path, heatmap_path, mask_path, created_at) VALUES (5, 1, 'm.md', 'j.json', 'p.pdf', 'o.png', 'h.png', 'm.png', '2026-08-08')"
            )
            conn.commit()
        finally:
            conn.close()

        headers = {"Authorization": f"Bearer {self.doctor_token}"}
        # Call visuals overlay endpoint (will return 404 image placeholder, but access should be logged!)
        res = self.client.get("/api/report/5/visuals/overlay", headers=headers)
        self.assertEqual(res.status_code, 404)

        # Check logs
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT * FROM security_audit_logs WHERE event_type = ? AND details LIKE ?;",
                ("REPORT_VIEWED", "%Viewed report visual: overlay%")
            ).fetchone()
            self.assertIsNotNone(row)
        finally:
            conn.close()

    def test_flask_visuals_route_logs_audit(self):
        """Verifies requesting report visuals logs REPORT_VIEWED in Flask."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at) VALUES (6, 'REP-6', 'pat-uuid-999', 1, 'FINAL', '2026-08-08', '2026-08-08')"
            )
            conn.execute(
                "INSERT INTO clinical_reports (id, prediction_id, markdown_path, json_path, pdf_path, overlay_path, heatmap_path, mask_path, created_at) VALUES (6, 1, 'm.md', 'j.json', 'p.pdf', 'o.png', 'h.png', 'm.png', '2026-08-08')"
            )
            conn.commit()
        finally:
            conn.close()

        # Simulate Flask cookie for login_required
        self.flask_client.set_cookie("access_token", self.doctor_token)

        # Call Flask visuals route
        res = self.flask_client.get("/api/report/6/visuals/heatmap")
        self.assertEqual(res.status_code, 404)

        # Check logs
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT * FROM security_audit_logs WHERE event_type = ? AND details LIKE ?;",
                ("REPORT_VIEWED", "%Viewed report visual: heatmap%")
            ).fetchone()
            self.assertIsNotNone(row)
        finally:
            conn.close()

    def test_get_audit_history_patient_isolation(self):
        """Verifies audit trail isolation boundaries where patients can only see permitted reports and their own actions."""
        service = ReportService(db_path=self.db_path)
        conn = sqlite3.connect(self.db_path)
        try:
            # Seed report associations
            conn.execute(
                "INSERT INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at) VALUES (10, 'REP-10', 'pat-uuid-999', 1, 'FINAL', '2026-08-08', '2026-08-08')"
            )
            conn.execute(
                "INSERT INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at) VALUES (20, 'REP-20', 'pat-uuid-888', 1, 'FINAL', '2026-08-08', '2026-08-08')"
            )
            conn.execute(
                "INSERT OR IGNORE INTO doctor_patient_assignments (doctor_id, patient_id, created_at) VALUES (?, 'pat-uuid-999', '2026-08-08');",
                (self.doctor_user.id,)
            )
            conn.execute(
                "INSERT OR IGNORE INTO doctor_patient_assignments (doctor_id, patient_id, created_at) VALUES (?, 'pat-uuid-888', '2026-08-08');",
                (self.doctor_user.id,)
            )

            # 1. Bob Jones (patient 999) views his report
            service._log_security_audit_event(
                conn, "REPORT_VIEWED", self.patient_user, 10, "SUCCESS", "Viewed report details."
            )
            # 2. Alice Cooper (patient 888) views her report
            service._log_security_audit_event(
                conn, "REPORT_VIEWED", self.wrong_patient_user, 20, "SUCCESS", "Viewed report details."
            )
            # 3. Bob Jones tries to view Alice's report (access denied)
            service._log_security_audit_event(
                conn, "REPORT_ACCESS_DENIED", self.patient_user, 20, "FAILED", "Access denied to patient report."
            )
            conn.commit()
        finally:
            conn.close()

        # Doctor -> Can see all 3 logs
        headers_doc = {"Authorization": f"Bearer {self.doctor_token}"}
        res_doc = self.client.get("/api/reports/audit-history", headers=headers_doc)
        self.assertEqual(res_doc.status_code, 200)
        self.assertEqual(len(res_doc.json()), 3)

        # Patient Bob Jones -> Can see only logs relating to Report ID 10 OR his own actions (the denial on report 20 is his own action!)
        headers_pat = {"Authorization": f"Bearer {self.patient_token}"}
        res_pat = self.client.get("/api/reports/audit-history", headers=headers_pat)
        self.assertEqual(res_pat.status_code, 200)
        bob_logs = res_pat.json()
        self.assertEqual(len(bob_logs), 2)
        # Check that Alice's view (report 20) is NOT visible to Bob
        for log in bob_logs:
            self.assertNotEqual(log["email"], "unauthorized_patient@aurascan.ai")

if __name__ == "__main__":
    unittest.main()
