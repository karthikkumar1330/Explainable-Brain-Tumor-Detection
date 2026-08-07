import unittest
import os
import tempfile
import sqlite3
from flask import url_for
from dashboard.infrastructure.web_server import create_app
from security.domain.entities import Role, User, TokenType
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.jwt_service import JWTService
from security.infrastructure.password import PasswordHasher


class TestRBACIntegration(unittest.TestCase):
    """Integration test suite to verify Role-Based Access Control redirects, page routing, and endpoint protection."""

    def setUp(self):
        # Create a temp database
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        self.app = create_app(db_path=self.db_path)
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

        # Bootstrap database schema and admin
        self.repo = SQLiteUserRepository(db_path=self.db_path)
        self.repo.initialize_security_tables()
        
        # Bootstrap default admin
        self.admin = self.repo.bootstrap_admin(admin_email="admin@aurascan.ai", admin_pass="Admin@123456")
        
        # Create test doctor user (pre-verified and active)
        self.doctor = User(
            id=None,
            uuid="doc-uuid-111",
            email="doctor@hospital.org",
            password_hash=PasswordHasher.hash_password("DoctorPass@123"),
            full_name="Dr. Alice",
            role=Role.DOCTOR,
            is_verified=True,
            is_active=True
        )
        self.repo.create_user(self.doctor)

        # Create test patient user (pre-verified and active)
        self.patient = User(
            id=None,
            uuid="pat-uuid-222",
            email="patient@health.org",
            password_hash=PasswordHasher.hash_password("PatientPass@123"),
            full_name="John Patient",
            role=Role.PATIENT,
            is_verified=True,
            is_active=True
        )
        self.repo.create_user(self.patient)

        # Create another test patient user for cross-resource check
        self.other_patient = User(
            id=None,
            uuid="pat-uuid-333",
            email="other@health.org",
            password_hash=PasswordHasher.hash_password("OtherPass@123"),
            full_name="Other Patient",
            role=Role.PATIENT,
            is_verified=True,
            is_active=True
        )
        self.repo.create_user(self.other_patient)

        # Set up JWT tokens
        self.jwt_svc = JWTService()
        self.admin_token = self.jwt_svc.create_access_token(
            user_uuid=self.admin.uuid,
            user_id=self.admin.id,
            email=self.admin.email,
            role=self.admin.role
        )
        self.doctor_token = self.jwt_svc.create_access_token(
            user_uuid=self.doctor.uuid,
            user_id=self.doctor.id,
            email=self.doctor.email,
            role=self.doctor.role
        )
        self.patient_token = self.jwt_svc.create_access_token(
            user_uuid=self.patient.uuid,
            user_id=self.patient.id,
            email=self.patient.email,
            role=self.patient.role
        )
        self.other_patient_token = self.jwt_svc.create_access_token(
            user_uuid=self.other_patient.uuid,
            user_id=self.other_patient.id,
            email=self.other_patient.email,
            role=self.other_patient.role
        )

        # Seed clinical report database structure for checking PDF owner checks
        self.seed_clinical_report()

    def tearDown(self):
        os.close(self.db_fd)
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def seed_clinical_report(self):
        """Seeds a mock patient scan and report linked to the first patient."""
        from persistence.infrastructure.repository import SQLitePersistenceRepository
        clinical_repo = SQLitePersistenceRepository(db_path=self.db_path)
        clinical_repo.initialize_db()

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # 1. Insert patient matching first patient uuid/name
            cursor.execute(
                "INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
                ("pat-uuid-222", "John Patient", 45, "Male", "2026-08-06T00:00:00")
            )
            # 2. Insert scan
            cursor.execute(
                "INSERT INTO mri_scans (patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at) VALUES (?, ?, ?, ?, ?, ?);",
                ("pat-uuid-222", "outputs/scans/test.png", 1.0, "Dr. Alice", "2026-08-06", "2026-08-06T00:00:00")
            )
            scan_id = cursor.lastrowid
            # 3. Insert prediction
            cursor.execute(
                """
                INSERT INTO predictions (scan_id, predicted_class, confidence_score, prob_glioma, prob_meningioma, prob_pituitary, prob_no_tumor, tumor_pixel_count, tumor_area_mm2, tumor_percentage_brain, tumor_percentage_image, estimated_brain_pixel_count, rule_based_severity, severity_rule_description, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (scan_id, "Glioma", 0.98, 0.98, 0.01, 0.01, 0.0, 100, 120.5, 0.05, 0.02, 2000, "Medium", "Rule description", "2026-08-06T00:00:00")
            )
            pred_id = cursor.lastrowid
            # 4. Insert report (id = 1)
            cursor.execute(
                """
                INSERT INTO clinical_reports (prediction_id, markdown_path, json_path, pdf_path, overlay_path, heatmap_path, mask_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (pred_id, "outputs/reports/rep.md", "outputs/reports/rep.json", "outputs/reports/rep.pdf", "outputs/reports/rep_overlay.png", "outputs/reports/rep_heatmap.png", "outputs/reports/rep_mask.png", "2026-08-06T00:00:00")
            )
            conn.commit()
        finally:
            conn.close()

    # --- 1. Test Web Page Route Redirection & Route Protection ---

    def test_unauthenticated_page_redirects(self):
        """Verify that accessing dashboard pages without login redirects to landing page."""
        for path in ["/admin", "/doctor", "/patient"]:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 302)
            self.assertIn("/", response.headers.get("Location", ""))

    def test_admin_page_authorization(self):
        """Verify role protection and redirects on the /admin web page route."""
        # Logged in as Patient -> Should redirect back to landing page
        self.client.set_cookie("access_token", self.patient_token)
        response = self.client.get("/admin")
        self.assertEqual(response.status_code, 302)
        self.assertTrue(response.headers.get("Location", "").endswith("/"))

        # Logged in as Doctor -> Should redirect back to landing page
        self.client.set_cookie("access_token", self.doctor_token)
        response = self.client.get("/admin")
        self.assertEqual(response.status_code, 302)

        # Logged in as Admin -> Should load page successfully (status 200)
        self.client.set_cookie("access_token", self.admin_token)
        response = self.client.get("/admin")
        self.assertEqual(response.status_code, 200)

    def test_doctor_page_authorization(self):
        """Verify role protection and redirects on the /doctor web page route."""
        # Logged in as Patient -> Redirect
        self.client.set_cookie("access_token", self.patient_token)
        response = self.client.get("/doctor")
        self.assertEqual(response.status_code, 302)

        # Logged in as Doctor -> 200 OK
        self.client.set_cookie("access_token", self.doctor_token)
        response = self.client.get("/doctor")
        self.assertEqual(response.status_code, 200)

    def test_patient_page_authorization(self):
        """Verify role protection and redirects on the /patient web page route."""
        # Logged in as Doctor -> Redirect
        self.client.set_cookie("access_token", self.doctor_token)
        response = self.client.get("/patient")
        self.assertEqual(response.status_code, 302)

        # Logged in as Patient -> 200 OK
        self.client.set_cookie("access_token", self.patient_token)
        response = self.client.get("/patient")
        self.assertEqual(response.status_code, 200)

    # --- 2. Test API Endpoints & Role Permissions ---

    def test_admin_api_endpoints_rbac(self):
        """Verify Admin APIs are strictly blocked for Patients/Doctors and permitted for Admin."""
        for token, expected_status in [
            (self.patient_token, 403),
            (self.doctor_token, 403),
            (self.admin_token, 200)
        ]:
            self.client.set_cookie("access_token", token)
            response = self.client.get("/api/admin/users")
            self.assertEqual(response.status_code, expected_status)

            response_logs = self.client.get("/api/admin/audit-logs")
            self.assertEqual(response_logs.status_code, expected_status)

    def test_report_visuals_api_rbac(self):
        """Verify that Patient role is blocked from viewing raw scan visuals (CAM overlays / UNeXt masks)."""
        for token, expected_status in [
            (self.patient_token, 403),
            (self.doctor_token, 404),  # Returns 404 instead of 403 because scan file doesn't physically exist, indicating request passed authorization!
            (self.admin_token, 404)
        ]:
            self.client.set_cookie("access_token", token)
            response = self.client.get("/api/report/1/visuals/overlay")
            self.assertEqual(response.status_code, expected_status)

    def test_report_pdf_owner_check_rbac(self):
        """Verify that a Patient can ONLY access their own report PDFs, whereas Doctors and Admins can access any."""
        # 1. Patient requests their own report PDF (Report ID = 1 matches patient 'John Patient')
        self.client.set_cookie("access_token", self.patient_token)
        response = self.client.get("/api/report/1/pdf")
        self.assertEqual(response.status_code, 404)  # 404 file not physically present indicating permission check passed!

        # 2. Other Patient requests Patient's report PDF -> Blocked with 403 Forbidden!
        self.client.set_cookie("access_token", self.other_patient_token)
        response = self.client.get("/api/report/1/pdf")
        self.assertEqual(response.status_code, 403)

        # 3. Doctor requests Patient's report PDF -> Allowed (404 indicate permission passed)
        self.client.set_cookie("access_token", self.doctor_token)
        response = self.client.get("/api/report/1/pdf")
        self.assertEqual(response.status_code, 404)

        # 4. Admin requests Patient's report PDF -> Allowed (404 indicate permission passed)
        self.client.set_cookie("access_token", self.admin_token)
        response = self.client.get("/api/report/1/pdf")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
