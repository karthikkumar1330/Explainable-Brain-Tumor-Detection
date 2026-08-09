import unittest
import os
import tempfile
import sqlite3
import math
from dashboard.infrastructure.web_server import create_app
from security.domain.entities import Role, User, TokenType
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.jwt_service import JWTService
from security.infrastructure.password import PasswordHasher
from clinical_reporting.application.services import ReportService

class TestClinicalAnalyticsAPI(unittest.TestCase):
    """Integration test suite to verify Doctor Clinical Analytics Dashboard API endpoints and templates."""

    def setUp(self):
        # Create a temp database
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        self.app = create_app(db_path=self.db_path)
        self.app.config["TESTING"] = True
        self.app.config["DISABLE_CSRF"] = True
        self.client = self.app.test_client()

        # Bootstrap repositories
        self.repo = SQLiteUserRepository(db_path=self.db_path)
        self.repo.initialize_security_tables()

        # Create doctor
        self.doctor = User(
            id=None,
            uuid="doc-uuid-123",
            email="doc@aurascan.ai",
            password_hash=PasswordHasher.hash_password("DoctorPass@123"),
            full_name="Dr. Gregory House",
            role=Role.DOCTOR,
            is_verified=True,
            is_active=True
        )
        self.repo.create_user(self.doctor)

        # Create patient A (multiple scans)
        self.patient_a = User(
            id=None,
            uuid="pat-uuid-aaa",
            email="pat_a@aurascan.ai",
            password_hash=PasswordHasher.hash_password("PatientPass@123"),
            full_name="Patient Alpha",
            role=Role.PATIENT,
            is_verified=True,
            is_active=True
        )
        self.repo.create_user(self.patient_a)

        # Create patient B (0 scans)
        self.patient_b = User(
            id=None,
            uuid="pat-uuid-bbb",
            email="pat_b@aurascan.ai",
            password_hash=PasswordHasher.hash_password("PatientPass@123"),
            full_name="Patient Beta",
            role=Role.PATIENT,
            is_verified=True,
            is_active=True
        )
        self.repo.create_user(self.patient_b)

        # Generate JWT Tokens
        self.jwt_svc = JWTService()
        self.doctor_token = self.jwt_svc.create_access_token(
            user_uuid=self.doctor.uuid, user_id=self.doctor.id, email=self.doctor.email, role=self.doctor.role
        )
        self.patient_a_token = self.jwt_svc.create_access_token(
            user_uuid=self.patient_a.uuid, user_id=self.patient_a.id, email=self.patient_a.email, role=self.patient_a.role
        )
        self.patient_b_token = self.jwt_svc.create_access_token(
            user_uuid=self.patient_b.uuid, user_id=self.patient_b.id, email=self.patient_b.email, role=self.patient_b.role
        )

        self.seed_clinical_data()

    def tearDown(self):
        os.close(self.db_fd)
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def seed_clinical_data(self):
        from persistence.infrastructure.repository import SQLitePersistenceRepository
        clinical_repo = SQLitePersistenceRepository(db_path=self.db_path)
        clinical_repo.initialize_db()

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # Insert patient A
            cursor.execute(
                "INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
                ("pat-uuid-aaa", "Patient Alpha", 38, "Male", "2026-08-01T00:00:00")
            )
            # Insert patient B
            cursor.execute(
                "INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
                ("pat-uuid-bbb", "Patient Beta", 50, "Female", "2026-08-01T00:00:00")
            )

            # Insert Scan 1 for patient A
            cursor.execute(
                "INSERT INTO mri_scans (patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at) VALUES (?, ?, ?, ?, ?, ?);",
                ("pat-uuid-aaa", "outputs/scans/1.png", 1.0, "Dr. House", "2026-08-02", "2026-08-02T10:00:00")
            )
            scan_1_id = cursor.lastrowid

            # Insert Prediction 1
            cursor.execute(
                """
                INSERT INTO predictions (scan_id, predicted_class, confidence_score, prob_glioma, prob_meningioma, prob_pituitary, prob_no_tumor, tumor_pixel_count, tumor_area_mm2, tumor_percentage_brain, tumor_percentage_image, estimated_brain_pixel_count, rule_based_severity, severity_rule_description, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (scan_1_id, "Glioma", 0.95, 0.95, 0.02, 0.02, 0.01, 200, 250.0, 0.10, 0.05, 2000, "Medium", "Tumor identified", "2026-08-02T10:05:00")
            )
            pred_1_id = cursor.lastrowid

            # Insert Clinical Report 1
            cursor.execute(
                """
                INSERT INTO clinical_reports (id, prediction_id, markdown_path, json_path, pdf_path, overlay_path, heatmap_path, mask_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (101, pred_1_id, "outputs/r1.md", "outputs/r1.json", "outputs/r1.pdf", "outputs/o1.png", "outputs/h1.png", "outputs/m1.png", "2026-08-02T10:05:00")
            )

            # Insert Version & Report 1
            cursor.execute(
                "INSERT INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                (101, "RPT-101", "pat-uuid-aaa", 1, "FINAL", "2026-08-02T10:05:00", "2026-08-02T10:05:00")
            )
            cursor.execute(
                """
                INSERT INTO report_versions (report_id, version_number, prediction_id, status, reason, pdf_path, json_path, checksum, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (101, 1, pred_1_id, "FINAL", "Baseline", "outputs/r1.pdf", "outputs/r1.json", "checksum1", "2026-08-02T10:05:00")
            )

            # Insert Scan 2 for patient A (Progression)
            cursor.execute(
                "INSERT INTO mri_scans (patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at) VALUES (?, ?, ?, ?, ?, ?);",
                ("pat-uuid-aaa", "outputs/scans/2.png", 1.0, "Dr. House", "2026-08-08", "2026-08-08T10:00:00")
            )
            scan_2_id = cursor.lastrowid

            # Insert Prediction 2 (Increase in tumor size)
            cursor.execute(
                """
                INSERT INTO predictions (scan_id, predicted_class, confidence_score, prob_glioma, prob_meningioma, prob_pituitary, prob_no_tumor, tumor_pixel_count, tumor_area_mm2, tumor_percentage_brain, tumor_percentage_image, estimated_brain_pixel_count, rule_based_severity, severity_rule_description, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (scan_2_id, "Glioma", 0.99, 0.99, 0.0, 0.0, 0.01, 300, 375.0, 0.15, 0.07, 2000, "High", "Significant area increase", "2026-08-08T10:05:00")
            )
            pred_2_id = cursor.lastrowid

            # Insert Clinical Report 2
            cursor.execute(
                """
                INSERT INTO clinical_reports (id, prediction_id, markdown_path, json_path, pdf_path, overlay_path, heatmap_path, mask_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (102, pred_2_id, "outputs/r2.md", "outputs/r2.json", "outputs/r2.pdf", "outputs/o2.png", "outputs/h2.png", "outputs/m2.png", "2026-08-08T10:05:00")
            )

            # Insert Version & Report 2
            cursor.execute(
                "INSERT INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                (102, "RPT-102", "pat-uuid-aaa", 1, "FINAL", "2026-08-08T10:05:00", "2026-08-08T10:05:00")
            )
            cursor.execute(
                """
                INSERT INTO report_versions (report_id, version_number, prediction_id, status, reason, pdf_path, json_path, checksum, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (102, 1, pred_2_id, "FINAL", "Follow-up scan", "outputs/r2.pdf", "outputs/r2.json", "checksum2", "2026-08-08T10:05:00")
            )

            conn.commit()
        finally:
            conn.close()

    def test_population_analytics_endpoint_security(self):
        """Verify that only authorized roles (Doctor/Admin) can access population overview analytics."""
        # 1. Unauthenticated request denied
        res = self.client.get("/api/analytics/overview")
        self.assertEqual(res.status_code, 401)

        # 2. Patient role denied (Forbidden)
        self.client.set_cookie("access_token", self.patient_a_token)
        res = self.client.get("/api/analytics/overview")
        self.assertEqual(res.status_code, 403)

        # 3. Doctor role permitted (Success)
        self.client.set_cookie("access_token", self.doctor_token)
        res = self.client.get("/api/analytics/overview")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("total_patients", data)
        self.assertEqual(data["total_patients"], 2)
        self.assertEqual(data["total_reports"], 2)
        self.assertEqual(data["total_scans"], 2)

    def test_population_analytics_response_payload(self):
        """Verify the JSON fields of population analytics overview endpoint."""
        self.client.set_cookie("access_token", self.doctor_token)
        res = self.client.get("/api/analytics/overview")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        # Schema keys assertion
        expected_keys = [
            "total_patients", "total_reports", "total_scans", 
            "classification_distribution", "severity_distribution",
            "progression_distribution", "average_confidence", 
            "average_tumor_area", "activity_over_time"
        ]
        for key in expected_keys:
            self.assertIn(key, data)

        self.assertIn("Glioma", data["classification_distribution"])
        self.assertIn("High", data["severity_distribution"])
        self.assertIn("PROGRESSION", data["progression_distribution"])
        self.assertTrue(len(data["activity_over_time"]) >= 2)

    def test_patient_specific_analytics_isolation(self):
        """Verify patient isolation on patient-specific analytics endpoints."""
        # Patient A tries to access Patient A's own analytics (Allowed)
        self.client.set_cookie("access_token", self.patient_a_token)
        res = self.client.get("/api/patients/pat-uuid-aaa/analytics")
        self.assertEqual(res.status_code, 200)
        
        # Patient A tries to access Patient B's analytics (Denied)
        res = self.client.get("/api/patients/pat-uuid-bbb/analytics")
        self.assertEqual(res.status_code, 403)

        # Doctor accesses Patient A's analytics (Allowed)
        self.client.set_cookie("access_token", self.doctor_token)
        res = self.client.get("/api/patients/pat-uuid-aaa/analytics")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["patient_id"], "pat-uuid-aaa")
        self.assertEqual(data["total_scans"], 2)

    def test_patient_timeseries_analytics(self):
        """Verify patient timeseries points data payload."""
        self.client.set_cookie("access_token", self.doctor_token)
        res = self.client.get("/api/patients/pat-uuid-aaa/analytics/timeseries")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        self.assertEqual(data["patient_id"], "pat-uuid-aaa")
        self.assertEqual(len(data["points"]), 2)
        p1 = data["points"][0]
        self.assertIn("date", p1)
        self.assertEqual(p1["classification"], "Glioma")
        self.assertEqual(p1["tumor_area"], 250.0)

    def test_doctor_dashboard_template_contains_analytics_components(self):
        """Verify that the dashboard doctor HTML template has all UI components and skeletons."""
        self.client.set_cookie("access_token", self.doctor_token)
        res = self.client.get("/doctor")
        self.assertEqual(res.status_code, 200)
        html = res.data.decode("utf-8")

        # 1. Verification of KPI Card containers
        self.assertIn('id="analytics-stat-patients"', html)
        self.assertIn('id="analytics-stat-reports"', html)
        self.assertIn('id="analytics-stat-scans"', html)
        self.assertIn('id="analytics-stat-avg-confidence"', html)

        # 2. Verification of skeletons
        self.assertIn('id="analytics-skeleton-patients"', html)
        self.assertIn('id="analytics-skeleton-reports"', html)
        self.assertIn('id="analytics-skeleton-scans"', html)
        self.assertIn('id="analytics-skeleton-avg-confidence"', html)

        # 3. Verification of charts elements
        self.assertIn('id="chart-progression"', html)
        self.assertIn('id="chart-patient-area"', html)
        self.assertIn('id="chart-patient-occupancy"', html)
        self.assertIn('id="chart-patient-confidence"', html)

    def test_safe_formatting_nan_infinity_handling(self):
        """Verify that entity JSON serialization sanitizes NaN/Infinity/None correctly."""
        from clinical_reporting.domain.entities import sanitize_json_value
        dirty_payload = {
            "nan_val": float("nan"),
            "inf_val": float("inf"),
            "neg_inf": float("-inf"),
            "normal": 42.5,
            "nested": {
                "nested_nan": float("nan")
            }
        }
        clean_payload = sanitize_json_value(dirty_payload)
        self.assertIsNone(clean_payload["nan_val"])
        self.assertIsNone(clean_payload["inf_val"])
        self.assertIsNone(clean_payload["neg_inf"])
        self.assertEqual(clean_payload["normal"], 42.5)
        self.assertIsNone(clean_payload["nested"]["nested_nan"])

if __name__ == "__main__":
    unittest.main()
