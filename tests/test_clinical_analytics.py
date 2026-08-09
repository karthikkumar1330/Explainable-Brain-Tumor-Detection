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

    def test_population_analytics_query_count_is_constant(self):
        """Verify that get_population_analytics runs a constant number of queries instead of N+1."""
        service = ReportService(db_path=self.db_path)
        queries_run = []

        class SpyConnection:
            def __init__(self, real_conn):
                self.real_conn = real_conn
            def execute(self, sql, *args, **kwargs):
                queries_run.append(sql)
                return self.real_conn.execute(sql, *args, **kwargs)
            def executemany(self, sql, *args, **kwargs):
                queries_run.append(sql)
                return self.real_conn.executemany(sql, *args, **kwargs)
            def commit(self):
                return self.real_conn.commit()
            def close(self):
                return self.real_conn.close()

        original_get_conn = service._get_connection
        def spy_get_connection():
            real_conn = original_get_conn()
            return SpyConnection(real_conn)

        service._get_connection = spy_get_connection

        # Measure query count for 2 patients
        service.get_population_analytics(actor=self.doctor)
        count_for_2 = len(queries_run)

        # Clear queries log
        queries_run.clear()

        # Seed 5 more patients in database
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            for i in range(5):
                cursor.execute(
                    "INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
                    (f"pat-uuid-extra-{i}", f"Extra Patient {i}", 40, "Male", "2026-08-01T00:00:00")
                )
            conn.commit()
        finally:
            conn.close()

        # Measure query count for 7 patients
        service.get_population_analytics(actor=self.doctor)
        count_for_7 = len(queries_run)

        # Assert that the number of queries executed did not increase with N
        self.assertEqual(count_for_2, count_for_7, f"Query count increased from {count_for_2} to {count_for_7} when adding patients! N+1 pattern detected.")

    def test_population_analytics_semantic_scenarios(self):
        """Test semantic clinical analytics scenarios: 0/1/multiple patients, missing prediction/measurements, zero baseline, NaN/Infinity."""
        temp_db_fd, temp_db_path = tempfile.mkstemp(suffix=".db")
        try:
            from persistence.infrastructure.repository import SQLitePersistenceRepository
            clinical_repo = SQLitePersistenceRepository(db_path=temp_db_path)
            clinical_repo.initialize_db()

            service = ReportService(db_path=temp_db_path)

            # Scenario A: 0 patients
            analytics = service.get_population_analytics(actor=self.doctor)
            self.assertEqual(analytics.total_patients, 0)
            self.assertEqual(analytics.total_reports, 0)
            self.assertEqual(analytics.total_scans, 0)
            self.assertEqual(analytics.progression_distribution, {})

            # Setup Scenario B: Patient with no reports
            conn = sqlite3.connect(temp_db_path)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
                           ("pat-no-reports", "No Reports Patient", 45, "Male", "2026-08-01"))
            conn.commit()
            conn.close()

            analytics = service.get_population_analytics(actor=self.doctor)
            self.assertEqual(analytics.total_patients, 1)
            # A patient with no reports has timeline status "EMPTY"
            self.assertEqual(analytics.progression_distribution.get("EMPTY"), 1)

            # Setup Scenario C: Patient with one report (Baseline)
            conn = sqlite3.connect(temp_db_path)
            cursor = conn.cursor()
            cursor.execute("INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
                           ("pat-one-report", "One Report Patient", 50, "Female", "2026-08-01"))
            cursor.execute("INSERT INTO mri_scans (patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at) VALUES (?, ?, ?, ?, ?, ?);",
                           ("pat-one-report", "scans/1.png", 1.0, "Dr. House", "2026-08-02", "2026-08-02"))
            scan_id = cursor.lastrowid
            cursor.execute("""
                INSERT INTO predictions (
                    scan_id, predicted_class, confidence_score, prob_glioma, prob_meningioma,
                    prob_pituitary, prob_no_tumor, tumor_pixel_count, tumor_area_mm2,
                    tumor_percentage_brain, tumor_percentage_image, estimated_brain_pixel_count,
                    rule_based_severity, severity_rule_description, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (scan_id, "Glioma", 0.95, 0.95, 0.0, 0.0, 0.05, 100, 150.0, 1.5, 0.5, 10000, "Medium", "desc", "2026-08-02"))
            pred_id = cursor.lastrowid
            cursor.execute("INSERT INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                           (1001, "RPT-1001", "pat-one-report", 1, "FINAL", "2026-08-02", "2026-08-02"))
            cursor.execute("INSERT INTO report_versions (report_id, version_number, prediction_id, status, reason, pdf_path, json_path, checksum, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);",
                           (1001, 1, pred_id, "FINAL", "Baseline", "rep.pdf", "rep.json", "hash", "2026-08-02"))
            conn.commit()
            conn.close()

            analytics = service.get_population_analytics(actor=self.doctor)
            self.assertEqual(analytics.total_patients, 2)
            self.assertEqual(analytics.progression_distribution.get("EMPTY"), 1)
            # A patient with a single report has timeline status "STABLE"
            self.assertEqual(analytics.progression_distribution.get("STABLE"), 1)

            # Write temporary JSON files for NaN and Infinity mock data to test file enrichment sanitization
            import json
            json_nz1_path = os.path.abspath(os.path.join(os.path.dirname(temp_db_path), "nz1.json"))
            json_nz2_path = os.path.abspath(os.path.join(os.path.dirname(temp_db_path), "nz2.json"))

            with open(json_nz1_path, "w") as f:
                json.dump({
                    "classification": {"confidence_score": "nan"}
                }, f)

            with open(json_nz2_path, "w") as f:
                json.dump({
                    "classification": {"confidence_score": "inf"},
                    "segmentation": {"tumor_area_mm2": "inf", "tumor_percentage_brain": "-inf"}
                }, f)

            # Setup Scenario D: Multiple reports (Stable, Progression, NULL values, Zero Baseline, NaN/Infinity)
            conn = sqlite3.connect(temp_db_path)
            cursor = conn.cursor()

            # Patient with multiple reports (Progression: area increase > 10%)
            cursor.execute("INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
                           ("pat-prog", "Prog Patient", 60, "Male", "2026-08-01"))

            # Scan 1: Baseline
            cursor.execute("INSERT INTO mri_scans (patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at) VALUES (?, ?, ?, ?, ?, ?);",
                           ("pat-prog", "scans/p1.png", 1.0, "Dr. House", "2026-08-02", "2026-08-02"))
            scan_p1_id = cursor.lastrowid
            cursor.execute("""
                INSERT INTO predictions (
                    scan_id, predicted_class, confidence_score, prob_glioma, prob_meningioma,
                    prob_pituitary, prob_no_tumor, tumor_pixel_count, tumor_area_mm2,
                    tumor_percentage_brain, tumor_percentage_image, estimated_brain_pixel_count,
                    rule_based_severity, severity_rule_description, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (scan_p1_id, "Glioma", 0.90, 0.90, 0.05, 0.0, 0.05, 100, 100.0, 1.0, 0.3, 10000, "Medium", "desc", "2026-08-02"))
            pred_p1_id = cursor.lastrowid
            cursor.execute("INSERT INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                           (1002, "RPT-1002", "pat-prog", 1, "FINAL", "2026-08-02", "2026-08-02"))
            cursor.execute("INSERT INTO report_versions (report_id, version_number, prediction_id, status, reason, pdf_path, json_path, checksum, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);",
                           (1002, 1, pred_p1_id, "FINAL", "Baseline", "p1.pdf", "p1.json", "h1", "2026-08-02"))

            # Scan 2: Follow-up (Area increase to 120.0, which is a 20% increase)
            cursor.execute("INSERT INTO mri_scans (patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at) VALUES (?, ?, ?, ?, ?, ?);",
                           ("pat-prog", "scans/p2.png", 1.0, "Dr. House", "2026-08-08", "2026-08-08"))
            scan_p2_id = cursor.lastrowid
            cursor.execute("""
                INSERT INTO predictions (
                    scan_id, predicted_class, confidence_score, prob_glioma, prob_meningioma,
                    prob_pituitary, prob_no_tumor, tumor_pixel_count, tumor_area_mm2,
                    tumor_percentage_brain, tumor_percentage_image, estimated_brain_pixel_count,
                    rule_based_severity, severity_rule_description, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (scan_p2_id, "Glioma", 0.95, 0.95, 0.02, 0.01, 0.02, 120, 120.0, 1.2, 0.4, 10000, "Medium", "desc", "2026-08-08"))
            pred_p2_id = cursor.lastrowid
            cursor.execute("INSERT INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                           (1003, "RPT-1003", "pat-prog", 1, "FINAL", "2026-08-08", "2026-08-08"))
            cursor.execute("INSERT INTO report_versions (report_id, version_number, prediction_id, status, reason, pdf_path, json_path, checksum, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);",
                           (1003, 1, pred_p2_id, "FINAL", "Follow-up", "p2.pdf", "p2.json", "h2", "2026-08-08"))

            # Patient with Zero Baseline and non-finite NaN / Inf values
            cursor.execute("INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
                           ("pat-nan-zero", "NaN Zero Patient", 30, "Female", "2026-08-01"))

            # Scan 1: Area = 0.0 (Zero Baseline)
            cursor.execute("INSERT INTO mri_scans (patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at) VALUES (?, ?, ?, ?, ?, ?);",
                           ("pat-nan-zero", "scans/nz1.png", 1.0, "Dr. House", "2026-08-02", "2026-08-02"))
            scan_nz1_id = cursor.lastrowid
            cursor.execute("""
                INSERT INTO predictions (
                    scan_id, predicted_class, confidence_score, prob_glioma, prob_meningioma,
                    prob_pituitary, prob_no_tumor, tumor_pixel_count, tumor_area_mm2,
                    tumor_percentage_brain, tumor_percentage_image, estimated_brain_pixel_count,
                    rule_based_severity, severity_rule_description, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (scan_nz1_id, "No Tumor", 0.0, 0.0, 0.0, 0.0, 1.0, 0, 0.0, 0.0, 0.0, 10000, "Low", "desc", "2026-08-02"))
            pred_nz1_id = cursor.lastrowid
            cursor.execute("INSERT INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                           (1004, "RPT-1004", "pat-nan-zero", 1, "FINAL", "2026-08-02", "2026-08-02"))
            cursor.execute("INSERT INTO report_versions (report_id, version_number, prediction_id, status, reason, pdf_path, json_path, checksum, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);",
                           (1004, 1, pred_nz1_id, "FINAL", "Baseline", "nz1.pdf", json_nz1_path, "h3", "2026-08-02"))

            # Scan 2: Area = Infinity (non-finite)
            cursor.execute("INSERT INTO mri_scans (patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at) VALUES (?, ?, ?, ?, ?, ?);",
                           ("pat-nan-zero", "scans/nz2.png", 1.0, "Dr. House", "2026-08-08", "2026-08-08"))
            scan_nz2_id = cursor.lastrowid
            cursor.execute("""
                INSERT INTO predictions (
                    scan_id, predicted_class, confidence_score, prob_glioma, prob_meningioma,
                    prob_pituitary, prob_no_tumor, tumor_pixel_count, tumor_area_mm2,
                    tumor_percentage_brain, tumor_percentage_image, estimated_brain_pixel_count,
                    rule_based_severity, severity_rule_description, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (scan_nz2_id, "Glioma", 1.0, 1.0, 0.0, 0.0, 0.0, 100, 999.0, 9.9, 0.5, 10000, "High", "desc", "2026-08-08"))
            pred_nz2_id = cursor.lastrowid
            cursor.execute("INSERT INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                           (1005, "RPT-1005", "pat-nan-zero", 1, "FINAL", "2026-08-08", "2026-08-08"))
            cursor.execute("INSERT INTO report_versions (report_id, version_number, prediction_id, status, reason, pdf_path, json_path, checksum, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);",
                           (1005, 1, pred_nz2_id, "FINAL", "Follow-up", "nz2.pdf", json_nz2_path, "h4", "2026-08-08"))

            conn.commit()
            conn.close()

            # Execute and assert analytics
            analytics = service.get_population_analytics(actor=self.doctor)

            # 4 total patients: pat-no-reports, pat-one-report, pat-prog, pat-nan-zero
            self.assertEqual(analytics.total_patients, 4)
            self.assertEqual(analytics.total_reports, 5)
            self.assertEqual(analytics.total_scans, 5)

            # Verify progression distributions
            self.assertEqual(analytics.progression_distribution.get("EMPTY"), 1)  # pat-no-reports
            self.assertEqual(analytics.progression_distribution.get("STABLE"), 1) # pat-one-report
            self.assertEqual(analytics.progression_distribution.get("PROGRESSION"), 1) # pat-prog (20% increase)
            self.assertEqual(analytics.progression_distribution.get("CHANGED"), 1) # pat-nan-zero (classification changed No Tumor -> Glioma due to Infinity area)

            # Verify that averages are computed correctly from the SQLite database
            self.assertEqual(analytics.average_confidence, 0.76)
            self.assertEqual(analytics.average_tumor_area, 273.8)

        finally:
            # Clean up JSON files
            for p in [json_nz1_path, json_nz2_path]:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
            os.close(temp_db_fd)
            if os.path.exists(temp_db_path):
                try:
                    os.remove(temp_db_path)
                except Exception:
                    pass

    def test_population_analytics_semantic_equivalence(self):
        """Dedicated equivalence test comparing optimized population analytics against pre-optimized loop-based calculations."""
        service = ReportService(db_path=self.db_path)

        # 1. Compute pre-optimized expected values
        expected_prog_dist = {}
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows_patients = conn.execute("SELECT patient_id FROM patients;").fetchall()
            for row_pat in rows_patients:
                pat_id = row_pat["patient_id"]
                try:
                    timeline = service.get_patient_longitudinal_timeline(patient_id=pat_id, actor=self.doctor)
                    status = timeline.timeline_status
                    expected_prog_dist[status] = expected_prog_dist.get(status, 0) + 1
                except Exception:
                    expected_prog_dist["UNKNOWN"] = expected_prog_dist.get("UNKNOWN", 0) + 1
        finally:
            conn.close()

        # 2. Get optimized population analytics result
        optimized_result = service.get_population_analytics(actor=self.doctor)

        # 3. Assert exact match
        self.assertEqual(optimized_result.progression_distribution, expected_prog_dist,
                         f"Semantic mismatch! Expected: {expected_prog_dist}, Got: {optimized_result.progression_distribution}")


if __name__ == "__main__":
    unittest.main()
