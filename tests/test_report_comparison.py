import os
import unittest
import sqlite3
import datetime
import json
from fastapi.testclient import TestClient

os.environ["DB_PATH"] = os.path.abspath("outputs/test_report_comparison.db")

from run_api import app
from clinical_reporting.application.services import ReportService, ReportNotFoundException, VersionNotFoundException
from clinical_reporting.domain.entities import PatientMismatchException, ComparisonMetric, ReportComparison
from persistence.infrastructure.repository import SQLitePersistenceRepository
from security.domain.entities import Role, User
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.jwt_service import JWTService


class TestReportComparisonEngine(unittest.TestCase):
    """F3.1 Advanced Report Comparison Engine compliance tests."""

    def _create_test_hierarchy(self, conn, patient_id, prediction_id, scan_id, classification, area, severity):
        conn.execute(
            "INSERT OR IGNORE INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
            (patient_id, f"Patient {patient_id}", 45, "Male", datetime.datetime.utcnow().isoformat())
        )
        conn.execute(
            "INSERT OR IGNORE INTO mri_scans (id, patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
            (scan_id, patient_id, f"scan_{scan_id}.png", 1.0, "Dr. Smith", "2026-08-08", datetime.datetime.utcnow().isoformat())
        )
        conn.execute(
            """INSERT OR IGNORE INTO predictions (
                id, scan_id, predicted_class, confidence_score, prob_glioma, prob_meningioma,
                prob_pituitary, prob_no_tumor, tumor_pixel_count, tumor_area_mm2,
                tumor_percentage_brain, tumor_percentage_image, estimated_brain_pixel_count,
                rule_based_severity, severity_rule_description, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
            (prediction_id, scan_id, classification, 0.95, 0.95, 0.02, 0.02, 0.01, 500, area, 5.0, 2.0, 10000, severity, f"Severity is {severity}", datetime.datetime.utcnow().isoformat())
        )

    def setUp(self):
        self.db_path = os.environ.get("DB_PATH", "outputs/test_report_comparison.db")
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

        import api.infrastructure.routes as api_routes
        import api.routes.auth_routes as auth_routes
        api_routes.DEFAULT_DB_PATH = self.db_path
        auth_routes.DEFAULT_DB_PATH = self.db_path

        self.persistence_repo = SQLitePersistenceRepository(db_path=self.db_path)
        self.persistence_repo.initialize_db()

        self.test_client_ctx = TestClient(app)
        self.client = self.test_client_ctx.__enter__()

        self.user_repo = SQLiteUserRepository(db_path=self.db_path)
        self.user_repo.initialize_security_tables()
        self.admin_user = self.user_repo.bootstrap_admin()

        from security.infrastructure.password import PasswordHasher
        pass_hash = PasswordHasher.hash_password("Doctor@123")

        # Create reports output directory
        os.makedirs("outputs/clinical_reports", exist_ok=True)
        self.prev_json_path = os.path.abspath("outputs/clinical_reports/prev.json")
        self.curr_json_path = os.path.abspath("outputs/clinical_reports/curr.json")
        self.other_json_path = os.path.abspath("outputs/clinical_reports/other.json")

        # Write prev.json
        prev_json_data = {
            "patient": {"patient_id": "pat-uuid-999"},
            "classification": {"predicted_class": "Glioma", "confidence_score": 0.95},
            "segmentation": {
                "tumor_area_mm2": 50.0,
                "tumor_percentage_brain": 5.0,
                "shape_statistics": {
                    "major_axis_mm": 10.0,
                    "minor_axis_mm": 8.0,
                    "eccentricity": 0.6,
                    "orientation_deg": 45.0,
                    "perimeter_mm": 30.0,
                    "solidity": 0.95,
                    "circularity": 0.8
                }
            },
            "severity": {"category": "Medium"},
            "clinical_insight": {
                "key_findings": ["Initial tumor observation"],
                "summary_narrative": "Initial tumor observation"
            }
        }
        with open(self.prev_json_path, "w") as f:
            json.dump(prev_json_data, f)

        # Write curr.json
        curr_json_data = {
            "patient": {"patient_id": "pat-uuid-999"},
            "classification": {"predicted_class": "Glioma", "confidence_score": 0.95},
            "segmentation": {
                "tumor_area_mm2": 75.0,
                "tumor_percentage_brain": 7.5,
                "shape_statistics": {
                    "major_axis_mm": 12.0,
                    "minor_axis_mm": 9.0,
                    "eccentricity": 0.65,
                    "orientation_deg": 50.0,
                    "perimeter_mm": 35.0,
                    "solidity": 0.96,
                    "circularity": 0.82
                }
            },
            "severity": {"category": "High"},
            "clinical_insight": {
                "key_findings": ["Tumor enlargement observed"],
                "summary_narrative": "Tumor enlargement observed"
            }
        }
        with open(self.curr_json_path, "w") as f:
            json.dump(curr_json_data, f)

        # Write other.json
        other_json_data = {
            "patient": {"patient_id": "pat-uuid-888"},
            "classification": {"predicted_class": "Meningioma", "confidence_score": 0.95},
            "segmentation": {
                "tumor_area_mm2": 30.0,
                "tumor_percentage_brain": 3.0,
                "shape_statistics": {
                    "major_axis_mm": 7.0,
                    "minor_axis_mm": 6.0,
                    "eccentricity": 0.5,
                    "orientation_deg": 30.0,
                    "perimeter_mm": 20.0,
                    "solidity": 0.90,
                    "circularity": 0.75
                }
            },
            "severity": {"category": "Low"},
            "clinical_insight": {
                "key_findings": ["Initial check"],
                "summary_narrative": "Initial check"
            }
        }
        with open(self.other_json_path, "w") as f:
            json.dump(other_json_data, f)

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
                ("pat-uuid-888", "wrong_patient@aurascan.ai", pass_hash, "Alice Cooper", Role.PATIENT.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            
            # Create two scans/predictions for pat-uuid-999
            self._create_test_hierarchy(conn, "pat-uuid-999", prediction_id=1, scan_id=1, classification="Glioma", area=50.0, severity="Medium")
            self._create_test_hierarchy(conn, "pat-uuid-999", prediction_id=2, scan_id=2, classification="Glioma", area=75.0, severity="High")
            
            # Create one scan/prediction for pat-uuid-888
            self._create_test_hierarchy(conn, "pat-uuid-888", prediction_id=3, scan_id=3, classification="Meningioma", area=30.0, severity="Low")
            
            # Create reports
            # Report 1: previous report for pat-uuid-999
            conn.execute(
                "INSERT INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                (101, "REP-2026-0001", "pat-uuid-999", 1, "FINALIZED", "2026-08-08 10:00:00", "2026-08-08 10:00:00")
            )
            conn.execute(
                """INSERT INTO report_versions (
                    report_id, version_number, prediction_id, status, created_by, reason, created_at,
                    pdf_path, json_path, checksum, integrity_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
                (101, 1, 1, "FINAL", "doc-uuid-1", "Initial draft", "2026-08-08 10:00:00",
                 "outputs/clinical_reports/prev.pdf", self.prev_json_path, "dummy_checksum", "dummy_integrity")
            )
            
            # Report 2: current report for pat-uuid-999
            conn.execute(
                "INSERT INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                (102, "REP-2026-0002", "pat-uuid-999", 1, "FINALIZED", "2026-08-09 10:00:00", "2026-08-09 10:00:00")
            )
            conn.execute(
                """INSERT INTO report_versions (
                    report_id, version_number, prediction_id, status, created_by, reason, created_at,
                    pdf_path, json_path, checksum, integrity_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
                (102, 1, 2, "FINAL", "doc-uuid-1", "Followup scan", "2026-08-09 10:00:00",
                 "outputs/clinical_reports/curr.pdf", self.curr_json_path, "dummy_checksum", "dummy_integrity")
            )

            # Report 3: report for patient 888
            conn.execute(
                "INSERT INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                (103, "REP-2026-0003", "pat-uuid-888", 1, "FINALIZED", "2026-08-09 11:00:00", "2026-08-09 11:00:00")
            )
            conn.execute(
                """INSERT INTO report_versions (
                    report_id, version_number, prediction_id, status, created_by, reason, created_at,
                    pdf_path, json_path, checksum, integrity_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
                (103, 1, 3, "FINAL", "doc-uuid-1", "Initial report", "2026-08-09 11:00:00",
                 "outputs/clinical_reports/other.pdf", self.other_json_path, "dummy_checksum", "dummy_integrity")
            )
            
            conn.commit()
        finally:
            conn.close()

        self.doctor_user = self.user_repo.get_by_email("doctor@aurascan.ai")
        self.patient_user = self.user_repo.get_by_email("patient@aurascan.ai")
        self.wrong_patient_user = self.user_repo.get_by_email("wrong_patient@aurascan.ai")

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
        self.test_client_ctx.__exit__(None, None, None)
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass
        
        # Cleanup mock JSON files
        for p in [self.prev_json_path, self.curr_json_path, self.other_json_path]:
            if os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass

    def test_compare_reports_service_success(self):
        """Tests successful report comparison at the application service layer."""
        service = ReportService(db_path=self.db_path)
        res = service.compare_reports(
            previous_report_id=101,
            current_report_id=102,
            actor=self.doctor_user
        )
        self.assertEqual(res["patient_id"], "pat-uuid-999")
        self.assertEqual(res["previous_report"]["report_id"], 101)
        self.assertEqual(res["current_report"]["report_id"], 102)

        # Check numeric comparison of tumor_area_mm2
        area_metric = next(m for m in res["metrics"] if m["name"] == "tumor_area_mm2")
        self.assertEqual(area_metric["previous_value"], 50.0)
        self.assertEqual(area_metric["current_value"], 75.0)
        self.assertEqual(area_metric["absolute_difference"], 25.0)
        self.assertEqual(area_metric["percentage_difference"], 50.0)
        self.assertEqual(area_metric["direction"], "INCREASED")
        self.assertEqual(area_metric["interpretation_category"], "OBSERVED_INCREASE")

        # Check severity comparison (Medium -> High)
        sev_metric = next(m for m in res["metrics"] if m["name"] == "severity")
        self.assertEqual(sev_metric["previous_value"], "MEDIUM")
        self.assertEqual(sev_metric["current_value"], "HIGH")
        self.assertEqual(sev_metric["direction"], "INCREASED")
        self.assertEqual(sev_metric["interpretation_category"], "OBSERVED_INCREASE")

        # Check shape statistics parsed from JSON
        major_axis = next(m for m in res["metrics"] if m["name"] == "major_axis")
        self.assertEqual(major_axis["previous_value"], 10.0)
        self.assertEqual(major_axis["current_value"], 12.0)
        self.assertEqual(major_axis["absolute_difference"], 2.0)
        self.assertEqual(major_axis["percentage_difference"], 20.0)
        self.assertEqual(major_axis["direction"], "INCREASED")

        # Check summary status and disclaimer
        self.assertEqual(res["summary"]["status"], "MULTIPLE_CHANGES")
        self.assertIn("tumor area increased", res["summary"]["text"].lower())
        self.assertIn("is not a medical diagnosis", res["disclaimer"])

    def test_compare_reports_patient_mismatch(self):
        """Tests that trying to compare reports of different patients raises PatientMismatchException."""
        service = ReportService(db_path=self.db_path)
        with self.assertRaises(PatientMismatchException):
            service.compare_reports(
                previous_report_id=101,
                current_report_id=103,
                actor=self.doctor_user
            )

    def test_compare_reports_rbac_enforcement(self):
        """Tests that patients cannot compare reports if they don't own them."""
        service = ReportService(db_path=self.db_path)
        
        # Patient 999 comparing their own reports -> success
        res = service.compare_reports(101, 102, actor=self.patient_user)
        self.assertEqual(res["patient_id"], "pat-uuid-999")

        # Patient 888 comparing patient 999's reports -> access denied Exception
        from clinical_reporting.application.services import ReportServiceException
        with self.assertRaises(ReportServiceException) as ctx:
            service.compare_reports(101, 102, actor=self.wrong_patient_user)
        self.assertIn("Access denied", str(ctx.exception))

    def test_compare_reports_api_endpoints(self):
        """Tests the FastAPI routes for report comparison."""
        headers = {"Authorization": f"Bearer {self.doctor_token}"}
        
        # Valid comparison
        res = self.client.get("/api/reports/101/compare/102", headers=headers)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["summary"]["status"], "MULTIPLE_CHANGES")

        # Different patients
        res = self.client.get("/api/reports/101/compare/103", headers=headers)
        self.assertEqual(res.status_code, 400)
        self.assertIn("different patients", res.json()["detail"].lower())

        # Non-existent report
        res = self.client.get("/api/reports/101/compare/999", headers=headers)
        self.assertEqual(res.status_code, 404)

        # RBAC unauthorized check
        wrong_headers = {"Authorization": f"Bearer {self.wrong_patient_token}"}
        res = self.client.get("/api/reports/101/compare/102", headers=wrong_headers)
        self.assertEqual(res.status_code, 403)

    def test_compare_reports_non_finite_values(self):
        """Tests that non-finite float values (NaN, Inf, -Inf) are serialized as JSON null."""
        nan_json_path = os.path.abspath("outputs/clinical_reports/nan_test.json")
        nan_json_data = {
            "patient": {"patient_id": "pat-uuid-999"},
            "classification": {"predicted_class": "Glioma", "confidence_score": float("nan")},
            "segmentation": {
                "tumor_area_mm2": float("inf"),
                "tumor_percentage_brain": float("-inf"),
                "shape_statistics": {
                    "major_axis_mm": float("nan"),
                    "minor_axis_mm": 8.0,
                    "eccentricity": float("inf"),
                    "orientation_deg": 45.0,
                    "perimeter_mm": float("-inf"),
                    "solidity": 0.95,
                    "circularity": 0.8
                }
            },
            "severity": {"category": "Medium"},
            "clinical_insight": {
                "key_findings": ["Non-finite test findings"],
                "summary_narrative": "Non-finite test findings"
            }
        }
        with open(nan_json_path, "w") as f:
            json.dump(nan_json_data, f)

        conn = sqlite3.connect(self.db_path)
        try:
            # Report 104: report with NaN/Inf for pat-uuid-999
            conn.execute(
                "INSERT INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                (104, "REP-2026-0004", "pat-uuid-999", 1, "FINALIZED", "2026-08-09 12:00:00", "2026-08-09 12:00:00")
            )
            conn.execute(
                """INSERT INTO report_versions (
                    report_id, version_number, prediction_id, status, created_by, reason, created_at,
                    pdf_path, json_path, checksum, integrity_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
                (104, 1, 2, "FINAL", "doc-uuid-1", "Non-finite test version", "2026-08-09 12:00:00",
                 "outputs/clinical_reports/nan_test.pdf", nan_json_path, "dummy_checksum", "dummy_integrity")
            )
            conn.commit()
        finally:
            conn.close()

        headers = {"Authorization": f"Bearer {self.doctor_token}"}
        # Compare 101 (valid) with 104 (has NaN/Inf)
        res = self.client.get("/api/reports/101/compare/104", headers=headers)
        self.assertEqual(res.status_code, 200)

        # Verify response is valid JSON and contains no NaN/Infinity literal tokens
        raw_content = res.content.decode("utf-8")
        self.assertNotIn("NaN", raw_content)
        self.assertNotIn("Infinity", raw_content)
        self.assertNotIn("-Infinity", raw_content)

        # Parse JSON and check that sanitized values are null
        data = res.json()
        
        confidence = next(m for m in data["metrics"] if m["name"] == "confidence")
        self.assertIsNone(confidence["current_value"])
        self.assertIsNone(confidence["absolute_difference"])
        self.assertIsNone(confidence["percentage_difference"])

        area = next(m for m in data["metrics"] if m["name"] == "tumor_area_mm2")
        self.assertIsNone(area["current_value"])
        self.assertIsNone(area["absolute_difference"])
        self.assertIsNone(area["percentage_difference"])

        major_axis = next(m for m in data["metrics"] if m["name"] == "major_axis")
        self.assertIsNone(major_axis["current_value"])
        self.assertIsNone(major_axis["absolute_difference"])
        self.assertIsNone(major_axis["percentage_difference"])

        # Clean up nan file
        if os.path.exists(nan_json_path):
            try:
                os.remove(nan_json_path)
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
