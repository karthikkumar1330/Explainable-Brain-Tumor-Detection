import os
import unittest
import sqlite3
import datetime
import json
import math
from fastapi.testclient import TestClient

os.environ["DB_PATH"] = os.path.abspath("outputs/test_followup_comparison.db")

from run_api import app
from clinical_reporting.application.services import ReportService, ReportNotFoundException, VersionNotFoundException, ReportServiceException
from clinical_reporting.domain.entities import PatientMismatchException
from persistence.infrastructure.repository import SQLitePersistenceRepository
from security.domain.entities import Role, User
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.jwt_service import JWTService


class TestFollowUpComparisonEngine(unittest.TestCase):
    """F3.2 Follow-up MRI / Scan Comparison compliance tests."""

    def _create_test_hierarchy(self, conn, patient_id, prediction_id, scan_id, classification, area, severity, scan_date="2026-08-08"):
        conn.execute(
            "INSERT OR IGNORE INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
            (patient_id, f"Patient {patient_id}", 45, "Male", datetime.datetime.utcnow().isoformat())
        )
        conn.execute(
            "INSERT OR IGNORE INTO mri_scans (id, patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
            (scan_id, patient_id, f"scan_{scan_id}.png", 1.0, "Dr. Smith", scan_date, datetime.datetime.utcnow().isoformat())
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
        self.db_path = os.environ.get("DB_PATH", "outputs/test_followup_comparison.db")
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
        self.prev_json_path = os.path.abspath("outputs/clinical_reports/followup_prev.json")
        self.curr_json_path = os.path.abspath("outputs/clinical_reports/followup_curr.json")
        self.other_json_path = os.path.abspath("outputs/clinical_reports/followup_other.json")

        # Write prev.json (baseline)
        prev_json_data = {
            "patient": {"patient_id": "pat-uuid-999", "scan_date": "2026-08-08"},
            "classification": {"predicted_class": "Glioma", "confidence_score": 0.95},
            "segmentation": {
                "tumor_area_mm2": 50.0,
                "tumor_percentage_brain": 0.05,
                "shape_statistics": {
                    "major_axis_mm": 10.0,
                    "minor_axis_mm": 8.0,
                    "eccentricity": 0.6,
                    "orientation_deg": 45.0,
                    "perimeter_mm": 30.0,
                    "solidity": 0.95,
                    "circularity": 0.8,
                    "bbox_w_mm": 12.0,
                    "bbox_h_mm": 10.0
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

        # Write curr.json (follow-up)
        curr_json_data = {
            "patient": {"patient_id": "pat-uuid-999", "scan_date": "2026-08-09"},
            "classification": {"predicted_class": "Glioma", "confidence_score": 0.96},
            "segmentation": {
                "tumor_area_mm2": 55.0, # +10%
                "tumor_percentage_brain": 0.055,
                "shape_statistics": {
                    "major_axis_mm": 10.5,
                    "minor_axis_mm": 8.2,
                    "eccentricity": 0.61,
                    "orientation_deg": 46.0,
                    "perimeter_mm": 31.0,
                    "solidity": 0.94,
                    "circularity": 0.79,
                    "bbox_w_mm": 12.5,
                    "bbox_h_mm": 10.2
                }
            },
            "severity": {"category": "High"},
            "clinical_insight": {
                "key_findings": ["Enlargement noted"],
                "summary_narrative": "Enlargement noted"
            }
        }
        with open(self.curr_json_path, "w") as f:
            json.dump(curr_json_data, f)

        # Write other.json
        other_json_data = {
            "patient": {"patient_id": "pat-uuid-888", "scan_date": "2026-08-09"},
            "classification": {"predicted_class": "Meningioma", "confidence_score": 0.95},
            "segmentation": {
                "tumor_area_mm2": 30.0,
                "tumor_percentage_brain": 0.03,
                "shape_statistics": {
                    "major_axis_mm": 7.0,
                    "minor_axis_mm": 6.0,
                    "eccentricity": 0.5,
                    "orientation_deg": 30.0,
                    "perimeter_mm": 20.0,
                    "solidity": 0.90,
                    "circularity": 0.75,
                    "bbox_w_mm": 8.0,
                    "bbox_h_mm": 7.0
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
            self._create_test_hierarchy(conn, "pat-uuid-999", prediction_id=1, scan_id=1, classification="Glioma", area=50.0, severity="Medium", scan_date="2026-08-08")
            self._create_test_hierarchy(conn, "pat-uuid-999", prediction_id=2, scan_id=2, classification="Glioma", area=55.0, severity="High", scan_date="2026-08-09")

            # Create one scan/prediction for pat-uuid-888
            self._create_test_hierarchy(conn, "pat-uuid-888", prediction_id=3, scan_id=3, classification="Meningioma", area=30.0, severity="Low", scan_date="2026-08-09")

            # Create reports
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

            from tests.helpers.authorization_fixtures import assign_doctor_to_patient
            assign_doctor_to_patient(conn, "doctor@aurascan.ai", "pat-uuid-999")
            assign_doctor_to_patient(conn, "doctor@aurascan.ai", "pat-uuid-888")
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

    def test_successful_followup_comparison(self):
        """1. successful follow-up comparison"""
        service = ReportService(db_path=self.db_path)
        res = service.compare_followup_reports(
            previous_report_id=101,
            current_report_id=102,
            actor=self.doctor_user
        )
        self.assertEqual(res["patient_id"], "pat-uuid-999")
        self.assertEqual(res["previous_report"]["report_id"], 101)
        self.assertEqual(res["current_report"]["report_id"], 102)
        self.assertEqual(res["previous_report"]["scan_date"], "2026-08-08")
        self.assertEqual(res["current_report"]["scan_date"], "2026-08-09")

        # Verify tumor area metric
        area_m = next(m for m in res["metrics"] if m["name"] == "tumor_area_mm2")
        self.assertEqual(area_m["previous_value"], 50.0)
        self.assertEqual(area_m["current_value"], 55.0)
        self.assertEqual(area_m["absolute_change"], 5.0)
        self.assertEqual(area_m["percentage_change"], 10.0)
        self.assertEqual(area_m["status"], "INCREASED")

        # Verify bounding box dimensions
        bbox_w = next(m for m in res["metrics"] if m["name"] == "bbox_w_mm")
        self.assertEqual(bbox_w["previous_value"], 12.0)
        self.assertEqual(bbox_w["current_value"], 12.5)
        self.assertEqual(bbox_w["status"], "INCREASED")

        # Verify solidity and circularity
        sol = next(m for m in res["metrics"] if m["name"] == "solidity")
        self.assertEqual(sol["status"], "DECREASED")

        circ = next(m for m in res["metrics"] if m["name"] == "circularity")
        self.assertEqual(circ["status"], "DECREASED")

        # Verify deterministic summary text
        self.assertEqual(res["summary"]["text"], "Tumor area increased by 10.0%")
        self.assertEqual(res["summary"]["status"], "MEASUREMENTS_INCREASED")

    def test_same_patient_enforcement(self):
        """2. same-patient enforcement"""
        service = ReportService(db_path=self.db_path)
        with self.assertRaises(PatientMismatchException):
            service.compare_followup_reports(
                previous_report_id=101,
                current_report_id=103,
                actor=self.doctor_user
            )

    def test_rbac_authorization(self):
        """3. RBAC checks for Doctor, Patient, and wrong patient"""
        service = ReportService(db_path=self.db_path)

        # Doctor can compare
        res = service.compare_followup_reports(101, 102, actor=self.doctor_user)
        self.assertEqual(res["patient_id"], "pat-uuid-999")

        # Admin can compare (via admin user)
        res_admin = service.compare_followup_reports(101, 102, actor=self.admin_user)
        self.assertEqual(res_admin["patient_id"], "pat-uuid-999")

        # Owning Patient can compare
        res_pat = service.compare_followup_reports(101, 102, actor=self.patient_user)
        self.assertEqual(res_pat["patient_id"], "pat-uuid-999")

        # Wrong Patient gets ReportServiceException
        with self.assertRaises(ReportServiceException) as ctx:
            service.compare_followup_reports(101, 102, actor=self.wrong_patient_user)
        self.assertIn("Access denied", str(ctx.exception))

    def test_missing_previous_measurements(self):
        """4. missing previous measurements (UNAVAILABLE status)"""
        missing_json_path = os.path.abspath("outputs/clinical_reports/missing.json")
        missing_json = {
            "patient": {"patient_id": "pat-uuid-999"},
            "classification": {"predicted_class": "Glioma", "confidence_score": 0.95},
            "segmentation": {
                "tumor_area_mm2": None, # Missing!
                "shape_statistics": {
                    "bbox_w_mm": None,
                    "solidity": None
                }
            }
        }
        with open(missing_json_path, "w") as f:
            json.dump(missing_json, f)

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                (105, "REP-2026-0005", "pat-uuid-999", 1, "FINALIZED", "2026-08-08 10:00:00", "2026-08-08 10:00:00")
            )
            conn.execute(
                """INSERT INTO report_versions (
                    report_id, version_number, prediction_id, status, created_by, reason, created_at,
                    pdf_path, json_path, checksum, integrity_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
                (105, 1, 1, "FINAL", "doc-uuid-1", "Missing values test", "2026-08-08 10:00:00",
                 "outputs/clinical_reports/missing.pdf", missing_json_path, "dummy", "dummy")
            )
            conn.commit()
        finally:
            conn.close()

        service = ReportService(db_path=self.db_path)
        # Compare 105 (missing area) with 102 (valid area)
        res = service.compare_followup_reports(105, 102, actor=self.doctor_user)

        area_m = next(m for m in res["metrics"] if m["name"] == "tumor_area_mm2")
        self.assertIsNone(area_m["previous_value"])
        self.assertEqual(area_m["status"], "UNAVAILABLE")
        self.assertEqual(res["summary"]["status"], "UNAVAILABLE")
        self.assertEqual(res["summary"]["text"], "Comparison unavailable because previous measurement is missing")

        if os.path.exists(missing_json_path):
            os.remove(missing_json_path)

    def test_zero_baseline_values(self):
        """5. zero baseline values (null percentage change)"""
        zero_json_path = os.path.abspath("outputs/clinical_reports/zero.json")
        zero_json = {
            "patient": {"patient_id": "pat-uuid-999"},
            "classification": {"predicted_class": "Glioma", "confidence_score": 0.0},
            "segmentation": {
                "tumor_area_mm2": 0.0,
                "shape_statistics": {
                    "bbox_w_mm": 0.0
                }
            }
        }
        with open(zero_json_path, "w") as f:
            json.dump(zero_json, f)

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                (106, "REP-2026-0006", "pat-uuid-999", 1, "FINALIZED", "2026-08-08 10:00:00", "2026-08-08 10:00:00")
            )
            conn.execute(
                """INSERT INTO report_versions (
                    report_id, version_number, prediction_id, status, created_by, reason, created_at,
                    pdf_path, json_path, checksum, integrity_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
                (106, 1, 1, "FINAL", "doc-uuid-1", "Zero values test", "2026-08-08 10:00:00",
                 "outputs/clinical_reports/zero.pdf", zero_json_path, "dummy", "dummy")
            )
            conn.commit()
        finally:
            conn.close()

        service = ReportService(db_path=self.db_path)
        # Compare 106 (zero baseline) with 102 (valid current)
        res = service.compare_followup_reports(106, 102, actor=self.doctor_user)

        area_m = next(m for m in res["metrics"] if m["name"] == "tumor_area_mm2")
        self.assertEqual(area_m["previous_value"], 0.0)
        self.assertIsNone(area_m["percentage_change"])
        self.assertEqual(area_m["status"], "INCREASED")

        if os.path.exists(zero_json_path):
            os.remove(zero_json_path)

    def test_null_nan_infinity_values(self):
        """6, 7, 8, 9: Null, NaN, Infinity, -Infinity value handling"""
        special_json_path = os.path.abspath("outputs/clinical_reports/special.json")
        special_json = {
            "patient": {"patient_id": "pat-uuid-999"},
            "classification": {"predicted_class": "Glioma", "confidence_score": float("nan")},
            "segmentation": {
                "tumor_area_mm2": float("inf"),
                "tumor_percentage_brain": float("-inf"),
                "shape_statistics": {
                    "bbox_w_mm": None,
                    "bbox_h_mm": 10.0
                }
            }
        }
        with open(special_json_path, "w") as f:
            json.dump(special_json, f)

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                (107, "REP-2026-0007", "pat-uuid-999", 1, "FINALIZED", "2026-08-08 10:00:00", "2026-08-08 10:00:00")
            )
            conn.execute(
                """INSERT INTO report_versions (
                    report_id, version_number, prediction_id, status, created_by, reason, created_at,
                    pdf_path, json_path, checksum, integrity_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
                (107, 1, 1, "FINAL", "doc-uuid-1", "Special values test", "2026-08-08 10:00:00",
                 "outputs/clinical_reports/special.pdf", special_json_path, "dummy", "dummy")
            )
            conn.commit()
        finally:
            conn.close()

        service = ReportService(db_path=self.db_path)
        res = service.compare_followup_reports(101, 107, actor=self.doctor_user)

        # Verify NaN is null in returned dict
        conf = next(m for m in res["metrics"] if m["name"] == "confidence")
        self.assertIsNone(conf["current_value"])
        self.assertEqual(conf["status"], "UNAVAILABLE")

        # Verify Inf/ -Inf are null
        area = next(m for m in res["metrics"] if m["name"] == "tumor_area_mm2")
        self.assertIsNone(area["current_value"])
        self.assertEqual(area["status"], "UNAVAILABLE")

        pct = next(m for m in res["metrics"] if m["name"] == "tumor_percentage_brain")
        self.assertIsNone(pct["current_value"])
        self.assertEqual(pct["status"], "UNAVAILABLE")

        if os.path.exists(special_json_path):
            os.remove(special_json_path)

    def test_stability_increased_decreased_classifications(self):
        """10, 11, 12: stable, increased, and decreased classifications"""
        service = ReportService(db_path=self.db_path)

        # Use custom tolerance where 6.0 is stable, others are not
        res = service.compare_followup_reports(
            previous_report_id=101,
            current_report_id=102,
            actor=self.doctor_user,
            tolerances={"tumor_area_mm2": 6.0} # 50 -> 55 (+5) is stable under 6.0 tolerance
        )
        area_m = next(m for m in res["metrics"] if m["name"] == "tumor_area_mm2")
        self.assertEqual(area_m["status"], "STABLE")
        self.assertEqual(res["summary"]["text"], "Tumor area remained stable")

        # Compare 102 -> 101 to check DECREASED
        res_dec = service.compare_followup_reports(
            previous_report_id=102,
            current_report_id=101,
            actor=self.doctor_user,
            tolerances={"tumor_area_mm2": 1.0} # 55 -> 50 (-5) is decreased
        )
        area_dec = next(m for m in res_dec["metrics"] if m["name"] == "tumor_area_mm2")
        self.assertEqual(area_dec["status"], "DECREASED")
        self.assertEqual(res_dec["summary"]["text"], "Tumor area decreased by 9.1%")

    def test_invalid_report_ids(self):
        """13. invalid report IDs (HTTP 404)"""
        headers = {"Authorization": f"Bearer {self.doctor_token}"}
        res = self.client.get("/api/reports/101/compare-followup/999", headers=headers)
        self.assertEqual(res.status_code, 404)

        res = self.client.get("/api/reports/999/compare-followup/102", headers=headers)
        self.assertEqual(res.status_code, 404)

    def test_missing_patient_in_db(self):
        """14. missing patient (HTTP 404)"""
        headers = {"Authorization": f"Bearer {self.doctor_token}"}
        # Delete patients and check response
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DELETE FROM reports;")
            conn.commit()
        finally:
            conn.close()

        res = self.client.get("/api/reports/101/compare-followup/102", headers=headers)
        self.assertEqual(res.status_code, 404)

    def test_api_serialization_safety(self):
        """15. API serialization (no NaN/Infinity literal tokens in JSON string)"""
        special_json_path = os.path.abspath("outputs/clinical_reports/serialization_special.json")
        special_json = {
            "patient": {"patient_id": "pat-uuid-999"},
            "classification": {"predicted_class": "Glioma", "confidence_score": float("nan")},
            "segmentation": {
                "tumor_area_mm2": float("inf"),
                "tumor_percentage_brain": float("-inf"),
                "shape_statistics": {
                    "bbox_w_mm": 12.0
                }
            }
        }
        with open(special_json_path, "w") as f:
            json.dump(special_json, f)

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                (101, "REP-2026-0001", "pat-uuid-999", 1, "FINALIZED", "2026-08-08 10:00:00", "2026-08-08 10:00:00")
            )
            conn.execute(
                "INSERT INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                (108, "REP-2026-0008", "pat-uuid-999", 1, "FINALIZED", "2026-08-09 10:00:00", "2026-08-09 10:00:00")
            )
            conn.execute(
                """INSERT INTO report_versions (
                    report_id, version_number, prediction_id, status, created_by, reason, created_at,
                    pdf_path, json_path, checksum, integrity_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
                (108, 1, 2, "FINAL", "doc-uuid-1", "Serialization special test", "2026-08-09 10:00:00",
                 "outputs/clinical_reports/special_ser.pdf", special_json_path, "dummy", "dummy")
            )
            conn.commit()
        finally:
            conn.close()

        headers = {"Authorization": f"Bearer {self.doctor_token}"}
        res = self.client.get("/api/reports/101/compare-followup/108", headers=headers)
        self.assertEqual(res.status_code, 200)

        # Check response body raw content for literals
        raw_text = res.content.decode("utf-8")
        self.assertNotIn("NaN", raw_text)
        self.assertNotIn("Infinity", raw_text)
        self.assertNotIn("-Infinity", raw_text)

        if os.path.exists(special_json_path):
            os.remove(special_json_path)

    def test_audit_event_generation(self):
        """16. audit event generation"""
        service = ReportService(db_path=self.db_path)
        # Execute comparison
        service.compare_followup_reports(101, 102, actor=self.doctor_user)

        # Query audit log
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute("SELECT * FROM security_audit_logs WHERE event_type = 'REPORT_COMPARED' ORDER BY timestamp DESC LIMIT 1;").fetchone()
            self.assertIsNotNone(row)
            self.assertIn("Compared follow-up with current report ID: 102", row["details"])
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
