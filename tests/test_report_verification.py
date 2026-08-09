import os
import unittest
import sqlite3
import datetime
import hashlib
import json
from fastapi.testclient import TestClient

os.environ["DB_PATH"] = os.path.abspath("outputs/test_report_verification.db")

from run_api import app
from clinical_reporting.application.services import ReportService, ReportNotFoundException
from clinical_reporting.domain.entities import (
    ReportStatus,
    canonicalize_report_data,
    generate_integrity_hash,
    VerificationState
)
from clinical_reporting.application.qr_service import QRGeneratorService
from persistence.infrastructure.repository import SQLitePersistenceRepository
from security.domain.entities import Role, User
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.jwt_service import JWTService


class TestReportVerificationAndIntegrity(unittest.TestCase):
    """Phase F2.2 Automated Unit and Integration tests."""

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
        self.db_path = os.environ.get("DB_PATH", "outputs/test_report_verification.db")
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

        import api.infrastructure.routes as api_routes
        import api.routes.auth_routes as auth_routes
        api_routes.DEFAULT_DB_PATH = self.db_path
        auth_routes.DEFAULT_DB_PATH = self.db_path

        # Initialize DB
        self.persistence_repo = SQLitePersistenceRepository(db_path=self.db_path)
        self.persistence_repo.initialize_db()

        # Set up fastapi test client
        self.test_client_ctx = TestClient(app)
        self.client = self.test_client_ctx.__enter__()

        # Bootstrap user repo & doctor / admin users
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
            self._create_test_hierarchy(conn)
            conn.commit()
        finally:
            conn.close()

        self.doctor_user = self.user_repo.get_by_email("doctor@aurascan.ai")
        self.patient_user = self.user_repo.get_by_email("patient@aurascan.ai")

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

        self.admin_headers = {"Authorization": f"Bearer {self.admin_token}"}
        self.doctor_headers = {"Authorization": f"Bearer {self.doctor_token}"}
        self.patient_headers = {"Authorization": f"Bearer {self.patient_token}"}

        self.service = ReportService(db_path=self.db_path)

    def tearDown(self):
        self.test_client_ctx.__exit__()
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def test_01_canonicalization_is_deterministic(self):
        """1. Verify canonicalization is deterministic and ignores unstable keys."""
        report_data_1 = {
            "patient": {"patient_id": "P001", "name": "Alice"},
            "classification": {"predicted_class": "Glioma", "confidence_score": 0.95},
            "files": {"pdf_path": "outputs/some_path.pdf", "json_path": "outputs/json.json"},
            "timestamp": "2026-08-09 07:19:13",
            "latency_sec": {"classification": 0.045}
        }
        report_data_2 = {
            "classification": {"confidence_score": 0.95, "predicted_class": "Glioma"},
            "patient": {"name": "Alice", "patient_id": "P001"},
            "files": {"pdf_path": "outputs/diff_path.pdf", "json_path": "outputs/diff.json"},
            "timestamp": "2026-08-09 10:00:00",
            "latency_sec": {"classification": 0.999}
        }
        c1 = canonicalize_report_data(report_data_1)
        c2 = canonicalize_report_data(report_data_2)
        self.assertEqual(c1, c2)

    def test_02_identical_report_produces_identical_hash(self):
        """2. Verify identical report payload produces identical hash."""
        report_data = {
            "patient": {"patient_id": "P001", "name": "Alice"},
            "classification": {"predicted_class": "Glioma"}
        }
        h1 = generate_integrity_hash(report_data)
        h2 = generate_integrity_hash(report_data)
        self.assertEqual(h1, h2)

    def test_03_modified_report_produces_different_hash(self):
        """3. Verify modified report payload produces different hash."""
        report_data_1 = {
            "patient": {"patient_id": "P001", "name": "Alice"},
            "classification": {"predicted_class": "Glioma"}
        }
        report_data_2 = {
            "patient": {"patient_id": "P001", "name": "Alice"},
            "classification": {"predicted_class": "Meningioma"}
        }
        h1 = generate_integrity_hash(report_data_1)
        h2 = generate_integrity_hash(report_data_2)
        self.assertNotEqual(h1, h2)

    def test_04_sha256_format_is_valid(self):
        """4. Verify integrity hash is a valid 64-character SHA-256 hex string."""
        report_data = {"test": "data"}
        h = generate_integrity_hash(report_data)
        self.assertEqual(len(h), 64)
        self.assertTrue(all(c in "0123456789abcdef" for c in h.lower()))

    def test_05_verification_token_is_generated(self):
        """5. Verify verification token is generated on report creation."""
        os.makedirs("outputs", exist_ok=True)
        json_path = "outputs/test_05.json"
        with open(json_path, "w") as f:
            json.dump({"patient": {"patient_id": "pat-uuid-999"}}, f)

        report = self.service.create_report(
            patient_id="pat-uuid-999",
            created_by="doctor@aurascan.ai",
            report_type="MRI Brain Scan",
            pdf_path="outputs/test_05.pdf",
            json_path=json_path,
            prediction_id=1,
            initial_status=ReportStatus.GENERATED
        )
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute("SELECT verification_token, integrity_hash FROM reports WHERE report_id = ?;", (report.report_id,)).fetchone()
            self.assertIsNotNone(row["verification_token"])
            self.assertIsNotNone(row["integrity_hash"])
        finally:
            conn.close()

    def test_06_token_is_not_based_only_on_sequential_id(self):
        """6. Verify token is sufficiently secure and unpredictable, not sequential."""
        os.makedirs("outputs", exist_ok=True)
        json_path_1 = "outputs/test_06_1.json"
        with open(json_path_1, "w") as f:
            json.dump({"patient": {"patient_id": "pat-1"}}, f)
        json_path_2 = "outputs/test_06_2.json"
        with open(json_path_2, "w") as f:
            json.dump({"patient": {"patient_id": "pat-2"}}, f)

        r1 = self.service.create_report("pat-uuid-999", "doc", "Type", "outputs/test_06_1.pdf", json_path_1, 1)
        r2 = self.service.create_report("pat-uuid-999", "doc", "Type", "outputs/test_06_2.pdf", json_path_2, 1)

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            tok1 = conn.execute("SELECT verification_token FROM reports WHERE report_id = ?;", (r1.report_id,)).fetchone()[0]
            tok2 = conn.execute("SELECT verification_token FROM reports WHERE report_id = ?;", (r2.report_id,)).fetchone()[0]
            self.assertNotEqual(tok1, tok2)
            self.assertGreater(len(tok1), 10)
        finally:
            conn.close()

    def test_07_qr_payload_contains_no_patient_data(self):
        """7. Verify QR payload contains no clinical/patient data."""
        qr_svc = QRGeneratorService(verification_base_url="http://aurascan.ai/verify")
        token = "securetokenabc123"
        url = qr_svc.generate_verification_url(token)
        self.assertNotIn("Bob", url)
        self.assertNotIn("Glioma", url)
        self.assertEqual(url, "http://aurascan.ai/verify/securetokenabc123")

    def test_08_qr_url_identifies_exact_report_version(self):
        """8. Verify QR URL routes to the correct token path."""
        qr_svc = QRGeneratorService()
        url = qr_svc.generate_verification_url("abcdef")
        self.assertTrue(url.endswith("/abcdef"))

    def test_09_valid_report_verifies_successfully(self):
        """9. Verify validation returns VALID status for matching hashes."""
        os.makedirs("outputs", exist_ok=True)
        json_path = "outputs/test_09.json"
        with open(json_path, "w") as f:
            json.dump({"patient": {"patient_id": "pat-999"}}, f)

        report = self.service.create_report("pat-uuid-999", "doctor@aurascan.ai", "Type", "outputs/test_09.pdf", json_path, 1)

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            token = conn.execute("SELECT verification_token FROM reports WHERE report_id = ?;", (report.report_id,)).fetchone()[0]
        finally:
            conn.close()

        res = self.service.verify_report_by_token(token)
        self.assertEqual(res["verification_state"], "VALID")

    def test_10_invalid_token_is_rejected(self):
        """10. Verify verification of non-existent token returns INVALID."""
        res = self.service.verify_report_by_token("fake-token-does-not-exist")
        self.assertEqual(res["verification_state"], "INVALID")

    def test_11_tampered_report_is_detected(self):
        """11. Verify tampered JSON files raise TAMPERED status."""
        os.makedirs("outputs", exist_ok=True)
        json_path = "outputs/test_11.json"
        with open(json_path, "w") as f:
            json.dump({"patient": {"patient_id": "pat-999"}}, f)

        report = self.service.create_report("pat-uuid-999", "doctor@aurascan.ai", "Type", "outputs/test_11.pdf", json_path, 1)

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            token = conn.execute("SELECT verification_token FROM reports WHERE report_id = ?;", (report.report_id,)).fetchone()[0]
        finally:
            conn.close()

        # Tamper the file contents
        with open(json_path, "w") as f:
            json.dump({"patient": {"patient_id": "pat-TAMPERED-ID"}}, f)

        res = self.service.verify_report_by_token(token)
        self.assertEqual(res["verification_state"], "TAMPERED")

    def test_12_superseded_version_is_correctly_identified(self):
        """12. Verify superseded older version returns SUPERSEDED state."""
        os.makedirs("outputs", exist_ok=True)
        json_path = "outputs/test_12.json"
        with open(json_path, "w") as f:
            json.dump({"patient": {"patient_id": "pat-999"}}, f)

        report = self.service.create_report("pat-uuid-999", "doctor@aurascan.ai", "Type", "outputs/test_12.pdf", json_path, 1, ReportStatus.GENERATED)

        # Save token of version 1
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            token_v1 = conn.execute("SELECT verification_token FROM report_versions WHERE report_id = ? AND version_number = 1;", (report.report_id,)).fetchone()[0]
        finally:
            conn.close()

        # Create version 2
        self.service.create_new_version(report.report_id, "doctor@aurascan.ai", "Update notes")

        res = self.service.verify_report_by_token(token_v1)
        self.assertEqual(res["verification_state"], "SUPERSEDED")

    def test_13_archived_version_is_correctly_identified(self):
        """13. Verify archived report returns ARCHIVED state."""
        os.makedirs("outputs", exist_ok=True)
        json_path = "outputs/test_13.json"
        with open(json_path, "w") as f:
            json.dump({"patient": {"patient_id": "pat-999"}}, f)

        report = self.service.create_report("pat-uuid-999", "doctor@aurascan.ai", "Type", "outputs/test_13.pdf", json_path, 1, ReportStatus.GENERATED)
        self.service.transition_status(report.report_id, ReportStatus.REVIEWED, "doctor@aurascan.ai")
        self.service.transition_status(report.report_id, ReportStatus.FINAL, "doctor@aurascan.ai")
        self.service.transition_status(report.report_id, ReportStatus.ARCHIVED, "doctor@aurascan.ai")

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            token = conn.execute("SELECT verification_token FROM reports WHERE report_id = ?;", (report.report_id,)).fetchone()[0]
        finally:
            conn.close()

        res = self.service.verify_report_by_token(token)
        self.assertEqual(res["verification_state"], "ARCHIVED")

    def test_14_version_1_token_cannot_resolve_to_version_2(self):
        """14. Verify version 1 token checks version 1 and doesn't map to version 2 details."""
        os.makedirs("outputs", exist_ok=True)
        json_path_v1 = "outputs/test_14_v1.json"
        with open(json_path_v1, "w") as f:
            json.dump({"patient": {"patient_id": "pat-999"}, "data": "v1"}, f)

        report = self.service.create_report("pat-uuid-999", "doctor@aurascan.ai", "Type", "outputs/test_14.pdf", json_path_v1, 1, ReportStatus.GENERATED)

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            token_v1 = conn.execute("SELECT verification_token FROM report_versions WHERE report_id = ? AND version_number = 1;", (report.report_id,)).fetchone()[0]
        finally:
            conn.close()

        json_path_v2 = "outputs/test_14_v2.json"
        with open(json_path_v2, "w") as f:
            json.dump({"patient": {"patient_id": "pat-999"}, "data": "v2"}, f)

        v2 = self.service.create_new_version(report.report_id, "doctor@aurascan.ai", "notes", json_path=json_path_v2)

        res = self.service.verify_report_by_token(token_v1)
        self.assertEqual(res["verification_state"], "SUPERSEDED")
        self.assertEqual(res["version"], 1)

    def test_15_public_verification_does_not_expose_patient_data(self):
        """15. Verify public verification endpoint returns safe non-PII metadata only."""
        os.makedirs("outputs", exist_ok=True)
        json_path = "outputs/test_15.json"
        with open(json_path, "w") as f:
            json.dump({"patient": {"patient_id": "pat-999", "name": "Secret Patient"}}, f)

        report = self.service.create_report("pat-uuid-999", "doctor@aurascan.ai", "Type", "outputs/test_15.pdf", json_path, 1)

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            token = conn.execute("SELECT verification_token FROM reports WHERE report_id = ?;", (report.report_id,)).fetchone()[0]
        finally:
            conn.close()

        # Call FastAPI Endpoint without auth headers
        response = self.client.get(f"/api/reports/verify/{token}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertNotIn("Secret Patient", str(data))
        self.assertNotIn("name", data)
        self.assertIn("verification_state", data)
        self.assertIn("report_number", data)

    def test_16_rbac_remains_correct(self):
        """16. Verify authenticated report routes still isolate users correctly."""
        os.makedirs("outputs", exist_ok=True)
        json_path = "outputs/test_16.json"
        with open(json_path, "w") as f:
            json.dump({"patient": {"patient_id": "pat-uuid-999"}}, f)

        report = self.service.create_report("pat-uuid-999", "doctor@aurascan.ai", "Type", "outputs/test_16.pdf", json_path, 1)

        # Authorized doctor can view
        res = self.client.get(f"/api/reports/{report.report_id}", headers=self.doctor_headers)
        self.assertEqual(res.status_code, 200)

        # Authorized patient can view
        res = self.client.get(f"/api/reports/{report.report_id}", headers=self.patient_headers)
        self.assertEqual(res.status_code, 200)

    def test_17_pdf_generation_still_works(self):
        """17. Verify PDF document compiles correctly with verification cards added."""
        os.makedirs("outputs", exist_ok=True)
        json_path = "outputs/test_17.json"
        with open(json_path, "w") as f:
            json.dump({"patient": {"patient_id": "pat-999"}}, f)

        # Create report to trigger builder and PDF generation
        report = self.service.create_report("pat-uuid-999", "doctor@aurascan.ai", "Type", "outputs/test_17.pdf", json_path, 1)
        self.assertTrue(os.path.exists("outputs/test_17.pdf"))

    def test_18_qr_generation_failure_is_handled_safely(self):
        """18. Verify QR code generation failures default cleanly without crashing PDF rendering."""
        from clinical_reporting.domain.entities import Metadata, PatientInformation, ClassificationResult, SegmentationResult, TumorStatistics, ExplainabilityResult, ClinicalInsights, ProcessingMetrics, SystemStatus, Disclaimer, ReportData

        patient = PatientInformation("pat-1", "Bob", 45, "Male", "Dr.")
        cls_r = ClassificationResult("Glioma", 0.9, "EfficientNet-B0")
        seg_r = SegmentationResult("Available", 10.0, 1.0)
        t_stats = TumorStatistics()
        xai_r = ExplainabilityResult(None, "Unavailable")
        insights = ClinicalInsights("narrative")
        metrics = ProcessingMetrics()
        sys_status = SystemStatus("Available", "Available", "Unavailable", "Successful")
        disclaimer = Disclaimer("Disclaimer text")

        metadata = Metadata(
            report_id="RPT-12345",
            report_title="TITLE",
            report_type="TYPE",
            brand_name="BRAND",
            timestamp="2026-08-09",
            verification_token="invalid/token\\with/slashes*", # invalid characters for path
            integrity_hash="some-hash"
        )
        data = ReportData(patient, cls_r, seg_r, t_stats, xai_r, insights, metrics, sys_status, disclaimer, metadata)

        from clinical_reporting.infrastructure.pdf_generator import ReportLabPDFGenerator
        pdf_gen = ReportLabPDFGenerator()

        try:
            pdf_gen.generate_pdf(data, "outputs/test_18.pdf")
            self.assertTrue(os.path.exists("outputs/test_18.pdf"))
        except Exception as e:
            self.fail(f"PDF generation crashed: {e}")

    def test_19_database_migration_is_idempotent(self):
        """19. Verify database migration can run multiple times safely."""
        db_repo = SQLitePersistenceRepository(db_path=self.db_path)
        try:
            db_repo.initialize_db()
            db_repo.initialize_db()
        except Exception as e:
            self.fail(f"Database migration was not idempotent: {e}")


if __name__ == "__main__":
    unittest.main()
