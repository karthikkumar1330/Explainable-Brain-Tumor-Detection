import os
import unittest
import sqlite3
import datetime
import hashlib
from fastapi.testclient import TestClient

os.environ["DB_PATH"] = os.path.abspath("outputs/test_report_lifecycle.db")

from run_api import app
from clinical_reporting.application.services import (
    ReportService,
    ReportServiceException,
    InvalidTransitionException,
    ReportNotFoundException,
    FinalizedReportException
)
from clinical_reporting.domain.entities import ReportStatus, Report, ReportVersion
from security.domain.entities import Role, User
from security.infrastructure.jwt_service import JWTService
from security.infrastructure.repository import SQLiteUserRepository
from persistence.infrastructure.repository import SQLitePersistenceRepository


class TestReportLifecycleAndVersionManagement(unittest.TestCase):
    """Automated tests for Phase F2.1 - Enterprise Report Lifecycle and Version Management."""

    def _create_test_hierarchy(self, conn, patient_id="pat-uuid-999", prediction_id=1, scan_id=1):
        # Insert patients
        conn.execute(
            "INSERT OR IGNORE INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
            (patient_id, "Bob Jones", 45, "Male", datetime.datetime.utcnow().isoformat())
        )
        conn.execute(
            "INSERT OR IGNORE INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
            ("pat-1", "Patient One", 30, "Male", datetime.datetime.utcnow().isoformat())
        )
        conn.execute(
            "INSERT OR IGNORE INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
            ("pat-2", "Patient Two", 40, "Female", datetime.datetime.utcnow().isoformat())
        )
        conn.execute(
            "INSERT OR IGNORE INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
            ("pat-999", "Patient 999", 50, "Male", datetime.datetime.utcnow().isoformat())
        )

        # Insert scans
        conn.execute(
            "INSERT OR IGNORE INTO mri_scans (id, patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
            (scan_id, patient_id, "dummy_scan.png", 1.0, "Dr. Smith", "2026-08-08", datetime.datetime.utcnow().isoformat())
        )
        conn.execute(
            "INSERT OR IGNORE INTO mri_scans (id, patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
            (2, "pat-2", "dummy_scan2.png", 1.0, "Dr. Smith", "2026-08-08", datetime.datetime.utcnow().isoformat())
        )

        # Insert predictions
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
            """INSERT OR IGNORE INTO predictions (
                id, scan_id, predicted_class, confidence_score, prob_glioma, prob_meningioma,
                prob_pituitary, prob_no_tumor, tumor_pixel_count, tumor_area_mm2,
                tumor_percentage_brain, tumor_percentage_image, estimated_brain_pixel_count,
                rule_based_severity, severity_rule_description, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
            (2, 2, "Meningioma", 0.90, 0.02, 0.90, 0.05, 0.03, 300, 30.0, 3.0, 1.0, 10000, "Medium", "Medium severity rule", datetime.datetime.utcnow().isoformat())
        )

    def setUp(self):
        self.db_path = os.environ.get("DB_PATH", "outputs/test_report_lifecycle.db")
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

        # Setup users for RBAC testing
        self.user_repo = SQLiteUserRepository(db_path=self.db_path)
        self.user_repo.initialize_security_tables()
        self.admin_user = self.user_repo.bootstrap_admin()

        # Bootstrap a doctor user
        from security.infrastructure.password import PasswordHasher
        pass_hash = PasswordHasher.hash_password("Doctor@123")

        # Insert users directly in DB to bypass JWT verification dependencies on DB state
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
                ("pat-uuid-OTHER", "other@patient.com", pass_hash, "Other Patient", Role.PATIENT.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            self._create_test_hierarchy(conn)
            conn.commit()
        finally:
            conn.close()

        self.doctor_user = self.user_repo.get_by_email("doctor@aurascan.ai")
        self.patient_user = self.user_repo.get_by_email("patient@aurascan.ai")
        self.other_patient_user = self.user_repo.get_by_email("other@patient.com")

        # Generate JWT tokens
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
        self.other_patient_token = self.jwt_svc.create_access_token(
            user_uuid=self.other_patient_user.uuid, user_id=self.other_patient_user.id, email=self.other_patient_user.email, role=Role.PATIENT
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

    def test_report_creation_and_retrieval(self):
        """Verify report creation and basic retrieval operations."""
        report = self.service.create_report(
            patient_id="pat-uuid-999",
            created_by="doctor@aurascan.ai",
            report_type="MRI Brain Scan",
            pdf_path="test_report_1.pdf",
            json_path="test_report_1.json",
            prediction_id=1,
            initial_status=ReportStatus.GENERATED
        )
        self.assertEqual(report.patient_id, "pat-uuid-999")
        self.assertEqual(report.status, ReportStatus.GENERATED)
        self.assertEqual(report.current_version, 1)
        self.assertTrue(report.report_number.startswith("RPT-2026-"))

        # Retrieve report
        retrieved = self.service.get_report(report.report_id)
        self.assertEqual(retrieved.report_number, report.report_number)
        self.assertEqual(retrieved.status, ReportStatus.GENERATED)

    def test_report_number_uniqueness(self):
        """Verify sequential, collision-safe report number generation."""
        r1 = self.service.create_report("pat-1", "doc", "Type", "pdf1", "json1", 1)
        r2 = self.service.create_report("pat-2", "doc", "Type", "pdf2", "json2", 2)
        self.assertNotEqual(r1.report_number, r2.report_number)

        # Verify deterministic sequence numbers
        num1 = int(r1.report_number.split("-")[-1])
        num2 = int(r2.report_number.split("-")[-1])
        self.assertEqual(num2, num1 + 1)

    def test_initial_version_creation(self):
        """Verify that creating a report automatically creates the version 1 record."""
        report = self.service.create_report("pat-999", "doc", "Type", "pdf", "json", 1)
        versions = self.service.get_report_versions(report.report_id)
        self.assertEqual(len(versions), 1)
        self.assertEqual(versions[0].version_number, 1)
        self.assertEqual(versions[0].reason, "Initial report generation")
        self.assertEqual(versions[0].status, ReportStatus.GENERATED)

    def test_version_increment_and_preservation(self):
        """Verify report version incrementing and copying/preserving files on disk."""
        pdf_path = "outputs/test_lifecycle_pdf.pdf"
        os.makedirs("outputs", exist_ok=True)
        with open(pdf_path, "w") as f:
            f.write("PDF v1 content")

        report = self.service.create_report("pat-uuid-999", "doctor@aurascan.ai", "Type", pdf_path, "json_path", 1, ReportStatus.GENERATED)

        # Create version 2
        new_v = self.service.create_new_version(
            report_id=report.report_id,
            created_by="doctor@aurascan.ai",
            reason="Additional findings",
            status=ReportStatus.DRAFT
        )
        self.assertEqual(new_v.version_number, 2)
        self.assertEqual(new_v.reason, "Additional findings")
        self.assertEqual(new_v.status, ReportStatus.DRAFT)
        self.assertIn("_v2.pdf", new_v.pdf_path)

        # Verify physical file preservation
        self.assertTrue(os.path.exists(pdf_path))
        self.assertTrue(os.path.exists(new_v.pdf_path))

        # Clean up files
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        if os.path.exists(new_v.pdf_path):
            os.remove(new_v.pdf_path)

    def test_status_transitions_validation(self):
        """Verify transition rules and invalid state change blocks."""
        report = self.service.create_report("pat-999", "doc", "Type", "pdf", "json", 1, ReportStatus.GENERATED)

        # Valid: GENERATED -> REVIEWED
        updated = self.service.transition_status(report.report_id, ReportStatus.REVIEWED, "doc")
        self.assertEqual(updated.status, ReportStatus.REVIEWED)

        # Invalid: REVIEWED -> GENERATED (cannot move backwards arbitrarily)
        with self.assertRaises(InvalidTransitionException):
            self.service.transition_status(report.report_id, ReportStatus.GENERATED, "doc")

        # Valid: REVIEWED -> FINAL
        updated = self.service.transition_status(report.report_id, ReportStatus.FINAL, "doc")
        self.assertEqual(updated.status, ReportStatus.FINAL)
        self.assertIsNotNone(updated.finalized_at)

    def test_archive_behavior(self):
        """Verify archiving a finalized report prevents edits/new versions."""
        report = self.service.create_report("pat-999", "doc", "Type", "pdf", "json", 1, ReportStatus.GENERATED)
        self.service.transition_status(report.report_id, ReportStatus.REVIEWED, "doc")
        self.service.transition_status(report.report_id, ReportStatus.FINAL, "doc")

        # Transition: FINAL -> ARCHIVED
        archived = self.service.transition_status(report.report_id, ReportStatus.ARCHIVED, "doc")
        self.assertEqual(archived.status, ReportStatus.ARCHIVED)
        self.assertIsNotNone(archived.archived_at)

        # Editing or adding version to ARCHIVED report should fail
        with self.assertRaises(FinalizedReportException):
            self.service.create_new_version(report.report_id, "doc", "Try edit")

    def test_rbac_and_patient_isolation(self):
        """Verify role-based access controls and patient tenant boundary checks."""
        report = self.service.create_report(
            patient_id="pat-uuid-999",
            created_by="doctor@aurascan.ai",
            report_type="MRI Brain Scan",
            pdf_path="test_pat.pdf",
            json_path="test_pat.json",
            prediction_id=1,
            initial_status=ReportStatus.GENERATED
        )

        # Test endpoint GET /api/reports/{report_id}
        # Doctor can access
        res = self.client.get(f"/api/reports/{report.report_id}", headers=self.doctor_headers)
        self.assertEqual(res.status_code, 200)

        # Patient (Bob Jones, whose patient_id is pat-uuid-999) can access his own report
        res = self.client.get(f"/api/reports/{report.report_id}", headers=self.patient_headers)
        self.assertEqual(res.status_code, 200)

        # Another patient (unauthorized) cannot access Bob's report
        res = self.client.get(f"/api/reports/{report.report_id}", headers={"Authorization": f"Bearer {self.other_patient_token}"})
        self.assertEqual(res.status_code, 403)

        # Patient cannot update status
        res = self.client.patch(f"/api/reports/{report.report_id}/status", json={"status": "reviewed"}, headers=self.patient_headers)
        self.assertEqual(res.status_code, 403)

    def test_database_migration_idempotency(self):
        """Verify migration can run multiple times safely without destroying or duplicating data."""
        conn = sqlite3.connect(self.db_path)
        try:
            # Re-run migration
            self.persistence_repo._migrate_legacy_reports(conn)
            # Ensure tables exist and data is fine
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM reports;")
            self.assertEqual(cursor.fetchone()[0], 0)  # Since no legacy clinical_reports existed in this test DB yet
        finally:
            conn.close()

    def test_degraded_state_segmentation_failure(self):
        """Verify report behavior when classification succeeds but segmentation fails completely."""
        # 1. Test severity classifier directly
        from severity_assessment.infrastructure.classifier import RuleBasedSeverityClassifier
        from severity_assessment.domain.entities import SeverityCategory

        classifier = RuleBasedSeverityClassifier()
        assessment = classifier.assess(
            tumor_type="Pituitary",
            tumor_area_mm2=0.0,
            tumor_percentage=0.0,
            segmentation_failed=True
        )
        # Verify category is MEDIUM (no false Low/No tumor baseline category)
        self.assertEqual(assessment.category, SeverityCategory.MEDIUM)
        self.assertIn("Quantitative tumor morphology is unavailable", assessment.rule_description)
        self.assertNotIn("No active tumor mass detected", assessment.rule_description)

        # 2. Test clinical guidelines generation directly
        from tumor_analysis.application.use_cases import AnalyzeTumorUseCase
        from tumor_analysis.infrastructure.analyzer import OpenCVTumorAnalyzer
        import logging

        analyzer = OpenCVTumorAnalyzer()
        use_case = AnalyzeTumorUseCase(analyzer=analyzer, logger=logging.getLogger("test"))

        # Simulated all-zeros mask
        import numpy as np
        mask = np.zeros((224, 224), dtype=np.uint8)

        report_data = use_case.execute(
            mask=mask,
            patient_id="pat-uuid-999",
            tumor_class="Pituitary",
            segmentation_failed=True
        )
        self.assertIn("Quantitative tumor morphology is unavailable", report_data.clinical_notes)
        self.assertIn("neurosurgical/oncological clinical team", report_data.recommendations or "")

        # 3. Test GenerateClinicalInsightUseCase directly
        from clinical_insight.application.use_cases import GenerateClinicalInsightUseCase
        insight_use_case = GenerateClinicalInsightUseCase()
        insight = insight_use_case.execute(
            predicted_class="Pituitary",
            confidence_score=0.95,
            is_calibrated=True,
            probabilities={"Pituitary": 0.95},
            tumor_area_mm2=0.0,
            pixel_count=0,
            solidity=None,
            circularity=None,
            xai_method="gradcam",
            xai_overlap_percentage=0.0,
            segmentation_failed=True
        )
        self.assertIn("Quantitative tumor morphology is unavailable", insight.summary_narrative)
        self.assertIn("Quantitative tumor morphology is unavailable", insight.key_findings[1])
        self.assertIn("Refer patient to neurosurgical/oncological", insight.recommendations[0])


if __name__ == "__main__":
    unittest.main()
