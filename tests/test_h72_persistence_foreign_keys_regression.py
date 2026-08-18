import unittest
import sqlite3
import os
import tempfile
from persistence.infrastructure.repository import SQLitePersistenceRepository
from security.infrastructure.repository import SQLiteUserRepository
from security.application.authorization_service import AuthorizationService
from clinical_reporting.application.followup_service import FollowupScheduleService
from clinical_reporting.application.mri_annotation_service import MriAnnotationService
from clinical_reporting.application.segmentation_review_service import SegmentationReviewService
from security.domain.entities import Role, User

class TestSQLiteForeignKeysRegression(unittest.TestCase):
    def setUp(self):
        # Create a unique temp file for the SQLite database to avoid locking conflicts
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(self.db_fd)

        # Initialize databases
        self.repo = SQLitePersistenceRepository(db_path=self.db_path)
        self.user_repo = SQLiteUserRepository(db_path=self.db_path)

        # Ensure triggers are disabled during setup to prevent auto-assignment pollution
        os.environ["DISABLE_TEST_AUTO_ASSIGN"] = "1"

        self.repo.initialize_db()
        self.user_repo.initialize_security_tables()

    def tearDown(self):
        # Make sure connections are garbage collected before removing
        self.repo = None
        self.user_repo = None

        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass
        if "DISABLE_TEST_AUTO_ASSIGN" in os.environ:
            del os.environ["DISABLE_TEST_AUTO_ASSIGN"]

    def test_all_connections_enable_foreign_keys(self):
        """1. Verify that every repository and service SQLite connection enables foreign keys."""
        # Persistence Repo
        conn = self.repo._get_connection()
        fk = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
        self.assertEqual(fk, 1, "PersistenceRepository connection must enable foreign_keys")
        conn.close()

        # User Repo
        conn = self.user_repo._get_connection()
        fk = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
        self.assertEqual(fk, 1, "UserRepository connection must enable foreign_keys")
        conn.close()

        # Authorization Service
        auth_svc = AuthorizationService(db_path=self.db_path)
        conn = auth_svc._get_connection()
        fk = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
        self.assertEqual(fk, 1, "AuthorizationService connection must enable foreign_keys")
        conn.close()

        # Followup Service
        followup_svc = FollowupScheduleService(db_path=self.db_path)
        conn = followup_svc._get_connection()
        fk = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
        self.assertEqual(fk, 1, "FollowupScheduleService connection must enable foreign_keys")
        conn.close()

        # Mri Annotation Service
        anno_svc = MriAnnotationService(db_path=self.db_path)
        conn = anno_svc._get_connection()
        fk = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
        self.assertEqual(fk, 1, "MriAnnotationService connection must enable foreign_keys")
        conn.close()

        # Segmentation Review Service
        rev_svc = SegmentationReviewService(db_path=self.db_path)
        conn = rev_svc._get_connection()
        fk = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
        self.assertEqual(fk, 1, "SegmentationReviewService connection must enable foreign_keys")
        conn.close()

    def test_parent_deletion_cascades(self):
        """2. Verify that deleting a parent record cascade-deletes related children when foreign keys are ON."""
        conn = self.repo._get_connection()

        # 1. Insert patient
        conn.execute("INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES ('P1', 'Test Patient', 30, 'Male', '2026-08-18 12:00:00');")
        # 2. Insert scan
        conn.execute("INSERT INTO mri_scans (id, patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at) VALUES (10, 'P1', 'path', 1.0, 'Doctor', '2026-08-18', '2026-08-18 12:00:00');")
        # 3. Insert prediction
        conn.execute("INSERT INTO predictions (id, scan_id, predicted_class, confidence_score, prob_glioma, prob_meningioma, prob_pituitary, prob_no_tumor, tumor_pixel_count, tumor_area_mm2, tumor_percentage_brain, tumor_percentage_image, estimated_brain_pixel_count, rule_based_severity, severity_rule_description, created_at) VALUES (20, 10, 'Glioma', 0.9, 0.9, 0.0, 0.0, 0.1, 100, 10.0, 1.0, 1.0, 1000, 'Low', 'None', '2026-08-18 12:00:00');")
        # 4. Insert report
        conn.execute("INSERT INTO reports (report_id, report_number, patient_id, created_by, report_type, current_version, status, created_at, updated_at, pdf_path, json_path, checksum) VALUES (30, 'RPT-001', 'P1', 'Doctor', 'MRI Brain Scan', 1, 'GENERATED', '2026-08-18 12:00:00', '2026-08-18 12:00:00', 'pdf', 'json', 'check');")
        # 5. Insert report version
        conn.execute("INSERT INTO report_versions (version_id, report_id, version_number, created_at, created_by, reason, pdf_path, json_path, checksum, status, prediction_id) VALUES (40, 30, 1, '2026-08-18 12:00:00', 'Doctor', 'Migration', 'pdf', 'json', 'check', 'GENERATED', 20);")

        # Verify they exist
        self.assertIsNotNone(conn.execute("SELECT 1 FROM mri_scans WHERE id = 10;").fetchone())
        self.assertIsNotNone(conn.execute("SELECT 1 FROM predictions WHERE id = 20;").fetchone())
        self.assertIsNotNone(conn.execute("SELECT 1 FROM reports WHERE report_id = 30;").fetchone())
        self.assertIsNotNone(conn.execute("SELECT 1 FROM report_versions WHERE version_id = 40;").fetchone())

        # Delete Patient -> should cascade to all tables
        conn.execute("DELETE FROM patients WHERE patient_id = 'P1';")
        conn.commit()

        # Check they are deleted
        self.assertIsNone(conn.execute("SELECT 1 FROM mri_scans WHERE id = 10;").fetchone())
        self.assertIsNone(conn.execute("SELECT 1 FROM predictions WHERE id = 20;").fetchone())
        self.assertIsNone(conn.execute("SELECT 1 FROM reports WHERE report_id = 30;").fetchone())
        self.assertIsNone(conn.execute("SELECT 1 FROM report_versions WHERE version_id = 40;").fetchone())

        conn.close()

    def test_legacy_migration_handles_missing_lineage_safely_and_is_idempotent(self):
        """3 & 4. Verify that legacy migration skips records with missing parents and is idempotent."""
        conn = self.repo._get_connection()

        # Disable foreign keys temporarily to mock E2E cleanup leak (deleting scan while leaving prediction)
        conn.execute("PRAGMA foreign_keys = OFF;")

        # 1. Insert valid patient
        conn.execute("INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES ('P_MIG', 'Migration Patient', 45, 'Female', '2026');")
        # 2. Insert scan
        conn.execute("INSERT INTO mri_scans (id, patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at) VALUES (100, 'P_MIG', 'path', 1.0, 'Doctor', '2026', '2026');")
        # 3. Insert prediction
        conn.execute("INSERT INTO predictions (id, scan_id, predicted_class, confidence_score, prob_glioma, prob_meningioma, prob_pituitary, prob_no_tumor, tumor_pixel_count, tumor_area_mm2, tumor_percentage_brain, tumor_percentage_image, estimated_brain_pixel_count, rule_based_severity, severity_rule_description, created_at) VALUES (200, 100, 'Glioma', 0.9, 0.9, 0.0, 0.0, 0.1, 100, 10.0, 1.0, 1.0, 1000, 'Low', 'None', '2026');")
        # 4. Insert clinical report
        conn.execute("INSERT INTO clinical_reports (id, prediction_id, markdown_path, json_path, pdf_path, created_at) VALUES (99, 200, 'md', 'json', 'pdf', '2026-08-18 12:00:00');")

        # 5. Delete scan while foreign keys are OFF (simulates cleanup leak leaving prediction dangling)
        conn.execute("DELETE FROM mri_scans WHERE id = 100;")
        conn.commit()

        # Re-enable foreign keys
        conn.execute("PRAGMA foreign_keys = ON;")

        # Run migration once (should log warning but not raise exception)
        try:
            self.repo._migrate_legacy_reports(conn)
        except sqlite3.IntegrityError as e:
            self.fail(f"Migration raised IntegrityError on missing lineage: {e}")

        # Assert no report was created for this invalid ID
        row = conn.execute("SELECT 1 FROM reports WHERE report_id = 99;").fetchone()
        self.assertIsNone(row, "Report should not have been created for invalid legacy record")

        # Run migration a second time to verify idempotency
        try:
            self.repo._migrate_legacy_reports(conn)
        except Exception as e:
            self.fail(f"Second migration run raised exception: {e}")

        conn.close()

    def test_existing_authorization_unchanged(self):
        """5. Verify existing doctor-patient authorization rules are unchanged."""
        auth_svc = AuthorizationService(db_path=self.db_path)

        # Create users
        doc_u = User(id=3, uuid="doc-uuid", email="doc@test.com", password_hash="hash", full_name="Dr. Test", role=Role.DOCTOR, is_verified=True, is_active=True)

        # Insert Doctor into users table so the foreign key constraint passes
        conn_user = self.user_repo._get_connection()
        conn_user.execute("INSERT INTO users (id, uuid, email, password_hash, full_name, role, is_verified, is_active, created_at, updated_at) VALUES (3, 'doc-uuid', 'doc@test.com', 'hash', 'Dr. Test', 'doctor', 1, 1, '2026', '2026');")
        conn_user.commit()
        conn_user.close()

        conn = self.repo._get_connection()
        # Doctor A has no assignment yet -> must be denied access to 'pat-uuid' if it exists as a patient
        conn.execute("INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES ('pat-uuid', 'Pat Test', 30, 'Male', '2026');")
        conn.commit()

        # Check authorization boundary
        allowed = auth_svc.can_access_patient(doc_u, "pat-uuid", conn=conn)
        self.assertFalse(allowed, "Doctor should be denied access to unassigned patient record")

        # Assign Doctor to Patient
        conn.execute("INSERT INTO doctor_patient_assignments (doctor_id, patient_id, created_at) VALUES (3, 'pat-uuid', '2026');")
        conn.commit()

        allowed_assigned = auth_svc.can_access_patient(doc_u, "pat-uuid", conn=conn)
        self.assertTrue(allowed_assigned, "Doctor should be allowed access to assigned patient record")

        conn.close()

if __name__ == "__main__":
    unittest.main()
