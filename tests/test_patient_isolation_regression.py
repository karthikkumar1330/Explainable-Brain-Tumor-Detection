import os
import unittest
import sqlite3
import datetime
import tempfile
from security.domain.entities import Role, User
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.jwt_service import JWTService
from security.infrastructure.password import PasswordHasher
from security.application.authorization_service import AuthorizationService
from persistence.infrastructure.repository import SQLitePersistenceRepository
from clinical_reporting.application.services import ReportService

class TestPatientIsolationRegression(unittest.TestCase):
    """Regression tests to verify patient isolation invariants and prevent automatic/implicit assignment creation."""

    def setUp(self):
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.environ["DB_PATH"] = self.db_path
        os.environ["DISABLE_TEST_AUTO_ASSIGN"] = "1"

        # Initialize persistence database
        self.persistence_repo = SQLitePersistenceRepository(db_path=self.db_path)
        self.persistence_repo.initialize_db()

        # Initialize security database
        self.user_repo = SQLiteUserRepository(db_path=self.db_path)
        self.user_repo.initialize_security_tables()
        self.admin = self.user_repo.bootstrap_admin()

        # Create doctor A, doctor B, and patient
        pass_hash = PasswordHasher.hash_password("Doctor@123")
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO users (uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                ("doc-a", "doc_a@aurascan.ai", pass_hash, "Dr. Alice", Role.DOCTOR.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO users (uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                ("doc-b", "doc_b@aurascan.ai", pass_hash, "Dr. Bob", Role.DOCTOR.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO users (uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                ("pat-x", "pat_x@aurascan.ai", pass_hash, "Patient X", Role.PATIENT.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            # Create patient demographics
            conn.execute(
                "INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
                ("pat-x", "Patient X", 45, "Male", datetime.datetime.utcnow().isoformat())
            )
            # Create scan and prediction
            conn.execute(
                "INSERT INTO mri_scans (id, patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                (1, "pat-x", "scan.png", 1.0, "Dr. Alice", "2026-08-08", datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                """INSERT INTO predictions (
                    id, scan_id, predicted_class, confidence_score, prob_glioma, prob_meningioma,
                    prob_pituitary, prob_no_tumor, tumor_pixel_count, tumor_area_mm2,
                    tumor_percentage_brain, tumor_percentage_image, estimated_brain_pixel_count,
                    rule_based_severity, severity_rule_description, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);""",
                (1, 1, "Glioma", 0.95, 0.95, 0.02, 0.02, 0.01, 500, 50.0, 5.0, 2.0, 10000, "High", "High severity", datetime.datetime.utcnow().isoformat())
            )
            conn.commit()
        finally:
            conn.close()

        self.doc_a = self.user_repo.get_by_email("doc_a@aurascan.ai")
        self.doc_b = self.user_repo.get_by_email("doc_b@aurascan.ai")

        self.auth_svc = AuthorizationService(db_path=self.db_path)
        self.report_svc = ReportService(db_path=self.db_path)

    def tearDown(self):
        os.close(self.db_fd)
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass
        if "DISABLE_TEST_AUTO_ASSIGN" in os.environ:
            del os.environ["DISABLE_TEST_AUTO_ASSIGN"]

    def test_patient_isolation_regression_invariants(self):
        # 1. Proves no implicit assignment
        conn = sqlite3.connect(self.db_path)
        count = conn.execute("SELECT COUNT(*) FROM doctor_patient_assignments;").fetchone()[0]
        self.assertEqual(count, 0, "Initially, there must be no assignments.")

        # Doctor A and Doctor B are blocked
        self.assertFalse(self.auth_svc.can_access_patient(self.doc_a, "pat-x"), "Doctor A must not have access without explicit assignment.")
        self.assertFalse(self.auth_svc.can_access_patient(self.doc_b, "pat-x"), "Doctor B must not have access without explicit assignment.")
        conn.close()

        # 2. Proves explicit assignment required
        from tests.helpers.authorization_fixtures import assign_doctor_to_patient
        assign_doctor_to_patient(self.db_path, "doc_a@aurascan.ai", "pat-x")

        # Doctor A has access, Doctor B does not (implements "Doctor A cannot access Doctor B's patient")
        self.assertTrue(self.auth_svc.can_access_patient(self.doc_a, "pat-x"), "Doctor A must have access after explicit assignment.")
        self.assertFalse(self.auth_svc.can_access_patient(self.doc_b, "pat-x"), "Doctor B must still be blocked.")

        # 3. Proves assignment persists across restart
        # We instantiate a new authorization service to simulate server restart
        new_auth_svc = AuthorizationService(db_path=self.db_path)
        self.assertTrue(new_auth_svc.can_access_patient(self.doc_a, "pat-x"), "Access must persist across service re-instantiation.")
        self.assertFalse(new_auth_svc.can_access_patient(self.doc_b, "pat-x"), "Doctor B must still be blocked after restart.")

        # 4. Proves report creation does not create assignment automatically
        # Setup report path
        os.makedirs("outputs/clinical_reports", exist_ok=True)
        json_path = os.path.abspath("outputs/clinical_reports/test_regression.json")
        with open(json_path, "w") as f:
            f.write('{"patient": {"patient_id": "pat-x"}}')

        conn = sqlite3.connect(self.db_path)
        before_count = conn.execute("SELECT COUNT(*) FROM doctor_patient_assignments;").fetchone()[0]
        self.assertEqual(before_count, 1, "There should be exactly 1 assignment before report creation.")
        conn.close()

        report = self.report_svc.create_report(
            patient_id="pat-x",
            created_by="doc_a@aurascan.ai",
            report_type="MRI Brain Scan",
            pdf_path="outputs/clinical_reports/test_regression.pdf",
            json_path=json_path,
            prediction_id=1
        )

        conn = sqlite3.connect(self.db_path)
        after_count = conn.execute("SELECT COUNT(*) FROM doctor_patient_assignments;").fetchone()[0]
        self.assertEqual(before_count, after_count, "Creating a report must not create any assignments automatically.")
        conn.close()

        # Cleanup created files
        for p in ["outputs/clinical_reports/test_regression.pdf", json_path]:
            if os.path.exists(p):
                os.remove(p)

        # 5. Proves reading reports does not create assignment
        conn = sqlite3.connect(self.db_path)
        before_read_count = conn.execute("SELECT COUNT(*) FROM doctor_patient_assignments;").fetchone()[0]
        conn.close()

        # Doctor A reads report
        self.report_svc.get_report(report.report_id)

        conn = sqlite3.connect(self.db_path)
        after_read_count = conn.execute("SELECT COUNT(*) FROM doctor_patient_assignments;").fetchone()[0]
        self.assertEqual(before_read_count, after_read_count, "Reading report must not mutate assignments.")
        conn.close()

        # 6. Proves exports do not create assignment
        # JSON export
        try:
            self.report_svc.get_report_json_for_export(report.report_id, actor=self.doc_a)
        except Exception:
            pass

        # CSV export
        try:
            self.report_svc.get_report_csv_for_export(report.report_id, actor=self.doc_a)
        except Exception:
            pass

        conn = sqlite3.connect(self.db_path)
        after_export_count = conn.execute("SELECT COUNT(*) FROM doctor_patient_assignments;").fetchone()[0]
        self.assertEqual(before_read_count, after_export_count, "Exporting reports must not mutate assignments.")
        conn.close()

        # 7. Proves analytics do not create assignment
        try:
            self.report_svc.get_population_analytics(actor=self.doc_a)
        except Exception:
            pass

        conn = sqlite3.connect(self.db_path)
        after_analytics_count = conn.execute("SELECT COUNT(*) FROM doctor_patient_assignments;").fetchone()[0]
        self.assertEqual(before_read_count, after_analytics_count, "Viewing analytics must not mutate assignments.")
        conn.close()

        # 8. Proves email delivery does not create assignment
        # Deliver email
        try:
            self.report_svc.send_report_email(report.report_id, "doc_a@aurascan.ai", actor=self.doc_a)
        except Exception:
            pass

        conn = sqlite3.connect(self.db_path)
        after_email_count = conn.execute("SELECT COUNT(*) FROM doctor_patient_assignments;").fetchone()[0]
        self.assertEqual(before_read_count, after_email_count, "Sending email must not mutate assignments.")
        conn.close()
