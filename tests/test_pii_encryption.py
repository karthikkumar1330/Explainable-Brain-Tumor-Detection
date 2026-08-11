import os
import unittest
import sqlite3
import datetime
import json
import secrets
from unittest.mock import patch, MagicMock
from cryptography.fernet import Fernet

# Set DB_PATH for tests
os.environ["DB_PATH"] = os.path.abspath("outputs/test_pii_encryption.db")

from security.infrastructure.encryption_service import PIIEncryptionService, EncryptionKeyMissingError
from persistence.infrastructure.repository import SQLitePersistenceRepository
from prediction_history.infrastructure.repository import SQLitePredictionHistoryRepository
from prediction_history.domain.entities import HistorySearchCriteria
from clinical_reporting.application.services import ReportService
from security.domain.entities import Role, User
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.jwt_service import JWTService
from clinical_reporting.domain.entities import PatientInfo, ClinicalReport, ProcessingSummary, PredictionResult
from dashboard.infrastructure.web_server import create_app

class TestPIIEncryption(unittest.TestCase):
    def setUp(self):
        # Generate and configure key
        self.encryption_key = Fernet.generate_key().decode("utf-8")
        os.environ["PII_ENCRYPTION_KEY"] = self.encryption_key
        
        self.db_path = os.environ["DB_PATH"]
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

        # Update path configurations
        import api.infrastructure.routes as api_routes
        import api.routes.auth_routes as auth_routes
        api_routes.DEFAULT_DB_PATH = self.db_path
        auth_routes.DEFAULT_DB_PATH = self.db_path

        self.persistence_repo = SQLitePersistenceRepository(db_path=self.db_path)
        self.persistence_repo.initialize_db()
        
        self.user_repo = SQLiteUserRepository(db_path=self.db_path)
        self.user_repo.initialize_security_tables()
        self.admin_user = self.user_repo.bootstrap_admin()

        self.jwt_service = JWTService()
        self.app = create_app(db_path=self.db_path)
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    def tearDown(self):
        # Clean up database tables to avoid unique constraint violations on Windows
        if os.path.exists(self.db_path):
            try:
                conn = sqlite3.connect(self.db_path)
                with conn:
                    conn.execute("DELETE FROM prediction_quality_flags;")
                    conn.execute("DELETE FROM report_versions;")
                    conn.execute("DELETE FROM reports;")
                    conn.execute("DELETE FROM clinical_reports;")
                    conn.execute("DELETE FROM predictions;")
                    conn.execute("DELETE FROM mri_scans;")
                    conn.execute("DELETE FROM patients;")
                    conn.execute("DELETE FROM users;")
                conn.close()
            except Exception:
                pass
        os.environ.pop("PII_ENCRYPTION_KEY", None)

    def _insert_plaintext_patient_hierarchy(self, patient_id, name, age, gender, scan_id, pred_id, report_id):
        conn = sqlite3.connect(self.db_path)
        conn.execute("INSERT OR IGNORE INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
                     (patient_id, name, age, gender, '2026-08-11'))
        conn.execute("INSERT OR IGNORE INTO mri_scans (id, patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at) VALUES (?, ?, 'dummy.png', 1.0, 'Dr. Smith', '2026-08-11', '2026-08-11');",
                     (scan_id, patient_id))
        conn.execute("INSERT OR IGNORE INTO predictions (id, scan_id, predicted_class, confidence_score, prob_glioma, prob_meningioma, prob_pituitary, prob_no_tumor, tumor_pixel_count, tumor_area_mm2, tumor_percentage_brain, tumor_percentage_image, estimated_brain_pixel_count, rule_based_severity, severity_rule_description, created_at) VALUES (?, ?, 'Glioma', 0.9, 0.9, 0.0, 0.0, 0.1, 100, 10.0, 1.0, 1.0, 1000, 'Low', 'Low rule', '2026-08-11');",
                     (pred_id, scan_id))
        conn.execute("INSERT OR IGNORE INTO clinical_reports (id, prediction_id, markdown_path, json_path, pdf_path, xai_method, created_at) VALUES (?, ?, 'dummy.md', 'dummy.json', 'dummy.pdf', 'LRP', '2026-08-11');",
                     (report_id, pred_id))
        conn.commit()
        conn.close()

    # 1. test_encrypt_decrypt_string
    def test_encrypt_decrypt_string(self):
        service = PIIEncryptionService(self.encryption_key)
        plaintext = "Alice Smith"
        ciphertext = service.encrypt(plaintext)
        self.assertTrue(service.is_encrypted(ciphertext))
        self.assertNotEqual(plaintext, ciphertext)
        self.assertEqual(plaintext, service.decrypt(ciphertext))

    # 2. test_encrypt_decrypt_age
    def test_encrypt_decrypt_age(self):
        service = PIIEncryptionService(self.encryption_key)
        plaintext = 45
        ciphertext = service.encrypt(plaintext)
        self.assertTrue(service.is_encrypted(ciphertext))
        self.assertEqual("45", service.decrypt(ciphertext))

    # 3. test_encrypt_decrypt_gender
    def test_encrypt_decrypt_gender(self):
        service = PIIEncryptionService(self.encryption_key)
        plaintext = "M"
        ciphertext = service.encrypt(plaintext)
        self.assertTrue(service.is_encrypted(ciphertext))
        self.assertEqual(plaintext, service.decrypt(ciphertext))

    # 4. test_none_value
    def test_none_value(self):
        service = PIIEncryptionService(self.encryption_key)
        self.assertIsNone(service.encrypt(None))
        self.assertIsNone(service.decrypt(None))

    # 5. test_empty_value
    def test_empty_value(self):
        service = PIIEncryptionService(self.encryption_key)
        self.assertEqual("enc:v1:", service.encrypt("")[:7])
        self.assertEqual("", service.decrypt(service.encrypt("")))

    # 6. test_invalid_ciphertext
    def test_invalid_ciphertext(self):
        service = PIIEncryptionService(self.encryption_key)
        with self.assertRaises(ValueError):
            service.decrypt("enc:v1:invalidciphertext")

    # 7. test_missing_key
    def test_missing_key(self):
        with patch.dict(os.environ, {"DISABLE_TEST_PII_FALLBACK_KEY": "True"}, clear=True):
            with self.assertRaises(EncryptionKeyMissingError):
                PIIEncryptionService(None)

    # 8. test_key_not_logged
    def test_key_not_logged(self):
        service = PIIEncryptionService(self.encryption_key)
        self.assertNotIn(self.encryption_key, str(service))

    # 9. test_new_patient_stores_ciphertext
    def test_new_patient_stores_ciphertext(self):
        patient_info = PatientInfo(
            patient_id="pat-1",
            name="Alice",
            age=30,
            gender="F",
            scan_date="2026-08-11",
            ref_physician="Dr. Smith"
        )
        report = ClinicalReport(
            patient_info=patient_info,
            processing_summary=ProcessingSummary("cpu", 1.0, "cls", "seg", 0.5, 0.5, 0.0),
            classification=PredictionResult(
                label=1,
                class_name="Glioma",
                confidence_score=0.9,
                probabilities={"Glioma": 0.9, "Meningioma": 0.0, "Pituitary": 0.0, "No Tumor": 0.1}
            ),
            segmentation_metrics=None,
            severity_assessment=None,
            original_image_path="dummy.png",
            heatmap_image_path="dummy_heatmap.png",
            overlay_image_path="dummy_overlay.png",
            segmentation_mask_path="dummy_mask.png"
        )
        self.persistence_repo.save_report(report, "outputs")
        
        # Query directly in SQLite
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("SELECT name, age, gender FROM patients WHERE patient_id = 'pat-1';").fetchone()
        conn.close()
        
        self.assertTrue(row[0].startswith("enc:v1:"))
        self.assertTrue(row[1].startswith("enc:v1:"))
        self.assertTrue(row[2].startswith("enc:v1:"))

    # 10. test_patient_read_returns_plaintext_value
    def test_patient_read_returns_plaintext_value(self):
        patient_info = PatientInfo(
            patient_id="pat-2",
            name="Bob",
            age=40,
            gender="M",
            scan_date="2026-08-11",
            ref_physician="Dr. Smith"
        )
        report = ClinicalReport(
            patient_info=patient_info,
            processing_summary=ProcessingSummary("cpu", 1.0, "cls", "seg", 0.5, 0.5, 0.0),
            classification=PredictionResult(
                label=1,
                class_name="Glioma",
                confidence_score=0.9,
                probabilities={"Glioma": 0.9, "Meningioma": 0.0, "Pituitary": 0.0, "No Tumor": 0.1}
            ),
            segmentation_metrics=None,
            severity_assessment=None,
            original_image_path="dummy.png",
            heatmap_image_path="dummy_heatmap.png",
            overlay_image_path="dummy_overlay.png",
            segmentation_mask_path="dummy_mask.png"
        )
        self.persistence_repo.save_report(report, "outputs")
        
        # Read patient history
        history = self.persistence_repo.get_patient_history("pat-2")
        self.assertGreater(len(history), 0)
        self.assertEqual(history[0]["name"], "Bob")
        self.assertEqual(history[0]["age"], 40)
        self.assertEqual(history[0]["gender"], "M")

    # 11. test_patient_update_encrypts_value
    def test_patient_update_encrypts_value(self):
        # Save first
        patient_info = PatientInfo("pat-3", "Charlie", 50, "M", "2026-08-11", "Dr. Smith")
        report = ClinicalReport(
            patient_info, 
            ProcessingSummary("cpu", 1.0, "cls", "seg", 0.5, 0.5, 0.0), 
            PredictionResult(
                label=1,
                class_name="Glioma",
                confidence_score=0.9,
                probabilities={"Glioma": 0.9, "Meningioma": 0.0, "Pituitary": 0.0, "No Tumor": 0.1}
            ), 
            None, 
            None, 
            "dummy.png",
            "dummy_heatmap.png",
            "dummy_overlay.png",
            "dummy_mask.png"
        )
        self.persistence_repo.save_report(report, "outputs")
        
        # Update name/age
        updated_patient_info = PatientInfo("pat-3", "Charlie Updated", 51, "M", "2026-08-11", "Dr. Smith")
        updated_report = ClinicalReport(
            updated_patient_info, 
            ProcessingSummary("cpu", 1.0, "cls", "seg", 0.5, 0.5, 0.0), 
            PredictionResult(
                label=1,
                class_name="Glioma",
                confidence_score=0.9,
                probabilities={"Glioma": 0.9, "Meningioma": 0.0, "Pituitary": 0.0, "No Tumor": 0.1}
            ), 
            None, 
            None, 
            "dummy.png",
            "dummy_heatmap.png",
            "dummy_overlay.png",
            "dummy_mask.png"
        )
        self.persistence_repo.save_report(updated_report, "outputs")
        
        # Direct check
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("SELECT name, age FROM patients WHERE patient_id = 'pat-3';").fetchone()
        conn.close()
        self.assertTrue(row[0].startswith("enc:v1:"))
        
        # Read back
        history = self.persistence_repo.get_patient_history("pat-3")
        self.assertEqual(history[0]["name"], "Charlie Updated")
        self.assertEqual(history[0]["age"], 51)

    # 12. test_existing_plaintext_record_migration
    def test_existing_plaintext_record_migration(self):
        # Insert plaintext record hierarchy directly
        self._insert_plaintext_patient_hierarchy("pat-mig", "Dave", 60, "M", 401, 501, 601)

        # Retrieve prior to migration (should return plaintext successfully)
        history_before = self.persistence_repo.get_patient_history("pat-mig")
        self.assertEqual(history_before[0]["name"], "Dave")
        self.assertEqual(history_before[0]["age"], 60)

        # Run migration tool
        from scripts.migrate_pii_encryption import main as run_migration
        with patch("sys.argv", ["migrate_pii_encryption.py", "--encrypt", "--db", self.db_path]):
            run_migration()

        # Verify it is now encrypted in DB
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("SELECT name, age, gender FROM patients WHERE patient_id = 'pat-mig';").fetchone()
        conn.close()
        self.assertTrue(row[0].startswith("enc:v1:"))

        # Verify it is decrypted correctly on read
        history_after = self.persistence_repo.get_patient_history("pat-mig")
        self.assertEqual(history_after[0]["name"], "Dave")
        self.assertEqual(history_after[0]["age"], 60)

    # 13. test_migration_is_idempotent
    def test_migration_is_idempotent(self):
        self._insert_plaintext_patient_hierarchy("pat-idemp", "Emma", 25, "F", 402, 502, 602)

        from scripts.migrate_pii_encryption import main as run_migration
        # Migrate once
        with patch("sys.argv", ["migrate_pii_encryption.py", "--encrypt", "--db", self.db_path]):
            run_migration()
        
        # Get ciphertext after first migration
        conn = sqlite3.connect(self.db_path)
        name1 = conn.execute("SELECT name FROM patients WHERE patient_id = 'pat-idemp';").fetchone()[0]
        conn.close()

        # Migrate again
        with patch("sys.argv", ["migrate_pii_encryption.py", "--encrypt", "--db", self.db_path]):
            run_migration()

        conn = sqlite3.connect(self.db_path)
        name2 = conn.execute("SELECT name FROM patients WHERE patient_id = 'pat-idemp';").fetchone()[0]
        conn.close()

        # Verify it remains exactly the same ciphertext
        self.assertEqual(name1, name2)

    # 14. test_database_contains_no_plaintext_target_values_after_migration
    def test_database_contains_no_plaintext_target_values_after_migration(self):
        self._insert_plaintext_patient_hierarchy("pat-sec", "Frank PII", 75, "M", 403, 503, 603)

        # Migrate
        from scripts.migrate_pii_encryption import main as run_migration
        with patch("sys.argv", ["migrate_pii_encryption.py", "--encrypt", "--db", self.db_path]):
            run_migration()

        # Direct database verification
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name, age, gender FROM patients WHERE patient_id = 'pat-sec';")
        rows = cursor.fetchall()
        conn.close()

        for name, age, gender in rows:
            self.assertNotEqual(name, "Frank PII")
            self.assertNotEqual(age, 75)
            self.assertNotEqual(gender, "M")
            self.assertTrue(str(name).startswith("enc:v1:"))
            self.assertTrue(str(age).startswith("enc:v1:"))
            self.assertTrue(str(gender).startswith("enc:v1:"))

    # 15. test_patient_api_contract_unchanged
    def test_patient_api_contract_unchanged(self):
        # Register a doctor user to fetch API
        doctor = User(None, "doc-uuid-contract", "doctor_contract@hospital.org", "password", "Dr. Gregory", Role.DOCTOR)
        self.user_repo.create_user(doctor)
        
        # Save a report for pat-api
        patient_info = PatientInfo("pat-api", "API Patient", 35, "M", "2026-08-11", "Dr. Gregory")
        report = ClinicalReport(
            patient_info, 
            ProcessingSummary("cpu", 1.0, "cls", "seg", 0.5, 0.5, 0.0), 
            PredictionResult(
                label=1,
                class_name="Glioma",
                confidence_score=0.9,
                probabilities={"Glioma": 0.9, "Meningioma": 0.0, "Pituitary": 0.0, "No Tumor": 0.1}
            ), 
            None, 
            None, 
            "dummy.png",
            "dummy_heatmap.png",
            "dummy_overlay.png",
            "dummy_mask.png"
        )
        self.persistence_repo.save_report(report, "outputs")
        
        # Create token
        token = self.jwt_service.create_access_token(
            user_uuid=doctor.uuid,
            user_id=doctor.id,
            email=doctor.email,
            role=doctor.role
        )
        headers = {"Authorization": f"Bearer {token}"}
        
        # Search via API search endpoint
        response = self.client.get("/api/search?patient_name=api", headers=headers)
        self.assertEqual(response.status_code, 200)
        
        res_data = json.loads(response.data.decode("utf-8"))
        self.assertEqual(res_data[0]["patient_name"], "API Patient")

    # 16. test_report_generation_still_works
    def test_report_generation_still_works(self):
        # Seed the patient first to satisfy FK
        conn = sqlite3.connect(self.db_path)
        conn.execute("INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES ('pat-rep', 'Plaintext Patient', 35, 'M', '2026-08-11');")
        conn.commit()
        conn.close()

        service = ReportService(db_path=self.db_path)
        report = service.create_report("pat-rep", "Dr. Gregory", "MRI Brain Scan", "dummy.pdf", "dummy.json", None)
        self.assertIsNotNone(report.report_number)
        
        # Read back
        retrieved = service.get_report(report.report_id)
        self.assertEqual(retrieved.report_number, report.report_number)

    # 17. test_patient_search_behavior
    def test_patient_search_behavior(self):
        history_repo = SQLitePredictionHistoryRepository(db_path=self.db_path)
        
        # Insert patients
        patient_info1 = PatientInfo("pat-s1", "John Doe", 32, "M", "2026-08-11", "Dr. Smith")
        report1 = ClinicalReport(
            patient_info1, 
            ProcessingSummary("cpu", 1.0, "cls", "seg", 0.5, 0.5, 0.0), 
            PredictionResult(
                label=1,
                class_name="Glioma",
                confidence_score=0.9,
                probabilities={"Glioma": 0.9, "Meningioma": 0.0, "Pituitary": 0.0, "No Tumor": 0.1}
            ), 
            None, 
            None, 
            "dummy.png",
            "dummy_heatmap.png",
            "dummy_overlay.png",
            "dummy_mask.png"
        )
        self.persistence_repo.save_report(report1, "outputs")
        
        patient_info2 = PatientInfo("pat-s2", "Jane Smith", 28, "F", "2026-08-11", "Dr. Smith")
        report2 = ClinicalReport(
            patient_info2, 
            ProcessingSummary("cpu", 1.0, "cls", "seg", 0.5, 0.5, 0.0), 
            PredictionResult(
                label=2,
                class_name="Meningioma",
                confidence_score=0.9,
                probabilities={"Glioma": 0.0, "Meningioma": 0.9, "Pituitary": 0.0, "No Tumor": 0.1}
            ), 
            None, 
            None, 
            "dummy2.png",
            "dummy_heatmap.png",
            "dummy_overlay.png",
            "dummy_mask.png"
        )
        self.persistence_repo.save_report(report2, "outputs")
        
        # Search by exact name
        criteria = HistorySearchCriteria(patient_name="jane")
        paginated_res = history_repo.search_history(criteria)
        self.assertEqual(paginated_res.total_count, 1)
        self.assertEqual(paginated_res[0].patient_name, "Jane Smith")

        # Search by global query q
        criteria_q = HistorySearchCriteria(q="john")
        paginated_res_q = history_repo.search_history(criteria_q)
        self.assertEqual(paginated_res_q.total_count, 1)
        self.assertEqual(paginated_res_q[0].patient_name, "John Doe")

    # 18. test_role_authorization_unchanged
    def test_role_authorization_unchanged(self):
        # Create doctor in DB
        doctor = User(None, "doc-auth-uuid", "doctor@hospital.org", "password", "Dr. Gregory", Role.DOCTOR)
        self.user_repo.create_user(doctor)
        # Create patient in DB
        patient = User(None, "pat-auth-uuid", "patient@hospital.org", "password", "Patient A", Role.PATIENT)
        self.user_repo.create_user(patient)

        # Token for Doctor
        token_doc = self.jwt_service.create_access_token(
            user_uuid=doctor.uuid,
            user_id=doctor.id,
            email=doctor.email,
            role=doctor.role
        )
        headers_doc = {"Authorization": f"Bearer {token_doc}"}
        response = self.client.get("/api/history", headers=headers_doc)
        self.assertEqual(response.status_code, 200)

        # Token for Patient
        token_pat = self.jwt_service.create_access_token(
            user_uuid=patient.uuid,
            user_id=patient.id,
            email=patient.email,
            role=patient.role
        )
        headers_pat = {"Authorization": f"Bearer {token_pat}"}
        # /api/admin/users is admin-only, so a Patient gets 403 Forbidden
        response_admin = self.client.get("/api/admin/users", headers=headers_pat)
        self.assertEqual(response_admin.status_code, 403)

    # 19. test_cross_patient_access_denied
    def test_cross_patient_access_denied(self):
        patient_info = PatientInfo("pat-a", "Patient A", 30, "F", "2026-08-11", "Dr. Gregory")
        report = ClinicalReport(
            patient_info, 
            ProcessingSummary("cpu", 1.0, "cls", "seg", 0.5, 0.5, 0.0), 
            PredictionResult(
                label=1,
                class_name="Glioma",
                confidence_score=0.9,
                probabilities={"Glioma": 0.9, "Meningioma": 0.0, "Pituitary": 0.0, "No Tumor": 0.1}
            ), 
            None, 
            None, 
            "dummy.png",
            "dummy_heatmap.png",
            "dummy_overlay.png",
            "dummy_mask.png"
        )
        report_id = self.persistence_repo.save_report(report, "outputs")
        
        service = ReportService(db_path=self.db_path)
        patient_b = User(None, "pat-b-uuid", "patientB@hospital.org", "password", "Patient B", Role.PATIENT)
        
        access = service.check_report_access(report_id, patient_b)
        self.assertEqual(access, "FORBIDDEN")

    # 20. test_encryption_key_not_in_repository
    def test_encryption_key_not_in_repository(self):
        import subprocess
        result = subprocess.run(["git", "grep", self.encryption_key], capture_output=True, text=True, cwd="d:\\BrainTumorProject\\UNeXt-pytorch")
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn(self.encryption_key, result.stdout)
