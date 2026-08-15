import os
import unittest
import sqlite3
import datetime
from unittest.mock import patch

from security.domain.entities import Role, User
from security.infrastructure.jwt_service import JWTService
from security.infrastructure.repository import SQLiteUserRepository
from security.application.authorization_service import AuthorizationService
from persistence.infrastructure.repository import SQLitePersistenceRepository
from prediction_history.infrastructure.repository import SQLitePredictionHistoryRepository, HistorySearchCriteria
from clinical_reporting.application.followup_service import FollowupScheduleService, FollowupScheduleServiceException
from clinical_reporting.application.services import ReportService
from api.infrastructure.routes import get_prediction_history
from dashboard.infrastructure.web_server import create_app

class TestH72EPerformanceOptimization(unittest.TestCase):
    """Focused test suite for Phase H7.2-E: Database Performance Optimization."""

    def setUp(self):
        self.db_path = os.path.abspath("outputs/test_h72e_telemetry.db")
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        # Override DB paths in routes
        import api.infrastructure.routes as api_routes
        api_routes.DEFAULT_DB_PATH = self.db_path

        self.persistence_repo = SQLitePersistenceRepository(db_path=self.db_path)
        self.persistence_repo.initialize_db()

        self.user_repo = SQLiteUserRepository(db_path=self.db_path)
        self.user_repo.initialize_security_tables()

        # Drop auto assign triggers to have manual control over doctor assignment boundaries
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DROP TRIGGER IF EXISTS auto_assign_doctor_on_insert;")
            conn.execute("DROP TRIGGER IF EXISTS auto_assign_patient_on_insert;")
            conn.execute("DELETE FROM doctor_patient_assignments;")
            conn.commit()
        finally:
            conn.close()

        # Create users
        self.admin = self.user_repo.bootstrap_admin()
        self.doctor = self.user_repo.create_user(User(
            id=None, uuid="doc-opt", email="doc-opt@aurascan.ai",
            password_hash="hash", full_name="Dr. Opt", role=Role.DOCTOR,
            is_verified=True, is_active=True,
            created_at=datetime.datetime.utcnow().isoformat(),
            updated_at=datetime.datetime.utcnow().isoformat()
        ))
        self.other_doctor = self.user_repo.create_user(User(
            id=None, uuid="doc-other", email="doc-other@aurascan.ai",
            password_hash="hash", full_name="Dr. Other", role=Role.DOCTOR,
            is_verified=True, is_active=True,
            created_at=datetime.datetime.utcnow().isoformat(),
            updated_at=datetime.datetime.utcnow().isoformat()
        ))
        self.patient = self.user_repo.create_user(User(
            id=None, uuid="pat-opt", email="pat-opt@aurascan.ai",
            password_hash="hash", full_name="Patient Bob", role=Role.PATIENT,
            is_verified=True, is_active=True,
            created_at=datetime.datetime.utcnow().isoformat(),
            updated_at=datetime.datetime.utcnow().isoformat()
        ))
        self.other_patient = self.user_repo.create_user(User(
            id=None, uuid="pat-other", email="pat-other@aurascan.ai",
            password_hash="hash", full_name="Patient Alice", role=Role.PATIENT,
            is_verified=True, is_active=True,
            created_at=datetime.datetime.utcnow().isoformat(),
            updated_at=datetime.datetime.utcnow().isoformat()
        ))

        # Assign Patient Bob to Dr. Opt (Patient Alice remains unassigned or assigned to other doc)
        conn = sqlite3.connect(self.db_path)
        now_str = datetime.datetime.utcnow().isoformat()
        try:
            conn.execute("INSERT INTO doctor_patient_assignments (doctor_id, patient_id, created_at) VALUES (?, ?, ?);", (self.doctor.id, self.patient.uuid, now_str))
            conn.execute("INSERT INTO doctor_patient_assignments (doctor_id, patient_id, created_at) VALUES (?, ?, ?);", (self.other_doctor.id, self.other_patient.uuid, now_str))
            conn.commit()
        finally:
            conn.close()

        self.auth_svc = AuthorizationService(db_path=self.db_path)
        self.history_repo = SQLitePredictionHistoryRepository(db_path=self.db_path)
        self.followup_service = FollowupScheduleService(db_path=self.db_path)
        self.report_service = ReportService(db_path=self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def _seed_mock_history_records(self, count=5):
        """Helper to seed history database rows."""
        conn = sqlite3.connect(self.db_path)
        now_str = datetime.datetime.utcnow().isoformat()
        try:
            from security.infrastructure.encryption_service import PIIEncryptionService
            enc_svc = PIIEncryptionService()
        except Exception:
            enc_svc = None

        try:
            for i in range(count):
                pat_id = self.patient.uuid if i % 2 == 0 else self.other_patient.uuid
                pat_name = f"encryptedname_{i}"
                if enc_svc:
                    pat_name = enc_svc.encrypt(pat_name)
                # Insert Patient
                conn.execute("""
                    INSERT OR IGNORE INTO patients (patient_id, name, age, gender, created_at)
                    VALUES (?, ?, 'enc:v1:encage', 'enc:v1:encgender', ?);
                """, (pat_id, pat_name, now_str))
                # Insert MRI Scan
                c = conn.execute("""
                    INSERT INTO mri_scans (patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at)
                    VALUES (?, ?, 1.5, ?, '2026-08-15', ?);
                """, (pat_id, f"image_{i}.png", f"Dr. Smith_{i}", now_str))
                scan_id = c.lastrowid
                # Insert Prediction
                c = conn.execute("""
                    INSERT INTO predictions (scan_id, predicted_class, confidence_score, prob_glioma, prob_meningioma, prob_pituitary, prob_no_tumor, tumor_pixel_count, tumor_area_mm2, tumor_percentage_brain, tumor_percentage_image, estimated_brain_pixel_count, rule_based_severity, severity_rule_description, created_at)
                    VALUES (?, 'Glioblastoma', 0.95, 0.95, 0.0, 0.0, 0.05, 100, 150.0, 5.0, 10.0, 2000, 'HIGH', 'Rule description', ?);
                """, (scan_id, now_str))
                pred_id = c.lastrowid
                # Insert Clinical Report
                conn.execute("""
                    INSERT INTO clinical_reports (prediction_id, markdown_path, json_path, pdf_path, created_at)
                    VALUES (?, ?, ?, ?, ?);
                """, (pred_id, f"report_{i}.md", f"report_{i}.json", f"report_{i}.pdf", now_str))
            conn.commit()
        finally:
            conn.close()

    def test_database_indices_exist(self):
        """Index Validation: Verifies that required indexes exist on foreign keys."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='index';")
            indexes = {r[0] for r in cursor.fetchall()}
            
            # Assert missing indexes are created
            self.assertIn("idx_mri_scans_patient_id", indexes)
            self.assertIn("idx_predictions_scan_id", indexes)
            self.assertIn("idx_clinical_reports_prediction_id", indexes)
            self.assertIn("idx_doctor_patient_assignments_doc_pat", indexes)
        finally:
            conn.close()

    def test_get_authorized_patient_ids(self):
        """Authorization Service: Verifies correct patient IDs sets are returned for RBAC roles."""
        # 1. Admin returns None (unrestricted)
        self.assertIsNone(self.auth_svc.get_authorized_patient_ids(self.admin))

        # 2. Patient returns only their own uuid
        self.assertEqual(self.auth_svc.get_authorized_patient_ids(self.patient), {self.patient.uuid.lower()})

        # 3. Doctor returns assigned patient UUIDs
        self.assertEqual(self.auth_svc.get_authorized_patient_ids(self.doctor), {self.patient.uuid.lower()})
        self.assertEqual(self.auth_svc.get_authorized_patient_ids(self.other_doctor), {self.other_patient.uuid.lower()})

    def test_n1_query_elimination_history(self):
        """N+1 Reduction: Verifies history endpoint does not perform O(N) database connections inside the loop."""
        self._seed_mock_history_records(count=10)

        # Track connect calls
        connect_count = 0
        original_connect = sqlite3.connect

        def mock_connect(*args, **kwargs):
            nonlocal connect_count
            connect_count += 1
            return original_connect(*args, **kwargs)

        with patch("sqlite3.connect", side_effect=mock_connect):
            # Call history endpoint logic directly
            res = get_prediction_history(patient_id=None, current_user=self.doctor)
            
            # Since doctor is assigned to pat-opt, and i % 2 == 0 records belong to pat-opt (5 out of 10)
            self.assertEqual(len(res), 5)
            
            # The query count must be low and constant, not O(N) where N=10
            # Expected:
            # 1. get_authorized_patient_ids() -> 1 connection
            # 2. search_history() -> 1 connection (runs 2 queries internally: 1 for count, 1 for select)
            # Total connections should be <= 3 (not 12+ connection setups)
            self.assertLessEqual(connect_count, 3)

    def test_database_side_pagination_active(self):
        """Database-side Filtering/Pagination: Verifies direct database pagination when PII is not searched."""
        self._seed_mock_history_records(count=60)

        # Retrieve page 2 with page size 10
        criteria = HistorySearchCriteria(page=2, page_size=10)
        res = self.history_repo.search_history(criteria)

        self.assertEqual(len(res), 10)
        self.assertEqual(res.total_count, 60)

    def test_pii_filtering_fallback_correctness(self):
        """Database-side Filtering Fallback: Verifies correct fallback when PII filter is active."""
        self._seed_mock_history_records(count=10)

        # Patient name in mock record for Bob is encryptedname_0. Since Bob has 5 scans, searching for Bob's name should yield 5 results.
        criteria = HistorySearchCriteria(patient_name="encryptedname_0")
        res = self.history_repo.search_history(criteria)
        self.assertEqual(len(res), 5)

    def test_patient_isolation_idor(self):
        """Patient Isolation: Verifies patient cannot view other patient's history."""
        self._seed_mock_history_records(count=4)

        # Retrieve prediction history as Patient Bob
        res = get_prediction_history(patient_id=None, current_user=self.patient)
        
        # Verify Bob only sees Bob's records (2 out of 4)
        self.assertEqual(len(res), 2)
        for r in res:
            self.assertEqual(r["patient_id"], self.patient.uuid)

        # IDOR attempt: Patient Bob requests Patient Alice's history explicitly
        # The API endpoint overrides patient_id to the patient's own UUID, successfully preventing the IDOR
        res_idor = get_prediction_history(patient_id=self.other_patient.uuid, current_user=self.patient)
        self.assertEqual(len(res_idor), 2)
        for r in res_idor:
            self.assertEqual(r["patient_id"], self.patient.uuid)

    def test_doctor_assignment_isolation(self):
        """Doctor Assignment Isolation: Verifies doctor cannot view unassigned patients history."""
        self._seed_mock_history_records(count=4)

        # Dr. Smith is assigned only to Patient Bob. Should only see Bob's records.
        res = get_prediction_history(patient_id=None, current_user=self.doctor)
        self.assertEqual(len(res), 2)
        for r in res:
            self.assertEqual(r["patient_id"], self.patient.uuid)

        # IDOR attempt: Dr. Smith requests Patient Alice's records explicitly
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            get_prediction_history(patient_id=self.other_patient.uuid, current_user=self.doctor)
        self.assertEqual(ctx.exception.status_code, 403)

    def test_followup_connection_consolidation(self):
        """Follow-up Connection Reuse: Verifies follow-up service methods reuse a single connection."""
        # Seeding a patient record is required first
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("INSERT OR IGNORE INTO patients (patient_id, name, age, gender, created_at) VALUES (?, 'enc:v1:pat', 'enc:v1:30', 'enc:v1:M', '2026-08-15');", (self.patient.uuid,))
            conn.commit()
        finally:
            conn.close()

        # Track connect calls during create_followup
        connect_count = 0
        original_connect = sqlite3.connect

        def mock_connect(*args, **kwargs):
            nonlocal connect_count
            connect_count += 1
            return original_connect(*args, **kwargs)

        with patch.object(self.followup_service, "_log_audit_event"), \
             patch.object(self.followup_service, "_notify_affected_parties"), \
             patch("sqlite3.connect", side_effect=mock_connect):
            res = self.followup_service.create_followup(
                actor=self.doctor,
                patient_id=self.patient.uuid,
                scheduled_date="2026-09-15 10:00:00",
                reason="Routine checkup",
                notes="Check tumor progression"
            )
            
            # The query count must be exactly 1 connection context in followup_service
            self.assertEqual(connect_count, 1)

            # Let's test update_followup query count
            connect_count = 0
            self.followup_service.update_followup(
                actor=self.doctor,
                followup_id=res["followup_id"],
                status="completed"
            )
            self.assertEqual(connect_count, 1)

    def test_empty_database_behavior(self):
        """Empty Database behavior: Verify systems handle empty state gracefully without exceptions."""
        criteria = HistorySearchCriteria()
        res = self.history_repo.search_history(criteria)
        self.assertEqual(len(res), 0)

if __name__ == "__main__":
    unittest.main()
