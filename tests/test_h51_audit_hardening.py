import os
os.environ["DB_PATH"] = os.path.abspath("outputs/test_h51_audit_hardening.db")

import unittest
import sqlite3
import datetime
from fastapi.testclient import TestClient
from run_api import app
from security.domain.entities import Role, User
from security.infrastructure.jwt_service import JWTService
from security.infrastructure.repository import SQLiteUserRepository
from persistence.infrastructure.repository import SQLitePersistenceRepository
from clinical_reporting.application.services import ReportService

class TestH51AuditHardening(unittest.TestCase):
    """Phase H5.1: Audit Authorization Hardening focused tests."""

    def setUp(self):
        self.db_path = os.environ.get("DB_PATH", "outputs/test_h51_audit_hardening.db")
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

        # Override default DB paths in API
        import api.infrastructure.routes as api_routes
        import api.routes.auth_routes as auth_routes
        api_routes.DEFAULT_DB_PATH = self.db_path
        auth_routes.DEFAULT_DB_PATH = self.db_path

        # Disable trigger auto assignment for strict doctor assignment tests
        os.environ["DISABLE_TEST_AUTO_ASSIGN"] = "1"

        # 1. Initialize databases
        persistence_repo = SQLitePersistenceRepository(db_path=self.db_path)
        persistence_repo.initialize_db()

        self.user_repo = SQLiteUserRepository(db_path=self.db_path)
        self.user_repo.initialize_security_tables()

        # Disable triggers manually
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DROP TRIGGER IF EXISTS auto_assign_doctor_on_insert;")
            conn.execute("DROP TRIGGER IF EXISTS auto_assign_patient_on_insert;")
            conn.execute("DELETE FROM doctor_patient_assignments;")
            conn.commit()
        finally:
            conn.close()

        # 2. Setup users
        self.admin = self.user_repo.bootstrap_admin()

        # Same-name patients
        # Patient A (Jane Doe, UUID = pat-uuid-aaaa)
        self.patient_a = User(
            id=None,
            uuid="pat-uuid-aaaa",
            email="patienta@aurascan.ai",
            password_hash="fakehash",
            full_name="Jane Doe",
            role=Role.PATIENT,
            is_verified=True,
            is_active=True,
            created_at=datetime.datetime.utcnow().isoformat(),
            updated_at=datetime.datetime.utcnow().isoformat()
        )
        self.patient_a = self.user_repo.create_user(self.patient_a)

        # Patient B (Jane Doe, UUID = pat-uuid-bbbb)
        self.patient_b = User(
            id=None,
            uuid="pat-uuid-bbbb",
            email="patientb@aurascan.ai",
            password_hash="fakehash",
            full_name="Jane Doe",
            role=Role.PATIENT,
            is_verified=True,
            is_active=True,
            created_at=datetime.datetime.utcnow().isoformat(),
            updated_at=datetime.datetime.utcnow().isoformat()
        )
        self.patient_b = self.user_repo.create_user(self.patient_b)

        # Doctor 1 (assigned to Patient A only)
        self.doc_assigned = User(
            id=None,
            uuid="doc-uuid-assigned",
            email="doc_assigned@aurascan.ai",
            password_hash="fakehash",
            full_name="Dr. House",
            role=Role.DOCTOR,
            is_verified=True,
            is_active=True,
            created_at=datetime.datetime.utcnow().isoformat(),
            updated_at=datetime.datetime.utcnow().isoformat()
        )
        self.doc_assigned = self.user_repo.create_user(self.doc_assigned)

        # Doctor 2 (unassigned)
        self.doc_unassigned = User(
            id=None,
            uuid="doc-uuid-unassigned",
            email="doc_unassigned@aurascan.ai",
            password_hash="fakehash",
            full_name="Dr. Stranger",
            role=Role.DOCTOR,
            is_verified=True,
            is_active=True,
            created_at=datetime.datetime.utcnow().isoformat(),
            updated_at=datetime.datetime.utcnow().isoformat()
        )
        self.doc_unassigned = self.user_repo.create_user(self.doc_unassigned)

        # 3. Create tokens
        jwt_svc = JWTService()
        self.admin_token = jwt_svc.create_access_token(self.admin.uuid, self.admin.id, self.admin.email, self.admin.role)
        self.pat_a_token = jwt_svc.create_access_token(self.patient_a.uuid, self.patient_a.id, self.patient_a.email, self.patient_a.role)
        self.pat_b_token = jwt_svc.create_access_token(self.patient_b.uuid, self.patient_b.id, self.patient_b.email, self.patient_b.role)
        self.doc_ass_token = jwt_svc.create_access_token(self.doc_assigned.uuid, self.doc_assigned.id, self.doc_assigned.email, self.doc_assigned.role)
        self.doc_unass_token = jwt_svc.create_access_token(self.doc_unassigned.uuid, self.doc_unassigned.id, self.doc_unassigned.email, self.doc_unassigned.role)

        self.pat_a_headers = {"Authorization": f"Bearer {self.pat_a_token}"}
        self.pat_b_headers = {"Authorization": f"Bearer {self.pat_b_token}"}
        self.doc_ass_headers = {"Authorization": f"Bearer {self.doc_ass_token}"}
        self.doc_unass_headers = {"Authorization": f"Bearer {self.doc_unass_token}"}
        self.admin_headers = {"Authorization": f"Bearer {self.admin_token}"}

        # 4. Seeding data
        conn = sqlite3.connect(self.db_path)
        try:
            # Patients metadata
            conn.execute("INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES ('pat-uuid-aaaa', 'Jane Doe', 30, 'Female', '2026-08-14T00:00:00');")
            conn.execute("INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES ('pat-uuid-bbbb', 'Jane Doe', 40, 'Female', '2026-08-14T00:00:00');")

            # Report 1 owned by Patient A
            conn.execute("INSERT INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at) VALUES (1, 'RPT-A-0001', 'pat-uuid-aaaa', 1, 'FINAL', '2026-08-14T00:00:00', '2026-08-14T00:00:00');")
            # Report 2 owned by Patient B
            conn.execute("INSERT INTO reports (report_id, report_number, patient_id, current_version, status, created_at, updated_at) VALUES (2, 'RPT-B-0002', 'pat-uuid-bbbb', 1, 'FINAL', '2026-08-14T00:00:00', '2026-08-14T00:00:00');")

            # Doctor 1 assigned to Patient A
            conn.execute("INSERT INTO doctor_patient_assignments (doctor_id, patient_id, created_at) VALUES (?, 'pat-uuid-aaaa', '2026-08-14T00:00:00');", (self.doc_assigned.id,))

            # Seed Security Audit Logs
            now_str = datetime.datetime.utcnow().isoformat()
            # Log 1: Patient A views Report 1
            conn.execute("""
                INSERT INTO security_audit_logs (timestamp, event_type, user_id, email, ip_address, status, details, user_agent)
                VALUES (?, 'REPORT_VIEWED', ?, 'patienta@aurascan.ai', '127.0.0.1', 'SUCCESS', 'Report ID: 1, viewed by patient.', 'Browser');
            """, (now_str, self.patient_a.id))

            # Log 2: Patient B views Report 2
            conn.execute("""
                INSERT INTO security_audit_logs (timestamp, event_type, user_id, email, ip_address, status, details, user_agent)
                VALUES (?, 'REPORT_VIEWED', ?, 'patientb@aurascan.ai', '127.0.0.1', 'SUCCESS', 'Report ID: 2, viewed by patient.', 'Browser');
            """, (now_str, self.patient_b.id))

            # Log 3: Doctor Assigned views Report 1
            conn.execute("""
                INSERT INTO security_audit_logs (timestamp, event_type, user_id, email, ip_address, status, details, user_agent)
                VALUES (?, 'REPORT_VIEWED', ?, 'doc_assigned@aurascan.ai', '127.0.0.1', 'SUCCESS', 'Report ID: 1, viewed by doctor.', 'Browser');
            """, (now_str, self.doc_assigned.id))

            # Log 4: Non-existent Report log
            conn.execute("""
                INSERT INTO security_audit_logs (timestamp, event_type, user_id, email, ip_address, status, details, user_agent)
                VALUES (?, 'REPORT_VIEWED', ?, 'admin@aurascan.ai', '127.0.0.1', 'SUCCESS', 'Report ID: 99999, viewed by admin.', 'Browser');
            """, (now_str, self.admin.id))

            conn.commit()
        finally:
            conn.close()

        self.client = TestClient(app)

    def test_01_patient_a_cannot_access_patient_b_audit_history(self):
        """1. Patient A cannot access Patient B's report audit history logs."""
        resp = self.client.get("/api/reports/audit-history", headers=self.pat_a_headers)
        self.assertEqual(resp.status_code, 200)
        logs = resp.json()

        # Patient A should see Log 1 (own action)
        details = [l["details"] for l in logs]
        self.assertTrue(any("Report ID: 1, viewed by patient" in d for d in details))
        # Patient A must NEVER see Log 2 (Patient B's report viewed)
        self.assertFalse(any("Report ID: 2, viewed by patient" in d for d in details))

    def test_02_identical_names_isolation(self):
        """2. Two patients with identical names cannot cross-access audit history."""
        resp = self.client.get("/api/reports/audit-history", headers=self.pat_a_headers)
        self.assertEqual(resp.status_code, 200)
        logs = resp.json()

        # Verify that Jane Doe (Patient A) does not receive Jane Doe (Patient B)'s audit logs
        details = [l["details"] for l in logs]
        self.assertFalse(any("Report ID: 2, viewed by patient" in d for d in details))

    def test_03_doctor_cannot_access_unassigned_patient_audit_history(self):
        """3. Doctor cannot access unassigned patient's audit history."""
        resp = self.client.get("/api/reports/audit-history", headers=self.doc_unass_headers)
        self.assertEqual(resp.status_code, 200)
        logs = resp.json()

        # Doctor 2 is unassigned and did not perform any actions.
        # Should not see any logs related to Report 1 or Report 2.
        details = [l["details"] for l in logs]
        self.assertFalse(any("Report ID: 1" in d for d in details))
        self.assertFalse(any("Report ID: 2" in d for d in details))

    def test_04_doctor_can_access_assigned_patient_audit_history(self):
        """4. Doctor can access assigned patient's audit history."""
        resp = self.client.get("/api/reports/audit-history", headers=self.doc_ass_headers)
        self.assertEqual(resp.status_code, 200)
        logs = resp.json()

        # Doctor 1 is assigned to Patient A (Report 1).
        # Doctor 1 should see Log 1 and Log 3.
        details = [l["details"] for l in logs]
        self.assertTrue(any("Report ID: 1, viewed by patient" in d for d in details))
        self.assertTrue(any("Report ID: 1, viewed by doctor" in d for d in details))
        # Doctor 1 must not see Patient B's report logs (Report 2)
        self.assertFalse(any("Report ID: 2" in d for d in details))

    def test_05_patient_can_access_own_authorized_audit_history(self):
        """5. Patient can access own authorized audit history."""
        resp = self.client.get("/api/reports/audit-history", headers=self.pat_a_headers)
        self.assertEqual(resp.status_code, 200)
        logs = resp.json()

        details = [l["details"] for l in logs]
        self.assertTrue(any("Report ID: 1, viewed by patient" in d for d in details))

    def test_06_unauthenticated_access_is_denied(self):
        """6. Unauthenticated access is denied."""
        resp = self.client.get("/api/reports/audit-history")
        self.assertIn(resp.status_code, [401, 403])

    def test_07_invalid_resource_identifiers_fail_safely(self):
        """7. Invalid resource identifiers fail safely."""
        # Patient A should not see the log with invalid report ID 99999
        resp = self.client.get("/api/reports/audit-history", headers=self.pat_a_headers)
        self.assertEqual(resp.status_code, 200)
        logs = resp.json()
        details = [l["details"] for l in logs]
        self.assertFalse(any("Report ID: 99999" in d for d in details))

    def test_08_admin_can_access_all_audit_history(self):
        """8. Admin can access all audit history logs (wildcard)."""
        resp = self.client.get("/api/reports/audit-history", headers=self.admin_headers)
        self.assertEqual(resp.status_code, 200)
        logs = resp.json()

        details = [l["details"] for l in logs]
        self.assertTrue(any("Report ID: 1, viewed by patient" in d for d in details))
        self.assertTrue(any("Report ID: 2, viewed by patient" in d for d in details))
        self.assertTrue(any("Report ID: 99999" in d for d in details))
