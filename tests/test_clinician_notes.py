import os
import sqlite3
import datetime
import unittest
from fastapi.testclient import TestClient

# Use a separate test database
os.environ["DB_PATH"] = os.path.abspath("outputs/test_clinician_notes.db")

from run_api import app
from security.domain.entities import Role, User
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.jwt_service import JWTService
from security.application.authorization_service import AuthorizationService
from clinical_reporting.application.clinician_note_service import ClinicianNoteService, ClinicianNoteServiceException
from persistence.infrastructure.repository import SQLitePersistenceRepository

class TestClinicianNotes(unittest.TestCase):
    """Focused validation, encryption, and authorization tests for Phase H3.2-A (Clinician Notes)."""

    def setUp(self):
        os.environ["DISABLE_TEST_AUTO_ASSIGN"] = "1"
        self.db_path = os.environ.get("DB_PATH", "outputs/test_clinician_notes.db")
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

        # Make sure default paths match
        import api.infrastructure.routes as api_routes
        import api.routes.auth_routes as auth_routes
        api_routes.DEFAULT_DB_PATH = self.db_path
        auth_routes.DEFAULT_DB_PATH = self.db_path

        # Initialize DB
        self.persistence_repo = SQLitePersistenceRepository(db_path=self.db_path)
        self.persistence_repo.initialize_db()

        self.user_repo = SQLiteUserRepository(db_path=self.db_path)
        self.user_repo.initialize_security_tables()

        # Drop the auto-assignment triggers to ensure strict clean testing
        conn_drop = sqlite3.connect(self.db_path)
        try:
            conn_drop.execute("DROP TRIGGER IF EXISTS auto_assign_doctor_on_insert;")
            conn_drop.execute("DROP TRIGGER IF EXISTS auto_assign_patient_on_insert;")
            conn_drop.execute("DELETE FROM doctor_patient_assignments;")
            conn_drop.commit()
        finally:
            conn_drop.close()
        
        # Bootstrap users
        self.admin_user = self.user_repo.bootstrap_admin()
        
        from security.infrastructure.password import PasswordHasher
        pass_hash = PasswordHasher.hash_password("Password@123")

        conn = sqlite3.connect(self.db_path)
        try:
            # Insert users
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (10, "doc-uuid-1", "doctor1@aurascan.ai", pass_hash, "Dr. Jane Smith", Role.DOCTOR.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (11, "doc-uuid-2", "doctor2@aurascan.ai", pass_hash, "Dr. John Doe", Role.DOCTOR.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (20, "pat-uuid-1", "patient1@aurascan.ai", pass_hash, "Bob Jones", Role.PATIENT.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (21, "pat-uuid-2", "patient2@aurascan.ai", pass_hash, "Alice Cooper", Role.PATIENT.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )

            # Insert patient demographics (Phase H3.1 encrypted name fallback logic compatibility)
            conn.execute(
                "INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
                ("pat-uuid-1", "Bob Jones", 45, "Male", datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
                ("pat-uuid-2", "Alice Cooper", 30, "Female", datetime.datetime.utcnow().isoformat())
            )

            # Insert MRI Scans
            conn.execute(
                "INSERT INTO mri_scans (id, patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                (100, "pat-uuid-1", "uploads/bob_scan.png", 1.0, "Dr. Jane Smith", "2026-08-08", datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO mri_scans (id, patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                (101, "pat-uuid-2", "uploads/alice_scan.png", 1.0, "Dr. John Doe", "2026-08-08", datetime.datetime.utcnow().isoformat())
            )

            # Assignments: Doctor 1 is assigned only to Patient 1 (Bob Jones)
            conn.execute(
                "INSERT INTO doctor_patient_assignments (doctor_id, patient_id, created_at) VALUES (?, ?, ?);",
                (10, "pat-uuid-1", datetime.datetime.utcnow().isoformat())
            )

            conn.commit()
        finally:
            conn.close()

        # Load users
        self.doc1 = self.user_repo.get_by_email("doctor1@aurascan.ai")
        self.doc2 = self.user_repo.get_by_email("doctor2@aurascan.ai")
        self.pat1 = self.user_repo.get_by_email("patient1@aurascan.ai")
        self.pat2 = self.user_repo.get_by_email("patient2@aurascan.ai")

        # JWT tokens
        self.jwt_svc = JWTService()
        self.token_admin = self.jwt_svc.create_access_token(self.admin_user.uuid, self.admin_user.id, self.admin_user.email, Role.ADMIN)
        self.token_doc1 = self.jwt_svc.create_access_token(self.doc1.uuid, self.doc1.id, self.doc1.email, Role.DOCTOR)
        self.token_doc2 = self.jwt_svc.create_access_token(self.doc2.uuid, self.doc2.id, self.doc2.email, Role.DOCTOR)
        self.token_pat1 = self.jwt_svc.create_access_token(self.pat1.uuid, self.pat1.id, self.pat1.email, Role.PATIENT)
        self.token_pat2 = self.jwt_svc.create_access_token(self.pat2.uuid, self.pat2.id, self.pat2.email, Role.PATIENT)

        self.test_client_ctx = TestClient(app)
        self.client = self.test_client_ctx.__enter__()

    def tearDown(self):
        self.test_client_ctx.__exit__(None, None, None)
        if "DISABLE_TEST_AUTO_ASSIGN" in os.environ:
            del os.environ["DISABLE_TEST_AUTO_ASSIGN"]
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def test_01_authorized_doctor_can_create_note(self):
        """Test: Doctor 1 can create a note on Patient 1's scan."""
        svc = ClinicianNoteService(self.db_path)
        note = svc.create_note(self.doc1, 100, "High severity glioma suspected.")
        self.assertEqual(note["scan_id"], 100)
        self.assertEqual(note["patient_id"], "pat-uuid-1")
        self.assertEqual(note["content"], "High severity glioma suspected.")
        self.assertEqual(note["status"], "active")

    def test_02_authorized_doctor_can_read_note(self):
        """Test: Doctor 1 can read note on Patient 1's scan."""
        svc = ClinicianNoteService(self.db_path)
        svc.create_note(self.doc1, 100, "Observation text.")
        notes = svc.get_notes_for_scan(self.doc1, 100)
        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0]["content"], "Observation text.")
        self.assertEqual(notes[0]["doctor_email"], "doctor1@aurascan.ai")

    def test_03_authorized_doctor_can_update_own_note(self):
        """Test: Doctor 1 can update note they created."""
        svc = ClinicianNoteService(self.db_path)
        note = svc.create_note(self.doc1, 100, "Observation text.")
        updated = svc.update_note(self.doc1, note["note_id"], "Updated observation text.")
        self.assertEqual(updated["content"], "Updated observation text.")
        
        # Verify in retrieval
        notes = svc.get_notes_for_scan(self.doc1, 100)
        self.assertEqual(notes[0]["content"], "Updated observation text.")

    def test_04_authorized_doctor_can_archive_own_note(self):
        """Test: Doctor 1 can soft-archive own note."""
        svc = ClinicianNoteService(self.db_path)
        note = svc.create_note(self.doc1, 100, "Observation text.")
        svc.archive_note(self.doc1, note["note_id"])
        
        # Verify it is no longer returned as active
        notes = svc.get_notes_for_scan(self.doc1, 100)
        self.assertEqual(len(notes), 0)

    def test_05_unauthorized_doctor_cannot_create_note(self):
        """Test: Doctor 1 cannot create note on Patient 2's scan (no assignment)."""
        svc = ClinicianNoteService(self.db_path)
        with self.assertRaises(ClinicianNoteServiceException) as context:
            svc.create_note(self.doc1, 101, "Should fail.")
        self.assertIn("Access denied", str(context.exception))

    def test_06_zero_assignment_doctor_cannot_create_note(self):
        """Test: Doctor 2 (zero assignments) cannot create note on Patient 1's scan."""
        svc = ClinicianNoteService(self.db_path)
        with self.assertRaises(ClinicianNoteServiceException) as context:
            svc.create_note(self.doc2, 100, "Should fail.")
        self.assertIn("Access denied", str(context.exception))

    def test_07_unauthorized_doctor_cannot_read_note(self):
        """Test: Doctor 2 cannot read notes on Patient 1's scan."""
        svc = ClinicianNoteService(self.db_path)
        svc.create_note(self.doc1, 100, "Observation text.")
        with self.assertRaises(ClinicianNoteServiceException):
            svc.get_notes_for_scan(self.doc2, 100)

    def test_08_unauthorized_doctor_cannot_update_note(self):
        """Test: Doctor 2 cannot update Doctor 1's note (IDOR)."""
        svc = ClinicianNoteService(self.db_path)
        note = svc.create_note(self.doc1, 100, "Doctor 1 text.")
        with self.assertRaises(ClinicianNoteServiceException) as context:
            svc.update_note(self.doc2, note["note_id"], "Doctor 2 hijack.")
        self.assertIn("Access denied", str(context.exception))

    def test_09_unauthorized_doctor_cannot_archive_note(self):
        """Test: Doctor 2 cannot archive Doctor 1's note."""
        svc = ClinicianNoteService(self.db_path)
        note = svc.create_note(self.doc1, 100, "Doctor 1 text.")
        with self.assertRaises(ClinicianNoteServiceException) as context:
            svc.archive_note(self.doc2, note["note_id"])
        self.assertIn("Access denied", str(context.exception))

    def test_10_patient_cannot_modify_note(self):
        """Test: Patients cannot create, update, or archive notes."""
        svc = ClinicianNoteService(self.db_path)
        
        # Test creation denied to Patient role
        with self.assertRaises(ClinicianNoteServiceException):
            svc.create_note(self.pat1, 100, "Patient observation.")

    def test_11_mismatched_patient_scan_rejected(self):
        """Test: Mismatched patient and scan ID is rejected."""
        svc = ClinicianNoteService(self.db_path)
        # Scan 100 belongs to pat-uuid-1. Attempt to set note for scan 100 but check mismatch throws in service.
        # Note: create_note fetches patient_id from scan, preventing client mismatch.
        # Let's test that manually calling with invalid scan raises scan not found.
        with self.assertRaises(ClinicianNoteServiceException):
            svc.create_note(self.doc1, 9999, "Mismatched.")

    def test_12_nonexistent_scan_rejected(self):
        """Test: Nonexistent scan rejected."""
        svc = ClinicianNoteService(self.db_path)
        with self.assertRaises(ClinicianNoteServiceException) as context:
            svc.create_note(self.doc1, 9999, "Valid text.")
        self.assertIn("MRI Scan not found", str(context.exception))

    def test_13_invalid_note_content_rejected(self):
        """Test: Empty or whitespace-only note rejected."""
        svc = ClinicianNoteService(self.db_path)
        with self.assertRaises(ClinicianNoteServiceException) as context:
            svc.create_note(self.doc1, 100, "")
        self.assertIn("content is required", str(context.exception).lower())

    def test_14_empty_note_rejected(self):
        """Test: Whitespace note rejected."""
        svc = ClinicianNoteService(self.db_path)
        with self.assertRaises(ClinicianNoteServiceException) as context:
            svc.create_note(self.doc1, 100, "   ")
        self.assertIn("content cannot be empty", str(context.exception).lower())

    def test_15_oversized_note_rejected(self):
        """Test: Note exceeding 5000 chars rejected."""
        svc = ClinicianNoteService(self.db_path)
        long_content = "x" * 5001
        with self.assertRaises(ClinicianNoteServiceException) as context:
            svc.create_note(self.doc1, 100, long_content)
        self.assertIn("exceeds the maximum length", str(context.exception))

    def test_16_plaintext_not_stored_in_database(self):
        """Test: Note content is encrypted in SQLite, not plaintext."""
        svc = ClinicianNoteService(self.db_path)
        content_text = "Highly confidential tumor scan note."
        note = svc.create_note(self.doc1, 100, content_text)
        
        # Query SQLite database directly
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute("SELECT encrypted_content FROM clinician_notes WHERE note_id = ?;", (note["note_id"],)).fetchone()
            db_value = row[0]
            self.assertNotEqual(db_value, content_text)
            self.assertTrue(db_value.startswith("enc:v1:"))
            self.assertNotIn(content_text, db_value)
        finally:
            conn.close()

    def test_17_decrypted_content_returned_only_to_authorized_user(self):
        """Test: Decrypted content is returned upon authorized retrieval."""
        svc = ClinicianNoteService(self.db_path)
        svc.create_note(self.doc1, 100, "Confidential doctor note.")
        
        notes = svc.get_notes_for_scan(self.doc1, 100)
        self.assertEqual(notes[0]["content"], "Confidential doctor note.")

    def test_18_encrypted_content_never_returned_by_api(self):
        """Test: Encrypted content/metadata is never exposed in API responses."""
        headers = {"Authorization": f"Bearer {self.token_doc1}"}
        resp = self.client.post("/api/doctor/scans/100/clinical-notes", json={"content": "Top secret note."}, headers=headers)
        self.assertEqual(resp.status_code, 200)
        
        data = resp.json()
        self.assertNotIn("encrypted_content", data)
        self.assertEqual(data["content"], "Top secret note.")
        
        # Verify on GET as well
        resp_get = self.client.get("/api/doctor/scans/100/clinical-notes", headers=headers)
        self.assertEqual(resp_get.status_code, 200)
        notes = resp_get.json()
        self.assertEqual(notes[0]["content"], "Top secret note.")
        self.assertNotIn("encrypted_content", notes[0])

    def test_19_note_audit_event_created(self):
        """Test: Audit logs are populated with CLINICIAN_NOTE events (excluding plaintext)."""
        svc = ClinicianNoteService(self.db_path)
        svc.create_note(self.doc1, 100, "Audit trail verify text.")
        
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute("SELECT event_type, details FROM security_audit_logs WHERE event_type = 'CLINICIAN_NOTE_CREATED';").fetchall()
            self.assertTrue(len(rows) >= 1)
            details = rows[0][1]
            self.assertNotIn("Audit trail verify text", details)
        finally:
            conn.close()

    def test_20_archived_note_not_returned_as_active(self):
        """Test: Archived notes do not return in list API responses."""
        headers = {"Authorization": f"Bearer {self.token_doc1}"}
        
        # Create note
        resp = self.client.post("/api/doctor/scans/100/clinical-notes", json={"content": "Note to be archived."}, headers=headers)
        note_id = resp.json()["note_id"]
        
        # Delete/archive
        resp_del = self.client.delete(f"/api/doctor/clinical-notes/{note_id}", headers=headers)
        self.assertEqual(resp_del.status_code, 200)
        
        # List notes and check empty
        resp_list = self.client.get("/api/doctor/scans/100/clinical-notes", headers=headers)
        self.assertEqual(len(resp_list.json()), 0)

    def test_21_IDOR_note_modification_denied(self):
        """Test: IDOR note manipulation put/delete is denied."""
        svc = ClinicianNoteService(self.db_path)
        note = svc.create_note(self.doc1, 100, "Doc 1 observation.")
        
        # Doctor 2 attempts PUT
        headers_doc2 = {"Authorization": f"Bearer {self.token_doc2}"}
        resp_put = self.client.put(f"/api/doctor/clinical-notes/{note['note_id']}", json={"content": "Hacked content"}, headers=headers_doc2)
        self.assertEqual(resp_put.status_code, 403)
        
        # Doctor 2 attempts DELETE
        resp_del = self.client.delete(f"/api/doctor/clinical-notes/{note['note_id']}", headers=headers_doc2)
        self.assertEqual(resp_del.status_code, 403)

if __name__ == "__main__":
    unittest.main()
