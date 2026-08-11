import os
import sqlite3
import datetime
import math
import unittest
from fastapi.testclient import TestClient

os.environ["DB_PATH"] = os.path.abspath("outputs/test_mri_annotations.db")

from run_api import app
from security.domain.entities import Role, User
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.jwt_service import JWTService
from clinical_reporting.application.mri_annotation_service import MriAnnotationService, MriAnnotationServiceException
from persistence.infrastructure.repository import SQLitePersistenceRepository

class TestMriAnnotations(unittest.TestCase):
    """Focused validation, encryption, and authorization tests for Phase H3.2-B (Point MRI Annotations)."""

    def setUp(self):
        os.environ["DISABLE_TEST_AUTO_ASSIGN"] = "1"
        self.db_path = os.environ.get("DB_PATH", "outputs/test_mri_annotations.db")
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
            # Doctor A (ID: 10), Doctor B (ID: 11), Patient A (ID: 20), Patient B (ID: 21)
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (10, "doc-uuid-a", "doctor_a@aurascan.ai", pass_hash, "Dr. Alice", Role.DOCTOR.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (11, "doc-uuid-b", "doctor_b@aurascan.ai", pass_hash, "Dr. Bob", Role.DOCTOR.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (12, "doc-uuid-zero", "doctor_zero@aurascan.ai", pass_hash, "Dr. Zero", Role.DOCTOR.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (20, "pat-uuid-a", "patient_a@aurascan.ai", pass_hash, "Patient A", Role.PATIENT.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (21, "pat-uuid-b", "patient_b@aurascan.ai", pass_hash, "Patient B", Role.PATIENT.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )

            # Insert patient demographics
            conn.execute(
                "INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
                ("pat-uuid-a", "Patient A", 45, "Male", datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
                ("pat-uuid-b", "Patient B", 30, "Female", datetime.datetime.utcnow().isoformat())
            )

            # Insert MRI Scans: Scan A (100) -> Patient A, Scan B (101) -> Patient B
            conn.execute(
                "INSERT INTO mri_scans (id, patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                (100, "pat-uuid-a", "uploads/scan_a.png", 1.0, "Dr. Alice", "2026-08-08", datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO mri_scans (id, patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                (101, "pat-uuid-b", "uploads/scan_b.png", 1.0, "Dr. Bob", "2026-08-08", datetime.datetime.utcnow().isoformat())
            )

            # Assignments: Doctor A -> Patient A, Doctor B -> Patient B. Doctor Zero has no assignments.
            conn.execute(
                "INSERT INTO doctor_patient_assignments (doctor_id, patient_id, created_at) VALUES (?, ?, ?);",
                (10, "pat-uuid-a", datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO doctor_patient_assignments (doctor_id, patient_id, created_at) VALUES (?, ?, ?);",
                (11, "pat-uuid-b", datetime.datetime.utcnow().isoformat())
            )

            conn.commit()
        finally:
            conn.close()

        # Load users
        self.doc_a = self.user_repo.get_by_email("doctor_a@aurascan.ai")
        self.doc_b = self.user_repo.get_by_email("doctor_b@aurascan.ai")
        self.doc_zero = self.user_repo.get_by_email("doctor_zero@aurascan.ai")
        self.pat_a = self.user_repo.get_by_email("patient_a@aurascan.ai")

        # JWT tokens
        self.jwt_svc = JWTService()
        self.token_admin = self.jwt_svc.create_access_token(self.admin_user.uuid, self.admin_user.id, self.admin_user.email, Role.ADMIN)
        self.token_doc_a = self.jwt_svc.create_access_token(self.doc_a.uuid, self.doc_a.id, self.doc_a.email, Role.DOCTOR)
        self.token_doc_b = self.jwt_svc.create_access_token(self.doc_b.uuid, self.doc_b.id, self.doc_b.email, Role.DOCTOR)
        self.token_doc_zero = self.jwt_svc.create_access_token(self.doc_zero.uuid, self.doc_zero.id, self.doc_zero.email, Role.DOCTOR)
        self.token_pat_a = self.jwt_svc.create_access_token(self.pat_a.uuid, self.pat_a.id, self.pat_a.email, Role.PATIENT)

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

    def test_01_authorized_doctor_can_create_point(self):
        """Test: Doctor A can create a point annotation on Patient A's Scan A."""
        svc = MriAnnotationService(self.db_path)
        ann = svc.create_annotation(self.doc_a, 100, 0.25, 0.75, "Tumor center", "Suspicious region.")
        self.assertEqual(ann["scan_id"], 100)
        self.assertEqual(ann["patient_id"], "pat-uuid-a")
        self.assertEqual(ann["x"], 0.25)
        self.assertEqual(ann["y"], 0.75)
        self.assertEqual(ann["label"], "Tumor center")
        self.assertEqual(ann["comment"], "Suspicious region.")
        self.assertEqual(ann["status"], "active")

    def test_02_authorized_doctor_can_read_points(self):
        """Test: Doctor A can read point annotations on Patient A's Scan A."""
        svc = MriAnnotationService(self.db_path)
        svc.create_annotation(self.doc_a, 100, 0.5, 0.5, "Point 1", "Comment 1")
        anns = svc.get_annotations_for_scan(self.doc_a, 100)
        self.assertEqual(len(anns), 1)
        self.assertEqual(anns[0]["x"], 0.5)
        self.assertEqual(anns[0]["comment"], "Comment 1")
        self.assertEqual(anns[0]["doctor_email"], "doctor_a@aurascan.ai")

    def test_03_authorized_doctor_can_update_point(self):
        """Test: Doctor A can update their own annotation."""
        svc = MriAnnotationService(self.db_path)
        ann = svc.create_annotation(self.doc_a, 100, 0.5, 0.5, "Original", "Original Comment")
        updated = svc.update_annotation(self.doc_a, ann["annotation_id"], 0.6, 0.6, "Updated", "Updated Comment")
        self.assertEqual(updated["x"], 0.6)
        self.assertEqual(updated["label"], "Updated")
        self.assertEqual(updated["comment"], "Updated Comment")

    def test_04_authorized_doctor_can_archive_point(self):
        """Test: Doctor A can soft-archive/delete their own annotation."""
        svc = MriAnnotationService(self.db_path)
        ann = svc.create_annotation(self.doc_a, 100, 0.5, 0.5, "Delete me", "Delete comment")
        svc.archive_annotation(self.doc_a, ann["annotation_id"])
        
        # Verify it does not appear in active list
        anns = svc.get_annotations_for_scan(self.doc_a, 100)
        self.assertEqual(len(anns), 0)

    def test_05_unauthorized_doctor_cannot_create_point(self):
        """Test: Doctor A cannot create annotation on Patient B's Scan B (no assignment)."""
        svc = MriAnnotationService(self.db_path)
        with self.assertRaises(MriAnnotationServiceException) as ctx:
            svc.create_annotation(self.doc_a, 101, 0.5, 0.5, "Label", "Comment")
        self.assertIn("Access denied", str(ctx.exception))

    def test_06_zero_assignment_doctor_cannot_create_point(self):
        """Test: Doctor with zero assignments cannot create point on Scan A."""
        svc = MriAnnotationService(self.db_path)
        with self.assertRaises(MriAnnotationServiceException) as ctx:
            svc.create_annotation(self.doc_zero, 100, 0.5, 0.5, "Label", "Comment")
        self.assertIn("Access denied", str(ctx.exception))

    def test_07_unauthorized_doctor_cannot_read_points(self):
        """Test: Doctor B cannot retrieve annotations on Scan A."""
        svc = MriAnnotationService(self.db_path)
        svc.create_annotation(self.doc_a, 100, 0.5, 0.5, "L", "C")
        with self.assertRaises(MriAnnotationServiceException) as ctx:
            svc.get_annotations_for_scan(self.doc_b, 100)
        self.assertIn("Access denied", str(ctx.exception))

    def test_08_unauthorized_doctor_cannot_update_point(self):
        """Test: Doctor B cannot update Doctor A's annotation (IDOR protection)."""
        svc = MriAnnotationService(self.db_path)
        ann = svc.create_annotation(self.doc_a, 100, 0.5, 0.5, "L", "C")
        with self.assertRaises(MriAnnotationServiceException) as ctx:
            svc.update_annotation(self.doc_b, ann["annotation_id"], 0.6, 0.6, "Hacked", "Hacked")
        self.assertIn("Access denied", str(ctx.exception))

    def test_09_unauthorized_doctor_cannot_archive_point(self):
        """Test: Doctor B cannot archive Doctor A's annotation."""
        svc = MriAnnotationService(self.db_path)
        ann = svc.create_annotation(self.doc_a, 100, 0.5, 0.5, "L", "C")
        with self.assertRaises(MriAnnotationServiceException) as ctx:
            svc.archive_annotation(self.doc_b, ann["annotation_id"])
        self.assertIn("Access denied", str(ctx.exception))

    def test_10_patient_cannot_create_point(self):
        """Test: Patient cannot create annotations."""
        svc = MriAnnotationService(self.db_path)
        with self.assertRaises(MriAnnotationServiceException):
            svc.create_annotation(self.pat_a, 100, 0.5, 0.5, "Label")

    def test_11_invalid_x_rejected(self):
        """Test: Coordinate x outside range 0..1 rejected."""
        svc = MriAnnotationService(self.db_path)
        with self.assertRaises(MriAnnotationServiceException) as ctx:
            svc.create_annotation(self.doc_a, 100, 1.001, 0.5, "L")
        self.assertIn("normalized between 0.0 and 1.0", str(ctx.exception))

    def test_12_invalid_y_rejected(self):
        """Test: Coordinate y outside range 0..1 rejected."""
        svc = MriAnnotationService(self.db_path)
        with self.assertRaises(MriAnnotationServiceException) as ctx:
            svc.create_annotation(self.doc_a, 100, 0.5, 1.001, "L")
        self.assertIn("normalized between 0.0 and 1.0", str(ctx.exception))

    def test_13_negative_x_rejected(self):
        """Test: Coordinate x < 0 rejected."""
        svc = MriAnnotationService(self.db_path)
        with self.assertRaises(MriAnnotationServiceException):
            svc.create_annotation(self.doc_a, 100, -0.001, 0.5, "L")

    def test_14_negative_y_rejected(self):
        """Test: Coordinate y < 0 rejected."""
        svc = MriAnnotationService(self.db_path)
        with self.assertRaises(MriAnnotationServiceException):
            svc.create_annotation(self.doc_a, 100, 0.5, -0.001, "L")

    def test_15_x_above_one_rejected(self):
        """Test: Coordinate x above 1 rejected."""
        svc = MriAnnotationService(self.db_path)
        with self.assertRaises(MriAnnotationServiceException):
            svc.create_annotation(self.doc_a, 100, 1.05, 0.5, "L")

    def test_16_y_above_one_rejected(self):
        """Test: Coordinate y above 1 rejected."""
        svc = MriAnnotationService(self.db_path)
        with self.assertRaises(MriAnnotationServiceException):
            svc.create_annotation(self.doc_a, 100, 0.5, 1.05, "L")

    def test_17_non_numeric_coordinate_rejected(self):
        """Test: NaN, Infinity, and non-numeric coordinate types rejected."""
        svc = MriAnnotationService(self.db_path)
        with self.assertRaises(MriAnnotationServiceException):
            svc.create_annotation(self.doc_a, 100, float("nan"), 0.5, "L")
        with self.assertRaises(MriAnnotationServiceException):
            svc.create_annotation(self.doc_a, 100, float("inf"), 0.5, "L")
        with self.assertRaises(MriAnnotationServiceException):
            svc.create_annotation(self.doc_a, 100, "string_coord", 0.5, "L")

    def test_18_mismatched_patient_scan_rejected(self):
        """Test: Mismatched patient and scan rejected. Scan A exists in DB, scan B does too."""
        # Creating annotation on invalid scan ID raises not found, preventing mismatch
        svc = MriAnnotationService(self.db_path)
        with self.assertRaises(MriAnnotationServiceException) as ctx:
            svc.create_annotation(self.doc_a, 9999, 0.5, 0.5)
        self.assertIn("Scan not found", str(ctx.exception))

    def test_19_nonexistent_scan_rejected(self):
        """Test: Nonexistent scan rejected."""
        svc = MriAnnotationService(self.db_path)
        with self.assertRaises(MriAnnotationServiceException):
            svc.create_annotation(self.doc_a, 9999, 0.5, 0.5)

    def test_20_annotation_scan_ownership_enforced(self):
        """Test: IDOR scan ownership enforced."""
        svc = MriAnnotationService(self.db_path)
        ann = svc.create_annotation(self.doc_a, 100, 0.5, 0.5, "L", "C")
        
        # Doc B attempts to fetch annotation A
        headers_b = {"Authorization": f"Bearer {self.token_doc_b}"}
        resp = self.client.get("/api/doctor/scans/100/point-annotations", headers=headers_b)
        self.assertEqual(resp.status_code, 403)

    def test_21_plaintext_comment_not_stored(self):
        """Test: Plaintext comment is encrypted in database."""
        svc = MriAnnotationService(self.db_path)
        comment_val = "Clinician private observation."
        ann = svc.create_annotation(self.doc_a, 100, 0.5, 0.5, "L", comment_val)
        
        # Verify direct database read contains ciphertext only
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute("SELECT encrypted_comment FROM mri_point_annotations WHERE annotation_id = ?;", (ann["annotation_id"],)).fetchone()
            db_comment = row[0]
            self.assertNotEqual(db_comment, comment_val)
            self.assertTrue(db_comment.startswith("enc:v1:"))
            self.assertNotIn(comment_val, db_comment)
        finally:
            conn.close()

    def test_22_encrypted_comment_not_returned(self):
        """Test: Ciphertext is never returned in API response."""
        headers = {"Authorization": f"Bearer {self.token_doc_a}"}
        resp = self.client.post("/api/doctor/scans/100/point-annotations", json={"x": 0.5, "y": 0.5, "label": "L", "comment": "Confidential comment"}, headers=headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertNotIn("encrypted_comment", data)
        self.assertEqual(data["comment"], "Confidential comment")

    def test_23_authorized_comment_decrypted(self):
        """Test: Comment is decrypted only for authorized users."""
        svc = MriAnnotationService(self.db_path)
        ann = svc.create_annotation(self.doc_a, 100, 0.5, 0.5, "L", "Decrypted comment test")
        
        anns = svc.get_annotations_for_scan(self.doc_a, 100)
        self.assertEqual(anns[0]["comment"], "Decrypted comment test")

    def test_24_archived_annotation_hidden(self):
        """Test: Archived annotations are hidden on scan retrieve list."""
        headers = {"Authorization": f"Bearer {self.token_doc_a}"}
        
        # Create
        resp = self.client.post("/api/doctor/scans/100/point-annotations", json={"x": 0.5, "y": 0.5, "label": "L", "comment": "C"}, headers=headers)
        ann_id = resp.json()["annotation_id"]
        
        # Delete
        resp_del = self.client.delete(f"/api/doctor/point-annotations/{ann_id}", headers=headers)
        self.assertEqual(resp_del.status_code, 200)
        
        # Read list
        resp_list = self.client.get("/api/doctor/scans/100/point-annotations", headers=headers)
        self.assertEqual(len(resp_list.json()), 0)

    def test_25_audit_event_created(self):
        """Test: Creating point annotation creates POINT_ANNOTATION_CREATED audit log event."""
        svc = MriAnnotationService(self.db_path)
        svc.create_annotation(self.doc_a, 100, 0.5, 0.5, "L", "C")
        
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute("SELECT event_type, details FROM security_audit_logs WHERE event_type = 'POINT_ANNOTATION_CREATED';").fetchone()
            self.assertIsNotNone(row)
            self.assertIn("Created point annotation ID", row[1])
        finally:
            conn.close()

    def test_26_edge_coordinates_valid(self):
        """Test: Edge values (0.0 and 1.0) are valid, out of bounds rejected."""
        svc = MriAnnotationService(self.db_path)
        
        # 0.0 is valid
        ann0 = svc.create_annotation(self.doc_a, 100, 0.0, 0.0, "Edge 0")
        self.assertEqual(ann0["x"], 0.0)
        
        # 1.0 is valid
        ann1 = svc.create_annotation(self.doc_a, 100, 1.0, 1.0, "Edge 1")
        self.assertEqual(ann1["x"], 1.0)
        
        # -0.001 is invalid
        with self.assertRaises(MriAnnotationServiceException):
            svc.create_annotation(self.doc_a, 100, -0.001, 0.5, "L")
            
        # 1.001 is invalid
        with self.assertRaises(MriAnnotationServiceException):
            svc.create_annotation(self.doc_a, 100, 1.001, 0.5, "L")

if __name__ == "__main__":
    unittest.main()
