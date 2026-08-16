import os
import sqlite3
import datetime
import unittest
from fastapi.testclient import TestClient

os.environ["DB_PATH"] = os.path.abspath("outputs/test_segmentation_reviews.db")

from run_api import app
from security.domain.entities import Role, User
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.jwt_service import JWTService
from clinical_reporting.application.segmentation_review_service import SegmentationReviewService, SegmentationReviewServiceException
from persistence.infrastructure.repository import SQLitePersistenceRepository

class TestSegmentationReviews(unittest.TestCase):
    """Focused integration and security validation tests for Phase I2 Clinician Segmentation Reviews."""

    def setUp(self):
        os.environ["DISABLE_TEST_AUTO_ASSIGN"] = "1"
        self.db_path = os.environ.get("DB_PATH", "outputs/test_segmentation_reviews.db")
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

        import api.infrastructure.routes as api_routes
        import api.routes.auth_routes as auth_routes
        api_routes.DEFAULT_DB_PATH = self.db_path
        auth_routes.DEFAULT_DB_PATH = self.db_path

        # Initialize persistence database and tables
        self.persistence_repo = SQLitePersistenceRepository(db_path=self.db_path)
        self.persistence_repo.initialize_db()

        self.user_repo = SQLiteUserRepository(db_path=self.db_path)
        self.user_repo.initialize_security_tables()

        # Clean/drop auto-assign triggers
        conn_drop = sqlite3.connect(self.db_path)
        try:
            conn_drop.execute("DROP TRIGGER IF EXISTS auto_assign_doctor_on_insert;")
            conn_drop.execute("DROP TRIGGER IF EXISTS auto_assign_patient_on_insert;")
            conn_drop.execute("DELETE FROM doctor_patient_assignments;")
            conn_drop.commit()
        finally:
            conn_drop.close()

        # Bootstrap Admin
        self.admin_user = self.user_repo.bootstrap_admin()

        from security.infrastructure.password import PasswordHasher
        pass_hash = PasswordHasher.hash_password("Password@123")

        conn = sqlite3.connect(self.db_path)
        try:
            # Doctor A (10), Doctor B (11), Patient A (20), Patient B (21)
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
                (20, "pat-uuid-a", "patient_a@aurascan.ai", pass_hash, "Patient A", Role.PATIENT.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (21, "pat-uuid-b", "patient_b@aurascan.ai", pass_hash, "Patient B", Role.PATIENT.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )

            # Insert Patients demographics
            conn.execute(
                "INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
                ("pat-uuid-a", "Patient A", 45, "Male", datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
                ("pat-uuid-b", "Patient B", 30, "Female", datetime.datetime.utcnow().isoformat())
            )

            # Insert scans
            conn.execute(
                "INSERT INTO mri_scans (id, patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                (100, "pat-uuid-a", "uploads/scan_a.png", 1.0, "Dr. Alice", "2026-08-08", datetime.datetime.utcnow().isoformat())
            )
            conn.execute(
                "INSERT INTO mri_scans (id, patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at) VALUES (?, ?, ?, ?, ?, ?, ?);",
                (101, "pat-uuid-b", "uploads/scan_b.png", 1.0, "Dr. Bob", "2026-08-08", datetime.datetime.utcnow().isoformat())
            )

            # Insert prediction rows for scans
            conn.execute("""
                INSERT INTO predictions (
                    id, scan_id, predicted_class, confidence_score, prob_glioma, prob_meningioma, prob_pituitary,
                    prob_no_tumor, tumor_pixel_count, tumor_area_mm2, tumor_percentage_brain, tumor_percentage_image,
                    estimated_brain_pixel_count, rule_based_severity, severity_rule_description, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (500, 100, "Glioma", 0.95, 0.95, 0.02, 0.02, 0.01, 1200, 300.0, 5.0, 3.0, 24000, "MEDIUM", "Observed glioma tumor mass", datetime.datetime.utcnow().isoformat()))

            conn.execute("""
                INSERT INTO predictions (
                    id, scan_id, predicted_class, confidence_score, prob_glioma, prob_meningioma, prob_pituitary,
                    prob_no_tumor, tumor_pixel_count, tumor_area_mm2, tumor_percentage_brain, tumor_percentage_image,
                    estimated_brain_pixel_count, rule_based_severity, severity_rule_description, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (501, 101, "No Tumor", 0.99, 0.0, 0.0, 0.0, 0.99, 0, 0.0, 0.0, 0.0, 24000, "LOW", "No active tumor mass", datetime.datetime.utcnow().isoformat()))

            # Establish assignments: Alice assigned to Patient A, Bob to Patient B
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

        # Auth Tokens
        self.doc_a = self.user_repo.get_by_email("doctor_a@aurascan.ai")
        self.doc_b = self.user_repo.get_by_email("doctor_b@aurascan.ai")
        self.pat_a = self.user_repo.get_by_email("patient_a@aurascan.ai")
        self.pat_b = self.user_repo.get_by_email("patient_b@aurascan.ai")

        jwt_svc = JWTService()
        self.token_a = jwt_svc.create_access_token(self.doc_a.uuid, self.doc_a.id, self.doc_a.email, Role.DOCTOR)
        self.token_b = jwt_svc.create_access_token(self.doc_b.uuid, self.doc_b.id, self.doc_b.email, Role.DOCTOR)
        self.token_pat_a = jwt_svc.create_access_token(self.pat_a.uuid, self.pat_a.id, self.pat_a.email, Role.PATIENT)

        self.test_client_ctx = TestClient(app)
        self.client = self.test_client_ctx.__enter__()

    def tearDown(self):
        self.test_client_ctx.__exit__(None, None, None)
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def test_authorized_doctor_review_asis(self):
        """Test that an authorized doctor can submit and retrieve an APPROVED_AS_IS review."""
        headers = {"Authorization": f"Bearer {self.token_a}"}
        payload = {
            "patient_id": "pat-uuid-a",
            "review_status": "APPROVED_AS_IS",
            "comment": "Perfect AI segmentation overlay output."
        }

        # Submit
        resp = self.client.post("/api/doctor/scans/100/reviews", json=payload, headers=headers)
        self.assertEqual(resp.status_code, 200, resp.json())
        data = resp.json()
        self.assertEqual(data["review_status"], "APPROVED_AS_IS")
        self.assertEqual(data["comment"], "Perfect AI segmentation overlay output.")

        # Get
        resp_get = self.client.get("/api/doctor/scans/100/reviews", headers=headers)
        self.assertEqual(resp_get.status_code, 200)
        data_get = resp_get.json()
        self.assertEqual(data_get["review_status"], "APPROVED_AS_IS")
        self.assertEqual(data_get["reviewer_name"], "Dr. Alice")
        self.assertEqual(data_get["comment"], "Perfect AI segmentation overlay output.")

    def test_unauthorized_doctor_review_blocked(self):
        """Test that Doctor B (unassigned) cannot review or view reviews for Patient A's scan (100)."""
        headers = {"Authorization": f"Bearer {self.token_b}"}
        payload = {
            "patient_id": "pat-uuid-a",
            "review_status": "APPROVED_AS_IS",
            "comment": "Unauthorized review attempt"
        }

        # POST blocked
        resp = self.client.post("/api/doctor/scans/100/reviews", json=payload, headers=headers)
        self.assertEqual(resp.status_code, 403)

        # GET blocked
        resp_get = self.client.get("/api/doctor/scans/100/reviews", headers=headers)
        self.assertEqual(resp_get.status_code, 403)

    def test_patient_access_denied(self):
        """Test that a patient cannot submit reviews or fetch review data."""
        headers = {"Authorization": f"Bearer {self.token_pat_a}"}
        payload = {
            "patient_id": "pat-uuid-a",
            "review_status": "APPROVED_AS_IS",
            "comment": "Patient trying to self-approve"
        }

        # POST blocked
        resp = self.client.post("/api/doctor/scans/100/reviews", json=payload, headers=headers)
        self.assertEqual(resp.status_code, 403)

        # GET blocked
        resp_get = self.client.get("/api/doctor/scans/100/reviews", headers=headers)
        self.assertEqual(resp_get.status_code, 403)

    def test_idor_wrong_scan_patient_match(self):
        """Test IDOR check: returns error when scan and patient_id mismatch."""
        # Establish assignment of Doctor A to Patient B, so Doctor A has authorization to both.
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO doctor_patient_assignments (doctor_id, patient_id, created_at) VALUES (?, ?, ?);",
                (10, "pat-uuid-b", datetime.datetime.utcnow().isoformat())
            )
            conn.commit()
        finally:
            conn.close()

        headers = {"Authorization": f"Bearer {self.token_a}"}
        payload = {
            "patient_id": "pat-uuid-b", # Scan 100 belongs to pat-uuid-a, not pat-uuid-b
            "review_status": "APPROVED_AS_IS",
            "comment": "IDOR attempt"
        }

        resp = self.client.post("/api/doctor/scans/100/reviews", json=payload, headers=headers)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("Mismatched patient and scan", resp.json()["detail"])

    def test_status_approved_with_corrections_gate(self):
        """Test that APPROVED_WITH_CORRECTIONS requires active annotations to be present."""
        headers = {"Authorization": f"Bearer {self.token_a}"}
        payload = {
            "patient_id": "pat-uuid-a",
            "review_status": "APPROVED_WITH_CORRECTIONS",
            "comment": "Approve with corrections but missing actual corrections"
        }

        # Fails initially because no annotations are saved
        resp = self.client.post("/api/doctor/scans/100/reviews", json=payload, headers=headers)
        self.assertEqual(resp.status_code, 400)
        self.assertIn("no active point or rectangle annotations were found", resp.json()["detail"])

        # Add a point annotation
        point_payload = {
            "x": 0.5,
            "y": 0.5,
            "label": "Correction point",
            "comment": "Observed tumor boundary shift"
        }
        resp_pt = self.client.post("/api/doctor/scans/100/point-annotations", json=point_payload, headers=headers)
        self.assertEqual(resp_pt.status_code, 200)

        # Now succeeds
        resp_retry = self.client.post("/api/doctor/scans/100/reviews", json=payload, headers=headers)
        self.assertEqual(resp_retry.status_code, 200)
        self.assertEqual(resp_retry.json()["review_status"], "APPROVED_WITH_CORRECTIONS")

    def test_original_ai_prediction_immutable(self):
        """Test that submitting a review preserves the original AI predictions in the database."""
        headers = {"Authorization": f"Bearer {self.token_a}"}

        # Verify initial prediction metrics
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        pred_before = conn.execute("SELECT * FROM predictions WHERE id = 500;").fetchone()
        conn.close()

        payload = {
            "patient_id": "pat-uuid-a",
            "review_status": "REJECTED",
            "comment": "Incorrect segmentation contour"
        }
        resp = self.client.post("/api/doctor/scans/100/reviews", json=payload, headers=headers)
        self.assertEqual(resp.status_code, 200)

        # Verify prediction is completely untouched
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        pred_after = conn.execute("SELECT * FROM predictions WHERE id = 500;").fetchone()
        conn.close()

        for col in pred_before.keys():
            self.assertEqual(pred_before[col], pred_after[col], f"Column {col} was modified!")

    def test_encryption_at_rest_and_audit_logging(self):
        """Verify that comments are encrypted at rest in DB and review actions log security audits."""
        headers = {"Authorization": f"Bearer {self.token_a}"}
        payload = {
            "patient_id": "pat-uuid-a",
            "review_status": "APPROVED_AS_IS",
            "comment": "Clinician sensitive notes about glioma tumor size"
        }
        resp = self.client.post("/api/doctor/scans/100/reviews", json=payload, headers=headers)
        self.assertEqual(resp.status_code, 200)

        # Check DB directly for encryption
        conn = sqlite3.connect(self.db_path)
        row = conn.execute("SELECT encrypted_comment FROM segmentation_reviews WHERE scan_id = 100;").fetchone()
        encrypted_val = row[0]

        self.assertTrue(encrypted_val.startswith("enc:v1:"), "Comment is not version encrypted in DB!")
        self.assertNotEqual(encrypted_val, "Clinician sensitive notes about glioma tumor size")

        # Verify security audit log exists
        logs = conn.execute("SELECT event_type, details, status FROM security_audit_logs;").fetchall()
        conn.close()

        event_types = [l[0] for l in logs]
        self.assertIn("SEGMENTATION_REVIEW_SUBMITTED", event_types)

    def test_review_immutable_no_delete(self):
        """Test that triggers prevent clinician reviews from being deleted in SQLite."""
        headers = {"Authorization": f"Bearer {self.token_a}"}
        payload = {
            "patient_id": "pat-uuid-a",
            "review_status": "APPROVED_AS_IS",
            "comment": "Final audit check"
        }
        resp = self.client.post("/api/doctor/scans/100/reviews", json=payload, headers=headers)
        self.assertEqual(resp.status_code, 200)

        # Attempt SQLite DELETE
        conn = sqlite3.connect(self.db_path)
        with self.assertRaises(sqlite3.IntegrityError):
            conn.execute("DELETE FROM segmentation_reviews WHERE scan_id = 100;")
        conn.close()

    def test_point_and_rectangle_annotation_regression(self):
        """Ensure that point and rectangle CRUD operations function correctly without regressions."""
        headers = {"Authorization": f"Bearer {self.token_a}"}

        # Test point annotations
        pt_payload = {"x": 0.1, "y": 0.2, "label": "Point A", "comment": "Point comment"}
        resp_pt = self.client.post("/api/doctor/scans/100/point-annotations", json=pt_payload, headers=headers)
        self.assertEqual(resp_pt.status_code, 200)
        pt_id = resp_pt.json()["annotation_id"]

        # Test rectangle annotations
        rect_payload = {"x": 0.3, "y": 0.3, "width": 0.2, "height": 0.2, "label": "Rect A", "comment": "Rect comment"}
        resp_rect = self.client.post("/api/doctor/scans/100/rectangle-annotations", json=rect_payload, headers=headers)
        self.assertEqual(resp_rect.status_code, 200)
        rect_id = resp_rect.json()["annotation_id"]

        # Verify GET points & rectangles
        resp_pts_get = self.client.get("/api/doctor/scans/100/point-annotations", headers=headers)
        self.assertEqual(len(resp_pts_get.json()), 1)
        self.assertEqual(resp_pts_get.json()[0]["comment"], "Point comment")

        resp_rects_get = self.client.get("/api/doctor/scans/100/rectangle-annotations", headers=headers)
        self.assertEqual(len(resp_rects_get.json()), 1)
        self.assertEqual(resp_rects_get.json()[0]["comment"], "Rect comment")
