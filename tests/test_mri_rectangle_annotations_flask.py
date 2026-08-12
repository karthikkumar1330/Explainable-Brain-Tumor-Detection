import os
import sqlite3
import datetime
import math
import unittest
from flask import json
from dashboard.infrastructure.web_server import create_app
from security.domain.entities import Role, User
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.jwt_service import JWTService
from persistence.infrastructure.repository import SQLitePersistenceRepository

class TestMriRectangleAnnotationsFlask(unittest.TestCase):
    """Focused integration tests for Flask rectangle annotation endpoints (Phase H3.2-C)."""

    def setUp(self):
        self.db_path = os.path.abspath("outputs/test_mri_rect_flask.db")
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

        # Configure Flask App
        self.app = create_app(db_path=self.db_path)
        self.app.config["TESTING"] = True
        self.app.config["DISABLE_CSRF"] = True
        self.client = self.app.test_client()

        # Initialize DB
        self.persistence_repo = SQLitePersistenceRepository(db_path=self.db_path)
        self.persistence_repo.initialize_db()

        self.user_repo = SQLiteUserRepository(db_path=self.db_path)
        self.user_repo.initialize_security_tables()

        # Drop auto triggers if they exist
        conn_drop = sqlite3.connect(self.db_path)
        try:
            conn_drop.execute("DROP TRIGGER IF EXISTS auto_assign_doctor_on_insert;")
            conn_drop.execute("DROP TRIGGER IF EXISTS auto_assign_patient_on_insert;")
            conn_drop.execute("DELETE FROM doctor_patient_assignments;")
            conn_drop.commit()
        finally:
            conn_drop.close()

        # Bootstrap admin
        self.admin_user = self.user_repo.bootstrap_admin()

        from security.infrastructure.password import PasswordHasher
        pass_hash = PasswordHasher.hash_password("Password@123")

        conn = sqlite3.connect(self.db_path)
        try:
            # Doctor A (ID: 10), Doctor B (ID: 11), Doctor Zero (ID: 12), Patient A (ID: 20), Patient B (ID: 21)
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

    def tearDown(self):
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def _get_headers(self, token):
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

    def test_01_authorized_doctor_can_create_rectangle(self):
        """Test Flask Route: Authorized Doctor A can create a rectangle annotation on Scan A."""
        headers = self._get_headers(self.token_doc_a)
        payload = {
            "x": 0.2,
            "y": 0.3,
            "width": 0.4,
            "height": 0.25,
            "label": "Tumor ROI",
            "comment": "Suspicious region."
        }
        resp = self.client.post("/api/doctor/scans/100/rectangle-annotations", json=payload, headers=headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["scan_id"], 100)
        self.assertEqual(data["x"], 0.2)
        self.assertEqual(data["width"], 0.4)
        self.assertEqual(data["label"], "Tumor ROI")
        self.assertEqual(data["comment"], "Suspicious region.")

    def test_02_unauthorized_doctor_cannot_create_rectangle(self):
        """Test Flask Route: Doctor A cannot create annotation on Scan B (no doctor-patient assignment)."""
        headers = self._get_headers(self.token_doc_a)
        payload = {
            "x": 0.2,
            "y": 0.3,
            "width": 0.4,
            "height": 0.25,
            "label": "Test"
        }
        resp = self.client.post("/api/doctor/scans/101/rectangle-annotations", json=payload, headers=headers)
        self.assertEqual(resp.status_code, 403)
        self.assertIn("Access denied", resp.get_json()["error"])

    def test_03_patient_cannot_create_rectangle(self):
        """Test Flask Route: Patient role is unauthorized (403)."""
        headers = self._get_headers(self.token_pat_a)
        payload = {
            "x": 0.2,
            "y": 0.3,
            "width": 0.4,
            "height": 0.25,
            "label": "Test"
        }
        resp = self.client.post("/api/doctor/scans/100/rectangle-annotations", json=payload, headers=headers)
        self.assertEqual(resp.status_code, 403)

    def test_04_nonexistent_scan_returns_404(self):
        """Test Flask Route: Nonexistent scan ID returns 404."""
        headers = self._get_headers(self.token_doc_a)
        payload = {
            "x": 0.2,
            "y": 0.3,
            "width": 0.4,
            "height": 0.25,
            "label": "Test"
        }
        resp = self.client.post("/api/doctor/scans/9999/rectangle-annotations", json=payload, headers=headers)
        self.assertEqual(resp.status_code, 404)

    def test_05_invalid_coordinates_rejected(self):
        """Test Flask Route: Invalid coordinates (out of bounds, NaN, negative) return 400."""
        headers = self._get_headers(self.token_doc_a)

        # x out of bounds (> 1.0)
        resp1 = self.client.post("/api/doctor/scans/100/rectangle-annotations", json={"x": 1.1, "y": 0.3, "width": 0.2, "height": 0.2}, headers=headers)
        self.assertEqual(resp1.status_code, 400)

        # negative width
        resp2 = self.client.post("/api/doctor/scans/100/rectangle-annotations", json={"x": 0.1, "y": 0.3, "width": -0.2, "height": 0.2}, headers=headers)
        self.assertEqual(resp2.status_code, 400)

        # width + x exceeds 1.0
        resp3 = self.client.post("/api/doctor/scans/100/rectangle-annotations", json={"x": 0.9, "y": 0.3, "width": 0.2, "height": 0.2}, headers=headers)
        self.assertEqual(resp3.status_code, 400)

    def test_06_invalid_label_length_rejected(self):
        """Test Flask Route: Label exceeding 100 characters returns 400."""
        headers = self._get_headers(self.token_doc_a)
        payload = {
            "x": 0.2,
            "y": 0.3,
            "width": 0.4,
            "height": 0.25,
            "label": "a" * 101
        }
        resp = self.client.post("/api/doctor/scans/100/rectangle-annotations", json=payload, headers=headers)
        self.assertEqual(resp.status_code, 400)

    def test_07_invalid_comment_length_rejected(self):
        """Test Flask Route: Comment exceeding 2000 characters returns 400."""
        headers = self._get_headers(self.token_doc_a)
        payload = {
            "x": 0.2,
            "y": 0.3,
            "width": 0.4,
            "height": 0.25,
            "label": "Test",
            "comment": "a" * 2001
        }
        resp = self.client.post("/api/doctor/scans/100/rectangle-annotations", json=payload, headers=headers)
        self.assertEqual(resp.status_code, 400)

    def test_08_rectangle_persisted_and_encrypted_at_rest(self):
        """Test Flask Route: Comment is encrypted in DB and decrypted upon retrieval."""
        headers = self._get_headers(self.token_doc_a)
        payload = {
            "x": 0.2,
            "y": 0.3,
            "width": 0.4,
            "height": 0.25,
            "label": "Encrypted ROI",
            "comment": "Secret diagnostic note."
        }
        resp = self.client.post("/api/doctor/scans/100/rectangle-annotations", json=payload, headers=headers)
        self.assertEqual(resp.status_code, 200)
        ann_id = resp.get_json()["annotation_id"]

        # Check DB directly - must be encrypted
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute("SELECT encrypted_comment FROM mri_rectangle_annotations WHERE annotation_id = ?;", (ann_id,)).fetchone()
            db_val = row[0]
            self.assertNotEqual(db_val, "Secret diagnostic note.")
            self.assertTrue(db_val.startswith("enc:v1:"))
        finally:
            conn.close()

        # Check authorized GET retrieval - must be decrypted
        resp_get = self.client.get("/api/doctor/scans/100/rectangle-annotations", headers=headers)
        self.assertEqual(resp_get.status_code, 200)
        self.assertEqual(len(resp_get.get_json()), 1)
        self.assertEqual(resp_get.get_json()[0]["comment"], "Secret diagnostic note.")

    def test_09_unauthorized_retrieval_is_blocked(self):
        """Test Flask Route: Unauthorized doctor cannot GET annotations on Scan A."""
        headers_a = self._get_headers(self.token_doc_a)
        payload = {"x": 0.2, "y": 0.3, "width": 0.4, "height": 0.25, "label": "L"}
        resp_post = self.client.post("/api/doctor/scans/100/rectangle-annotations", json=payload, headers=headers_a)
        self.assertEqual(resp_post.status_code, 200)

        # Doc B attempts to fetch Scan A annotations
        headers_b = self._get_headers(self.token_doc_b)
        resp_get = self.client.get("/api/doctor/scans/100/rectangle-annotations", headers=headers_b)
        self.assertEqual(resp_get.status_code, 403)

    def test_10_update_and_delete_operations(self):
        """Test Flask Route: PUT and DELETE operations on rectangle annotations."""
        headers = self._get_headers(self.token_doc_a)
        payload = {"x": 0.2, "y": 0.3, "width": 0.4, "height": 0.25, "label": "Orig", "comment": "Orig comment"}
        resp_post = self.client.post("/api/doctor/scans/100/rectangle-annotations", json=payload, headers=headers)
        self.assertEqual(resp_post.status_code, 200)
        ann_id = resp_post.get_json()["annotation_id"]

        # Update
        payload_update = {"x": 0.25, "y": 0.35, "width": 0.4, "height": 0.25, "label": "New", "comment": "New comment"}
        resp_put = self.client.put(f"/api/doctor/rectangle-annotations/{ann_id}", json=payload_update, headers=headers)
        self.assertEqual(resp_put.status_code, 200)
        self.assertEqual(resp_put.get_json()["label"], "New")

        # Archive (Delete)
        resp_del = self.client.delete(f"/api/doctor/rectangle-annotations/{ann_id}", headers=headers)
        self.assertEqual(resp_del.status_code, 200)

        # GET should be empty
        resp_get = self.client.get("/api/doctor/scans/100/rectangle-annotations", headers=headers)
        self.assertEqual(len(resp_get.get_json()), 0)

    def test_11_route_registration(self):
        """Test Flask Route Registration: Prove the route is present on the Flask app."""
        routes = [rule.rule for rule in self.app.url_map.iter_rules()]
        expected_route = "/api/doctor/scans/<int:scan_id>/rectangle-annotations"
        self.assertIn(expected_route, routes, f"Route {expected_route} is missing from the Flask app's registered routes!")

if __name__ == "__main__":
    unittest.main()
