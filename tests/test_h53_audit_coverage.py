import os
os.environ["DB_PATH"] = os.path.abspath("outputs/test_h53_audit_coverage.db")

import unittest
import sqlite3
import datetime
import cv2
import numpy as np
from fastapi.testclient import TestClient
from run_api import app
from security.domain.entities import Role, User
from security.infrastructure.jwt_service import JWTService
from security.infrastructure.repository import SQLiteUserRepository
from persistence.infrastructure.repository import SQLitePersistenceRepository

class TestH53AuditCoverage(unittest.TestCase):
    """Phase H5.3: Audit Event Coverage focused tests."""

    def setUp(self):
        self.db_path = os.environ.get("DB_PATH", "outputs/test_h53_audit_coverage.db")
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

        # Doctor
        self.doctor = User(
            id=None,
            uuid="doc-uuid-aaaa",
            email="doctora@aurascan.ai",
            password_hash="fakehash",
            full_name="Dr. House",
            role=Role.DOCTOR,
            is_verified=True,
            is_active=True,
            created_at=datetime.datetime.utcnow().isoformat(),
            updated_at=datetime.datetime.utcnow().isoformat()
        )
        self.doctor = self.user_repo.create_user(self.doctor)

        # Patient
        self.patient = User(
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
        self.patient = self.user_repo.create_user(self.patient)

        # JWT Tokens
        jwt_svc = JWTService()
        self.doc_token = jwt_svc.create_access_token(self.doctor.uuid, self.doctor.id, self.doctor.email, self.doctor.role)
        self.pat_token = jwt_svc.create_access_token(self.patient.uuid, self.patient.id, self.patient.email, self.patient.role)

        self.doc_headers = {"Authorization": f"Bearer {self.doc_token}"}
        self.pat_headers = {"Authorization": f"Bearer {self.pat_token}"}

        self.client = TestClient(app)

    def _create_mock_brain_mri_slice(self) -> bytes:
        """Generates a mock brain MRI slice programmatically (centered ellipse with tissue texture)."""
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.ellipse(mask, (128, 128), (60, 80), 0, 0, 360, 255, -1)

        np.random.seed(42)
        half_width = 128
        left_texture = np.random.randint(80, 200, size=(256, half_width), dtype=np.uint8)
        right_texture = np.fliplr(left_texture)
        texture = np.hstack([left_texture, right_texture])

        brain_tissue = cv2.bitwise_and(texture, texture, mask=mask)
        brain_tissue_blurred = cv2.GaussianBlur(brain_tissue, (5, 5), 0)
        brain_tissue = np.where(mask > 0, brain_tissue_blurred, 0)

        img[:, :, 0] = brain_tissue
        img[:, :, 1] = brain_tissue
        img[:, :, 2] = brain_tissue

        cv2.ellipse(img, (128, 110), (15, 8), 0, 0, 360, (40, 40, 40), -1)
        cv2.ellipse(img, (128, 140), (12, 6), 0, 0, 360, (40, 40, 40), -1)

        _, buf = cv2.imencode(".png", img)
        return buf.tobytes()

    def _create_mock_non_brain_image(self) -> bytes:
        """Generates a non-brain natural-like photo (solid white canvas with text)."""
        img = np.ones((256, 256, 3), dtype=np.uint8) * 240
        cv2.putText(img, "MEDICAL INC", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        _, buf = cv2.imencode(".png", img)
        return buf.tobytes()

    def test_01_successful_mri_upload_generates_audit_event(self):
        """1. Successful MRI upload generates the intended audit event."""
        file_bytes = self._create_mock_brain_mri_slice()
        files = {"file": ("mock_mri.png", file_bytes, "image/png")}

        resp = self.client.post("/api/upload", files=files, headers=self.doc_headers)
        self.assertEqual(resp.status_code, 200)

        # Verify the success audit event
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute("SELECT * FROM security_audit_logs WHERE event_type = 'MRI_UPLOAD' AND status = 'SUCCESS';").fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["user_id"], self.doctor.id)
            self.assertEqual(row["email"], self.doctor.email)
            self.assertIn("mock_mri.png", row["details"])
            # Assert sensitive local path is not logged
            self.assertNotIn("outputs/temp_uploads", row["details"])
        finally:
            conn.close()

    def test_02_failed_mri_validation_generates_failure_event(self):
        """2. Failed MRI validation generates an appropriate failure event."""
        # Non-brain image triggers validation failure
        file_bytes = self._create_mock_non_brain_image()
        files = {"file": ("invalid_mri.png", file_bytes, "image/png")}

        resp = self.client.post("/api/upload", files=files, headers=self.doc_headers)
        self.assertEqual(resp.status_code, 400)

        # Verify the failure audit event
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute("SELECT * FROM security_audit_logs WHERE event_type = 'MRI_UPLOAD' AND status = 'FAILED';").fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["user_id"], self.doctor.id)
            self.assertEqual(row["email"], self.doctor.email)
            self.assertIn("invalid_mri.png", row["details"])
            self.assertIn("validation failed", row["details"].lower())
            # Assert sensitive local path is not logged
            self.assertNotIn("outputs/temp_uploads", row["details"])
        finally:
            conn.close()

    def test_03_unauthorized_upload_does_not_log_success(self):
        """3. Unauthorized upload/access does not create misleading success events."""
        file_bytes = self._create_mock_brain_mri_slice()
        files = {"file": ("unauth_mri.png", file_bytes, "image/png")}

        # Call with Patient headers (Patients are not authorized to upload)
        resp = self.client.post("/api/upload", files=files, headers=self.pat_headers)
        self.assertEqual(resp.status_code, 403)

        # Verify no success audit event exists for this file
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute("SELECT * FROM security_audit_logs WHERE event_type = 'MRI_UPLOAD' AND details LIKE '%unauth_mri.png%';").fetchone()
            self.assertIsNone(row)
        finally:
            conn.close()
