import os
os.environ["DB_PATH"] = os.path.abspath("outputs/test_h54_telemetry.db")

import unittest
import sqlite3
import datetime
import cv2
import numpy as np
from fastapi.testclient import TestClient as FastAPITestClient
from flask import g
from run_api import app as fastapi_app
from dashboard.infrastructure.web_server import create_app
from security.domain.entities import Role, User
from security.infrastructure.jwt_service import JWTService
from security.infrastructure.repository import SQLiteUserRepository
from persistence.infrastructure.repository import SQLitePersistenceRepository
from clinical_reporting.application.services import ReportService

class TestH54Telemetry(unittest.TestCase):
    """Phase H5.4: Request Telemetry focused tests."""

    def setUp(self):
        self.db_path = os.environ.get("DB_PATH", "outputs/test_h54_telemetry.db")
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

        # Override default DB paths in API and Flask
        import api.infrastructure.routes as api_routes
        import api.routes.auth_routes as auth_routes
        api_routes.DEFAULT_DB_PATH = self.db_path
        auth_routes.DEFAULT_DB_PATH = self.db_path

        self.flask_app = create_app(self.db_path)
        self.flask_app.config["DATABASE_PATH"] = self.db_path
        self.flask_app.config["TESTING"] = True
        self.flask_app.config["DISABLE_CSRF"] = False

        # Initialize databases
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

        # Setup users
        self.admin = self.user_repo.bootstrap_admin()

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

        jwt_svc = JWTService()
        self.doc_token = jwt_svc.create_access_token(self.doctor.uuid, self.doctor.id, self.doctor.email, self.doctor.role)

        self.fastapi_client = FastAPITestClient(fastapi_app)
        self.flask_client = self.flask_app.test_client()

    def _create_mock_brain_mri_slice(self) -> bytes:
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

    def test_01_flask_propagates_request_telemetry(self):
        """1. Flask endpoint propagates client IP and User-Agent to CSRF audit event."""
        self.flask_app.config["TESTING"] = False  # Enable CSRF middleware checks
        self.flask_client.set_cookie("csrf_token", "matching_token")

        # Call with mismatched header value -> triggers CSRF_ATTEMPT log
        headers = {
            "X-CSRF-Token": "mismatched_token",
            "User-Agent": "Flask-Test-Agent"
        }
        resp = self.flask_client.post("/api/reports/export", json={"dummy": "data"}, headers=headers)
        self.assertEqual(resp.status_code, 403)

        # Verify the logged CSRF attempt details and telemetry
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute("SELECT * FROM security_audit_logs WHERE event_type = 'CSRF_ATTEMPT';").fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["user_agent"], "Flask-Test-Agent")
        finally:
            conn.close()

    def test_02_fastapi_propagates_request_telemetry(self):
        """2. FastAPI endpoint propagates client IP and User-Agent to upload audit event."""
        file_bytes = self._create_mock_brain_mri_slice()
        files = {"file": ("mock_mri.png", file_bytes, "image/png")}
        headers = {
            "Authorization": f"Bearer {self.doc_token}",
            "User-Agent": "FastAPI-Test-Agent"
        }

        resp = self.fastapi_client.post("/api/upload", files=files, headers=headers)
        self.assertEqual(resp.status_code, 200)

        # Verify the success audit event
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute("SELECT * FROM security_audit_logs WHERE event_type = 'MRI_UPLOAD';").fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["user_agent"], "FastAPI-Test-Agent")
        finally:
            conn.close()

    def test_03_background_job_falls_back_safely(self):
        """3. Logging in a background job context falls back safely to Loopback/System."""
        # Call logging directly on ReportService (which runs without HTTP request context)
        service = ReportService(db_path=self.db_path)

        conn = sqlite3.connect(self.db_path)
        try:
            # Trigger report view check which writes security event
            service._log_security_audit_event(conn, "REPORT_VIEWED", self.doctor, 999, "SUCCESS", "View report metadata")
            conn.commit()
        finally:
            conn.close()

        # Check database logs
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute("SELECT * FROM security_audit_logs WHERE event_type = 'REPORT_VIEWED';").fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["ip_address"], "127.0.0.1")
            self.assertEqual(row["user_agent"], "System")
        finally:
            conn.close()

    def test_04_proxy_ip_handling_based_on_trust(self):
        """4. Verify that proxy header X-Forwarded-For is parsed based on TRUST_PROXY value."""
        file_bytes = self._create_mock_brain_mri_slice()

        # Test case A: TRUST_PROXY is enabled
        os.environ["TRUST_PROXY"] = "1"
        files = {"file": ("mri_a.png", file_bytes, "image/png")}
        headers = {
            "Authorization": f"Bearer {self.doc_token}",
            "X-Forwarded-For": "203.0.113.195, 198.51.100.1",
            "User-Agent": "FastAPI-Test-Agent"
        }
        resp = self.fastapi_client.post("/api/upload", files=files, headers=headers)
        self.assertEqual(resp.status_code, 200)

        # Test case B: TRUST_PROXY is disabled
        os.environ["TRUST_PROXY"] = "0"
        files = {"file": ("mri_b.png", file_bytes, "image/png")}
        headers = {
            "Authorization": f"Bearer {self.doc_token}",
            "X-Forwarded-For": "203.0.113.195, 198.51.100.1",
            "User-Agent": "FastAPI-Test-Agent"
        }
        resp = self.fastapi_client.post("/api/upload", files=files, headers=headers)
        self.assertEqual(resp.status_code, 200)

        # Verify database logs
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row_a = conn.execute("SELECT * FROM security_audit_logs WHERE details LIKE '%mri_a.png%';").fetchone()
            self.assertIsNotNone(row_a)
            self.assertEqual(row_a["ip_address"], "203.0.113.195")

            row_b = conn.execute("SELECT * FROM security_audit_logs WHERE details LIKE '%mri_b.png%';").fetchone()
            self.assertIsNotNone(row_b)
            self.assertNotEqual(row_b["ip_address"], "203.0.113.195")
            self.assertEqual(row_b["ip_address"], "testclient")  # FastAPITestClient default host
        finally:
            conn.close()

        # Clean environment
        if "TRUST_PROXY" in os.environ:
            del os.environ["TRUST_PROXY"]
