import os
os.environ["DB_PATH"] = os.path.abspath("outputs/test_h72a_telemetry.db")

import unittest
import sqlite3
import datetime
import time
from unittest.mock import patch
from fastapi.testclient import TestClient as FastAPITestClient
from run_api import app as fastapi_app
from dashboard.infrastructure.web_server import create_app
from security.domain.entities import Role, User
from security.infrastructure.jwt_service import JWTService
from security.infrastructure.repository import SQLiteUserRepository
from persistence.infrastructure.repository import SQLitePersistenceRepository

class TestH72ARequestTelemetry(unittest.TestCase):
    """Phase H7.2-A: Standardized HTTP Request Performance Telemetry tests."""

    def setUp(self):
        self.db_path = os.environ.get("DB_PATH", "outputs/test_h72a_telemetry.db")
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

        # Override default DB paths in API, Auth routes, and dashboard
        import api.infrastructure.routes as api_routes
        import api.routes.auth_routes as auth_routes
        api_routes.DEFAULT_DB_PATH = self.db_path
        auth_routes.DEFAULT_DB_PATH = self.db_path

        self.flask_app = create_app(self.db_path)
        self.flask_app.config["DATABASE_PATH"] = self.db_path
        self.flask_app.config["TESTING"] = True
        self.flask_app.config["DISABLE_CSRF"] = True  # Simplify calls for request testing

        self.persistence_repo = SQLitePersistenceRepository(db_path=self.db_path)
        self.persistence_repo.initialize_db()

        self.user_repo = SQLiteUserRepository(db_path=self.db_path)
        self.user_repo.initialize_security_tables()

        # Disable auto-assignment triggers
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DROP TRIGGER IF EXISTS auto_assign_doctor_on_insert;")
            conn.execute("DROP TRIGGER IF EXISTS auto_assign_patient_on_insert;")
            conn.execute("DELETE FROM doctor_patient_assignments;")
            conn.commit()
        finally:
            conn.close()

        # Create admin user
        self.admin = self.user_repo.bootstrap_admin()

        # Create doctor user
        self.doctor = User(
            id=None,
            uuid="doctor-test-uuid",
            email="doctor-test@aurascan.ai",
            password_hash="fakehash",
            full_name="Dr. Tester",
            role=Role.DOCTOR,
            is_verified=True,
            is_active=True,
            created_at=datetime.datetime.utcnow().isoformat(),
            updated_at=datetime.datetime.utcnow().isoformat()
        )
        self.doctor = self.user_repo.create_user(self.doctor)

        # Create patient user
        self.patient = User(
            id=None,
            uuid="patient-test-uuid",
            email="patient-test@aurascan.ai",
            password_hash="fakehash",
            full_name="Patient Jane",
            role=Role.PATIENT,
            is_verified=True,
            is_active=True,
            created_at=datetime.datetime.utcnow().isoformat(),
            updated_at=datetime.datetime.utcnow().isoformat()
        )
        self.patient = self.user_repo.create_user(self.patient)

        jwt_svc = JWTService()
        self.admin_token = jwt_svc.create_access_token(self.admin.uuid, self.admin.id, self.admin.email, self.admin.role)
        self.doc_token = jwt_svc.create_access_token(self.doctor.uuid, self.doctor.id, self.doctor.email, self.doctor.role)
        self.pat_token = jwt_svc.create_access_token(self.patient.uuid, self.patient.id, self.patient.email, self.patient.role)

        self.fastapi_client = FastAPITestClient(fastapi_app)
        self.flask_client = self.flask_app.test_client()

    def tearDown(self):
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def test_01_fastapi_request_records_timing_telemetry(self):
        """Verify FastAPI request produces timing telemetry on success/failure."""
        # Clean existing telemetry
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DELETE FROM http_request_telemetry;")
            conn.commit()
        finally:
            conn.close()

        # Trigger a FastAPI call that gets captured (e.g. invalid endpoint or login attempt)
        headers = {"Authorization": f"Bearer {self.doc_token}"}
        resp = self.fastapi_client.get("/api/reports/audit-history", headers=headers)
        self.assertIn(resp.status_code, [200, 401, 403, 404])

        # Verify telemetry entry in DB
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute("SELECT * FROM http_request_telemetry;").fetchall()
            self.assertTrue(len(rows) > 0)
            row = rows[0]
            self.assertIsNotNone(row["timestamp"])
            self.assertGreaterEqual(row["duration_ms"], 0.0)
            self.assertEqual(row["method"], "GET")
            self.assertIn("/api/reports/audit-history", row["route"])
            self.assertEqual(row["status_code"], resp.status_code)
        finally:
            conn.close()

    def test_02_flask_request_records_timing_telemetry(self):
        """Verify Flask request produces timing telemetry on success/failure."""
        # Clean existing telemetry
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DELETE FROM http_request_telemetry;")
            conn.commit()
        finally:
            conn.close()

        # Trigger a Flask call
        headers = {"Authorization": f"Bearer {self.doc_token}"}
        resp = self.flask_client.get("/api/history", headers=headers)
        self.assertIn(resp.status_code, [200, 401, 403, 404])

        # Verify telemetry in DB
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute("SELECT * FROM http_request_telemetry;").fetchall()
            self.assertTrue(len(rows) > 0)
            row = rows[0]
            self.assertIsNotNone(row["timestamp"])
            self.assertGreaterEqual(row["duration_ms"], 0.0)
            self.assertEqual(row["method"], "GET")
            self.assertEqual(row["route"], "/api/history")
            self.assertEqual(row["status_code"], resp.status_code)
        finally:
            conn.close()

    def test_03_route_normalization_fastapi(self):
        """Verify FastAPI route parameters are properly normalized in telemetry."""
        # Clean existing telemetry
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DELETE FROM http_request_telemetry;")
            conn.commit()
        finally:
            conn.close()

        headers = {"Authorization": f"Bearer {self.doc_token}"}
        # Request a route that has route path parameters, e.g. /api/reports/{report_id}/versions
        # We query /api/reports/99/versions -> should normalize to /api/reports/<report_id>/versions
        resp = self.fastapi_client.get("/api/reports/99/versions", headers=headers)

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute("SELECT * FROM http_request_telemetry WHERE route LIKE '%/versions';").fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["route"], "/api/reports/<report_id>/versions")
        finally:
            conn.close()

    def test_04_route_normalization_flask(self):
        """Verify Flask route parameters are properly normalized in telemetry."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DELETE FROM http_request_telemetry;")
            conn.commit()
        finally:
            conn.close()

        headers = {"Authorization": f"Bearer {self.doc_token}"}
        # /api/report/<int:report_id>/compare/<int:other_report_id>
        resp = self.flask_client.get("/api/report/123/compare/456", headers=headers)

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute("SELECT * FROM http_request_telemetry WHERE route LIKE '%compare%';").fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["route"], "/api/report/<report_id>/compare/<other_report_id>")
        finally:
            conn.close()

    def test_05_failed_and_error_requests(self):
        """Verify failed/4xx/5xx requests are timed and stored safely."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DELETE FROM http_request_telemetry;")
            conn.commit()
        finally:
            conn.close()

        # FastAPI trigger invalid route (404)
        self.fastapi_client.get("/api/does-not-exist")
        # Flask trigger invalid route (404)
        self.flask_client.get("/api/does-not-exist")

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute("SELECT * FROM http_request_telemetry WHERE status_code = 404;").fetchall()
            self.assertTrue(len(rows) >= 2)
        finally:
            conn.close()

    def test_06_telemetry_recording_failure_does_not_break_request(self):
        """Verify that database telemetry write failures do not affect API response success."""
        # Mock save_http_request_telemetry to raise error
        with patch.object(SQLitePersistenceRepository, "save_http_request_telemetry", side_effect=Exception("DB connection error")):
            headers = {"Authorization": f"Bearer {self.doc_token}"}
            # FastAPI
            resp_fast = self.fastapi_client.get("/api/reports/audit-history", headers=headers)
            self.assertIn(resp_fast.status_code, [200, 401, 403])

            # Flask
            resp_flask = self.flask_client.get("/api/history", headers=headers)
            self.assertIn(resp_flask.status_code, [200, 401, 403])

    def test_07_no_credentials_or_pii_logged(self):
        """Verify Authorization headers, query params, cookies, and PII are NOT saved."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DELETE FROM http_request_telemetry;")
            conn.commit()
        finally:
            conn.close()

        headers = {
            "Authorization": "Bearer super_secret_jwt_token_12345",
            "Cookie": "session_id=abcdefg"
        }
        self.flask_client.get("/api/history?patient_id=123&patient_name=JaneDoe", headers=headers)

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute("SELECT * FROM http_request_telemetry;").fetchone()
            self.assertIsNotNone(row)
            # Ensure query parameters are not in the stored route
            self.assertEqual(row["route"], "/api/history")
            # Verify table columns only contain standard parameters
            keys = row.keys()
            for key in keys:
                self.assertIn(key, ["id", "timestamp", "duration_ms", "method", "route", "status_code"])
        finally:
            conn.close()

    def test_08_rbac_telemetry_access_enforced(self):
        """Verify Role-based Access Control limits telemetry endpoints to Admin role."""
        # 1. Admin must be allowed access
        headers = {"Authorization": f"Bearer {self.admin_token}"}
        resp_admin = self.flask_client.get("/api/health-telemetry", headers=headers)
        self.assertEqual(resp_admin.status_code, 200)
        data = resp_admin.get_json()
        self.assertIn("total_http_requests", data)
        self.assertIn("avg_http_latency_ms", data)

        # 2. Doctor must be denied access
        headers_doc = {"Authorization": f"Bearer {self.doc_token}"}
        resp_doc = self.flask_client.get("/api/health-telemetry", headers=headers_doc)
        self.assertEqual(resp_doc.status_code, 403)

        # 3. Patient must be denied access
        headers_pat = {"Authorization": f"Bearer {self.pat_token}"}
        resp_pat = self.flask_client.get("/api/health-telemetry", headers=headers_pat)
        self.assertEqual(resp_pat.status_code, 403)

    def test_09_health_endpoints_exempt_from_telemetry(self):
        """Verify health checks (/health, /ping) do not write telemetry records and remain safe."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DELETE FROM http_request_telemetry;")
            conn.commit()
        finally:
            conn.close()

        # Call health endpoints
        self.fastapi_client.get("/api/health")
        self.fastapi_client.get("/ping")
        self.flask_client.get("/health")
        self.flask_client.get("/ping")

        # Telemetry should be empty
        conn = sqlite3.connect(self.db_path)
        try:
            count = conn.execute("SELECT COUNT(*) FROM http_request_telemetry;").fetchone()[0]
            self.assertEqual(count, 0)
        finally:
            conn.close()

if __name__ == "__main__":
    unittest.main()
