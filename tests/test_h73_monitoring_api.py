import os
os.environ["DB_PATH"] = os.path.abspath("outputs/test_h73_monitoring.db")

import unittest
import sqlite3
import datetime
from fastapi.testclient import TestClient as FastAPITestClient
from run_api import app as fastapi_app
from dashboard.infrastructure.web_server import create_app
from security.domain.entities import Role, User
from security.infrastructure.jwt_service import JWTService
from security.infrastructure.repository import SQLiteUserRepository
from persistence.infrastructure.repository import SQLitePersistenceRepository

class TestH73MonitoringAPI(unittest.TestCase):
    """focused tests validating H7.3 monitoring metrics and trend aggregation API correctness & security."""

    def setUp(self):
        self.db_path = os.environ.get("DB_PATH", "outputs/test_h73_monitoring.db")
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

        # Override route variables to use our test DB
        import api.infrastructure.routes as api_routes
        import api.routes.auth_routes as auth_routes
        api_routes.DEFAULT_DB_PATH = self.db_path
        auth_routes.DEFAULT_DB_PATH = self.db_path

        self.flask_app = create_app(self.db_path)
        self.flask_app.config["DATABASE_PATH"] = self.db_path
        self.flask_app.config["TESTING"] = True
        self.flask_app.config["DISABLE_CSRF"] = True

        self.persistence_repo = SQLitePersistenceRepository(db_path=self.db_path)
        self.persistence_repo.initialize_db()

        self.user_repo = SQLiteUserRepository(db_path=self.db_path)
        self.user_repo.initialize_security_tables()

        # Seed test users
        self.admin = self.user_repo.bootstrap_admin(admin_email="admin@aurascan.ai", admin_pass="AdminPass@123")

        self.doctor = User(
            id=None,
            uuid="doc-uuid-73",
            email="doc73@aurascan.ai",
            password_hash="fakehash",
            full_name="Dr. Auditor",
            role=Role.DOCTOR,
            is_verified=True,
            is_active=True
        )
        self.user_repo.create_user(self.doctor)

        self.patient = User(
            id=None,
            uuid="pat-uuid-73",
            email="patient73@aurascan.ai",
            password_hash="fakehash",
            full_name="Patient Bob",
            role=Role.PATIENT,
            is_verified=True,
            is_active=True
        )
        self.user_repo.create_user(self.patient)

        # Generate tokens
        jwt_svc = JWTService()
        self.admin_token = jwt_svc.create_access_token(
            user_uuid=self.admin.uuid, user_id=self.admin.id, email=self.admin.email, role=self.admin.role
        )
        self.doc_token = jwt_svc.create_access_token(
            user_uuid=self.doctor.uuid, user_id=self.doctor.id, email=self.doctor.email, role=self.doctor.role
        )
        self.pat_token = jwt_svc.create_access_token(
            user_uuid=self.patient.uuid, user_id=self.patient.id, email=self.patient.email, role=self.patient.role
        )

        self.fastapi_client = FastAPITestClient(fastapi_app)
        self.flask_client = self.flask_app.test_client()

    def tearDown(self):
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def test_01_security_access_controls(self):
        """Verify Admin role has access to health telemetry and trends, while other roles are rejected."""
        headers_adm = {"Authorization": f"Bearer {self.admin_token}"}
        headers_doc = {"Authorization": f"Bearer {self.doc_token}"}
        headers_pat = {"Authorization": f"Bearer {self.pat_token}"}

        # 1. FastAPI Admin Success
        resp = self.fastapi_client.get("/api/health-telemetry", headers=headers_adm)
        self.assertEqual(resp.status_code, 200)

        resp_trends = self.fastapi_client.get("/api/admin/monitoring/trends", headers=headers_adm)
        self.assertEqual(resp_trends.status_code, 200)

        # 2. Flask Admin Success
        resp_flask = self.flask_client.get("/api/health-telemetry", headers=headers_adm)
        self.assertEqual(resp_flask.status_code, 200)

        resp_flask_trends = self.flask_client.get("/api/admin/monitoring/trends", headers=headers_adm)
        self.assertEqual(resp_flask_trends.status_code, 200)

        # 3. Doctor Rejected (403)
        self.assertEqual(self.fastapi_client.get("/api/health-telemetry", headers=headers_doc).status_code, 403)
        self.assertEqual(self.fastapi_client.get("/api/admin/monitoring/trends", headers=headers_doc).status_code, 403)
        self.assertEqual(self.flask_client.get("/api/health-telemetry", headers=headers_doc).status_code, 403)
        self.assertEqual(self.flask_client.get("/api/admin/monitoring/trends", headers=headers_doc).status_code, 403)

        # 4. Patient Rejected (403)
        self.assertEqual(self.fastapi_client.get("/api/health-telemetry", headers=headers_pat).status_code, 403)
        self.assertEqual(self.fastapi_client.get("/api/admin/monitoring/trends", headers=headers_pat).status_code, 403)
        self.assertEqual(self.flask_client.get("/api/health-telemetry", headers=headers_pat).status_code, 403)
        self.assertEqual(self.flask_client.get("/api/admin/monitoring/trends", headers=headers_pat).status_code, 403)

        # 5. Anonymous Rejected (401/403)
        self.assertIn(self.fastapi_client.get("/api/health-telemetry").status_code, [401, 403])
        self.assertIn(self.fastapi_client.get("/api/admin/monitoring/trends").status_code, [401, 403])
        self.assertIn(self.flask_client.get("/api/health-telemetry").status_code, [401, 403])
        self.assertIn(self.flask_client.get("/api/admin/monitoring/trends").status_code, [401, 403])

    def test_02_empty_telemetry_safe(self):
        """Assert metrics retrieve safely when telemetry tables are empty."""
        headers = {"Authorization": f"Bearer {self.admin_token}"}
        resp = self.fastapi_client.get("/api/health-telemetry", headers=headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        hist = data.get("historical_telemetry", {})
        self.assertEqual(hist.get("total_http_requests"), 0)
        self.assertEqual(hist.get("avg_http_latency_ms"), 0.0)
        self.assertEqual(hist.get("http_error_rate"), 0.0)
        self.assertEqual(hist.get("total_batches"), 0)
        self.assertEqual(hist.get("batch_error_rate"), 0.0)
        self.assertIsNone(hist.get("http_p50_latency_ms"))

    def test_03_http_and_batch_aggregates(self):
        """Assert counts, latencies, percentiles, and error-rates compute correctly."""
        # Seed telemetry rows
        conn = sqlite3.connect(self.db_path)
        try:
            # Seed 5 HTTP requests: 4 success, 1 error (status_code=500)
            conn.execute("""
                INSERT INTO http_request_telemetry (timestamp, duration_ms, method, route, status_code)
                VALUES
                    (?, 100.0, 'GET', '/api/users', 200),
                    (?, 200.0, 'GET', '/api/users', 200),
                    (?, 300.0, 'GET', '/api/users', 200),
                    (?, 400.0, 'GET', '/api/users', 200),
                    (?, 1000.0, 'POST', '/api/report', 500);
            """, [datetime.datetime.utcnow().isoformat()] * 5)

            # Seed 2 batches: batch 1 has 10 items (8 successful, 2 failed), batch 2 has 5 items (5 successful)
            # Total failed = 2, total items = 15. error rate = 2/15 = 0.1333
            conn.execute("""
                INSERT INTO batch_performance_telemetry (batch_id, timestamp, total_items, successful_items, failed_items, total_duration_ms, average_item_duration_ms)
                VALUES
                    ('b1', ?, 10, 8, 2, 5000.0, 500.0),
                    ('b2', ?, 5, 5, 0, 2000.0, 400.0);
            """, [datetime.datetime.utcnow().isoformat()] * 2)
            conn.commit()
        finally:
            conn.close()

        headers = {"Authorization": f"Bearer {self.admin_token}"}
        resp = self.fastapi_client.get("/api/health-telemetry", headers=headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.json().get("historical_telemetry", {})

        self.assertEqual(data.get("total_http_requests"), 5)
        # Average HTTP latency: (100+200+300+400+1000)/5 = 400.0
        self.assertAlmostEqual(data.get("avg_http_latency_ms"), 400.0)
        # HTTP error rate: 1/5 = 0.2
        self.assertAlmostEqual(data.get("http_error_rate"), 0.2)

        # Batch counts
        self.assertEqual(data.get("total_batches"), 2)
        # Average batch duration: (5000 + 2000)/2 = 3500.0
        self.assertAlmostEqual(data.get("avg_batch_latency_ms"), 3500.0)
        # Total batch items: 10 + 5 = 15
        self.assertEqual(data.get("total_batch_items"), 15)
        # Batch error rate: 2/15 = 0.13333333333333333
        self.assertAlmostEqual(data.get("batch_error_rate"), 2.0 / 15.0)

        # Percentiles
        # Latencies sorted: 100, 200, 300, 400, 1000.
        # P50 (median) = 300
        self.assertEqual(data.get("http_p50_latency_ms"), 300.0)

    def test_04_historical_aggregates_grouping(self):
        """Assert daily grouping aggregation operates correctly and is bounded."""
        now = datetime.datetime.utcnow()
        # Seed HTTP requests across three distinct dates dynamically relative to utcnow
        date1 = (now - datetime.timedelta(days=3)).strftime("%Y-%m-%dT10:00:00.000")
        date2 = (now - datetime.timedelta(days=2)).strftime("%Y-%m-%dT10:00:00.000")
        date3 = (now - datetime.timedelta(days=1)).strftime("%Y-%m-%dT10:00:00.000")

        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT INTO http_request_telemetry (timestamp, duration_ms, method, route, status_code)
                VALUES
                    (?, 100.0, 'GET', '/api/users', 200),
                    (?, 200.0, 'GET', '/api/users', 200),
                    (?, 300.0, 'GET', '/api/users', 200),
                    (?, 400.0, 'GET', '/api/users', 200);
            """, [date1, date1, date2, date3])
            conn.commit()
        finally:
            conn.close()

        headers = {"Authorization": f"Bearer {self.admin_token}"}
        resp = self.fastapi_client.get("/api/admin/monitoring/trends?days=4", headers=headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        http_trends = data.get("http", [])
        # We expect 3 periods matching distinct dates
        self.assertEqual(len(http_trends), 3)
        self.assertEqual(http_trends[0]["period"], (now - datetime.timedelta(days=3)).strftime("%Y-%m-%d"))
        self.assertEqual(http_trends[0]["request_count"], 2)
        self.assertAlmostEqual(http_trends[0]["average_latency_ms"], 150.0)

        self.assertEqual(http_trends[1]["period"], (now - datetime.timedelta(days=2)).strftime("%Y-%m-%d"))
        self.assertEqual(http_trends[1]["request_count"], 1)

    def test_05_provenance_and_security_leakage(self):
        """Verify endpoint payload does not leak any file paths, keys, database paths, or PII."""
        # Seed active model provenance using the repository register method to satisfy all NOT NULL constraints
        self.persistence_repo.register_model_provenance(
            model_type="CLASSIFICATION",
            model_name="BrainNet-B0",
            architecture="EfficientNet-B0",
            model_version="v1.0.0",
            checkpoint_identifier="efficientnet_b0_brain_tumor.pth",
            checkpoint_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            device="cpu",
            loaded_at=datetime.datetime.utcnow().isoformat()
        )

        headers = {"Authorization": f"Bearer {self.admin_token}"}
        resp = self.fastapi_client.get("/api/health-telemetry", headers=headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        # Check model details
        cls_model = data.get("historical_telemetry", {}).get("active_classification_model")
        self.assertIsNotNone(cls_model)
        self.assertEqual(cls_model["model_name"], "BrainNet-B0")
        self.assertEqual(cls_model["checkpoint_sha256"], "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")

        # Assert no paths or secrets leak anywhere in payload
        payload_str = resp.text.lower()
        forbidden_terms = [
            ".db", ".sqlite", "outputs/", "users/", "windows", "c:", "d:",
            "secret", "token", "password", "key", "smtp", "patient_email"
        ]
        for term in forbidden_terms:
            self.assertNotIn(term, payload_str)

    def test_06_cpu_fallback_behaves_safely(self):
        """Verify CPU loading functions operate smoothly returning valid hardware profiles."""
        headers = {"Authorization": f"Bearer {self.admin_token}"}
        resp = self.fastapi_client.get("/api/health-telemetry", headers=headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertIn("cpu_cores", data)
        self.assertIn("cpu_usage_percent", data)
        self.assertIsInstance(data["cpu_usage_percent"], (int, float))
        self.assertTrue(0.0 <= data["cpu_usage_percent"] <= 100.0)

if __name__ == "__main__":
    unittest.main()
