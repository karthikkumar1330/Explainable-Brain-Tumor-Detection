import os
os.environ["DB_PATH"] = os.path.abspath("outputs/test_h732_dashboard.db")

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

class TestH732MonitoringDashboard(unittest.TestCase):
    """focused tests validating H7.3.2 Admin Monitoring Dashboard UI structure, APIs and role security."""

    def setUp(self):
        self.db_path = os.environ.get("DB_PATH", "outputs/test_h732_dashboard.db")
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
            uuid="doc-uuid-732",
            email="doc732@aurascan.ai",
            password_hash="fakehash",
            full_name="Dr. Tester",
            role=Role.DOCTOR,
            is_verified=True,
            is_active=True
        )
        self.user_repo.create_user(self.doctor)

        self.patient = User(
            id=None,
            uuid="pat-uuid-732",
            email="patient732@aurascan.ai",
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

    def test_01_dashboard_role_based_access(self):
        """1. Verify Admin can access dashboard and telemetry while Doctor/Patient are blocked from telemetry APIs."""
        headers_adm = {"Authorization": f"Bearer {self.admin_token}"}
        headers_doc = {"Authorization": f"Bearer {self.doc_token}"}
        headers_pat = {"Authorization": f"Bearer {self.pat_token}"}

        # Admin: ALLOWED
        self.assertEqual(self.flask_client.get("/api/health-telemetry", headers=headers_adm).status_code, 200)
        self.assertEqual(self.flask_client.get("/api/admin/monitoring/trends", headers=headers_adm).status_code, 200)
        self.assertEqual(self.fastapi_client.get("/api/health-telemetry", headers=headers_adm).status_code, 200)
        self.assertEqual(self.fastapi_client.get("/api/admin/monitoring/trends", headers=headers_adm).status_code, 200)

        # Doctor: FORBIDDEN (403)
        self.assertEqual(self.flask_client.get("/api/health-telemetry", headers=headers_doc).status_code, 403)
        self.assertEqual(self.flask_client.get("/api/admin/monitoring/trends", headers=headers_doc).status_code, 403)
        self.assertEqual(self.fastapi_client.get("/api/health-telemetry", headers=headers_doc).status_code, 403)
        self.assertEqual(self.fastapi_client.get("/api/admin/monitoring/trends", headers=headers_doc).status_code, 403)

        # Patient: FORBIDDEN (403)
        self.assertEqual(self.flask_client.get("/api/health-telemetry", headers=headers_pat).status_code, 403)
        self.assertEqual(self.flask_client.get("/api/admin/monitoring/trends", headers=headers_pat).status_code, 403)
        self.assertEqual(self.fastapi_client.get("/api/health-telemetry", headers=headers_pat).status_code, 403)
        self.assertEqual(self.fastapi_client.get("/api/admin/monitoring/trends", headers=headers_pat).status_code, 403)

        # Unauthenticated: UNAUTHORIZED (401/403)
        self.assertIn(self.flask_client.get("/api/health-telemetry").status_code, [401, 403])
        self.assertIn(self.flask_client.get("/api/admin/monitoring/trends").status_code, [401, 403])
        self.assertIn(self.fastapi_client.get("/api/health-telemetry").status_code, [401, 403])
        self.assertIn(self.fastapi_client.get("/api/admin/monitoring/trends").status_code, [401, 403])

    def test_02_dashboard_template_contains_monitoring_tab(self):
        """2. Verify dashboard template contains the required DOM elements for the monitoring tab."""
        template_path = "dashboard/presentation/templates/dashboard_admin.html"
        self.assertTrue(os.path.exists(template_path))
        
        with open(template_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Navigation and containers exist
        self.assertIn('switchTab(\'monitoring\')', content)
        self.assertIn('id="nav-monitoring"', content)
        self.assertIn('id="tab-monitoring"', content)

        # Main KPI/Widgets exist
        self.assertIn('id="mon-system-health"', content)
        self.assertIn('id="mon-db-health"', content)
        self.assertIn('id="mon-cpu-usage-val"', content)
        self.assertIn('id="mon-ram-usage-val"', content)
        self.assertIn('id="mon-disk-usage-val"', content)
        self.assertIn('id="mon-gpu-status"', content)
        self.assertIn('id="chart-http-volume"', content)
        self.assertIn('id="chart-http-latency"', content)

    def test_03_no_hardcoded_secrets_or_paths_in_template(self):
        """3. Assert the HTML template does not leak system paths, DB names, or passwords."""
        template_path = "dashboard/presentation/templates/dashboard_admin.html"
        with open(template_path, "r", encoding="utf-8") as f:
            content = f.read().lower()

        # Sensitive keywords checks
        self.assertNotIn("clinical_reports.db", content)
        self.assertNotIn("identifier.sqlite", content)
        self.assertNotIn("c:\\users\\", content)
        self.assertNotIn("c:/users/", content)

if __name__ == "__main__":
    unittest.main()
