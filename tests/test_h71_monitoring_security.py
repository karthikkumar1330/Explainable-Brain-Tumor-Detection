import os
import unittest
import tempfile
import sqlite3
import datetime
from fastapi.testclient import TestClient
from run_api import app
from security.domain.entities import Role, User
from security.infrastructure.jwt_service import JWTService
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.password import PasswordHasher
from persistence.infrastructure.repository import SQLitePersistenceRepository


class TestH71MonitoringSecurity(unittest.TestCase):
    """Phase H7.1 focused integration tests validating liveness, basic health, and detailed telemetry access security."""

    def setUp(self):
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.environ["DB_PATH"] = self.db_path

        # Override route variables to use our test DB
        import api.infrastructure.routes as api_routes
        import api.routes.auth_routes as auth_routes
        api_routes.DEFAULT_DB_PATH = self.db_path
        auth_routes.DEFAULT_DB_PATH = self.db_path

        # Initialize persistence database structure
        self.persistence_repo = SQLitePersistenceRepository(db_path=self.db_path)
        self.persistence_repo.initialize_db()

        self.user_repo = SQLiteUserRepository(db_path=self.db_path)
        self.user_repo.initialize_security_tables()

        # Seed test users
        self.admin = self.user_repo.bootstrap_admin(admin_email="admin@aurascan.ai", admin_pass="AdminPass@123")

        self.doctor = User(
            id=None,
            uuid="doc-uuid-71",
            email="doc71@aurascan.ai",
            password_hash=PasswordHasher.hash_password("DoctorPass@123"),
            full_name="Dr. Hardened",
            role=Role.DOCTOR,
            is_verified=True,
            is_active=True
        )
        self.user_repo.create_user(self.doctor)

        self.patient = User(
            id=None,
            uuid="pat-uuid-71",
            email="patient71@aurascan.ai",
            password_hash=PasswordHasher.hash_password("PatientPass@123"),
            full_name="Patient Hardened",
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
        self.doctor_token = jwt_svc.create_access_token(
            user_uuid=self.doctor.uuid, user_id=self.doctor.id, email=self.doctor.email, role=self.doctor.role
        )
        self.patient_token = jwt_svc.create_access_token(
            user_uuid=self.patient.uuid, user_id=self.patient.id, email=self.patient.email, role=self.patient.role
        )

        # Create token for invalid/unknown role (direct string injection)
        self.invalid_role_token = jwt_svc.create_access_token(
            user_uuid=self.patient.uuid, user_id=self.patient.id, email=self.patient.email, role="visitor"
        )

        self.test_client_ctx = TestClient(app)
        self.client = self.test_client_ctx.__enter__()

    def tearDown(self):
        self.test_client_ctx.__exit__()
        os.close(self.db_fd)
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def test_01_health_is_public_and_minimal(self):
        """Validate GET /health (exposed via /api/health) works without auth and returns minimal safe status check."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data, {"status": "healthy"})

    def test_02_health_does_not_leak_telemetry(self):
        """Verify GET /health response does not expose sensitive infrastructure metrics."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Ensure zero exposure of detailed hardware metrics
        forbidden_keys = [
            "cpu_cores", "cpu_threads", "ram_total_gb", "ram_used_gb",
            "ram_usage_percent", "disk_total_gb", "disk_free_gb",
            "disk_usage_percent", "cuda_available", "gpu_device_name",
            "gpu_vram_total_mb", "gpu_vram_used_mb", "gpu_vram_free_mb",
            "historical_telemetry", "model_cls", "model_seg"
        ]
        for key in forbidden_keys:
            self.assertNotIn(key, data)

    def test_03_ping_is_unchanged_and_safe(self):
        """Validate GET /ping is publicly accessible and returns status and timestamp."""
        response = self.client.get("/ping")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertIn("timestamp", data)
        self.assertNotIn("cpu_cores", data)
        self.assertNotIn("ram_total_gb", data)

    def test_04_admin_can_access_detailed_telemetry(self):
        """Verify ADMIN role is allowed to access GET /api/health-telemetry."""
        headers = {"Authorization": f"Bearer {self.admin_token}"}
        response = self.client.get("/api/health-telemetry", headers=headers)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Verify infrastructure details are present
        self.assertIn("cpu_cores", data)
        self.assertIn("ram_total_gb", data)
        self.assertIn("disk_usage_percent", data)
        self.assertIn("historical_telemetry", data)

    def test_05_doctor_is_denied_detailed_telemetry(self):
        """Verify DOCTOR role is forbidden (403) from GET /api/health-telemetry."""
        headers = {"Authorization": f"Bearer {self.doctor_token}"}
        response = self.client.get("/api/health-telemetry", headers=headers)
        self.assertEqual(response.status_code, 403)
        self.assertNotIn("cpu_cores", response.json())

    def test_06_patient_is_denied_detailed_telemetry(self):
        """Verify PATIENT role is forbidden (403) from GET /api/health-telemetry."""
        headers = {"Authorization": f"Bearer {self.patient_token}"}
        response = self.client.get("/api/health-telemetry", headers=headers)
        self.assertEqual(response.status_code, 403)
        self.assertNotIn("cpu_cores", response.json())

    def test_07_unauthenticated_is_denied_detailed_telemetry(self):
        """Verify requests without JWT authorization return 401 Unauthorized."""
        response = self.client.get("/api/health-telemetry")
        self.assertEqual(response.status_code, 401)

    def test_08_invalid_role_is_denied_detailed_telemetry(self):
        """Verify unknown roles return 403 Forbidden for detailed telemetry."""
        headers = {"Authorization": f"Bearer {self.invalid_role_token}"}
        response = self.client.get("/api/health-telemetry", headers=headers)
        self.assertEqual(response.status_code, 403)

    def test_09_unauthorized_responses_do_not_leak_details_or_exceptions(self):
        """Verify that blocked responses do not leak system telemetry or stack traces."""
        # Unauthenticated check
        resp1 = self.client.get("/api/health-telemetry")
        self.assertEqual(resp1.status_code, 401)
        data1 = resp1.json()
        self.assertNotIn("cpu_cores", data1)
        self.assertNotIn("traceback", data1)

        # Doctor check
        headers_doc = {"Authorization": f"Bearer {self.doctor_token}"}
        resp2 = self.client.get("/api/health-telemetry", headers=headers_doc)
        self.assertEqual(resp2.status_code, 403)
        data2 = resp2.json()
        self.assertNotIn("cpu_cores", data2)
        self.assertNotIn("traceback", data2)


if __name__ == "__main__":
    unittest.main()
