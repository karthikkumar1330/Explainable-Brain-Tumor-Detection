import unittest
import os
from fastapi.testclient import TestClient
from run_api import app
from security.domain.entities import Role
from security.infrastructure.jwt_service import JWTService
from security.infrastructure.repository import SQLiteUserRepository


class TestFastAPIRoutes(unittest.TestCase):
    """FastAPI REST routing parameters, security protection, and data schemas validation checks."""

    def setUp(self):
        self.test_client_ctx = TestClient(app)
        self.client = self.test_client_ctx.__enter__()
        
        repo = SQLiteUserRepository(db_path="outputs/clinical_reports.db")
        repo.initialize_security_tables()
        admin = repo.bootstrap_admin()
        jwt_svc = JWTService()
        self.auth_token = jwt_svc.create_access_token(
            user_uuid=admin.uuid,
            user_id=admin.id,
            email=admin.email,
            role=admin.role
        )
        self.headers = {"Authorization": f"Bearer {self.auth_token}"}

    def tearDown(self):
        self.test_client_ctx.__exit__()

    def test_dashboard_analytics_api(self):
        """Verify that dashboard analytics telemetry returns correct HTTP status."""
        response = self.client.get("/api/dashboard/analytics")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("total_patients", data)
        self.assertIn("total_scans", data)

    def test_database_history_api(self):
        """Verify that history search listing filters respond correctly."""
        response = self.client.get("/api/database/history?patient_id=PATIENT_001")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, list)

    def test_protected_report_endpoint_unauthorized(self):
        """Verify that unauthenticated requests to prediction endpoints return 401 Unauthorized."""
        payload = {
            "patient_id": "TEST_INVALID",
            "name": "Invalid Patient Name",
            "age": 45,
            "gender": "Female",
            "ref_physician": "Dr. Sarah",
            "pixel_spacing_mm": 1.0
        }
        response = self.client.post("/api/report?filepath=dummy_path.png", json=payload)
        self.assertEqual(response.status_code, 401)

    def test_invalid_report_intake_validation(self):
        """Verify that out-of-bounds inputs return HTTP 422 Unprocessable Entity when authorized."""
        payload = {
            "patient_id": "TEST_INVALID",
            "name": "Invalid Patient Name",
            "age": 150,  # Invalid age > 120
            "gender": "Female",
            "ref_physician": "Dr. Sarah",
            "pixel_spacing_mm": 1.0
        }
        response = self.client.post(
            "/api/report?filepath=dummy_path.png",
            json=payload,
            headers=self.headers
        )
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
