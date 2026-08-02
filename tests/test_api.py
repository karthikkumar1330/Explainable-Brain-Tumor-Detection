import unittest
import os
from fastapi.testclient import TestClient
from run_api import app


class TestFastAPIRoutes(unittest.TestCase):
    """FastAPI REST routing parameters and data schemas validation checks."""

    def setUp(self):
        # Create in-memory test client context
        self.client = TestClient(app)

    def test_dashboard_analytics_api(self):
        """Verify that dashboard analytics telemetry returns correct HTTP status."""
        response = self.client.get("/api/dashboard/analytics")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("total_patients", data)
        self.assertIn("total_scans", data)
        self.assertIn("classification_distribution", data)
        self.assertIn("severity_distribution", data)

    def test_database_history_api(self):
        """Verify that history search listing filters respond correctly."""
        response = self.client.get("/api/database/history?patient_id=PATIENT_001")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIsInstance(data, list)

    def test_invalid_report_intake_validation(self):
        """Verify that out-of-bounds inputs return HTTP 422 Unprocessable Entity."""
        # Age exceeds maximum allowable intake threshold of 120
        payload = {
            "patient_id": "TEST_INVALID",
            "name": "Invalid Patient Name",
            "age": 150,  # Invalid
            "gender": "Female",
            "ref_physician": "Dr. Sarah",
            "pixel_spacing_mm": 1.0
        }
        response = self.client.post("/api/report?filepath=dummy_path.png", json=payload)
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
