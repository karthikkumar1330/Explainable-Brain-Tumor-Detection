import os
os.environ["DB_PATH"] = os.path.abspath("outputs/test_h72b_telemetry.db")

import unittest
import sqlite3
import datetime
import time
import numpy as np
import cv2
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient as FastAPITestClient
from run_api import app as fastapi_app
from dashboard.infrastructure.web_server import create_app
from security.domain.entities import Role, User
from security.infrastructure.jwt_service import JWTService
from security.infrastructure.repository import SQLiteUserRepository
from persistence.infrastructure.repository import SQLitePersistenceRepository

class TestH72BBatchTelemetry(unittest.TestCase):
    """Phase H7.2-B: Batch Performance Telemetry Persistence tests."""

    def setUp(self):
        self.db_path = os.environ.get("DB_PATH", "outputs/test_h72b_telemetry.db")
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

        # Override DB paths
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

        # Disable auto-assignment triggers
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DROP TRIGGER IF EXISTS auto_assign_doctor_on_insert;")
            conn.execute("DROP TRIGGER IF EXISTS auto_assign_patient_on_insert;")
            conn.execute("DELETE FROM doctor_patient_assignments;")
            conn.commit()
        finally:
            conn.close()

        # Create roles
        self.admin = self.user_repo.bootstrap_admin()
        self.doctor = User(
            id=None,
            uuid="doc-test-uuid-b",
            email="doctor-b@aurascan.ai",
            password_hash="fakehash",
            full_name="Dr. Smith",
            role=Role.DOCTOR,
            is_verified=True,
            is_active=True,
            created_at=datetime.datetime.utcnow().isoformat(),
            updated_at=datetime.datetime.utcnow().isoformat()
        )
        self.doctor = self.user_repo.create_user(self.doctor)

        self.patient = User(
            id=None,
            uuid="pat-test-uuid-b",
            email="patient-b@aurascan.ai",
            password_hash="fakehash",
            full_name="Patient Bob",
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

        # Clean temporary uploads directory
        self.temp_dir = "outputs/temp_uploads"
        if os.path.exists(self.temp_dir):
            for filename in os.listdir(self.temp_dir):
                if filename.startswith("batch_"):
                    try:
                        os.remove(os.path.join(self.temp_dir, filename))
                    except Exception:
                        pass

    def tearDown(self):
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass
        if os.path.exists(self.temp_dir):
            for filename in os.listdir(self.temp_dir):
                if filename.startswith("batch_"):
                    try:
                        os.remove(os.path.join(self.temp_dir, filename))
                    except Exception:
                        pass

    def _create_mock_mri_bytes(self, label_text: str = "BRAIN_MRI", seed: int = 42, axes: tuple = (60, 80)) -> bytes:
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        mask = np.zeros((256, 256), dtype=np.uint8)
        cv2.ellipse(mask, (128, 128), axes, 0, 0, 360, 255, -1)
        np.random.seed(seed)
        texture = np.random.randint(90, 210, size=(256, 256), dtype=np.uint8)
        brain_tissue = cv2.bitwise_and(texture, texture, mask=mask)
        brain_tissue_blurred = cv2.GaussianBlur(brain_tissue, (5, 5), 0)
        brain_tissue = np.where(mask > 0, brain_tissue_blurred, 0)
        img[:, :, 0] = brain_tissue
        img[:, :, 1] = brain_tissue
        img[:, :, 2] = brain_tissue
        cv2.ellipse(img, (128, 110), (15, 8), 0, 0, 360, (40, 40, 40), -1)
        cv2.ellipse(img, (128, 140), (12, 6), 0, 0, 360, (40, 40, 40), -1)
        cv2.putText(img, label_text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        _, buf = cv2.imencode(".png", img)
        return buf.tobytes()

    @patch("api.infrastructure.routes._run_single_report_pipeline")
    def test_01_successful_batch_persists_telemetry(self, mock_pipeline):
        """1. Verify successful batch processing saves telemetry correctly with exact counts."""
        # Set up mock pipeline response
        mock_pipeline.return_value = {
            "report_id": 999,
            "patient_id": "PAT_BATCH_TEST",
            "diagnosis": "Glioma",
            "confidence": 0.85,
            "severity": "Mild",
            "tumor_area_mm2": 150.0,
            "execution_time_sec": 1.2,
            "xai_method": "gradcam",
            "xai_explanation": "Sample text explanation",
            "xai_overlap_percentage": 78.5,
            "quality_warnings": [],
            "ensemble": None
        }

        # Clear existing telemetry
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DELETE FROM batch_performance_telemetry;")
            conn.commit()
        finally:
            conn.close()

        # Prepare payload
        file1 = self._create_mock_mri_bytes("SLICE_1", seed=10, axes=(60, 80))
        file2 = self._create_mock_mri_bytes("SLICE_2", seed=20, axes=(45, 65))
        payload = {
            "patient_id": "PAT_BATCH_TEST",
            "name": "Jane Patient",
            "age": 42,
            "gender": "Female",
            "ref_physician": "Dr. Smith",
            "pixel_spacing_mm": 1.0,
            "xai_method": "gradcam"
        }
        files = [
            ("files", ("slice1.png", file1, "image/png")),
            ("files", ("slice2.png", file2, "image/png"))
        ]

        # Use admin token to bypass patient assignment auth checks
        headers = {"Authorization": f"Bearer {self.admin_token}"}
        resp = self.fastapi_client.post("/api/report/batch", data=payload, files=files, headers=headers)
        self.assertEqual(resp.status_code, 200)

        # Verify response structure compatibility (Step 9/11)
        data = resp.json()
        self.assertIn("stats", data)
        self.assertIn("results", data)
        self.assertEqual(data["stats"]["total_files"], 2)
        self.assertEqual(data["stats"]["successful"], 2)
        self.assertEqual(data["stats"]["failed"], 0)

        # Verify telemetry table contains correct record
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute("SELECT * FROM batch_performance_telemetry;").fetchall()
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["batch_id"], data["stats"]["batch_id"])
            self.assertEqual(row["total_items"], 2)
            self.assertEqual(row["successful_items"], 2)
            self.assertEqual(row["failed_items"], 0)
            self.assertGreaterEqual(row["total_duration_ms"], 0.0)
            # average_item_duration_ms should be total_duration / 2
            self.assertAlmostEqual(row["average_item_duration_ms"], row["total_duration_ms"] / 2.0, places=3)
        finally:
            conn.close()

    @patch("api.infrastructure.routes._run_single_report_pipeline")
    def test_02_batch_telemetry_captures_partial_failures(self, mock_pipeline):
        """2. Verify batch telemetry captures both successful, failed, and duplicate items."""
        # Set up mock pipeline response
        mock_pipeline.return_value = {
            "report_id": 999,
            "patient_id": "PAT_BATCH_TEST",
            "diagnosis": "Glioma",
            "confidence": 0.85,
            "severity": "Mild",
            "tumor_area_mm2": 150.0,
            "execution_time_sec": 1.2,
            "xai_method": "gradcam",
            "xai_explanation": "Sample text explanation",
            "xai_overlap_percentage": 78.5,
            "quality_warnings": [],
            "ensemble": None
        }

        # Clear existing telemetry
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DELETE FROM batch_performance_telemetry;")
            conn.commit()
        finally:
            conn.close()

        # Send duplicates and empty files to trigger validation failures
        file1 = self._create_mock_mri_bytes("SLICE_A", seed=10, axes=(60, 80))
        # Empty file will fail validation
        file_empty = b""

        payload = {
            "patient_id": "PAT_BATCH_TEST",
            "name": "Jane Patient",
            "age": 42,
            "gender": "Female",
            "ref_physician": "Dr. Smith",
            "pixel_spacing_mm": 1.0,
            "xai_method": "gradcam"
        }
        files = [
            ("files", ("slice1.png", file1, "image/png")),
            ("files", ("slice2.png", file_empty, "image/png")),
            ("files", ("slice_dup.png", file1, "image/png"))  # Identical content hash to slice1
        ]

        # Use admin token to bypass patient assignment auth checks
        headers = {"Authorization": f"Bearer {self.admin_token}"}
        resp = self.fastapi_client.post("/api/report/batch", data=payload, files=files, headers=headers)
        self.assertEqual(resp.status_code, 200)

        # Verify DB entries
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute("SELECT * FROM batch_performance_telemetry;").fetchall()
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["total_items"], 3)
            # successful_items should be 1 because slice1 runs successfully through the real pipeline
            # failed_items should be 2 because slice2 (empty) and slice_dup (duplicate) fail validation
            self.assertEqual(row["successful_items"], 1)
            self.assertEqual(row["failed_items"], 2)
            self.assertGreaterEqual(row["total_duration_ms"], 0.0)
        finally:
            conn.close()

    def test_03_telemetry_contains_no_sensitive_or_pii_fields(self):
        """3. Verify batch telemetry stores ONLY operational data and no patient PII/clinical results."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            # Query the table info to check column names
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(batch_performance_telemetry);")
            columns = [col["name"] for col in cursor.fetchall()]

            # Stored columns must strictly be: id, batch_id, timestamp, total_items, successful_items, failed_items, total_duration_ms, average_item_duration_ms
            allowed_columns = ["id", "batch_id", "timestamp", "total_items", "successful_items", "failed_items", "total_duration_ms", "average_item_duration_ms"]
            for col in columns:
                self.assertIn(col, allowed_columns)

            # Specifically ensure no clinical/user parameters exist
            self.assertNotIn("patient_id", columns)
            self.assertNotIn("name", columns)
            self.assertNotIn("patient_name", columns)
            self.assertNotIn("diagnosis", columns)
            self.assertNotIn("confidence", columns)
            self.assertNotIn("mri_path", columns)
        finally:
            conn.close()

    def test_04_telemetry_persistence_failure_does_not_fail_batch(self):
        """4. Verify that a database telemetry insertion failure does not crash the API response."""
        with patch.object(SQLitePersistenceRepository, "save_batch_performance_telemetry", side_effect=Exception("Database lock error")):
            payload = {
                "patient_id": "PAT_BATCH_TEST",
                "name": "Jane Patient",
                "age": 42,
                "gender": "Female",
                "ref_physician": "Dr. Smith",
                "pixel_spacing_mm": 1.0,
                "xai_method": "gradcam"
            }
            files = [
                ("files", ("slice1.png", b"invalid_slice_format", "image/png"))
            ]
            # Use admin token to bypass patient assignment auth checks
            headers = {"Authorization": f"Bearer {self.admin_token}"}
            resp = self.fastapi_client.post("/api/report/batch", data=payload, files=files, headers=headers)
            # Endpoint must return 200 successfully even though telemetry saving crashed
            self.assertEqual(resp.status_code, 200)

    def test_05_duplicate_telemetry_prevented_by_unique_constraint(self):
        """5. Verify that duplicate telemetry insertions for the same batch_id are safely ignored."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("DELETE FROM batch_performance_telemetry;")
            conn.commit()
        finally:
            conn.close()

        # Insert once
        ts = datetime.datetime.utcnow().isoformat()
        self.persistence_repo.save_batch_performance_telemetry(
            batch_id="stable-batch-123",
            timestamp=ts,
            total_items=5,
            successful_items=4,
            failed_items=1,
            total_duration_ms=500.0,
            average_item_duration_ms=100.0
        )

        # Attempt to insert same batch_id again (should be ignored due to INSERT OR IGNORE)
        self.persistence_repo.save_batch_performance_telemetry(
            batch_id="stable-batch-123",
            timestamp=ts,
            total_items=5,
            successful_items=4,
            failed_items=1,
            total_duration_ms=600.0,
            average_item_duration_ms=120.0
        )

        # Verify only one row is written, and it contains the first write's values
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute("SELECT * FROM batch_performance_telemetry WHERE batch_id = 'stable-batch-123';").fetchall()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["total_duration_ms"], 500.0)
        finally:
            conn.close()

    def test_06_rbac_detailed_telemetry_access(self):
        """6. Verify Role-based Access Control limits batch performance aggregates to Admin."""
        # Add a test batch log
        ts = datetime.datetime.utcnow().isoformat()
        self.persistence_repo.save_batch_performance_telemetry(
            batch_id="audit-batch-999",
            timestamp=ts,
            total_items=10,
            successful_items=9,
            failed_items=1,
            total_duration_ms=1200.0,
            average_item_duration_ms=120.0
        )

        # 1. Admin must be allowed access and see the stats
        headers = {"Authorization": f"Bearer {self.admin_token}"}
        resp = self.flask_client.get("/api/health-telemetry", headers=headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("total_batches", data)
        self.assertIn("avg_batch_latency_ms", data)
        self.assertIn("total_batch_items", data)
        self.assertEqual(data["total_batches"], 1)
        self.assertEqual(data["total_batch_items"], 10)
        self.assertAlmostEqual(data["avg_batch_latency_ms"], 1200.0, places=3)

        # 2. Doctor must be denied access (403)
        headers_doc = {"Authorization": f"Bearer {self.doc_token}"}
        resp_doc = self.flask_client.get("/api/health-telemetry", headers=headers_doc)
        self.assertEqual(resp_doc.status_code, 403)

        # 3. Patient must be denied access (403)
        headers_pat = {"Authorization": f"Bearer {self.pat_token}"}
        resp_pat = self.flask_client.get("/api/health-telemetry", headers=headers_pat)
        self.assertEqual(resp_pat.status_code, 403)

if __name__ == "__main__":
    unittest.main()
