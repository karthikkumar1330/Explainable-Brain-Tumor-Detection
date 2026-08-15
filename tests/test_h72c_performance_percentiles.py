import os
os.environ["DB_PATH"] = os.path.abspath("outputs/test_h72c_telemetry.db")

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

class TestH72CPerformancePercentiles(unittest.TestCase):
    """Phase H7.2-C: Performance Percentile Aggregation tests."""

    def setUp(self):
        self.db_path = os.environ.get("DB_PATH", "outputs/test_h72c_telemetry.db")
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

        # Disable triggers
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
            uuid="doc-test-c",
            email="doctor-c@aurascan.ai",
            password_hash="fakehash",
            full_name="Dr. Tester",
            role=Role.DOCTOR,
            is_verified=True,
            is_active=True,
            created_at=datetime.datetime.utcnow().isoformat(),
            updated_at=datetime.datetime.utcnow().isoformat()
        )
        self.doctor = self.user_repo.create_user(self.doctor)

        self.patient = User(
            id=None,
            uuid="pat-test-c",
            email="patient-c@aurascan.ai",
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

        self.flask_client = self.flask_app.test_client()

    def tearDown(self):
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def test_01_empty_dataset_returns_none(self):
        """1. Empty values array returns None for all percentiles."""
        self.assertIsNone(SQLitePersistenceRepository.calculate_percentile([], 50.0))
        self.assertIsNone(SQLitePersistenceRepository.calculate_percentile([], 95.0))
        self.assertIsNone(SQLitePersistenceRepository.calculate_percentile([], 99.0))

    def test_02_single_observation_returns_itself(self):
        """2. Single item list returns the item itself for all percentiles."""
        vals = [123.45]
        self.assertEqual(SQLitePersistenceRepository.calculate_percentile(vals, 50.0), 123.45)
        self.assertEqual(SQLitePersistenceRepository.calculate_percentile(vals, 95.0), 123.45)
        self.assertEqual(SQLitePersistenceRepository.calculate_percentile(vals, 99.0), 123.45)

    def test_03_algorithm_correctness_small_dataset(self):
        """3. Validate linear interpolation percentile values on small known datasets."""
        # Dataset [1, 2, 3, 4, 5]
        vals_small = [1, 2, 3, 4, 5]
        self.assertEqual(SQLitePersistenceRepository.calculate_percentile(vals_small, 50.0), 3.0)
        self.assertAlmostEqual(SQLitePersistenceRepository.calculate_percentile(vals_small, 95.0), 4.8, places=5)
        self.assertAlmostEqual(SQLitePersistenceRepository.calculate_percentile(vals_small, 99.0), 4.96, places=5)

        # Dataset [100, 200, 300, 400, 500]
        vals_hundreds = [100, 200, 300, 400, 500]
        self.assertEqual(SQLitePersistenceRepository.calculate_percentile(vals_hundreds, 50.0), 300.0)
        self.assertAlmostEqual(SQLitePersistenceRepository.calculate_percentile(vals_hundreds, 95.0), 480.0, places=5)
        self.assertAlmostEqual(SQLitePersistenceRepository.calculate_percentile(vals_hundreds, 99.0), 496.0, places=5)

    def test_04_algorithm_sorting_independence(self):
        """4. Verify unsorted list returns identical results to sorted list."""
        unsorted = [400, 100, 500, 200, 300]
        self.assertEqual(SQLitePersistenceRepository.calculate_percentile(unsorted, 50.0), 300.0)
        self.assertAlmostEqual(SQLitePersistenceRepository.calculate_percentile(unsorted, 95.0), 480.0, places=5)
        self.assertAlmostEqual(SQLitePersistenceRepository.calculate_percentile(unsorted, 99.0), 496.0, places=5)

    def test_05_duplicate_values_handling(self):
        """5. Duplicate numeric values are handled correctly."""
        dups = [10, 10, 20, 20, 30, 30]
        self.assertEqual(SQLitePersistenceRepository.calculate_percentile(dups, 50.0), 20.0)
        self.assertEqual(SQLitePersistenceRepository.calculate_percentile(dups, 95.0), 30.0)

    def test_06_corrupt_and_invalid_data_ignored_gracefully(self):
        """6. Malformed data (None, NaN, Infinity, strings) is safely ignored without crashing."""
        bad_data = [10.0, None, float("nan"), float("inf"), "corrupt_string", 20.0, 30.0]
        self.assertEqual(SQLitePersistenceRepository.calculate_percentile(bad_data, 50.0), 20.0)
        self.assertAlmostEqual(SQLitePersistenceRepository.calculate_percentile(bad_data, 95.0), 29.0, places=5)

    def test_07_http_and_batch_telemetry_isolation(self):
        """7. Verify HTTP and Batch telemetry percentiles are calculated separately over correct tables."""
        # Insert HTTP request timings (Dataset: 100, 200, 300, 400, 500 ms)
        # Insert Batch timings (Dataset: 1000, 2000, 3000, 4000, 5000 ms)
        conn = sqlite3.connect(self.db_path)
        try:
            # HTTP request writes
            for idx, dur in enumerate([100.0, 200.0, 300.0, 400.0, 500.0]):
                conn.execute("""
                    INSERT INTO http_request_telemetry (timestamp, duration_ms, method, route, status_code)
                    VALUES (?, ?, 'GET', '/api/report', 200);
                """, (f"2026-08-15T12:00:0{idx}", dur))
            # Batch performance writes
            for idx, dur in enumerate([1000.0, 2000.0, 3000.0, 4000.0, 5000.0]):
                conn.execute("""
                    INSERT INTO batch_performance_telemetry (batch_id, timestamp, total_items, successful_items, failed_items, total_duration_ms, average_item_duration_ms)
                    VALUES (?, ?, 1, 1, 0, ?, ?);
                """, (f"batch-{idx}", f"2026-08-15T12:10:0{idx}", dur, dur))
            conn.commit()
        finally:
            conn.close()

        # Query health telemetry as ADMIN
        headers = {"Authorization": f"Bearer {self.admin_token}"}
        resp = self.flask_client.get("/api/health-telemetry", headers=headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()

        # Verify HTTP Request percentiles (P50 = 300.0, P95 = 480.0, P99 = 496.0)
        self.assertEqual(data["http_p50_latency_ms"], 300.0)
        self.assertAlmostEqual(data["http_p95_latency_ms"], 480.0, places=5)
        self.assertAlmostEqual(data["http_p99_latency_ms"], 496.0, places=5)
        # Verify existing HTTP averages are preserved
        self.assertEqual(data["total_http_requests"], 5)
        self.assertAlmostEqual(data["avg_http_latency_ms"], 300.0, places=5)

        # Verify Batch latency percentiles (P50 = 3000.0, P95 = 4800.0, P99 = 4960.0)
        self.assertEqual(data["batch_p50_latency_ms"], 3000.0)
        self.assertAlmostEqual(data["batch_p95_latency_ms"], 4800.0, places=5)
        self.assertAlmostEqual(data["batch_p99_latency_ms"], 4960.0, places=5)
        # Verify existing Batch averages are preserved
        self.assertEqual(data["total_batches"], 5)
        self.assertAlmostEqual(data["avg_batch_latency_ms"], 3000.0, places=5)

    def test_08_rbac_detailed_telemetry_access(self):
        """8. Verify only admin users can retrieve detailed telemetry with percentiles."""
        # 1. Admin ALLOWED
        headers = {"Authorization": f"Bearer {self.admin_token}"}
        resp = self.flask_client.get("/api/health-telemetry", headers=headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("http_p50_latency_ms", data)
        self.assertIn("batch_p50_latency_ms", data)

        # 2. Doctor DENIED (403)
        headers_doc = {"Authorization": f"Bearer {self.doc_token}"}
        resp_doc = self.flask_client.get("/api/health-telemetry", headers=headers_doc)
        self.assertEqual(resp_doc.status_code, 403)

        # 3. Patient DENIED (403)
        headers_pat = {"Authorization": f"Bearer {self.pat_token}"}
        resp_pat = self.flask_client.get("/api/health-telemetry", headers=headers_pat)
        self.assertEqual(resp_pat.status_code, 403)

    def test_09_no_secrets_or_pii_disclosed(self):
        """9. Verify health telemetry endpoint outputs only operational aggregates and no PII."""
        headers = {"Authorization": f"Bearer {self.admin_token}"}
        resp = self.flask_client.get("/api/health-telemetry", headers=headers)
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()

        # Stored keys
        for key in data.keys():
            self.assertNotIn(key, ["patient_name", "password", "token", "jwt", "cookie", "mri_file", "key"])

if __name__ == "__main__":
    unittest.main()
