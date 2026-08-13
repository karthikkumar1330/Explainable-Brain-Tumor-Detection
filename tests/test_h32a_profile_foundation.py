import os
import sqlite3
import datetime
import unittest
from fastapi.testclient import TestClient

# Configure test database environment
os.environ["DB_PATH"] = os.path.abspath("outputs/test_h32a_foundation.db")
os.environ["DISABLE_TEST_AUTO_ASSIGN"] = "1"

from run_api import app
from dashboard.infrastructure.web_server import create_app
from security.domain.entities import Role, User
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.jwt_service import JWTService
from security.application.use_cases import AuthUseCases
from persistence.infrastructure.repository import SQLitePersistenceRepository
from security.infrastructure.encryption_service import PIIEncryptionService

class TestH32APatientProfileFoundation(unittest.TestCase):
    """Focused integration tests for Phase H3.2-A patient profile foundation."""

    def setUp(self):
        self.db_path = os.environ.get("DB_PATH", "outputs/test_h32a_foundation.db")
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

        # Make sure FastAPI and routes default paths point to our test database
        import api.infrastructure.routes as api_routes
        import api.routes.auth_routes as auth_routes
        api_routes.DEFAULT_DB_PATH = self.db_path
        auth_routes.DEFAULT_DB_PATH = self.db_path

        # Configure Flask App
        self.flask_app = create_app(db_path=self.db_path)
        self.flask_app.config["TESTING"] = True
        self.flask_app.config["DISABLE_CSRF"] = True
        self.flask_client = self.flask_app.test_client()

        # Initialize DB schemas
        self.persistence_repo = SQLitePersistenceRepository(db_path=self.db_path)
        self.persistence_repo.initialize_db()

        self.user_repo = SQLiteUserRepository(db_path=self.db_path)
        self.user_repo.initialize_security_tables()

        # Drop the auto-assignment triggers
        conn_drop = sqlite3.connect(self.db_path)
        try:
            conn_drop.execute("DROP TRIGGER IF EXISTS auto_assign_doctor_on_insert;")
            conn_drop.execute("DROP TRIGGER IF EXISTS auto_assign_patient_on_insert;")
            conn_drop.commit()
        finally:
            conn_drop.close()

        # Setup Password Hasher
        from security.infrastructure.password import PasswordHasher
        pass_hash = PasswordHasher.hash_password("Password@123")

        # Bootstrap a Patient user
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO users (id, uuid, email, password_hash, full_name, role, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (20, "pat-uuid-a", "patient_a@aurascan.ai", pass_hash, "Original Patient Name", Role.PATIENT.value, datetime.datetime.utcnow().isoformat(), datetime.datetime.utcnow().isoformat())
            )
            # Insert corresponding encrypted patient demographic record
            encryption_service = PIIEncryptionService()
            enc_name = encryption_service.encrypt("Original Patient Name")
            enc_age = encryption_service.encrypt("45")
            enc_gender = encryption_service.encrypt("Male")
            conn.execute(
                "INSERT INTO patients (patient_id, name, age, gender, created_at) VALUES (?, ?, ?, ?, ?);",
                ("pat-uuid-a", enc_name, enc_age, enc_gender, datetime.datetime.utcnow().isoformat())
            )
            conn.commit()
        finally:
            conn.close()

        # Load patient user and generate token
        self.patient_user = self.user_repo.get_by_email("patient_a@aurascan.ai")
        self.jwt_svc = JWTService()
        self.token_pat_a = self.jwt_svc.create_access_token(
            self.patient_user.uuid,
            self.patient_user.id,
            self.patient_user.email,
            Role.PATIENT
        )

        self.fastapi_ctx = TestClient(app)
        self.fastapi_client = self.fastapi_ctx.__enter__()

    def tearDown(self):
        self.fastapi_ctx.__exit__(None, None, None)
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def _get_headers(self, token):
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

    def test_patient_name_sync_on_profile_update(self):
        """Test: Updating a patient user's profile full_name synchronizes it to the patients table."""
        headers = self._get_headers(self.token_pat_a)

        # 1. Perform PUT /api/auth/profile via Flask App
        resp = self.flask_client.put(
            "/api/auth/profile",
            json={"full_name": "Synchronized Patient Name"},
            headers=headers
        )
        self.assertEqual(resp.status_code, 200)

        # 2. Verify that users table is updated
        updated_user = self.user_repo.get_by_email("patient_a@aurascan.ai")
        self.assertEqual(updated_user.full_name, "Synchronized Patient Name")

        # 3. Verify that patients table is updated and encrypted correctly
        conn = sqlite3.connect(self.db_path)
        try:
            row = conn.execute("SELECT name FROM patients WHERE patient_id = ?;", ("pat-uuid-a",)).fetchone()
            self.assertIsNotNone(row)
            enc_name = row[0]

            # Assert name is stored encrypted with 'enc:v1:' prefix
            self.assertTrue(str(enc_name).startswith("enc:v1:"))
            self.assertNotEqual(enc_name, "Synchronized Patient Name")

            # Decrypt name and verify it matches the updated name
            encryption_service = PIIEncryptionService()
            decrypted_name = encryption_service.decrypt(enc_name)
            self.assertEqual(decrypted_name, "Synchronized Patient Name")
        finally:
            conn.close()

    def test_timeline_route_deduplication(self):
        """Test: FastAPI router does not have duplicate timeline endpoint registrations."""
        # Find all routes matching the longitudinal-timeline pattern inside original_routers
        timeline_routes = []
        for route in app.router.routes:
            if "IncludedRouter" in type(route).__name__:
                orig = getattr(route, "original_router", None)
                if orig:
                    for sub_route in orig.routes:
                        if getattr(sub_route, "path", None) == "/patients/{patient_id}/longitudinal-timeline":
                            timeline_routes.append(sub_route)

        # Must be exactly one route registered
        self.assertEqual(len(timeline_routes), 1, f"There should be exactly one timeline route registered, found: {len(timeline_routes)}")
        # Check endpoint function name is get_patient_timeline_api
        self.assertEqual(timeline_routes[0].endpoint.__name__, "get_patient_timeline_api")

if __name__ == "__main__":
    unittest.main()
