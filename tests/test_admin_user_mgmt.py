import unittest
import os
import tempfile
from dashboard.infrastructure.web_server import create_app
from security.domain.entities import Role, User
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.password import PasswordHasher


class TestAdminUserMgmt(unittest.TestCase):
    """Test suite to verify Administrator operations, paginated queries, CSV database exports, and resets."""

    def setUp(self):
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        self.app = create_app(db_path=self.db_path)
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

        # Initialize tables
        self.repo = SQLiteUserRepository(db_path=self.db_path)
        self.repo.initialize_security_tables()

        # Create Admin
        self.admin = User(
            id=None,
            uuid="admin-uuid",
            email="testadmin@aurascan.ai",
            password_hash=PasswordHasher.hash_password("AdminSecurePassword@123"),
            full_name="Admin User",
            role=Role.ADMIN,
            is_verified=True,
            is_active=True
        )
        self.repo.create_user(self.admin)

        # Create Patient
        self.patient = User(
            id=None,
            uuid="patient-uuid",
            email="patient@aurascan.ai",
            password_hash=PasswordHasher.hash_password("PatientSecurePassword@123"),
            full_name="Patient User",
            role=Role.PATIENT,
            is_verified=True,
            is_active=True
        )
        self.repo.create_user(self.patient)

    def tearDown(self):
        os.close(self.db_fd)
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def _login_admin(self):
        resp = self.client.post("/api/auth/login", json={
            "email": "testadmin@aurascan.ai",
            "password": "AdminSecurePassword@123"
        })
        self.assertEqual(resp.status_code, 200)

    def test_list_users_pagination(self):
        """GET /api/admin/users should support limit and offset variables."""
        self._login_admin()
        
        # Test full fetch
        resp = self.client.get("/api/admin/users")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(len(resp.get_json()["users"]), 3)

        # Test pagination limits
        resp_paginated = self.client.get("/api/admin/users?limit=1&offset=1")
        self.assertEqual(resp_paginated.status_code, 200)
        self.assertEqual(len(resp_paginated.get_json()["users"]), 1)

    def test_admin_reset_user_password(self):
        """PUT /api/admin/users/<id>/password should allow an Admin to override a user's password directly."""
        self._login_admin()
        patient_in_db = self.repo.get_by_email("patient@aurascan.ai")
        
        resp = self.client.put(f"/api/admin/users/{patient_in_db.id}/password", json={
            "password": "NewPatientPassword@999"
        })
        self.assertEqual(resp.status_code, 200)

        # Login as patient using newly set credentials to verify update worked
        resp_login = self.client.post("/api/auth/login", json={
            "email": "patient@aurascan.ai",
            "password": "NewPatientPassword@999"
        })
        self.assertEqual(resp_login.status_code, 200)

    def test_export_users_csv(self):
        """GET /api/admin/users/export should stream the full database as a CSV attachment."""
        self._login_admin()
        resp = self.client.get("/api/admin/users/export")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers["Content-Type"], "text/csv")
        csv_data = resp.data.decode("utf-8")
        self.assertIn("Email,Full Name,Role", csv_data)
        self.assertIn("patient@aurascan.ai", csv_data)
        self.assertIn("testadmin@aurascan.ai", csv_data)


if __name__ == "__main__":
    unittest.main()
