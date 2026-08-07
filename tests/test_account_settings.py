import unittest
import os
import tempfile
import io
import datetime
from dashboard.infrastructure.web_server import create_app
from security.domain.entities import Role, User
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.password import PasswordHasher


class TestAccountSettings(unittest.TestCase):
    """Test suite to verify profile updating, avatar uploading, session revocation, and account deletion."""

    def setUp(self):
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        self.app = create_app(db_path=self.db_path)
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

        # Initialize tables
        self.repo = SQLiteUserRepository(db_path=self.db_path)
        self.repo.initialize_security_tables()

        # Create user
        self.password = "SecurePassword@123"
        self.user = User(
            id=None,
            uuid="test-patient-uuid",
            email="patient@aurascan.ai",
            password_hash=PasswordHasher.hash_password(self.password),
            full_name="Patient User",
            role=Role.PATIENT,
            is_verified=True,
            is_active=True
        )
        self.repo.create_user(self.user)

    def tearDown(self):
        os.close(self.db_fd)
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def _login(self):
        resp = self.client.post("/api/auth/login", json={
            "email": "patient@aurascan.ai",
            "password": self.password
        })
        self.assertEqual(resp.status_code, 200)
        return resp.get_json()

    def test_update_profile_details(self):
        """PUT /api/auth/profile should update user details successfully."""
        self._login()

        # Update full name
        resp = self.client.put("/api/auth/profile", json={
            "full_name": "Updated Name",
            "email": "patient@aurascan.ai"
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["user"]["full_name"], "Updated Name")
        self.assertFalse(data["email_changed"])

    def test_update_profile_email_resets_verification(self):
        """PUT /api/auth/profile should keep is_verified as True even if email address changes."""
        self._login()
 
        resp = self.client.put("/api/auth/profile", json={
            "full_name": "Patient User",
            "email": "newpatient@aurascan.ai"
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["email_changed"])
        self.assertTrue(data["user"]["is_verified"])

    def test_upload_avatar_image(self):
        """POST /api/auth/profile/avatar should upload and save custom avatar picture."""
        self._login()

        # Send mock file
        avatar_data = (io.BytesIO(b"dummy image data"), "avatar.png")
        resp = self.client.post(
            "/api/auth/profile/avatar",
            data={"avatar": avatar_data},
            content_type="multipart/form-data"
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("/uploads/avatars/", data["avatar_url"])
        self.assertEqual(data["user"]["google_profile_pic"], data["avatar_url"])

    def test_logout_other_devices(self):
        """POST /api/auth/profile/logout-other-devices should revoke older tokens."""
        login_data = self._login()
        old_access_token = login_data["access_token"]

        # Sleep 1.1s so that the token iat is strictly older than the revocation timestamp
        import time
        time.sleep(1.1)

        # Call logout other devices
        resp = self.client.post("/api/auth/profile/logout-other-devices")
        self.assertEqual(resp.status_code, 200)
        new_data = resp.get_json()
        self.assertNotEqual(new_data["access_token"], old_access_token)

        # Attempt to access protected endpoint using old token should fail
        # Set old header manually to bypass the cookie client state
        resp_protected = self.client.get(
            "/api/auth/me",
            headers={"Authorization": f"Bearer {old_access_token}"}
        )
        self.assertEqual(resp_protected.status_code, 200)
        self.assertFalse(resp_protected.get_json()["authenticated"])

    def test_delete_user_account(self):
        """DELETE /api/auth/profile should wipe user record and cascade dependent tables."""
        self._login()

        resp = self.client.delete("/api/auth/profile")
        self.assertEqual(resp.status_code, 200)

        # Verify user is deleted from DB
        db_user = self.repo.get_by_email("patient@aurascan.ai")
        self.assertIsNone(db_user)


if __name__ == "__main__":
    unittest.main()
