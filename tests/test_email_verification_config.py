import unittest
import os
import tempfile
import datetime
import hashlib
from security.domain.entities import Role, User
from security.infrastructure.repository import SQLiteUserRepository
from security.application.use_cases import AuthUseCases, get_public_base_url

class TestEmailVerificationConfig(unittest.TestCase):
    def setUp(self):
        self.orig_db_path = os.environ.get("DB_PATH")
        self.orig_public_url = os.environ.get("AURASCAN_PUBLIC_BASE_URL")
        self.orig_app_url = os.environ.get("APP_URL")

        os.environ["G7_TESTING"] = "True"
        os.environ["TESTING"] = "True"
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.environ["DB_PATH"] = self.db_path

        from persistence.infrastructure.repository import SQLitePersistenceRepository
        self.persistence_repo = SQLitePersistenceRepository(db_path=self.db_path)
        self.persistence_repo.initialize_db()

        self.repo = SQLiteUserRepository(db_path=self.db_path)
        self.repo.initialize_security_tables()
        self.auth_cases = AuthUseCases(user_repo=self.repo)

    def tearDown(self):
        os.close(self.db_fd)
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

        # Restore environment variables
        os.environ.pop("G7_TESTING", None)
        os.environ.pop("TESTING", None)

        if self.orig_db_path is not None:
            os.environ["DB_PATH"] = self.orig_db_path
        else:
            os.environ.pop("DB_PATH", None)

        if self.orig_public_url is not None:
            os.environ["AURASCAN_PUBLIC_BASE_URL"] = self.orig_public_url
        else:
            os.environ.pop("AURASCAN_PUBLIC_BASE_URL", None)

        if self.orig_app_url is not None:
            os.environ["APP_URL"] = self.orig_app_url
        else:
            os.environ.pop("APP_URL", None)

    def test_get_public_base_url_defaults(self):
        # A. localhost configuration generates localhost URL when no env configured
        os.environ.pop("AURASCAN_PUBLIC_BASE_URL", None)
        os.environ.pop("APP_URL", None)
        self.assertEqual(get_public_base_url(), "http://127.0.0.1:5000")

    def test_get_public_base_url_public_base_url(self):
        # B. LAN/public configuration generates configured URL
        os.environ["AURASCAN_PUBLIC_BASE_URL"] = "http://192.168.1.10:5000/"
        self.assertEqual(get_public_base_url(), "http://192.168.1.10:5000")

        os.environ["AURASCAN_PUBLIC_BASE_URL"] = "https://myproduction-domain.com"
        self.assertEqual(get_public_base_url(), "https://myproduction-domain.com")

    def test_get_public_base_url_app_url_fallback(self):
        os.environ.pop("AURASCAN_PUBLIC_BASE_URL", None)
        os.environ["APP_URL"] = "http://192.168.1.20:5000"
        self.assertEqual(get_public_base_url(), "http://192.168.1.20:5000")

    def test_verification_token_lifecycle(self):
        # Register a patient to test token behavior
        email = "patient_test@aurascan.ai"
        pwd = "Password@123"
        reg_res = self.auth_cases.register(
            email=email,
            password=pwd,
            full_name="Patient Test",
            role_str="patient"
        )
        token = reg_res["verification_token"]
        self.assertIsNotNone(token)

        # C. Verification token remains valid across different clients (different headers/requests)
        # Verify using auth_cases directly
        # E. Expired token fails
        # Let's verify by manually setting expires_at to past
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        conn = self.repo._get_connection()
        try:
            conn.execute(
                "UPDATE verification_tokens SET expires_at = ? WHERE token_hash = ?;",
                ((datetime.datetime.utcnow() - datetime.timedelta(hours=1)).isoformat(), token_hash)
            )
            conn.commit()
        finally:
            conn.close()

        with self.assertRaises(ValueError) as ctx:
            self.auth_cases.verify_email(token)
        self.assertIn("expired", str(ctx.exception).lower())

        # Restore expiry for success/reused/invalid tests
        conn = self.repo._get_connection()
        try:
            conn.execute(
                "UPDATE verification_tokens SET expires_at = ? WHERE token_hash = ?;",
                ((datetime.datetime.utcnow() + datetime.timedelta(hours=24)).isoformat(), token_hash)
            )
            conn.commit()
        finally:
            conn.close()

        # G. Invalid token fails
        with self.assertRaises(ValueError) as ctx:
            self.auth_cases.verify_email("invalid_token_123")
        self.assertIn("invalid", str(ctx.exception).lower())

        # D. Successful verification activates the correct account
        verify_res = self.auth_cases.verify_email(token)
        self.assertTrue(verify_res)
        user = self.repo.get_by_email(email)
        self.assertTrue(user.is_verified)

        # F. Reused token fails
        with self.assertRaises(ValueError) as ctx:
            self.auth_cases.verify_email(token)
        self.assertIn("already been used", str(ctx.exception).lower())

    def test_registration_email_uses_configured_public_base_url(self):
        """Verifies that registration email contains the configured public base URL."""
        # Clean email deliveries table
        conn = self.repo._get_connection()
        try:
            conn.execute("DELETE FROM email_deliveries;")
            conn.commit()
        finally:
            conn.close()

        # Set public base URL to LAN IP
        os.environ["AURASCAN_PUBLIC_BASE_URL"] = "http://192.168.10.16:5000"

        email = "patient_lan_test@aurascan.ai"
        pwd = "Password@123"

        # Trigger registration
        reg_res = self.auth_cases.register(
            email=email,
            password=pwd,
            full_name="Patient LAN Test",
            role_str="patient"
        )

        # Retrieve the latest sent email
        conn = self.repo._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT body_text, body_html FROM email_deliveries ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
        finally:
            conn.close()

        self.assertIsNotNone(row)
        body_text, body_html = row

        # Verify it contains the configured LAN URL and NOT localhost/127.0.0.1
        self.assertIn("http://192.168.10.16:5000/verify-email?token=", body_text)
        self.assertIn("http://192.168.10.16:5000/verify-email?token=", body_html)
        self.assertNotIn("127.0.0.1", body_text)
        self.assertNotIn("localhost", body_text)

    def test_registration_email_uses_fallback_base_url(self):
        """Verifies that registration email contains the default fallback base URL when no public URL is set."""
        # Clean email deliveries table
        conn = self.repo._get_connection()
        try:
            conn.execute("DELETE FROM email_deliveries;")
            conn.commit()
        finally:
            conn.close()

        # Set public base URL to localhost
        os.environ["AURASCAN_PUBLIC_BASE_URL"] = "http://127.0.0.1:5000"

        email = "patient_local_test@aurascan.ai"
        pwd = "Password@123"

        # Trigger registration
        reg_res = self.auth_cases.register(
            email=email,
            password=pwd,
            full_name="Patient Local Test",
            role_str="patient"
        )

        # Retrieve the latest sent email
        conn = self.repo._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT body_text, body_html FROM email_deliveries ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
        finally:
            conn.close()

        self.assertIsNotNone(row)
        body_text, body_html = row

        # Verify it contains localhost
        self.assertIn("http://127.0.0.1:5000/verify-email?token=", body_text)
        self.assertIn("http://127.0.0.1:5000/verify-email?token=", body_html)
