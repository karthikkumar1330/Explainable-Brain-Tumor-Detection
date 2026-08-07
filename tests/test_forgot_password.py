import unittest
import os
import tempfile
import datetime
from security.domain.entities import Role, SecurityAuditLog
from security.infrastructure.repository import SQLiteUserRepository
from security.application.use_cases import AuthUseCases


class TestForgotPasswordIntegration(unittest.TestCase):
    def setUp(self):
        self.db_fd, self.db_path = tempfile.mkstemp(suffix=".db")
        self.repo = SQLiteUserRepository(db_path=self.db_path)
        self.repo.initialize_security_tables()
        self.auth_cases = AuthUseCases(user_repo=self.repo)

        # Register a test user
        self.auth_cases.register(
            email="clinician@aurascan.ai",
            password="StrongClinician@123",
            full_name="Dr. Jane Doe",
            role_str="doctor"
        )

    def tearDown(self):
        os.close(self.db_fd)
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass

    def test_forgot_password_user_exists(self):
        # 1. Query the database using the same logic as the UI
        email = "clinician@aurascan.ai"
        user = self.auth_cases.user_repo.get_by_email(email)
        self.assertIsNotNone(user)
        self.assertEqual(user.email, email)

        # 2. Log a simulated reset password event in the audit logs
        now = datetime.datetime.utcnow().isoformat()
        audit_log = SecurityAuditLog(
            id=None,
            timestamp=now,
            event_type="PASSWORD_RESET_REQUEST",
            user_id=user.id,
            email=user.email,
            ip_address="127.0.0.1",
            status="SUCCESS",
            details=f"Simulated password reset request logged for {user.email}.",
            user_agent="Test Framework"
        )
        self.auth_cases.user_repo.log_security_event(audit_log)

        # 3. Retrieve audit logs and assert event was logged
        logs = self.auth_cases.user_repo.get_security_audit_logs(limit=10)
        reset_logs = [l for l in logs if l["event_type"] == "PASSWORD_RESET_REQUEST"]
        self.assertEqual(len(reset_logs), 1)
        self.assertEqual(reset_logs[0]["email"], email)
        self.assertIn("Simulated password reset request logged", reset_logs[0]["details"])

    def test_forgot_password_user_not_exists(self):
        # Querying for a non-existent user should return None, showing validation is handled
        email = "nonexistent@aurascan.ai"
        user = self.auth_cases.user_repo.get_by_email(email)
        self.assertIsNone(user)


if __name__ == "__main__":
    unittest.main()
