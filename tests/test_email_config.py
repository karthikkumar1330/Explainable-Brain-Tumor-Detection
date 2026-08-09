import os
import unittest
from unittest.mock import patch
from pathlib import Path
from clinical_reporting.infrastructure.email_config import EmailConfig, EmailConfigException


class TestEmailConfig(unittest.TestCase):
    """Phase G2 Configuration Validation and Environment isolation tests."""

    @patch.dict(os.environ, {
        "SMTP_HOST": "smtp.example.com",
        "SMTP_PORT": "587",
        "SMTP_USERNAME": "user@example.com",
        "SMTP_PASSWORD": "SecretPassword123!",
        "EMAIL_FROM": "noreply@example.com",
        "EMAIL_REPLY_TO": "support@example.com",
        "EMAIL_TIMEOUT": "5",
        "SMTP_USE_SSL": "False"
    })
    def test_01_valid_configuration(self) -> None:
        """Verify that a complete and correct configuration passes validation."""
        config = EmailConfig()
        try:
            config.validate()
        except EmailConfigException as e:
            self.fail(f"validate failed unexpectedly: {e}")
        
        self.assertEqual(config.smtp_host, "smtp.example.com")
        self.assertEqual(config.smtp_port, 587)
        self.assertEqual(config.smtp_username, "user@example.com")
        self.assertEqual(config.smtp_password, "SecretPassword123!")
        self.assertEqual(config.email_from, "noreply@example.com")
        self.assertEqual(config.email_reply_to, "support@example.com")
        self.assertEqual(config.email_timeout, 5.0)
        self.assertFalse(config.use_ssl)

    @patch.dict(os.environ, {
        "SMTP_HOST": "",
        "SMTP_PORT": "587",
        "EMAIL_FROM": "noreply@example.com"
    })
    def test_02_missing_smtp_host(self) -> None:
        """Verify that missing SMTP_HOST means inactive service and validation is skipped."""
        config = EmailConfig()
        try:
            config.validate()  # Should not raise because host is empty
        except EmailConfigException as e:
            self.fail(f"validate raised config exception when host is empty: {e}")

    @patch.dict(os.environ, {
        "SMTP_HOST": "smtp.example.com",
        "SMTP_PORT": "-10",
        "EMAIL_FROM": "noreply@example.com"
    })
    def test_03_invalid_smtp_port(self) -> None:
        """Verify that out-of-range port values raise Configuration error."""
        config = EmailConfig()
        with self.assertRaises(EmailConfigException):
            config.validate()

        with patch.dict(os.environ, {"SMTP_PORT": "999999"}):
            config = EmailConfig()
            with self.assertRaises(EmailConfigException):
                config.validate()

    @patch.dict(os.environ, {
        "SMTP_HOST": "smtp.example.com",
        "SMTP_PORT": "587",
        "EMAIL_TIMEOUT": "-5.0",
        "EMAIL_FROM": "noreply@example.com"
    })
    def test_04_invalid_timeout(self) -> None:
        """Verify that negative or zero timeouts are rejected."""
        config = EmailConfig()
        with self.assertRaises(EmailConfigException):
            config.validate()

        with patch.dict(os.environ, {"EMAIL_TIMEOUT": "0"}):
            config = EmailConfig()
            with self.assertRaises(EmailConfigException):
                config.validate()

    @patch.dict(os.environ, {
        "SMTP_HOST": "smtp.example.com",
        "SMTP_PORT": "587",
        "EMAIL_FROM": ""
    })
    def test_05_missing_email_from(self) -> None:
        """Verify that EMAIL_FROM is required when email service is active."""
        config = EmailConfig()
        with self.assertRaises(EmailConfigException):
            config.validate()

    @patch.dict(os.environ, {
        "SMTP_HOST": "smtp.example.com",
        "SMTP_PORT": "587",
        "EMAIL_FROM": "invalid-address"
    })
    def test_06_invalid_email_from(self) -> None:
        """Verify that malformed sender address is rejected."""
        config = EmailConfig()
        with self.assertRaises(EmailConfigException):
            config.validate()

    @patch.dict(os.environ, {
        "SMTP_HOST": "smtp.example.com",
        "SMTP_PORT": "587",
        "EMAIL_FROM": "sender@example.com",
        "EMAIL_REPLY_TO": "invalid-reply-to"
    })
    def test_07_invalid_email_reply_to(self) -> None:
        """Verify that malformed reply-to address is rejected."""
        config = EmailConfig()
        with self.assertRaises(EmailConfigException):
            config.validate()

    @patch.dict(os.environ, {
        "SMTP_HOST": "smtp.example.com",
        "SMTP_PORT": "587",
        "SMTP_USERNAME": "user@example.com",
        "SMTP_PASSWORD": "",
        "EMAIL_FROM": "sender@example.com"
    })
    def test_08_missing_password_when_required(self) -> None:
        """Verify that SMTP_PASSWORD is required when username is configured."""
        config = EmailConfig()
        with self.assertRaises(EmailConfigException):
            config.validate()

    @patch.dict(os.environ, {
        "SMTP_HOST": "smtp.example.com",
        "SMTP_PORT": "465",
        "EMAIL_FROM": "sender@example.com"
    })
    def test_09_ssl_configuration(self) -> None:
        """Verify implicit SSL resolution for port 465."""
        config = EmailConfig()
        self.assertTrue(config.use_ssl)

    @patch.dict(os.environ, {
        "SMTP_HOST": "smtp.example.com",
        "SMTP_PORT": "587",
        "EMAIL_FROM": "sender@example.com"
    })
    def test_10_starttls_configuration(self) -> None:
        """Verify default STARTTLS usage on standard port 587."""
        config = EmailConfig()
        self.assertFalse(config.use_ssl)
        self.assertEqual(config.smtp_port, 587)

    @patch.dict(os.environ, {
        "SMTP_HOST": "",
        "SMTP_PORT": "",
        "SMTP_USERNAME": "",
        "SMTP_PASSWORD": "",
        "EMAIL_FROM": ""
    })
    def test_11_test_environment_without_credentials(self) -> None:
        """Verify configuration does not fail if all mail variables are empty (e.g. mock test runs)."""
        config = EmailConfig()
        try:
            config.validate()
        except EmailConfigException as e:
            self.fail(f"validate failed on unconfigured environment: {e}")

    @patch.dict(os.environ, {
        "SMTP_HOST": "smtp.example.com",
        "SMTP_PORT": "587",
        "SMTP_USERNAME": "user@example.com",
        "SMTP_PASSWORD": "SuperSecretPassword123!",
        "EMAIL_FROM": "sender@example.com"
    })
    def test_12_secret_redaction(self) -> None:
        """Verify that SMTP_PASSWORD is not logged or exposed by config inspection."""
        config = EmailConfig()
        settings = config.get_non_secret_settings()
        
        # Verify password is not in keys
        self.assertNotIn("smtp_password", settings)
        self.assertNotIn("password", settings)
        
        # Verify no secret value matches in values
        for val in settings.values():
            self.assertNotEqual(val, "SuperSecretPassword123!")

        # Verify that validation exceptions do not contain the password
        with patch.dict(os.environ, {"SMTP_PASSWORD": "SecretPassword", "SMTP_USERNAME": "user", "SMTP_PASSWORD": ""}):
            config_err = EmailConfig()
            try:
                config_err.validate()
            except EmailConfigException as ex:
                self.assertNotIn("SecretPassword", str(ex))

    @patch.dict(os.environ, {
        "SMTP_HOST": "smtp.env.test",
        "SMTP_PORT": "2525",
        "EMAIL_FROM": "sender@env.test"
    })
    def test_13_environment_variable_loading(self) -> None:
        """Verify standard loading patterns from environment variables."""
        config = EmailConfig()
        self.assertEqual(config.smtp_host, "smtp.env.test")
        self.assertEqual(config.smtp_port, 2525)
        self.assertEqual(config.email_from, "sender@env.test")

    def test_14_env_example_correctness(self) -> None:
        """Verify that .env.example contains all required email variables."""
        env_example_path = Path(__file__).parent.parent / ".env.example"
        self.assertTrue(env_example_path.exists())
        
        content = env_example_path.read_text()
        required_vars = [
            "EMAIL_PROVIDER",
            "SMTP_HOST",
            "SMTP_PORT",
            "SMTP_USERNAME",
            "SMTP_PASSWORD",
            "EMAIL_FROM",
            "EMAIL_REPLY_TO",
            "EMAIL_TIMEOUT",
            "SMTP_USE_SSL",
            "ENV_MODE"
        ]
        for var in required_vars:
            self.assertIn(var, content, f"{var} is missing from .env.example template.")

    @patch.dict(os.environ, {
        "SMTP_HOST": "smtp.example.com",
        "SMTP_PORT": "25",
        "EMAIL_FROM": "sender@example.com",
        "ENV_MODE": "production",
        "SMTP_USE_SSL": "False"
    })
    def test_15_production_insecure_configuration_rejected(self) -> None:
        """Verify that insecure connections are rejected in production environment."""
        config = EmailConfig()
        with self.assertRaises(EmailConfigException):
            config.validate()

        # Should pass if ENV_MODE is set to development
        with patch.dict(os.environ, {"ENV_MODE": "development"}):
            config_dev = EmailConfig()
            try:
                config_dev.validate()
            except EmailConfigException as e:
                self.fail(f"validate failed on development environment: {e}")


if __name__ == "__main__":
    unittest.main()
