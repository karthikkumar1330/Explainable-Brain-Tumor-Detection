import os
import unittest
from unittest.mock import patch
import pytest
from importlib import reload

import security.application.config as config_module
from security.application.config import AppConfig, AppConfigException
from security.application.use_cases import get_public_base_url


class TestLanConfig(unittest.TestCase):
    """Unit tests validating centralized cross-device LAN configuration behaviors."""

    def setUp(self):
        # Save original environment variables to prevent test cross-contamination
        self.original_env = dict(os.environ)

    def tearDown(self):
        # Restore environment
        os.environ.clear()
        os.environ.update(self.original_env)

    def test_localhost_default_configuration(self):
        """Verifies default values are selected when no configuration env variables are present."""
        env_removals = ["AURASCAN_HOST", "HOST", "AURASCAN_PUBLIC_BASE_URL", "APP_URL", "REST_PORT", "DASHBOARD_PORT", "FASTAPI_URL"]
        for key in env_removals:
            if key in os.environ:
                del os.environ[key]

        cfg = AppConfig()
        self.assertEqual(cfg.host, "127.0.0.1")
        self.assertEqual(cfg.dashboard_port, 5000)
        self.assertEqual(cfg.api_port, 8000)
        self.assertEqual(cfg.public_base_url, "http://127.0.0.1:5000")
        self.assertEqual(cfg.fastapi_url, "http://127.0.0.1:8000")

    def test_explicit_env_overrides(self):
        """Verifies custom env overrides are loaded properly for LAN deployments."""
        os.environ["AURASCAN_HOST"] = "0.0.0.0"
        os.environ["AURASCAN_PUBLIC_BASE_URL"] = "http://192.168.10.16:5000"
        os.environ["REST_PORT"] = "9000"
        os.environ["DASHBOARD_PORT"] = "6000"

        cfg = AppConfig()
        self.assertEqual(cfg.host, "0.0.0.0")
        self.assertEqual(cfg.dashboard_port, 6000)
        self.assertEqual(cfg.api_port, 9000)
        self.assertEqual(cfg.public_base_url, "http://192.168.10.16:5000")
        self.assertEqual(cfg.fastapi_url, "http://192.168.10.16:9000")

    def test_app_url_fallback(self):
        """Verifies legacy APP_URL environment variable is respected when AURASCAN_PUBLIC_BASE_URL is missing."""
        if "AURASCAN_PUBLIC_BASE_URL" in os.environ:
            del os.environ["AURASCAN_PUBLIC_BASE_URL"]
        os.environ["APP_URL"] = "http://192.168.10.10:5000/"

        cfg = AppConfig()
        # Trailing slash must be normalized
        self.assertEqual(cfg.public_base_url, "http://192.168.10.10:5000")
        self.assertEqual(cfg.fastapi_url, "http://192.168.10.10:8000")

    def test_trailing_slash_normalization(self):
        """Verifies trailing slashes on public URLs are correctly stripped."""
        os.environ["AURASCAN_PUBLIC_BASE_URL"] = "http://mri-server.local:5000/"
        cfg = AppConfig()
        self.assertEqual(cfg.public_base_url, "http://mri-server.local:5000")
        self.assertEqual(cfg.fastapi_url, "http://mri-server.local:8000")

    def test_malformed_url_rejection(self):
        """Verifies obviously malformed and placeholder public URLs are rejected."""
        invalid_urls = [
            "http://",
            "192.168.10.16:5000",
            "ftp://192.168.1.1",
            "http://<server-lan-ip>:5000",
            "http://{server-ip}:5000",
            "http://[server-ip]:5000",
        ]
        for url in invalid_urls:
            os.environ["AURASCAN_PUBLIC_BASE_URL"] = url
            with self.assertRaises(AppConfigException):
                AppConfig()

    def test_verification_url_never_uses_host_header(self):
        """Verifies get_public_base_url returns the configured base URL, not deriving from headers."""
        os.environ["AURASCAN_PUBLIC_BASE_URL"] = "http://192.168.10.16:5000"

        # Reload configuration singleton to absorb mock environment
        reload(config_module)

        url = get_public_base_url()
        self.assertEqual(url, "http://192.168.10.16:5000")
        self.assertNotEqual(url, "http://127.0.0.1:5000")

    def test_no_secret_logging(self):
        """Verifies get_non_secret_settings exposes only safe variables, never keys or passwords."""
        os.environ["SMTP_PASSWORD"] = "secret-pass"
        os.environ["PII_ENCRYPTION_KEY"] = "secret-key"
        cfg = AppConfig()
        settings = cfg.get_non_secret_settings()

        self.assertNotIn("SMTP_PASSWORD", settings)
        self.assertNotIn("PII_ENCRYPTION_KEY", settings)
        self.assertNotIn("secret-pass", settings.values())
        self.assertNotIn("secret-key", settings.values())

        # Ensure bind variables are outputted
        self.assertIn("AURASCAN_HOST", settings)
        self.assertIn("AURASCAN_PUBLIC_BASE_URL", settings)

    def test_project_env_path(self):
        """Verifies that the resolved env_path points to the absolute project root .env."""
        import security.application.config as config_module
        expected_suffix = os.path.normpath("UNeXt-pytorch/.env")
        resolved_path = os.path.normpath(str(config_module.env_path))
        self.assertTrue(resolved_path.endswith(expected_suffix), f"Path {resolved_path} does not end with {expected_suffix}")

    def test_flask_fastapi_hosts_use_config(self):
        """Verifies that run_dashboard.py and run_api.py default hosts use the configured AppConfig host."""
        import run_dashboard
        import run_api

        # Test under custom environment
        with patch.dict(os.environ, {"AURASCAN_HOST": "0.0.0.0"}):
            # Reload to re-parse defaults
            reload(run_dashboard)
            reload(run_api)

            with patch("sys.argv", ["run_dashboard.py"]):
                dashboard_args = run_dashboard.parse_args()
            with patch("sys.argv", ["run_api.py"]):
                api_args = run_api.parse_args()

            self.assertEqual(dashboard_args.host, "0.0.0.0")
            self.assertEqual(api_args.host, "0.0.0.0")
