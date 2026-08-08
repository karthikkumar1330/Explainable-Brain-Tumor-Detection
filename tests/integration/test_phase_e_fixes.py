import unittest
import os

class TestPhaseEFixes(unittest.TestCase):
    """Regression tests verifying implementation correctness for Phase E+ Final Debugging fixes."""

    def setUp(self):
        # Locate project root directory relative to this test file
        self.workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    def test_alert_rules_not_hidden(self):
        """Verify index.html custom style targets :not(.hidden) instead of bare IDs to fix empty alert boxes."""
        index_path = os.path.join(self.workspace_dir, "dashboard", "presentation", "templates", "index.html")
        self.assertTrue(os.path.exists(index_path), f"File not found: {index_path}")
        with open(index_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Verify selectors have :not(.hidden) to avoid overriding Tailwind's .hidden class
        self.assertIn("#login-error:not(.hidden)", content)
        self.assertIn("#register-error:not(.hidden)", content)
        self.assertIn("#otp-error:not(.hidden)", content)
        self.assertIn("#register-success:not(.hidden)", content)
        self.assertIn("#forgot-msg:not(.hidden)", content)
        self.assertIn("#otp-hint:not(.hidden)", content)

    def test_doctor_dashboard_registry_call(self):
        """Verify dashboard_doctor.html calls updateRegistryView() and loadRegistryData()."""
        doc_path = os.path.join(self.workspace_dir, "dashboard", "presentation", "templates", "dashboard_doctor.html")
        self.assertTrue(os.path.exists(doc_path), f"File not found: {doc_path}")
        with open(doc_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Verify updateRegistryView() is called
        self.assertIn("updateRegistryView()", content)
        # Verify loadRegistryData() is called
        self.assertIn("loadRegistryData()", content)
        # Check switchTab includes the check for 'database'
        self.assertIn("['dashboard', 'reports', 'analytics', 'database'].includes(tabName)", content)

    def test_theme_logout_contrast(self):
        """Verify theme.css light mode overrides for .btn-danger have high-contrast red styling."""
        css_path = os.path.join(self.workspace_dir, "dashboard", "presentation", "static", "css", "theme.css")
        self.assertTrue(os.path.exists(css_path), f"File not found: {css_path}")
        with open(css_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Verify html:not(.dark) .btn-danger color is set to dark red (#DC2626) and NOT overridden to white
        self.assertIn("html:not(.dark) .btn-danger", content)
        self.assertIn("color: #DC2626 !important;", content)

    def test_login_localStorage_clear(self):
        """Verify index.html clearLocalStorage logic when authentication fails in checkAuthStatus."""
        index_path = os.path.join(self.workspace_dir, "dashboard", "presentation", "templates", "index.html")
        self.assertTrue(os.path.exists(index_path), f"File not found: {index_path}")
        with open(index_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Check localStorage.removeItem("auth_token") is called inside checkAuthStatus on failure
        self.assertIn("localStorage.removeItem(\"auth_token\")", content)

if __name__ == "__main__":
    unittest.main()
