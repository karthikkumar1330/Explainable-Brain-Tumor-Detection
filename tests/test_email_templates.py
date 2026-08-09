import unittest
from clinical_reporting.presentation.email_templates import (
    EmailTemplateRenderer,
    validate_url,
    render_cta_button,
    render_info_card,
    render_status_badge
)


class TestEmailTemplates(unittest.TestCase):
    """Phase G3 Presentation & Template rendering validation tests."""

    def test_01_base_template_renders(self) -> None:
        """Verify that basic rendering produces both HTML and Plain-Text with required strings."""
        html, text = EmailTemplateRenderer.render_generic_notification(
            subject="AuraScan Test",
            title="Scan Summary",
            message_body="Welcome to the scan portal."
        )
        self.assertIn("AuraScan Test", html)
        self.assertIn("Scan Summary", html)
        self.assertIn("Welcome to the scan portal.", html)
        
        self.assertIn("Scan Summary", text)
        self.assertIn("Welcome to the scan portal.", text)

    def test_02_required_variables_render_correctly(self) -> None:
        """Verify message elements render exactly as supplied."""
        html, text = EmailTemplateRenderer.render_generic_notification(
            subject="SubjectX",
            title="TitleY",
            message_body="BodyZ"
        )
        self.assertIn("SubjectX", html)
        self.assertIn("TitleY", html)
        self.assertIn("BodyZ", html)

    def test_03_missing_optional_variables_do_not_crash(self) -> None:
        """Verify missing optional parameters (badge, CTA, card) resolve to safe defaults without crashes."""
        try:
            html, text = EmailTemplateRenderer.render_generic_notification(
                subject="Subj",
                title=None,
                message_body="Msg"
            )
            # Verify no button, badge, or card tags exist when not configured
            self.assertNotIn("display: inline-block; padding: 12px 24px;", html)
            self.assertNotIn("display: inline-block; padding: 4px 8px;", html)
            self.assertNotIn("background-color: #f8fafc; border: 1px solid #e2e8f0;", html)
        except Exception as e:
            self.fail(f"Renderer crashed on missing optional parameters: {e}")

    def test_04_html_escaping_works(self) -> None:
        """Verify standard HTML auto-escaping rules for basic tags."""
        html, text = EmailTemplateRenderer.render_generic_notification(
            subject="Escape test",
            title="Test",
            message_body="Safe message <script>alert(1)</script>"
        )
        # Verify tag markers are fully escaped
        self.assertNotIn("<script>", html)
        self.assertIn("&lt;script&gt;alert(1)&lt;/script&gt;", html)

    def test_05_user_supplied_html_is_escaped(self) -> None:
        """Verify raw HTML formatting attempts from user variables are strictly escaped."""
        html, text = EmailTemplateRenderer.render_generic_notification(
            subject="Subj",
            title="<h1>Unsafe Title</h1>",
            message_body="<div style='color: red;'>Unsafe Body</div>"
        )
        self.assertNotIn("<h1>", html)
        self.assertNotIn("<div style='color: red;'>", html)
        self.assertIn("&lt;h1&gt;Unsafe Title&lt;/h1&gt;", html)
        self.assertIn("&lt;div style=&#39;color: red;&#39;&gt;Unsafe Body&lt;/div&gt;", html)

    def test_06_dangerous_url_schemes_are_rejected(self) -> None:
        """Verify javascript, data, vbscript, and file schemes raise ValueError."""
        unsafe_urls = [
            "javascript:alert(1)",
            "data:text/html,<script>alert(1)</script>",
            "vbscript:msgbox(1)",
            "file:///etc/passwd",
            "javascript://alert(1)"
        ]
        for url in unsafe_urls:
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    validate_url(url)

    def test_07_https_url_works(self) -> None:
        """Verify that standard safe URL paths are accepted."""
        safe_urls = [
            "https://portal.aurascan.ai/reports/123",
            "http://127.0.0.1:5000/dashboard",
            "/dashboard/reports/1"
        ]
        for url in safe_urls:
            with self.subTest(url=url):
                try:
                    validate_url(url)
                except ValueError:
                    self.fail(f"validate_url rejected a valid URL: {url}")

    def test_08_plain_text_fallback_renders(self) -> None:
        """Verify plain text fallbacks avoid HTML tags while preserving CTA links and values."""
        html, text = EmailTemplateRenderer.render_generic_notification(
            subject="Subj",
            title="Clinical Notice",
            message_body="Diagnosis finalized.",
            cta_text="View Report",
            cta_url="https://example.com/rep",
            card_items=[("Patient", "John Doe"), ("Severity", "High")],
            badge_label="Critical Alert",
            badge_type="danger"
        )
        # Verify no HTML tags in text
        self.assertNotIn("<html>", text)
        self.assertNotIn("</div>", text)
        self.assertNotIn("</a>", text)
        
        # Verify info card items and link details are preserved in text
        self.assertIn("Clinical Notice", text)
        self.assertIn("Diagnosis finalized.", text)
        self.assertIn("Status: [CRITICAL ALERT]", text)
        self.assertIn("- Patient: John Doe", text)
        self.assertIn("- Severity: High", text)
        self.assertIn("View Report: https://example.com/rep", text)

    def test_09_html_contains_expected_structural_elements(self) -> None:
        """Verify base HTML structure layout details are compiled correctly."""
        html, text = EmailTemplateRenderer.render_generic_notification(
            subject="Struct",
            title="Title",
            message_body="Body"
        )
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("<html lang=\"en\">", html)
        self.assertIn("<meta charset=\"utf-8\">", html)
        self.assertIn("<body", html)

    def test_10_header_exists(self) -> None:
        """Verify the healthcare portal header exists in the HTML body."""
        html, text = EmailTemplateRenderer.render_generic_notification(
            subject="S", title="T", message_body="B"
        )
        self.assertIn("AuraScan AI", html)
        self.assertIn("border-bottom: 3px solid #0284c7;", html)

    def test_11_footer_exists(self) -> None:
        """Verify that footer elements and clinical confidentiality notices are present."""
        html, text = EmailTemplateRenderer.render_generic_notification(
            subject="S", title="T", message_body="B"
        )
        self.assertIn("CONFIDENTIALITY NOTICE:", html)
        self.assertIn("AuraScan AI Platform", html)

    def test_12_cta_rendering_works(self) -> None:
        """Verify CTA buttons render properly and prevent quote breakouts."""
        html, text = EmailTemplateRenderer.render_generic_notification(
            subject="S",
            title="T",
            message_body="B",
            cta_text="Press Here",
            cta_url="https://example.com/test?param=1"
        )
        self.assertIn("Press Here", html)
        self.assertIn("https://example.com/test?param=1", html)
        self.assertIn("target=\"_blank\"", html)

        # Quote breakout test
        with self.assertRaises(ValueError):
            render_cta_button("Click", "https://example.com\" onclick=\"alert(1)")

    def test_13_information_card_rendering_works(self) -> None:
        """Verify info cards and lists compile into visual table format."""
        items = [("Patient ID", "9940"), ("Tumor Size", "2.4cm")]
        html, text = EmailTemplateRenderer.render_generic_notification(
            subject="S",
            title="T",
            message_body="B",
            card_items=items
        )
        self.assertIn("Patient ID:", html)
        self.assertIn("9940", html)
        self.assertIn("Tumor Size:", html)
        self.assertIn("2.4cm", html)

    def test_14_no_smtp_credentials_appear_in_output(self) -> None:
        """Verify credentials and passwords never leak into templates."""
        html, text = EmailTemplateRenderer.render_generic_notification(
            subject="S", title="T", message_body="B"
        )
        self.assertNotIn("SMTP_PASSWORD", html)
        self.assertNotIn("smtp_password", html)

    def test_15_no_internal_filesystem_paths_appear(self) -> None:
        """Verify local system files paths are completely absent."""
        html, text = EmailTemplateRenderer.render_generic_notification(
            subject="S", title="T", message_body="B"
        )
        self.assertNotIn("clinical_reporting/presentation", html)
        self.assertNotIn("outputs/clinical_reports", html)

    def test_16_template_renderer_handles_malformed_context_safely(self) -> None:
        """Verify passing unexpected types (such as None or numbers) is handled safely."""
        try:
            # None title and badge inputs
            html, text = EmailTemplateRenderer.render_generic_notification(
                subject="Test",
                title=None,
                message_body="Body",
                card_items=[("Count", 123)]  # Integer value
            )
            self.assertIn("Count:", html)
            self.assertIn("123", html)
        except Exception as e:
            self.fail(f"Template renderer failed on atypical types in context: {e}")


if __name__ == "__main__":
    unittest.main()
