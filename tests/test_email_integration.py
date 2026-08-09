import os
import io
import re
import ssl
import smtplib
import unittest
import tempfile
import logging
from unittest.mock import MagicMock, patch
from pathlib import Path

from clinical_reporting.infrastructure.email_service import (
    EmailService,
    EmailServiceException,
    InvalidAddressException,
    ConfigurationException,
    ConnectionException,
    AuthenticationException,
    AttachmentException,
    validate_email_address,
    validate_subject
)


class TestEmailService(unittest.TestCase):
    """Phase G1 Automated unit, mock, and security compliance tests for EmailService."""

    def setUp(self) -> None:
        self.smtp_host = "smtp.example.com"
        self.smtp_port = 587
        self.email_from = "sender@example.com"
        self.to_email = "receiver@example.com"
        self.subject = "Test Clinical Report"
        self.body_text = "This is a plain text clinical report."
        self.body_html = "<h1>This is a clinical report</h1>"

    def test_01_valid_recipient(self) -> None:
        """Verify address validation passes for typical email patterns."""
        valid_emails = [
            "test@example.com",
            "first.last@domain.co.uk",
            "user+box@sub.domain.org",
            "123@numbers.info"
        ]
        for email in valid_emails:
            with self.subTest(email=email):
                try:
                    validate_email_address(email)
                except InvalidAddressException:
                    self.fail(f"validate_email_address failed unexpectedly on valid email: {email}")

    def test_02_invalid_recipient(self) -> None:
        """Verify malformed email addresses are rejected."""
        invalid_emails = [
            "",
            "no_at_sign.com",
            "user@",
            "@domain.com",
            "user@domain",
            "user @domain.com",
            "user@domain..com"
        ]
        for email in invalid_emails:
            with self.subTest(email=email):
                with self.assertRaises(InvalidAddressException):
                    validate_email_address(email)

    def test_03_missing_configuration(self) -> None:
        """Verify connect throws ConfigurationException if config is missing."""
        # Missing host
        service1 = EmailService(smtp_host=None, smtp_port=587, email_from="sender@example.com")
        with self.assertRaises(ConfigurationException):
            service1.connect()

        # Missing port
        service2 = EmailService(smtp_host="smtp.example.com", smtp_port=None, email_from="sender@example.com")
        with self.assertRaises(ConfigurationException):
            service2.connect()

        # Missing sender
        service3 = EmailService(smtp_host="smtp.example.com", smtp_port=587, email_from=None)
        with self.assertRaises(ConfigurationException):
            service3.connect()

    def test_04_tls_configuration(self) -> None:
        """Verify implicit SSL/TLS and STARTTLS port resolution."""
        # 1. Port 465 resolves to implicit SSL/TLS
        with patch("smtplib.SMTP_SSL") as mock_smtp_ssl:
            service = EmailService(smtp_host="smtp.example.com", smtp_port=465, email_from="sender@example.com")
            service.connect()
            mock_smtp_ssl.assert_called_once()
            self.assertTrue(service.use_ssl)

        # 2. Port 587 uses standard SMTP with STARTTLS
        with patch("smtplib.SMTP") as mock_smtp:
            mock_conn = MagicMock()
            mock_conn.has_ext.return_value = True
            mock_smtp.return_value = mock_conn
            service = EmailService(smtp_host="smtp.example.com", smtp_port=587, email_from="sender@example.com")
            service.connect()
            mock_smtp.assert_called_once()
            mock_conn.starttls.assert_called_once()
            self.assertFalse(service.use_ssl)

    def test_05_smtp_authentication_failure(self) -> None:
        """Verify SMTP authentication failure raises AuthenticationException."""
        with patch("smtplib.SMTP") as mock_smtp:
            mock_conn = MagicMock()
            mock_conn.login.side_effect = smtplib.SMTPAuthenticationError(535, "Authentication failed")
            mock_smtp.return_value = mock_conn
            service = EmailService(
                smtp_host="smtp.example.com",
                smtp_port=587,
                smtp_username="user",
                smtp_password="pwd",
                email_from="sender@example.com"
            )
            with self.assertRaises(AuthenticationException):
                service.connect()

    def test_06_smtp_connection_failure(self) -> None:
        """Verify connection errors raise ConnectionException."""
        with patch("smtplib.SMTP", side_effect=ConnectionRefusedError("SMTP server down")):
            service = EmailService(smtp_host="smtp.example.com", smtp_port=587, email_from="sender@example.com")
            with self.assertRaises(ConnectionException):
                service.connect()

    def test_07_smtp_timeout(self) -> None:
        """Verify timeout values are set correctly on the SMTP transport."""
        with patch("smtplib.SMTP") as mock_smtp:
            mock_conn = MagicMock()
            mock_smtp.return_value = mock_conn
            service = EmailService(
                smtp_host="smtp.example.com",
                smtp_port=587,
                email_from="sender@example.com",
                email_timeout=4.5
            )
            service.connect()
            mock_smtp.assert_called_with(host="smtp.example.com", port=587, timeout=4.5)

    def test_08_plain_text_message_construction(self) -> None:
        """Verify MIME message format with plain text body only."""
        with patch("smtplib.SMTP") as mock_smtp:
            mock_conn = MagicMock()
            mock_smtp.return_value = mock_conn
            service = EmailService(smtp_host="smtp.example.com", smtp_port=587, email_from="sender@example.com")
            service.send(to_email="rec@example.com", subject="Hi", body_text="Hello plain text")
            
            args, kwargs = mock_conn.send_message.call_args
            sent_msg = args[0]
            self.assertEqual(sent_msg["To"], "rec@example.com")
            self.assertEqual(sent_msg["From"], "sender@example.com")
            self.assertEqual(sent_msg["Subject"], "Hi")
            self.assertEqual(sent_msg.get_content().strip(), "Hello plain text")
            self.assertFalse(sent_msg.is_multipart())

    def test_09_html_plus_plain_text_message_construction(self) -> None:
        """Verify MIME message alternative multipart construction with HTML body."""
        with patch("smtplib.SMTP") as mock_smtp:
            mock_conn = MagicMock()
            mock_smtp.return_value = mock_conn
            service = EmailService(smtp_host="smtp.example.com", smtp_port=587, email_from="sender@example.com")
            service.send(
                to_email="rec@example.com",
                subject="Hi",
                body_text="Hello plain text",
                body_html="<p>Hello HTML</p>"
            )
            
            args, kwargs = mock_conn.send_message.call_args
            sent_msg = args[0]
            self.assertTrue(sent_msg.is_multipart())
            parts = list(sent_msg.iter_parts())
            self.assertEqual(parts[0].get_content().strip(), "Hello plain text")
            self.assertEqual(parts[1].get_content().strip(), "<p>Hello HTML</p>")

    def test_10_attachment_construction(self) -> None:
        """Verify regular file loading and safe attachment serialization."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(b"Dummy PDF Content")
            tmp_path = tmp.name
        
        try:
            with patch("smtplib.SMTP") as mock_smtp:
                mock_conn = MagicMock()
                mock_smtp.return_value = mock_conn
                service = EmailService(smtp_host="smtp.example.com", smtp_port=587, email_from="sender@example.com")
                service.send(
                    to_email="rec@example.com",
                    subject="Hi",
                    body_text="Body",
                    attachment_path=tmp_path
                )
                
                args, kwargs = mock_conn.send_message.call_args
                sent_msg = args[0]
                attachments = list(sent_msg.iter_attachments())
                self.assertEqual(len(attachments), 1)
                self.assertEqual(attachments[0].get_filename(), os.path.basename(tmp_path))
                self.assertEqual(attachments[0].get_content(), b"Dummy PDF Content")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_11_missing_attachment(self) -> None:
        """Verify send raises AttachmentException if attachment file does not exist."""
        service = EmailService(smtp_host="smtp.example.com", smtp_port=587, email_from="sender@example.com")
        with self.assertRaises(AttachmentException):
            service.send(
                to_email="rec@example.com",
                subject="Hi",
                body_text="Body",
                attachment_path="nonexistent_report_file.pdf"
            )

    def test_12_invalid_attachment_path(self) -> None:
        """Verify directory paths are rejected as invalid attachments."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            service = EmailService(smtp_host="smtp.example.com", smtp_port=587, email_from="sender@example.com")
            with self.assertRaises(AttachmentException):
                service.send(
                    to_email="rec@example.com",
                    subject="Hi",
                    body_text="Body",
                    attachment_path=tmp_dir
                )

    def test_13_newline_header_injection_rejection(self) -> None:
        """Verify headers (subject, to, from, reply-to) block newline injection attempts."""
        service = EmailService(smtp_host="smtp.example.com", smtp_port=587, email_from="sender@example.com")

        # Recipient newline
        with self.assertRaises(InvalidAddressException):
            service.send(to_email="rec@example.com\nSubject: Inject", subject="Hi", body_text="Body")

        # Subject newline
        with self.assertRaises(EmailServiceException):
            service.send(to_email="rec@example.com", subject="Hi\r\nBcc: spy@example.com", body_text="Body")

        # Sender newline
        with self.assertRaises(InvalidAddressException):
            service = EmailService(smtp_host="smtp.example.com", smtp_port=587, email_from="sender@example.com\nCC: spy@example.com")
            service.send(to_email="rec@example.com", subject="Hi", body_text="Body")

    def test_14_successful_send(self) -> None:
        """Verify send transfers message smoothly when connection works."""
        with patch("smtplib.SMTP") as mock_smtp:
            mock_conn = MagicMock()
            mock_smtp.return_value = mock_conn
            service = EmailService(smtp_host="smtp.example.com", smtp_port=587, email_from="sender@example.com")
            
            # This should run without throwing errors
            service.send(to_email="rec@example.com", subject="Sub", body_text="Plain text")
            mock_conn.send_message.assert_called_once()

    def test_15_failed_send(self) -> None:
        """Verify SMTP send failures wrap and raise EmailServiceException."""
        with patch("smtplib.SMTP") as mock_smtp:
            mock_conn = MagicMock()
            mock_conn.send_message.side_effect = smtplib.SMTPException("Mailbox full")
            mock_smtp.return_value = mock_conn
            service = EmailService(smtp_host="smtp.example.com", smtp_port=587, email_from="sender@example.com")
            
            with self.assertRaises(EmailServiceException):
                service.send(to_email="rec@example.com", subject="Sub", body_text="Plain text")

    def test_16_smtp_connection_cleanup(self) -> None:
        """Verify context managers safely disconnect and clean up connections."""
        with patch("smtplib.SMTP") as mock_smtp:
            mock_conn = MagicMock()
            mock_smtp.return_value = mock_conn
            
            with EmailService(smtp_host="smtp.example.com", smtp_port=587, email_from="sender@example.com") as service:
                mock_conn.ehlo.assert_called()
            
            mock_conn.quit.assert_called_once()
            self.assertIsNone(service._connection)

    def test_17_secrets_are_never_logged(self) -> None:
        """Verify that SMTP credentials and secrets are never logged."""
        logger = logging.getLogger("email_service")
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

        smtp_pass = "MySecretSmtpPassword123!"
        
        try:
            with patch("smtplib.SMTP") as mock_smtp:
                mock_conn = MagicMock()
                mock_smtp.return_value = mock_conn
                service = EmailService(
                    smtp_host="smtp.example.com",
                    smtp_port=587,
                    smtp_username="user",
                    smtp_password=smtp_pass,
                    email_from="sender@example.com"
                )
                service.connect()
                service.send(to_email="rec@example.com", subject="Hi", body_text="Hello")
                service.disconnect()
        finally:
            logger.removeHandler(handler)

        logs_content = log_capture.getvalue()
        self.assertNotIn(smtp_pass, logs_content)
        self.assertNotIn("pwd", logs_content)


if __name__ == "__main__":
    unittest.main()
