import os
import re
import ssl
import smtplib
import logging
from pathlib import Path
from email.message import EmailMessage
from email.utils import make_msgid
from typing import Optional, Any
from dotenv import load_dotenv

# Load env variables on module import
load_dotenv()


class EmailServiceException(Exception):
    """Base exception for email service operations."""
    pass


class InvalidAddressException(EmailServiceException):
    """Raised when an invalid recipient or sender address is provided."""
    pass


class ConfigurationException(EmailServiceException):
    """Raised when email settings or parameters are invalid."""
    pass


class ConnectionException(EmailServiceException):
    """Raised when connecting to SMTP server fails."""
    pass


class AuthenticationException(EmailServiceException):
    """Raised when SMTP authentication fails."""
    pass


class AttachmentException(EmailServiceException):
    """Raised when an attachment cannot be loaded or processed."""
    pass


def validate_email_address(email: str) -> None:
    """Validates the structure of email address and protects against header injection."""
    if not email:
        raise InvalidAddressException("Email address cannot be empty.")
    if "\r" in email or "\n" in email:
        raise InvalidAddressException("Header injection detected: Email address contains newlines.")
    if ".." in email:
        raise InvalidAddressException("Malformed email address: consecutive dots are invalid.")
    
    # Match standard email address pattern
    email_regex = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    if not re.match(email_regex, email):
        raise InvalidAddressException(f"Malformed email address: {email}")


def validate_subject(subject: str) -> None:
    """Validates subject headers to prevent header injection."""
    if not subject:
        raise EmailServiceException("Email subject cannot be empty.")
    if "\r" in subject or "\n" in subject:
        raise EmailServiceException("Header injection detected: Subject contains newlines.")


_UNSET = object()


class EmailService:
    """Reusable email infrastructure transport service following Clean Architecture design."""

    def __init__(
        self,
        smtp_host: Optional[str] = _UNSET,
        smtp_port: Optional[int] = _UNSET,
        smtp_username: Optional[str] = _UNSET,
        smtp_password: Optional[str] = _UNSET,
        email_from: Optional[str] = _UNSET,
        email_reply_to: Optional[str] = _UNSET,
        email_timeout: Optional[float] = _UNSET,
        use_ssl: Optional[bool] = _UNSET
    ) -> None:
        """Initializes configuration properties, prioritizing explicit parameters over env vars."""
        self.logger = logging.getLogger("email_service")

        from clinical_reporting.infrastructure.email_config import EmailConfig
        self.config = EmailConfig()
        
        # Explicit parameters override environment settings
        if smtp_host is not _UNSET:
            self.config.smtp_host = smtp_host
        if smtp_port is not _UNSET:
            self.config.smtp_port = smtp_port
        if smtp_username is not _UNSET:
            self.config.smtp_username = smtp_username
        if smtp_password is not _UNSET:
            self.config.smtp_password = smtp_password
        if email_from is not _UNSET:
            self.config.email_from = email_from
        if email_reply_to is not _UNSET:
            self.config.email_reply_to = email_reply_to
        if email_timeout is not _UNSET:
            self.config.email_timeout = email_timeout
        if use_ssl is not _UNSET:
            self.config.use_ssl = use_ssl

        # Expose properties for external access/backwards compatibility
        self.smtp_host = self.config.smtp_host
        self.smtp_port = self.config.smtp_port
        self.smtp_username = self.config.smtp_username
        self.smtp_password = self.config.smtp_password
        self.email_from = self.config.email_from
        self.email_reply_to = self.config.email_reply_to
        self.email_timeout = self.config.email_timeout
        self.use_ssl = self.config.use_ssl

        self._connection: Optional[Any] = None

    def connect(self) -> "EmailService":
        """Establishes connection to the SMTP server with SSL/TLS or STARTTLS protocols."""
        if self._connection:
            return self
        
        from clinical_reporting.infrastructure.email_config import EmailConfigException
        try:
            # Validate settings on connection attempt
            self.config.validate(active=True)
        except EmailConfigException as e:
            raise ConfigurationException(str(e))

        # Synchronize properties
        self.smtp_host = self.config.smtp_host
        self.smtp_port = self.config.smtp_port
        self.smtp_username = self.config.smtp_username
        self.smtp_password = self.config.smtp_password
        self.email_from = self.config.email_from
        self.email_reply_to = self.config.email_reply_to
        self.email_timeout = self.config.email_timeout
        self.use_ssl = self.config.use_ssl

        try:
            self.logger.info(f"Connecting to SMTP server {self.smtp_host}:{self.smtp_port} (use_ssl={self.use_ssl}, timeout={self.email_timeout}s)...")
            
            if self.use_ssl:
                ssl_context = ssl.create_default_context()
                self._connection = smtplib.SMTP_SSL(
                    host=self.smtp_host,
                    port=self.smtp_port,
                    timeout=self.email_timeout,
                    context=ssl_context
                )
            else:
                self._connection = smtplib.SMTP(
                    host=self.smtp_host,
                    port=self.smtp_port,
                    timeout=self.email_timeout
                )
                self._connection.ehlo()
                if self._connection.has_ext("STARTTLS"):
                    ssl_context = ssl.create_default_context()
                    self._connection.starttls(context=ssl_context)
                    self._connection.ehlo()
                
            if self.smtp_username and self.smtp_password:
                self.logger.info("Authenticating with SMTP server...")
                self._connection.login(self.smtp_username, self.smtp_password)
                
            self.logger.info("SMTP connection established successfully.")
            return self
        except smtplib.SMTPAuthenticationError as sae:
            self._cleanup_connection_silent()
            raise AuthenticationException(f"SMTP authentication failed: {sae}")
        except (smtplib.SMTPConnectError, ConnectionRefusedError, TimeoutError) as ce:
            self._cleanup_connection_silent()
            raise ConnectionException(f"Failed to connect to SMTP server: {ce}")
        except Exception as e:
            self._cleanup_connection_silent()
            raise ConnectionException(f"Unexpected connection error: {e}")

    def disconnect(self) -> None:
        """Safely disconnects from the SMTP server."""
        if self._connection:
            try:
                self._connection.quit()
            except Exception:
                pass
            finally:
                self._connection = None
                self.logger.info("SMTP connection closed.")

    def _cleanup_connection_silent(self) -> None:
        """Internal helper to clean connection state on failure."""
        if self._connection:
            try:
                self._connection.close()
            except Exception:
                pass
            self._connection = None

    def __enter__(self) -> "EmailService":
        """Context manager hook to establish SMTP session."""
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager hook to safely tear down SMTP session."""
        self.disconnect()

    def send(
        self,
        to_email: str,
        subject: str,
        body_text: str,
        body_html: Optional[str] = None,
        attachment_path: Optional[str] = None
    ) -> None:
        """Constructs and transmits the email message. Auto-connects if not in active context."""
        # Validate headers to prevent injection attacks
        validate_email_address(to_email)

        if not self.email_from:
            raise ConfigurationException("Sender email address (EMAIL_FROM) is not configured.")
        try:
            validate_email_address(self.email_from)
        except InvalidAddressException as iae:
            raise ConfigurationException(f"Invalid sender email address configuration: {iae}")

        if self.email_reply_to:
            try:
                validate_email_address(self.email_reply_to)
            except InvalidAddressException as iae:
                raise ConfigurationException(f"Invalid reply-to email address configuration: {iae}")

        validate_subject(subject)

        # Build RFC MIME message
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = self.email_from
        msg["To"] = to_email
        if self.email_reply_to:
            msg["Reply-To"] = self.email_reply_to

        msg["Message-ID"] = make_msgid(domain=self.smtp_host or "aurascan.ai")

        if body_html:
            msg.set_content(body_text)
            msg.add_alternative(body_html, subtype="html")
        else:
            msg.set_content(body_text)

        # Attach file safely
        if attachment_path:
            p = Path(attachment_path)
            if not p.exists():
                raise AttachmentException(f"Attachment file not found: {attachment_path}")
            if not p.is_file():
                raise AttachmentException(f"Attachment path is not a file: {attachment_path}")
            
            try:
                file_bytes = p.read_bytes()
                filename = p.name
                msg.add_attachment(
                    file_bytes,
                    maintype="application",
                    subtype="pdf" if filename.lower().endswith(".pdf") else "octet-stream",
                    filename=filename
                )
                self.logger.info(f"Attached file: {filename} ({len(file_bytes)} bytes)")
            except Exception as e:
                raise AttachmentException(f"Failed to read attachment file: {e}")

        # Logs operation start (redacting recipient local name for compliance)
        domain = to_email.split("@")[1] if "@" in to_email else "unknown"
        self.logger.info(f"Initiating email send task. Recipient domain: {domain}")

        should_disconnect = False
        if not self._connection:
            self.connect()
            should_disconnect = True

        try:
            self._connection.send_message(msg)
            self.logger.info("Email message sent successfully.")
        except smtplib.SMTPException as se:
            raise EmailServiceException(f"SMTP send failure: {se}")
        except Exception as e:
            raise EmailServiceException(f"Unexpected send failure: {e}")
        finally:
            if should_disconnect:
                self.disconnect()
