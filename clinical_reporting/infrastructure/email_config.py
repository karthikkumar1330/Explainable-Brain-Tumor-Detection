import os
import re
from typing import Optional
from dotenv import load_dotenv

# Load environment variables on module import
load_dotenv()


class EmailConfigException(Exception):
    """Base exception for email configuration validation failures."""
    pass


class EmailConfig:
    """Configuration class that loads and validates email service settings securely."""

    def __init__(self) -> None:
        """Loads configuration variables from environment and applies secure defaults."""
        self.email_provider = os.environ.get("EMAIL_PROVIDER", "smtp").lower()
        self.smtp_host = os.environ.get("SMTP_HOST")
        
        # Resolve SMTP Port
        port_val = os.environ.get("SMTP_PORT")
        if port_val:
            try:
                self.smtp_port = int(port_val)
            except ValueError:
                raise EmailConfigException(f"Invalid SMTP_PORT value: {port_val}")
        else:
            self.smtp_port = 587  # Safe default port
            
        self.smtp_username = os.environ.get("SMTP_USERNAME")
        self.smtp_password = os.environ.get("SMTP_PASSWORD")
        self.email_from = os.environ.get("EMAIL_FROM")
        self.email_reply_to = os.environ.get("EMAIL_REPLY_TO")
        
        # Resolve Timeout
        timeout_val = os.environ.get("EMAIL_TIMEOUT")
        if timeout_val:
            try:
                self.email_timeout = float(timeout_val)
            except ValueError:
                raise EmailConfigException(f"Invalid EMAIL_TIMEOUT value: {timeout_val}")
        else:
            self.email_timeout = 10.0  # Safe default timeout
            
        # Resolve SSL Usage
        ssl_val = os.environ.get("SMTP_USE_SSL", "")
        if ssl_val.lower() in ("true", "1", "yes"):
            self.use_ssl = True
        elif self.smtp_port == 465:
            self.use_ssl = True
        else:
            self.use_ssl = False

    def validate(self, active: bool = False) -> None:
        """Validates loaded parameters strictly. Skipped if not active and SMTP_HOST is empty."""
        if not active and not self.smtp_host:
            # If not active and SMTP_HOST is not configured, do not fail startup
            return

        if not self.smtp_host:
            raise EmailConfigException("SMTP_HOST configuration parameter is missing.")

        if self.smtp_port is None:
            raise EmailConfigException("SMTP_PORT configuration parameter is missing.")
        if self.smtp_port <= 0 or self.smtp_port > 65535:
            raise EmailConfigException("SMTP_PORT must be a valid integer between 1 and 65535.")
            
        if self.smtp_port == 465:
            self.use_ssl = True
            
        if self.email_timeout <= 0:
            raise EmailConfigException("EMAIL_TIMEOUT must be a positive number of seconds.")
            
        if not self.email_from:
            raise EmailConfigException("EMAIL_FROM is required when email service is active.")
            
        # Validate format of EMAIL_FROM
        email_regex = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
        if not re.match(email_regex, self.email_from) or ".." in self.email_from:
            raise EmailConfigException("Invalid EMAIL_FROM address format.")
            
        if self.email_reply_to:
            if not re.match(email_regex, self.email_reply_to) or ".." in self.email_reply_to:
                raise EmailConfigException("Invalid EMAIL_REPLY_TO address format.")
                
        # Require password if username is configured
        if self.smtp_username and not self.smtp_password:
            raise EmailConfigException("SMTP_PASSWORD is required when SMTP_USERNAME is set.")

        # Require secure transport (STARTTLS on port 587 or SSL on port 465)
        # Insecure connections are allowed in development/testing only if explicitly configured
        if not self.use_ssl and self.smtp_port != 587:
            env_mode = os.environ.get("ENV_MODE", os.environ.get("APP_ENV", "production")).lower()
            if env_mode == "production":
                raise EmailConfigException(
                    "Insecure SMTP configuration rejected: SSL/TLS or STARTTLS (port 587) is required in production."
                )

    def get_non_secret_settings(self) -> dict:
        """Returns non-sensitive configuration settings for logging and diagnosis."""
        return {
            "email_provider": self.email_provider,
            "smtp_host": self.smtp_host,
            "smtp_port": self.smtp_port,
            "smtp_username": self.smtp_username,
            "email_from": self.email_from,
            "email_reply_to": self.email_reply_to,
            "email_timeout": self.email_timeout,
            "use_ssl": self.use_ssl
        }
