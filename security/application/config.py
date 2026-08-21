import os
import re
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv

# Resolve absolute project root to load .env deterministically
project_root = Path(__file__).resolve().parent.parent.parent
env_path = project_root / ".env"

# Load environment variables on module import (respects pre-existing process variables)
load_dotenv(dotenv_path=env_path)

class AppConfigException(ValueError):
    """Exception raised when application configuration validation fails."""
    pass

class AppConfig:
    """Manages application-wide configurations with robust validation and defaults."""

    def __init__(self) -> None:
        # Trigger initial validation of the public base URL on load
        _ = self.public_base_url

    @property
    def host(self) -> str:
        # Precedence: AURASCAN_HOST -> HOST -> default 127.0.0.1
        return os.environ.get("AURASCAN_HOST", os.environ.get("HOST", "127.0.0.1"))

    @property
    def dashboard_port(self) -> int:
        port_val = os.environ.get("DASHBOARD_PORT")
        if port_val:
            try:
                return int(port_val)
            except ValueError:
                raise AppConfigException(f"Invalid DASHBOARD_PORT configuration: {port_val}")
        return 5000

    @property
    def api_port(self) -> int:
        port_val = os.environ.get("REST_PORT")
        if port_val:
            try:
                return int(port_val)
            except ValueError:
                raise AppConfigException(f"Invalid REST_PORT configuration: {port_val}")
        return 8000

    @property
    def public_base_url(self) -> str:
        # Precedence: AURASCAN_PUBLIC_BASE_URL -> APP_URL -> default http://127.0.0.1:5000
        public_url = os.environ.get("AURASCAN_PUBLIC_BASE_URL", os.environ.get("APP_URL"))
        if public_url:
            url = public_url.rstrip("/")
        else:
            url = "http://127.0.0.1:5000"

        self.validate_public_url(url)
        return url

    @property
    def fastapi_url(self) -> str:
        fastapi_env_url = os.environ.get("FASTAPI_URL")
        if fastapi_env_url:
            return fastapi_env_url.rstrip("/")

        # Dynamically derive FastAPI URL from the public base URL
        base_url = self.public_base_url
        parsed = urlparse(base_url)
        if parsed.port:
            netloc_host = parsed.netloc.split(":")[0]
            return f"{parsed.scheme}://{netloc_host}:{self.api_port}"
        else:
            return f"{base_url}:{self.api_port}"

    def validate_public_url(self, url: str) -> None:
        """Enforces clean structure, scheme correctness, and rejects placeholders."""
        if not url:
            raise AppConfigException("Public base URL must not be empty.")

        try:
            parsed = urlparse(url)
        except ValueError as ve:
            raise AppConfigException(f"Malformed public URL: '{url}'. Parsing error: {ve}") from ve

        if not parsed.scheme or not parsed.netloc:
            raise AppConfigException(f"Malformed public URL: '{url}'. Must include a valid scheme and host.")

        if parsed.scheme not in ("http", "https"):
            raise AppConfigException(f"Invalid scheme in public URL: '{url}'. Only HTTP and HTTPS are allowed.")

        # Block any bracket-based configurations or typical example placeholders
        if any(char in url for char in ("<", ">", "{", "}", "[", "]")):
            raise AppConfigException(f"Public URL cannot contain placeholder/bracket characters: '{url}'")

    def get_non_secret_settings(self) -> dict:
        """Returns non-sensitive configurations safely for logging."""
        return {
            "AURASCAN_HOST": self.host,
            "AURASCAN_PUBLIC_BASE_URL": self.public_base_url,
            "DASHBOARD_PORT": self.dashboard_port,
            "REST_PORT": self.api_port,
            "FASTAPI_URL": self.fastapi_url
        }

# Global Configuration Singleton
app_config = AppConfig()
