import re
import hashlib
import hmac
import os
import secrets

import bcrypt


class PasswordHasher:
    """Enterprise password hashing engine enforcing OWASP security standards."""

    @classmethod
    def normalize_email(cls, email: str) -> str:
        """Normalizes and validates an email address. Enforces strict formats and rejects Markdown formatting."""
        if not isinstance(email, str):
            raise ValueError("Email must be a string.")

        email_clean = email.strip().lower()
        if not email_clean:
            raise ValueError("Email cannot be empty.")

        # Reject markdown links or mailto links explicitly
        if "[" in email_clean or "]" in email_clean or "(" in email_clean or ")" in email_clean or "mailto:" in email_clean:
            raise ValueError("Malformed email address containing links is not allowed.")

        # Basic validation: must contain exactly one @ and at least one dot in the domain part
        parts = email_clean.split("@")
        if len(parts) != 2 or not parts[0] or not parts[1] or "." not in parts[1]:
            raise ValueError("Invalid email format.")

        return email_clean

    @staticmethod
    def validate_password_strength(password: str) -> tuple[bool, str]:
        """Validates password against OWASP recommendations:
        - At least 8 characters long
        - At least one uppercase letter
        - At least one lowercase letter
        - At least one digit
        - At least one special character
        """
        if len(password) < 8:
            return False, "Password must be at least 8 characters long."
        if not re.search(r"[A-Z]", password):
            return False, "Password must contain at least one uppercase letter."
        if not re.search(r"[a-z]", password):
            return False, "Password must contain at least one lowercase letter."
        if not re.search(r"\d", password):
            return False, "Password must contain at least one number."
        if not re.search(r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>/?]", password):
            return False, "Password must contain at least one special character (!@#$%^&*...)."
        return True, ""

    @classmethod
    def hash_password(cls, password: str) -> str:
        """Hashes password using bcrypt with salt."""
        salt = bcrypt.gensalt(rounds=12)
        hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
        return hashed.decode("utf-8")

    @classmethod
    def verify_password(cls, password: str, password_hash: str) -> bool:
        """Verifies candidate password against hash in constant-time."""
        if not password_hash or not password:
            return False

        try:
            if password_hash.startswith("$2b$") or password_hash.startswith("$2a$") or password_hash.startswith("$2y$"):
                return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))

            if password_hash.startswith("pbkdf2_sha256$"):
                parts = password_hash.split("$")
                if len(parts) == 4:
                    _, rounds, salt_hex, key_hex = parts
                    salt = bytes.fromhex(salt_hex)
                    expected_key = bytes.fromhex(key_hex)
                    computed_key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(rounds))
                    return hmac.compare_digest(computed_key, expected_key)

            return False
        except Exception:
            return False
