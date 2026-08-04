import re
import hashlib
import hmac
import os
import secrets

try:
    import bcrypt
    HAS_BCRYPT = True
except ImportError:
    HAS_BCRYPT = False


class PasswordHasher:
    """Enterprise password hashing engine enforcing OWASP security standards."""

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
        """Hashes password using bcrypt with salt, or PBKDF2-HMAC-SHA256 fallback."""
        if HAS_BCRYPT:
            salt = bcrypt.gensalt(rounds=12)
            hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
            return hashed.decode("utf-8")
        else:
            # Fallback: PBKDF2 with 600,000 iterations (OWASP recommendation)
            salt = secrets.token_bytes(16)
            key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 600000)
            return f"pbkdf2_sha256$600000${salt.hex()}${key.hex()}"

    @classmethod
    def verify_password(cls, password: str, password_hash: str) -> bool:
        """Verifies candidate password against hash in constant-time."""
        if not password_hash or not password:
            return False

        try:
            if password_hash.startswith("$2b$") or password_hash.startswith("$2a$") or password_hash.startswith("$2y$"):
                if HAS_BCRYPT:
                    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
                return False

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
