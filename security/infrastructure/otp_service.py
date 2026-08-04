import secrets
import datetime
from typing import Tuple
from security.domain.entities import OtpPurpose, TokenType, OTPRecord, VerificationToken


class OTPService:
    """Cryptographically secure OTP and verification token generation engine."""

    @staticmethod
    def generate_6digit_otp() -> str:
        """Generates a 6-digit numeric OTP code using secrets module."""
        return f"{secrets.randbelow(1000000):06d}"

    @staticmethod
    def generate_url_token() -> str:
        """Generates a secure, 32-byte URL-safe token."""
        return secrets.token_urlsafe(32)

    @classmethod
    def create_otp_record(cls, user_id: int, purpose: OtpPurpose, validity_minutes: int = 10) -> Tuple[OTPRecord, str]:
        code = cls.generate_6digit_otp()
        expires_at = (datetime.datetime.utcnow() + datetime.timedelta(minutes=validity_minutes)).isoformat()
        record = OTPRecord(
            id=None,
            user_id=user_id,
            otp_code=code,
            purpose=purpose,
            expires_at=expires_at,
            used_at=None,
        )
        return record, code

    @classmethod
    def create_verification_token(cls, user_id: int, token_type: TokenType, validity_hours: int = 24) -> Tuple[VerificationToken, str]:
        token_str = cls.generate_url_token()
        expires_at = (datetime.datetime.utcnow() + datetime.timedelta(hours=validity_hours)).isoformat()
        record = VerificationToken(
            id=None,
            user_id=user_id,
            token=token_str,
            token_type=token_type,
            expires_at=expires_at,
            used_at=None,
        )
        return record, token_str
