import datetime
from enum import Enum
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


class Role(str, Enum):
    ADMIN = "admin"
    DOCTOR = "doctor"
    PATIENT = "patient"

    @classmethod
    def from_string(cls, role_str: str) -> "Role":
        try:
            return cls(role_str.lower().strip())
        except ValueError:
            return cls.PATIENT


class TokenType(str, Enum):
    ACCESS = "access"
    REFRESH = "refresh"
    EMAIL_VERIFICATION = "email_verification"
    PASSWORD_RESET = "password_reset"


class OtpPurpose(str, Enum):
    LOGIN_2FA = "login_2fa"
    PASSWORD_RESET = "password_reset"
    EMAIL_VERIFICATION = "email_verification"


@dataclass
class User:
    id: Optional[int]
    uuid: str
    email: str
    password_hash: str
    full_name: str
    role: Role
    is_verified: bool = False
    is_active: bool = True
    two_factor_enabled: bool = False
    two_factor_secret: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    last_login_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "uuid": self.uuid,
            "email": self.email,
            "full_name": self.full_name,
            "role": self.role.value if isinstance(self.role, Role) else self.role,
            "is_verified": self.is_verified,
            "is_active": self.is_active,
            "two_factor_enabled": self.two_factor_enabled,
            "created_at": self.created_at,
            "last_login_at": self.last_login_at,
        }


@dataclass
class TokenPayload:
    sub: str  # user uuid
    user_id: int
    email: str
    role: Role
    jti: str  # unique token ID
    type: TokenType
    exp: float
    iat: float


@dataclass
class VerificationToken:
    id: Optional[int]
    user_id: int
    token: str
    token_type: TokenType
    expires_at: str
    used_at: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())


@dataclass
class OTPRecord:
    id: Optional[int]
    user_id: int
    otp_code: str
    purpose: OtpPurpose
    expires_at: str
    used_at: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())


@dataclass
class SecurityAuditLog:
    id: Optional[int]
    timestamp: str
    event_type: str
    user_id: Optional[int]
    email: Optional[str]
    ip_address: str
    status: str
    details: str
