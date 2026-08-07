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


@dataclass
class User:
    id: Optional[int]
    uuid: str
    email: str
    password_hash: str
    full_name: str
    role: Role
    is_verified: bool = True
    is_active: bool = True
    profile_pic: Optional[str] = None
    sessions_revoked_at: Optional[str] = None
    failed_login_attempts: int = 0
    lockout_until: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.datetime.utcnow().isoformat())
    last_login_at: Optional[str] = None

    @property
    def email_verified(self) -> bool:
        return self.is_verified

    @email_verified.setter
    def email_verified(self, value: bool) -> None:
        self.is_verified = value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "uuid": self.uuid,
            "email": self.email,
            "full_name": self.full_name,
            "role": self.role.value if isinstance(self.role, Role) else self.role,
            "is_verified": self.is_verified,
            "email_verified": self.is_verified,
            "is_active": self.is_active,
            "google_profile_pic": self.profile_pic,
            "sessions_revoked_at": self.sessions_revoked_at,
            "failed_login_attempts": self.failed_login_attempts,
            "lockout_until": self.lockout_until,
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
class SecurityAuditLog:
    id: Optional[int]
    timestamp: str
    event_type: str
    user_id: Optional[int]
    email: Optional[str]
    ip_address: str
    status: str
    details: str
    user_agent: Optional[str] = None
