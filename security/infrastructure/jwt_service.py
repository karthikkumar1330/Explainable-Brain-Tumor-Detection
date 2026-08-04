import os
import time
import uuid
import jwt
from typing import Optional, Dict, Any
from security.domain.entities import Role, TokenType, TokenPayload

SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "aurascan_ai_enterprise_super_secret_jwt_key_2026_minimum_32bytes_required")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7


class JWTService:
    """Enterprise JWT Token generation and verification service."""

    def __init__(self, secret_key: str = SECRET_KEY):
        self.secret_key = secret_key

    def create_access_token(self, user_uuid: str, user_id: int, email: str, role: Role, expires_minutes: int = ACCESS_TOKEN_EXPIRE_MINUTES) -> str:
        now = time.time()
        payload = {
            "sub": user_uuid,
            "user_id": user_id,
            "email": email,
            "role": role.value if isinstance(role, Role) else role,
            "jti": str(uuid.uuid4()),
            "type": TokenType.ACCESS.value,
            "iat": int(now),
            "exp": int(now + expires_minutes * 60),
        }
        return jwt.encode(payload, self.secret_key, algorithm=ALGORITHM)

    def create_refresh_token(self, user_uuid: str, user_id: int, email: str, role: Role, expires_days: int = REFRESH_TOKEN_EXPIRE_DAYS) -> str:
        now = time.time()
        payload = {
            "sub": user_uuid,
            "user_id": user_id,
            "email": email,
            "role": role.value if isinstance(role, Role) else role,
            "jti": str(uuid.uuid4()),
            "type": TokenType.REFRESH.value,
            "iat": int(now),
            "exp": int(now + expires_days * 86400),
        }
        return jwt.encode(payload, self.secret_key, algorithm=ALGORITHM)

    def decode_token(self, token: str) -> Optional[TokenPayload]:
        try:
            payload_dict = jwt.decode(token, self.secret_key, algorithms=[ALGORITHM])
            return TokenPayload(
                sub=payload_dict["sub"],
                user_id=payload_dict["user_id"],
                email=payload_dict["email"],
                role=Role.from_string(payload_dict["role"]),
                jti=payload_dict["jti"],
                type=TokenType(payload_dict["type"]),
                exp=float(payload_dict["exp"]),
                iat=float(payload_dict["iat"]),
            )
        except (jwt.PyJWTError, KeyError, ValueError):
            return None
