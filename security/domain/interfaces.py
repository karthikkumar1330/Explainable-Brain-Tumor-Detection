from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
from security.domain.entities import User, VerificationToken, OTPRecord, SecurityAuditLog, TokenType, OtpPurpose, Role


class IUserRepository(ABC):
    @abstractmethod
    def create_user(self, user: User) -> User:
        pass

    @abstractmethod
    def get_by_id(self, user_id: int) -> Optional[User]:
        pass

    @abstractmethod
    def get_by_uuid(self, uuid: str) -> Optional[User]:
        pass

    @abstractmethod
    def get_by_email(self, email: str) -> Optional[User]:
        pass

    @abstractmethod
    def update_user(self, user: User) -> User:
        pass

    @abstractmethod
    def list_users(self, limit: int = 100, offset: int = 0) -> List[User]:
        pass

    @abstractmethod
    def save_verification_token(self, token_record: VerificationToken) -> VerificationToken:
        pass

    @abstractmethod
    def get_verification_token(self, token: str, token_type: TokenType) -> Optional[VerificationToken]:
        pass

    @abstractmethod
    def mark_token_used(self, token_id: int) -> None:
        pass

    @abstractmethod
    def save_otp(self, otp_record: OTPRecord) -> OTPRecord:
        pass

    @abstractmethod
    def get_latest_otp(self, user_id: int, purpose: OtpPurpose) -> Optional[OTPRecord]:
        pass

    @abstractmethod
    def mark_otp_used(self, otp_id: int) -> None:
        pass

    @abstractmethod
    def revoke_token(self, jti: str, user_id: int, expires_at: float) -> None:
        pass

    @abstractmethod
    def is_token_revoked(self, jti: str) -> bool:
        pass

    @abstractmethod
    def log_security_event(self, event: SecurityAuditLog) -> None:
        pass

    @abstractmethod
    def get_security_audit_logs(self, limit: int = 100) -> List[Dict[str, Any]]:
        pass
