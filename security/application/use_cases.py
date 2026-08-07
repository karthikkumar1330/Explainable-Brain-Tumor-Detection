import datetime
import uuid
from typing import Optional, Dict, Any, Tuple, List

from security.domain.entities import User, Role, TokenType, SecurityAuditLog
from security.domain.interfaces import IUserRepository
from security.infrastructure.password import PasswordHasher
from security.infrastructure.jwt_service import JWTService, REFRESH_TOKEN_EXPIRE_DAYS
from security.infrastructure.rate_limiter import global_rate_limiter


class AuthUseCases:
    """Application level use-cases coordinating simplified authentication, authorization, and user profile operations."""

    def __init__(self, user_repo: IUserRepository, jwt_service: Optional[JWTService] = None):
        self.user_repo = user_repo
        self.jwt_service = jwt_service or JWTService()

    def register(self, email: str, password: str, full_name: str, role_str: str = "patient", ip_address: str = "127.0.0.1") -> Dict[str, Any]:
        """Registers a new user account and activates them immediately."""
        # Check rate limit
        limited, remaining = global_rate_limiter.is_rate_limited(f"register:{ip_address}", max_requests=10, window_seconds=600)
        if limited:
            raise ValueError(f"Too many registration attempts. Please try again in {remaining} seconds.")

        # Validate email
        email_clean = email.lower().strip()
        if not email_clean or "@" not in email_clean or "." not in email_clean:
            raise ValueError("Invalid email format.")

        # Check duplicate
        existing = self.user_repo.get_by_email(email_clean)
        if existing:
            raise ValueError("An account with this email already exists.")

        # Validate password strength
        valid_pass, pass_err = PasswordHasher.validate_password_strength(password)
        if not valid_pass:
            raise ValueError(pass_err)

        # Build user
        user_role = Role.from_string(role_str)
        now = datetime.datetime.utcnow().isoformat()
        user_uuid = str(uuid.uuid4())
        pass_hash = PasswordHasher.hash_password(password)

        new_user = User(
            id=None,
            uuid=user_uuid,
            email=email_clean,
            password_hash=pass_hash,
            full_name=full_name.strip(),
            role=user_role,
            is_verified=True,  # Default to True to bypass verification entirely
            is_active=True,
            created_at=now,
            updated_at=now,
        )
        created_user = self.user_repo.create_user(new_user)

        # Generate tokens for immediate login
        access_token = self.jwt_service.create_access_token(created_user.uuid, created_user.id, created_user.email, created_user.role)
        refresh_token = self.jwt_service.create_refresh_token(created_user.uuid, created_user.id, created_user.email, created_user.role)

        # Audit log
        self.user_repo.log_security_event(SecurityAuditLog(
            id=None,
            timestamp=now,
            event_type="USER_REGISTRATION",
            user_id=created_user.id,
            email=email_clean,
            ip_address=ip_address,
            status="SUCCESS",
            details=f"User registered with role {user_role.value}",
        ))

        return {
            "message": "User registered successfully.",
            "user": created_user.to_dict(),
            "access_token": access_token,
            "refresh_token": refresh_token,
        }

    def login(self, email: str, password: str, ip_address: str = "127.0.0.1", remember_me: bool = False, user_agent: str = "Unknown") -> Dict[str, Any]:
        """Authenticates user credentials and returns JWT tokens immediately without 2FA OTP flow."""
        email_clean = email.lower().strip()
        now = datetime.datetime.utcnow().isoformat()

        # Brute force rate check
        limited, remaining = global_rate_limiter.is_rate_limited(f"login:{ip_address}", max_requests=10, window_seconds=300)
        if limited:
            self.user_repo.log_security_event(SecurityAuditLog(
                id=None, timestamp=now, event_type="LOGIN_ATTEMPT", user_id=None, email=email_clean,
                ip_address=ip_address, status="BLOCKED", details=f"Rate limited for {remaining} seconds", user_agent=user_agent
            ))
            raise ValueError(f"Too many failed login attempts. Account temporarily locked. Please try again in {remaining} seconds.")

        user = self.user_repo.get_by_email(email_clean)

        # Check Account Lockout status
        if user and user.lockout_until:
            try:
                lock_dt = datetime.datetime.fromisoformat(user.lockout_until.replace("Z", "+00:00"))
                if datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc) < lock_dt:
                    remaining_seconds = int((lock_dt - datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc)).total_seconds())
                    self.user_repo.log_security_event(SecurityAuditLog(
                        id=None, timestamp=now, event_type="LOGIN_ATTEMPT", user_id=user.id, email=email_clean,
                        ip_address=ip_address, status="BLOCKED", details=f"Account locked. Try again in {remaining_seconds}s.", user_agent=user_agent
                    ))
                    raise ValueError(f"Account is temporarily locked due to too many failed login attempts. Please try again in {remaining_seconds} seconds.")
            except ValueError:
                raise
            except Exception:
                pass

        if not user or not PasswordHasher.verify_password(password, user.password_hash):
            if user:
                user.failed_login_attempts += 1
                if user.failed_login_attempts >= 5:
                    lock_time = (datetime.datetime.utcnow() + datetime.timedelta(minutes=15)).isoformat() + "Z"
                    user.lockout_until = lock_time
                    self.user_repo.update_user(user)
                    self.user_repo.log_security_event(SecurityAuditLog(
                        id=None, timestamp=now, event_type="ACCOUNT_LOCKED", user_id=user.id,
                        email=email_clean, ip_address=ip_address, status="LOCKED", details="Account locked due to 5 consecutive login failures", user_agent=user_agent
                    ))
                    raise ValueError("Account is temporarily locked due to too many failed login attempts. Please try again in 15 minutes.")
                else:
                    self.user_repo.update_user(user)
            
            self.user_repo.log_security_event(SecurityAuditLog(
                id=None, timestamp=now, event_type="LOGIN_ATTEMPT", user_id=user.id if user else None,
                email=email_clean, ip_address=ip_address, status="FAILURE", details=f"Invalid credentials. Attempt {user.failed_login_attempts if user else 1}/5", user_agent=user_agent
            ))
            if user:
                raise ValueError(f"Invalid email or password. Attempt {user.failed_login_attempts}/5.")
            raise ValueError("Invalid email or password.")

        if not user.is_active:
            raise ValueError("Your account has been deactivated. Please contact system administrator.")

        # Successful Login - Reset Lockout parameters
        user.last_login_at = now
        user.failed_login_attempts = 0
        user.lockout_until = None
        self.user_repo.update_user(user)
        global_rate_limiter.reset_key(f"login:{ip_address}")

        refresh_expires_days = 30 if remember_me else REFRESH_TOKEN_EXPIRE_DAYS
        access_token = self.jwt_service.create_access_token(user.uuid, user.id, user.email, user.role)
        refresh_token = self.jwt_service.create_refresh_token(user.uuid, user.id, user.email, user.role, expires_days=refresh_expires_days)

        self.user_repo.log_security_event(SecurityAuditLog(
            id=None, timestamp=now, event_type="LOGIN_SUCCESS", user_id=user.id, email=user.email,
            ip_address=ip_address, status="SUCCESS", details=f"User logged in successfully with role {user.role.value}", user_agent=user_agent
        ))

        return {
            "requires_2fa": False,
            "message": "Login successful.",
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "user": user.to_dict(),
        }

    def refresh_token(self, refresh_token: str, ip_address: str = "127.0.0.1") -> Dict[str, Any]:
        """Validates a refresh token, revokes it (rotation), and issues a new access/refresh token pair."""
        now = datetime.datetime.utcnow().isoformat()
        try:
            payload = self.jwt_service.decode_token(refresh_token, expected_type=TokenType.REFRESH)
        except Exception as e:
            raise ValueError(f"Invalid refresh token: {str(e)}")

        # Check JTI revocation (Token Replay Attack check)
        if self.user_repo.is_token_revoked(payload.jti):
            self.user_repo.log_security_event(SecurityAuditLog(
                id=None, timestamp=now, event_type="TOKEN_REPLAY_ATTACK_SUSPECTED", user_id=payload.user_id,
                email=payload.email, ip_address=ip_address, status="REVOKED",
                details=f"Revoked refresh token reuse attempt for JTI: {payload.jti}"
            ))
            raise ValueError("Refresh token has been revoked or already used.")

        # Get user
        user = self.user_repo.get_by_id(payload.user_id)
        if not user or not user.is_active:
            raise ValueError("User is inactive or does not exist.")

        # Check session invalidation timestamp
        if user.sessions_revoked_at:
            try:
                rev_str = user.sessions_revoked_at
                if not rev_str.endswith("Z") and "+00:00" not in rev_str:
                    rev_str += "Z"
                rev_str = rev_str.replace("Z", "+00:00")
                rev_dt = datetime.datetime.fromisoformat(rev_str)
                if payload.iat < rev_dt.timestamp():
                    raise ValueError("Refresh token has been revoked due to session invalidation.")
            except Exception:
                pass

        # Revoke the old refresh token JTI immediately (rotation)
        self.user_repo.revoke_token(payload.jti, user.id, payload.exp)

        # Calculate time remaining on the refresh token, or check if we should keep it long-lived
        duration_days = (payload.exp - payload.iat) / 86400.0
        remember_me = duration_days > 8.0
        refresh_expires_days = 30 if remember_me else REFRESH_TOKEN_EXPIRE_DAYS

        # Generate new access and refresh tokens
        new_access_token = self.jwt_service.create_access_token(user.uuid, user.id, user.email, user.role)
        new_refresh_token = self.jwt_service.create_refresh_token(
            user.uuid, user.id, user.email, user.role, expires_days=refresh_expires_days
        )

        self.user_repo.log_security_event(SecurityAuditLog(
            id=None, timestamp=now, event_type="TOKEN_REFRESH", user_id=user.id, email=user.email,
            ip_address=ip_address, status="SUCCESS", details="Access and refresh tokens rotated successfully"
        ))

        return {
            "access_token": new_access_token,
            "refresh_token": new_refresh_token,
            "token_type": "bearer",
            "user": user.to_dict(),
        }

    def logout(self, token: str, refresh_token: Optional[str] = None, ip_address: str = "127.0.0.1") -> Dict[str, Any]:
        """Revokes JWT access and refresh token JTIs upon logout."""
        now = datetime.datetime.utcnow().isoformat()
        access_payload = None
        refresh_payload = None

        # Revoke access token
        try:
            access_payload = self.jwt_service.decode_token(token, verify_exp=False)
            if access_payload:
                self.user_repo.revoke_token(access_payload.jti, access_payload.user_id, access_payload.exp)
        except Exception:
            pass

        # Revoke refresh token
        if refresh_token:
            try:
                refresh_payload = self.jwt_service.decode_token(refresh_token, verify_exp=False)
                if refresh_payload:
                    self.user_repo.revoke_token(refresh_payload.jti, refresh_payload.user_id, refresh_payload.exp)
            except Exception:
                pass

        user_id = access_payload.user_id if access_payload else (refresh_payload.user_id if refresh_payload else None)
        email = access_payload.email if access_payload else (refresh_payload.email if refresh_payload else None)

        self.user_repo.log_security_event(SecurityAuditLog(
            id=None, timestamp=now, event_type="LOGOUT", user_id=user_id, email=email,
            ip_address=ip_address, status="SUCCESS", details="Access/Refresh tokens revoked on logout"
        ))

        return {"message": "Logout successful."}

    def update_profile(self, user_id: int, full_name: Optional[str] = None, email: Optional[str] = None) -> Dict[str, Any]:
        """Updates user profile preferences, enforcing email uniqueness."""
        user = self.user_repo.get_by_id(user_id)
        if not user:
            raise ValueError("User not found.")

        if full_name and full_name.strip():
            user.full_name = full_name.strip()

        email_changed = False
        if email and email.strip() and email.lower().strip() != user.email:
            new_email = email.lower().strip()
            # Check uniqueness
            existing = self.user_repo.get_by_email(new_email)
            if existing and existing.id != user.id:
                raise ValueError("This email is already in use by another account.")
            user.email = new_email
            email_changed = True

        updated = self.user_repo.update_user(user)
        return {
            "message": "Profile updated successfully.",
            "email_changed": email_changed,
            "user": updated.to_dict()
        }

    def change_password(self, user_id: int, current_password: str, new_password: str, ip_address: str = "127.0.0.1") -> Dict[str, Any]:
        """Changes user password after validating current password."""
        now = datetime.datetime.utcnow().isoformat()
        user = self.user_repo.get_by_id(user_id)
        if not user or not PasswordHasher.verify_password(current_password, user.password_hash):
            raise ValueError("Current password is incorrect.")

        valid_pass, pass_err = PasswordHasher.validate_password_strength(new_password)
        if not valid_pass:
            raise ValueError(pass_err)

        user.password_hash = PasswordHasher.hash_password(new_password)
        self.user_repo.update_user(user)

        self.user_repo.log_security_event(SecurityAuditLog(
            id=None, timestamp=now, event_type="PASSWORD_CHANGE", user_id=user.id, email=user.email,
            ip_address=ip_address, status="SUCCESS", details="User changed password"
        ))

        return {"message": "Password changed successfully."}
