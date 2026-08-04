import datetime
import uuid
from typing import Optional, Dict, Any, Tuple, List

from security.domain.entities import User, Role, TokenType, OtpPurpose, SecurityAuditLog
from security.domain.interfaces import IUserRepository
from security.infrastructure.password import PasswordHasher
from security.infrastructure.jwt_service import JWTService
from security.infrastructure.otp_service import OTPService
from security.infrastructure.rate_limiter import global_rate_limiter


class AuthUseCases:
    """Application level use-cases coordinating authentication, authorization, OTP, and user profile operations."""

    def __init__(self, user_repo: IUserRepository, jwt_service: Optional[JWTService] = None):
        self.user_repo = user_repo
        self.jwt_service = jwt_service or JWTService()

    def register(self, email: str, password: str, full_name: str, role_str: str = "patient", ip_address: str = "127.0.0.1") -> Dict[str, Any]:
        """Registers a new user account with OWASP password complexity and email verification token."""
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
            is_verified=False,
            is_active=True,
            two_factor_enabled=False,
            created_at=now,
            updated_at=now,
        )
        created_user = self.user_repo.create_user(new_user)

        # Create Email Verification Token & OTP
        token_record, token_str = OTPService.create_verification_token(created_user.id, TokenType.EMAIL_VERIFICATION, validity_hours=24)
        self.user_repo.save_verification_token(token_record)

        otp_record, otp_code = OTPService.create_otp_record(created_user.id, OtpPurpose.EMAIL_VERIFICATION, validity_minutes=15)
        self.user_repo.save_otp(otp_record)

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
            "message": "User registered successfully. Please verify your email using the verification code/token provided.",
            "user": created_user.to_dict(),
            "verification_token": token_str,
            "verification_otp": otp_code,
        }

    def verify_email(self, token_or_otp: str, email: Optional[str] = None, ip_address: str = "127.0.0.1") -> Dict[str, Any]:
        """Verifies email using URL verification token or 6-digit OTP code."""
        now = datetime.datetime.utcnow().isoformat()

        # Try URL token first
        token_record = self.user_repo.get_verification_token(token_or_otp, TokenType.EMAIL_VERIFICATION)
        if token_record:
            if token_record.used_at:
                raise ValueError("Verification token has already been used.")
            if token_record.expires_at < now:
                raise ValueError("Verification token has expired.")

            user = self.user_repo.get_by_id(token_record.user_id)
            if not user:
                raise ValueError("User not found.")

            user.is_verified = True
            self.user_repo.update_user(user)
            self.user_repo.mark_token_used(token_record.id)

            self.user_repo.log_security_event(SecurityAuditLog(
                id=None, timestamp=now, event_type="EMAIL_VERIFICATION", user_id=user.id, email=user.email,
                ip_address=ip_address, status="SUCCESS", details="Email verified via link token"
            ))
            return {"message": "Email verified successfully.", "user": user.to_dict()}

        # Try OTP code with email
        if email:
            user = self.user_repo.get_by_email(email)
            if user:
                otp_rec = self.user_repo.get_latest_otp(user.id, OtpPurpose.EMAIL_VERIFICATION)
                if otp_rec and otp_rec.otp_code == token_or_otp.strip():
                    if otp_rec.expires_at < now:
                        raise ValueError("Verification OTP code has expired.")

                    user.is_verified = True
                    self.user_repo.update_user(user)
                    self.user_repo.mark_otp_used(otp_rec.id)

                    self.user_repo.log_security_event(SecurityAuditLog(
                        id=None, timestamp=now, event_type="EMAIL_VERIFICATION", user_id=user.id, email=user.email,
                        ip_address=ip_address, status="SUCCESS", details="Email verified via OTP code"
                    ))
                    return {"message": "Email verified successfully.", "user": user.to_dict()}

        raise ValueError("Invalid or expired email verification token / OTP code.")

    def login(self, email: str, password: str, ip_address: str = "127.0.0.1") -> Dict[str, Any]:
        """Authenticates user credentials and returns JWT tokens or triggers 2FA OTP flow."""
        email_clean = email.lower().strip()
        now = datetime.datetime.utcnow().isoformat()

        # Brute force rate check
        limited, remaining = global_rate_limiter.is_rate_limited(f"login:{ip_address}", max_requests=5, window_seconds=300)
        if limited:
            self.user_repo.log_security_event(SecurityAuditLog(
                id=None, timestamp=now, event_type="LOGIN_ATTEMPT", user_id=None, email=email_clean,
                ip_address=ip_address, status="BLOCKED", details=f"Rate limited for {remaining} seconds"
            ))
            raise ValueError(f"Too many failed login attempts. Account temporarily locked. Please try again in {remaining} seconds.")

        user = self.user_repo.get_by_email(email_clean)
        if not user or not PasswordHasher.verify_password(password, user.password_hash):
            self.user_repo.log_security_event(SecurityAuditLog(
                id=None, timestamp=now, event_type="LOGIN_ATTEMPT", user_id=user.id if user else None,
                email=email_clean, ip_address=ip_address, status="FAILURE", details="Invalid credentials"
            ))
            raise ValueError("Invalid email or password.")

        if not user.is_active:
            raise ValueError("Your account has been deactivated. Please contact system administrator.")

        # If 2FA OTP is required for account
        if user.two_factor_enabled:
            otp_record, otp_code = OTPService.create_otp_record(user.id, OtpPurpose.LOGIN_2FA, validity_minutes=10)
            self.user_repo.save_otp(otp_record)

            self.user_repo.log_security_event(SecurityAuditLog(
                id=None, timestamp=now, event_type="LOGIN_2FA_REQUIRED", user_id=user.id, email=email_clean,
                ip_address=ip_address, status="PENDING_2FA", details="2FA OTP generated for login"
            ))

            return {
                "requires_2fa": True,
                "message": "Two-factor authentication required. Please verify the OTP code sent to your device/email.",
                "user_id": user.id,
                "email": user.email,
                "otp_code": otp_code,  # Returned for mock/dev notification support
            }

        # Successful Login
        user.last_login_at = now
        self.user_repo.update_user(user)
        global_rate_limiter.reset_key(f"login:{ip_address}")

        access_token = self.jwt_service.create_access_token(user.uuid, user.id, user.email, user.role)
        refresh_token = self.jwt_service.create_refresh_token(user.uuid, user.id, user.email, user.role)

        self.user_repo.log_security_event(SecurityAuditLog(
            id=None, timestamp=now, event_type="LOGIN_SUCCESS", user_id=user.id, email=user.email,
            ip_address=ip_address, status="SUCCESS", details=f"User logged in successfully with role {user.role.value}"
        ))

        return {
            "requires_2fa": False,
            "message": "Login successful.",
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "user": user.to_dict(),
        }

    def verify_login_otp(self, user_id: int, otp_code: str, ip_address: str = "127.0.0.1") -> Dict[str, Any]:
        """Verifies 2FA OTP code and completes login."""
        now = datetime.datetime.utcnow().isoformat()
        user = self.user_repo.get_by_id(user_id)
        if not user:
            raise ValueError("User not found.")

        otp_rec = self.user_repo.get_latest_otp(user.id, OtpPurpose.LOGIN_2FA)
        if not otp_rec or otp_rec.otp_code != otp_code.strip():
            self.user_repo.log_security_event(SecurityAuditLog(
                id=None, timestamp=now, event_type="LOGIN_2FA_VERIFICATION", user_id=user.id, email=user.email,
                ip_address=ip_address, status="FAILURE", details="Invalid 2FA OTP code"
            ))
            raise ValueError("Invalid 2FA verification OTP code.")

        if otp_rec.expires_at < now:
            raise ValueError("2FA OTP code has expired.")

        self.user_repo.mark_otp_used(otp_rec.id)
        user.last_login_at = now
        self.user_repo.update_user(user)

        access_token = self.jwt_service.create_access_token(user.uuid, user.id, user.email, user.role)
        refresh_token = self.jwt_service.create_refresh_token(user.uuid, user.id, user.email, user.role)

        self.user_repo.log_security_event(SecurityAuditLog(
            id=None, timestamp=now, event_type="LOGIN_2FA_SUCCESS", user_id=user.id, email=user.email,
            ip_address=ip_address, status="SUCCESS", details="2FA OTP verified, login complete"
        ))

        return {
            "message": "2FA OTP verified successfully.",
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "user": user.to_dict(),
        }

    def logout(self, token: str, ip_address: str = "127.0.0.1") -> Dict[str, Any]:
        """Revokes JWT access/refresh token JTI upon logout."""
        payload = self.jwt_service.decode_token(token)
        now = datetime.datetime.utcnow().isoformat()

        if payload:
            self.user_repo.revoke_token(payload.jti, payload.user_id, payload.exp)
            self.user_repo.log_security_event(SecurityAuditLog(
                id=None, timestamp=now, event_type="LOGOUT", user_id=payload.user_id, email=payload.email,
                ip_address=ip_address, status="SUCCESS", details="Token revoked on logout"
            ))

        return {"message": "Logout successful."}

    def forgot_password(self, email: str, ip_address: str = "127.0.0.1") -> Dict[str, Any]:
        """Initiates password reset flow with token & OTP code."""
        email_clean = email.lower().strip()
        now = datetime.datetime.utcnow().isoformat()

        user = self.user_repo.get_by_email(email_clean)
        if not user:
            # OWASP recommendation: do not leak user existence
            return {"message": "If an account exists with that email, password reset instructions have been sent."}

        token_record, reset_token = OTPService.create_verification_token(user.id, TokenType.PASSWORD_RESET, validity_hours=1)
        self.user_repo.save_verification_token(token_record)

        otp_record, otp_code = OTPService.create_otp_record(user.id, OtpPurpose.PASSWORD_RESET, validity_minutes=15)
        self.user_repo.save_otp(otp_record)

        self.user_repo.log_security_event(SecurityAuditLog(
            id=None, timestamp=now, event_type="PASSWORD_RESET_REQUEST", user_id=user.id, email=user.email,
            ip_address=ip_address, status="SUCCESS", details="Password reset token and OTP generated"
        ))

        return {
            "message": "If an account exists with that email, password reset instructions have been sent.",
            "reset_token": reset_token,
            "reset_otp": otp_code,
        }

    def reset_password(self, reset_token_or_otp: str, new_password: str, email: Optional[str] = None, ip_address: str = "127.0.0.1") -> Dict[str, Any]:
        """Resets user password using URL token or OTP code, enforcing OWASP complexity."""
        now = datetime.datetime.utcnow().isoformat()

        valid_pass, pass_err = PasswordHasher.validate_password_strength(new_password)
        if not valid_pass:
            raise ValueError(pass_err)

        user: Optional[User] = None
        token_id_to_mark: Optional[int] = None
        otp_id_to_mark: Optional[int] = None

        # Check URL token
        t_rec = self.user_repo.get_verification_token(reset_token_or_otp, TokenType.PASSWORD_RESET)
        if t_rec:
            if t_rec.used_at:
                raise ValueError("Password reset token has already been used.")
            if t_rec.expires_at < now:
                raise ValueError("Password reset token has expired.")
            user = self.user_repo.get_by_id(t_rec.user_id)
            token_id_to_mark = t_rec.id

        # Check OTP
        elif email:
            user = self.user_repo.get_by_email(email)
            if user:
                o_rec = self.user_repo.get_latest_otp(user.id, OtpPurpose.PASSWORD_RESET)
                if o_rec and o_rec.otp_code == reset_token_or_otp.strip():
                    if o_rec.expires_at < now:
                        raise ValueError("Password reset OTP has expired.")
                    otp_id_to_mark = o_rec.id

        if not user:
            raise ValueError("Invalid or expired password reset token / OTP code.")

        # Update password
        user.password_hash = PasswordHasher.hash_password(new_password)
        self.user_repo.update_user(user)

        if token_id_to_mark:
            self.user_repo.mark_token_used(token_id_to_mark)
        if otp_id_to_mark:
            self.user_repo.mark_otp_used(otp_id_to_mark)

        self.user_repo.log_security_event(SecurityAuditLog(
            id=None, timestamp=now, event_type="PASSWORD_RESET_SUCCESS", user_id=user.id, email=user.email,
            ip_address=ip_address, status="SUCCESS", details="Password reset completed"
        ))

        return {"message": "Password reset successfully. You can now login with your new password."}

    def update_profile(self, user_id: int, full_name: Optional[str] = None, enable_2fa: Optional[bool] = None) -> Dict[str, Any]:
        """Updates user profile preferences."""
        user = self.user_repo.get_by_id(user_id)
        if not user:
            raise ValueError("User not found.")

        if full_name and full_name.strip():
            user.full_name = full_name.strip()
        if enable_2fa is not None:
            user.two_factor_enabled = enable_2fa

        updated = self.user_repo.update_user(user)
        return {"message": "Profile updated successfully.", "user": updated.to_dict()}

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
