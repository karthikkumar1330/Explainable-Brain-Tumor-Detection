import os
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Header, Request, status
from pydantic import BaseModel, EmailStr, Field

from security.domain.entities import Role, User
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.jwt_service import JWTService
from security.application.use_cases import AuthUseCases

DEFAULT_DB_PATH = os.environ.get("DB_PATH", "outputs/clinical_reports.db")

auth_router = APIRouter(prefix="/auth", tags=["Authentication"])
admin_router = APIRouter(prefix="/admin", tags=["Admin User Management"])


def get_auth_use_cases() -> AuthUseCases:
    repo = SQLiteUserRepository(db_path=DEFAULT_DB_PATH)
    repo.initialize_security_tables()
    jwt_svc = JWTService()
    return AuthUseCases(user_repo=repo, jwt_service=jwt_svc)



# Dependencies for Auth & RBAC
def extract_token_from_header(authorization: Optional[str] = Header(None)) -> Optional[str]:
    if not authorization:
        return None
    parts = authorization.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return authorization


def get_current_user(
    authorization: Optional[str] = Header(None),
    use_cases: AuthUseCases = Depends(get_auth_use_cases)
) -> User:
    token = extract_token_from_header(authorization)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token missing in Authorization header.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check JTI revocation
    payload = use_cases.jwt_service.decode_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if use_cases.user_repo.is_token_revoked(payload.jti):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token has been revoked (logged out).",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = use_cases.user_repo.get_by_id(payload.user_id)
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account does not exist or is inactive.",
        )

    return user


def require_roles(allowed_roles: List[Role]):
    def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Privilege level '{current_user.role.value}' is not authorized. Required: {[r.value for r in allowed_roles]}"
            )
        return current_user
    return role_checker


# Pydantic Schemas
class RegisterSchema(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: str = Field(..., min_length=2)
    role: str = "patient"


class VerifyEmailSchema(BaseModel):
    token_or_otp: str
    email: Optional[str] = None


class LoginSchema(BaseModel):
    email: EmailStr
    password: str


class VerifyOtpSchema(BaseModel):
    user_id: int
    otp_code: str


class LogoutSchema(BaseModel):
    token: Optional[str] = None


class ForgotPasswordSchema(BaseModel):
    email: EmailStr


class ResetPasswordSchema(BaseModel):
    reset_token_or_otp: str
    new_password: str = Field(..., min_length=8)
    email: Optional[str] = None


class ProfileUpdateSchema(BaseModel):
    full_name: Optional[str] = None
    enable_2fa: Optional[bool] = None


class ChangePasswordSchema(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8)


class AdminRoleUpdateSchema(BaseModel):
    role: str


class AdminStatusUpdateSchema(BaseModel):
    is_active: bool


# Router Endpoints
@auth_router.post("/register")
def register(data: RegisterSchema, request: Request, use_cases: AuthUseCases = Depends(get_auth_use_cases)):
    client_ip = request.client.host if request.client else "127.0.0.1"
    try:
        res = use_cases.register(
            email=data.email,
            password=data.password,
            full_name=data.full_name,
            role_str=data.role,
            ip_address=client_ip
        )
        return res
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@auth_router.post("/verify-email")
def verify_email(data: VerifyEmailSchema, request: Request, use_cases: AuthUseCases = Depends(get_auth_use_cases)):
    client_ip = request.client.host if request.client else "127.0.0.1"
    try:
        return use_cases.verify_email(token_or_otp=data.token_or_otp, email=data.email, ip_address=client_ip)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@auth_router.post("/login")
def login(data: LoginSchema, request: Request, use_cases: AuthUseCases = Depends(get_auth_use_cases)):
    client_ip = request.client.host if request.client else "127.0.0.1"
    try:
        return use_cases.login(email=data.email, password=data.password, ip_address=client_ip)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@auth_router.post("/verify-otp")
def verify_otp(data: VerifyOtpSchema, request: Request, use_cases: AuthUseCases = Depends(get_auth_use_cases)):
    client_ip = request.client.host if request.client else "127.0.0.1"
    try:
        return use_cases.verify_login_otp(user_id=data.user_id, otp_code=data.otp_code, ip_address=client_ip)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@auth_router.post("/logout")
def logout(
    data: LogoutSchema,
    request: Request,
    authorization: Optional[str] = Header(None),
    use_cases: AuthUseCases = Depends(get_auth_use_cases)
):
    token = data.token or extract_token_from_header(authorization)
    if not token:
        return {"message": "Logout successful (no active token)."}
    client_ip = request.client.host if request.client else "127.0.0.1"
    return use_cases.logout(token=token, ip_address=client_ip)


@auth_router.post("/forgot-password")
def forgot_password(data: ForgotPasswordSchema, request: Request, use_cases: AuthUseCases = Depends(get_auth_use_cases)):
    client_ip = request.client.host if request.client else "127.0.0.1"
    return use_cases.forgot_password(email=data.email, ip_address=client_ip)


@auth_router.post("/reset-password")
def reset_password(data: ResetPasswordSchema, request: Request, use_cases: AuthUseCases = Depends(get_auth_use_cases)):
    client_ip = request.client.host if request.client else "127.0.0.1"
    try:
        return use_cases.reset_password(
            reset_token_or_otp=data.reset_token_or_otp,
            new_password=data.new_password,
            email=data.email,
            ip_address=client_ip
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@auth_router.get("/profile")
def get_profile(current_user: User = Depends(get_current_user)):
    return {"user": current_user.to_dict()}


@auth_router.put("/profile")
def update_profile(
    data: ProfileUpdateSchema,
    current_user: User = Depends(get_current_user),
    use_cases: AuthUseCases = Depends(get_auth_use_cases)
):
    try:
        return use_cases.update_profile(
            user_id=current_user.id,
            full_name=data.full_name,
            enable_2fa=data.enable_2fa
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@auth_router.post("/change-password")
def change_password(
    data: ChangePasswordSchema,
    request: Request,
    current_user: User = Depends(get_current_user),
    use_cases: AuthUseCases = Depends(get_auth_use_cases)
):
    client_ip = request.client.host if request.client else "127.0.0.1"
    try:
        return use_cases.change_password(
            user_id=current_user.id,
            current_password=data.current_password,
            new_password=data.new_password,
            ip_address=client_ip
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# Admin Endpoints
@admin_router.get("/users")
def list_users(
    limit: int = 100,
    offset: int = 0,
    current_user: User = Depends(require_roles([Role.ADMIN])),
    use_cases: AuthUseCases = Depends(get_auth_use_cases)
):
    users = use_cases.user_repo.list_users(limit=limit, offset=offset)
    return {"users": [u.to_dict() for u in users]}


@admin_router.put("/users/{target_user_id}/role")
def update_user_role(
    target_user_id: int,
    data: AdminRoleUpdateSchema,
    current_user: User = Depends(require_roles([Role.ADMIN])),
    use_cases: AuthUseCases = Depends(get_auth_use_cases)
):
    user = use_cases.user_repo.get_by_id(target_user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    user.role = Role.from_string(data.role)
    updated = use_cases.user_repo.update_user(user)
    return {"message": f"User role updated to {user.role.value}", "user": updated.to_dict()}


@admin_router.put("/users/{target_user_id}/status")
def update_user_status(
    target_user_id: int,
    data: AdminStatusUpdateSchema,
    current_user: User = Depends(require_roles([Role.ADMIN])),
    use_cases: AuthUseCases = Depends(get_auth_use_cases)
):
    user = use_cases.user_repo.get_by_id(target_user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    user.is_active = data.is_active
    updated = use_cases.user_repo.update_user(user)
    return {"message": f"User active status set to {user.is_active}", "user": updated.to_dict()}


@admin_router.get("/audit-logs")
def get_audit_logs(
    limit: int = 100,
    current_user: User = Depends(require_roles([Role.ADMIN])),
    use_cases: AuthUseCases = Depends(get_auth_use_cases)
):
    logs = use_cases.user_repo.get_security_audit_logs(limit=limit)
    return {"audit_logs": logs}
