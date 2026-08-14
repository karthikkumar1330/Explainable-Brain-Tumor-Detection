import os
import sqlite3
import functools
from flask import Flask, jsonify, request, send_file, render_template_string, session, redirect, url_for
from typing import Optional, List, Dict, Any, Tuple

from persistence.infrastructure.repository import SQLitePersistenceRepository
from prediction_history.infrastructure.repository import SQLitePredictionHistoryRepository
from prediction_history.domain.entities import HistorySearchCriteria

from security.domain.entities import Role, User, TokenType
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.jwt_service import JWTService, TokenExpiredError, TokenInvalidError
from security.application.use_cases import AuthUseCases
from security.application.profile_image_service import validate_and_resize_avatar

def validate_version_param(version_val: Any) -> Optional[int]:
    """Validates the version parameter strictly.
    Returns:
        The validated version as an int, or None if the parameter was not provided (to use current version).
    Raises:
        ValueError if the version parameter is malformed.
    """
    if version_val is None:
        return None

    # If it's a string, strip it and check if it's empty
    if isinstance(version_val, str):
        val_str = version_val.strip()
        if not val_str:
            raise ValueError("Version cannot be empty or whitespace.")
        # Check if it has a decimal point, reject
        if "." in val_str:
            raise ValueError("Version must be a positive integer, not a decimal.")
        # Reject booleans represented as strings
        if val_str.lower() in ("true", "false"):
            raise ValueError("Version cannot be a boolean.")
        try:
            val_int = int(val_str)
        except ValueError:
            raise ValueError("Version must be a valid integer.")
    elif isinstance(version_val, bool):
        # Python's bool is a subclass of int, so isinstance(True, int) is True! We must check bool explicitly.
        raise ValueError("Version cannot be a boolean.")
    elif isinstance(version_val, float):
        raise ValueError("Version must be an integer, not a float.")
    elif isinstance(version_val, int):
        val_int = version_val
    else:
        raise ValueError("Invalid version type.")

    if val_int <= 0:
        raise ValueError("Version must be a positive integer.")
    return val_int


def create_app(db_path: str) -> Flask:
    """Factory function to build and configure the Flask web dashboard application with OWASP security.

    Args:
        db_path: Path to the SQLite database.

    Returns:
        Configured Flask application instance.
    """
    template_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "presentation", "templates")
    )
    static_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "presentation", "static")
    )
    app = Flask(__name__, template_folder=template_dir, static_folder=static_dir, static_url_path="/static")
    app.config["DB_PATH"] = db_path
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", "aurascan_dashboard_secret_key_84739201923")

    # Repositories & Security Services
    persistence_repo = SQLitePersistenceRepository(db_path=db_path)
    history_repo = SQLitePredictionHistoryRepository(db_path=db_path)
    user_repo = SQLiteUserRepository(db_path=db_path)
    jwt_svc = JWTService()
    auth_use_cases = AuthUseCases(user_repo=user_repo, jwt_service=jwt_svc)

    # Initialize database schemas and bootstrap admin
    persistence_repo.initialize_db()
    user_repo.initialize_security_tables()
    user_repo.bootstrap_admin()

    # Security Headers and CSRF Token Cookie Middleware
    @app.after_request
    def set_security_headers_and_csrf(response):
        # Set CSRF Token Cookie if not exists
        if not request.cookies.get("csrf_token"):
            import secrets
            csrf_val = secrets.token_hex(32)
            response.set_cookie(
                "csrf_token",
                csrf_val,
                samesite="Lax",
                secure=False,  # False for local dev server compatibility
                httponly=False,  # Must be False so frontend JS can read and submit it
                path="/"
            )

        # Configure robust Enterprise Security Headers
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net https://cdn.tailwindcss.com https://cdnjs.cloudflare.com; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com https://fonts.googleapis.com; "
            "font-src 'self' https://cdnjs.cloudflare.com https://fonts.gstatic.com; "
            "img-src 'self' data: https://lh3.googleusercontent.com; "
            "connect-src 'self';"
        )
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response

    @app.before_request
    def csrf_protect():
        if app.config.get("DISABLE_CSRF", app.config.get("TESTING", False)):
            return

        # Exclude public sign-in and registration from CSRF
        exempt_paths = [
            "/api/auth/login",
            "/api/auth/register",
            "/api/auth/verify-email",
            "/api/auth/resend-verification",
            "/api/auth/forgot-password",
            "/api/auth/reset-password"
        ]

        if request.method in ["POST", "PUT", "DELETE", "PATCH"]:
            if not request.path.startswith("/api/") or any(request.path.startswith(p) for p in exempt_paths):
                return

            cookie_csrf = request.cookies.get("csrf_token")
            header_csrf = request.headers.get("X-CSRF-Token")

            # Support bypass in standard unit tests if needed
            if app.config.get("TESTING") and header_csrf == "SKIP_CSRF_FOR_TESTS":
                return

            if not cookie_csrf or not header_csrf or cookie_csrf != header_csrf:
                user, _ = get_current_user_from_request()
                import datetime
                from security.domain.entities import SecurityAuditLog
                user_repo.log_security_event(SecurityAuditLog(
                    id=None,
                    timestamp=datetime.datetime.utcnow().isoformat(),
                    event_type="CSRF_ATTEMPT",
                    user_id=user.id if user else None,
                    email=user.email if user else None,
                    ip_address=request.remote_addr or "127.0.0.1",
                    status="BLOCKED",
                    details="CSRF token mismatch",
                    user_agent=request.headers.get("User-Agent", "Unknown")
                ))
                return jsonify({"error": "CSRF verification failed. Request blocked."}), 403

    # Auth Helper
    def get_current_user_from_request() -> Tuple[Optional[User], Optional[str]]:
        # Check Bearer Header
        auth_header = request.headers.get("Authorization")
        token = None
        if auth_header:
            parts = auth_header.split()
            if len(parts) == 2 and parts[0].lower() == "bearer":
                token = parts[1]
            else:
                token = auth_header
        else:
            token = request.cookies.get("access_token")

        if not token:
            return None, "TOKEN_MISSING"

        try:
            payload = jwt_svc.decode_token(token, expected_type=TokenType.ACCESS)
        except TokenExpiredError:
            return None, "TOKEN_EXPIRED"
        except TokenInvalidError:
            return None, "TOKEN_INVALID"

        if user_repo.is_token_revoked(payload.jti):
            return None, "TOKEN_REVOKED"

        user = user_repo.get_by_id(payload.user_id)
        if not user:
            return None, "USER_NOT_FOUND"
        if not user.is_active:
            return None, "USER_INACTIVE"

        if user.sessions_revoked_at:
            import datetime
            try:
                rev_str = user.sessions_revoked_at.replace("Z", "")
                if "+" in rev_str:
                    rev_dt = datetime.datetime.fromisoformat(rev_str)
                else:
                    rev_dt = datetime.datetime.fromisoformat(rev_str).replace(tzinfo=datetime.timezone.utc)
                if payload.iat <= int(rev_dt.timestamp()):
                    return None, "TOKEN_REVOKED"
            except Exception as e:
                # Fail-closed: treat parsing failures as revoked session
                return None, "TOKEN_REVOKED"

        return user, None

    def login_required(f):
        @functools.wraps(f)
        def decorated(*args, **kwargs):
            user, err_code = get_current_user_from_request()
            if not user:
                error_msg = "Authentication required. Please login."
                if err_code == "TOKEN_EXPIRED":
                    error_msg = "Your session has expired. Please login again."
                return jsonify({"error": error_msg, "code": err_code}), 401
            return f(user, *args, **kwargs)
        return decorated

    def roles_accepted(*roles: Role):
        def decorator(f):
            @functools.wraps(f)
            def decorated(*args, **kwargs):
                user, err_code = get_current_user_from_request()
                if not user:
                    error_msg = "Authentication required. Please login."
                    if err_code == "TOKEN_EXPIRED":
                        error_msg = "Your session has expired. Please login again."
                    return jsonify({"error": error_msg, "code": err_code}), 401
                if user.role not in roles and user.role != Role.ADMIN:
                    return jsonify({"error": f"Access denied. Required privileges: {[r.value for r in roles]}"}), 403
                return f(user, *args, **kwargs)
            return decorated
        return decorator

    # --- Web Routes ---
    @app.route("/")
    def index():
        user, err_code = get_current_user_from_request()
        if user:
            if user.role == Role.ADMIN:
                return redirect(url_for("admin_dashboard"))
            elif user.role == Role.DOCTOR:
                return redirect(url_for("doctor_dashboard"))
            elif user.role == Role.PATIENT:
                return redirect(url_for("patient_dashboard"))

        index_path = os.path.join(template_dir, "index.html")
        if not os.path.exists(index_path):
            return f"Error: index.html presentation template not found at {index_path}", 404

        with open(index_path, "r", encoding="utf-8") as f:
            content = f.read()
        return render_template_string(content)

    @app.route("/verify-email")
    def page_verify_email():
        token = request.args.get("token")
        from flask import render_template
        if not token:
            return render_template("verify_email.html", success=False, error="Verification token is missing.")
        try:
            auth_use_cases.verify_email(raw_token=token)
            return render_template("verify_email.html", success=True)
        except ValueError as e:
            return render_template("verify_email.html", success=False, error=str(e))

    @app.route("/reset-password")
    def page_reset_password():
        token = request.args.get("token")
        from flask import render_template
        if not token:
            return render_template("reset_password.html", valid=False, error="Password reset token is missing.")

        import hashlib
        import datetime
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        token_rec = user_repo.get_password_reset_token(token_hash)

        if not token_rec or token_rec["used_at"]:
            return render_template("reset_password.html", valid=False, error="Password reset token is invalid or has already been used.")

        expires_at = datetime.datetime.fromisoformat(token_rec["expires_at"])
        if datetime.datetime.utcnow() > expires_at:
            return render_template("reset_password.html", valid=False, error="Password reset token has expired.")

        user = user_repo.get_by_id(token_rec["user_id"])
        if not user:
            return render_template("reset_password.html", valid=False, error="User account not found.")

        return render_template("reset_password.html", valid=True, token=token, email=user.email)

    @app.route("/admin")
    def admin_dashboard():
        user, err_code = get_current_user_from_request()
        if not user or user.role != Role.ADMIN:
            return redirect(url_for("index"))

        path = os.path.join(template_dir, "dashboard_admin.html")
        if not os.path.exists(path):
            return f"Error: dashboard_admin.html not found", 404
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return render_template_string(content)

    @app.route("/doctor")
    def doctor_dashboard():
        user, err_code = get_current_user_from_request()
        if not user or user.role != Role.DOCTOR:
            return redirect(url_for("index"))

        path = os.path.join(template_dir, "dashboard_doctor.html")
        if not os.path.exists(path):
            return f"Error: dashboard_doctor.html not found", 404
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return render_template_string(content)

    @app.route("/patient")
    def patient_dashboard():
        user, err_code = get_current_user_from_request()
        if not user or user.role != Role.PATIENT:
            return redirect(url_for("index"))

        path = os.path.join(template_dir, "dashboard_patient.html")
        if not os.path.exists(path):
            return f"Error: dashboard_patient.html not found", 404
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return render_template_string(content)

    # --- Authentication API Routes ---
    @app.route("/api/auth/register", methods=["POST"])
    def auth_register():
        data = request.get_json() or {}
        ip_addr = request.remote_addr or "127.0.0.1"
        try:
            res = auth_use_cases.register(
                email=data.get("email", ""),
                password=data.get("password", ""),
                full_name=data.get("full_name", ""),
                role_str=data.get("role", "patient"),
                ip_address=ip_addr
            )
            response = jsonify(res)
            if "access_token" in res:
                is_secure = request.is_secure or request.headers.get("X-Forwarded-Proto", "").lower() == "https"
                response.set_cookie(
                    "access_token",
                    res["access_token"],
                    max_age=30 * 60,
                    httponly=True,
                    secure=is_secure,
                    samesite="Lax",
                    path="/"
                )
                response.set_cookie(
                    "refresh_token",
                    res["refresh_token"],
                    max_age=30 * 86400,
                    httponly=True,
                    secure=is_secure,
                    samesite="Lax",
                    path="/"
                )
            return response
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/api/auth/login", methods=["POST"])
    def auth_login():
        data = request.get_json() or {}
        ip_addr = request.remote_addr or "127.0.0.1"
        remember_me = data.get("remember_me", False)
        try:
            res = auth_use_cases.login(
                email=data.get("email", ""),
                password=data.get("password", ""),
                ip_address=ip_addr,
                remember_me=remember_me,
                user_agent=request.headers.get("User-Agent", "Unknown")
            )
            response = jsonify(res)
            if "access_token" in res:
                is_secure = request.is_secure or request.headers.get("X-Forwarded-Proto", "").lower() == "https"
                response.set_cookie(
                    "access_token",
                    res["access_token"],
                    max_age=30 * 60,
                    httponly=True,
                    secure=is_secure,
                    samesite="Lax",
                    path="/"
                )
                refresh_max_age = 30 * 86400 if remember_me else None
                response.set_cookie(
                    "refresh_token",
                    res["refresh_token"],
                    max_age=refresh_max_age,
                    httponly=True,
                    secure=is_secure,
                    samesite="Lax",
                    path="/"
                )
            return response
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/api/auth/verify-email", methods=["POST"])
    def auth_verify_email():
        ip_addr = request.remote_addr or "127.0.0.1"
        from security.application.use_cases import is_testing_env
        if not is_testing_env():
            from security.infrastructure.rate_limiter import global_rate_limiter
            limited, remaining = global_rate_limiter.is_rate_limited(f"verify_email:{ip_addr}", max_requests=5, window_seconds=300)
            if limited:
                return jsonify({"error": f"Rate limit exceeded. Please try again in {remaining} seconds."}), 429

        data = request.get_json() or {}
        token = data.get("token")
        if not token:
            return jsonify({"error": "Token is required."}), 400
        try:
            res = auth_use_cases.verify_email(raw_token=token)
            return jsonify(res)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/api/auth/resend-verification", methods=["POST"])
    def auth_resend_verification():
        ip_addr = request.remote_addr or "127.0.0.1"
        from security.application.use_cases import is_testing_env
        if not is_testing_env():
            from security.infrastructure.rate_limiter import global_rate_limiter
            limited, remaining = global_rate_limiter.is_rate_limited(f"resend_verification:{ip_addr}", max_requests=5, window_seconds=300)
            if limited:
                return jsonify({"error": f"Rate limit exceeded. Please try again in {remaining} seconds."}), 429

        data = request.get_json() or {}
        email = data.get("email")
        if not email:
            return jsonify({"error": "Email is required."}), 400
        res = auth_use_cases.resend_verification(email=email)
        return jsonify(res)

    @app.route("/api/auth/forgot-password", methods=["POST"])
    def auth_forgot_password():
        ip_addr = request.remote_addr or "127.0.0.1"
        from security.application.use_cases import is_testing_env
        if not is_testing_env():
            from security.infrastructure.rate_limiter import global_rate_limiter
            limited, remaining = global_rate_limiter.is_rate_limited(f"forgot_password:{ip_addr}", max_requests=5, window_seconds=300)
            if limited:
                return jsonify({"error": f"Rate limit exceeded. Please try again in {remaining} seconds."}), 429

        data = request.get_json() or {}
        email = data.get("email")
        if not email:
            return jsonify({"error": "Email is required."}), 400
        try:
            res = auth_use_cases.forgot_password(email=email, ip_address=ip_addr)
            return jsonify(res)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/api/auth/reset-password", methods=["POST"])
    def auth_reset_password():
        ip_addr = request.remote_addr or "127.0.0.1"
        from security.application.use_cases import is_testing_env
        if not is_testing_env():
            from security.infrastructure.rate_limiter import global_rate_limiter
            limited, remaining = global_rate_limiter.is_rate_limited(f"reset_password:{ip_addr}", max_requests=5, window_seconds=300)
            if limited:
                return jsonify({"error": f"Rate limit exceeded. Please try again in {remaining} seconds."}), 429

        data = request.get_json() or {}
        code = data.get("reset_token_or_otp") or data.get("token")
        email = data.get("email")
        new_password = data.get("new_password")

        if not code or not email or not new_password:
            return jsonify({"error": "Missing required fields."}), 400

        try:
            res = auth_use_cases.reset_password(
                reset_token_or_otp=code,
                email=email,
                new_password=new_password,
                ip_address=ip_addr
            )
            return jsonify(res)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400


    @app.route("/api/auth/refresh", methods=["POST"])
    def auth_refresh():
        refresh_token = request.cookies.get("refresh_token")
        if not refresh_token:
            data = request.get_json() or {}
            refresh_token = data.get("refresh_token")

        if not refresh_token:
            return jsonify({"error": "Refresh token is missing.", "code": "REFRESH_TOKEN_MISSING"}), 401

        ip_addr = request.remote_addr or "127.0.0.1"
        try:
            res = auth_use_cases.refresh_token(refresh_token=refresh_token, ip_address=ip_addr)
            response = jsonify(res)

            is_secure = request.is_secure or request.headers.get("X-Forwarded-Proto", "").lower() == "https"

            try:
                payload = jwt_svc.decode_token(res["refresh_token"], expected_type=TokenType.REFRESH)
                duration_days = (payload.exp - payload.iat) / 86400.0
                remember_me = duration_days > 8.0
            except Exception:
                remember_me = False

            response.set_cookie(
                "access_token",
                res["access_token"],
                max_age=30 * 60,
                httponly=True,
                secure=is_secure,
                samesite="Lax",
                path="/"
            )

            refresh_max_age = 30 * 86400 if remember_me else None
            response.set_cookie(
                "refresh_token",
                res["refresh_token"],
                max_age=refresh_max_age,
                httponly=True,
                secure=is_secure,
                samesite="Lax",
                path="/"
            )
            return response
        except ValueError as e:
            app.logger.warning(f"Refresh token validation failed: {e}")
            from flask import make_response
            response = make_response(jsonify({"error": "Invalid or expired refresh token.", "code": "REFRESH_TOKEN_INVALID"}), 401)
            response.set_cookie("access_token", "", expires=0, httponly=True, samesite="Lax", path="/")
            response.set_cookie("refresh_token", "", expires=0, httponly=True, samesite="Lax", path="/")
            return response
        except Exception as e:
            app.logger.error(f"Unexpected error in refresh token: {e}", exc_info=True)
            from flask import make_response
            response = make_response(jsonify({"error": "An unexpected error occurred.", "code": "INTERNAL_SERVER_ERROR"}), 500)
            response.set_cookie("access_token", "", expires=0, httponly=True, samesite="Lax", path="/")
            response.set_cookie("refresh_token", "", expires=0, httponly=True, samesite="Lax", path="/")
            return response

    @app.route("/api/auth/logout", methods=["POST"])
    def auth_logout():
        token = request.cookies.get("access_token")
        auth_header = request.headers.get("Authorization")
        if auth_header:
            parts = auth_header.split()
            if len(parts) == 2 and parts[0].lower() == "bearer":
                token = parts[1]
            else:
                token = auth_header

        refresh_token = request.cookies.get("refresh_token")
        ip_addr = request.remote_addr or "127.0.0.1"
        if token or refresh_token:
            auth_use_cases.logout(token=token or "", refresh_token=refresh_token, ip_address=ip_addr)

        response = jsonify({"message": "Logout successful."})
        response.set_cookie("access_token", "", expires=0, httponly=True, samesite="Lax", path="/")
        response.set_cookie("refresh_token", "", expires=0, httponly=True, samesite="Lax", path="/")
        return response

    @app.route("/api/auth/me", methods=["GET"])
    def auth_me():
        user, err_code = get_current_user_from_request()
        if not user:
            response = jsonify({"authenticated": False, "code": err_code})
            response.set_cookie("access_token", "", expires=0, httponly=True, samesite="Lax", path="/")
            return response

        response = jsonify({"authenticated": True, "user": user.to_dict()})
        auth_header = request.headers.get("Authorization")
        if auth_header:
            parts = auth_header.split()
            token = parts[1] if len(parts) == 2 and parts[0].lower() == "bearer" else auth_header
            if request.cookies.get("access_token") != token:
                is_secure = request.is_secure or request.headers.get("X-Forwarded-Proto", "").lower() == "https"
                response.set_cookie(
                    "access_token",
                    token,
                    max_age=30 * 60,
                    httponly=True,
                    secure=is_secure,
                    samesite="Lax",
                    path="/"
                )
        return response

    @app.route("/api/auth/profile", methods=["PUT"])
    @login_required
    def auth_update_profile(current_user: User):
        data = request.get_json() or {}
        try:
            res = auth_use_cases.update_profile(
                user_id=current_user.id,
                full_name=data.get("full_name"),
                email=data.get("email"),
                enable_2fa=data.get("enable_2fa")
            )
            return jsonify(res)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/api/auth/verify-otp", methods=["POST"])
    def auth_verify_otp():
        data = request.get_json() or {}
        user_id = data.get("user_id")
        otp_code = data.get("otp_code", "")
        remember_me = data.get("remember_me", False)
        ip_addr = request.remote_addr or "127.0.0.1"

        if not user_id:
            return jsonify({"error": "User ID is required."}), 400

        try:
            res = auth_use_cases.verify_tfa_otp(
                user_id=int(user_id),
                otp_code=otp_code,
                ip_address=ip_addr,
                remember_me=remember_me,
                user_agent=request.headers.get("User-Agent", "Unknown")
            )
            response = jsonify(res)

            # If access token was successfully generated, set the cookies
            if "access_token" in res:
                is_secure = request.is_secure or request.headers.get("X-Forwarded-Proto", "").lower() == "https"
                response.set_cookie(
                    "access_token",
                    res["access_token"],
                    max_age=30 * 60,
                    httponly=True,
                    secure=is_secure,
                    samesite="Lax",
                    path="/"
                )
                refresh_max_age = 30 * 86400 if remember_me else None
                response.set_cookie(
                    "refresh_token",
                    res["refresh_token"],
                    max_age=refresh_max_age,
                    httponly=True,
                    secure=is_secure,
                    samesite="Lax",
                    path="/"
                )
            return response
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/api/auth/profile/avatar", methods=["POST"])
    @login_required
    def auth_upload_avatar(current_user: User):
        if "avatar" not in request.files:
            return jsonify({"error": "No avatar file provided."}), 400
        file = request.files["avatar"]
        if file.filename == "":
            return jsonify({"error": "No selected file."}), 400

        # Create uploads folder programmatically
        upload_dir = os.path.abspath(os.path.join(app.root_path, "..", "uploads", "avatars"))
        os.makedirs(upload_dir, exist_ok=True)

        # Validate file extension and format
        file_ext = os.path.splitext(file.filename)[1].lower()
        try:
            resized_bytes = validate_and_resize_avatar(file.stream, file.filename)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        filename = f"{current_user.uuid}{file_ext}"
        filepath = os.path.join(upload_dir, filename)

        # Save the validated and resized image bytes
        with open(filepath, "wb") as f:
            f.write(resized_bytes)

        # Update profile URL
        avatar_url = f"/uploads/avatars/{filename}"
        current_user.profile_pic = avatar_url
        user_repo.update_user(current_user)

        return jsonify({
            "message": "Avatar uploaded successfully.",
            "avatar_url": avatar_url,
            "user": current_user.to_dict()
        })

    @app.route("/uploads/avatars/<filename>")
    def serve_avatar(filename):
        upload_dir = os.path.abspath(os.path.join(app.root_path, "..", "uploads", "avatars"))
        filepath = os.path.join(upload_dir, filename)
        if not os.path.exists(filepath):
            return "File not found", 404
        return send_file(filepath)

    @app.route("/api/auth/profile/logout-other-devices", methods=["POST"])
    @login_required
    def auth_logout_other_devices(current_user: User):
        import datetime
        import time
        now = datetime.datetime.utcnow().isoformat()
        # Set sessions_revoked_at to 1 second ago so the new token's iat (which is now) is valid
        current_user.sessions_revoked_at = (datetime.datetime.utcnow() - datetime.timedelta(seconds=1)).isoformat()
        user_repo.update_user(current_user)

        # Get current session JTI from request token if possible, to avoid revoking ourselves globally!
        token = request.cookies.get("access_token")
        auth_header = request.headers.get("Authorization")
        if auth_header:
            parts = auth_header.split()
            if len(parts) == 2 and parts[0].lower() == "bearer":
                token = parts[1]
            else:
                token = auth_header

        current_jti = None
        if token:
            try:
                payload = jwt_svc.decode_token(token, verify_exp=False)
                current_jti = payload.jti
            except Exception:
                pass

        # Revoke other sessions in database & update their logout times
        user_repo.revoke_other_sessions(current_user.id, current_jti, now)

        # Revoke the current token itself (so the old token is explicitly blocked, but the fresh new one is valid!)
        if current_jti:
            user_repo.revoke_token(current_jti, current_user.id, time.time() + 86400)
            user_repo.update_session_logout(current_jti, now)

        # Generate fresh tokens for the current session
        access_token = jwt_svc.create_access_token(
            user_uuid=current_user.uuid,
            user_id=current_user.id,
            email=current_user.email,
            role=current_user.role
        )
        refresh_token = jwt_svc.create_refresh_token(
            user_uuid=current_user.uuid,
            user_id=current_user.id,
            email=current_user.email,
            role=current_user.role,
            expires_days=30
        )

        response = jsonify({
            "message": "Logged out of all other devices successfully.",
            "access_token": access_token,
            "refresh_token": refresh_token,
            "user": current_user.to_dict()
        })

        is_secure = request.is_secure or request.headers.get("X-Forwarded-Proto", "").lower() == "https"
        response.set_cookie("access_token", access_token, max_age=30*60, httponly=True, secure=is_secure, samesite="Lax", path="/")
        response.set_cookie("refresh_token", refresh_token, max_age=30*86400, httponly=True, secure=is_secure, samesite="Lax", path="/")

        # Log audit trail
        from security.domain.entities import SecurityAuditLog
        audit_log = SecurityAuditLog(
            id=None,
            timestamp=now,
            event_type="REVOKE_ALL_SESSIONS",
            user_id=current_user.id,
            email=current_user.email,
            ip_address=request.remote_addr or "127.0.0.1",
            status="SUCCESS",
            details="All other active sessions revoked."
        )
        user_repo.log_security_event(audit_log)

        return response

    @app.route("/api/auth/profile", methods=["DELETE"])
    @login_required
    def auth_delete_account(current_user: User):
        import datetime
        now = datetime.datetime.utcnow().isoformat()
        try:
            # Audit log user deletion before wiping record
            from security.domain.entities import SecurityAuditLog
            audit_log = SecurityAuditLog(
                id=None,
                timestamp=now,
                event_type="ACCOUNT_DELETE",
                user_id=current_user.id,
                email=current_user.email,
                ip_address=request.remote_addr or "127.0.0.1",
                status="SUCCESS",
                details="User account deleted successfully."
            )
            user_repo.log_security_event(audit_log)

            user_repo.delete_user(current_user.id)

            # Clear session cookies
            response = jsonify({"message": "Your account has been deleted permanently."})
            response.set_cookie("access_token", "", expires=0, httponly=True, samesite="Lax", path="/")
            response.set_cookie("refresh_token", "", expires=0, httponly=True, samesite="Lax", path="/")
            return response
        except Exception as e:
            app.logger.error(f"Error deleting user account: {e}", exc_info=True)
            return jsonify({"error": "Internal profile deletion error"}), 500

    @app.route("/api/auth/profile/sessions", methods=["GET"])
    @login_required
    def auth_get_sessions(current_user: User):
        import json
        from dashboard.utils.timezone import convert_utc_to_ist
        conn = sqlite3.connect(app.config["DB_PATH"])
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT timestamp, event_type, ip_address, details
                FROM security_audit_logs
                WHERE user_id = ? AND (event_type LIKE 'LOGIN%' OR event_type = 'LOGOUT' OR event_type = 'REVOKE_ALL_SESSIONS')
                ORDER BY id DESC LIMIT 15;
            """, (current_user.id,))
            rows = cursor.fetchall()

            sessions = []
            for r in rows:
                d = dict(r)
                details_str = d.get("details", "{}")
                try:
                    details = json.loads(details_str)
                except Exception:
                    details = {}

                sessions.append({
                    "timestamp": convert_utc_to_ist(d.get("timestamp")),
                    "event_type": d.get("event_type"),
                    "ip_address": d.get("ip_address"),
                    "browser": details.get("browser", "Unknown Browser"),
                    "device": details.get("device", "Unknown Device"),
                    "location": details.get("location", "Unknown Location"),
                    "login_time": convert_utc_to_ist(details.get("login_time")),
                    "logout_time": convert_utc_to_ist(details.get("logout_time")),
                })
            return jsonify({"sessions": sessions})
        except Exception as e:
            app.logger.error(f"Error retrieving user sessions: {e}", exc_info=True)
            return jsonify({"error": "Internal profile sessions retrieval error"}), 500
        finally:
            conn.close()



    @app.route("/api/auth/change-password", methods=["POST"])
    @login_required
    def auth_change_password(current_user: User):
        data = request.get_json() or {}
        ip_addr = request.remote_addr or "127.0.0.1"
        try:
            res = auth_use_cases.change_password(
                user_id=current_user.id,
                current_password=data.get("current_password", ""),
                new_password=data.get("new_password", ""),
                ip_address=ip_addr
            )
            return jsonify(res)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    # --- Admin API Routes ---
    @app.route("/api/admin/users", methods=["GET"])
    @roles_accepted(Role.ADMIN)
    def admin_list_users(current_user: User):
        try:
            limit = int(request.args.get("limit", 200))
            offset = int(request.args.get("offset", 0))
        except ValueError:
            limit = 200
            offset = 0
        users = user_repo.list_users(limit=limit, offset=offset)
        return jsonify({"users": [u.to_dict() for u in users]})

    @app.route("/api/admin/users/<int:target_id>/password", methods=["PUT"])
    @roles_accepted(Role.ADMIN)
    def admin_reset_password(current_user: User, target_id: int):
        data = request.get_json() or {}
        new_pass = data.get("password")
        if not new_pass:
            return jsonify({"error": "Password is required."}), 400

        from security.infrastructure.password import PasswordHasher
        valid_pass, pass_err = PasswordHasher.validate_password_strength(new_pass)
        if not valid_pass:
            return jsonify({"error": pass_err}), 400

        user = user_repo.get_by_id(target_id)
        if not user:
            return jsonify({"error": "User not found"}), 404

        user.password_hash = PasswordHasher.hash_password(new_pass)
        user.failed_login_attempts = 0
        user.lockout_until = None
        user_repo.update_user(user)

        # Log audit trail
        import datetime
        from security.domain.entities import SecurityAuditLog
        audit_log = SecurityAuditLog(
            id=None,
            timestamp=datetime.datetime.utcnow().isoformat(),
            event_type="ADMIN_PASSWORD_RESET",
            user_id=current_user.id,
            email=current_user.email,
            ip_address=request.remote_addr or "127.0.0.1",
            status="SUCCESS",
            details=f"Admin reset password for user ID: {target_id} ({user.email}).",
            user_agent=request.headers.get("User-Agent", "Unknown")
        )
        user_repo.log_security_event(audit_log)

        return jsonify({"message": "Password reset successfully."})

    @app.route("/api/admin/users/export", methods=["GET"])
    @roles_accepted(Role.ADMIN)
    def admin_export_users_csv(current_user: User):
        import csv
        import io
        from flask import make_response

        users = user_repo.list_users(limit=1000)

        dest = io.StringIO()
        writer = csv.writer(dest)
        writer.writerow(["ID", "UUID", "Email", "Full Name", "Role", "Is Verified", "Is Active", "Created At", "Last Login At"])

        for u in users:
            writer.writerow([
                u.id, u.uuid, u.email, u.full_name, u.role.value if hasattr(u.role, 'value') else u.role,
                u.is_verified, u.is_active, u.created_at, u.last_login_at
            ])

        output = make_response(dest.getvalue())
        output.headers["Content-Disposition"] = "attachment; filename=users_export.csv"
        output.headers["Content-type"] = "text/csv"
        return output

    @app.route("/api/admin/users/<int:target_id>/role", methods=["PUT"])
    @roles_accepted(Role.ADMIN)
    def admin_update_role(current_user: User, target_id: int):
        data = request.get_json() or {}
        user = user_repo.get_by_id(target_id)
        if not user:
            return jsonify({"error": "User not found"}), 404
        user.role = Role.from_string(data.get("role", "patient"))
        updated = user_repo.update_user(user)
        return jsonify({"message": f"Updated role to {user.role.value}", "user": updated.to_dict()})

    @app.route("/api/admin/users/<int:target_id>/status", methods=["PUT"])
    @roles_accepted(Role.ADMIN)
    def admin_update_status(current_user: User, target_id: int):
        data = request.get_json() or {}
        user = user_repo.get_by_id(target_id)
        if not user:
            return jsonify({"error": "User not found"}), 404
        user.is_active = bool(data.get("is_active", True))
        updated = user_repo.update_user(user)
        return jsonify({"message": f"Updated active status to {user.is_active}", "user": updated.to_dict()})

    @app.route("/api/admin/audit-logs", methods=["GET"])
    @roles_accepted(Role.ADMIN)
    def admin_audit_logs(current_user: User):
        logs = user_repo.get_security_audit_logs(limit=100)
        return jsonify({"audit_logs": logs})

    # --- Existing Clinical Endpoints (Protected with RBAC) ---
    @app.route("/api/analytics")
    @login_required
    def analytics(current_user: User):
        try:
            summary = persistence_repo.get_analytics_summary()
            return jsonify(summary)
        except Exception as e:
            app.logger.error(f"Error compiling analytics widgets: {e}", exc_info=True)
            return jsonify({"error": "Internal analytics telemetry error"}), 500

    @app.route("/api/health-telemetry")
    @roles_accepted(Role.ADMIN, Role.DOCTOR)
    def health_telemetry(current_user: User):
        try:
            telemetry = persistence_repo.get_health_telemetry()
            return jsonify(telemetry)
        except Exception as e:
            app.logger.error(f"Error compiling health telemetry: {e}", exc_info=True)
            return jsonify({"error": "Internal health telemetry compilation error"}), 500

    @app.route("/api/history")
    @login_required
    def history(current_user: User):
        try:
            patient_id = request.args.get("patient_id", "").strip() or None
            if current_user.role == Role.PATIENT:
                patient_id = current_user.uuid

            restrict_doctor_id = current_user.id if current_user.role == Role.DOCTOR else None
            criteria = HistorySearchCriteria(patient_id=patient_id, restrict_to_doctor_id=restrict_doctor_id)
            summaries = history_repo.search_history(criteria)

            data = []
            for s in summaries:
                # If Patient role, only return matching patient records (strictly verified by unique ID to prevent name-collision IDOR)
                if current_user.role == Role.PATIENT:
                    if s.patient_id.lower() != current_user.uuid.lower():
                        continue
                if current_user.role == Role.DOCTOR:
                    from security.application.authorization_service import AuthorizationService
                    auth_svc = AuthorizationService(db_path=app.config["DB_PATH"])
                    if not auth_svc.can_access_patient(current_user, s.patient_id):
                        continue

                data.append({
                    "report_id": s.report_id,
                    "prediction_id": s.prediction_id,
                    "patient_id": s.patient_id,
                    "patient_name": s.patient_name,
                    "scan_date": s.scan_date,
                    "predicted_class": s.predicted_class,
                    "confidence_score": s.confidence_score,
                    "tumor_area_mm2": s.tumor_area_mm2,
                    "rule_based_severity": s.rule_based_severity,
                    "created_at": s.created_at,
                })
            return jsonify(data)
        except Exception as e:
            app.logger.error(f"Error querying prediction history: {e}", exc_info=True)
            return jsonify({"error": "Internal history query failure"}), 500

    @app.route("/api/search")
    @login_required
    def search(current_user: User):
        # 1. Parsing query parameters
        q = request.args.get("q", "").strip() or None
        patient_id = request.args.get("patient_id", "").strip() or None
        patient_name = request.args.get("patient_name", "").strip() or None
        referring_doctor = request.args.get("referring_doctor", "").strip() or None
        classification = request.args.get("classification", "").strip() or None
        severity = request.args.get("severity", "").strip() or None

        # Validate min_confidence range
        min_confidence_str = request.args.get("min_confidence", "").strip()
        min_confidence = None
        if min_confidence_str:
            try:
                min_confidence = float(min_confidence_str)
                if not (0.0 <= min_confidence <= 1.0):
                    return jsonify({"error": "min_confidence must be between 0.0 and 1.0"}), 400
            except ValueError:
                return jsonify({"error": "min_confidence must be a valid float"}), 400

        # Validate scan date formats
        import datetime
        start_date = request.args.get("start_date", "").strip() or None
        if start_date:
            try:
                datetime.datetime.strptime(start_date, "%Y-%m-%d")
            except ValueError:
                return jsonify({"error": "start_date must be in YYYY-MM-DD format"}), 400

        end_date = request.args.get("end_date", "").strip() or None
        if end_date:
            try:
                datetime.datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                return jsonify({"error": "end_date must be in YYYY-MM-DD format"}), 400

        report_status = request.args.get("report_status", "").strip() or None

        # Validate sorting fields
        sort_by = request.args.get("sort_by", "").strip() or None
        valid_sort_fields = [
            "report_id", "prediction_id", "patient_id", "patient_name",
            "scan_date", "predicted_class", "confidence_score", "tumor_area_mm2",
            "rule_based_severity", "created_at", "referring_doctor", "report_status"
        ]
        if sort_by and sort_by not in valid_sort_fields:
            return jsonify({"error": f"Invalid sort_by field. Must be one of {valid_sort_fields}"}), 400

        sort_order = request.args.get("sort_order", "").strip() or None
        if sort_order and sort_order.lower() not in ["asc", "desc"]:
            return jsonify({"error": "sort_order must be 'asc' or 'desc'"}), 400

        # Validate page / page_size
        page_str = request.args.get("page", "").strip()
        page = None
        if page_str:
            try:
                page = int(page_str)
                if page < 1:
                    return jsonify({"error": "page must be greater than or equal to 1"}), 400
            except ValueError:
                return jsonify({"error": "page must be a valid integer"}), 400

        page_size_str = request.args.get("page_size", "").strip()
        page_size = None
        if page_size_str:
            try:
                page_size = int(page_size_str)
                if not (1 <= page_size <= 100):
                    return jsonify({"error": "page_size must be between 1 and 100"}), 400
            except ValueError:
                return jsonify({"error": "page_size must be a valid integer"}), 400

        # Apply Patient RBAC restrictions
        restrict_uuid = None
        restrict_name = None
        if current_user.role == Role.PATIENT:
            restrict_uuid = current_user.uuid
            restrict_name = current_user.full_name

        restrict_doctor_id = current_user.id if current_user.role == Role.DOCTOR else None

        try:
            criteria = HistorySearchCriteria(
                patient_id=patient_id,
                report_id=None,
                scan_date=None,
                q=q,
                patient_name=patient_name,
                referring_doctor=referring_doctor,
                classification=classification,
                severity=severity,
                min_confidence=min_confidence,
                start_date=start_date,
                end_date=end_date,
                report_status=report_status,
                sort_by=sort_by,
                sort_order=sort_order,
                page=page,
                page_size=page_size,
                restrict_to_patient_uuid=restrict_uuid,
                restrict_to_patient_name=restrict_name,
                restrict_to_doctor_id=restrict_doctor_id
            )
            summaries = history_repo.search_history(criteria)

            data = []
            for s in summaries:
                # Defensive check in case database returned unfiltered records (RBAC boundary check)
                if current_user.role == Role.PATIENT:
                    if s.patient_name.lower() != current_user.full_name.lower() and s.patient_id.lower() != current_user.uuid.lower():
                        continue
                if current_user.role == Role.DOCTOR:
                    from security.application.authorization_service import AuthorizationService
                    auth_svc = AuthorizationService(db_path=app.config["DB_PATH"])
                    if not auth_svc.can_access_patient(current_user, s.patient_id):
                        continue
                data.append({
                    "report_id": s.report_id,
                    "prediction_id": s.prediction_id,
                    "patient_id": s.patient_id,
                    "patient_name": s.patient_name,
                    "scan_date": s.scan_date,
                    "predicted_class": s.predicted_class,
                    "confidence_score": s.confidence_score,
                    "tumor_area_mm2": s.tumor_area_mm2,
                    "rule_based_severity": s.rule_based_severity,
                    "created_at": s.created_at,
                    "referring_doctor": s.referring_doctor,
                    "report_status": s.report_status
                })

            if page is not None:
                total_count = getattr(summaries, "total_count", len(summaries))
                import math
                p_size = page_size or 10
                total_pages = math.ceil(total_count / p_size)
                return jsonify({
                    "items": data,
                    "total_count": total_count,
                    "page": page,
                    "page_size": p_size,
                    "total_pages": total_pages
                })
            else:
                return jsonify(data)
        except Exception as e:
            app.logger.error(f"Error during search query: {e}", exc_info=True)
            return jsonify({"error": "Internal search failure"}), 500

    @app.route("/api/notifications", methods=["GET"])
    @login_required
    def api_list_notifications(current_user: User):
        from clinical_reporting.application.notification_service import NotificationService
        service = NotificationService(db_path=app.config["DB_PATH"])
        try:
            page = int(request.args.get("page", 1))
            page_size = int(request.args.get("page_size", 10))
            type_filter = request.args.get("type") or None
            unread_only = request.args.get("unread_only", "").lower() == "true"

            res = service.list_notifications(
                user_id=current_user.id,
                page=page,
                page_size=page_size,
                type_filter=type_filter,
                unread_only=unread_only
            )
            return jsonify(res)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            app.logger.error(f"Error retrieving notifications: {e}", exc_info=True)
            return jsonify({"error": "Internal notifications fetch error"}), 500

    @app.route("/api/notifications/unread-count", methods=["GET"])
    @login_required
    def api_notifications_unread_count(current_user: User):
        from clinical_reporting.application.notification_service import NotificationService
        service = NotificationService(db_path=app.config["DB_PATH"])
        try:
            count = service.get_unread_count(user_id=current_user.id)
            return jsonify({"unread_count": count})
        except Exception as e:
            app.logger.error(f"Error checking unread notifications count: {e}", exc_info=True)
            return jsonify({"error": "Internal unread count compilation error"}), 500

    @app.route("/api/notifications/<int:notification_id>/read", methods=["PATCH"])
    @login_required
    def api_mark_notification_read(current_user: User, notification_id: int):
        from clinical_reporting.application.notification_service import NotificationService
        service = NotificationService(db_path=app.config["DB_PATH"])
        try:
            success = service.mark_notification_read(user_id=current_user.id, notification_id=notification_id)
            if not success:
                return jsonify({"error": "Notification not found or unauthorized access"}), 404
            return jsonify({"success": True})
        except Exception as e:
            app.logger.error(f"Error marking notification as read: {e}", exc_info=True)
            return jsonify({"error": "Internal notification modification error"}), 500

    @app.route("/api/notifications/read-all", methods=["PATCH"])
    @login_required
    def api_mark_all_notifications_read(current_user: User):
        from clinical_reporting.application.notification_service import NotificationService
        service = NotificationService(db_path=app.config["DB_PATH"])
        try:
            service.mark_all_notifications_read(user_id=current_user.id)
            return jsonify({"success": True})
        except Exception as e:
            app.logger.error(f"Error marking all notifications as read: {e}", exc_info=True)
            return jsonify({"error": "Internal notification read-all modification error"}), 500

    @app.route("/api/notifications/<int:notification_id>", methods=["DELETE"])
    @login_required
    def api_delete_notification(current_user: User, notification_id: int):
        from clinical_reporting.application.notification_service import NotificationService
        service = NotificationService(db_path=app.config["DB_PATH"])
        try:
            success = service.delete_notification(user_id=current_user.id, notification_id=notification_id)
            if not success:
                return jsonify({"error": "Notification not found or unauthorized access"}), 404
            return jsonify({"success": True})
        except Exception as e:
            app.logger.error(f"Error deleting notification: {e}", exc_info=True)
            return jsonify({"error": "Internal notification deletion error"}), 500

    @app.route("/api/notifications/preferences", methods=["GET"])
    @login_required
    def api_get_notification_preferences(current_user: User):
        from clinical_reporting.application.notification_service import NotificationService
        service = NotificationService(db_path=app.config["DB_PATH"])
        try:
            prefs = service.get_preferences(user_id=current_user.id)
            return jsonify(prefs)
        except Exception as e:
            app.logger.error(f"Error fetching notification preferences: {e}", exc_info=True)
            return jsonify({"error": "Internal notification preferences fetch error"}), 500

    @app.route("/api/notifications/preferences", methods=["PUT"])
    @login_required
    def api_update_notification_preferences(current_user: User):
        from clinical_reporting.application.notification_service import NotificationService
        service = NotificationService(db_path=app.config["DB_PATH"])
        try:
            data = request.get_json() or {}
            analysis = data.get("analysis", True)
            report = data.get("report", True)
            security = data.get("security", True)
            account = data.get("account", True)

            # Never allow a client-side preference to bypass critical security alerts
            security = True

            service.update_preferences(
                user_id=current_user.id,
                analysis=analysis,
                report=report,
                security=security,
                account=account
            )
            return jsonify({"success": True})
        except Exception as e:
            app.logger.error(f"Error updating notification preferences: {e}", exc_info=True)
            return jsonify({"error": "Internal notification preferences update error"}), 500

    @app.route("/api/predictions/<int:prediction_id>/feedback", methods=["POST"])
    @login_required
    def api_submit_feedback(current_user: User, prediction_id: int):
        import html
        import datetime
        from clinical_reporting.application.services import ReportService
        from security.domain.entities import SecurityAuditLog

        data = request.get_json() or {}
        rating = data.get("rating")
        feedback_type = data.get("feedback_type")
        comment = data.get("comment", "")

        # Server-side validations
        if rating is None or not isinstance(rating, int) or rating < 1 or rating > 5:
            return jsonify({"error": "Rating must be an integer between 1 and 5"}), 400

        valid_feedback_types = ["ACCURATE", "INACCURATE", "UNCERTAIN", "TECHNICAL_QUALITY_ISSUE", "CLINICAL_REVIEW_REQUIRED"]
        if feedback_type not in valid_feedback_types:
            return jsonify({"error": "Invalid feedback type"}), 400

        if comment:
            comment_str = str(comment).strip()
            if len(comment_str) > 1000:
                return jsonify({"error": "Comment exceeds 1000 characters limit"}), 400
            comment = html.escape(comment_str)
        else:
            comment = ""

        conn = sqlite3.connect(app.config["DB_PATH"])
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute("""
                SELECT cr.id as report_id, s.patient_id
                FROM clinical_reports cr
                JOIN predictions pr ON cr.prediction_id = pr.id
                JOIN mri_scans s ON pr.scan_id = s.id
                WHERE pr.id = ?;
            """, (prediction_id,)).fetchone()
            if not row:
                return jsonify({"error": "Prediction not found"}), 404

            report_id = row["report_id"]
            patient_id = row["patient_id"]

            # Enforce report access controls
            service = ReportService(db_path=app.config["DB_PATH"])
            access_status = service.check_report_access(report_id, current_user)
            if access_status == "FORBIDDEN":
                return jsonify({"error": "Access denied to prediction report"}), 403
            elif access_status == "NOT_FOUND":
                return jsonify({"error": "Report not found"}), 404
            elif access_status == "UNAUTHORIZED":
                return jsonify({"error": "Authentication required"}), 401

            now_str = datetime.datetime.utcnow().isoformat()

            # Idempotency / duplicate check (G8.3.9)
            existing = conn.execute("""
                SELECT id FROM prediction_feedback
                WHERE prediction_id = ? AND submitted_by_user_id = ?;
            """, (prediction_id, current_user.id)).fetchone()

            role_val = current_user.role.value if hasattr(current_user.role, 'value') else str(current_user.role)

            if existing:
                conn.execute("""
                    UPDATE prediction_feedback
                    SET rating = ?, feedback_type = ?, comment = ?, updated_at = ?
                    WHERE id = ?;
                """, (rating, feedback_type, comment, now_str, existing["id"]))
                conn.commit()
                feedback_id = existing["id"]
            else:
                cursor = conn.execute("""
                    INSERT INTO prediction_feedback (
                        prediction_id, report_id, patient_id, submitted_by_user_id, submitted_by_role,
                        rating, feedback_type, comment, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """, (prediction_id, report_id, patient_id, current_user.id, role_val, rating, feedback_type, comment, now_str, now_str))
                conn.commit()
                feedback_id = cursor.lastrowid

            # Log audit event
            user_repo.log_security_event(SecurityAuditLog(
                id=None,
                timestamp=now_str,
                event_type="FEEDBACK_SUBMITTED",
                user_id=current_user.id,
                email=current_user.email,
                ip_address=request.remote_addr or "127.0.0.1",
                status="SUCCESS",
                details=f"User {current_user.email} submitted feedback ID {feedback_id} for prediction {prediction_id}",
                user_agent=request.headers.get("User-Agent", "Unknown")
            ))

            return jsonify({"success": True, "feedback_id": feedback_id})
        finally:
            conn.close()

    @app.route("/api/predictions/<int:prediction_id>/feedback", methods=["GET"])
    @login_required
    def api_get_feedback(current_user: User, prediction_id: int):
        from clinical_reporting.application.services import ReportService

        conn = sqlite3.connect(app.config["DB_PATH"])
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute("""
                SELECT cr.id as report_id
                FROM clinical_reports cr
                WHERE cr.prediction_id = ?;
            """, (prediction_id,)).fetchone()
            if not row:
                return jsonify({"error": "Prediction not found"}), 404

            report_id = row["report_id"]

            # Enforce access controls
            service = ReportService(db_path=app.config["DB_PATH"])
            access_status = service.check_report_access(report_id, current_user)
            if access_status == "FORBIDDEN":
                return jsonify({"error": "Access denied"}), 403
            elif access_status == "NOT_FOUND":
                return jsonify({"error": "Report not found"}), 404
            elif access_status == "UNAUTHORIZED":
                return jsonify({"error": "Authentication required"}), 401

            role_val = current_user.role.value if hasattr(current_user.role, 'value') else str(current_user.role).lower()

            if role_val == "patient":
                # Patients can only see their own feedback
                rows = conn.execute("""
                    SELECT * FROM prediction_feedback
                    WHERE prediction_id = ? AND submitted_by_user_id = ?
                    ORDER BY created_at DESC;
                """, (prediction_id, current_user.id)).fetchall()
            else:
                # Doctors/Admins see all feedback for this prediction
                rows = conn.execute("""
                    SELECT * FROM prediction_feedback
                    WHERE prediction_id = ?
                    ORDER BY created_at DESC;
                """, (prediction_id,)).fetchall()

            res = []
            for r in rows:
                res.append(dict(r))
            return jsonify(res)
        finally:
            conn.close()

    @app.route("/api/predictions/<int:prediction_id>/flags", methods=["POST"])
    @login_required
    def api_submit_quality_flag(current_user: User, prediction_id: int):
        import html
        import json
        import datetime
        from clinical_reporting.application.services import ReportService
        from security.domain.entities import SecurityAuditLog

        data = request.get_json() or {}
        flag_type = data.get("flag_type")
        severity = data.get("severity")
        description = data.get("description", "")

        valid_flag_types = [
            "LOW_CONFIDENCE", "POSSIBLE_FALSE_POSITIVE", "POSSIBLE_FALSE_NEGATIVE",
            "POOR_IMAGE_QUALITY", "SEGMENTATION_CONCERN", "CLASSIFICATION_CONCERN",
            "REPORT_CONCERN", "CLINICAL_REVIEW_REQUIRED", "OTHER"
        ]
        valid_severities = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

        if flag_type not in valid_flag_types:
            return jsonify({"error": f"Invalid flag type: {flag_type}"}), 400

        if severity not in valid_severities:
            return jsonify({"error": f"Invalid severity: {severity}"}), 400

        if not description or not str(description).strip():
            return jsonify({"error": "Description is required"}), 400

        desc_str = str(description).strip()
        if len(desc_str) > 1000:
            return jsonify({"error": "Description exceeds 1000 characters limit"}), 400
        description = html.escape(desc_str)

        conn = sqlite3.connect(app.config["DB_PATH"])
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute("""
                SELECT cr.id as report_id, s.patient_id
                FROM clinical_reports cr
                JOIN predictions pr ON cr.prediction_id = pr.id
                JOIN mri_scans s ON pr.scan_id = s.id
                WHERE pr.id = ?;
            """, (prediction_id,)).fetchone()
            if not row:
                return jsonify({"error": "Prediction not found"}), 404

            report_id = row["report_id"]
            patient_id = row["patient_id"]

            # Enforce access controls
            service = ReportService(db_path=app.config["DB_PATH"])
            access_status = service.check_report_access(report_id, current_user)
            if access_status == "FORBIDDEN":
                return jsonify({"error": "Access denied"}), 403
            elif access_status == "NOT_FOUND":
                return jsonify({"error": "Report not found"}), 404
            elif access_status == "UNAUTHORIZED":
                return jsonify({"error": "Authentication required"}), 401

            now_str = datetime.datetime.utcnow().isoformat()

            # Idempotency / duplicate check (G8.3.9) - if same user, prediction, and flag_type is already OPEN
            existing = conn.execute("""
                SELECT id FROM prediction_quality_flags
                WHERE prediction_id = ? AND flagged_by_user_id = ? AND flag_type = ? AND status = 'OPEN';
            """, (prediction_id, current_user.id, flag_type)).fetchone()

            if existing:
                conn.execute("""
                    UPDATE prediction_quality_flags
                    SET severity = ?, description = ?, updated_at = ?
                    WHERE id = ?;
                """, (severity, description, now_str, existing["id"]))
                conn.commit()
                flag_id = existing["id"]
            else:
                cursor = conn.execute("""
                    INSERT INTO prediction_quality_flags (
                        prediction_id, report_id, patient_id, flagged_by_user_id, flag_type,
                        severity, description, status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'OPEN', ?, ?);
                """, (prediction_id, report_id, patient_id, current_user.id, flag_type, severity, description, now_str, now_str))
                conn.commit()
                flag_id = cursor.lastrowid

            # Log audit trail
            user_repo.log_security_event(SecurityAuditLog(
                id=None,
                timestamp=now_str,
                event_type="QUALITY_FLAG_CREATED",
                user_id=current_user.id,
                email=current_user.email,
                ip_address=request.remote_addr or "127.0.0.1",
                status="SUCCESS",
                details=f"User {current_user.email} raised quality flag ID {flag_id} (severity: {severity}, type: {flag_type}) for prediction {prediction_id}",
                user_agent=request.headers.get("User-Agent", "Unknown")
            ))

            # Notification integration (G8.3.12) - notify doctors/admins for HIGH/CRITICAL flags
            if severity in ["HIGH", "CRITICAL"]:
                try:
                    from clinical_reporting.application.notification_service import NotificationService
                    notif_svc = NotificationService(db_path=app.config["DB_PATH"])
                    # Retrieve all Doctor and Admin users
                    reviewers = [u for u in user_repo.list_users(limit=100) if u.role in [Role.DOCTOR, Role.ADMIN]]
                    for rev in reviewers:
                        notif_svc.create_notification(
                            user_id=rev.id,
                            type_="QUALITY_FLAG_CREATED",
                            title="Urgent Quality Flag Raised",
                            message=f"A {severity} severity quality flag ({flag_type}) was raised for patient {patient_id}.",
                            metadata_json=json.dumps({"prediction_id": prediction_id, "report_id": report_id, "severity": severity, "flag_id": flag_id})
                        )
                except Exception as notif_err:
                    app.logger.error(f"Failed to send flag creation notification: {notif_err}")

            return jsonify({"success": True, "flag_id": flag_id})
        finally:
            conn.close()

    @app.route("/api/predictions/<int:prediction_id>/flags", methods=["GET"])
    @login_required
    def api_get_prediction_flags(current_user: User, prediction_id: int):
        from clinical_reporting.application.services import ReportService

        conn = sqlite3.connect(app.config["DB_PATH"])
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute("""
                SELECT cr.id as report_id
                FROM clinical_reports cr
                WHERE cr.prediction_id = ?;
            """, (prediction_id,)).fetchone()
            if not row:
                return jsonify({"error": "Prediction not found"}), 404

            report_id = row["report_id"]

            # Enforce access controls
            service = ReportService(db_path=app.config["DB_PATH"])
            access_status = service.check_report_access(report_id, current_user)
            if access_status == "FORBIDDEN":
                return jsonify({"error": "Access denied"}), 403
            elif access_status == "NOT_FOUND":
                return jsonify({"error": "Report not found"}), 404
            elif access_status == "UNAUTHORIZED":
                return jsonify({"error": "Authentication required"}), 401

            role_val = current_user.role.value if hasattr(current_user.role, 'value') else str(current_user.role).lower()

            if role_val == "patient":
                # Patients can only see their own flags
                rows = conn.execute("""
                    SELECT * FROM prediction_quality_flags
                    WHERE prediction_id = ? AND flagged_by_user_id = ?
                    ORDER BY created_at DESC;
                """, (prediction_id, current_user.id)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT * FROM prediction_quality_flags
                    WHERE prediction_id = ?
                    ORDER BY created_at DESC;
                """, (prediction_id,)).fetchall()

            res = []
            for r in rows:
                res.append(dict(r))
            return jsonify(res)
        finally:
            conn.close()

    @app.route("/api/quality-flags", methods=["GET"])
    @login_required
    def api_list_quality_flags(current_user: User):
        status_filter = request.args.get("status")
        severity_filter = request.args.get("severity")
        flag_type_filter = request.args.get("flag_type")
        start_date = request.args.get("start_date")
        end_date = request.args.get("end_date")

        conn = sqlite3.connect(app.config["DB_PATH"])
        conn.row_factory = sqlite3.Row
        try:
            where_clauses = []
            params = []

            role_val = current_user.role.value if hasattr(current_user.role, 'value') else str(current_user.role).lower()

            if role_val == "patient":
                # Patients can only see their own flags
                where_clauses.append("flagged_by_user_id = ?")
                params.append(current_user.id)

            if status_filter:
                where_clauses.append("status = ?")
                params.append(status_filter)

            if severity_filter:
                where_clauses.append("severity = ?")
                params.append(severity_filter)

            if flag_type_filter:
                where_clauses.append("flag_type = ?")
                params.append(flag_type_filter)

            if start_date:
                where_clauses.append("created_at >= ?")
                params.append(start_date)

            if end_date:
                where_clauses.append("created_at <= ?")
                params.append(end_date)

            where_str = ""
            if where_clauses:
                where_str = " WHERE " + " AND ".join(where_clauses)

            query = f"""
                SELECT f.*, p.name as patient_name, pr.predicted_class, pr.confidence_score
                FROM prediction_quality_flags f
                JOIN patients p ON f.patient_id = p.patient_id
                JOIN predictions pr ON f.prediction_id = pr.id
                {where_str}
                ORDER BY f.created_at DESC;
            """
            rows = conn.execute(query, params).fetchall()

            try:
                from security.infrastructure.encryption_service import PIIEncryptionService
                encryption_service = PIIEncryptionService()
            except Exception:
                encryption_service = None

            res = []
            for r in rows:
                row_dict = dict(r)
                p_name = row_dict.get("patient_name")
                if p_name is not None:
                    if str(p_name).startswith("enc:v1:"):
                        if encryption_service is None:
                            raise ValueError("Patient name is encrypted but PII_ENCRYPTION_KEY is missing.")
                        row_dict["patient_name"] = encryption_service.decrypt(p_name)
                res.append(row_dict)
            return jsonify(res)
        finally:
            conn.close()

    @app.route("/api/quality-flags/<int:flag_id>", methods=["GET"])
    @login_required
    def api_get_quality_flag_details(current_user: User, flag_id: int):
        conn = sqlite3.connect(app.config["DB_PATH"])
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute("""
                SELECT f.*, p.name as patient_name, pr.predicted_class, pr.confidence_score
                FROM prediction_quality_flags f
                JOIN patients p ON f.patient_id = p.patient_id
                JOIN predictions pr ON f.prediction_id = pr.id
                WHERE f.id = ?;
            """, (flag_id,)).fetchone()
            if not row:
                return jsonify({"error": "Quality flag not found"}), 404

            role_val = current_user.role.value if hasattr(current_user.role, 'value') else str(current_user.role).lower()

            if role_val == "patient":
                # Verify patient ownership
                if row["flagged_by_user_id"] != current_user.id:
                    return jsonify({"error": "Access denied to quality flag"}), 403

            row_dict = dict(row)
            try:
                from security.infrastructure.encryption_service import PIIEncryptionService
                encryption_service = PIIEncryptionService()
            except Exception:
                encryption_service = None

            p_name = row_dict.get("patient_name")
            if p_name is not None:
                if str(p_name).startswith("enc:v1:"):
                    if encryption_service is None:
                        raise ValueError("Patient name is encrypted but PII_ENCRYPTION_KEY is missing.")
                    row_dict["patient_name"] = encryption_service.decrypt(p_name)

            return jsonify(row_dict)
        finally:
            conn.close()

    @app.route("/api/quality-flags/<int:flag_id>", methods=["PATCH"])
    @login_required
    def api_update_quality_flag(current_user: User, flag_id: int):
        import html
        import json
        import datetime
        from security.domain.entities import SecurityAuditLog

        role_val = current_user.role.value if hasattr(current_user.role, 'value') else str(current_user.role).lower()
        if role_val == "patient":
            # Patients MUST NOT review or resolve flags
            return jsonify({"error": "Access denied. Patient role cannot update quality flag status."}), 403

        data = request.get_json() or {}
        new_status = data.get("status")
        resolution_note = data.get("resolution_note", "")

        valid_statuses = ["OPEN", "IN_REVIEW", "RESOLVED", "DISMISSED"]
        if new_status not in valid_statuses:
            return jsonify({"error": f"Invalid status: {new_status}"}), 400

        # Enforce resolution note for RESOLVED
        if new_status == "RESOLVED" and (not resolution_note or not str(resolution_note).strip()):
            return jsonify({"error": "Resolution note is required when resolving a quality flag."}), 400

        if resolution_note:
            resolution_note = html.escape(str(resolution_note).strip())
            if len(resolution_note) > 1000:
                return jsonify({"error": "Resolution note exceeds 1000 characters limit"}), 400

        conn = sqlite3.connect(app.config["DB_PATH"])
        conn.row_factory = sqlite3.Row
        try:
            flag_record = conn.execute("SELECT * FROM prediction_quality_flags WHERE id = ?;", (flag_id,)).fetchone()
            if not flag_record:
                return jsonify({"error": "Quality flag not found"}), 404

            current_status = flag_record["status"]
            now_str = datetime.datetime.utcnow().isoformat()

            conn.execute("""
                UPDATE prediction_quality_flags
                SET status = ?, resolution_note = ?, reviewed_by_user_id = ?, reviewed_at = ?, updated_at = ?
                WHERE id = ?;
            """, (new_status, resolution_note, current_user.id, now_str, now_str, flag_id))
            conn.commit()

            audit_event_type = "QUALITY_FLAG_STATUS_CHANGED"
            if new_status == "RESOLVED":
                audit_event_type = "QUALITY_FLAG_RESOLVED"
            elif new_status == "DISMISSED":
                audit_event_type = "QUALITY_FLAG_DISMISSED"

            user_repo.log_security_event(SecurityAuditLog(
                id=None,
                timestamp=now_str,
                event_type=audit_event_type,
                user_id=current_user.id,
                email=current_user.email,
                ip_address=request.remote_addr or "127.0.0.1",
                status="SUCCESS",
                details=f"User {current_user.email} updated quality flag ID {flag_id} status from {current_status} to {new_status}.",
                user_agent=request.headers.get("User-Agent", "Unknown")
            ))

            # Notify the relevant user who raised the flag
            try:
                from clinical_reporting.application.notification_service import NotificationService
                notif_svc = NotificationService(db_path=app.config["DB_PATH"])
                notif_svc.create_notification(
                    user_id=flag_record["flagged_by_user_id"],
                    type_="QUALITY_FLAG_RESOLVED",
                    title=f"Quality Flag {new_status.capitalize()}",
                    message=f"Your quality flag for prediction {flag_record['prediction_id']} has been {new_status.lower()}.",
                    metadata_json=json.dumps({"flag_id": flag_id, "status": new_status, "resolution_note": resolution_note})
                )
            except Exception as notif_err:
                app.logger.error(f"Failed to send flag status update notification: {notif_err}")

            return jsonify({"success": True})
        finally:
            conn.close()

    @app.route("/api/report/<int:report_id>")
    @login_required
    def get_report_details(current_user: User, report_id: int):
        from clinical_reporting.application.services import ReportService
        service = ReportService(db_path=app.config["DB_PATH"])

        # Enforce report access check
        access_status = service.check_report_access(report_id, current_user)
        if access_status == "NOT_FOUND":
            service.log_report_access_event("REPORT_NOT_FOUND", current_user, report_id, "FAILED", "Requested nonexistent report details.")
            return jsonify({"error": "Report not found"}), 404
        elif access_status == "UNAUTHORIZED":
            return jsonify({"error": "Authentication required"}), 401
        elif access_status == "FORBIDDEN":
            service.log_report_access_event("REPORT_ACCESS_DENIED", current_user, report_id, "FAILED", "Access denied to patient report details.")
            return jsonify({"error": "Access denied to patient report"}), 403

        try:
            # 1. Fetch current active report details
            conn = sqlite3.connect(app.config["DB_PATH"])
            conn.row_factory = sqlite3.Row
            try:
                query = """
                SELECT
                    cr.id as report_id, cr.prediction_id, s.id as scan_id, p.patient_id, p.name as patient_name,
                    pr.predicted_class, pr.confidence_score, pr.tumor_area_mm2, pr.tumor_percentage_brain,
                    pr.rule_based_severity, pr.severity_rule_description, cr.created_at
                FROM clinical_reports cr
                JOIN predictions pr ON cr.prediction_id = pr.id
                JOIN mri_scans s ON pr.scan_id = s.id
                JOIN patients p ON s.patient_id = p.patient_id
                WHERE cr.id = ?;
                """
                row = conn.execute(query, (report_id,)).fetchone()
                if not row:
                    return jsonify({"error": "Report not found"}), 404
                report_dict = dict(row)
                try:
                    from security.infrastructure.encryption_service import PIIEncryptionService
                    encryption_service = PIIEncryptionService()
                except Exception:
                    encryption_service = None

                p_name = report_dict.get("patient_name")
                if p_name is not None:
                    if str(p_name).startswith("enc:v1:"):
                        if encryption_service is None:
                            raise ValueError("Patient name is encrypted but PII_ENCRYPTION_KEY is missing.")
                        report_dict["patient_name"] = encryption_service.decrypt(p_name)
            finally:
                conn.close()

            # 2. Enrich with ReportService metadata & versions
            report = service.get_report(report_id)
            versions = service.get_report_versions(report_id)

            report_dict["report_number"] = report.report_number
            report_dict["status"] = report.status.value
            report_dict["current_version"] = report.current_version
            report_dict["updated_at"] = report.updated_at

            versions_list = [v.to_dict() for v in versions]
            role_str = current_user.role.value if hasattr(current_user.role, "value") else str(current_user.role).lower()
            if role_str == "patient":
                report_dict.pop("pdf_path", None)
                report_dict.pop("json_path", None)
                for v_dict in versions_list:
                    v_dict.pop("pdf_path", None)
                    v_dict.pop("json_path", None)

            report_dict["versions"] = versions_list

            service.log_report_access_event("REPORT_VIEWED", current_user, report_id, "SUCCESS", "Viewed report details.")
            return jsonify(report_dict)
        except Exception as e:
            app.logger.error(f"Error fetching report metadata: {e}", exc_info=True)
            return jsonify({"error": "Internal report fetch error"}), 500

    @app.route("/api/doctor/scans/<int:scan_id>/clinical-notes", methods=["POST", "GET"])
    @login_required
    def clinician_notes_scan_flask(current_user: User, scan_id: int):
        from clinical_reporting.application.clinician_note_service import ClinicianNoteService, ClinicianNoteServiceException
        service = ClinicianNoteService(db_path=app.config["DB_PATH"])

        role_val = current_user.role.value if hasattr(current_user.role, 'value') else str(current_user.role).lower()
        if role_val != "doctor" and role_val != "admin":
            return jsonify({"error": "Access denied. Doctor role required."}), 403

        if request.method == "POST":
            data = request.get_json() or {}
            content = data.get("content")
            try:
                note = service.create_note(actor=current_user, scan_id=scan_id, content=content)
                return jsonify(note), 200
            except ClinicianNoteServiceException as e:
                err_msg = str(e)
                if "Access denied" in err_msg:
                    return jsonify({"error": err_msg}), 403
                elif "Authentication required" in err_msg:
                    return jsonify({"error": err_msg}), 401
                elif "not found" in err_msg.lower():
                    return jsonify({"error": err_msg}), 404
                else:
                    return jsonify({"error": err_msg}), 400
            except Exception as e:
                app.logger.error(f"Error creating clinical note: {e}", exc_info=True)
                return jsonify({"error": "Internal notes operation failure"}), 500
        else: # GET
            try:
                notes = service.get_notes_for_scan(actor=current_user, scan_id=scan_id)
                return jsonify(notes), 200
            except ClinicianNoteServiceException as e:
                err_msg = str(e)
                if "Access denied" in err_msg:
                    return jsonify({"error": err_msg}), 403
                elif "Authentication required" in err_msg:
                    return jsonify({"error": err_msg}), 401
                elif "not found" in err_msg.lower():
                    return jsonify({"error": err_msg}), 404
                else:
                    return jsonify({"error": err_msg}), 400
            except Exception as e:
                app.logger.error(f"Error querying clinical notes: {e}", exc_info=True)
                return jsonify({"error": "Internal notes query failure"}), 500

    @app.route("/api/doctor/clinical-notes/<int:note_id>", methods=["PUT", "DELETE"])
    @login_required
    def clinician_note_operations_flask(current_user: User, note_id: int):
        from clinical_reporting.application.clinician_note_service import ClinicianNoteService, ClinicianNoteServiceException
        service = ClinicianNoteService(db_path=app.config["DB_PATH"])

        role_val = current_user.role.value if hasattr(current_user.role, 'value') else str(current_user.role).lower()
        if role_val != "doctor" and role_val != "admin":
            return jsonify({"error": "Access denied. Doctor role required."}), 403

        if request.method == "PUT":
            data = request.get_json() or {}
            content = data.get("content")
            try:
                note = service.update_note(actor=current_user, note_id=note_id, content=content)
                return jsonify(note), 200
            except ClinicianNoteServiceException as e:
                err_msg = str(e)
                if "Access denied" in err_msg:
                    return jsonify({"error": err_msg}), 403
                elif "Authentication required" in err_msg:
                    return jsonify({"error": err_msg}), 401
                elif "not found" in err_msg.lower():
                    return jsonify({"error": err_msg}), 404
                else:
                    return jsonify({"error": err_msg}), 400
            except Exception as e:
                app.logger.error(f"Error updating clinical note: {e}", exc_info=True)
                return jsonify({"error": "Internal note modification failure"}), 500
        else: # DELETE
            try:
                result = service.archive_note(actor=current_user, note_id=note_id)
                return jsonify(result), 200
            except ClinicianNoteServiceException as e:
                err_msg = str(e)
                if "Access denied" in err_msg:
                    return jsonify({"error": err_msg}), 403
                elif "Authentication required" in err_msg:
                    return jsonify({"error": err_msg}), 401
                elif "not found" in err_msg.lower():
                    return jsonify({"error": err_msg}), 404
                else:
                    return jsonify({"error": err_msg}), 400
            except Exception as e:
                app.logger.error(f"Error deleting clinical note: {e}", exc_info=True)
                return jsonify({"error": "Internal note deletion failure"}), 500

    @app.route("/api/doctor/scans/<int:scan_id>/point-annotations", methods=["POST", "GET"])
    @login_required
    def clinician_point_annotations_scan_flask(current_user: User, scan_id: int):
        from clinical_reporting.application.mri_annotation_service import MriAnnotationService, MriAnnotationServiceException
        service = MriAnnotationService(db_path=app.config["DB_PATH"])

        role_val = current_user.role.value if hasattr(current_user.role, 'value') else str(current_user.role).lower()
        if role_val != "doctor" and role_val != "admin":
            return jsonify({"error": "Access denied. Doctor role required."}), 403

        if request.method == "POST":
            data = request.get_json() or {}
            x = data.get("x")
            y = data.get("y")
            label = data.get("label")
            comment = data.get("comment")
            try:
                note = service.create_annotation(
                    actor=current_user,
                    scan_id=scan_id,
                    x=x,
                    y=y,
                    label=label,
                    comment=comment
                )
                return jsonify(note), 200
            except MriAnnotationServiceException as e:
                err_msg = str(e)
                if "Access denied" in err_msg:
                    return jsonify({"error": err_msg}), 403
                elif "Authentication required" in err_msg:
                    return jsonify({"error": err_msg}), 401
                elif "not found" in err_msg.lower():
                    return jsonify({"error": err_msg}), 404
                else:
                    return jsonify({"error": err_msg}), 400
            except Exception as e:
                app.logger.error(f"Error creating point annotation: {e}", exc_info=True)
                return jsonify({"error": "Internal annotation operation failure"}), 500
        else: # GET
            try:
                notes = service.get_annotations_for_scan(actor=current_user, scan_id=scan_id)
                return jsonify(notes), 200
            except MriAnnotationServiceException as e:
                err_msg = str(e)
                if "Access denied" in err_msg:
                    return jsonify({"error": err_msg}), 403
                elif "Authentication required" in err_msg:
                    return jsonify({"error": err_msg}), 401
                elif "not found" in err_msg.lower():
                    return jsonify({"error": err_msg}), 404
                else:
                    return jsonify({"error": err_msg}), 400
            except Exception as e:
                app.logger.error(f"Error querying point annotations: {e}", exc_info=True)
                return jsonify({"error": "Internal annotations query failure"}), 500

    @app.route("/api/doctor/point-annotations/<int:annotation_id>", methods=["PUT", "DELETE"])
    @login_required
    def clinician_point_annotation_operations_flask(current_user: User, annotation_id: int):
        from clinical_reporting.application.mri_annotation_service import MriAnnotationService, MriAnnotationServiceException
        service = MriAnnotationService(db_path=app.config["DB_PATH"])

        role_val = current_user.role.value if hasattr(current_user.role, 'value') else str(current_user.role).lower()
        if role_val != "doctor" and role_val != "admin":
            return jsonify({"error": "Access denied. Doctor role required."}), 403

        if request.method == "PUT":
            data = request.get_json() or {}
            x = data.get("x")
            y = data.get("y")
            label = data.get("label")
            comment = data.get("comment")
            try:
                note = service.update_annotation(
                    actor=current_user,
                    annotation_id=annotation_id,
                    x=x,
                    y=y,
                    label=label,
                    comment=comment
                )
                return jsonify(note), 200
            except MriAnnotationServiceException as e:
                err_msg = str(e)
                if "Access denied" in err_msg:
                    return jsonify({"error": err_msg}), 403
                elif "Authentication required" in err_msg:
                    return jsonify({"error": err_msg}), 401
                elif "not found" in err_msg.lower():
                    return jsonify({"error": err_msg}), 404
                else:
                    return jsonify({"error": err_msg}), 400
            except Exception as e:
                app.logger.error(f"Error updating point annotation: {e}", exc_info=True)
                return jsonify({"error": "Internal annotation modification failure"}), 500
        else: # DELETE
            try:
                result = service.archive_annotation(actor=current_user, annotation_id=annotation_id)
                return jsonify(result), 200
            except MriAnnotationServiceException as e:
                err_msg = str(e)
                if "Access denied" in err_msg:
                    return jsonify({"error": err_msg}), 403
                elif "Authentication required" in err_msg:
                    return jsonify({"error": err_msg}), 401
                elif "not found" in err_msg.lower():
                    return jsonify({"error": err_msg}), 404
                else:
                    return jsonify({"error": err_msg}), 400
            except Exception as e:
                app.logger.error(f"Error deleting point annotation: {e}", exc_info=True)
                return jsonify({"error": "Internal annotation deletion failure"}), 500

    @app.route("/api/doctor/scans/<int:scan_id>/rectangle-annotations", methods=["POST", "GET"])
    @login_required
    def clinician_rectangle_annotations_scan_flask(current_user: User, scan_id: int):
        from clinical_reporting.application.mri_annotation_service import MriAnnotationService, MriAnnotationServiceException
        service = MriAnnotationService(db_path=app.config["DB_PATH"])

        role_val = current_user.role.value if hasattr(current_user.role, 'value') else str(current_user.role).lower()
        if role_val != "doctor" and role_val != "admin":
            return jsonify({"error": "Access denied. Doctor role required."}), 403

        if request.method == "POST":
            data = request.get_json() or {}
            x = data.get("x")
            y = data.get("y")
            width = data.get("width")
            height = data.get("height")
            label = data.get("label")
            comment = data.get("comment")
            try:
                note = service.create_rectangle_annotation(
                    actor=current_user,
                    scan_id=scan_id,
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    label=label,
                    comment=comment
                )
                return jsonify(note), 200
            except MriAnnotationServiceException as e:
                err_msg = str(e)
                if "Access denied" in err_msg:
                    return jsonify({"error": err_msg}), 403
                elif "Authentication required" in err_msg:
                    return jsonify({"error": err_msg}), 401
                elif "not found" in err_msg.lower():
                    return jsonify({"error": err_msg}), 404
                else:
                    return jsonify({"error": err_msg}), 400
            except Exception as e:
                app.logger.error(f"Error creating rectangle annotation: {e}", exc_info=True)
                return jsonify({"error": "Internal rectangle annotation operation failure"}), 500
        else: # GET
            try:
                notes = service.get_rectangle_annotations_for_scan(actor=current_user, scan_id=scan_id)
                return jsonify(notes), 200
            except MriAnnotationServiceException as e:
                err_msg = str(e)
                if "Access denied" in err_msg:
                    return jsonify({"error": err_msg}), 403
                elif "Authentication required" in err_msg:
                    return jsonify({"error": err_msg}), 401
                elif "not found" in err_msg.lower():
                    return jsonify({"error": err_msg}), 404
                else:
                    return jsonify({"error": err_msg}), 400
            except Exception as e:
                app.logger.error(f"Error querying rectangle annotations: {e}", exc_info=True)
                return jsonify({"error": "Internal rectangle annotations query failure"}), 500

    @app.route("/api/doctor/rectangle-annotations/<int:annotation_id>", methods=["PUT", "DELETE"])
    @login_required
    def clinician_rectangle_annotation_operations_flask(current_user: User, annotation_id: int):
        from clinical_reporting.application.mri_annotation_service import MriAnnotationService, MriAnnotationServiceException
        service = MriAnnotationService(db_path=app.config["DB_PATH"])

        role_val = current_user.role.value if hasattr(current_user.role, 'value') else str(current_user.role).lower()
        if role_val != "doctor" and role_val != "admin":
            return jsonify({"error": "Access denied. Doctor role required."}), 403

        if request.method == "PUT":
            data = request.get_json() or {}
            x = data.get("x")
            y = data.get("y")
            width = data.get("width")
            height = data.get("height")
            label = data.get("label")
            comment = data.get("comment")
            try:
                note = service.update_rectangle_annotation(
                    actor=current_user,
                    annotation_id=annotation_id,
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    label=label,
                    comment=comment
                )
                return jsonify(note), 200
            except MriAnnotationServiceException as e:
                err_msg = str(e)
                if "Access denied" in err_msg:
                    return jsonify({"error": err_msg}), 403
                elif "Authentication required" in err_msg:
                    return jsonify({"error": err_msg}), 401
                elif "not found" in err_msg.lower():
                    return jsonify({"error": err_msg}), 404
                else:
                    return jsonify({"error": err_msg}), 400
            except Exception as e:
                app.logger.error(f"Error updating rectangle annotation: {e}", exc_info=True)
                return jsonify({"error": "Internal rectangle annotation modification failure"}), 500
        else: # DELETE
            try:
                result = service.archive_rectangle_annotation(actor=current_user, annotation_id=annotation_id)
                return jsonify(result), 200
            except MriAnnotationServiceException as e:
                err_msg = str(e)
                if "Access denied" in err_msg:
                    return jsonify({"error": err_msg}), 403
                elif "Authentication required" in err_msg:
                    return jsonify({"error": err_msg}), 401
                elif "not found" in err_msg.lower():
                    return jsonify({"error": err_msg}), 404
                else:
                    return jsonify({"error": err_msg}), 400
            except Exception as e:
                app.logger.error(f"Error deleting rectangle annotation: {e}", exc_info=True)
                return jsonify({"error": "Internal rectangle annotation deletion failure"}), 500

    @app.route("/api/report/<int:report_id>/compare/<int:other_report_id>")
    @login_required
    def compare_reports_flask(current_user: User, report_id: int, other_report_id: int):
        from clinical_reporting.application.services import ReportService, ReportNotFoundException, VersionNotFoundException, ReportServiceException
        from clinical_reporting.domain.entities import PatientMismatchException

        prev_version_str = request.args.get("previous_version")
        curr_version_str = request.args.get("current_version")

        prev_version = None
        curr_version = None
        if prev_version_str:
            try:
                prev_version = int(prev_version_str)
            except ValueError:
                pass
        if curr_version_str:
            try:
                curr_version = int(curr_version_str)
            except ValueError:
                pass

        service = ReportService(db_path=app.config["DB_PATH"])
        try:
            result = service.compare_reports(
                previous_report_id=report_id,
                current_report_id=other_report_id,
                previous_version=prev_version,
                current_version=curr_version,
                actor=current_user
            )
            return jsonify(result)
        except (ReportNotFoundException, VersionNotFoundException) as nfe:
            return jsonify({"error": str(nfe)}), 404
        except PatientMismatchException as pme:
            return jsonify({"error": str(pme)}), 400
        except ReportServiceException as rse:
            err_msg = str(rse)
            if "Access denied" in err_msg or "denied" in err_msg.lower():
                return jsonify({"error": err_msg}), 403
            elif "Authentication required" in err_msg or "unauthenticated" in err_msg.lower():
                return jsonify({"error": err_msg}), 401
            else:
                return jsonify({"error": err_msg}), 422
        except Exception as e:
            app.logger.error(f"Error in Flask comparison API: {e}")
            return jsonify({"error": "Internal comparison engine error"}), 500


    @app.route("/api/report/<int:report_id>/compare-followup/<int:other_report_id>")
    @login_required
    def compare_followup_reports_flask(current_user: User, report_id: int, other_report_id: int):
        from clinical_reporting.application.services import ReportService, ReportNotFoundException, VersionNotFoundException, ReportServiceException
        from clinical_reporting.domain.entities import PatientMismatchException

        prev_version_str = request.args.get("previous_version")
        curr_version_str = request.args.get("current_version")

        prev_version = None
        curr_version = None
        if prev_version_str:
            try:
                prev_version = int(prev_version_str)
            except ValueError:
                pass
        if curr_version_str:
            try:
                curr_version = int(curr_version_str)
            except ValueError:
                pass

        service = ReportService(db_path=app.config["DB_PATH"])
        try:
            result = service.compare_followup_reports(
                previous_report_id=report_id,
                current_report_id=other_report_id,
                previous_version=prev_version,
                current_version=curr_version,
                actor=current_user
            )
            return jsonify(result)
        except (ReportNotFoundException, VersionNotFoundException) as nfe:
            return jsonify({"error": str(nfe)}), 404
        except PatientMismatchException as pme:
            return jsonify({"error": str(pme)}), 400
        except ReportServiceException as rse:
            err_msg = str(rse)
            if "Access denied" in err_msg or "denied" in err_msg.lower():
                return jsonify({"error": err_msg}), 403
            elif "Authentication required" in err_msg or "unauthenticated" in err_msg.lower():
                return jsonify({"error": err_msg}), 401
            else:
                return jsonify({"error": err_msg}), 422
        except Exception as e:
            app.logger.error(f"Error in Flask follow-up comparison API: {e}")
            return jsonify({"error": "Internal follow-up comparison engine error"}), 500

    @app.route("/api/doctor/patients/<patient_id>", methods=["GET"])
    @roles_accepted(Role.DOCTOR)
    def get_doctor_patient_profile_flask(current_user: User, patient_id: str):
        from clinical_reporting.application.services import ReportService, ReportServiceException
        service = ReportService(db_path=app.config["DB_PATH"])
        try:
            profile = service.get_patient_profile(patient_id=patient_id, actor=current_user)
            return jsonify(profile), 200
        except ReportServiceException as e:
            err_msg = str(e)
            if "Access denied" in err_msg:
                return jsonify({"error": err_msg}), 403
            elif "Authentication required" in err_msg:
                return jsonify({"error": err_msg}), 401
            elif "not found" in err_msg.lower():
                return jsonify({"error": err_msg}), 404
            else:
                return jsonify({"error": err_msg}), 400
        except Exception as e:
            app.logger.error(f"Error fetching patient profile: {e}", exc_info=True)
            return jsonify({"error": "Internal patient profile fetch error"}), 500

    @app.route("/api/doctor/patients/<patient_id>/followups", methods=["POST"])
    @roles_accepted(Role.DOCTOR)
    def create_doctor_patient_followup_flask(current_user: User, patient_id: str):
        from clinical_reporting.application.followup_service import FollowupScheduleService, FollowupScheduleServiceException
        service = FollowupScheduleService(db_path=app.config["DB_PATH"])
        try:
            data = request.get_json() or {}
        except Exception:
            return jsonify({"error": "Malformed or empty JSON body."}), 400
        scheduled_date = data.get("scheduled_date")
        reason = data.get("reason")
        notes = data.get("notes")
        try:
            followup = service.create_followup(
                actor=current_user,
                patient_id=patient_id,
                scheduled_date=scheduled_date,
                reason=reason,
                notes=notes
            )
            return jsonify(followup), 200
        except FollowupScheduleServiceException as e:
            err_msg = str(e)
            if "Access denied" in err_msg:
                return jsonify({"error": err_msg}), 403
            elif "Authentication required" in err_msg:
                return jsonify({"error": err_msg}), 401
            elif "not found" in err_msg.lower():
                return jsonify({"error": err_msg}), 404
            else:
                return jsonify({"error": err_msg}), 400
        except Exception as e:
            app.logger.error(f"Error creating patient follow-up: {e}", exc_info=True)
            return jsonify({"error": "Internal follow-up schedule creation failure"}), 500

    @app.route("/api/doctor/patients/<patient_id>/followups", methods=["GET"])
    @roles_accepted(Role.DOCTOR)
    def get_doctor_patient_followups_flask(current_user: User, patient_id: str):
        from clinical_reporting.application.followup_service import FollowupScheduleService, FollowupScheduleServiceException
        service = FollowupScheduleService(db_path=app.config["DB_PATH"])
        try:
            followups = service.get_followups_for_patient(actor=current_user, patient_id=patient_id)
            return jsonify(followups), 200
        except FollowupScheduleServiceException as e:
            err_msg = str(e)
            if "Access denied" in err_msg:
                return jsonify({"error": err_msg}), 403
            elif "Authentication required" in err_msg:
                return jsonify({"error": err_msg}), 401
            elif "not found" in err_msg.lower():
                return jsonify({"error": err_msg}), 404
            else:
                return jsonify({"error": err_msg}), 400
        except Exception as e:
            app.logger.error(f"Error fetching patient follow-ups: {e}", exc_info=True)
            return jsonify({"error": "Internal follow-up schedules query failure"}), 500

    @app.route("/api/patient/followups", methods=["GET"])
    @roles_accepted(Role.PATIENT)
    def get_patient_followups_flask(current_user: User):
        from clinical_reporting.application.followup_service import FollowupScheduleService, FollowupScheduleServiceException
        service = FollowupScheduleService(db_path=app.config["DB_PATH"])
        try:
            patient_id = current_user.uuid
            followups = service.get_followups_for_patient(actor=current_user, patient_id=patient_id)
            return jsonify(followups), 200
        except FollowupScheduleServiceException as e:
            err_msg = str(e)
            if "Access denied" in err_msg:
                return jsonify({"error": "Access denied."}), 403
            elif "Authentication required" in err_msg:
                return jsonify({"error": "Authentication required."}), 401
            elif "not found" in err_msg.lower():
                return jsonify({"error": "Patient record not found."}), 404
            else:
                return jsonify({"error": err_msg}), 400
        except Exception as e:
            app.logger.error(f"Error fetching patient follow-ups: {e}", exc_info=True)
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/api/patient/profile", methods=["GET"])
    @roles_accepted(Role.PATIENT)
    def get_patient_profile_flask(current_user: User):
        import sqlite3
        from security.infrastructure.encryption_service import PIIEncryptionService
        db_path = app.config["DB_PATH"]
        conn = None
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT name, age, gender FROM patients WHERE patient_id = ?;",
                (current_user.uuid,)
            ).fetchone()
            if not row:
                return jsonify({"error": "Patient record not found."}), 404

            try:
                encryption_service = PIIEncryptionService()
            except Exception:
                encryption_service = None

            pat_name_raw = row["name"]
            pat_age_raw = row["age"]
            pat_gender_raw = row["gender"]

            name = encryption_service.decrypt(pat_name_raw) if encryption_service and pat_name_raw and str(pat_name_raw).startswith("enc:v1:") else pat_name_raw
            age = encryption_service.decrypt(pat_age_raw) if encryption_service and pat_age_raw and str(pat_age_raw).startswith("enc:v1:") else pat_age_raw
            gender = encryption_service.decrypt(pat_gender_raw) if encryption_service and pat_gender_raw and str(pat_gender_raw).startswith("enc:v1:") else pat_gender_raw

            try:
                if age is not None:
                    age = int(age)
            except ValueError:
                pass

            return jsonify({
                "patient_id": current_user.uuid,
                "name": name,
                "email": current_user.email,
                "age": age,
                "gender": gender
            }), 200
        except Exception as e:
            app.logger.error(f"Error retrieving patient profile: {e}", exc_info=True)
            return jsonify({"error": "Internal server error"}), 500
        finally:
            if conn:
                conn.close()

    @app.route("/api/doctor/followups/<int:followup_id>", methods=["PUT"])
    @roles_accepted(Role.DOCTOR)
    def update_doctor_followup_flask(current_user: User, followup_id: int):
        from clinical_reporting.application.followup_service import FollowupScheduleService, FollowupScheduleServiceException
        service = FollowupScheduleService(db_path=app.config["DB_PATH"])
        try:
            data = request.get_json() or {}
        except Exception:
            return jsonify({"error": "Malformed or empty JSON body."}), 400
        scheduled_date = data.get("scheduled_date")
        status = data.get("status")
        reason = data.get("reason")
        notes = data.get("notes")
        try:
            followup = service.update_followup(
                actor=current_user,
                followup_id=followup_id,
                scheduled_date=scheduled_date,
                status=status,
                reason=reason,
                notes=notes
            )
            return jsonify(followup), 200
        except FollowupScheduleServiceException as e:
            err_msg = str(e)
            if "Access denied" in err_msg:
                return jsonify({"error": err_msg}), 403
            elif "Authentication required" in err_msg:
                return jsonify({"error": err_msg}), 401
            elif "not found" in err_msg.lower():
                return jsonify({"error": err_msg}), 404
            else:
                return jsonify({"error": err_msg}), 400
        except Exception as e:
            app.logger.error(f"Error updating follow-up: {e}", exc_info=True)
            return jsonify({"error": "Internal follow-up schedule update failure"}), 500

    @app.route("/api/patients/<patient_id>/longitudinal-timeline")
    @login_required
    def get_patient_longitudinal_timeline_flask(current_user: User, patient_id: str):
        """Flask Endpoint: Retrieves a patient's longitudinal timeline, enforcing authorization boundaries."""
        from clinical_reporting.application.services import ReportService, ReportServiceException

        service = ReportService(db_path=app.config["DB_PATH"])
        try:
            timeline = service.get_patient_longitudinal_timeline(
                patient_id=patient_id,
                actor=current_user
            )
            return jsonify(timeline.to_dict())
        except ReportServiceException as rse:
            err_msg = str(rse)
            if "Access denied" in err_msg or "denied" in err_msg.lower():
                return jsonify({"error": "Access denied to patient timeline."}), 403
            elif "Authentication required" in err_msg or "unauthenticated" in err_msg.lower():
                return jsonify({"error": err_msg}), 401
            elif "not found" in err_msg.lower():
                return jsonify({"error": err_msg}), 404
            else:
                return jsonify({"error": err_msg}), 422
        except Exception as e:
            app.logger.error(f"Error in Flask longitudinal timeline API: {e}")
            return jsonify({"error": "Internal timeline engine error"}), 500

    @app.route("/api/patients/<patient_id>/analytics")
    @login_required
    def get_patient_analytics_flask(current_user: User, patient_id: str):
        """Flask Endpoint: Retrieves patient clinical analytics, enforcing authorization boundaries."""
        from clinical_reporting.application.services import ReportService, ReportServiceException

        service = ReportService(db_path=app.config["DB_PATH"])
        try:
            analytics = service.get_patient_analytics(
                patient_id=patient_id,
                actor=current_user
            )
            return jsonify(analytics.to_dict())
        except ReportServiceException as rse:
            err_msg = str(rse)
            if "Access denied" in err_msg or "denied" in err_msg.lower():
                return jsonify({"error": "Access denied to patient analytics."}), 403
            elif "Authentication required" in err_msg or "unauthenticated" in err_msg.lower():
                return jsonify({"error": err_msg}), 401
            elif "not found" in err_msg.lower():
                return jsonify({"error": err_msg}), 404
            else:
                return jsonify({"error": err_msg}), 422
        except Exception as e:
            app.logger.error(f"Error in Flask patient analytics API: {e}")
            return jsonify({"error": "Internal analytics engine error"}), 500

    @app.route("/api/analytics/overview")
    @login_required
    def get_population_analytics_flask(current_user: User):
        """Flask Endpoint: Retrieves population overview analytics, enforcing authorization boundaries."""
        if current_user.role not in [Role.ADMIN, Role.DOCTOR]:
            return jsonify({"error": f"Access denied. Privilege level '{current_user.role.value}' is not authorized."}), 403

        from clinical_reporting.application.services import ReportService, ReportServiceException

        service = ReportService(db_path=app.config["DB_PATH"])
        try:
            pop_analytics = service.get_population_analytics(actor=current_user)
            return jsonify(pop_analytics.to_dict())
        except ReportServiceException as rse:
            err_msg = str(rse)
            if "Access denied" in err_msg or "denied" in err_msg.lower():
                return jsonify({"error": err_msg}), 403
            elif "Authentication required" in err_msg or "unauthenticated" in err_msg.lower():
                return jsonify({"error": err_msg}), 401
            else:
                return jsonify({"error": err_msg}), 422
        except Exception as e:
            app.logger.error(f"Error in Flask population analytics API: {e}")
            return jsonify({"error": "Internal population analytics engine error"}), 500

    @app.route("/api/patients/<patient_id>/analytics/timeseries")
    @login_required
    def get_patient_timeseries_flask(current_user: User, patient_id: str):
        """Flask Endpoint: Retrieves patient timeseries points, enforcing authorization boundaries."""
        from clinical_reporting.application.services import ReportService, ReportServiceException

        service = ReportService(db_path=app.config["DB_PATH"])
        try:
            ts = service.get_patient_timeseries(
                patient_id=patient_id,
                actor=current_user
            )
            return jsonify(ts.to_dict())
        except ReportServiceException as rse:
            err_msg = str(rse)
            if "Access denied" in err_msg or "denied" in err_msg.lower():
                return jsonify({"error": "Access denied to patient timeseries."}), 403
            elif "Authentication required" in err_msg or "unauthenticated" in err_msg.lower():
                return jsonify({"error": err_msg}), 401
            elif "not found" in err_msg.lower():
                return jsonify({"error": err_msg}), 404
            else:
                return jsonify({"error": err_msg}), 422
        except Exception as e:
            app.logger.error(f"Error in Flask patient timeseries API: {e}")
            return jsonify({"error": "Internal timeseries engine error"}), 500


    @app.route("/api/report/<int:report_id>/status", methods=["PATCH"])
    @login_required
    def update_report_status(current_user: User, report_id: int):
        from clinical_reporting.application.services import ReportService
        from clinical_reporting.domain.entities import ReportStatus

        # Enforce RBAC
        if current_user.role not in [Role.ADMIN, Role.DOCTOR]:
            return jsonify({"error": "Access denied. Action restricted to Doctors and Admins."}), 403

        data = request.get_json() or {}
        status_str = data.get("status")
        if not status_str:
            return jsonify({"error": "Status is required."}), 400

        service = ReportService(db_path=app.config["DB_PATH"])
        try:
            target_status = ReportStatus(status_str.upper())
            updated_report = service.transition_status(
                report_id=report_id,
                target_status=target_status,
                actor=current_user.email
            )
            return jsonify(updated_report.to_dict())
        except ValueError:
            return jsonify({"error": f"Invalid status: {status_str}"}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/api/report/<int:report_id>/versions", methods=["POST"])
    @login_required
    def create_report_version(current_user: User, report_id: int):
        from clinical_reporting.application.services import ReportService
        from clinical_reporting.domain.entities import ReportStatus

        # Enforce RBAC
        if current_user.role not in [Role.ADMIN, Role.DOCTOR]:
            return jsonify({"error": "Access denied. Action restricted to Doctors and Admins."}), 403

        service = ReportService(db_path=app.config["DB_PATH"])
        access_status = service.check_report_access(report_id, current_user)
        if access_status == "NOT_FOUND":
            return jsonify({"error": "Report not found"}), 404
        elif access_status == "UNAUTHORIZED":
            return jsonify({"error": "Authentication required"}), 401
        elif access_status == "FORBIDDEN":
            return jsonify({"error": "Access denied to patient report"}), 403

        data = request.get_json() or {}
        reason = data.get("reason")
        if not reason:
            return jsonify({"error": "Reason is required."}), 400

        pdf_path = data.get("pdf_path")
        json_path = data.get("json_path")
        prediction_id = data.get("prediction_id")
        status_str = data.get("status", "DRAFT")
        try:
            status_enum = ReportStatus(status_str.upper())
            new_version = service.create_new_version(
                report_id=report_id,
                created_by=current_user.email,
                reason=reason,
                pdf_path=pdf_path,
                json_path=json_path,
                prediction_id=prediction_id,
                status=status_enum
            )
            return jsonify(new_version.to_dict())
        except ValueError:
            return jsonify({"error": f"Invalid status: {status_str}"}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 400


    @app.route("/api/report/<int:report_id>/pdf")
    @login_required
    def get_pdf(current_user: User, report_id: int):
        import logging
        logger = logging.getLogger("dashboard.web_server.get_pdf")

        version_str = request.args.get("version")
        try:
            version = validate_version_param(version_str)
        except ValueError as ve:
            return jsonify({"error": str(ve)}), 400

        logger.info(f"Flask API request received for PDF. Report ID: {report_id}, User: {current_user.email}, Version: {version}")

        from clinical_reporting.application.services import (
            ReportService, ReportNotFoundException, VersionNotFoundException,
            PathTraversalException, IntegrityFailureException
        )
        service = ReportService(db_path=app.config["DB_PATH"])

        # 1. Enforce access check
        access_status = service.check_report_access(report_id, current_user)
        if access_status == "NOT_FOUND":
            service.log_report_access_event("REPORT_NOT_FOUND", current_user, report_id, "FAILED", "Requested nonexistent report.")
            return jsonify({"error": "Report not found"}), 404
        elif access_status == "UNAUTHORIZED":
            return jsonify({"error": "Authentication required"}), 401
        elif access_status == "FORBIDDEN":
            service.log_report_access_event("REPORT_ACCESS_DENIED", current_user, report_id, "FAILED", "Access denied to patient report.")
            return jsonify({"error": "Access denied to patient report"}), 403

        # 2. Resolve PDF path with security checks and integrity verification
        try:
            pdf_path = service.resolve_secure_pdf_path(report_id, version)
        except (ReportNotFoundException, VersionNotFoundException):
            service.log_report_access_event("REPORT_NOT_FOUND", current_user, report_id, "FAILED", f"Report version {version if version is not None else 'latest'} not found.")
            return jsonify({"error": "Report not found"}), 404
        except PathTraversalException as pte:
            service.log_report_access_event("REPORT_ACCESS_DENIED", current_user, report_id, "FAILED", f"Path traversal attempt: {pte}")
            return jsonify({"error": "Invalid report path"}), 400
        except IntegrityFailureException as ife:
            service.log_report_access_event("REPORT_ACCESS_DENIED", current_user, report_id, "FAILED", f"Integrity failure: {ife}")
            return jsonify({"error": "Report integrity verification failed."}), 422
        except FileNotFoundError:
            return jsonify({"error": "PDF report file not found on server disk"}), 404
        except Exception as e:
            logger.error(f"Error resolving PDF path: {e}")
            return jsonify({"error": "Internal error resolving PDF report."}), 500

        # 3. Log download success and serve
        service.log_report_access_event("REPORT_DOWNLOADED", current_user, report_id, "SUCCESS", f"Downloaded report version {version if version is not None else 'latest'}")
        filename = os.path.basename(pdf_path)

        response = send_file(pdf_path, mimetype="application/pdf", download_name=filename, as_attachment=True)
        response.headers["Content-Disposition"] = f"attachment; filename={filename}"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.route("/api/report/<int:report_id>/json")
    @login_required
    def get_json_report(current_user: User, report_id: int):
        import logging
        logger = logging.getLogger("dashboard.web_server.get_json_report")

        version_str = request.args.get("version")
        try:
            version = validate_version_param(version_str)
        except ValueError as ve:
            return jsonify({"error": str(ve)}), 400

        logger.info(f"Flask API request received for JSON. Report ID: {report_id}, User: {current_user.email}, Version: {version}")

        from clinical_reporting.application.services import (
            ReportService, ReportNotFoundException, VersionNotFoundException,
            PathTraversalException, IntegrityFailureException
        )
        service = ReportService(db_path=app.config["DB_PATH"])

        # 1. Enforce access check
        access_status = service.check_report_access(report_id, current_user)
        if access_status == "NOT_FOUND":
            service.log_report_access_event("REPORT_NOT_FOUND", current_user, report_id, "FAILED", "Requested nonexistent report JSON.")
            return jsonify({"error": "Report not found"}), 404
        elif access_status == "UNAUTHORIZED":
            return jsonify({"error": "Authentication required"}), 401
        elif access_status == "FORBIDDEN":
            service.log_report_access_event("REPORT_ACCESS_DENIED", current_user, report_id, "FAILED", "Access denied to patient report JSON.")
            return jsonify({"error": "Access denied to patient report"}), 403

        # 2. Get and sanitize JSON payload
        try:
            sanitized_json = service.get_report_json_for_export(report_id, version, current_user)
        except (ReportNotFoundException, VersionNotFoundException):
            return jsonify({"error": "Report not found"}), 404
        except PathTraversalException as pte:
            return jsonify({"error": "Invalid report path"}), 400
        except IntegrityFailureException as ife:
            return jsonify({"error": "Report integrity verification failed."}), 422
        except FileNotFoundError:
            return jsonify({"error": "JSON report file not found on server disk"}), 404
        except Exception as e:
            logger.error(f"Error exporting JSON report: {e}")
            return jsonify({"error": "Internal error exporting JSON report."}), 500

        resolved_version = version
        if resolved_version is None:
            try:
                report = service.get_report(report_id)
                resolved_version = report.current_version
            except Exception:
                resolved_version = 1

        filename = f"report_{report_id}_v{resolved_version}.json"

        response = jsonify(sanitized_json)
        response.headers["Content-Disposition"] = f"attachment; filename={filename}"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.route("/api/report/<int:report_id>/csv")
    @login_required
    def get_csv_report(current_user: User, report_id: int):
        import logging
        logger = logging.getLogger("dashboard.web_server.get_csv_report")

        version_str = request.args.get("version")
        try:
            version = validate_version_param(version_str)
        except ValueError as ve:
            return jsonify({"error": str(ve)}), 400

        logger.info(f"Flask API request received for CSV. Report ID: {report_id}, User: {current_user.email}, Version: {version}")

        from clinical_reporting.application.services import (
            ReportService, ReportNotFoundException, VersionNotFoundException,
            PathTraversalException, IntegrityFailureException
        )
        service = ReportService(db_path=app.config["DB_PATH"])

        # 1. Enforce access check
        access_status = service.check_report_access(report_id, current_user)
        if access_status == "NOT_FOUND":
            service.log_report_access_event("REPORT_NOT_FOUND", current_user, report_id, "FAILED", "Requested nonexistent report CSV.")
            return jsonify({"error": "Report not found"}), 404
        elif access_status == "UNAUTHORIZED":
            return jsonify({"error": "Authentication required"}), 401
        elif access_status == "FORBIDDEN":
            service.log_report_access_event("REPORT_ACCESS_DENIED", current_user, report_id, "FAILED", "Access denied to patient report CSV.")
            return jsonify({"error": "Access denied to patient report"}), 403

        # 2. Get CSV payload
        try:
            csv_content = service.get_report_csv_for_export(report_id, version, current_user)
        except (ReportNotFoundException, VersionNotFoundException):
            return jsonify({"error": "Report not found"}), 404
        except PathTraversalException as pte:
            return jsonify({"error": "Invalid report path"}), 400
        except IntegrityFailureException as ife:
            return jsonify({"error": "Report integrity verification failed."}), 422
        except FileNotFoundError:
            return jsonify({"error": "CSV report source file not found"}), 404
        except Exception as e:
            logger.error(f"Error exporting CSV report: {e}")
            return jsonify({"error": "Internal error exporting CSV report."}), 500

        resolved_version = version
        if resolved_version is None:
            try:
                report = service.get_report(report_id)
                resolved_version = report.current_version
            except Exception:
                resolved_version = 1

        filename = f"report_{report_id}_v{resolved_version}.csv"

        from flask import make_response
        response = make_response(csv_content)
        response.headers["Content-Disposition"] = f"attachment; filename={filename}"
        response.headers["Content-Type"] = "text/csv; charset=utf-8"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.route("/api/reports/audit-history", methods=["GET"])
    @login_required
    def get_report_audit_history_flask(current_user: User):
        from clinical_reporting.application.services import ReportService
        service = ReportService(db_path=app.config["DB_PATH"])
        try:
            logs = service.get_report_audit_history(current_user)
            return jsonify(logs)
        except Exception as e:
            app.logger.error(f"Error fetching report audit history: {e}", exc_info=True)
            return jsonify({"error": "Internal report audit history retrieval error"}), 500

    @app.route("/api/report/<int:report_id>/visuals/<image_type>")
    @roles_accepted(Role.ADMIN, Role.DOCTOR)
    def get_visual_scan(current_user: User, report_id: int, image_type: str):
        from clinical_reporting.application.services import ReportService
        service = ReportService(db_path=app.config["DB_PATH"])
        access_status = service.check_report_access(report_id, current_user)
        if access_status == "NOT_FOUND":
            return jsonify({"error": "Report not found"}), 404
        elif access_status == "UNAUTHORIZED":
            return jsonify({"error": "Authentication required"}), 401
        elif access_status == "FORBIDDEN":
            return jsonify({"error": "Access denied to patient report"}), 403

        conn = sqlite3.connect(app.config["DB_PATH"])
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT cr.overlay_path, cr.heatmap_path, cr.mask_path, s.image_path as raw_path
                FROM clinical_reports cr
                JOIN predictions pr ON cr.prediction_id = pr.id
                JOIN mri_scans s ON pr.scan_id = s.id
                WHERE cr.id = ?;
                """,
                (report_id,)
            )
            row = cursor.fetchone()
            if not row:
                try:
                    service.log_report_access_event(
                        "REPORT_NOT_FOUND",
                        current_user,
                        report_id,
                        "FAILED",
                        f"Report not found for visual fetch: {image_type}"
                    )
                except Exception as e:
                    app.logger.error(f"Audit log failed: {e}")
            else:
                try:
                    service.log_report_access_event(
                        "REPORT_VIEWED",
                        current_user,
                        report_id,
                        "SUCCESS",
                        f"Viewed report visual: {image_type}"
                    )
                except Exception as e:
                    app.logger.error(f"Audit log failed: {e}")

            img_path = None

            if row:
                if image_type == "overlay":
                    img_path = row["overlay_path"]
                elif image_type == "heatmap":
                    img_path = row["heatmap_path"]
                elif image_type == "mask":
                    img_path = row["mask_path"]
                elif image_type == "raw":
                    img_path = row["raw_path"]
                else:
                    img_path = None

            if img_path:
                img_path = os.path.abspath(img_path)

            import logging
            logger = logging.getLogger("dashboard.web_server.get_visual_scan")

            if not img_path or not os.path.exists(img_path):
                import numpy as np
                import cv2
                import io

                # Generate placeholder image on the fly
                placeholder = np.zeros((400, 400, 3), dtype=np.uint8)
                placeholder[:] = [42, 23, 15]  # Slate color
                cv2.circle(placeholder, (200, 200), 120, (85, 65, 51), thickness=2)
                cv2.line(placeholder, (150, 200), (250, 200), (105, 85, 71), 1)
                cv2.line(placeholder, (200, 150), (200, 250), (105, 85, 71), 1)
                text = "IMAGE NOT FOUND"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = (400 - text_size[0]) // 2
                text_y = (400 + text_size[1]) // 2
                cv2.putText(placeholder, text, (text_x, text_y), font, font_scale, (139, 116, 100), thickness, cv2.LINE_AA)

                success, encoded_img = cv2.imencode('.png', placeholder)
                if success:
                    return send_file(io.BytesIO(encoded_img.tobytes()), mimetype="image/png"), 404
                return f"Image file not physically present on server: {img_path}", 404

            if img_path.lower().endswith(('.tif', '.tiff')):
                import numpy as np
                import cv2
                import io
                try:
                    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        if img.dtype != np.uint8:
                            img_min, img_max = img.min(), img.max()
                            if img_max > img_min:
                                img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                            else:
                                img = np.zeros_like(img, dtype=np.uint8)
                        success, encoded_img = cv2.imencode('.png', img)
                        if success:
                            return send_file(io.BytesIO(encoded_img.tobytes()), mimetype="image/png")
                except Exception as ex:
                    logger.error(f"Failed to convert TIFF {img_path} to PNG: {ex}")

            mimetype = "image/png"
            if img_path.lower().endswith(".jpg") or img_path.lower().endswith(".jpeg"):
                mimetype = "image/jpeg"
            return send_file(img_path, mimetype=mimetype)
        except Exception as e:
            return str(e), 500
        finally:
            conn.close()

    @app.route("/api/doctor/generate-report", methods=["POST"])
    @roles_accepted(Role.ADMIN, Role.DOCTOR)
    def doctor_generate_report(current_user: User):
        import requests

        # 1. Get input data and files
        if "mri_file" not in request.files:
            return jsonify({"error": "No MRI file uploaded."}), 400

        mri_file = request.files["mri_file"]
        if not mri_file or mri_file.filename == "":
            return jsonify({"error": "Empty MRI file."}), 400

        patient_id = request.form.get("patient_id", "").strip()
        patient_name = request.form.get("patient_name", "").strip()
        patient_age_str = request.form.get("patient_age", "45").strip()
        patient_gender = request.form.get("patient_gender", "Female").strip()
        ref_physician = request.form.get("ref_physician", "").strip()
        pixel_spacing_str = request.form.get("pixel_spacing_mm", "1.0").strip()
        xai_method = request.form.get("xai_method", "gradcam").strip()
        ensemble_mode_str = request.form.get("ensemble_mode", "false").strip()

        # Validate inputs basic
        if not patient_id or not patient_name:
            return jsonify({"error": "Patient ID and Name are required."}), 400
        try:
            patient_age = int(patient_age_str)
        except ValueError:
            return jsonify({"error": "Patient Age must be an integer."}), 400
        try:
            pixel_spacing_mm = float(pixel_spacing_str)
        except ValueError:
            return jsonify({"error": "Pixel spacing must be a float."}), 400

        ensemble_mode = ensemble_mode_str.lower() == "true"

        # Forward token
        token = request.cookies.get("access_token")
        auth_header = request.headers.get("Authorization")
        if auth_header:
            parts = auth_header.split()
            if len(parts) == 2 and parts[0].lower() == "bearer":
                token = parts[1]
            else:
                token = auth_header

        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        # 2. Forward to FastAPI
        api_url = os.environ.get("FASTAPI_URL", "http://127.0.0.1:8000")
        try:
            files = {
                "file": (mri_file.filename, mri_file.read(), mri_file.content_type or "image/png")
            }
            upload_resp = requests.post(f"{api_url}/api/upload", files=files, headers=headers, timeout=30)
            if not upload_resp.ok:
                try:
                    err_msg = upload_resp.json().get("detail", "Failed upload to core API.")
                    if isinstance(err_msg, dict) and "message" in err_msg:
                        err_msg = f"{err_msg['message']}: {', '.join(err_msg.get('errors', []))}"
                except Exception:
                    err_msg = upload_resp.text
                return jsonify({"error": f"MRI Ingestion Failed: {err_msg}"}), upload_resp.status_code

            upload_data = upload_resp.json()
            filepath = upload_data["filepath"]

            intake_payload = {
                "patient_id": patient_id,
                "name": patient_name,
                "age": patient_age,
                "gender": patient_gender,
                "ref_physician": ref_physician,
                "pixel_spacing_mm": pixel_spacing_mm,
                "xai_method": xai_method,
                "ensemble_mode": ensemble_mode
            }

            report_resp = requests.post(
                f"{api_url}/api/report",
                params={"filepath": filepath},
                json=intake_payload,
                headers=headers,
                timeout=60
            )

            if not report_resp.ok:
                try:
                    err_msg = report_resp.json().get("detail", "Failed report execution.")
                    if isinstance(err_msg, dict) and "message" in err_msg:
                        err_msg = f"{err_msg['message']}: {', '.join(err_msg.get('errors', []))}"
                except Exception:
                    err_msg = report_resp.text
                return jsonify({"error": f"AI Diagnostic Failure: {err_msg}"}), report_resp.status_code

            return jsonify(report_resp.json())

        except requests.exceptions.ConnectionError:
            return jsonify({"error": "Failed to connect to AI Inference REST API. Ensure FastAPI server is running on http://127.0.0.1:8000"}), 503
        except Exception as e:
            return jsonify({"error": f"An unexpected pipeline error occurred: {str(e)}"}), 500

    @app.route("/api/doctor/batch-predict", methods=["POST"])
    @roles_accepted(Role.ADMIN, Role.DOCTOR)
    def doctor_batch_predict(current_user: User):
        import requests

        if "mri_files" not in request.files:
            return jsonify({"error": "No MRI files uploaded in batch."}), 400

        mri_files = request.files.getlist("mri_files")
        if not mri_files or len(mri_files) == 0:
            return jsonify({"error": "Empty MRI files batch."}), 400

        patient_id = request.form.get("patient_id", "").strip()
        patient_name = request.form.get("patient_name", "").strip()
        patient_age_str = request.form.get("patient_age", "45").strip()
        patient_gender = request.form.get("patient_gender", "Female").strip()
        ref_physician = request.form.get("ref_physician", "").strip()
        pixel_spacing_str = request.form.get("pixel_spacing_mm", "1.0").strip()
        xai_method = request.form.get("xai_method", "gradcam").strip()
        ensemble_mode_str = request.form.get("ensemble_mode", "false").strip()

        # Validate inputs basic
        if not patient_id or not patient_name:
            return jsonify({"error": "Patient ID and Name are required."}), 400
        try:
            patient_age = int(patient_age_str)
        except ValueError:
            return jsonify({"error": "Patient Age must be an integer."}), 400
        try:
            pixel_spacing_mm = float(pixel_spacing_str)
        except ValueError:
            return jsonify({"error": "Pixel spacing must be a float."}), 400

        # Forward token
        token = request.cookies.get("access_token")
        auth_header = request.headers.get("Authorization")
        if auth_header:
            parts = auth_header.split()
            if len(parts) == 2 and parts[0].lower() == "bearer":
                token = parts[1]
            else:
                token = auth_header

        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        api_url = os.environ.get("FASTAPI_URL", "http://127.0.0.1:8000")
        try:
            # Build files payload for requests library
            files_payload = []
            for f in mri_files:
                if f.filename != "":
                    files_payload.append(
                        ("files", (f.filename, f.read(), f.content_type or "image/png"))
                    )

            if not files_payload:
                return jsonify({"error": "No valid files in batch request."}), 400

            data_payload = {
                "patient_id": patient_id,
                "name": patient_name,
                "age": str(patient_age),
                "gender": patient_gender,
                "ref_physician": ref_physician,
                "pixel_spacing_mm": str(pixel_spacing_mm),
                "xai_method": xai_method,
                "ensemble_mode": ensemble_mode_str
            }

            resp = requests.post(
                f"{api_url}/api/report/batch",
                files=files_payload,
                data=data_payload,
                headers=headers,
                timeout=300 # longer timeout for batch
            )

            if not resp.ok:
                try:
                    err_detail = resp.json().get("detail", "Failed executing batch prediction.")
                except Exception:
                    err_detail = resp.text
                return jsonify({"error": f"Batch Prediction Failed: {err_detail}"}), resp.status_code

            return jsonify(resp.json())

        except requests.exceptions.ConnectionError:
            return jsonify({"error": "Failed to connect to AI Inference REST API. Ensure FastAPI server is running on http://127.0.0.1:8000"}), 503
        except Exception as e:
            return jsonify({"error": f"An unexpected batch error occurred: {str(e)}"}), 500

    @app.route("/verify/<token>")
    def verify_report_page(token: str):
        """Web page for public verification of a report version by secure token."""
        from clinical_reporting.application.services import ReportService
        service = ReportService(db_path=app.config["DB_PATH"])

        result = service.verify_report_by_token(token)
        state = result.get("verification_state", "INVALID")

        if state == "INVALID":
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Report Verification Failed</title>
                <style>
                    body { font-family: 'Helvetica Neue', Arial, sans-serif; background-color: #FDEDEC; color: #78281F; padding: 50px; text-align: center; }
                    .card { background: white; padding: 40px; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); display: inline-block; max-width: 500px; border-top: 5px solid #C0392B; }
                    h1 { color: #C0392B; font-size: 24px; margin-bottom: 20px; }
                    p { font-size: 16px; line-height: 1.5; color: #5D6D7E; }
                    .icon { font-size: 48px; color: #C0392B; margin-bottom: 20px; }
                </style>
            </head>
            <body>
                <div class="card">
                    <div class="icon">✕</div>
                    <h1>Verification Failed</h1>
                    <p>The verification token is invalid or does not exist in the system database.</p>
                </div>
            </body>
            </html>
            """
            return render_template_string(html_content), 404

        elif state == "TAMPERED":
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Report Integrity Warning</title>
                <style>
                    body {{ font-family: 'Helvetica Neue', Arial, sans-serif; background-color: #FDEDEC; color: #78281F; padding: 50px; text-align: center; }}
                    .card {{ background: white; padding: 40px; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); display: inline-block; max-width: 500px; border-top: 5px solid #C0392B; text-align: left; }}
                    h1 {{ color: #C0392B; font-size: 24px; margin-bottom: 20px; text-align: center; }}
                    p {{ font-size: 14px; line-height: 1.5; color: #2C3E50; margin: 10px 0; }}
                    .icon {{ font-size: 48px; color: #C0392B; margin-bottom: 20px; text-align: center; }}
                    .label {{ font-weight: bold; color: #34495E; }}
                </style>
            </head>
            <body>
                <div class="card">
                    <div class="icon">⚠️</div>
                    <h1>TAMPERED REPORT</h1>
                    <p style="text-align: center; font-weight: bold; color: #C0392B; margin-bottom: 20px;">✕ Integrity mismatch detected!</p>
                    <p><span class="label">Report:</span> {result['report_number']}</p>
                    <p><span class="label">Version:</span> {result['version']}</p>
                    <p><span class="label">Status:</span> {result['status']}</p>
                    <p><span class="label">Integrity:</span> <span style="color: #C0392B; font-weight: bold;">TAMPERED</span></p>
                    <p><span class="label">Generated:</span> {result['created_at']}</p>
                </div>
            </body>
            </html>
            """
            return render_template_string(html_content), 200

        elif state == "SUPERSEDED":
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Report Verified (Superseded)</title>
                <style>
                    body {{ font-family: 'Helvetica Neue', Arial, sans-serif; background-color: #FEF9E7; color: #7D6608; padding: 50px; text-align: center; }}
                    .card {{ background: white; padding: 40px; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); display: inline-block; max-width: 500px; border-top: 5px solid #F39C12; text-align: left; }}
                    h1 {{ color: #D35400; font-size: 24px; margin-bottom: 20px; text-align: center; }}
                    p {{ font-size: 14px; line-height: 1.5; color: #2C3E50; margin: 10px 0; }}
                    .icon {{ font-size: 48px; color: #F39C12; margin-bottom: 20px; text-align: center; }}
                    .label {{ font-weight: bold; color: #34495E; }}
                </style>
            </head>
            <body>
                <div class="card">
                    <div class="icon">ℹ️</div>
                    <h1>Report Verified</h1>
                    <p style="text-align: center; font-weight: bold; color: #D35400; margin-bottom: 20px;">Report verified but superseded by a newer version</p>
                    <p><span class="label">Report:</span> {result['report_number']}</p>
                    <p><span class="label">Version:</span> {result['version']}</p>
                    <p><span class="label">Status:</span> SUPERSEDED</p>
                    <p><span class="label">Integrity:</span> <span style="color: #27AE60; font-weight: bold;">VERIFIED</span></p>
                    <p><span class="label">Generated:</span> {result['created_at']}</p>
                </div>
            </body>
            </html>
            """
            return render_template_string(html_content), 200

        elif state == "ARCHIVED":
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Report Verified (Archived)</title>
                <style>
                    body {{ font-family: 'Helvetica Neue', Arial, sans-serif; background-color: #EAEDED; color: #2C3E50; padding: 50px; text-align: center; }}
                    .card {{ background: white; padding: 40px; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); display: inline-block; max-width: 500px; border-top: 5px solid #7F8C8D; text-align: left; }}
                    h1 {{ color: #34495E; font-size: 24px; margin-bottom: 20px; text-align: center; }}
                    p {{ font-size: 14px; line-height: 1.5; color: #2C3E50; margin: 10px 0; }}
                    .icon {{ font-size: 48px; color: #7F8C8D; margin-bottom: 20px; text-align: center; }}
                    .label {{ font-weight: bold; color: #34495E; }}
                </style>
            </head>
            <body>
                <div class="card">
                    <div class="icon">📦</div>
                    <h1>Report Verified</h1>
                    <p style="text-align: center; font-weight: bold; color: #7F8C8D; margin-bottom: 20px;">Report verified (Status: ARCHIVED)</p>
                    <p><span class="label">Report:</span> {result['report_number']}</p>
                    <p><span class="label">Version:</span> {result['version']}</p>
                    <p><span class="label">Status:</span> ARCHIVED</p>
                    <p><span class="label">Integrity:</span> <span style="color: #27AE60; font-weight: bold;">VERIFIED</span></p>
                    <p><span class="label">Generated:</span> {result['created_at']}</p>
                </div>
            </body>
            </html>
            """
            return render_template_string(html_content), 200

        else: # VALID
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Report Verified</title>
                <style>
                    body {{ font-family: 'Helvetica Neue', Arial, sans-serif; background-color: #E8F8F5; color: #117864; padding: 50px; text-align: center; }}
                    .card {{ background: white; padding: 40px; border-radius: 8px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); display: inline-block; max-width: 500px; border-top: 5px solid #27AE60; text-align: left; }}
                    h1 {{ color: #27AE60; font-size: 24px; margin-bottom: 20px; text-align: center; }}
                    p {{ font-size: 14px; line-height: 1.5; color: #2C3E50; margin: 10px 0; }}
                    .icon {{ font-size: 48px; color: #27AE60; margin-bottom: 20px; text-align: center; }}
                    .label {{ font-weight: bold; color: #34495E; }}
                </style>
            </head>
            <body>
                <div class="card">
                    <div class="icon">✓</div>
                    <h1>VALID REPORT</h1>
                    <p style="text-align: center; font-weight: bold; color: #27AE60; margin-bottom: 20px;">✓ Report authenticity verified</p>
                    <p><span class="label">Report:</span> {result['report_number']}</p>
                    <p><span class="label">Version:</span> {result['version']}</p>
                    <p><span class="label">Status:</span> {result['status']}</p>
                    <p><span class="label">Integrity:</span> <span style="color: #27AE60; font-weight: bold;">VERIFIED</span></p>
                    <p><span class="label">Generated:</span> {result['created_at']}</p>
                </div>
            </body>
            </html>
            """
            return render_template_string(html_content), 200

    @app.route("/api/reports/<int:report_id>/email", methods=["POST"])
    @login_required
    def email_report_flask(current_user: User, report_id: int):
        from clinical_reporting.application.services import ReportService, ReportNotFoundException, VersionNotFoundException
        from clinical_reporting.infrastructure.email_service import ConfigurationException
        service = ReportService(db_path=app.config["DB_PATH"])
        data = request.get_json() or {}
        recipient_email = data.get("recipient_email")
        version = data.get("version")

        # Enforce version validation if provided
        if version is not None:
            try:
                version = validate_version_param(version)
            except ValueError as e:
                return jsonify({"error": str(e)}), 400

        try:
            result = service.send_report_email(
                report_id=report_id,
                actor=current_user,
                recipient_email=recipient_email,
                version=version
            )
            return jsonify(result)
        except ReportNotFoundException as e:
            return jsonify({"error": str(e)}), 404
        except VersionNotFoundException as e:
            return jsonify({"error": str(e)}), 404
        except PermissionError as e:
            return jsonify({"error": str(e)}), 403
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except FileNotFoundError as e:
            return jsonify({"error": "PDF report file not found on the server."}), 404
        except ConfigurationException as e:
            return jsonify({"error": str(e)}), 503
        except Exception as e:
            app.logger.error(f"Error emailing report: {e}")
            return jsonify({"error": "Internal server error sending report email."}), 500

    @app.route("/api/reports/email/history", methods=["GET"])
    @login_required
    def get_email_history_flask(current_user: User):
        from clinical_reporting.application.services import ReportService
        service = ReportService(db_path=app.config["DB_PATH"])

        try:
            page = int(request.args.get("page", 1))
            per_page = int(request.args.get("per_page", 10))
            if page < 1:
                page = 1
            if per_page < 1 or per_page > 100:
                per_page = 10
        except ValueError:
            page = 1
            per_page = 10

        status = request.args.get("status")
        search = request.args.get("search")
        report_id_str = request.args.get("report_id")
        report_id = None
        if report_id_str:
            try:
                report_id = int(report_id_str)
            except ValueError:
                return jsonify({"error": "Invalid report_id parameter."}), 400

        try:
            result = service.get_email_history(
                actor=current_user,
                page=page,
                per_page=per_page,
                status=status,
                search=search,
                report_id=report_id
            )
            return jsonify(result)
        except PermissionError as e:
            return jsonify({"error": str(e)}), 403
        except Exception as e:
            app.logger.error(f"Error fetching email history: {e}")
            return jsonify({"error": "Internal server error fetching email history."}), 500

    return app
