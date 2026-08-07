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
    app = Flask(__name__, template_folder=template_dir)
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
                httponly=False  # Must be False so frontend JS can read and submit it
            )
        
        # Configure robust Enterprise Security Headers
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://cdnjs.cloudflare.com https://fonts.googleapis.com; "
            "font-src 'self' https://cdnjs.cloudflare.com https://fonts.gstatic.com; "
            "img-src 'self' data:; "
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
            "/api/auth/register"
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
                    details=f"CSRF mismatch. Header: {header_csrf}, Cookie: {cookie_csrf}",
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
                rev_str = user.sessions_revoked_at
                if not rev_str.endswith("Z") and "+00:00" not in rev_str:
                    rev_str += "Z"
                rev_str = rev_str.replace("Z", "+00:00")
                rev_dt = datetime.datetime.fromisoformat(rev_str)
                if payload.iat < rev_dt.timestamp():
                    return None, "TOKEN_REVOKED"
            except Exception:
                pass

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
                    samesite="Lax"
                )
                response.set_cookie(
                    "refresh_token",
                    res["refresh_token"],
                    max_age=30 * 86400,
                    httponly=True,
                    secure=is_secure,
                    samesite="Lax"
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
                    samesite="Lax"
                )
                refresh_max_age = 30 * 86400 if remember_me else None
                response.set_cookie(
                    "refresh_token",
                    res["refresh_token"],
                    max_age=refresh_max_age,
                    httponly=True,
                    secure=is_secure,
                    samesite="Lax"
                )
            return response
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
            res = auth_use_cases.refresh_token(refresh_token=refresh_token, ip_addr=ip_addr)
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
                samesite="Lax"
            )
            
            refresh_max_age = 30 * 86400 if remember_me else None
            response.set_cookie(
                "refresh_token",
                res["refresh_token"],
                max_age=refresh_max_age,
                httponly=True,
                secure=is_secure,
                samesite="Lax"
            )
            return response
        except ValueError as e:
            response = jsonify({"error": str(e), "code": "REFRESH_TOKEN_INVALID"}), 401
            response.set_cookie("access_token", "", expires=0, httponly=True, samesite="Lax")
            response.set_cookie("refresh_token", "", expires=0, httponly=True, samesite="Lax")
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
        response.set_cookie("access_token", "", expires=0, httponly=True, samesite="Lax")
        response.set_cookie("refresh_token", "", expires=0, httponly=True, samesite="Lax")
        return response

    @app.route("/api/auth/me", methods=["GET"])
    def auth_me():
        user, err_code = get_current_user_from_request()
        if not user:
            return jsonify({"authenticated": False, "code": err_code}), 200
        return jsonify({"authenticated": True, "user": user.to_dict()})

    @app.route("/api/auth/profile", methods=["PUT"])
    @login_required
    def auth_update_profile(current_user: User):
        data = request.get_json() or {}
        try:
            res = auth_use_cases.update_profile(
                user_id=current_user.id,
                full_name=data.get("full_name"),
                email=data.get("email")
            )
            return jsonify(res)
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

        # Validate file extension
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in [".png", ".jpg", ".jpeg", ".gif", ".webp"]:
            return jsonify({"error": "Invalid image format. Allowed formats: PNG, JPG, JPEG, GIF, WEBP."}), 400

        filename = f"{current_user.uuid}{file_ext}"
        filepath = os.path.join(upload_dir, filename)
        file.save(filepath)

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
        now = datetime.datetime.utcnow().isoformat()
        current_user.sessions_revoked_at = now
        user_repo.update_user(current_user)

        # Generate fresh tokens for the current session (issued at now, which is >= sessions_revoked_at)
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
        response.set_cookie("access_token", access_token, max_age=30*60, httponly=True, secure=is_secure, samesite="Lax")
        response.set_cookie("refresh_token", refresh_token, max_age=30*86400, httponly=True, secure=is_secure, samesite="Lax")

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
            response.set_cookie("access_token", "", expires=0, httponly=True, samesite="Lax")
            response.set_cookie("refresh_token", "", expires=0, httponly=True, samesite="Lax")
            return response
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/auth/profile/sessions", methods=["GET"])
    @login_required
    def auth_get_sessions(current_user: User):
        conn = sqlite3.connect(app.config["DB_PATH"])
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT timestamp, event_type, ip_address, details
                FROM security_audit_logs
                WHERE user_id = ? AND status = 'SUCCESS' AND (event_type LIKE 'LOGIN%' OR event_type = 'REVOKE_ALL_SESSIONS')
                ORDER BY id DESC LIMIT 15;
            """, (current_user.id,))
            rows = cursor.fetchall()
            sessions = [dict(r) for r in rows]
            return jsonify({"sessions": sessions})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
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
            return jsonify({"error": str(e)}), 500

    @app.route("/api/history")
    @login_required
    def history(current_user: User):
        try:
            criteria = HistorySearchCriteria()
            summaries = history_repo.search_history(criteria)
            
            data = []
            for s in summaries:
                # If Patient role, only return matching patient records
                if current_user.role == Role.PATIENT:
                    if s.patient_name.lower() != current_user.full_name.lower() and s.patient_id.lower() != current_user.uuid.lower():
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
            return jsonify({"error": str(e)}), 500

    @app.route("/api/search")
    @login_required
    def search(current_user: User):
        q = request.args.get("q", "").strip()
        try:
            criteria = HistorySearchCriteria(patient_id=q if q else None)
            summaries = history_repo.search_history(criteria)
            
            data = []
            for s in summaries:
                if current_user.role == Role.PATIENT:
                    if s.patient_name.lower() != current_user.full_name.lower() and s.patient_id.lower() != current_user.uuid.lower():
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
            return jsonify({"error": str(e)}), 500

    @app.route("/api/report/<int:report_id>")
    @login_required
    def get_report_details(current_user: User, report_id: int):
        conn = sqlite3.connect(app.config["DB_PATH"])
        conn.row_factory = sqlite3.Row
        try:
            query = """
            SELECT 
                cr.id as report_id, p.patient_id, p.name as patient_name,
                pr.predicted_class, pr.confidence_score, pr.tumor_area_mm2, pr.tumor_percentage_brain,
                pr.rule_based_severity, pr.severity_rule_description, cr.created_at
            FROM clinical_reports cr
            JOIN predictions pr ON cr.prediction_id = pr.id
            JOIN mri_scans s ON pr.scan_id = s.id
            JOIN patients p ON s.patient_id = p.patient_id
            WHERE cr.id = ?;
            """
            cursor = conn.cursor()
            cursor.execute(query, (report_id,))
            row = cursor.fetchone()
            if not row:
                return jsonify({"error": "Report not found"}), 404

            report_dict = dict(row)
            if current_user.role == Role.PATIENT:
                if report_dict["patient_name"].lower() != current_user.full_name.lower() and report_dict["patient_id"].lower() != current_user.uuid.lower():
                    return jsonify({"error": "Access denied to patient report"}), 403

            return jsonify(report_dict)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            conn.close()

    @app.route("/api/report/<int:report_id>/pdf")
    @login_required
    def get_pdf(current_user: User, report_id: int):
        if current_user.role == Role.PATIENT:
            conn = sqlite3.connect(app.config["DB_PATH"])
            conn.row_factory = sqlite3.Row
            try:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT p.patient_id, p.name as patient_name
                    FROM clinical_reports cr
                    JOIN predictions pr ON cr.prediction_id = pr.id
                    JOIN mri_scans s ON pr.scan_id = s.id
                    JOIN patients p ON s.patient_id = p.patient_id
                    WHERE cr.id = ?;
                    """,
                    (report_id,)
                )
                row = cursor.fetchone()
                if not row:
                    return "Report not found", 404
                if row["patient_name"].lower() != current_user.full_name.lower() and row["patient_id"].lower() != current_user.uuid.lower():
                    return "Access denied to patient report PDF", 403
            finally:
                conn.close()

        paths = history_repo.get_report_paths(report_id)
        if not paths or not paths[2]:
            return "PDF report not found in database record", 404
        pdf_path = paths[2]
        if not os.path.exists(pdf_path):
            return f"PDF file not physically present on server: {pdf_path}", 404
        return send_file(pdf_path, mimetype="application/pdf")

    @app.route("/api/report/<int:report_id>/visuals/<image_type>")
    @roles_accepted(Role.ADMIN, Role.DOCTOR)
    def get_visual_scan(current_user: User, report_id: int, image_type: str):
        conn = sqlite3.connect(app.config["DB_PATH"])
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT overlay_path, heatmap_path, mask_path FROM clinical_reports WHERE id = ?;",
                (report_id,)
            )
            row = cursor.fetchone()
            if not row:
                return "Report visuals not found", 404

            if image_type == "overlay":
                img_path = row["overlay_path"]
            elif image_type == "heatmap":
                img_path = row["heatmap_path"]
            elif image_type == "mask":
                img_path = row["mask_path"]
            else:
                return "Invalid visual image type. Choose 'overlay', 'heatmap', or 'mask'.", 400

            if not img_path or not os.path.exists(img_path):
                return f"Image file not physically present on server: {img_path}", 404

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

    return app
