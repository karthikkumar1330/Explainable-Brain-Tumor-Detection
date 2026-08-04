import os
import sqlite3
import functools
from flask import Flask, jsonify, request, send_file, render_template_string, session
from typing import Optional, List, Dict, Any

from persistence.infrastructure.repository import SQLitePersistenceRepository
from prediction_history.infrastructure.repository import SQLitePredictionHistoryRepository
from prediction_history.domain.entities import HistorySearchCriteria

from security.domain.entities import Role, User
from security.infrastructure.repository import SQLiteUserRepository
from security.infrastructure.jwt_service import JWTService
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

    # Security Headers Middleware
    @app.after_request
    def set_security_headers(response):
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self' 'unsafe-inline' 'unsafe-eval' data: blob: https://fonts.googleapis.com https://fonts.gstatic.com;"
        return response

    # Auth Helper
    def get_current_user_from_request() -> Optional[User]:
        # Check Bearer Header
        auth_header = request.headers.get("Authorization")
        token = None
        if auth_header:
            parts = auth_header.split()
            if len(parts) == 2 and parts[0].lower() == "bearer":
                token = parts[1]
            else:
                token = auth_header
        elif "access_token" in session:
            token = session["access_token"]

        if not token:
            return None

        payload = jwt_svc.decode_token(token)
        if not payload:
            return None

        if user_repo.is_token_revoked(payload.jti):
            return None

        user = user_repo.get_by_id(payload.user_id)
        if not user or not user.is_active:
            return None

        return user

    def login_required(f):
        @functools.wraps(f)
        def decorated(*args, **kwargs):
            user = get_current_user_from_request()
            if not user:
                return jsonify({"error": "Authentication required. Please login."}), 401
            return f(user, *args, **kwargs)
        return decorated

    def roles_accepted(*roles: Role):
        def decorator(f):
            @functools.wraps(f)
            def decorated(*args, **kwargs):
                user = get_current_user_from_request()
                if not user:
                    return jsonify({"error": "Authentication required. Please login."}), 401
                if user.role not in roles and Role.ADMIN not in roles:
                    return jsonify({"error": f"Access denied. Required privileges: {[r.value for r in roles]}"}), 403
                return f(user, *args, **kwargs)
            return decorated
        return decorator

    # --- Web Routes ---
    @app.route("/")
    def index():
        index_path = os.path.join(template_dir, "index.html")
        if not os.path.exists(index_path):
            return f"Error: index.html presentation template not found at {index_path}", 404
        
        with open(index_path, "r", encoding="utf-8") as f:
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
            return jsonify(res)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/api/auth/verify-email", methods=["POST"])
    def auth_verify_email():
        data = request.get_json() or {}
        ip_addr = request.remote_addr or "127.0.0.1"
        try:
            res = auth_use_cases.verify_email(
                token_or_otp=data.get("token_or_otp", ""),
                email=data.get("email"),
                ip_address=ip_addr
            )
            return jsonify(res)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/api/auth/login", methods=["POST"])
    def auth_login():
        data = request.get_json() or {}
        ip_addr = request.remote_addr or "127.0.0.1"
        try:
            res = auth_use_cases.login(
                email=data.get("email", ""),
                password=data.get("password", ""),
                ip_address=ip_addr
            )
            if not res.get("requires_2fa") and "access_token" in res:
                session["access_token"] = res["access_token"]
                session["user"] = res["user"]
            return jsonify(res)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/api/auth/verify-otp", methods=["POST"])
    def auth_verify_otp():
        data = request.get_json() or {}
        ip_addr = request.remote_addr or "127.0.0.1"
        try:
            res = auth_use_cases.verify_login_otp(
                user_id=int(data.get("user_id", 0)),
                otp_code=data.get("otp_code", ""),
                ip_address=ip_addr
            )
            if "access_token" in res:
                session["access_token"] = res["access_token"]
                session["user"] = res["user"]
            return jsonify(res)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/api/auth/logout", methods=["POST"])
    def auth_logout():
        token = session.get("access_token")
        auth_header = request.headers.get("Authorization")
        if auth_header:
            token = auth_header.split()[-1]
        ip_addr = request.remote_addr or "127.0.0.1"
        if token:
            auth_use_cases.logout(token=token, ip_address=ip_addr)
        session.clear()
        return jsonify({"message": "Logout successful."})

    @app.route("/api/auth/me", methods=["GET"])
    def auth_me():
        user = get_current_user_from_request()
        if not user:
            return jsonify({"authenticated": False}), 200
        return jsonify({"authenticated": True, "user": user.to_dict()})

    @app.route("/api/auth/forgot-password", methods=["POST"])
    def auth_forgot_password():
        data = request.get_json() or {}
        ip_addr = request.remote_addr or "127.0.0.1"
        res = auth_use_cases.forgot_password(email=data.get("email", ""), ip_address=ip_addr)
        return jsonify(res)

    @app.route("/api/auth/reset-password", methods=["POST"])
    def auth_reset_password():
        data = request.get_json() or {}
        ip_addr = request.remote_addr or "127.0.0.1"
        try:
            res = auth_use_cases.reset_password(
                reset_token_or_otp=data.get("reset_token_or_otp", ""),
                new_password=data.get("new_password", ""),
                email=data.get("email"),
                ip_address=ip_addr
            )
            return jsonify(res)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    @app.route("/api/auth/profile", methods=["PUT"])
    @login_required
    def auth_update_profile(current_user: User):
        data = request.get_json() or {}
        try:
            res = auth_use_cases.update_profile(
                user_id=current_user.id,
                full_name=data.get("full_name"),
                enable_2fa=data.get("enable_2fa")
            )
            return jsonify(res)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

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
        users = user_repo.list_users(limit=200)
        return jsonify({"users": [u.to_dict() for u in users]})

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
        paths = history_repo.get_report_paths(report_id)
        if not paths or not paths[2]:
            return "PDF report not found in database record", 404
        pdf_path = paths[2]
        if not os.path.exists(pdf_path):
            return f"PDF file not physically present on server: {pdf_path}", 404
        return send_file(pdf_path, mimetype="application/pdf")

    @app.route("/api/report/<int:report_id>/visuals/<image_type>")
    @login_required
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

    return app
