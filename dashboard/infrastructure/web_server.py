import os
import sqlite3
from flask import Flask, jsonify, request, send_file, render_template_string
from typing import Optional

from persistence.infrastructure.repository import SQLitePersistenceRepository
from prediction_history.infrastructure.repository import SQLitePredictionHistoryRepository
from prediction_history.domain.entities import HistorySearchCriteria


def create_app(db_path: str) -> Flask:
    """Factory function to build and configure the Flask web dashboard application.

    Args:
        db_path: Path to the SQLite database.

    Returns:
        Configured Flask application instance.
    """
    # Configure relative templates directory pointing to presentation folder
    template_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "presentation", "templates")
    )
    app = Flask(__name__, template_folder=template_dir)
    app.config["DB_PATH"] = db_path

    # Initialize repositories
    persistence_repo = SQLitePersistenceRepository(db_path=db_path)
    history_repo = SQLitePredictionHistoryRepository(db_path=db_path)

    # Make sure DB schema is set up
    persistence_repo.initialize_db()

    @app.route("/")
    def index():
        """Serves the dashboard single page application."""
        # Find index.html inside the configured template directory
        index_path = os.path.join(template_dir, "index.html")
        if not os.path.exists(index_path):
            return f"Error: index.html presentation template not found at {index_path}", 404
        
        with open(index_path, "r", encoding="utf-8") as f:
            content = f.read()
        return render_template_string(content)

    @app.route("/api/analytics")
    def analytics():
        """API Endpoint: returns high-level diagnostic counts and averages."""
        try:
            summary = persistence_repo.get_analytics_summary()
            return jsonify(summary)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/history")
    def history():
        """API Endpoint: returns the list of all prediction summaries."""
        try:
            # Query all records (empty criteria)
            criteria = HistorySearchCriteria()
            summaries = history_repo.search_history(criteria)
            
            # Serialize to dict list
            data = []
            for s in summaries:
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
    def search():
        """API Endpoint: searches prediction summaries matching a patient ID query."""
        q = request.args.get("q", "").strip()
        try:
            criteria = HistorySearchCriteria(patient_id=q if q else None)
            summaries = history_repo.search_history(criteria)
            
            data = []
            for s in summaries:
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
    def get_report_details(report_id: int):
        """API Endpoint: returns the clinical findings description and rules for a Report ID."""
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
            return jsonify(dict(row))
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        finally:
            conn.close()

    @app.route("/api/report/<int:report_id>/pdf")
    def get_pdf(report_id: int):
        """Streams the generated clinical PDF directly to the web client."""
        paths = history_repo.get_report_paths(report_id)
        if not paths or not paths[2]:
            return "PDF report not found in database record", 404
        pdf_path = paths[2]
        if not os.path.exists(pdf_path):
            return f"PDF file not physically present on server: {pdf_path}", 404
        return send_file(pdf_path, mimetype="application/pdf")

    @app.route("/api/report/<int:report_id>/visuals/<image_type>")
    def get_visual_scan(report_id: int, image_type: str):
        """Streams the original scan, overlay, or mask image to the browser.

        Args:
            report_id: Clinical report ID.
            image_type: One of 'overlay', 'heatmap', 'mask'.
        """
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

            # Map selection
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

            # Determine mimetype
            mimetype = "image/png"
            if img_path.lower().endswith(".jpg") or img_path.lower().endswith(".jpeg"):
                mimetype = "image/jpeg"
            return send_file(img_path, mimetype=mimetype)
        except Exception as e:
            return str(e), 500
        finally:
            conn.close()

    return app
