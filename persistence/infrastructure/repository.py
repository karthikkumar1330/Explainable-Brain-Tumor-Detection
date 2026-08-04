import sqlite3
import datetime
import logging
import os
from typing import List, Dict, Any, Optional
from clinical_reporting.domain.entities import ClinicalReport
from persistence.domain.interfaces import IPersistenceRepository


class SQLitePersistenceRepository(IPersistenceRepository):
    """SQLite implementation of the persistence repository layer."""

    def __init__(self, db_path: str, logger: Optional[logging.Logger] = None) -> None:
        """Initializes the SQLite repository.

        Args:
            db_path: Path to the SQLite database file.
            logger: Optional logger instance.
        """
        self.db_path = db_path
        self.logger = logger or logging.getLogger("db_persistence")

    def _get_connection(self) -> sqlite3.Connection:
        """Creates a database connection with foreign keys and performance pragmas enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        # Enforce foreign keys
        conn.execute("PRAGMA foreign_keys = ON;")
        
        # Optimize writes and read-concurrency using WAL mode
        try:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.execute("PRAGMA cache_size = -2000;")
            conn.execute("PRAGMA temp_store = MEMORY;")
        except Exception:
            pass
            
        return conn

    def initialize_db(self) -> None:
        """Initializes the database schema tables and optimization indices."""
        self.logger.info(f"Initializing SQLite database at: {self.db_path}")
        
        create_patients_sql = """
        CREATE TABLE IF NOT EXISTS patients (
            patient_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            gender TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """

        create_scans_sql = """
        CREATE TABLE IF NOT EXISTS mri_scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            image_path TEXT NOT NULL,
            pixel_spacing_mm REAL NOT NULL,
            ref_physician TEXT NOT NULL,
            scan_date TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
        );
        """

        create_predictions_sql = """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_id INTEGER NOT NULL,
            predicted_class TEXT NOT NULL,
            confidence_score REAL NOT NULL,
            prob_glioma REAL NOT NULL,
            prob_meningioma REAL NOT NULL,
            prob_pituitary REAL NOT NULL,
            prob_no_tumor REAL NOT NULL,
            tumor_pixel_count INTEGER NOT NULL,
            tumor_area_mm2 REAL NOT NULL,
            tumor_percentage_brain REAL NOT NULL,
            tumor_percentage_image REAL NOT NULL,
            estimated_brain_pixel_count INTEGER NOT NULL,
            rule_based_severity TEXT NOT NULL,
            severity_rule_description TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (scan_id) REFERENCES mri_scans(id) ON DELETE CASCADE
        );
        """

        create_reports_sql = """
        CREATE TABLE IF NOT EXISTS clinical_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER NOT NULL,
            markdown_path TEXT NOT NULL,
            json_path TEXT NOT NULL,
            pdf_path TEXT NOT NULL,
            heatmap_path TEXT,
            overlay_path TEXT,
            mask_path TEXT,
            xai_method TEXT,
            xai_overlap_percentage REAL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
        );
        """

        create_validation_sql = """
        CREATE TABLE IF NOT EXISTS mri_scan_validation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER,
            file_hash TEXT UNIQUE NOT NULL,
            p_hash TEXT NOT NULL,
            is_valid INTEGER NOT NULL,
            scorecard_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (prediction_id) REFERENCES predictions(id) ON DELETE SET NULL
        );
        """

        create_audit_logs_sql = """
        CREATE TABLE IF NOT EXISTS ai_audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER,
            timestamp TEXT NOT NULL,
            user_id TEXT NOT NULL,
            patient_id TEXT NOT NULL,
            model_version_cls TEXT NOT NULL,
            model_version_seg TEXT NOT NULL,
            runtime_sec REAL NOT NULL,
            gpu_active INTEGER NOT NULL,
            cpu_threads INTEGER NOT NULL,
            warnings_json TEXT NOT NULL,
            errors_json TEXT NOT NULL,
            report_status TEXT NOT NULL,
            database_status TEXT NOT NULL,
            FOREIGN KEY (prediction_id) REFERENCES predictions(id) ON DELETE SET NULL
        );
        """

        create_timeline_traces_sql = """
        CREATE TABLE IF NOT EXISTS timeline_traces (
            prediction_id INTEGER PRIMARY KEY,
            trace_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
        );
        """

        # Analytics Indices
        indices = [
            "CREATE INDEX IF NOT EXISTS idx_patients_age_gender ON patients(age, gender);",
            "CREATE INDEX IF NOT EXISTS idx_mri_scans_date ON mri_scans(scan_date);",
            "CREATE INDEX IF NOT EXISTS idx_predictions_class_severity ON predictions(predicted_class, rule_based_severity);",
            "CREATE INDEX IF NOT EXISTS idx_predictions_area ON predictions(tumor_area_mm2);",
            "CREATE INDEX IF NOT EXISTS idx_mri_scan_validation_hash ON mri_scan_validation(file_hash, p_hash);",
            "CREATE INDEX IF NOT EXISTS idx_ai_audit_logs_timestamp ON ai_audit_logs(timestamp);",
            "CREATE INDEX IF NOT EXISTS idx_ai_audit_logs_patient ON ai_audit_logs(patient_id);"
        ]

        conn = self._get_connection()
        try:
            with conn:
                conn.execute(create_patients_sql)
                conn.execute(create_scans_sql)
                conn.execute(create_predictions_sql)
                conn.execute(create_reports_sql)
                conn.execute(create_validation_sql)
                conn.execute(create_audit_logs_sql)
                conn.execute(create_timeline_traces_sql)
                for idx_sql in indices:
                    conn.execute(idx_sql)
                
                # Check and migrate existing clinical_reports schema
                try:
                    conn.execute("SELECT xai_method FROM clinical_reports LIMIT 1;")
                except sqlite3.OperationalError:
                    try:
                        conn.execute("ALTER TABLE clinical_reports ADD COLUMN xai_method TEXT;")
                        conn.execute("ALTER TABLE clinical_reports ADD COLUMN xai_overlap_percentage REAL;")
                    except Exception as alt_err:
                        self.logger.warning(f"Could not migrate clinical_reports schema: {alt_err}")
            self.logger.info("Database schema and analytics indices verified successfully.")
        except Exception as e:
            self.logger.error(f"Failed to initialize SQLite database: {e}")
            raise e
        finally:
            conn.close()

        # Initialize security tables and bootstrap admin
        try:
            from security.infrastructure.repository import SQLiteUserRepository
            user_repo = SQLiteUserRepository(db_path=self.db_path, logger=self.logger)
            user_repo.initialize_security_tables()
            user_repo.bootstrap_admin()
        except Exception as sec_err:
            self.logger.warning(f"Could not initialize security tables: {sec_err}")


    def save_report(self, report: ClinicalReport, output_dir: str) -> int:
        """Persists the full integrated clinical report findings into SQLite.

        Args:
            report: The combined ClinicalReport entity.
            output_dir: Destination folder where reports are physically written.

        Returns:
            The saved clinical report record ID.
        """
        self.logger.info(f"Persisting findings for patient: {report.patient_info.patient_id}")
        conn = self._get_connection()
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            with conn:
                # 1. Insert/Update Patient Record
                patient_sql = """
                INSERT INTO patients (patient_id, name, age, gender, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(patient_id) DO UPDATE SET
                    name=excluded.name,
                    age=excluded.age,
                    gender=excluded.gender;
                """
                conn.execute(patient_sql, (
                    report.patient_info.patient_id,
                    report.patient_info.name,
                    report.patient_info.age,
                    report.patient_info.gender,
                    now_str
                ))

                # 2. Insert Scan Record
                scan_sql = """
                INSERT INTO mri_scans (patient_id, image_path, pixel_spacing_mm, ref_physician, scan_date, created_at)
                VALUES (?, ?, ?, ?, ?, ?);
                """
                cursor = conn.execute(scan_sql, (
                    report.patient_info.patient_id,
                    report.original_image_path,
                    report.processing_summary.segmentation_latency_sec, # placeholder spatial spacing or actual
                    report.patient_info.ref_physician,
                    report.patient_info.scan_date,
                    now_str
                ))
                scan_id = cursor.lastrowid

                # 3. Extract metrics
                pixel_count = 0
                tumor_area = 0.0
                pct_brain = 0.0
                pct_image = 0.0
                brain_pixels = 0
                if report.segmentation_metrics is not None:
                    pixel_count = report.segmentation_metrics.pixel_count
                    tumor_area = report.segmentation_metrics.tumor_area_mm2
                    pct_brain = report.segmentation_metrics.tumor_percentage_brain
                    pct_image = report.segmentation_metrics.tumor_percentage_image
                    brain_pixels = report.segmentation_metrics.estimated_brain_pixel_count

                severity = "LOW"
                rule_desc = "No active tumor mass detected"
                if report.severity_assessment is not None:
                    severity = report.severity_assessment.category.value
                    rule_desc = report.severity_assessment.rule_description

                probs = report.classification.probabilities
                prob_glioma = probs.get("Glioma", 0.0)
                prob_meningioma = probs.get("Meningioma", 0.0)
                prob_pituitary = probs.get("Pituitary", 0.0)
                prob_no_tumor = probs.get("No Tumor", 0.0)

                # 4. Insert Prediction Record
                pred_sql = """
                INSERT INTO predictions (
                    scan_id, predicted_class, confidence_score,
                    prob_glioma, prob_meningioma, prob_pituitary, prob_no_tumor,
                    tumor_pixel_count, tumor_area_mm2, tumor_percentage_brain, tumor_percentage_image,
                    estimated_brain_pixel_count, rule_based_severity, severity_rule_description, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """
                cursor = conn.execute(pred_sql, (
                    scan_id,
                    report.classification.class_name,
                    report.classification.confidence_score,
                    prob_glioma, prob_meningioma, prob_pituitary, prob_no_tumor,
                    pixel_count, tumor_area, pct_brain, pct_image,
                    brain_pixels, severity, rule_desc, now_str
                ))
                pred_id = cursor.lastrowid

                # 5. Insert Report Record
                report_sql = """
                INSERT INTO clinical_reports (
                    prediction_id, markdown_path, json_path, pdf_path,
                    heatmap_path, overlay_path, mask_path, xai_method, xai_overlap_percentage, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """
                # Extract paths from report object
                # Markdown & JSON generated outputs are generated using PatientID prefix
                # The caller should make sure actual paths are populated.
                base_name = f"{report.patient_info.patient_id}_clinical_report"
                md_p = os.path.join(output_dir, f"{base_name}.md")
                js_p = os.path.join(output_dir, f"{base_name}.json")
                pdf_p = os.path.join(output_dir, f"{base_name}.pdf")

                cursor = conn.execute(report_sql, (
                    pred_id,
                    md_p,
                    js_p,
                    pdf_p,
                    report.heatmap_image_path,
                    report.overlay_image_path,
                    report.segmentation_mask_path,
                    getattr(report, "xai_method", None),
                    getattr(report, "xai_overlap_percentage", None),
                    now_str
                ))
                report_id = cursor.lastrowid
                
            self.logger.info(f"Report findings saved successfully. Assigned Database Report ID: {report_id}")
            return report_id
        except Exception as e:
            self.logger.error(f"Transaction failed when saving clinical findings: {e}")
            raise e
        finally:
            conn.close()

    def get_patient_history(self, patient_id: str) -> List[Dict[str, Any]]:
        """Queries database records to pull scan and severity history for a patient.

        Args:
            patient_id: Patient ID.

        Returns:
            A list of dictionary records containing diagnostic details.
        """
        conn = self._get_connection()
        query = """
        SELECT 
            p.name, p.age, p.gender,
            s.scan_date, s.ref_physician, s.image_path,
            pr.predicted_class, pr.confidence_score, pr.tumor_area_mm2, pr.rule_based_severity, pr.created_at,
            cr.pdf_path, cr.json_path, cr.mask_path, cr.overlay_path
        FROM patients p
        JOIN mri_scans s ON p.patient_id = s.patient_id
        JOIN predictions pr ON s.id = pr.scan_id
        JOIN clinical_reports cr ON pr.id = cr.prediction_id
        WHERE p.patient_id = ?
        ORDER BY pr.created_at DESC;
        """
        try:
            cursor = conn.cursor()
            cursor.execute(query, (patient_id,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            self.logger.error(f"Failed to query history for patient {patient_id}: {e}")
            raise e
        finally:
            conn.close()

    def get_analytics_summary(self) -> Dict[str, Any]:
        """Runs dashboard analytical aggregations on scan metrics.

        Returns:
            A dictionary containing counts and distributions.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()

            # 1. Total Patients & Scans
            cursor.execute("SELECT COUNT(*) FROM patients;")
            total_patients = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM mri_scans;")
            total_scans = cursor.fetchone()[0]

            # 2. Diagnosis Distribution
            cursor.execute("""
            SELECT predicted_class, COUNT(*) as cnt 
            FROM predictions 
            GROUP BY predicted_class;
            """)
            diag_rows = cursor.fetchall()
            diag_dist = {row['predicted_class']: row['cnt'] for row in diag_rows}

            # 3. Severity Distribution
            cursor.execute("""
            SELECT rule_based_severity, COUNT(*) as cnt 
            FROM predictions 
            GROUP BY rule_based_severity;
            """)
            sev_rows = cursor.fetchall()
            sev_dist = {row['rule_based_severity']: row['cnt'] for row in sev_rows}

            # 4. Average Tumor Area by Class (for tumors with pixel count > 0)
            cursor.execute("""
            SELECT predicted_class, AVG(tumor_area_mm2) as avg_area
            FROM predictions
            WHERE tumor_pixel_count > 0
            GROUP BY predicted_class;
            """)
            area_rows = cursor.fetchall()
            avg_area = {row['predicted_class']: round(row['avg_area'], 2) for row in area_rows}

            return {
                "total_patients": total_patients,
                "total_scans": total_scans,
                "classification_distribution": diag_dist,
                "severity_distribution": sev_dist,
                "average_tumor_area_mm2": avg_area,
            }
        except Exception as e:
            self.logger.error(f"Failed to fetch database analytics: {e}")
            raise e
        finally:
            conn.close()

    def save_validation_scorecard(
        self,
        file_hash: str,
        p_hash: str,
        is_valid: bool,
        scorecard_json: str,
        prediction_id: Optional[int] = None
    ) -> None:
        """Persists validation scorecard findings and perceptual hash references."""
        conn = self._get_connection()
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        sql = """
        INSERT INTO mri_scan_validation (
            prediction_id, file_hash, p_hash, is_valid, scorecard_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(file_hash) DO UPDATE SET
            prediction_id=coalesce(excluded.prediction_id, prediction_id),
            is_valid=excluded.is_valid,
            scorecard_json=excluded.scorecard_json;
        """
        try:
            with conn:
                conn.execute(sql, (
                    prediction_id,
                    file_hash,
                    p_hash,
                    1 if is_valid else 0,
                    scorecard_json,
                    now_str
                ))
            self.logger.info("MRI scan validation scorecard saved successfully.")
        except Exception as e:
            self.logger.error(f"Failed to save validation scorecard: {e}")
            raise e
        finally:
            conn.close()

    def find_duplicate_scan(
        self,
        file_hash: str,
        p_hash: str
    ) -> Optional[dict]:
        """Performs cryptographic and perceptual hashing lookup to identify duplicates."""
        conn = self._get_connection()
        
        # 1. First, check direct cryptographic SHA256 match
        crypto_query = """
        SELECT 
            v.file_hash,
            p.patient_id,
            s.scan_date
        FROM mri_scan_validation v
        JOIN predictions pr ON v.prediction_id = pr.id
        JOIN mri_scans s ON pr.scan_id = s.id
        JOIN patients p ON s.patient_id = p.patient_id
        WHERE v.file_hash = ?;
        """
        try:
            cursor = conn.cursor()
            cursor.execute(crypto_query, (file_hash,))
            row = cursor.fetchone()
            if row:
                return {
                    "type": "cryptographic",
                    "hash": row["file_hash"],
                    "patient_id": row["patient_id"],
                    "scan_date": row["scan_date"]
                }
            
            # 2. Check perceptual hash matching with hamming distance threshold <= 2 bits
            perceptual_query = """
            SELECT 
                v.file_hash,
                v.p_hash,
                p.patient_id,
                s.scan_date
            FROM mri_scan_validation v
            JOIN predictions pr ON v.prediction_id = pr.id
            JOIN mri_scans s ON pr.scan_id = s.id
            JOIN patients p ON s.patient_id = p.patient_id
            WHERE v.is_valid = 1;
            """
            cursor.execute(perceptual_query)
            rows = cursor.fetchall()
            
            # Helper to calculate hamming distance between two hex-string hashes
            def hamming_distance(h1: str, h2: str) -> int:
                if len(h1) != len(h2):
                    return 64  # mismatch length, maximum distance
                try:
                    bin1 = bin(int(h1, 16))[2:].zfill(64)
                    bin2 = bin(int(h2, 16))[2:].zfill(64)
                    return sum(c1 != c2 for c1, c2 in zip(bin1, bin2))
                except ValueError:
                    return 64

            for row in rows:
                db_phash = row["p_hash"]
                if not db_phash or not p_hash:
                    continue
                dist = hamming_distance(db_phash, p_hash)
                if dist <= 2:  # Extremely high similarity threshold
                    return {
                        "type": "perceptual",
                        "hash": row["file_hash"],
                        "patient_id": row["patient_id"],
                        "scan_date": row["scan_date"],
                        "hamming_distance": dist
                    }
            return None
        except Exception as e:
            self.logger.error(f"Failed to query database for duplicates: {e}")
            raise e
        finally:
            conn.close()

    def save_audit_log(self, entry: Dict[str, Any]) -> None:
        """Persists a telemetry audit record log in SQLite database."""
        conn = self._get_connection()
        sql = """
        INSERT INTO ai_audit_logs (
            prediction_id, timestamp, user_id, patient_id, model_version_cls,
            model_version_seg, runtime_sec, gpu_active, cpu_threads, warnings_json,
            errors_json, report_status, database_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        try:
            with conn:
                conn.execute(sql, (
                    entry.get("prediction_id"),
                    entry["timestamp"],
                    entry["user_id"],
                    entry["patient_id"],
                    entry["model_version_cls"],
                    entry["model_version_seg"],
                    entry["runtime_sec"],
                    entry["gpu_active"],
                    entry["cpu_threads"],
                    entry["warnings_json"],
                    entry["errors_json"],
                    entry["report_status"],
                    entry["database_status"]
                ))
            self.logger.info("AI Audit Log record saved successfully.")
        except Exception as e:
            self.logger.error(f"Failed to insert AI Audit Log record: {e}")
            raise e
        finally:
            conn.close()

    def get_health_telemetry(self) -> Dict[str, Any]:
        """Queries SQLite database to compile historical AI health, pipeline metrics and analytics statistics."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            # 1. Total predictions count
            cursor.execute("SELECT COUNT(*) FROM predictions;")
            total_predictions = cursor.fetchone()[0]
            
            # 2. Average confidence score
            cursor.execute("SELECT AVG(confidence_score) FROM predictions;")
            row = cursor.fetchone()
            avg_confidence = row[0] if row[0] is not None else 0.0
            
            # 3. Average runtime (from audit logs)
            cursor.execute("SELECT AVG(runtime_sec) FROM ai_audit_logs;")
            row = cursor.fetchone()
            avg_runtime = row[0] if row[0] is not None else 0.0
            
            # 4. Duplicate upload counts
            cursor.execute("SELECT COUNT(*) FROM mri_scan_validation WHERE is_valid = 0 AND scorecard_json LIKE '%Duplicate scan detected%';")
            duplicate_uploads = cursor.fetchone()[0]
            
            # 5. Database health (returns number of tables, size estimation)
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [r["name"] for r in cursor.fetchall()]
            db_healthy = "clinical_reports" in tables and "predictions" in tables
            
            # 6. Diagnosis Distribution for charts
            cursor.execute("SELECT predicted_class, COUNT(*) as cnt FROM predictions GROUP BY predicted_class;")
            diag_rows = cursor.fetchall()
            diag_dist = {r["predicted_class"]: r["cnt"] for r in diag_rows}
            
            # 7. Average tumor area
            cursor.execute("SELECT AVG(tumor_area_mm2) FROM predictions WHERE tumor_pixel_count > 0;")
            row = cursor.fetchone()
            avg_tumor_area = row[0] if row[0] is not None else 0.0

            # 8. Avg Grad-CAM Overlap
            cursor.execute("SELECT AVG(xai_overlap_percentage) FROM clinical_reports WHERE xai_overlap_percentage IS NOT NULL;")
            row = cursor.fetchone()
            avg_xai_overlap = row[0] if row[0] is not None else 0.0
            
            # 9. Active XAI methods count
            cursor.execute("SELECT xai_method, COUNT(*) as cnt FROM clinical_reports WHERE xai_method IS NOT NULL GROUP BY xai_method;")
            xai_rows = cursor.fetchall()
            xai_methods = {r["xai_method"]: r["cnt"] for r in xai_rows}
            
            return {
                "total_predictions": total_predictions,
                "avg_confidence": avg_confidence,
                "avg_runtime": avg_runtime,
                "duplicate_uploads": duplicate_uploads,
                "db_healthy": db_healthy,
                "tables_count": len(tables),
                "diagnosis_distribution": diag_dist,
                "avg_tumor_area": avg_tumor_area,
                "avg_xai_overlap": avg_xai_overlap,
                "xai_methods": xai_methods
            }
        except Exception as e:
            self.logger.error(f"Failed to query health telemetry: {e}")
            return {
                "total_predictions": 0,
                "avg_confidence": 0.0,
                "avg_runtime": 0.0,
                "duplicate_uploads": 0,
                "db_healthy": False,
                "tables_count": 0,
                "diagnosis_distribution": {},
                "avg_tumor_area": 0.0,
                "avg_xai_overlap": 0.0,
                "xai_methods": {}
            }
        finally:
            conn.close()

    def save_timeline_trace(self, prediction_id: int, timeline_data: Dict[str, float]) -> None:
        """Persists the latency timeline traces for a prediction run."""
        import json
        import datetime
        conn = self._get_connection()
        sql = "INSERT OR REPLACE INTO timeline_traces (prediction_id, trace_json, created_at) VALUES (?, ?, ?);"
        try:
            with conn:
                conn.execute(sql, (prediction_id, json.dumps(timeline_data), datetime.datetime.now().isoformat()))
            self.logger.info(f"Timeline trace saved for prediction ID: {prediction_id}")
        except Exception as e:
            self.logger.error(f"Failed to save timeline trace: {e}")
        finally:
            conn.close()

    def get_timeline_trace(self, prediction_id: int) -> Optional[Dict[str, float]]:
        """Loads execution timelines for a given prediction ID."""
        import json
        conn = self._get_connection()
        sql = "SELECT trace_json FROM timeline_traces WHERE prediction_id = ?;"
        try:
            row = conn.execute(sql, (prediction_id,)).fetchone()
            if row:
                return json.loads(row["trace_json"])
            return None
        except Exception as e:
            self.logger.error(f"Failed to retrieve timeline trace: {e}")
            return None
        finally:
            conn.close()
