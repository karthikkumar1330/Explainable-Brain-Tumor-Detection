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
            created_at TEXT NOT NULL,
            FOREIGN KEY (prediction_id) REFERENCES predictions(id) ON DELETE CASCADE
        );
        """

        # Analytics Indices
        indices = [
            "CREATE INDEX IF NOT EXISTS idx_patients_age_gender ON patients(age, gender);",
            "CREATE INDEX IF NOT EXISTS idx_mri_scans_date ON mri_scans(scan_date);",
            "CREATE INDEX IF NOT EXISTS idx_predictions_class_severity ON predictions(predicted_class, rule_based_severity);",
            "CREATE INDEX IF NOT EXISTS idx_predictions_area ON predictions(tumor_area_mm2);"
        ]

        conn = self._get_connection()
        try:
            with conn:
                conn.execute(create_patients_sql)
                conn.execute(create_scans_sql)
                conn.execute(create_predictions_sql)
                conn.execute(create_reports_sql)
                for idx_sql in indices:
                    conn.execute(idx_sql)
            self.logger.info("Database schema and analytics indices verified successfully.")
        except Exception as e:
            self.logger.error(f"Failed to initialize SQLite database: {e}")
            raise e
        finally:
            conn.close()

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
                    heatmap_path, overlay_path, mask_path, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
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
            cr.pdf_path
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
