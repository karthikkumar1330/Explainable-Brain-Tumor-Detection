import sqlite3
import logging
from typing import List, Optional, Tuple, Any
from prediction_history.domain.entities import HistorySearchCriteria, PredictionSummary
from prediction_history.domain.interfaces import IPredictionHistoryRepository


class SQLitePredictionHistoryRepository(IPredictionHistoryRepository):
    """SQLite concrete repository querying clinical and prediction database records."""

    def __init__(self, db_path: str, logger: Optional[logging.Logger] = None) -> None:
        """Initializes the prediction history repository.

        Args:
            db_path: Path to the SQLite database.
            logger: Optional logger.
        """
        self.db_path = db_path
        self.logger = logger or logging.getLogger("prediction_history_repo")

    def _get_connection(self) -> sqlite3.Connection:
        """Creates a database connection with performance pragmas enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        # Optimize read concurrency
        try:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
            conn.execute("PRAGMA cache_size = -2000;")
            conn.execute("PRAGMA temp_store = MEMORY;")
        except Exception:
            pass
            
        return conn

    def search_history(self, criteria: HistorySearchCriteria) -> List[PredictionSummary]:
        """Searches prediction summaries matching the filter criteria.

        Args:
            criteria: Search criteria constraints.

        Returns:
            A list of PredictionSummary dataclasses.
        """
        self.logger.info(f"Querying SQLite database: {self.db_path}")
        
        base_query = """
        SELECT 
            cr.id as report_id,
            pr.id as prediction_id,
            p.patient_id,
            p.name as patient_name,
            s.scan_date,
            pr.predicted_class,
            pr.confidence_score,
            pr.tumor_area_mm2,
            pr.rule_based_severity,
            cr.created_at
        FROM patients p
        JOIN mri_scans s ON p.patient_id = s.patient_id
        JOIN predictions pr ON s.id = pr.scan_id
        JOIN clinical_reports cr ON pr.id = cr.prediction_id
        """

        conditions = []
        params: List[Any] = []

        if criteria.patient_id is not None:
            conditions.append("p.patient_id LIKE ?")
            params.append(f"%{criteria.patient_id}%")

        if criteria.report_id is not None:
            conditions.append("cr.id = ?")
            params.append(criteria.report_id)

        if criteria.scan_date is not None:
            conditions.append("s.scan_date LIKE ?")
            params.append(f"%{criteria.scan_date}%")

        # Compile final query
        if conditions:
            query = base_query + " WHERE " + " AND ".join(conditions)
        else:
            query = base_query

        # Order by newest reports first
        query += " ORDER BY cr.created_at DESC"

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                results.append(PredictionSummary(
                    report_id=row["report_id"],
                    prediction_id=row["prediction_id"],
                    patient_id=row["patient_id"],
                    patient_name=row["patient_name"],
                    scan_date=row["scan_date"],
                    predicted_class=row["predicted_class"],
                    confidence_score=row["confidence_score"],
                    tumor_area_mm2=row["tumor_area_mm2"],
                    rule_based_severity=row["rule_based_severity"],
                    created_at=row["created_at"]
                ))
            return results
        except Exception as e:
            self.logger.error(f"Failed to query database for history: {e}")
            raise e
        finally:
            conn.close()

    def get_report_paths(self, report_id: int) -> Optional[Tuple[str, str, str]]:
        """Retrieves file paths for Markdown, JSON, and PDF reports for a Report ID.

        Args:
            report_id: Database clinical report identifier.

        Returns:
            A tuple of (markdown_path, json_path, pdf_path) if found, otherwise None.
        """
        conn = self._get_connection()
        query = """
        SELECT markdown_path, json_path, pdf_path
        FROM clinical_reports
        WHERE id = ?;
        """
        try:
            cursor = conn.cursor()
            cursor.execute(query, (report_id,))
            row = cursor.fetchone()
            if row:
                return row["markdown_path"], row["json_path"], row["pdf_path"]
            return None
        except Exception as e:
            self.logger.error(f"Failed to query report paths for report_id {report_id}: {e}")
            raise e
        finally:
            conn.close()
