import sqlite3
import logging
from typing import List, Optional, Tuple, Any
from prediction_history.domain.entities import HistorySearchCriteria, PredictionSummary
from prediction_history.domain.interfaces import IPredictionHistoryRepository


class PaginatedList(list):
    """Subclass of list that carries total count metadata for pagination."""
    def __init__(self, items, total_count: int) -> None:
        super().__init__(items)
        self.total_count = total_count


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
            A list of PredictionSummary dataclasses (or PaginatedList).
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
            cr.created_at,
            s.ref_physician as referring_doctor,
            r.status as report_status
        FROM patients p
        JOIN mri_scans s ON p.patient_id = s.patient_id
        JOIN predictions pr ON s.id = pr.scan_id
        JOIN clinical_reports cr ON pr.id = cr.prediction_id
        LEFT JOIN reports r ON cr.id = r.report_id
        """

        count_base_query = """
        SELECT COUNT(*) as cnt
        FROM patients p
        JOIN mri_scans s ON p.patient_id = s.patient_id
        JOIN predictions pr ON s.id = pr.scan_id
        JOIN clinical_reports cr ON pr.id = cr.prediction_id
        LEFT JOIN reports r ON cr.id = r.report_id
        """

        conditions = []
        params: List[Any] = []

        # Non-PII Exact/Partial matches
        if criteria.patient_id is not None:
            conditions.append("p.patient_id LIKE ?")
            params.append(f"%{criteria.patient_id}%")

        if criteria.referring_doctor is not None:
            conditions.append("s.ref_physician LIKE ?")
            params.append(f"%{criteria.referring_doctor}%")

        if criteria.report_id is not None:
            conditions.append("cr.id = ?")
            params.append(criteria.report_id)

        if criteria.scan_date is not None:
            conditions.append("s.scan_date LIKE ?")
            params.append(f"%{criteria.scan_date}%")

        # Specific Dropdowns / Ranges
        if criteria.classification is not None:
            conditions.append("LOWER(pr.predicted_class) = LOWER(?)")
            params.append(criteria.classification)

        if criteria.severity is not None:
            conditions.append("LOWER(pr.rule_based_severity) = LOWER(?)")
            params.append(criteria.severity)

        if criteria.min_confidence is not None:
            conditions.append("pr.confidence_score >= ?")
            params.append(criteria.min_confidence)

        if criteria.start_date is not None:
            conditions.append("s.scan_date >= ?")
            params.append(criteria.start_date)

        if criteria.end_date is not None:
            conditions.append("s.scan_date <= ?")
            params.append(criteria.end_date)

        if criteria.report_status is not None:
            conditions.append("LOWER(r.status) = LOWER(?)")
            params.append(criteria.report_status)

        # Compile final queries
        where_clause = ""
        if conditions:
            where_clause = " WHERE " + " AND ".join(conditions)

        main_query = base_query + where_clause

        # Safe Sorting Whitelist
        sort_by_map = {
            "report_id": "cr.id",
            "prediction_id": "pr.id",
            "patient_id": "p.patient_id",
            "scan_date": "s.scan_date",
            "predicted_class": "pr.predicted_class",
            "confidence_score": "pr.confidence_score",
            "tumor_area_mm2": "pr.tumor_area_mm2",
            "rule_based_severity": "pr.rule_based_severity",
            "created_at": "cr.created_at",
            "referring_doctor": "s.ref_physician",
            "report_status": "r.status",
        }
        # If sorted by patient_name, we sort in Python, otherwise let DB do base sorting
        sort_col = sort_by_map.get(criteria.sort_by, "cr.created_at")
        sort_dir = "ASC" if criteria.sort_order and criteria.sort_order.lower() == "asc" else "DESC"
        order_clause = f" ORDER BY {sort_col} {sort_dir}"

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            # Fetch all candidate records for the non-PII query
            query = main_query + order_clause
            cursor.execute(query, params)
            rows = cursor.fetchall()

            # Instantiate PII encryption service
            try:
                from security.infrastructure.encryption_service import PIIEncryptionService
                encryption_service = PIIEncryptionService()
            except Exception:
                encryption_service = None

            # Decrypt in memory
            candidates = []
            for row in rows:
                p_name = row["patient_name"]
                if p_name is not None:
                    is_enc = str(p_name).startswith("enc:v1:")
                    if is_enc:
                        if encryption_service is None:
                            raise ValueError("PII patient_name is encrypted but PII_ENCRYPTION_KEY is missing.")
                        decrypted_name = encryption_service.decrypt(p_name)
                    else:
                        decrypted_name = p_name
                else:
                    decrypted_name = ""

                candidates.append({
                    "report_id": row["report_id"],
                    "prediction_id": row["prediction_id"],
                    "patient_id": row["patient_id"],
                    "patient_name": decrypted_name,
                    "scan_date": row["scan_date"],
                    "predicted_class": row["predicted_class"],
                    "confidence_score": row["confidence_score"],
                    "tumor_area_mm2": row["tumor_area_mm2"],
                    "rule_based_severity": row["rule_based_severity"],
                    "created_at": row["created_at"],
                    "referring_doctor": row["referring_doctor"],
                    "report_status": row["report_status"]
                })

            # Filter in Python
            filtered_candidates = []
            for c in candidates:
                # Patient RBAC filter
                if criteria.restrict_to_patient_uuid is not None and criteria.restrict_to_patient_name is not None:
                    uuid_match = str(c["patient_id"]).lower() == str(criteria.restrict_to_patient_uuid).lower()
                    name_match = str(c["patient_name"]).lower() == str(criteria.restrict_to_patient_name).lower()
                    if not (uuid_match or name_match):
                        continue

                # Global query q filter
                if criteria.q is not None:
                    q_val = str(criteria.q).lower()
                    pat_id_match = q_val in str(c["patient_id"]).lower()
                    pat_name_match = q_val in str(c["patient_name"]).lower()
                    ref_doc_match = q_val in str(c["referring_doctor"]).lower() if c["referring_doctor"] else False
                    if not (pat_id_match or pat_name_match or ref_doc_match):
                        continue

                # Exact/Partial name filter
                if criteria.patient_name is not None:
                    name_search = str(criteria.patient_name).lower()
                    if name_search not in str(c["patient_name"]).lower():
                        continue

                filtered_candidates.append(c)

            # Sort in Python if patient_name was requested
            if criteria.sort_by == "patient_name":
                reverse_sort = not (criteria.sort_order and criteria.sort_order.lower() == "asc")
                filtered_candidates.sort(key=lambda x: str(x["patient_name"]).lower(), reverse=reverse_sort)

            # Paginate in Python
            total_count = len(filtered_candidates)
            results = []

            if criteria.page is not None and criteria.page_size is not None:
                offset = (criteria.page - 1) * criteria.page_size
                page_items = filtered_candidates[offset : offset + criteria.page_size]
            else:
                page_items = filtered_candidates

            for c in page_items:
                results.append(PredictionSummary(
                    report_id=c["report_id"],
                    prediction_id=c["prediction_id"],
                    patient_id=c["patient_id"],
                    patient_name=c["patient_name"],
                    scan_date=c["scan_date"],
                    predicted_class=c["predicted_class"],
                    confidence_score=c["confidence_score"],
                    tumor_area_mm2=c["tumor_area_mm2"],
                    rule_based_severity=c["rule_based_severity"],
                    created_at=c["created_at"],
                    referring_doctor=c["referring_doctor"],
                    report_status=c["report_status"]
                ))
            return PaginatedList(results, total_count)
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
