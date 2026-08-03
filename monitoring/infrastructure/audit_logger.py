import json
import datetime
import logging
from typing import List, Optional, Dict, Any
from persistence.infrastructure.repository import SQLitePersistenceRepository


class AuditLogger:
    """Service to log pipeline execution telemetry and Quality Assurance check results to SQLite."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.logger = logging.getLogger("audit_logger")

    def log_execution(
        self,
        patient_id: str,
        user: str,
        model_version_cls: str,
        model_version_seg: str,
        runtime_sec: float,
        gpu_active: bool,
        cpu_threads: int,
        warnings: List[str],
        errors: List[str],
        report_status: str,
        database_status: str,
        prediction_id: Optional[int] = None
    ) -> None:
        """Saves a structured audit record into the database, handling failures gracefully."""
        db_repo = SQLitePersistenceRepository(db_path=self.db_path)
        
        entry = {
            "prediction_id": prediction_id,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_id": user,
            "patient_id": patient_id,
            "model_version_cls": model_version_cls,
            "model_version_seg": model_version_seg,
            "runtime_sec": float(runtime_sec),
            "gpu_active": 1 if gpu_active else 0,
            "cpu_threads": int(cpu_threads),
            "warnings_json": json.dumps(warnings),
            "errors_json": json.dumps(errors),
            "report_status": report_status,
            "database_status": database_status
        }
        
        try:
            db_repo.save_audit_log(entry)
            self.logger.info(f"AI Audit Log saved successfully for patient {patient_id}.")
        except Exception as e:
            self.logger.error(f"Failed to persist AI audit log to database: {e}")
