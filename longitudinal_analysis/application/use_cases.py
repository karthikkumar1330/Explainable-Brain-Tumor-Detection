import os
import json
import logging
from typing import Optional, Dict, Any
from longitudinal_analysis.domain.entities import LongitudinalComparison
from longitudinal_analysis.domain.interfaces import ILongitudinalAnalyzer
from persistence.infrastructure.repository import SQLitePersistenceRepository


class CompareScansUseCase:
    """Use case to compare the current scan with the patient's previous historical scan."""

    def __init__(
        self,
        analyzer: ILongitudinalAnalyzer,
        db_path: str,
        logger: Optional[logging.Logger] = None
    ) -> None:
        self.analyzer = analyzer
        self.db_path = db_path
        self.logger = logger or logging.getLogger("compare_scans_use_case")

    def execute(
        self,
        patient_id: str,
        current_report_data: Dict[str, Any],
        output_image_path: Optional[str] = None
    ) -> Optional[LongitudinalComparison]:
        """Runs the comparison using patient history from the database.

        Args:
            patient_id: Unique identifier for the patient.
            current_report_data: Dictionary of the current scan report.
            output_image_path: Path to save comparison canvas.

        Returns:
            A LongitudinalComparison object if a previous scan exists, else None.
        """
        db_repo = SQLitePersistenceRepository(db_path=self.db_path)
        try:
            history = db_repo.get_patient_history(patient_id)
        except Exception as e:
            self.logger.error(f"Failed to retrieve patient history from DB: {e}")
            return None
        
        current_scan_date = current_report_data.get("patient", {}).get("scan_date", "")

        # Find the immediately preceding scan record chronologically older than current
        previous_record = None
        for record in history:
            rec_date = record.get("scan_date", "")
            if rec_date < current_scan_date:
                previous_record = record
                break

        if previous_record is None:
            self.logger.info(f"No previous historical scan found for patient {patient_id} before {current_scan_date}.")
            return None

        # Load previous JSON file
        json_path = previous_record.get("json_path")
        if not json_path or not os.path.exists(json_path):
            self.logger.warning(f"Previous report JSON file not found at: {json_path}")
            # Try to build fallback from database record fields
            previous_report_data = {
                "patient": {
                    "patient_id": patient_id,
                    "scan_date": previous_record.get("scan_date", "")
                },
                "classification": {
                    "predicted_class": previous_record.get("predicted_class", "No Tumor"),
                    "confidence_score": previous_record.get("confidence_score", 0.0)
                },
                "segmentation": {
                    "tumor_area_mm2": previous_record.get("tumor_area_mm2", 0.0),
                    "tumor_percentage_brain": 0.0
                },
                "files": {
                    "original_image": previous_record.get("image_path", ""),
                    "segmentation_mask": previous_record.get("mask_path", "")
                }
            }
        else:
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    previous_report_data = json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load previous report JSON file: {e}")
                return None

        # Run comparison analysis
        return self.analyzer.compare(
            current_data=current_report_data,
            previous_data=previous_report_data,
            output_image_path=output_image_path
        )
