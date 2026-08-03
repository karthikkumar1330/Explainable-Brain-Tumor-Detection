import json
import logging
from typing import Optional, Dict, Any
from input_validation.domain.entities import (
    ValidationScorecard,
    DuplicateCheckResult
)
from input_validation.domain.interfaces import IMriValidator
from persistence.infrastructure.repository import SQLitePersistenceRepository


class ValidateMriUploadUseCase:
    """Use case to run magic numbers checks, brain detection, quality QA, and duplicate checks."""

    def __init__(
        self,
        validator: IMriValidator,
        db_path: str,
        logger: Optional[logging.Logger] = None
    ) -> None:
        self.validator = validator
        self.db_path = db_path
        self.logger = logger or logging.getLogger("validate_mri_use_case")

    def execute(
        self,
        filepath: str,
        file_bytes: bytes,
        filename: str,
        prediction_id: Optional[int] = None
    ) -> ValidationScorecard:
        """Executes the validation workflow, checks for duplicates, and persists scorecard results.

        Args:
            filepath: Path where the file is temporarily cached.
            file_bytes: Raw binary bytes of the upload.
            filename: Original name of the uploaded scan.
            prediction_id: Optional database prediction ID to link validation telemetry.

        Returns:
            A ValidationScorecard detailing verification results.
        """
        # 1. Run baseline file, format, brain detector, and quality metrics
        scorecard = self.validator.validate_file(filepath, file_bytes, filename)
        
        # 2. Extract hashes for duplicate verification
        file_hash = scorecard.duplicate_check.duplicate_hash
        p_hash = ""
        if hasattr(self.validator, "compute_perceptual_hash"):
            p_hash = self.validator.compute_perceptual_hash(file_bytes)

        # 3. Check for duplicates in SQLite database
        db_repo = SQLitePersistenceRepository(db_path=self.db_path)
        duplicate_info = None
        try:
            duplicate_info = db_repo.find_duplicate_scan(file_hash, p_hash)
        except Exception as e:
            self.logger.error(f"Duplicate scan check failed: {e}")

        # Re-compile error list if duplicate is found
        errors = list(scorecard.errors)
        is_valid = scorecard.is_valid

        if duplicate_info:
            is_valid = False
            dup_type = duplicate_info["type"]
            pat_id = duplicate_info["patient_id"]
            scan_date = duplicate_info["scan_date"]
            
            dup_msg = (
                f"Duplicate scan detected ({dup_type} match). This slice is identical or perceptually similar to "
                f"a scan already analyzed for Patient ID: '{pat_id}' on {scan_date}."
            )
            errors.append(dup_msg)
            
            duplicate_check = DuplicateCheckResult(
                is_duplicate=True,
                duplicate_hash=file_hash,
                duplicate_patient_id=pat_id,
                duplicate_scan_date=scan_date
            )
        else:
            duplicate_check = DuplicateCheckResult(
                is_duplicate=False,
                duplicate_hash=file_hash
            )

        # Re-instantiate scorecard with final duplicate status
        updated_scorecard = ValidationScorecard(
            is_valid=is_valid,
            file_validation=scorecard.file_validation,
            image_validation=scorecard.image_validation,
            brain_detection=scorecard.brain_detection,
            quality_assessment=scorecard.quality_assessment,
            duplicate_check=duplicate_check,
            errors=errors
        )

        # 4. Persist validation scorecard into SQLite database
        try:
            scorecard_json_str = json.dumps(updated_scorecard.to_dict())
            db_repo.save_validation_scorecard(
                file_hash=file_hash,
                p_hash=p_hash,
                is_valid=is_valid,
                scorecard_json=scorecard_json_str,
                prediction_id=prediction_id
            )
        except Exception as e:
            self.logger.error(f"Failed to save validation scorecard to database: {e}")

        return updated_scorecard
