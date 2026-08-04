from typing import Dict, Any, List, Optional
from monitoring.domain.entities import (
    SystemHealthReport,
    SegmentationValidationReport,
    ConsistencyReport
)
from monitoring.infrastructure.health_monitor import PipelineHealthMonitor
from monitoring.infrastructure.segmentation_validator import SegmentationValidator
from monitoring.infrastructure.consistency_checker import ConfidenceConsistencyChecker


class RunPipelineHealthCheckUseCase:
    """Use case to execute end-to-end system telemetry and model health checks."""

    def __init__(self, monitor: PipelineHealthMonitor) -> None:
        self.monitor = monitor

    def execute(
        self,
        model_cls: Optional[Any],
        model_seg: Optional[Any],
        device: str
    ) -> SystemHealthReport:
        """Invokes the pipeline health checks and returns a SystemHealthReport."""
        return self.monitor.run_health_checks(model_cls, model_seg, device)


class ValidateDiagnosticQualityUseCase:
    """Use case to validate segmentation outputs and verify classification confidence alignment."""

    def __init__(
        self,
        seg_validator: SegmentationValidator,
        consistency_checker: ConfidenceConsistencyChecker
    ) -> None:
        self.seg_validator = seg_validator
        self.consistency_checker = consistency_checker

    def execute(
        self,
        predicted_class: str,
        confidence_score: float,
        is_calibrated: bool,
        uncalibrated_confidence: Optional[float],
        probabilities: Optional[Dict[str, float]],
        mask: Any,
        expected_shape: tuple,
        tumor_area_mm2: float,
        brain_pixels: Optional[int] = None
    ) -> Dict[str, Any]:
        """Runs the segmentation checks and consistency logic, aggregating warning alerts.

        Args:
            predicted_class: Name of predicted tumor class.
            confidence_score: Classification confidence score.
            is_calibrated: Calibration active flag.
            uncalibrated_confidence: raw model confidence score before platt scaling.
            probabilities: Probability score mapping dictionary.
            mask: NumPy mask output.
            expected_shape: source BGR image resolution.
            tumor_area_mm2: Area in square millimeters.
            brain_pixels: Estimated patient brain slice pixel count.

        Returns:
            A dictionary containing validated status, combined warnings, and detailed reports.
        """
        seg_report = self.seg_validator.validate_mask(
            mask=mask,
            expected_shape=expected_shape,
            brain_pixels=brain_pixels
        )

        consistency_report = self.consistency_checker.check_consistency(
            predicted_class=predicted_class,
            confidence_score=confidence_score,
            is_calibrated=is_calibrated,
            uncalibrated_confidence=uncalibrated_confidence,
            tumor_area_mm2=tumor_area_mm2,
            pixel_count=seg_report.pixel_count,
            probabilities=probabilities
        )

        # Merge warnings
        all_warnings = list(seg_report.warnings) + list(consistency_report.warnings)
        is_valid = seg_report.is_valid and consistency_report.is_consistent

        return {
            "is_valid": is_valid,
            "warnings": all_warnings,
            "segmentation_report": seg_report,
            "consistency_report": consistency_report
        }
