from typing import Dict, Any, List, Optional
from monitoring.infrastructure.segmentation_validator import SegmentationValidator
from monitoring.infrastructure.consistency_checker import ConfidenceConsistencyChecker
from monitoring.infrastructure.explainability_validator import ExplainabilityValidator


class CentralWarningEngine:
    """Central engine collecting and prioritizing warnings across all pipeline checkers."""

    def __init__(
        self,
        seg_validator: SegmentationValidator,
        consistency_checker: ConfidenceConsistencyChecker,
        explain_validator: ExplainabilityValidator
    ) -> None:
        self.seg_validator = seg_validator
        self.consistency_checker = consistency_checker
        self.explain_validator = explain_validator

    def collect_warnings(
        self,
        input_errors: List[str],
        predicted_class: str,
        confidence_score: float,
        is_calibrated: bool,
        uncalibrated_confidence: Optional[float],
        probabilities: Optional[Dict[str, float]],
        mask: Any,
        expected_shape: tuple,
        tumor_area_mm2: float,
        heatmap: Any,
        overlap_percentage: Optional[float],
        pixel_spacing_mm: float = 1.0,
        brain_pixels: Optional[int] = None
    ) -> Dict[str, Any]:
        """Runs segmentation, consistency, and explainability checkers and aggregates all warnings.

        Args:
            input_errors: Pre-prediction file and image validation errors.
            predicted_class: Diagnostic class predicted by EfficientNet.
            confidence_score: Calibrated or uncalibrated confidence probability.
            is_calibrated: Boolean indicating calibration scaling active.
            uncalibrated_confidence: raw classification probability.
            probabilities: Probability score map dictionary.
            mask: UNeXt segmentation binary numpy mask.
            expected_shape: original MRI scan dimensions.
            tumor_area_mm2: quantitative area value.
            heatmap: Explainability Grad-CAM activation array.
            overlap_percentage: XAI spatial overlap value.
            pixel_spacing_mm: conversion scale millimeter metric.
            brain_pixels: total estimated parenchymal pixels count.

        Returns:
            A unified quality assessment dictionary.
        """
        all_warnings = list(input_errors)
        
        # 1. Run segmentation mask validation checks
        seg_report = self.seg_validator.validate_mask(
            mask=mask,
            expected_shape=expected_shape,
            brain_pixels=brain_pixels
        )
        all_warnings.extend(seg_report.warnings)

        # 2. Run classification consistency checks
        consistency_report = self.consistency_checker.check_consistency(
            predicted_class=predicted_class,
            confidence_score=confidence_score,
            is_calibrated=is_calibrated,
            uncalibrated_confidence=uncalibrated_confidence,
            tumor_area_mm2=tumor_area_mm2,
            pixel_count=seg_report.pixel_count,
            probabilities=probabilities
        )
        all_warnings.extend(consistency_report.warnings)

        # 3. Run explainability overlay validation checks
        exp_valid, exp_warnings = self.explain_validator.validate_explanation(
            heatmap=heatmap,
            tumor_mask=mask,
            overlap_percentage=overlap_percentage,
            pixel_spacing_mm=pixel_spacing_mm
        )
        all_warnings.extend(exp_warnings)

        is_safe = seg_report.is_valid and consistency_report.is_consistent and exp_valid and (len(input_errors) == 0)

        return {
            "is_safe": is_safe,
            "warnings": all_warnings,
            "segmentation_report": seg_report.to_dict(),
            "consistency_report": consistency_report.to_dict(),
            "explainability_valid": exp_valid
        }
