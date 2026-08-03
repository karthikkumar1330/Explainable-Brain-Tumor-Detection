from typing import List, Optional, Dict
from monitoring.domain.entities import ConsistencyReport


class ConfidenceConsistencyChecker:
    """Service to detect logic mismatches between classification outputs and segmentation contours."""

    def __init__(
        self,
        high_conf_threshold: float = 0.85,
        min_tumor_area_mm2: float = 5.0,
        false_neg_pixels: int = 100
    ) -> None:
        self.high_conf_threshold = high_conf_threshold
        self.min_tumor_area_mm2 = min_tumor_area_mm2
        self.false_neg_pixels = false_neg_pixels

    def check_consistency(
        self,
        predicted_class: str,
        confidence_score: float,
        is_calibrated: bool,
        uncalibrated_confidence: Optional[float],
        tumor_area_mm2: float,
        pixel_count: int,
        probabilities: Optional[Dict[str, float]] = None
    ) -> ConsistencyReport:
        """Cross-references prediction scores and morphology metrics to output quality warnings."""
        warnings: List[str] = []
        is_consistent = True

        # Convert label class name to uppercase for comparison safety
        p_class_upper = predicted_class.upper().strip()
        is_tumor_class = p_class_upper in ["GLIOMA", "MENINGIOMA", "PITUITARY"]

        # Case 1: High confidence of tumor but empty or negligible segmentation (False Positive Risk)
        if is_tumor_class and (confidence_score >= self.high_conf_threshold):
            if (pixel_count == 0) or (tumor_area_mm2 < self.min_tumor_area_mm2):
                msg = (
                    f"High confidence ({confidence_score:.1%}) for tumor type '{predicted_class}' "
                    f"but segmentation mask is empty or negligible ({tumor_area_mm2:.2f} mm²). "
                    f"Potential classification false alarm or localized detection failure."
                )
                warnings.append(msg)
                is_consistent = False

        # Case 2: No Tumor classified with high confidence but large segmentation outlined (False Negative Risk)
        if (p_class_upper == "NO TUMOR" or p_class_upper == "NO_TUMOR") and (confidence_score >= self.high_conf_threshold):
            if pixel_count >= self.false_neg_pixels:
                msg = (
                    f"Classification indicates 'No Tumor' with high confidence ({confidence_score:.1%}), "
                    f"but UNeXt outlines a significant lesion mass of {tumor_area_mm2:.2f} mm² "
                    f"({pixel_count} pixels). Risk of false negative."
                )
                warnings.append(msg)
                is_consistent = False

        # Case 3: Confidence calibration outliers
        if is_calibrated and (uncalibrated_confidence is not None):
            discrepancy = abs(uncalibrated_confidence - confidence_score)
            if discrepancy > 0.25:
                msg = (
                    f"Large calibration scaling: Uncalibrated confidence was {uncalibrated_confidence:.1%} "
                    f"but scaled down to {confidence_score:.1%} by platt calibration. "
                    f"Indicates high predictor uncertainty."
                )
                warnings.append(msg)
                # We flag this as a warning, but keep is_consistent = True if classification/segmentation align

        # Case 4: Narrow margin (Clinical ambiguity check)
        if probabilities:
            # Sort probabilities descending
            sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_probs) >= 2:
                primary_label, primary_val = sorted_probs[0]
                secondary_label, secondary_val = sorted_probs[1]
                margin = primary_val - secondary_val
                if margin < 0.10:
                    msg = (
                        f"Narrow diagnostic margin ({margin:.1%}) between primary classification "
                        f"'{primary_label}' ({primary_val:.1%}) and secondary '{secondary_label}' ({secondary_val:.1%}). "
                        f"High ambiguity, review recommended."
                    )
                    warnings.append(msg)
                    # Ambassador of clinical caution: not necessarily inconsistent, but warrants warning

        return ConsistencyReport(
            is_consistent=is_consistent,
            warnings=warnings,
            primary_class=predicted_class,
            confidence_score=confidence_score,
            calibrated=is_calibrated,
            tumor_area_mm2=tumor_area_mm2
        )
