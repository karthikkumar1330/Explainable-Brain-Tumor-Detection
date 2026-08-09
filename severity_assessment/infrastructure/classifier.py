from severity_assessment.domain.entities import SeverityAssessment, SeverityCategory
from severity_assessment.domain.interfaces import ISeverityClassifier
from severity_assessment.infrastructure.disclaimer import EDUCATIONAL_DISCLAIMER


class RuleBasedSeverityClassifier(ISeverityClassifier):
    """Clinical rule-based implementation of the ISeverityClassifier interface."""

    def assess(
        self,
        tumor_type: str,
        tumor_area_mm2: float,
        tumor_percentage: float,
        segmentation_failed: bool = False,
    ) -> SeverityAssessment:
        """Evaluates tumor risk profile using type, physical area, and brain tissue percentage.

        Args:
            tumor_type: Tumor class name (e.g. Glioma).
            tumor_area_mm2: Measured physical area in mm2.
            tumor_percentage: Brain occupancy percentage.
            segmentation_failed: True if the segmentation model failed during inference.

        Returns:
            A SeverityAssessment object.
        """
        t_type = tumor_type.strip().lower()

        # Rule 0: Segmentation failed
        if segmentation_failed:
            if t_type not in ["no tumor", "normal", "none"]:
                return SeverityAssessment(
                    category=SeverityCategory.MEDIUM,
                    rule_description=(
                        f"Classification predicts {tumor_type}. Quantitative tumor morphology is "
                        f"unavailable because the segmentation model failed during inference. Tumor area, "
                        f"occupancy, perimeter and morphology measurements should not be interpreted from this run."
                    ),
                    educational_disclaimer=EDUCATIONAL_DISCLAIMER
                )
            else:
                return SeverityAssessment(
                    category=SeverityCategory.LOW,
                    rule_description="No Tumor classification baseline. Segmentation model was bypassed or failed.",
                    educational_disclaimer=EDUCATIONAL_DISCLAIMER
                )

        # Rule 1: No Tumor or zero size is Low
        if t_type in ["no tumor", "normal", "none"] or tumor_area_mm2 <= 0.0 or tumor_percentage <= 0.0:
            category = SeverityCategory.LOW
            rule_desc = "No active tumor mass detected (Normal/No Tumor baseline)."
            return SeverityAssessment(
                category=category,
                rule_description=rule_desc,
                educational_disclaimer=EDUCATIONAL_DISCLAIMER,
            )

        # Rule 2: Critical Mass Effect or Large Scan Area Occupancy is High
        if tumor_percentage >= 8.0 or tumor_area_mm2 >= 1500.0:
            category = SeverityCategory.HIGH
            rule_desc = (
                f"High due to severe mass effect: tumor occupies {tumor_percentage:.2f}% of brain parenchyma "
                f"or physical size ({tumor_area_mm2:.2f} mm²) exceeds critical threshold of 1,500 mm²."
            )
            return SeverityAssessment(
                category=category,
                rule_description=rule_desc,
                educational_disclaimer=EDUCATIONAL_DISCLAIMER,
            )

        # Rule 3: Highly infiltrative Glioma with moderate size is High
        if t_type == "glioma" and (tumor_percentage >= 3.0 or tumor_area_mm2 >= 500.0):
            category = SeverityCategory.HIGH
            rule_desc = (
                f"High due to aggressive tumor class (Glioma) combined with significant size: "
                f"occupancy {tumor_percentage:.2f}% (>=3%) or area {tumor_area_mm2:.2f} mm² (>=500 mm²)."
            )
            return SeverityAssessment(
                category=category,
                rule_description=rule_desc,
                educational_disclaimer=EDUCATIONAL_DISCLAIMER,
            )

        # Rule 4: Slow-growing benign Meningioma/Pituitary with small sizes are Low
        if t_type in ["meningioma", "pituitary"] and tumor_percentage < 1.5 and tumor_area_mm2 < 200.0:
            category = SeverityCategory.LOW
            rule_desc = (
                f"Low risk profile: typically slow-growing tumor class ({tumor_type}) with minimal "
                f"parenchymal occupancy ({tumor_percentage:.2f}% < 1.5%) and small physical area ({tumor_area_mm2:.2f} mm² < 200 mm²)."
            )
            return SeverityAssessment(
                category=category,
                rule_description=rule_desc,
                educational_disclaimer=EDUCATIONAL_DISCLAIMER,
            )

        # Rule 5: Intermediate cases are classified as Medium
        category = SeverityCategory.MEDIUM
        rule_desc = (
            f"Medium risk profile: intermediate parameters. Type: {tumor_type}, "
            f"Area: {tumor_area_mm2:.2f} mm², Occupancy: {tumor_percentage:.2f}%."
        )
        return SeverityAssessment(
            category=category,
            rule_description=rule_desc,
            educational_disclaimer=EDUCATIONAL_DISCLAIMER,
        )
