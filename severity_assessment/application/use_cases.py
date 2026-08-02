import logging
from severity_assessment.domain.entities import SeverityAssessment
from severity_assessment.domain.interfaces import ISeverityClassifier


class AssessSeverityUseCase:
    """Use case to classify tumor severity based on clinical rules."""

    def __init__(
        self,
        classifier: ISeverityClassifier,
        logger: logging.Logger,
    ) -> None:
        """Initializes the severity use case.

        Args:
            classifier: Interface implementation for severity decision rules.
            logger: Logger instance.
        """
        self.classifier = classifier
        self.logger = logger

    def execute(
        self,
        tumor_type: str,
        tumor_area_mm2: float,
        tumor_percentage: float,
    ) -> SeverityAssessment:
        """Runs the severity assessment rules on the provided inputs.

        Args:
            tumor_type: Classification category of the tumor (e.g. Glioma).
            tumor_area_mm2: Measured physical tumor area.
            tumor_percentage: Percentage of brain parenchyma occupied.

        Returns:
            A SeverityAssessment containing the category, matched rules, and disclaimer.
        """
        self.logger.info(
            f"Evaluating severity for Type: {tumor_type}, "
            f"Area: {tumor_area_mm2:.2f} mm², Percentage: {tumor_percentage:.4f}%"
        )

        try:
            assessment = self.classifier.assess(
                tumor_type=tumor_type,
                tumor_area_mm2=tumor_area_mm2,
                tumor_percentage=tumor_percentage,
            )
            self.logger.info(f"Severity assessment complete. Assigned Category: {assessment.category.value}")
            return assessment
        except Exception as e:
            self.logger.error(f"Error during severity classification use case: {e}")
            raise e
