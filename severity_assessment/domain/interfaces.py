from abc import ABC, abstractmethod
from severity_assessment.domain.entities import SeverityAssessment


class ISeverityClassifier(ABC):
    """Interface for classifying tumor severity based on clinical rules."""

    @abstractmethod
    def assess(
        self,
        tumor_type: str,
        tumor_area_mm2: float,
        tumor_percentage: float,
    ) -> SeverityAssessment:
        """Classifies brain tumor severity based on type, area, and parenchymal percentage.

        Args:
            tumor_type: The classified type of tumor (e.g. Glioma).
            tumor_area_mm2: Physical area of tumor in square millimeters.
            tumor_percentage: Percentage of brain region occupied by the tumor.

        Returns:
            A SeverityAssessment entity.
        """
        pass
