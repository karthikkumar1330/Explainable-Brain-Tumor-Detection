from dataclasses import dataclass
from enum import Enum
from typing import Dict


class BrainTumorClass(Enum):
    """Enumeration representing the four target brain tumor categories."""
    GLIOMA = 0
    MENINGIOMA = 1
    PITUITARY = 2
    NO_TUMOR = 3

    @classmethod
    def get_name_by_value(cls, value: int) -> str:
        """Returns the user-friendly class name corresponding to an integer value.

        Args:
            value: The integer value of the brain tumor class.

        Returns:
            The string name of the class (e.g., 'No Tumor').
        """
        for item in cls:
            if item.value == value:
                if item == cls.NO_TUMOR:
                    return "No Tumor"
                return item.name.capitalize()
        raise ValueError(f"Value '{value}' is not a valid BrainTumorClass enum value.")


@dataclass(frozen=True)
class PredictionResult:
    """Dataclass holding the prediction results for a single image."""
    label: int
    class_name: str
    confidence_score: float
    probabilities: Dict[str, float]
