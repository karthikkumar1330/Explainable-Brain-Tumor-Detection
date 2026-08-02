from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any


class SeverityLevel(Enum):
    """Enumeration representing clinical severity classes for brain tumors."""
    NORMAL = "Normal/No Tumor"
    LOW = "Low Severity"
    MEDIUM = "Medium Severity"
    HIGH = "High Severity"
    CRITICAL = "Critical Severity"


@dataclass(frozen=True)
class TumorAnalysisResult:
    """Dataclass holding quantitative analysis results for a brain tumor mask."""
    pixel_count: int
    tumor_area_mm2: float
    tumor_percentage_image: float
    tumor_percentage_brain: float
    estimated_brain_pixel_count: int
    severity_level: SeverityLevel
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class ClinicalReportData:
    """Dataclass holding formatted clinical reports and recommendations."""
    patient_id: str
    tumor_class: str
    analysis: TumorAnalysisResult
    clinical_notes: str
    recommendations: str
