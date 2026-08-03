from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional


class SeverityLevel(Enum):
    """Enumeration representing clinical severity classes for brain tumors."""
    NORMAL = "Normal/No Tumor"
    LOW = "Low Severity"
    MEDIUM = "Medium Severity"
    HIGH = "High Severity"
    CRITICAL = "Critical Severity"


@dataclass(frozen=True)
class TumorStats:
    """Dataclass holding detailed shape, bounding box, and orientation measurements of the tumor."""
    area_pixels: int
    area_mm2: float
    perimeter_pixels: float
    perimeter_mm: float
    bbox_x_px: int
    bbox_y_px: int
    bbox_w_px: int
    bbox_h_px: int
    bbox_w_mm: float
    bbox_h_mm: float
    major_axis_mm: float
    minor_axis_mm: float
    eccentricity: float
    orientation_deg: float
    solidity: float
    circularity: float


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

    # Post-processing quality metrics
    quality_score: Optional[float] = None
    quality_category: Optional[str] = None  # e.g., 'HIGH', 'MEDIUM', 'LOW'
    post_processing_applied: bool = False
    post_processing_metadata: Optional[Dict[str, Any]] = None

    # Tumor Statistics Engine measurements
    stats: Optional[TumorStats] = None




@dataclass(frozen=True)
class ClinicalReportData:
    """Dataclass holding formatted clinical reports and recommendations."""
    patient_id: str
    tumor_class: str
    analysis: TumorAnalysisResult
    clinical_notes: str
    recommendations: str
