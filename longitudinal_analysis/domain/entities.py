from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class LongitudinalComparison:
    """Dataclass holding detailed scan comparisons and progression metrics across two historical MRI points."""
    patient_id: str
    current_scan_date: str
    previous_scan_date: str

    # Classification stability
    current_class: str
    previous_class: str
    class_changed: bool

    # Confidence stability
    current_confidence: float
    previous_confidence: float
    confidence_delta: float

    # Lesion Size/Area
    current_area_mm2: float
    previous_area_mm2: float
    area_delta_mm2: float
    area_percentage_change: float

    # Brain Occupancy
    current_pct_brain: float
    previous_pct_brain: float
    pct_brain_delta: float

    # Shape Descriptors
    current_perimeter_mm: Optional[float]
    previous_perimeter_mm: Optional[float]
    perimeter_delta_mm: Optional[float]

    current_solidity: Optional[float]
    previous_solidity: Optional[float]
    solidity_delta: Optional[float]

    current_circularity: Optional[float]
    previous_circularity: Optional[float]
    circularity_delta: Optional[float]

    current_major_axis_mm: Optional[float]
    previous_major_axis_mm: Optional[float]
    major_axis_delta_mm: Optional[float]

    current_minor_axis_mm: Optional[float]
    previous_minor_axis_mm: Optional[float]
    minor_axis_delta_mm: Optional[float]

    # Bounding Box Sizes
    current_bbox_w_mm: Optional[float]
    previous_bbox_w_mm: Optional[float]
    bbox_w_delta_mm: Optional[float]

    current_bbox_h_mm: Optional[float]
    previous_bbox_h_mm: Optional[float]
    bbox_h_delta_mm: Optional[float]

    # Spatial alignment/intersection
    dice_coefficient: Optional[float]

    # Visual comparison scan path
    comparison_canvas_path: Optional[str]

    # Progression Status
    progression_status: str  # "Progressive Disease", "Stable Disease", "Regression", "Complete Response", "No Tumor"
    summary_text: str
