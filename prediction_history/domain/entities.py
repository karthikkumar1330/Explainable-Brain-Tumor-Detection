from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class HistorySearchCriteria:
    """Dataclass encapsulating user search/filter options."""
    patient_id: Optional[str] = None
    report_id: Optional[int] = None
    scan_date: Optional[str] = None


@dataclass(frozen=True)
class PredictionSummary:
    """Dataclass representing a normalized history prediction summary record."""
    report_id: int
    prediction_id: int
    patient_id: str
    patient_name: str
    scan_date: str
    predicted_class: str
    confidence_score: float
    tumor_area_mm2: float
    rule_based_severity: str
    created_at: str
