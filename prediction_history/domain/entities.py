from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class HistorySearchCriteria:
    """Dataclass encapsulating user search/filter options."""
    patient_id: Optional[str] = None
    report_id: Optional[int] = None
    scan_date: Optional[str] = None

    # G8.1 parameters
    q: Optional[str] = None
    patient_name: Optional[str] = None
    referring_doctor: Optional[str] = None
    classification: Optional[str] = None
    severity: Optional[str] = None
    min_confidence: Optional[float] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    report_status: Optional[str] = None

    # Sorting & Pagination
    sort_by: Optional[str] = None
    sort_order: Optional[str] = None
    page: Optional[int] = None
    page_size: Optional[int] = None

    # RBAC boundary restrictions
    restrict_to_patient_uuid: Optional[str] = None
    restrict_to_patient_name: Optional[str] = None
    restrict_to_doctor_id: Optional[int] = None


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
    referring_doctor: Optional[str] = None
    report_status: Optional[str] = None
