import math
import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union
from enum import Enum
from tumor_analysis.domain.entities import TumorAnalysisResult
from severity_assessment.domain.entities import SeverityAssessment
from classification.domain.entities import PredictionResult
from longitudinal_analysis.domain.entities import LongitudinalComparison
from clinical_insight.domain.entities import ClinicalInsight


@dataclass(frozen=True)
class PatientInfo:
    """Dataclass holding demographics and clinical reference data for a patient scan."""
    patient_id: str
    name: str
    age: int
    gender: str
    scan_date: str
    ref_physician: str


@dataclass(frozen=True)
class ProcessingSummary:
    """Dataclass holding computation platform, timing, and model execution context."""
    device: str
    execution_time_sec: float
    classification_model_path: str
    segmentation_model_path: str
    classification_latency_sec: float
    segmentation_latency_sec: float
    explainability_latency_sec: float


@dataclass(frozen=True)
class ClinicalReport:
    """Aggregated entity representing the complete clinical brain MRI findings."""
    patient_info: PatientInfo
    processing_summary: ProcessingSummary
    classification: PredictionResult
    segmentation_metrics: Optional[TumorAnalysisResult]
    severity_assessment: Optional[SeverityAssessment]
    original_image_path: str
    heatmap_image_path: Optional[str]
    overlay_image_path: Optional[str]
    segmentation_mask_path: Optional[str]
    comparison_image_path: Optional[str] = None
    xai_method: Optional[str] = None
    xai_explanation_text: Optional[str] = None
    xai_overlap_percentage: Optional[float] = None
    longitudinal_comparison: Optional[LongitudinalComparison] = None
    quality_warnings: Optional[List[str]] = None
    clinical_insight: Optional[ClinicalInsight] = None
    report_number: Optional[str] = None
    version: Optional[int] = None
    status: Optional[str] = None
    verification_token: Optional[str] = None
    integrity_hash: Optional[str] = None


@dataclass(frozen=True)
class PatientInformation:
    patient_id: str
    name: str
    age: int
    gender: str
    referring_physician: str
    scan_id: Optional[str] = None
    scan_date: Optional[str] = None
    original_image_path: Optional[str] = None


@dataclass(frozen=True)
class ClassificationResult:
    predicted_class: str
    confidence_score: float
    classification_model: str
    model_version: Optional[str] = None


@dataclass(frozen=True)
class SegmentationResult:
    segmentation_status: str  # e.g., "Available", "Failed"
    tumor_area_mm2: float
    tumor_percentage_brain: float
    perimeter_mm: Optional[float] = None
    quality_score: Optional[float] = None
    quality_category: Optional[str] = None
    segmentation_mask_path: Optional[str] = None


@dataclass(frozen=True)
class TumorStatistics:
    shape_statistics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExplainabilityResult:
    xai_method: Optional[str]
    xai_status: str  # e.g., "Available", "Unavailable"
    explanation_text: Optional[str] = None
    overlap_percentage: Optional[float] = None
    heatmap_image_path: Optional[str] = None
    overlay_image_path: Optional[str] = None


@dataclass(frozen=True)
class ClinicalInsights:
    summary_narrative: str
    key_findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ProcessingMetrics:
    classification_latency_sec: Optional[float] = None
    segmentation_latency_sec: Optional[float] = None
    explainability_latency_sec: Optional[float] = None
    total_execution_time_sec: Optional[float] = None
    device: str = "CPU"


@dataclass(frozen=True)
class SystemStatus:
    classification: str  # "Available" or "Failed"
    segmentation: str  # "Available" or "Failed"
    explainability: str  # "Available" or "Unavailable"
    report_generation: str  # "Successful" or "Failed"


@dataclass(frozen=True)
class Disclaimer:
    text: str


@dataclass(frozen=True)
class Metadata:
    report_id: str
    report_title: str
    report_type: str
    brand_name: str
    timestamp: str
    version: Optional[int] = None
    status: Optional[str] = None
    integrity_hash: Optional[str] = None
    verification_token: Optional[str] = None



@dataclass(frozen=True)
class ReportData:
    patient_info: PatientInformation
    classification_result: ClassificationResult
    segmentation_result: SegmentationResult
    tumor_statistics: TumorStatistics
    explainability_result: ExplainabilityResult
    clinical_insights: ClinicalInsights
    processing_metrics: ProcessingMetrics
    system_status: SystemStatus
    disclaimer: Disclaimer
    metadata: Metadata


class ReportStatus(str, Enum):
    DRAFT = "DRAFT"
    GENERATED = "GENERATED"
    REVIEWED = "REVIEWED"
    FINAL = "FINAL"
    ARCHIVED = "ARCHIVED"


VALID_TRANSITIONS = {
    ReportStatus.DRAFT: {ReportStatus.GENERATED},
    ReportStatus.GENERATED: {ReportStatus.REVIEWED},
    ReportStatus.REVIEWED: {ReportStatus.FINAL},
    ReportStatus.FINAL: {ReportStatus.ARCHIVED}
}


def can_transition(current: ReportStatus, target: ReportStatus) -> bool:
    return target in VALID_TRANSITIONS.get(current, set())


@dataclass
class Report:
    report_id: int
    report_number: str
    patient_id: str
    created_by: Optional[str]
    report_type: str
    current_version: int
    status: ReportStatus
    created_at: str
    updated_at: str
    finalized_at: Optional[str]
    archived_at: Optional[str]
    pdf_path: Optional[str]
    json_path: Optional[str]
    checksum: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "report_number": self.report_number,
            "patient_id": self.patient_id,
            "created_by": self.created_by,
            "report_type": self.report_type,
            "current_version": self.current_version,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "finalized_at": self.finalized_at,
            "archived_at": self.archived_at,
            "pdf_path": self.pdf_path,
            "json_path": self.json_path,
            "checksum": self.checksum
        }


@dataclass
class ReportVersion:
    version_id: int
    report_id: int
    version_number: int
    created_at: str
    created_by: Optional[str]
    reason: Optional[str]
    pdf_path: str
    json_path: str
    checksum: str
    status: ReportStatus
    prediction_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "report_id": self.report_id,
            "version_number": self.version_number,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "reason": self.reason,
            "pdf_path": self.pdf_path,
            "json_path": self.json_path,
            "checksum": self.checksum,
            "status": self.status.value,
            "prediction_id": self.prediction_id
        }


class VerificationState(str, Enum):
    VALID = "VALID"
    INVALID = "INVALID"
    TAMPERED = "TAMPERED"
    SUPERSEDED = "SUPERSEDED"
    ARCHIVED = "ARCHIVED"


@dataclass
class ReportVerification:
    report_id: int
    version: int
    verification_token: str
    integrity_hash: str
    algorithm: str
    status: str
    created_at: str
    finalized_at: Optional[str]
    verification_state: VerificationState


def canonicalize_report_data(data: Dict[str, Any]) -> str:
    """Recursively sorts dictionary keys and returns a deterministic JSON string.
    Strips out unstable fields like paths, execution times, and temporary timestamps.
    """
    import copy
    import json
    cleaned = copy.deepcopy(data)

    def strip_unstable(d: Any) -> Any:
        if isinstance(d, dict):
            # Remove keys related to file paths, execution latency, and temporary timestamps
            keys_to_remove = [
                'files', 'pdf_path', 'json_path', 'checksum',
                'created_at', 'updated_at', 'finalized_at', 'archived_at',
                'timestamp', 'execution_time_sec', 'classification_latency_sec',
                'segmentation_latency_sec', 'explainability_latency_sec', 'total_execution_time_sec',
                'latency_sec', 'device'
            ]
            for key in keys_to_remove:
                d.pop(key, None)
            return {k: strip_unstable(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [strip_unstable(item) for item in d]
        else:
            return d

    cleaned = strip_unstable(cleaned)
    return json.dumps(cleaned, sort_keys=True, separators=(',', ':'))


def generate_integrity_hash(data: Dict[str, Any]) -> str:
    """Generates the SHA-256 integrity hash for canonicalized report data."""
    import hashlib
    canonical_str = canonicalize_report_data(data)
    return hashlib.sha256(canonical_str.encode('utf-8')).hexdigest()


class PatientMismatchException(ValueError):
    """Raised when trying to compare reports belonging to different patients."""
    pass


@dataclass(frozen=True)
class ComparisonMetric:
    name: str
    previous_value: Any
    current_value: Any
    absolute_difference: Optional[float] = None
    percentage_difference: Optional[float] = None
    direction: str = "UNCHANGED"  # "INCREASED", "DECREASED", "UNCHANGED", "CHANGED"
    interpretation_category: str = "UNCHANGED"  # "OBSERVED_INCREASE", "OBSERVED_DECREASE", "UNCHANGED"


@dataclass(frozen=True)
class ReportComparison:
    comparison_id: str
    patient_id: str
    previous_report_id: int
    current_report_id: int
    previous_version: int
    current_version: int
    created_at: str
    created_by: str
    metrics: List[ComparisonMetric]
    summary_status: str  # "STABLE", "MEASUREMENTS_INCREASED", "MEASUREMENTS_DECREASED", "CLASSIFICATION_CHANGED", "MULTIPLE_CHANGES"
    summary_text: str
    disclaimer: str


@dataclass(frozen=True)
class FollowUpComparisonMetric:
    name: str
    previous_value: Any
    current_value: Any
    absolute_change: Optional[float] = None
    percentage_change: Optional[float] = None
    status: str = "UNAVAILABLE"  # "INCREASED", "DECREASED", "STABLE", "UNAVAILABLE"


@dataclass(frozen=True)
class FollowUpComparison:
    comparison_id: str
    patient_id: str
    previous_report_id: int
    current_report_id: int
    previous_version: int
    current_version: int
    previous_scan_date: Optional[str]
    current_scan_date: Optional[str]
    created_at: str
    created_by: str
    metrics: List[FollowUpComparisonMetric]
    summary_status: str
    summary_text: str
    disclaimer: str


@dataclass(frozen=True)
class LongitudinalTimelineMetric:
    metric_name: str
    current_value: Any
    previous_value: Any
    absolute_change: Optional[float] = None
    percentage_change: Optional[float] = None
    trend: str = "UNAVAILABLE"
    unit: str = ""
    available: bool = False

    def __post_init__(self) -> None:
        def is_nan_or_inf(val: Any) -> bool:
            if val is None:
                return False
            if isinstance(val, float):
                return math.isnan(val) or math.isinf(val)
            if isinstance(val, str):
                return val.lower().strip() in ("nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity")
            return False

        def is_valid_numeric(val: Any) -> bool:
            if val is None or is_nan_or_inf(val):
                return False
            try:
                f = float(val)
                return not (math.isnan(f) or math.isinf(f))
            except (ValueError, TypeError):
                return False

        abs_change = None
        pct_change = None
        trend = "UNAVAILABLE"
        available = False

        if is_valid_numeric(self.current_value) and is_valid_numeric(self.previous_value):
            c = float(self.current_value)
            p = float(self.previous_value)
            abs_change = c - p

            if p == 0.0:
                if c == 0.0:
                    pct_change = 0.0
                    trend = "STABLE"
                else:
                    pct_change = None
                    trend = "INCREASED" if c > 0.0 else "DECREASED"
            else:
                try:
                    pct_change = (abs_change / abs(p)) * 100.0
                    if math.isnan(pct_change) or math.isinf(pct_change):
                        pct_change = None
                        trend = "UNAVAILABLE"
                    else:
                        if abs_change > 1e-7:
                            trend = "INCREASED"
                        elif abs_change < -1e-7:
                            trend = "DECREASED"
                        else:
                            trend = "STABLE"
                except ZeroDivisionError:
                    pct_change = None
                    trend = "UNAVAILABLE"
            available = True
        else:
            if (self.current_value is not None and self.previous_value is not None and
                not is_nan_or_inf(self.current_value) and not is_nan_or_inf(self.previous_value)):
                c_str = str(self.current_value)
                p_str = str(self.previous_value)
                trend = "UNCHANGED" if c_str == p_str else "CHANGED"
                available = True
            else:
                trend = "UNAVAILABLE"
                available = False

        # Set attributes for frozen dataclass
        object.__setattr__(self, 'absolute_change', abs_change)
        object.__setattr__(self, 'percentage_change', pct_change)
        object.__setattr__(self, 'trend', trend)
        object.__setattr__(self, 'available', available)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "previous_value": self.previous_value,
            "absolute_change": self.absolute_change,
            "percentage_change": self.percentage_change,
            "trend": self.trend,
            "unit": self.unit,
            "available": self.available
        }


@dataclass(frozen=True)
class LongitudinalTimelineEvent:
    report_id: Optional[int]
    scan_id: Optional[int]
    patient_id: str
    scan_date: Optional[str]
    classification: Optional[str]
    confidence: Optional[float]
    tumor_area: Optional[float]
    tumor_percentage: Optional[float]
    severity: Optional[str]
    severity_score: Optional[float] = None
    report_status: Optional[str] = None

    def get_parsed_date(self) -> datetime.datetime:
        if isinstance(self.scan_date, datetime.datetime):
            return self.scan_date
        if isinstance(self.scan_date, datetime.date):
            return datetime.datetime.combine(self.scan_date, datetime.time.min)
        if not self.scan_date or not isinstance(self.scan_date, str):
            return datetime.datetime.min

        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
            try:
                return datetime.datetime.strptime(self.scan_date.strip(), fmt)
            except ValueError:
                continue
        return datetime.datetime.min

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "scan_id": self.scan_id,
            "patient_id": self.patient_id,
            "scan_date": self.scan_date,
            "classification": self.classification,
            "confidence": self.confidence,
            "tumor_area": self.tumor_area,
            "tumor_percentage": self.tumor_percentage,
            "severity": self.severity,
            "severity_score": self.severity_score,
            "report_status": self.report_status
        }


def sanitize_json_value(val: Any) -> Any:
    """Recursively replaces non-finite float values (NaN, Infinity, -Infinity) with None."""
    if isinstance(val, dict):
        return {k: sanitize_json_value(v) for k, v in val.items()}
    if isinstance(val, list):
        return [sanitize_json_value(v) for v in val]
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return None
    return val


@dataclass(frozen=True)
class LongitudinalPatientTimeline:
    patient_id: str
    patient_name: Optional[str] = None
    events: List[LongitudinalTimelineEvent] = field(default_factory=list)
    total_events: int = 0
    first_scan_date: Optional[str] = None
    latest_scan_date: Optional[str] = None
    timeline_status: str = "UNKNOWN"
    trend_summary: str = ""
    metrics: Dict[str, LongitudinalTimelineMetric] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Sort events chronologically by parsed scan date
        sorted_events = sorted(self.events, key=lambda e: (e.get_parsed_date(), e.report_id or 0))
        object.__setattr__(self, 'events', sorted_events)
        object.__setattr__(self, 'total_events', len(sorted_events))

        first_date = None
        latest_date = None
        if sorted_events:
            first_date = sorted_events[0].scan_date
            latest_date = sorted_events[-1].scan_date

        object.__setattr__(self, 'first_scan_date', first_date)
        object.__setattr__(self, 'latest_scan_date', latest_date)

        status = self.timeline_status
        summary = self.trend_summary

        if status == "UNKNOWN":
            if not sorted_events:
                status = "EMPTY"
                summary = "No scan history available."
            elif len(sorted_events) == 1:
                status = "STABLE"
                summary = "Baseline scan established."
            else:
                first = sorted_events[0]
                latest = sorted_events[-1]

                # Compare tumor area change
                if first.tumor_area is not None and latest.tumor_area is not None:
                    def is_valid_num(v: Any) -> bool:
                        if v is None:
                            return False
                        try:
                            f = float(v)
                            return not (math.isnan(f) or math.isinf(f))
                        except (ValueError, TypeError):
                            return False

                    if is_valid_num(first.tumor_area) and is_valid_num(latest.tumor_area):
                        p_val = float(first.tumor_area)
                        c_val = float(latest.tumor_area)
                        if p_val == 0.0:
                            if c_val > 0.0:
                                status = "PROGRESSION"
                                summary = f"Tumor emerged. Latest area: {c_val} mm2."
                            else:
                                status = "STABLE"
                                summary = "No tumor detected."
                        else:
                            pct = ((c_val - p_val) / p_val) * 100.0
                            if pct >= 10.0:
                                status = "PROGRESSION"
                                summary = f"Tumor area increased by {pct:.1f}%."
                            elif pct <= -10.0:
                                status = "REGRESSION"
                                summary = f"Tumor area decreased by {abs(pct):.1f}%."
                            else:
                                status = "STABLE"
                                summary = f"Tumor area remains stable (change: {pct:+.1f}%)."
                    else:
                        status = "STABLE"
                        summary = "Comparison unavailable due to missing/invalid tumor area."
                else:
                    if first.classification != latest.classification:
                        status = "CHANGED"
                        summary = f"Classification changed from {first.classification} to {latest.classification}."
                    else:
                        status = "STABLE"
                        summary = "Patient condition stable."

            object.__setattr__(self, 'timeline_status', status)
            object.__setattr__(self, 'trend_summary', summary)

        # Derive first vs latest metrics
        derived_metrics = {}
        if len(sorted_events) >= 2:
            first = sorted_events[0]
            latest = sorted_events[-1]
            derived_metrics["tumor_area"] = LongitudinalTimelineMetric(
                metric_name="Tumor Area",
                current_value=latest.tumor_area,
                previous_value=first.tumor_area,
                unit="mm2"
            )
            derived_metrics["confidence"] = LongitudinalTimelineMetric(
                metric_name="Classification Confidence",
                current_value=latest.confidence,
                previous_value=first.confidence
            )
            derived_metrics["severity"] = LongitudinalTimelineMetric(
                metric_name="Severity Score",
                current_value=latest.severity_score,
                previous_value=first.severity_score
            )
            derived_metrics["classification"] = LongitudinalTimelineMetric(
                metric_name="Classification",
                current_value=latest.classification,
                previous_value=first.classification
            )
        else:
            derived_metrics["tumor_area"] = LongitudinalTimelineMetric(
                metric_name="Tumor Area", current_value=None, previous_value=None, unit="mm2"
            )
            derived_metrics["confidence"] = LongitudinalTimelineMetric(
                metric_name="Classification Confidence", current_value=None, previous_value=None
            )
            derived_metrics["severity"] = LongitudinalTimelineMetric(
                metric_name="Severity Score", current_value=None, previous_value=None
            )
            derived_metrics["classification"] = LongitudinalTimelineMetric(
                metric_name="Classification", current_value=None, previous_value=None
            )
        object.__setattr__(self, 'metrics', derived_metrics)

    @property
    def first_event(self) -> Optional[LongitudinalTimelineEvent]:
        return self.events[0] if self.events else None

    @property
    def latest_event(self) -> Optional[LongitudinalTimelineEvent]:
        return self.events[-1] if self.events else None

    @property
    def intermediate_events(self) -> List[LongitudinalTimelineEvent]:
        if len(self.events) <= 2:
            return []
        return self.events[1:-1]

    def to_dict(self) -> Dict[str, Any]:
        res = {
            "patient_id": self.patient_id,
            "patient_name": self.patient_name,
            "events": [event.to_dict() for event in self.events],
            "total_events": self.total_events,
            "first_scan_date": self.first_scan_date,
            "latest_scan_date": self.latest_scan_date,
            "timeline_status": self.timeline_status,
            "trend_summary": self.trend_summary,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()}
        }
        return sanitize_json_value(res)
