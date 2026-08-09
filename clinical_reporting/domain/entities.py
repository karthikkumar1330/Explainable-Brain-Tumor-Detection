from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
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
