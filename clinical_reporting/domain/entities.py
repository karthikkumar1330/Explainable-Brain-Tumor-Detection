from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
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



