from dataclasses import dataclass
from typing import Optional, List
from tumor_analysis.domain.entities import TumorAnalysisResult
from severity_assessment.domain.entities import SeverityAssessment
from classification.domain.entities import PredictionResult
from longitudinal_analysis.domain.entities import LongitudinalComparison


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



