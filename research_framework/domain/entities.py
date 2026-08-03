from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ModelProfile:
    """Dataclass holding descriptive metadata for a registered model in the research framework."""
    name: str
    description: str
    architecture: str
    framework: str
    is_production: bool
    checkpoint_path: Optional[str] = None


@dataclass(frozen=True)
class ModelPrediction:
    """Represents a single model's diagnostic outcome on a scan."""
    model_name: str
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    runtime_sec: float


@dataclass(frozen=True)
class AgreementMetrics:
    """Quantitative consensus metrics comparing research models against production baseline."""
    cosine_similarity: float
    jensen_shannon_divergence: float
    level: str  # e.g., "HIGH CONSENSUS", "MODERATE CONSENSUS", "DISCREPANCY WARNING"
    message: str


@dataclass(frozen=True)
class EnsembleResult:
    """Consolidated outcome of the ensemble voting process across all registered models."""
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    individual_predictions: List[ModelPrediction]
    agreement_metrics: Optional[AgreementMetrics] = None
