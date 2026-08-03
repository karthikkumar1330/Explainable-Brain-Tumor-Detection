from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np


@dataclass(frozen=True)
class ExplanationResult:
    """Dataclass holding detailed Explainable AI output, attention maps, and clinical statistics."""

    method: str
    target_class: int
    class_name: str
    heatmap: np.ndarray  # 2D heatmap in [0.0, 1.0]
    explanation_text: str  # Generated clinical explanation of model attention
    overlap_percentage: float  # Spatial overlap ratio between high attention and tumor mask
    quadrant_attention: Dict[str, float]  # Heatmap intensity distribution across 4 scan quadrants
    metadata: Dict[str, Any]
