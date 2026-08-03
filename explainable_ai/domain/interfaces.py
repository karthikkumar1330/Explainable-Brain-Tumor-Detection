from abc import ABC, abstractmethod
from typing import Any
import numpy as np
from classification.domain.interfaces import IExplainabilityService


class IXAIEngine(IExplainabilityService):
    """Interface for Explainable AI 2.0 Engine extending standard explainability service."""

    @abstractmethod
    def generate_explanation(
        self, image_tensor: Any, target_class: int, method: str = "gradcam"
    ) -> np.ndarray:
        """Generates a normalized heatmap (0.0 to 1.0) using the specified XAI method.

        Args:
            image_tensor: Normalized image tensor of shape (C, H, W) or (1, C, H, W).
            target_class: The class index to generate explanation for.
            method: The algorithm ('gradcam', 'gradcam_plus_plus', 'eigencam').

        Returns:
            A 2D numpy array representing the normalized heatmap in range [0, 1].
        """
        pass
