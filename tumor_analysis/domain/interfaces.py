from abc import ABC, abstractmethod
import numpy as np
from typing import Optional
from tumor_analysis.domain.entities import TumorAnalysisResult, TumorStats


class ITumorAnalyzer(ABC):
    """Interface for analyzing segmentation masks to compute medical metrics."""

    @abstractmethod
    def analyze(
        self,
        mask: np.ndarray,
        original_image: Optional[np.ndarray] = None,
        pixel_spacing_mm: float = 1.0,
    ) -> TumorAnalysisResult:
        """Analyzes a predicted tumor mask to calculate pixel count, area, and percentage.

        Args:
            mask: Binary or binarized segmentation mask (H, W).
            original_image: Optional original MRI image (H, W) or (H, W, C) for brain area calculation.
            pixel_spacing_mm: Physical size of one pixel side in millimeters.

        Returns:
            A TumorAnalysisResult object.
        """
        pass


class ITumorStatsEngine(ABC):
    """Interface for extracting detailed shape and morphometric stats from a tumor mask."""

    @abstractmethod
    def compute_stats(
        self, mask: np.ndarray, pixel_spacing_mm: float = 1.0
    ) -> Optional[TumorStats]:
        """Calculates quantitative shape and location metrics from a binary mask.

        Args:
            mask: Binary mask of shape (H, W) with active pixels.
            pixel_spacing_mm: Physical size of one pixel side in millimeters.

        Returns:
            A TumorStats object if a tumor is present, else None.
        """
        pass

