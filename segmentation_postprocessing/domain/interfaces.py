from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np


class ISegmentationPostProcessor(ABC):
    """Interface for segmentation mask post-processing operations under Clean Architecture."""

    @abstractmethod
    def process(
        self, binary_mask: np.ndarray, probability_map: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Processes the binary mask and computes quality metrics.

        Args:
            binary_mask: Binary numpy array of shape (H, W) with values in {0, 1} or {0, 255}.
            probability_map: Raw probability map of shape (H, W) with values in [0.0, 1.0].

        Returns:
            A tuple containing:
                - The post-processed binary mask of shape (H, W) with values in {0, 1}.
                - A dictionary containing quality scoring details and execution metadata.
        """
        pass
