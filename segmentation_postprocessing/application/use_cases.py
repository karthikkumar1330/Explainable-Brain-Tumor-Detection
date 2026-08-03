from typing import Dict, Any, Tuple
import numpy as np
from segmentation_postprocessing.domain.interfaces import ISegmentationPostProcessor


class PostProcessSegmentationUseCase:
    """Use case to manage and execute the segmentation post-processing pipeline."""

    def __init__(self, post_processor: ISegmentationPostProcessor) -> None:
        """Initializes the post-processing use case.

        Args:
            post_processor: An implementation of ISegmentationPostProcessor.
        """
        self.post_processor = post_processor

    def execute(
        self, binary_mask: np.ndarray, probability_map: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Runs post-processing on the provided segmentation maps.

        Args:
            binary_mask: The initial thresholded binary mask.
            probability_map: The model raw Sigmoid probability predictions.

        Returns:
            A tuple containing:
                - The processed binary mask.
                - Metadata detailing the quality metrics and post-processing steps.
        """
        return self.post_processor.process(binary_mask, probability_map)
