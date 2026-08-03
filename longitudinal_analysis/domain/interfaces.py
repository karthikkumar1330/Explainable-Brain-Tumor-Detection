from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
from longitudinal_analysis.domain.entities import LongitudinalComparison


class ILongitudinalAnalyzer(ABC):
    """Interface for analyzing and comparing two MRI scans over time for longitudinal progression."""

    @abstractmethod
    def compare(
        self,
        current_data: Dict[str, Any],
        previous_data: Dict[str, Any],
        current_mask: Optional[np.ndarray] = None,
        previous_mask: Optional[np.ndarray] = None,
        output_image_path: Optional[str] = None
    ) -> LongitudinalComparison:
        """Compares current and previous scan reports, calculating deltas and generating visual overlays.

        Args:
            current_data: Dict holding current patient scan JSON report data.
            previous_data: Dict holding previous patient scan JSON report data.
            current_mask: Optional binary mask of the current scan.
            previous_mask: Optional binary mask of the previous scan.
            output_image_path: Optional disk location to save visual comparison canvas.

        Returns:
            A LongitudinalComparison domain entity.
        """
        pass
