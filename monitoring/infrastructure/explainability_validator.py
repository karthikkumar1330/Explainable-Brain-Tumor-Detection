import numpy as np
from typing import List, Optional, Tuple


class ExplainabilityValidator:
    """Service to validate the integrity and anatomical alignment of Grad-CAM attention heatmaps."""

    def __init__(self, min_overlap_ratio: float = 0.10, max_distance_mm: float = 50.0) -> None:
        self.min_overlap_ratio = min_overlap_ratio
        self.max_distance_mm = max_distance_mm

    def validate_explanation(
        self,
        heatmap: np.ndarray,
        tumor_mask: Optional[np.ndarray],
        overlap_percentage: Optional[float],
        pixel_spacing_mm: float = 1.0
    ) -> Tuple[bool, List[str]]:
        """Verifies explainability outputs and checks for alignment discrepancies against tumor boundaries.

        Args:
            heatmap: 2D NumPy array representing Grad-CAM activation weights in [0, 1].
            tumor_mask: 2D binary tumor mask NumPy array.
            overlap_percentage: Heatmap overlap percentage with the tumor mask.
            pixel_spacing_mm: MRI voxel size conversion parameter.

        Returns:
            A tuple of (is_valid, warnings).
        """
        warnings: List[str] = []
        is_valid = True

        # 1. Integrity check
        if heatmap is None:
            return False, ["Grad-CAM heatmap output is None."]

        if np.isnan(heatmap).any() or np.isinf(heatmap).any():
            warnings.append("Grad-CAM heatmap contains invalid numerical outputs (NaN/Inf).")
            is_valid = False

        has_tumor = tumor_mask is not None and np.sum(tumor_mask > 0) > 0

        # 2. Overlap bounds check
        if has_tumor and overlap_percentage is not None:
            if overlap_percentage < self.min_overlap_ratio:
                msg = (
                    f"Low explainability target overlap: Grad-CAM attention focus covers only "
                    f"{overlap_percentage:.1%} of the segmented tumor (Threshold: >{self.min_overlap_ratio:.0%})."
                )
                warnings.append(msg)

        # 3. Spatial discrepancy distance check
        if has_tumor:
            try:
                # Calculate weighted center of mass for Grad-CAM activation heatmap
                h, w = heatmap.shape[:2]
                y_indices, x_indices = np.indices((h, w))
                total_activation = heatmap.sum()
                if total_activation > 0:
                    cy_cam = (y_indices * heatmap).sum() / total_activation
                    cx_cam = (x_indices * heatmap).sum() / total_activation
                    
                    # Calculate center of mass for segmented tumor contour
                    y_mask, x_mask = np.where(tumor_mask > 0)
                    cy_mask = y_mask.mean()
                    cx_mask = x_mask.mean()
                    
                    # Calculate spatial Euclidean distance
                    dist_px = np.sqrt((cx_cam - cx_mask)**2 + (cy_cam - cy_mask)**2)
                    dist_mm = dist_px * pixel_spacing_mm
                    
                    if dist_mm > self.max_distance_mm:
                        msg = (
                            f"Attention focus discrepancy: Grad-CAM attention centroid is "
                            f"{dist_mm:.1f} mm away from the segmented tumor centroid. "
                            f"The neural network may be focusing on peripheral features instead of the primary mass."
                        )
                        warnings.append(msg)
            except Exception as e:
                warnings.append(f"Failed to calculate explainability centroid distance check: {e}")

        return is_valid, warnings
