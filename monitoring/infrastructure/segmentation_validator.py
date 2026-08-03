import cv2
import numpy as np
from typing import List, Optional
from monitoring.domain.entities import SegmentationValidationReport


class SegmentationValidator:
    """Service to execute morphological and spatial validations on UNeXt output masks."""

    def __init__(self, tiny_threshold_px: int = 15, huge_brain_ratio: float = 0.45) -> None:
        self.tiny_threshold_px = tiny_threshold_px
        self.huge_brain_ratio = huge_brain_ratio

    def validate_mask(
        self,
        mask: np.ndarray,
        expected_shape: tuple,  # (height, width)
        brain_pixels: Optional[int] = None
    ) -> SegmentationValidationReport:
        """Runs checks on binary mask data integrity, values, shape components, and dimensions."""
        warnings: List[str] = []
        is_valid = True

        # 1. Corrupt/Invalid formats
        if mask is None:
            return SegmentationValidationReport(
                is_valid=False,
                warnings=["Segmentation mask is None / invalid reference."],
                pixel_count=0,
                num_components=0,
                min_val=0,
                max_val=0,
                unique_vals=[],
                height=0,
                width=0
            )

        if np.isnan(mask).any() or np.isinf(mask).any():
            warnings.append("Segmentation output contains corrupt numerical values (NaN/Inf).")
            is_valid = False

        # 2. Dimensions check
        h, w = mask.shape[:2]
        exp_h, exp_w = expected_shape[:2]
        if (h != exp_h) or (w != exp_w):
            warnings.append(f"Dimension mismatch: Expected {exp_w}x{exp_h}, got {w}x{h} pixels.")
            is_valid = False

        # 3. Invalid Pixel values check
        unique_vals = [int(v) for v in np.unique(mask)]
        # Binary masks should only contain values in {0, 1} or {0, 255}
        invalid_pixels = [v for v in unique_vals if v not in [0, 1, 255]]
        if invalid_pixels:
            warnings.append(f"Non-binary pixel values detected: {invalid_pixels}. Mask must contain only 0 and 1/255.")
            is_valid = False

        # Compute thresholded binary representation for structural analysis
        binary_mask = (mask > 0).astype(np.uint8)
        pixel_count = int(np.sum(binary_mask))

        # 4. Connected components analysis
        num_labels = 0
        if pixel_count > 0:
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)
            # Subtract 1 to exclude background label (0)
            num_components = num_labels - 1
        else:
            num_components = 0

        # Empty mask warning (non-critical, might represent actual absence of tumor)
        if pixel_count == 0:
            warnings.append("Segmentation mask is completely empty (no tumor pixels detected).")

        # Tiny mask check
        elif pixel_count < self.tiny_threshold_px:
            warnings.append(
                f"Extremely small segmentation region detected ({pixel_count} pixels). "
                f"This may represent machine noise or contour artifact."
            )

        # Huge mask check
        if brain_pixels and brain_pixels > 0:
            ratio = pixel_count / brain_pixels
            if ratio > self.huge_brain_ratio:
                warnings.append(
                    f"Abnormally large tumor occupancy: {ratio:.1%} of brain volume "
                    f"(Threshold: >{self.huge_brain_ratio:.0%}). Potential segmentation leakage."
                )
        else:
            # Fallback based on total image canvas
            ratio = pixel_count / mask.size
            if ratio > 0.35:
                warnings.append(
                    f"Abnormally large tumor size: {ratio:.1%} of total image area. "
                    f"Potential segmentation leakage."
                )

        # Disconnected regions check
        if num_components >= 3:
            warnings.append(
                f"Multi-focal segmentation warning: {num_components} disconnected tumor regions detected. "
                f"This might represent multi-focal disease or noise artifacts."
            )

        min_val = int(np.min(mask)) if mask.size > 0 else 0
        max_val = int(np.max(mask)) if mask.size > 0 else 0

        return SegmentationValidationReport(
            is_valid=is_valid,
            warnings=warnings,
            pixel_count=pixel_count,
            num_components=num_components,
            min_val=min_val,
            max_val=max_val,
            unique_vals=unique_vals,
            height=h,
            width=w
        )
