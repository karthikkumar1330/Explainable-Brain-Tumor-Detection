import numpy as np
import cv2
from typing import Optional, Dict, Any
from tumor_analysis.domain.entities import TumorAnalysisResult, SeverityLevel
from tumor_analysis.domain.interfaces import ITumorAnalyzer


class OpenCVTumorAnalyzer(ITumorAnalyzer):
    """OpenCV and NumPy implementation of the ITumorAnalyzer interface."""

    def __init__(
        self,
        low_thresh: float = 1.0,
        med_thresh: float = 5.0,
        high_thresh: float = 15.0,
    ) -> None:
        """Initializes the analyzer with threshold percentages for severity classification.

        Args:
            low_thresh: Percentage of brain parenchyma below which is LOW.
            med_thresh: Percentage of brain parenchyma below which is MEDIUM.
            high_thresh: Percentage of brain parenchyma below which is HIGH, above is CRITICAL.
        """
        self.low_thresh = low_thresh
        self.med_thresh = med_thresh
        self.high_thresh = high_thresh

    def analyze(
        self,
        mask: np.ndarray,
        original_image: Optional[np.ndarray] = None,
        pixel_spacing_mm: float = 1.0,
    ) -> TumorAnalysisResult:
        """Analyzes a predicted tumor mask to calculate metrics.

        Args:
            mask: Binary segmentation mask (values > 0 represent tumor).
            original_image: Optional original image to estimate brain region.
            pixel_spacing_mm: Spacing factor.

        Returns:
            TumorAnalysisResult object.
        """
        # Ensure mask is a 2D numpy array
        if len(mask.shape) == 3:
            mask_2d = mask[:, :, 0]
        else:
            mask_2d = mask

        # Binarize
        binary_mask = (mask_2d > 0).astype(np.uint8)

        # 1. Tumor pixel count
        tumor_pixel_count = int(np.sum(binary_mask))

        # 2. Tumor area in mm^2 (spacing * spacing * pixel_count)
        pixel_area_mm2 = pixel_spacing_mm * pixel_spacing_mm
        tumor_area_mm2 = float(tumor_pixel_count * pixel_area_mm2)

        # 3. Tumor percentage of total image
        total_pixels = mask_2d.size
        tumor_percentage_image = float((tumor_pixel_count / total_pixels) * 100.0)

        # 4. Estimated brain pixel count and tumor percentage of brain parenchyma
        estimated_brain_pixel_count = total_pixels
        if original_image is not None:
            # If color image, convert to grayscale
            if len(original_image.shape) == 3:
                gray_img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
            else:
                gray_img = original_image

            # Threshold to segment brain region (remove background black pixels)
            _, brain_mask = cv2.threshold(gray_img, 15, 255, cv2.THRESH_BINARY)
            estimated_brain_pixel_count = int(np.sum(brain_mask > 0))

            # Avoid division by zero
            if estimated_brain_pixel_count == 0:
                estimated_brain_pixel_count = total_pixels

        # Tumor percentage of brain
        tumor_percentage_brain = float((tumor_pixel_count / estimated_brain_pixel_count) * 100.0)

        # 5. Classify severity based on tumor percentage of brain parenchyma
        if tumor_pixel_count == 0:
            severity = SeverityLevel.NORMAL
        elif tumor_percentage_brain < self.low_thresh:
            severity = SeverityLevel.LOW
        elif tumor_percentage_brain < self.med_thresh:
            severity = SeverityLevel.MEDIUM
        elif tumor_percentage_brain < self.high_thresh:
            severity = SeverityLevel.HIGH
        else:
            severity = SeverityLevel.CRITICAL

        metadata: Dict[str, Any] = {
            "pixel_spacing_mm": pixel_spacing_mm,
            "pixel_area_mm2": pixel_area_mm2,
            "total_image_pixels": total_pixels,
        }

        return TumorAnalysisResult(
            pixel_count=tumor_pixel_count,
            tumor_area_mm2=tumor_area_mm2,
            tumor_percentage_image=tumor_percentage_image,
            tumor_percentage_brain=tumor_percentage_brain,
            estimated_brain_pixel_count=estimated_brain_pixel_count,
            severity_level=severity,
            metadata=metadata
        )
