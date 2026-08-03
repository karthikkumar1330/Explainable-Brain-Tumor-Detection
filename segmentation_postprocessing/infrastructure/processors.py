import cv2
import numpy as np
from typing import Dict, Any, Tuple
from segmentation_postprocessing.domain.interfaces import ISegmentationPostProcessor


class MedicalImagePostProcessor(ISegmentationPostProcessor):
    """Modular post-processing pipeline for brain tumor segmentation masks.

    Implements noise removal, hole filling, connected component selection,
    boundary smoothing, and shape-based quality scoring.
    """

    def __init__(
        self,
        noise_removal_kernel_size: int = 3,
        boundary_smoothing_kernel_size: int = 5,
        fill_holes: bool = True,
        keep_largest_component: bool = True,
    ) -> None:
        """Initializes the post-processor.

        Args:
            noise_removal_kernel_size: Size of kernel for morphological opening (0 to disable).
            boundary_smoothing_kernel_size: Size of Gaussian kernel for smoothing (0 to disable).
            fill_holes: Whether to perform contour hole filling.
            keep_largest_component: Whether to retain only the largest connected component.
        """
        self.noise_removal_kernel_size = noise_removal_kernel_size
        self.boundary_smoothing_kernel_size = boundary_smoothing_kernel_size
        self.fill_holes = fill_holes
        self.keep_largest_component = keep_largest_component

    def remove_noise(self, mask: np.ndarray) -> np.ndarray:
        """Removes small isolated noise pixels using morphological opening."""
        if mask.sum() == 0:
            return mask
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.noise_removal_kernel_size, self.noise_removal_kernel_size),
        )
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    def fill_internal_holes(self, mask: np.ndarray) -> np.ndarray:
        """Fills internal holes in the binary mask using contour drawing."""
        if mask.sum() == 0:
            return mask
        filled_mask = mask.copy()
        # Find contours including internal ones (CCOMP retrieval mode)
        contours, hierarchy = cv2.findContours(
            filled_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) > 0 and hierarchy is not None:
            for idx in range(len(contours)):
                # Draw contour filled (-1) with label 1 (white)
                cv2.drawContours(filled_mask, contours, idx, 1, -1)
        return filled_mask

    def smooth_boundaries(self, mask: np.ndarray) -> np.ndarray:
        """Smooths binary mask boundaries using Gaussian blur and re-thresholding."""
        if mask.sum() == 0:
            return mask
        ksize = self.boundary_smoothing_kernel_size
        if ksize % 2 == 0:
            ksize += 1
        # Apply Gaussian blur on float mask representation
        blurred = cv2.GaussianBlur(mask.astype(np.float32), (ksize, ksize), 0)
        # Re-threshold at 0.5 to restore a clean binary mask
        smoothed = (blurred >= 0.5).astype(np.uint8)
        return smoothed

    def get_largest_connected_component(self, mask: np.ndarray) -> np.ndarray:
        """Retains only the largest connected component in the mask, filtering out satellite noise."""
        if mask.sum() == 0:
            return mask
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        if num_labels <= 2:
            # Only background (0) and 1 foreground component (1) exist
            return mask

        # Stats array column CC_STAT_AREA holds the pixel counts.
        # Skip label 0 (background)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

        largest_cc_mask = (labels == largest_label).astype(np.uint8)
        return largest_cc_mask

    def calculate_quality_score(
        self, mask: np.ndarray, probability_map: np.ndarray
    ) -> Tuple[float, str]:
        """Calculates a surrogate segmentation quality score in [0.0, 1.0].

        Uses a combination of solidity (shape integrity), circularity (smoothness),
        and mean classification probability (model confidence).

        Returns:
            A tuple of (quality_score, quality_category)
        """
        if mask.sum() == 0:
            # Mask is empty. If probability map is also low overall, this is a clean prediction.
            mean_prob_all = float(np.mean(probability_map))
            if mean_prob_all < 0.1:
                return 1.0, "HIGH"
            elif mean_prob_all < 0.3:
                return 0.8, "HIGH"
            else:
                return 0.5, "LOW"

        mask_uint8 = (mask * 255).astype(np.uint8)

        # 1. Solidity
        solidity = 1.0
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) > 0:
            main_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(main_contour)
            if area > 0:
                hull = cv2.convexHull(main_contour)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    solidity = area / hull_area

        # 2. Circularity
        circularity = 1.0
        if len(contours) > 0:
            main_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(main_contour)
            perimeter = cv2.arcLength(main_contour, True)
            if area > 0 and perimeter > 0:
                circularity = (4.0 * np.pi * area) / (perimeter ** 2)
                circularity = min(circularity, 1.0)

        # 3. Mean confidence inside mask
        mask_indices = mask > 0
        mean_confidence = (
            float(np.mean(probability_map[mask_indices]))
            if mask_indices.sum() > 0
            else 0.0
        )

        # Multi-criteria weighted score: Solidity (30%), Circularity (20%), Confidence (50%)
        quality_score = (
            (0.3 * solidity) + (0.2 * circularity) + (0.5 * mean_confidence)
        )
        quality_score = float(np.clip(quality_score, 0.0, 1.0))

        if quality_score >= 0.85:
            category = "HIGH"
        elif quality_score >= 0.65:
            category = "MEDIUM"
        else:
            category = "LOW"

        return quality_score, category

    def process(
        self, binary_mask: np.ndarray, probability_map: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Executes the full post-processing pipeline on a binary mask."""
        mask = binary_mask.copy().astype(np.uint8)

        # Normalize binary mask to {0, 1}
        if mask.max() > 1:
            mask = (mask > 0).astype(np.uint8)

        steps = []

        # 1. Noise removal
        if self.noise_removal_kernel_size > 0:
            mask = self.remove_noise(mask)
            steps.append("Noise Removal")

        # 2. Hole filling
        if self.fill_holes:
            mask = self.fill_internal_holes(mask)
            steps.append("Hole Filling")

        # 3. Largest connected component selection
        if self.keep_largest_component:
            mask = self.get_largest_connected_component(mask)
            steps.append("Largest Component Selection")

        # 4. Boundary smoothing
        if self.boundary_smoothing_kernel_size > 0:
            mask = self.smooth_boundaries(mask)
            steps.append("Boundary Smoothing")

        # 5. Quality scoring
        quality_score, quality_category = self.calculate_quality_score(
            mask, probability_map
        )

        meta = {
            "post_processed": True,
            "steps_applied": steps,
            "quality_score": quality_score,
            "quality_category": quality_category,
            "noise_removal_kernel": self.noise_removal_kernel_size,
            "boundary_smoothing_kernel": self.boundary_smoothing_kernel_size,
        }

        return mask, meta
