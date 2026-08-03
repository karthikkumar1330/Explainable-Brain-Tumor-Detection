import cv2
import numpy as np
from typing import Optional
from tumor_analysis.domain.entities import TumorStats
from tumor_analysis.domain.interfaces import ITumorStatsEngine


class OpenCVTumorStatsEngine(ITumorStatsEngine):
    """OpenCV-based implementation of the ITumorStatsEngine interface."""

    def compute_stats(
        self, mask: np.ndarray, pixel_spacing_mm: float = 1.0
    ) -> Optional[TumorStats]:
        """Calculates detailed shape and region statistics from a binary mask.

        Args:
            mask: Binary mask of shape (H, W).
            pixel_spacing_mm: Real-world physical scale factor.

        Returns:
            A TumorStats object if a tumor contour is found, otherwise None.
        """
        # Ensure mask is a 2D binary numpy array
        if len(mask.shape) == 3:
            mask_2d = mask[:, :, 0]
        else:
            mask_2d = mask

        binary_mask = (mask_2d > 0).astype(np.uint8)

        if np.sum(binary_mask) == 0:
            return None

        # Find external contours
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours) == 0:
            return None

        # Choose the largest contour by area as the primary tumor mass
        main_contour = max(contours, key=cv2.contourArea)
        area_pixels = float(cv2.contourArea(main_contour))

        # If the contour has zero area, fallback to using total positive pixels
        if area_pixels == 0:
            area_pixels = float(np.sum(binary_mask))

        area_mm2 = area_pixels * pixel_spacing_mm * pixel_spacing_mm

        # 1. Perimeter
        perimeter_pixels = float(cv2.arcLength(main_contour, True))
        perimeter_mm = perimeter_pixels * pixel_spacing_mm

        # 2. Bounding Box
        bbox_x_px, bbox_y_px, bbox_w_px, bbox_h_px = cv2.boundingRect(main_contour)
        bbox_w_mm = bbox_w_px * pixel_spacing_mm
        bbox_h_mm = bbox_h_px * pixel_spacing_mm

        # 3. Shape Descriptors (Solidity & Circularity)
        # Solidity = Area / Convex Hull Area
        hull = cv2.convexHull(main_contour)
        hull_area = cv2.contourArea(hull)
        solidity = (
            float(area_pixels / hull_area)
            if hull_area > 0
            else 1.0
        )
        solidity = min(solidity, 1.0)

        # Circularity = 4 * pi * Area / Perimeter^2
        circularity = (
            float((4.0 * np.pi * area_pixels) / (perimeter_pixels ** 2))
            if perimeter_pixels > 0
            else 1.0
        )
        circularity = min(circularity, 1.0)

        # 4. Fit Equivalent Ellipse for Region Properties (Axes, Orientation, Eccentricity)
        # OpenCV fitEllipse requires at least 5 points in the contour.
        if len(main_contour) >= 5:
            try:
                (center_x, center_y), (major_axis, minor_axis), orientation = cv2.fitEllipse(main_contour)
                major_axis_mm = major_axis * pixel_spacing_mm
                minor_axis_mm = minor_axis * pixel_spacing_mm
                orientation_deg = float(orientation)

                # Eccentricity = sqrt(1 - (minor_axis / major_axis)^2)
                if major_axis > 0:
                    eccentricity = float(np.sqrt(1.0 - (minor_axis / major_axis) ** 2))
                else:
                    eccentricity = 0.0
            except Exception:
                major_axis_mm, minor_axis_mm, eccentricity, orientation_deg = self._compute_moments_fallback(
                    main_contour, pixel_spacing_mm
                )
        else:
            major_axis_mm, minor_axis_mm, eccentricity, orientation_deg = self._compute_moments_fallback(
                main_contour, pixel_spacing_mm
            )

        return TumorStats(
            area_pixels=int(area_pixels),
            area_mm2=area_mm2,
            perimeter_pixels=perimeter_pixels,
            perimeter_mm=perimeter_mm,
            bbox_x_px=bbox_x_px,
            bbox_y_px=bbox_y_px,
            bbox_w_px=bbox_w_px,
            bbox_h_px=bbox_h_px,
            bbox_w_mm=bbox_w_mm,
            bbox_h_mm=bbox_h_mm,
            major_axis_mm=major_axis_mm,
            minor_axis_mm=minor_axis_mm,
            eccentricity=eccentricity,
            orientation_deg=orientation_deg,
            solidity=solidity,
            circularity=circularity,
        )

    def _compute_moments_fallback(
        self, contour: np.ndarray, pixel_spacing_mm: float
    ) -> tuple[float, float, float, float]:
        """Calculates moment-based shape properties as a fallback for small contours."""
        moments = cv2.moments(contour)
        m00 = moments['m00']
        if m00 == 0:
            return 0.0, 0.0, 0.0, 0.0

        # Normalized central moments
        u20 = moments['mu20'] / m00
        u02 = moments['mu02'] / m00
        u11 = moments['mu11'] / m00

        # Compute equivalent ellipse characteristics from moments
        common_term = np.sqrt((u20 - u02) ** 2 + 4 * (u11 ** 2))
        major_axis = np.sqrt(8.0 * (u20 + u02 + common_term)) if (u20 + u02 + common_term) >= 0 else 0.0
        minor_axis = np.sqrt(8.0 * (u20 + u02 - common_term)) if (u20 + u02 - common_term) >= 0 else 0.0

        major_axis_mm = major_axis * pixel_spacing_mm
        minor_axis_mm = minor_axis * pixel_spacing_mm

        eccentricity = (
            float(np.sqrt(1.0 - (minor_axis / major_axis) ** 2))
            if major_axis > 0
            else 0.0
        )

        # Orientation angle in degrees
        orientation_deg = (
            float(0.5 * np.arctan2(2.0 * u11, u20 - u02) * (180.0 / np.pi))
            if (u20 - u02) != 0
            else 0.0
        )

        return major_axis_mm, minor_axis_mm, eccentricity, orientation_deg
