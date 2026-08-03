import cv2
import numpy as np


def overlay_tumor_contour(
    overlay_image: np.ndarray,
    tumor_mask: np.ndarray,
    color: tuple = (0, 255, 0),  # Green BGR
    thickness: int = 2,
) -> np.ndarray:
    """Overlays the contour/boundary of the tumor mask onto an overlay image.

    Args:
        overlay_image: BGR image of shape (H, W, 3).
        tumor_mask: Binary mask of shape (H, W) or (H, W, 1).
        color: Contour color (BGR).
        thickness: Contour line thickness.

    Returns:
        BGR image with the contour boundary overlaid.
    """
    img_contour = overlay_image.copy()
    if tumor_mask is not None and np.sum(tumor_mask) > 0:
        # Normalize mask to binary uint8 {0, 255}
        mask_uint8 = (tumor_mask > 0).astype(np.uint8) * 255

        # Ensure mask is 2D
        if len(mask_uint8.shape) == 3:
            mask_uint8 = mask_uint8[:, :, 0]

        # Resize mask to match overlay shape if they differ
        h, w = img_contour.shape[:2]
        if mask_uint8.shape[:2] != (h, w):
            mask_uint8 = cv2.resize(
                mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST
            )

        # Extract external boundary contours
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(img_contour, contours, -1, color, thickness)

    return img_contour
