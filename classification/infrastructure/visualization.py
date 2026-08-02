import os
import cv2
import numpy as np
import logging
from typing import Optional


def overlay_heatmap(
    original_image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.6
) -> np.ndarray:
    """Overlays a 2D Grad-CAM heatmap onto an original image.

    Args:
        original_image: The original input image as a numpy array (H, W, C) or (H, W).
        heatmap: The 2D normalized heatmap of shape (H_feat, W_feat) with values [0.0, 1.0].
        alpha: Blending factor for the original image. The heatmap blend factor is (1 - alpha).

    Returns:
        The blended overlay image as a numpy array (BGR).
    """
    # 1. Ensure original image is 3-channel (BGR)
    if len(original_image.shape) == 2:
        img_bgr = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    elif original_image.shape[2] == 1:
        img_bgr = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = original_image.copy()

    # 2. Resize heatmap to match the original image size
    height, width = img_bgr.shape[0], img_bgr.shape[1]
    heatmap_resized = cv2.resize(heatmap, (width, height))

    # 3. Convert heatmap to 0-255 uint8
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    # 4. Apply Jet colormap (it outputs a BGR image)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # 5. Blend the original image and the colormapped heatmap
    # blended = alpha * original_image + (1 - alpha) * heatmap
    beta = 1.0 - alpha
    overlay = cv2.addWeighted(img_bgr, alpha, heatmap_color, beta, 0)

    return overlay


def save_explainability_outputs(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    output_dir: str,
    base_filename: str,
    alpha: float = 0.6,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Saves the raw heatmap and overlaid visualization to disk.

    Args:
        original_image: The original input image (H, W, C).
        heatmap: The 2D normalized heatmap.
        output_dir: Directory where outputs will be saved.
        base_filename: Base filename prefix for saved files.
        alpha: Blending factor for the overlay.
        logger: Optional logger.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        # 1. Overlay heatmap
        overlay = overlay_heatmap(original_image, heatmap, alpha=alpha)

        # 2. Generate path names
        heatmap_path = os.path.join(output_dir, f"{base_filename}_heatmap.png")
        overlay_path = os.path.join(output_dir, f"{base_filename}_overlay.png")

        # 3. For the heatmap alone, convert to colored representation to save
        heatmap_uint8 = np.uint8(255 * cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0])))
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # 4. Write to disk
        cv2.imwrite(heatmap_path, heatmap_color)
        cv2.imwrite(overlay_path, overlay)

        if logger:
            logger.info(f"Saved explainability heatmap to: {heatmap_path}")
            logger.info(f"Saved explainability overlay to: {overlay_path}")

    except Exception as e:
        if logger:
            logger.error(f"Failed to save explainability outputs: {e}")
        raise e
