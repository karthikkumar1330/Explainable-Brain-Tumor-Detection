import os
import cv2
import numpy as np


def create_segmentation_comparison_image(
    original_image: np.ndarray,
    before_mask: np.ndarray,
    after_mask: np.ndarray,
    output_path: str,
) -> None:
    """Generates a side-by-side comparison image canvas.

    Left: Original brain MRI slice.
    Middle: Before post-processing (Initial UNeXt mask overlaid in Red).
    Right: After post-processing (Cleaned/Smoothed mask overlaid in Green).

    Args:
        original_image: NumPy array representing BGR image.
        before_mask: NumPy binary array representing original model output.
        after_mask: NumPy binary array representing post-processed mask.
        output_path: Destination path to save the comparison canvas.
    """
    h, w = original_image.shape[:2]

    # Ensure mask arrays are normalized uint8 and scaled to image size
    before_mask_resized = cv2.resize(
        before_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
    )
    after_mask_resized = cv2.resize(
        after_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
    )

    # Convert binary masks to True/False indices
    idx_before = before_mask_resized > 0
    idx_after = after_mask_resized > 0

    # 1. Overlay Before: Initial mask in Red (B=0, G=0, R=255)
    overlay_before = original_image.copy()
    overlay_before[idx_before] = [0, 0, 255]
    overlay_before = cv2.addWeighted(
        original_image, 0.7, overlay_before, 0.3, 0
    )

    # 2. Overlay After: Cleaned mask in Green (B=0, G=255, R=0)
    overlay_after = original_image.copy()
    overlay_after[idx_after] = [0, 255, 0]
    overlay_after = cv2.addWeighted(original_image, 0.7, overlay_after, 0.3, 0)

    # Create composite canvas: 3 columns + margins, and extra height for header
    canvas_w = w * 3 + 40
    canvas_h = h + 60
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas.fill(20)  # Dark background (20, 20, 20)

    # Draw image slices
    canvas[40 : 40 + h, 10 : 10 + w] = original_image
    canvas[40 : 40 + h, 20 + w : 20 + w * 2] = overlay_before
    canvas[40 : 40 + h, 30 + w * 2 : 30 + w * 3] = overlay_after

    # Draw text headers above each image
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 2
    color_text = (240, 240, 240)
    color_red = (100, 100, 255)
    color_green = (100, 255, 100)

    cv2.putText(
        canvas,
        "Original MRI Scan",
        (20, 26),
        font,
        font_scale,
        color_text,
        thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Raw UNeXt Output (Red)",
        (20 + w + 10, 26),
        font,
        font_scale,
        color_red,
        thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Post-Processed Mask (Green)",
        (30 + w * 2 + 10, 26),
        font,
        font_scale,
        color_green,
        thickness,
        cv2.LINE_AA,
    )

    # Save output to disk
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, canvas)
