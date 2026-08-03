import os
import cv2
import numpy as np
from typing import Dict, Any, Optional
from longitudinal_analysis.domain.entities import LongitudinalComparison
from longitudinal_analysis.domain.interfaces import ILongitudinalAnalyzer


class OpenCVLongitudinalAnalyzer(ILongitudinalAnalyzer):
    """OpenCV and NumPy implementation of ILongitudinalAnalyzer."""

    def compare(
        self,
        current_data: Dict[str, Any],
        previous_data: Dict[str, Any],
        current_mask: Optional[np.ndarray] = None,
        previous_mask: Optional[np.ndarray] = None,
        output_image_path: Optional[str] = None
    ) -> LongitudinalComparison:
        """Compares current and previous scans, calculates deltas, and saves comparison canvases."""
        
        # 1. Extract Classification & Confidence
        curr_class = current_data.get("classification", {}).get("predicted_class", "No Tumor")
        prev_class = previous_data.get("classification", {}).get("predicted_class", "No Tumor")
        class_changed = (curr_class != prev_class)

        curr_conf = current_data.get("classification", {}).get("confidence_score", 0.0)
        prev_conf = previous_data.get("classification", {}).get("confidence_score", 0.0)
        confidence_delta = curr_conf - prev_conf

        # 2. Extract Area & Parenchymal Percentage
        curr_area = current_data.get("segmentation", {}).get("tumor_area_mm2", 0.0)
        prev_area = previous_data.get("segmentation", {}).get("tumor_area_mm2", 0.0)
        area_delta_mm2 = curr_area - prev_area
        
        if prev_area > 0:
            area_percentage_change = (area_delta_mm2 / prev_area) * 100.0
        else:
            area_percentage_change = 100.0 if area_delta_mm2 > 0 else 0.0

        curr_pct_brain = current_data.get("segmentation", {}).get("tumor_percentage_brain", 0.0)
        prev_pct_brain = previous_data.get("segmentation", {}).get("tumor_percentage_brain", 0.0)
        pct_brain_delta = curr_pct_brain - prev_pct_brain

        # 3. Extract Detailed Shape Descriptors
        curr_stats = current_data.get("segmentation", {}).get("shape_statistics", {}) or {}
        prev_stats = previous_data.get("segmentation", {}).get("shape_statistics", {}) or {}

        curr_perimeter = curr_stats.get("perimeter_mm")
        prev_perimeter = prev_stats.get("perimeter_mm")
        perimeter_delta_mm = (curr_perimeter - prev_perimeter) if (curr_perimeter is not None and prev_perimeter is not None) else None

        curr_solidity = curr_stats.get("solidity")
        prev_solidity = prev_stats.get("solidity")
        solidity_delta = (curr_solidity - prev_solidity) if (curr_solidity is not None and prev_solidity is not None) else None

        curr_circularity = curr_stats.get("circularity")
        prev_circularity = prev_stats.get("circularity")
        circularity_delta = (curr_circularity - prev_circularity) if (curr_circularity is not None and prev_circularity is not None) else None

        curr_major = curr_stats.get("major_axis_mm")
        prev_major = prev_stats.get("major_axis_mm")
        major_axis_delta_mm = (curr_major - prev_major) if (curr_major is not None and prev_major is not None) else None

        curr_minor = curr_stats.get("minor_axis_mm")
        prev_minor = prev_stats.get("minor_axis_mm")
        minor_axis_delta_mm = (curr_minor - prev_minor) if (curr_minor is not None and prev_minor is not None) else None

        # Bounding Box sizes
        curr_bbox_w = curr_stats.get("bbox_w_mm")
        prev_bbox_w = prev_stats.get("bbox_w_mm")
        bbox_w_delta_mm = (curr_bbox_w - prev_bbox_w) if (curr_bbox_w is not None and prev_bbox_w is not None) else None

        curr_bbox_h = curr_stats.get("bbox_h_mm")
        prev_bbox_h = prev_stats.get("bbox_h_mm")
        bbox_h_delta_mm = (curr_bbox_h - prev_bbox_h) if (curr_bbox_h is not None and prev_bbox_h is not None) else None

        # 4. Resolve masks for spatial overlap and dice
        if current_mask is None:
            m_path = current_data.get("files", {}).get("segmentation_mask")
            if m_path and os.path.exists(m_path):
                current_mask = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
        if previous_mask is None:
            m_path = previous_data.get("files", {}).get("segmentation_mask")
            if m_path and os.path.exists(m_path):
                previous_mask = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)

        dice_coefficient = None
        if current_mask is not None and previous_mask is not None:
            if previous_mask.shape != current_mask.shape:
                previous_mask = cv2.resize(previous_mask, (current_mask.shape[1], current_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            bin_curr = (current_mask > 0)
            bin_prev = (previous_mask > 0)
            intersection = np.logical_and(bin_curr, bin_prev).sum()
            total = bin_curr.sum() + bin_prev.sum()
            dice_coefficient = float(2.0 * intersection / total) if total > 0 else 1.0

        # 5. Generate progression assessment status and summary text
        if curr_class == "No Tumor" and prev_class == "No Tumor":
            progression_status = "No Tumor Detected"
            summary_text = "No tumor mass detected in either the current slice or the previous scan history. Patient is clear."
        elif prev_class == "No Tumor" and curr_class != "No Tumor":
            progression_status = "Progressive Disease"
            summary_text = (
                f"New active lesion detected. The classification has progressed from 'No Tumor' to "
                f"'{curr_class}' (Confidence: {curr_conf:.2%}, Area: {curr_area:.1f} mm²). Immediate clinical evaluation recommended."
            )
        elif prev_class != "No Tumor" and curr_class == "No Tumor":
            progression_status = "Complete Response"
            summary_text = (
                f"Outstanding therapeutic regression. The tumor classification has regressed from "
                f"'{prev_class}' ({prev_area:.1f} mm²) to 'No Tumor' in the current scan, representing a 100.0% reduction."
            )
        else:
            # Both tumor
            summary_prefix = ""
            if class_changed:
                summary_prefix = f"Tumor class shift observed from '{prev_class}' to '{curr_class}'. "
            
            if area_percentage_change > 15.0:
                progression_status = "Progressive Disease"
                summary_text = (
                    f"{summary_prefix}Significant tumor enlargement of {area_percentage_change:+.1f}% "
                    f"(from {prev_area:.1f} mm² to {curr_area:.1f} mm²), indicating disease progression."
                )
            elif area_percentage_change < -15.0:
                progression_status = "Partial Response"
                summary_text = (
                    f"{summary_prefix}Significant tumor shrinkage of {area_percentage_change:.1f}% "
                    f"(from {prev_area:.1f} mm² to {curr_area:.1f} mm²), indicating partial response/regression."
                )
            else:
                progression_status = "Stable Disease"
                summary_text = (
                    f"{summary_prefix}Stable disease. Minimal area variation of {area_percentage_change:+.1f}% "
                    f"(from {prev_area:.1f} mm² to {curr_area:.1f} mm²)."
                )

            if dice_coefficient is not None:
                summary_text += f" The spatial Dice overlap coefficient is {dice_coefficient:.4f}."

        # 6. Build the Visual Comparison Canvas
        comparison_canvas_path = None
        if output_image_path and current_mask is not None and previous_mask is not None:
            try:
                curr_img_path = current_data.get("files", {}).get("original_image")
                prev_img_path = previous_data.get("files", {}).get("original_image")
                
                # Load images or default to black canvases
                img_curr = cv2.imread(curr_img_path) if (curr_img_path and os.path.exists(curr_img_path)) else None
                img_prev = cv2.imread(prev_img_path) if (prev_img_path and os.path.exists(prev_img_path)) else None
                
                h, w = current_mask.shape[:2]
                if img_curr is None:
                    img_curr = np.zeros((h, w, 3), dtype=np.uint8)
                if img_prev is None:
                    img_prev = cv2.resize(img_curr, (img_curr.shape[1], img_curr.shape[0]))
                
                # Ensure sizes match
                if img_prev.shape != img_curr.shape:
                    img_prev = cv2.resize(img_prev, (img_curr.shape[1], img_curr.shape[0]))
                if previous_mask.shape != current_mask.shape:
                    previous_mask = cv2.resize(previous_mask, (current_mask.shape[1], current_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                # Create Panel 1: Previous image with Red mask overlay
                panel_prev = img_prev.copy()
                # Find contours for drawing boundaries
                contours_prev, _ = cv2.findContours((previous_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(panel_prev, contours_prev, -1, (0, 0, 255), 2)
                cv2.putText(panel_prev, "Previous Scan (Red Mask)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Create Panel 2: Current image with Green mask overlay
                panel_curr = img_curr.copy()
                contours_curr, _ = cv2.findContours((current_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(panel_curr, contours_curr, -1, (0, 255, 0), 2)
                cv2.putText(panel_curr, "Current Scan (Green Mask)", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Create Panel 3: Spatial Overlap (Red previous, Green current, Yellow overlap)
                panel_overlap = img_curr.copy()
                
                overlay = panel_overlap.copy()
                overlay[previous_mask > 0] = [0, 0, 200]  # Red
                overlay[current_mask > 0] = [0, 200, 0]  # Green
                overlay[np.logical_and(previous_mask > 0, current_mask > 0)] = [0, 200, 200]  # Yellow overlap
                
                cv2.addWeighted(overlay, 0.5, panel_overlap, 0.5, 0, panel_overlap)
                cv2.drawContours(panel_overlap, contours_prev, -1, (0, 0, 255), 1)
                cv2.drawContours(panel_overlap, contours_curr, -1, (0, 255, 0), 1)
                
                dice_lbl = f"Dice: {dice_coefficient:.3f}" if dice_coefficient is not None else "Dice: N/A"
                cv2.putText(panel_overlap, f"Overlap comparison ({dice_lbl})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 2)

                # Combine side-by-side
                canvas = np.hstack([panel_prev, panel_curr, panel_overlap])
                cv2.imwrite(output_image_path, canvas)
                comparison_canvas_path = output_image_path
            except Exception as e:
                # Fallback to no image if drawing errors out
                pass

        return LongitudinalComparison(
            patient_id=current_data.get("patient", {}).get("patient_id", ""),
            current_scan_date=current_data.get("patient", {}).get("scan_date", ""),
            previous_scan_date=previous_data.get("patient", {}).get("scan_date", ""),
            current_class=curr_class,
            previous_class=prev_class,
            class_changed=class_changed,
            current_confidence=curr_conf,
            previous_confidence=prev_conf,
            confidence_delta=confidence_delta,
            current_area_mm2=curr_area,
            previous_area_mm2=prev_area,
            area_delta_mm2=area_delta_mm2,
            area_percentage_change=area_percentage_change,
            current_pct_brain=curr_pct_brain,
            previous_pct_brain=prev_pct_brain,
            pct_brain_delta=pct_brain_delta,
            current_perimeter_mm=curr_perimeter,
            previous_perimeter_mm=prev_perimeter,
            perimeter_delta_mm=perimeter_delta_mm,
            current_solidity=curr_solidity,
            previous_solidity=prev_solidity,
            solidity_delta=solidity_delta,
            current_circularity=curr_circularity,
            previous_circularity=prev_circularity,
            circularity_delta=circularity_delta,
            current_major_axis_mm=curr_major,
            previous_major_axis_mm=prev_major,
            major_axis_delta_mm=major_axis_delta_mm,
            current_minor_axis_mm=curr_minor,
            previous_minor_axis_mm=prev_minor,
            minor_axis_delta_mm=minor_axis_delta_mm,
            current_bbox_w_mm=curr_bbox_w,
            previous_bbox_w_mm=prev_bbox_w,
            bbox_w_delta_mm=bbox_w_delta_mm,
            current_bbox_h_mm=curr_bbox_h,
            previous_bbox_h_mm=prev_bbox_h,
            bbox_h_delta_mm=bbox_h_delta_mm,
            dice_coefficient=dice_coefficient,
            comparison_canvas_path=comparison_canvas_path,
            progression_status=progression_status,
            summary_text=summary_text,
        )
