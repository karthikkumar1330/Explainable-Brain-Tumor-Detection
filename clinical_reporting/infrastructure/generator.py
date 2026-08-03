import os
import json
from typing import Tuple
from clinical_reporting.domain.entities import ClinicalReport
from clinical_reporting.domain.interfaces import IClinicalReportGenerator
from clinical_reporting.infrastructure.pdf_generator import ReportLabPDFGenerator


class MarkdownJSONReportGenerator(IClinicalReportGenerator):
    """Generates structured Markdown and JSON reports for clinical brain MRI scans."""

    def generate(self, report: ClinicalReport, output_dir: str) -> Tuple[str, str, str]:
        """Saves clinical report in Markdown, JSON, and PDF formats.

        Args:
            report: The aggregated clinical report.
            output_dir: Destination folder.

        Returns:
            A tuple of (saved_markdown_path, saved_json_path, saved_pdf_path).
        """
        os.makedirs(output_dir, exist_ok=True)
        base_name = f"{report.patient_info.patient_id}_clinical_report"

        # 1. Generate JSON report
        json_path = os.path.join(output_dir, f"{base_name}.json")
        json_data = self._build_json_payload(report)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=4)

        # 2. Generate Markdown report
        md_path = os.path.join(output_dir, f"{base_name}.md")
        md_content = self._build_markdown_content(report)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        # 3. Generate PDF report
        pdf_path = os.path.join(output_dir, f"{base_name}.pdf")
        pdf_gen = ReportLabPDFGenerator()
        pdf_gen.generate_pdf(report, pdf_path)

        return md_path, json_path, pdf_path

    def _build_json_payload(self, report: ClinicalReport) -> dict:
        """Constructs a clean serializable dictionary for downstream ML/systems."""
        payload = {
            "patient": {
                "patient_id": report.patient_info.patient_id,
                "name": report.patient_info.name,
                "age": report.patient_info.age,
                "gender": report.patient_info.gender,
                "scan_date": report.patient_info.scan_date,
                "ref_physician": report.patient_info.ref_physician,
            },
            "processing": {
                "device": report.processing_summary.device,
                "total_execution_time_sec": report.processing_summary.execution_time_sec,
                "classification_model": report.processing_summary.classification_model_path,
                "segmentation_model": report.processing_summary.segmentation_model_path,
                "latency_sec": {
                    "classification": report.processing_summary.classification_latency_sec,
                    "segmentation": report.processing_summary.segmentation_latency_sec,
                    "explainability": report.processing_summary.explainability_latency_sec,
                }
            },
            "classification": {
                "predicted_class": report.classification.class_name,
                "confidence_score": report.classification.confidence_score,
                "probabilities": report.classification.probabilities,
                "is_calibrated": getattr(report.classification, "is_calibrated", False),
                "uncalibrated_confidence_score": getattr(report.classification, "uncalibrated_confidence_score", None),
                "uncalibrated_probabilities": getattr(report.classification, "uncalibrated_probabilities", None),
                "calibration_method": getattr(report.classification, "calibration_method", None),
                "calibration_parameters": getattr(report.classification, "calibration_parameters", None),
            },
            "files": {
                "original_image": report.original_image_path,
                "heatmap_image": report.heatmap_image_path,
                "overlay_image": report.overlay_image_path,
                "segmentation_mask": report.segmentation_mask_path,
                "comparison_image": report.comparison_image_path,
            }
        }

        # Add optional segmentation and severity assessment info if present
        if report.segmentation_metrics is not None:
            payload["segmentation"] = {
                "pixel_count": report.segmentation_metrics.pixel_count,
                "tumor_area_mm2": report.segmentation_metrics.tumor_area_mm2,
                "tumor_percentage_image": report.segmentation_metrics.tumor_percentage_image,
                "tumor_percentage_brain": report.segmentation_metrics.tumor_percentage_brain,
                "estimated_brain_pixel_count": report.segmentation_metrics.estimated_brain_pixel_count,
                "post_processing_applied": getattr(report.segmentation_metrics, "post_processing_applied", False),
                "quality_score": getattr(report.segmentation_metrics, "quality_score", None),
                "quality_category": getattr(report.segmentation_metrics, "quality_category", None),
                "post_processing_metadata": getattr(report.segmentation_metrics, "post_processing_metadata", None),
            }

            if getattr(report.segmentation_metrics, "stats", None) is not None:
                s = report.segmentation_metrics.stats
                payload["segmentation"]["shape_statistics"] = {
                    "area_pixels": s.area_pixels,
                    "area_mm2": s.area_mm2,
                    "perimeter_pixels": s.perimeter_pixels,
                    "perimeter_mm": s.perimeter_mm,
                    "bbox_x_px": s.bbox_x_px,
                    "bbox_y_px": s.bbox_y_px,
                    "bbox_w_px": s.bbox_w_px,
                    "bbox_h_px": s.bbox_h_px,
                    "bbox_w_mm": s.bbox_w_mm,
                    "bbox_h_mm": s.bbox_h_mm,
                    "major_axis_mm": s.major_axis_mm,
                    "minor_axis_mm": s.minor_axis_mm,
                    "eccentricity": s.eccentricity,
                    "orientation_deg": s.orientation_deg,
                    "solidity": s.solidity,
                    "circularity": s.circularity,
                }


        if report.severity_assessment is not None:
            payload["severity"] = {
                "category": report.severity_assessment.category.value,
                "rule_description": report.severity_assessment.rule_description,
                "educational_disclaimer": report.severity_assessment.educational_disclaimer,
            }

        # Add optional XAI 2.0 info if present
        if getattr(report, "xai_method", None) is not None:
            payload["explainability"] = {
                "method": report.xai_method,
                "explanation_text": report.xai_explanation_text,
                "overlap_percentage": report.xai_overlap_percentage,
            }

        # Add optional longitudinal comparison info if present
        if getattr(report, "longitudinal_comparison", None) is not None:
            lc = report.longitudinal_comparison
            payload["longitudinal_comparison"] = {
                "previous_scan_date": lc.previous_scan_date,
                "current_scan_date": lc.current_scan_date,
                "previous_class": lc.previous_class,
                "current_class": lc.current_class,
                "class_changed": lc.class_changed,
                "previous_confidence": lc.previous_confidence,
                "current_confidence": lc.current_confidence,
                "confidence_delta": lc.confidence_delta,
                "previous_area_mm2": lc.previous_area_mm2,
                "current_area_mm2": lc.current_area_mm2,
                "area_delta_mm2": lc.area_delta_mm2,
                "area_percentage_change": lc.area_percentage_change,
                "dice_coefficient": lc.dice_coefficient,
                "progression_status": lc.progression_status,
                "summary_text": lc.summary_text,
                "comparison_canvas_path": lc.comparison_canvas_path,
            }

        if getattr(report, "quality_warnings", None) is not None:
            payload["quality_warnings"] = report.quality_warnings

        return payload


    def _build_markdown_content(self, report: ClinicalReport) -> str:
        """Constructs a beautifully formatted Markdown clinical report."""
        # Visual outputs section
        images_section = ""
        if report.overlay_image_path or report.segmentation_mask_path:
            images_section = "\n## Visual Findings & Output Scans\n"
            if report.overlay_image_path:
                images_section += f"- **Grad-CAM Attention Overlay:** `{report.overlay_image_path}`  \n"
            if report.segmentation_mask_path:
                images_section += f"- **Tumor Segmentation Mask:** `{report.segmentation_mask_path}`  \n"
            if getattr(report, "comparison_image_path", None):
                images_section += f"- **Post-Processing Comparison Scan:** `{report.comparison_image_path}`  \n"

        # Morphological section
        morph_section = ""
        if report.segmentation_metrics is not None:
            quality_text = ""
            is_post_processed = getattr(report.segmentation_metrics, "post_processing_applied", False)
            if is_post_processed:
                q_score = report.segmentation_metrics.quality_score
                q_cat = report.segmentation_metrics.quality_category
                steps = ", ".join(report.segmentation_metrics.post_processing_metadata.get("steps_applied", []))
                quality_text = f"""
### Segmentation Quality Assessment
- **Quality Score:** {q_score:.2%} ({q_cat})
- **Morphological Pipeline Filters Applied:** {steps}
"""
            stats_text = ""
            if getattr(report.segmentation_metrics, "stats", None) is not None:
                s = report.segmentation_metrics.stats
                stats_text = f"""
### Lesion Shape & Bounding Box Properties
- **Perimeter:** {s.perimeter_mm:.2f} mm ({s.perimeter_pixels:.1f} pixels)
- **Bounding Box Size:** {s.bbox_w_mm:.2f} mm (W) x {s.bbox_h_mm:.2f} mm (H)
- **Bounding Box Coordinates:** X={s.bbox_x_px} px, Y={s.bbox_y_px} px
- **Equivalent Ellipse Major Axis:** {s.major_axis_mm:.2f} mm
- **Equivalent Ellipse Minor Axis:** {s.minor_axis_mm:.2f} mm
- **Lesion Eccentricity:** {s.eccentricity:.4f}
- **Lesion Orientation:** {s.orientation_deg:.1f}°
- **Solidity Index:** {s.solidity:.4f}
- **Circularity Index:** {s.circularity:.4f}
"""

            morph_section = f"""
## Quantitative Morphological Analysis
- **Tumor Pixel Count:** {report.segmentation_metrics.pixel_count:,} pixels
- **Estimated Tumor Area:** {report.segmentation_metrics.tumor_area_mm2:.2f} mm²
- **Parenchymal Space Occupied (Tumor % of Brain):** {report.segmentation_metrics.tumor_percentage_brain:.4f}%
- **Total Scan Area Occupied (Tumor % of Image):** {report.segmentation_metrics.tumor_percentage_image:.4f}%
- **Estimated Brain Parenchyma Size:** {report.segmentation_metrics.estimated_brain_pixel_count:,} pixels
{quality_text}
{stats_text}"""

        # Severity section
        severity_section = ""
        if report.severity_assessment is not None:
            severity_section = f"""
## Rule-Based AI Severity Assessment
- **Severity Category:** **{report.severity_assessment.category.value.upper()}**
- **Matched Rule:** {report.severity_assessment.rule_description}

{report.severity_assessment.educational_disclaimer}
"""

        # Detailed probabilities list
        probs_text = ""
        for cls_name, prob in report.classification.probabilities.items():
            probs_text += f"  - {cls_name:<16}: {prob:.4%}\n"

        # Calibration-related Markdown lines
        is_calibrated = getattr(report.classification, "is_calibrated", False)
        if is_calibrated:
            cal_params = report.classification.calibration_parameters or {}
            param_str = ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in cal_params.items())
            calibration_text = f"""- **Confidence Score (Calibrated):** {report.classification.confidence_score:.4%}
- **Confidence Score (Uncalibrated):** {report.classification.uncalibrated_confidence_score:.4%}
- **Calibration Method:** {report.classification.calibration_method} ({param_str})"""
        else:
            calibration_text = f"- **Confidence Score:** {report.classification.confidence_score:.4%}"

        # Explainable AI section
        xai_section = ""
        if getattr(report, "xai_method", None) is not None:
            method_display = (
                "Grad-CAM"
                if report.xai_method == "gradcam"
                else "Grad-CAM++"
                if report.xai_method in ["gradcam_plus_plus", "gradcam++"]
                else "EigenCAM"
            )
            overlap_str = f"{report.xai_overlap_percentage:.2%}" if report.xai_overlap_percentage is not None else "N/A"
            xai_section = f"""
## Explainable AI (XAI 2.0) Analysis
- **Active Explanation Method:** {method_display}
- **Lesion Spatial Overlap:** {overlap_str}
- **Clinical Interpretation:** {report.xai_explanation_text}
"""

        # Longitudinal comparison section
        longitudinal_section = ""
        if getattr(report, "longitudinal_comparison", None) is not None:
            lc = report.longitudinal_comparison
            img_comp = ""
            if lc.comparison_canvas_path:
                img_comp = f"- **Longitudinal Comparison Visual Canvas:** `{lc.comparison_canvas_path}`  \n"
            
            shape_deltas_text = ""
            if lc.perimeter_delta_mm is not None:
                shape_deltas_text = f"""  - **Perimeter Delta:** {lc.perimeter_delta_mm:+.2f} mm
  - **Major Axis Delta:** {lc.major_axis_delta_mm:+.2f} mm
  - **Minor Axis Delta:** {lc.minor_axis_delta_mm:+.2f} mm
  - **Solidity Delta:** {lc.solidity_delta:+.4f}
  - **Circularity Delta:** {lc.circularity_delta:+.4f}"""

            longitudinal_section = f"""
## Longitudinal Scan Comparison & Evolution
- **Previous Scan Date:** {lc.previous_scan_date}
- **Current Scan Date:** {lc.current_scan_date}
- **Evolution Status:** **{lc.progression_status.upper()}**
- **Clinical Follow-up Summary:** {lc.summary_text}
- **Quantitative Delta Metrics:**
  - **Classification Shift:** '{lc.previous_class}' -> '{lc.current_class}' (Changed: {lc.class_changed})
  - **Confidence Delta:** {lc.confidence_delta:+.2%}
  - **Tumor Area Delta:** {lc.area_delta_mm2:+.2f} mm² ({lc.area_percentage_change:+.1f}%)
  - **Brain Space Occupancy Delta:** {lc.pct_brain_delta:+.4f}%
{shape_deltas_text}
{img_comp}"""


        # Quality Warnings section
        warnings_section = ""
        if getattr(report, "quality_warnings", None):
            warnings_section = "## AI Diagnostic Quality & Coherence Warnings\n"
            for warning in report.quality_warnings:
                warnings_section += f"- ⚠️ {warning}  \n"
            warnings_section += "\n---\n\n"

        content = f"""# INTEGRATED CLINICAL BRAIN MRI REPORT

## Patient Demographics
- **Patient ID:** {report.patient_info.patient_id}
- **Patient Name:** {report.patient_info.name}
- **Age / Gender:** {report.patient_info.age} years / {report.patient_info.gender}
- **Referring Physician:** {report.patient_info.ref_physician}
- **Scan/Acquisition Date:** {report.patient_info.scan_date}

---

{warnings_section}## Clinical Classification Summary
- **Primary Diagnosis:** **{report.classification.class_name}**
{calibration_text}
- **Differential Classification Probabilities:**
{probs_text}
---
{xai_section}
---
{morph_section}
---
{severity_section}
---
{longitudinal_section}
---
{images_section}
---

## Technical Processing Summary
- **Execution Platform / Device:** {report.processing_summary.device.upper()}
- **Total Processing Time:** {report.processing_summary.execution_time_sec:.4f} seconds
- **Classification Model State:** `{report.processing_summary.classification_model_path}`
- **Segmentation Model State:** `{report.processing_summary.segmentation_model_path}`
- **Latency Breakdown:**
  - Classification Inference : {report.processing_summary.classification_latency_sec:.4f} s
  - Segmentation Inference   : {report.processing_summary.segmentation_latency_sec:.4f} s
  - Explainability Hooks     : {report.processing_summary.explainability_latency_sec:.4f} s

---
*Report generated automatically by the Integrated Clinical Report Generation Module.*
"""
        return content
