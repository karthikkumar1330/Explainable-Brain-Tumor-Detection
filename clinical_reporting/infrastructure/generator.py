import os
import json
from typing import Tuple
from clinical_reporting.domain.entities import ClinicalReport
from clinical_reporting.domain.interfaces import IClinicalReportGenerator


class MarkdownJSONReportGenerator(IClinicalReportGenerator):
    """Generates structured Markdown and JSON reports for clinical brain MRI scans."""

    def generate(self, report: ClinicalReport, output_dir: str) -> Tuple[str, str]:
        """Saves clinical report in both Markdown and JSON formats.

        Args:
            report: The aggregated clinical report.
            output_dir: Destination folder.

        Returns:
            A tuple of (saved_markdown_path, saved_json_path).
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

        return md_path, json_path

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
            },
            "files": {
                "original_image": report.original_image_path,
                "heatmap_image": report.heatmap_image_path,
                "overlay_image": report.overlay_image_path,
                "segmentation_mask": report.segmentation_mask_path,
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
            }

        if report.severity_assessment is not None:
            payload["severity"] = {
                "category": report.severity_assessment.category.value,
                "rule_description": report.severity_assessment.rule_description,
                "educational_disclaimer": report.severity_assessment.educational_disclaimer,
            }

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

        # Morphological section
        morph_section = ""
        if report.segmentation_metrics is not None:
            morph_section = f"""
## Quantitative Morphological Analysis
- **Tumor Pixel Count:** {report.segmentation_metrics.pixel_count:,} pixels
- **Estimated Tumor Area:** {report.segmentation_metrics.tumor_area_mm2:.2f} mm²
- **Parenchymal Space Occupied (Tumor % of Brain):** {report.segmentation_metrics.tumor_percentage_brain:.4f}%
- **Total Scan Area Occupied (Tumor % of Image):** {report.segmentation_metrics.tumor_percentage_image:.4f}%
- **Estimated Brain Parenchyma Size:** {report.segmentation_metrics.estimated_brain_pixel_count:,} pixels
"""

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

        content = f"""# INTEGRATED CLINICAL BRAIN MRI REPORT

## Patient Demographics
- **Patient ID:** {report.patient_info.patient_id}
- **Patient Name:** {report.patient_info.name}
- **Age / Gender:** {report.patient_info.age} years / {report.patient_info.gender}
- **Referring Physician:** {report.patient_info.ref_physician}
- **Scan/Acquisition Date:** {report.patient_info.scan_date}

---

## Clinical Classification Summary
- **Primary Diagnosis:** **{report.classification.class_name}**
- **Confidence Score:** {report.classification.confidence_score:.4%}
- **Differential Classification Probabilities:**
{probs_text}
---
{morph_section}
---
{severity_section}
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
