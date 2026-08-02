import json
import os
import logging
from typing import Optional
from tumor_analysis.domain.entities import ClinicalReportData


def save_clinical_report(
    report: ClinicalReportData,
    output_dir: str,
    base_filename: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Saves clinical report data in both JSON (machine-readable) and Markdown (human-readable) formats.

    Args:
        report: The ClinicalReportData object.
        output_dir: Directory where report should be saved.
        base_filename: Filename prefix.
        logger: Optional logger.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Save JSON report (for severity prediction models / electronic health records)
    json_path = os.path.join(output_dir, f"{base_filename}_analysis.json")
    json_data = {
        "patient_id": report.patient_id,
        "tumor_class": report.tumor_class,
        "severity_level": report.analysis.severity_level.value,
        "metrics": {
            "pixel_count": report.analysis.pixel_count,
            "tumor_area_mm2": report.analysis.tumor_area_mm2,
            "tumor_percentage_image": report.analysis.tumor_percentage_image,
            "tumor_percentage_brain": report.analysis.tumor_percentage_brain,
            "estimated_brain_pixel_count": report.analysis.estimated_brain_pixel_count,
        },
        "metadata": report.analysis.metadata,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=4)

    # 2. Save Markdown report (for clinicians)
    md_path = os.path.join(output_dir, f"{base_filename}_clinical_report.md")
    md_content = f"""# CLINICAL BRAIN MRI ANALYSIS REPORT

**Patient/Scan ID:** {report.patient_id}  
**Tumor Classification Type:** {report.tumor_class}  
**Assigned Severity Level:** {report.analysis.severity_level.value}  

---

## Quantitative Morphological Analysis
- **Tumor Pixel Count:** {report.analysis.pixel_count:,} pixels
- **Estimated Tumor Area:** {report.analysis.tumor_area_mm2:.2f} mm²
- **Parenchymal Space Occupied (Tumor % of Brain):** {report.analysis.tumor_percentage_brain:.4f}%
- **Total Scan Area Occupied (Tumor % of Image):** {report.analysis.tumor_percentage_image:.4f}%
- **Estimated Brain Parenchyma Size:** {report.analysis.estimated_brain_pixel_count:,} pixels

---

## Clinical Assessment Notes
{report.clinical_notes}

---

## Action Plan & Recommendations
{report.recommendations}

---
*Report generated automatically by Tumor Area Analysis Module.*
"""

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    if logger:
        logger.info(f"Saved machine-readable clinical report to: {json_path}")
        logger.info(f"Saved human-readable clinical report to: {md_path}")
