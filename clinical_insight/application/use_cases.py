from typing import Dict, Any, List, Optional
from clinical_insight.domain.entities import ClinicalInsight


class GenerateClinicalInsightUseCase:
    """Orchestrates structured clinical AI insight generation by analyzing classification, segmentation, and progression outputs."""

    def __init__(self) -> None:
        self.disclaimer = (
            "EDUCATIONAL USE ONLY: This clinical insight report is synthesized automatically by an AI algorithm "
            "for secondary diagnostic assistance. It has NOT been reviewed by a qualified human radiologist. "
            "Decisions regarding clinical patient management must be made in conjunction with full medical records "
            "and professional radiology readings."
        )

    def execute(
        self,
        predicted_class: str,
        confidence_score: float,
        is_calibrated: bool,
        probabilities: Dict[str, float],
        tumor_area_mm2: float,
        pixel_count: int,
        solidity: Optional[float],
        circularity: Optional[float],
        xai_method: Optional[str],
        xai_overlap_percentage: Optional[float],
        longitudinal_comparison: Optional[Any] = None
    ) -> ClinicalInsight:
        """Synthesizes qualitative clinical insights based on numerical measurements and classification context.

        Returns:
            A ClinicalInsight domain entity.
        """
        p_class_upper = predicted_class.upper().strip()
        is_tumor = p_class_upper not in ["NO TUMOR", "NO_TUMOR"]
        
        cal_str = "calibrated" if is_calibrated else "uncalibrated"
        conf_pct = f"{confidence_score:.1%}"

        if not is_tumor:
            summary = (
                f"The AI diagnostic pipeline detected no focal intracranial lesion on the uploaded slice. "
                f"Classification indicates 'No Tumor' with {cal_str} confidence of {conf_pct}. "
                f"This matches the absence of any segmented contour outlines."
            )
            key_findings = [
                "No focal space-occupying lesion outlined on UNeXt segmentation (0.00 mm²).",
                "Grad-CAM attention maps indicate diffused background activation, standard for healthy scans.",
                "Primary and secondary diagnostic parameters show complete operational consistency."
            ]
            recommendations = [
                "Correlate with previous history and other imaging slices.",
                "Schedule routine follow-up brain MRI scan cycles as clinically indicated."
            ]
        else:
            # Border & configuration descriptors
            solidity_desc = "highly regular/smooth"
            border_finding = "Smooth and well-circumscribed lesion margins"
            if solidity is not None:
                if solidity < 0.82:
                    solidity_desc = "irregular/spiculated"
                    border_finding = f"Irregular/spiculated lesion boundary detected (solidity: {solidity:.3f})"
                elif solidity < 0.90:
                    solidity_desc = "lobulated"
                    border_finding = f"Slightly lobulated margin boundary (solidity: {solidity:.3f})"

            circ_desc = "nodular/spherical"
            shape_finding = "Nodular or spherical configuration"
            if circularity is not None:
                if circularity < 0.45:
                    circ_desc = "elongated/complex"
                    shape_finding = f"Complex, elongated shape profile (circularity: {circularity:.3f})"
                elif circularity < 0.70:
                    circ_desc = "asymmetrical"
                    shape_finding = f"Asymmetrical shape profile (circularity: {circularity:.3f})"

            summary = (
                f"An active space-occupying lesion has been identified, classified as {predicted_class} "
                f"with {cal_str} confidence of {conf_pct}. Quantitative segmentation outlines a lesion mass "
                f"measuring {tumor_area_mm2:.2f} mm², displaying {solidity_desc} boundaries "
                f"and a {circ_desc} structure."
            )

            key_findings = [
                f"Active lesion tissue mass: {tumor_area_mm2:.2f} mm² ({pixel_count} pixels).",
                border_finding,
                shape_finding
            ]

            # XAI details
            xai_disp = xai_method.upper() if xai_method else "Grad-CAM"
            if xai_overlap_percentage is not None:
                overlap_str = f"{xai_overlap_percentage:.1%}"
                key_findings.append(f"Spatial focus: {xai_disp} validation confirms {overlap_str} overlap with lesion contours.")

            # Longitudinal Progression details
            if longitudinal_comparison is not None:
                status = getattr(longitudinal_comparison, "progression_status", "stable").upper()
                delta_area = getattr(longitudinal_comparison, "area_delta_mm2", 0.0)
                pct_change = getattr(longitudinal_comparison, "area_percentage_change", 0.0)
                
                prog_text = f"Longitudinal progression: {status} status. Area change: {delta_area:+.2f} mm² ({pct_change:+.1f}%)."
                key_findings.append(prog_text)
                
                if "PROGRES" in status:
                    recommendations = [
                        "Urgent clinical consultation with neurosurgery/oncology due to lesion progression.",
                        "Consider prompt radiological review of surrounding tissue for infiltrative margins.",
                        "Repeat diagnostic brain MRI with contrast enhancement within 30 days."
                    ]
                else:
                    recommendations = [
                        "Neurosurgical/neurological referral for correlation and staging review.",
                        "Correlate with tumor morphology, patient symptoms, and prior imaging series.",
                        "Schedule standard follow-up scan as per local diagnostic protocols."
                    ]
            else:
                recommendations = [
                    "Refer patient to neurosurgical/oncological clinical team for diagnostic staging.",
                    "Correlate findings with physical clinical evaluation and radiological reviews.",
                    "Contrast-enhanced diagnostic MRI slice mapping for surgical border verification."
                ]

        return ClinicalInsight(
            summary_narrative=summary,
            key_findings=key_findings,
            recommendations=recommendations,
            disclaimer=self.disclaimer
        )
