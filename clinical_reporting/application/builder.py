import datetime
from typing import Optional
from clinical_reporting.domain.entities import (
    ClinicalReport,
    ReportData,
    PatientInformation,
    ClassificationResult,
    SegmentationResult,
    TumorStatistics,
    ExplainabilityResult,
    ClinicalInsights,
    ProcessingMetrics,
    SystemStatus,
    Disclaimer,
    Metadata,
)


class ReportDataBuilder:
    """Builds and validates ReportData from ClinicalReport and pipeline outputs."""

    def build(self, report: ClinicalReport, report_id: Optional[str] = None) -> ReportData:
        """Normalizes and validates a ClinicalReport into a structured ReportData object.

        Args:
            report: The aggregated clinical report from the pipeline.
            report_id: Optional custom report identifier.

        Returns:
            A validated ReportData instance.

        Raises:
            ValueError: If critical data (e.g., patient info, classification results) is missing or invalid.
        """
        # 1. Validation checks
        if not report:
            raise ValueError("ClinicalReport cannot be None.")
        if not report.patient_info:
            raise ValueError("PatientInfo is missing from the ClinicalReport.")
        if not report.patient_info.patient_id or not report.patient_info.patient_id.strip():
            raise ValueError("Patient ID cannot be empty.")
        if not report.classification:
            raise ValueError("Classification PredictionResult is missing.")
        if not report.classification.class_name or not report.classification.class_name.strip():
            raise ValueError("Predicted class name cannot be empty.")

        # 2. Extract and Map Patient Information
        patient_info = PatientInformation(
            patient_id=report.patient_info.patient_id.strip(),
            name=report.patient_info.name.strip() if report.patient_info.name else "Anonymous Patient",
            age=report.patient_info.age,
            gender=report.patient_info.gender.strip() if report.patient_info.gender else "Unknown",
            referring_physician=report.patient_info.ref_physician.strip() if report.patient_info.ref_physician else "N/A",
            scan_date=report.patient_info.scan_date.strip() if report.patient_info.scan_date else None,
            original_image_path=report.original_image_path,
        )

        # 3. Extract and Map Classification Result
        classification_result = ClassificationResult(
            predicted_class=report.classification.class_name.strip(),
            confidence_score=report.classification.confidence_score,
            classification_model="EfficientNet-B0",
            model_version="v1.0.0",
        )

        # 4. Extract and Map Segmentation Result
        seg_status = "Available"
        if report.segmentation_metrics is None:
            seg_status = "Failed"
        else:
            # Check warnings for critical segmentation failures
            if report.quality_warnings:
                for warning in report.quality_warnings:
                    if "segmentation execution failed" in warning.lower() or "segmentation engine failed" in warning.lower():
                        seg_status = "Failed"
                        break

        tumor_area = 0.0
        tumor_pct = 0.0
        perimeter = None
        quality_score = None
        quality_category = None

        if report.segmentation_metrics:
            tumor_area = report.segmentation_metrics.tumor_area_mm2
            tumor_pct = report.segmentation_metrics.tumor_percentage_brain
            quality_score = getattr(report.segmentation_metrics, "quality_score", None)
            quality_category = getattr(report.segmentation_metrics, "quality_category", None)
            if report.segmentation_metrics.stats:
                perimeter = report.segmentation_metrics.stats.perimeter_mm

        segmentation_result = SegmentationResult(
            segmentation_status=seg_status,
            tumor_area_mm2=tumor_area,
            tumor_percentage_brain=tumor_pct,
            perimeter_mm=perimeter,
            quality_score=quality_score,
            quality_category=quality_category,
            segmentation_mask_path=report.segmentation_mask_path,
        )

        # 5. Extract Shape Statistics
        shape_stats = {}
        if report.segmentation_metrics and report.segmentation_metrics.stats:
            s = report.segmentation_metrics.stats
            shape_stats = {
                "perimeter_mm": s.perimeter_mm,
                "perimeter_pixels": s.perimeter_pixels,
                "area_pixels": s.area_pixels,
                "area_mm2": s.area_mm2,
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
        tumor_statistics = TumorStatistics(shape_statistics=shape_stats)

        # 6. Extract and Map Explainability Result
        import os
        xai_method = getattr(report, "xai_method", None)
        explanation_text = getattr(report, "xai_explanation_text", None)
        overlap_percentage = getattr(report, "xai_overlap_percentage", None)
        heatmap_image_path = getattr(report, "heatmap_image_path", None)
        overlay_image_path = getattr(report, "overlay_image_path", None)

        xai_status = "Unavailable"
        has_overlay = overlay_image_path and os.path.exists(overlay_image_path) and os.path.isfile(overlay_image_path)
        has_heatmap = heatmap_image_path and os.path.exists(heatmap_image_path) and os.path.isfile(heatmap_image_path)

        if has_overlay or has_heatmap:
            xai_status = "Available"
            if not xai_method:
                xai_method = "gradcam"
        elif xai_method:
            xai_failed = False
            if report.quality_warnings:
                for warning in report.quality_warnings:
                    if "explanation generation failed" in warning.lower() or "explainability engine failed" in warning.lower():
                        xai_failed = True
                        break
            if not xai_failed:
                xai_status = "Available"

        explainability_result = ExplainabilityResult(
            xai_method=xai_method,
            xai_status=xai_status,
            explanation_text=explanation_text,
            overlap_percentage=overlap_percentage,
            heatmap_image_path=heatmap_image_path,
            overlay_image_path=overlay_image_path,
        )

        # 7. Extract Clinical Insights
        summary_narrative = ""
        key_findings = []
        recommendations = []
        if report.clinical_insight:
            summary_narrative = report.clinical_insight.summary_narrative
            key_findings = report.clinical_insight.key_findings
            recommendations = report.clinical_insight.recommendations

        clinical_insights = ClinicalInsights(
            summary_narrative=summary_narrative,
            key_findings=key_findings,
            recommendations=recommendations,
        )

        # 8. Extract Processing Metrics
        device_val = "CPU"
        if report.processing_summary and report.processing_summary.device:
            device_val = report.processing_summary.device.upper()

        proc_metrics = ProcessingMetrics(
            classification_latency_sec=report.processing_summary.classification_latency_sec if report.processing_summary else None,
            segmentation_latency_sec=report.processing_summary.segmentation_latency_sec if report.processing_summary else None,
            explainability_latency_sec=report.processing_summary.explainability_latency_sec if report.processing_summary else None,
            total_execution_time_sec=report.processing_summary.execution_time_sec if report.processing_summary else None,
            device=device_val,
        )

        # 9. Formulate AI System Status
        cls_status = "Available"
        if not report.classification or not report.classification.class_name:
            cls_status = "Failed"

        system_status = SystemStatus(
            classification=cls_status,
            segmentation=seg_status,
            explainability=xai_status,
            report_generation="Successful",
        )

        # 10. Formulate Disclaimer
        disclaimer = Disclaimer(
            text=(
                "This report is AI-generated and intended for research, educational, and decision-support "
                "purposes only. It is not a medical diagnosis and does not replace assessment by a "
                "qualified healthcare professional."
            )
        )

        # 11. Generate Metadata
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # F2.2: Extract verification details if attached to the report object
        version_val = getattr(report, "version", 1)
        status_val = getattr(report, "status", "DRAFT")
        from enum import Enum
        if isinstance(status_val, Enum):
            status_val = status_val.value
        integrity_hash_val = getattr(report, "integrity_hash", "N/A")
        verification_token_val = getattr(report, "verification_token", None)
        report_number_val = getattr(report, "report_number", None)

        if not report_id:
            if report_number_val:
                report_id = report_number_val
            else:
                patient_slug = patient_info.patient_id.replace(" ", "_").upper()
                time_slug = timestamp_str.replace("-", "").replace(":", "").replace(" ", "")
                report_id = f"REP-{patient_slug}-{time_slug[:8]}"

        metadata = Metadata(
            report_id=report_id,
            report_title="ENTERPRISE CLINICAL REPORT",
            report_type="Brain MRI Classification & Segmentation Analysis",
            brand_name="AuraScan AI",
            timestamp=timestamp_str,
            version=version_val,
            status=status_val,
            integrity_hash=integrity_hash_val,
            verification_token=verification_token_val
        )

        # 12. Assemble ReportData
        return ReportData(
            patient_info=patient_info,
            classification_result=classification_result,
            segmentation_result=segmentation_result,
            tumor_statistics=tumor_statistics,
            explainability_result=explainability_result,
            clinical_insights=clinical_insights,
            processing_metrics=proc_metrics,
            system_status=system_status,
            disclaimer=disclaimer,
            metadata=metadata,
        )
