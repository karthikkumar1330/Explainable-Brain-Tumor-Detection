import os
import logging
from pathlib import Path
from typing import Tuple, Optional, Any
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    KeepTogether
)
from reportlab.pdfgen import canvas
from clinical_reporting.domain.entities import ClinicalReport


class NumberedCanvas(canvas.Canvas):
    """Canvas that draws the page numbers, running header, and educational disclaimer on every page."""

    # Class-level properties to pass metadata to canvas rendering
    report_id = "N/A"
    brand_name = "AuraScan AI"
    report_title = "CLINICAL REPORT"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self) -> None:
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int) -> None:
        self.saveState()

        # Draw running header
        self.setFont("Helvetica-Bold", 8)
        self.setFillColor(colors.HexColor("#1B365D"))
        self.drawString(54, 750, f"{self.brand_name.upper()} — {self.report_title.upper()}")
        self.setFont("Helvetica", 8)
        self.drawRightString(558, 750, f"Report ID: {self.report_id}")

        # Header line
        self.setStrokeColor(colors.HexColor("#BDC3C7"))
        self.setLineWidth(0.5)
        self.line(54, 742, 558, 742)

        # Footer line
        self.line(54, 60, 558, 60)

        # Draw running footer disclaimer
        self.setFont("Helvetica-Oblique", 7)
        self.setFillColor(colors.HexColor("#7F8C8D"))
        disclaimer_lines = [
            "CONFIDENTIAL MEDICAL REPORT — FOR CLINICAL DECISION-SUPPORT / RESEARCH USE ONLY.",
            "This report is AI-generated, is not a final diagnosis, and does not replace assessment by a healthcare professional."
        ]
        self.drawString(54, 48, disclaimer_lines[0])
        self.drawString(54, 38, disclaimer_lines[1])

        # Page numbers
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#1B365D"))
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 44, page_str)

        self.restoreState()


class ReportLabPDFGenerator:
    """Renders the ClinicalReport or ReportData entity into a professional PDF document."""

    def generate_pdf(self, report: Any, output_path: str) -> None:
        """Generates a styled clinical PDF report.

        Args:
            report: The ClinicalReport or ReportData object.
            output_path: Path where PDF will be saved.
        """
        logger = logging.getLogger("pdf_generator.ReportLabPDFGenerator")
        output_path_obj = Path(output_path).resolve()
        output_dir = output_path_obj.parent
        filename = output_path_obj.name
        
        logger.info(f"Generating clinical report PDF. Output directory: {output_dir}, Filename: {filename}")
        
        # Create missing directories automatically
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Normalize into ReportData
        from clinical_reporting.domain.entities import ReportData, ClinicalReport
        if isinstance(report, ClinicalReport):
            from clinical_reporting.application.builder import ReportDataBuilder
            builder = ReportDataBuilder()
            data = builder.build(report)
        elif isinstance(report, ReportData):
            data = report
        else:
            raise ValueError(f"Unsupported report type: {type(report)}. Expected ClinicalReport or ReportData.")

        # 2. Configure canvas-level values
        NumberedCanvas.report_id = data.metadata.report_id
        NumberedCanvas.brand_name = data.metadata.brand_name
        NumberedCanvas.report_title = data.metadata.report_title

        # Margins: 0.75 in (54 pt) top/bottom, 0.75 in (54 pt) left/right
        doc = SimpleDocTemplate(
            str(output_path_obj),
            pagesize=letter,
            leftMargin=54,
            rightMargin=54,
            topMargin=72,      # Leave room for running header (750 pt)
            bottomMargin=80,   # Leave room for running footer (60 pt)
        )

        # 3. Render flowables
        from clinical_reporting.infrastructure.renderer import EnterpriseReportRenderer
        renderer = EnterpriseReportRenderer()
        story = renderer.render(data)

        # 4. Build document
        doc.build(story, canvasmaker=NumberedCanvas)

        # Verify the PDF is actually written
        logger.info(f"Verifying PDF file existence check at: {output_path_obj}")
        if output_path_obj.is_file():
            logger.info(f"PDF successfully written and verified at: {output_path_obj}")
        else:
            logger.error(f"PDF file is missing on disk after generation: {output_path_obj}")


def load_or_regenerate_pdf(report_id: int, db_path: str) -> Optional[str]:
    """Retrieves the PDF path, and if the PDF file is missing on disk but the JSON report exists, re-compiles the PDF on the fly."""
    import sqlite3
    import logging
    from pathlib import Path
    
    logger = logging.getLogger("pdf_generator.load_or_regenerate_pdf")
    db_path_obj = Path(db_path).resolve()
    logger.info(f"Connecting to database at {db_path_obj} for report ID: {report_id}")
    
    if not db_path_obj.is_file():
        logger.error(f"Database file not found at: {db_path_obj}")
        return None

    conn = sqlite3.connect(str(db_path_obj))
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT json_path, pdf_path FROM clinical_reports WHERE id = ?;",
            (report_id,)
        )
        row = cursor.fetchone()
        if not row:
            logger.warning(f"No clinical report record found in DB for ID: {report_id}")
            return None
        json_path_str = row["json_path"]
        pdf_path_str = row["pdf_path"]
        
        pdf_path_obj = Path(pdf_path_str).resolve() if pdf_path_str else None
        json_path_obj = Path(json_path_str).resolve() if json_path_str else None
        
        # If PDF exists, return it
        if pdf_path_obj:
            logger.info(f"Checking if PDF exists at: {pdf_path_obj}")
            if pdf_path_obj.is_file():
                logger.info(f"PDF exists at: {pdf_path_obj}")
                return str(pdf_path_obj)
            else:
                logger.warning(f"PDF does not exist at: {pdf_path_obj}")
            
        # If PDF is missing but JSON exists, regenerate it!
        if json_path_obj and json_path_obj.is_file():
            logger.info(f"JSON file exists at: {json_path_obj}. Attempting to regenerate PDF.")
            try:
                import json
                with open(json_path_obj, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Reconstruct entities
                from clinical_reporting.domain.entities import PatientInfo, ProcessingSummary, ClinicalReport
                from classification.domain.entities import PredictionResult
                from tumor_analysis.domain.entities import TumorAnalysisResult, SeverityLevel
                from severity_assessment.domain.entities import SeverityAssessment, SeverityCategory
                from clinical_insight.domain.entities import ClinicalInsight
                
                p_data = data.get("patient", {})
                patient = PatientInfo(
                    patient_id=p_data.get("patient_id", "N/A"),
                    name=p_data.get("name", "N/A"),
                    age=p_data.get("age", 45),
                    gender=p_data.get("gender", "Female"),
                    scan_date=p_data.get("scan_date", "N/A"),
                    ref_physician=p_data.get("ref_physician", "N/A")
                )
                
                proc_data = data.get("processing", {})
                lat_data = proc_data.get("latency_sec", {})
                proc = ProcessingSummary(
                    device=proc_data.get("device", "CPU"),
                    execution_time_sec=proc_data.get("total_execution_time_sec"),
                    classification_model_path=proc_data.get("classification_model", ""),
                    segmentation_model_path=proc_data.get("segmentation_model", ""),
                    classification_latency_sec=lat_data.get("classification"),
                    segmentation_latency_sec=lat_data.get("segmentation"),
                    explainability_latency_sec=lat_data.get("explainability")
                )
                
                cls_data = data.get("classification", {})
                classification = PredictionResult(
                    label=0, # placeholder
                    class_name=cls_data.get("predicted_class", "No Tumor"),
                    confidence_score=cls_data.get("confidence_score", 0.0),
                    probabilities=cls_data.get("probabilities", {}),
                    is_calibrated=cls_data.get("is_calibrated", False),
                    uncalibrated_confidence_score=cls_data.get("uncalibrated_confidence_score"),
                    uncalibrated_probabilities=cls_data.get("uncalibrated_probabilities"),
                    calibration_method=cls_data.get("calibration_method"),
                    calibration_parameters=cls_data.get("calibration_parameters")
                )
                
                seg_metrics = None
                seg_data = data.get("segmentation")
                if seg_data:
                    try:
                        seg_metrics = TumorAnalysisResult(
                            pixel_count=seg_data.get("pixel_count", 0),
                            tumor_area_mm2=seg_data.get("tumor_area_mm2", 0.0),
                            tumor_percentage_brain=seg_data.get("tumor_percentage_brain", 0.0),
                            tumor_percentage_image=seg_data.get("tumor_percentage_image", 0.0),
                            estimated_brain_pixel_count=seg_data.get("estimated_brain_pixel_count", 50000),
                            severity_level=SeverityLevel.LOW, # placeholder
                            post_processing_applied=seg_data.get("post_processing_applied", False),
                            quality_score=seg_data.get("quality_score"),
                            quality_category=seg_data.get("quality_category"),
                            post_processing_metadata=seg_data.get("post_processing_metadata")
                        )
                    except Exception:
                        pass
                
                severity = None
                sev_data = data.get("severity") or data.get("classification", {}) # fallback keys
                if sev_data and "rule_based_severity" in sev_data:
                    try:
                        cat_str = sev_data.get("rule_based_severity", "LOW")
                        severity = SeverityAssessment(
                            category=SeverityCategory(cat_str.upper()) if cat_str.upper() in ["LOW", "MEDIUM", "HIGH"] else SeverityCategory.LOW,
                            rule_description=sev_data.get("severity_rule_description", "")
                        )
                    except Exception:
                        pass
                
                insight = None
                ins_data = data.get("clinical_insight") or data.get("insight")
                if ins_data:
                    try:
                        insight = ClinicalInsight(
                            summary_narrative=ins_data.get("summary_narrative", ""),
                            key_findings=ins_data.get("key_findings", []),
                            recommendations=ins_data.get("recommendations", [])
                        )
                    except Exception:
                        pass
                
                files_data = data.get("files", {})
                orig_p = files_data.get("original_image")
                heat_p = files_data.get("heatmap_image")
                over_p = files_data.get("overlay_image")
                mask_p = files_data.get("segmentation_mask")
                comp_p = files_data.get("comparison_image")
                
                report = ClinicalReport(
                    patient_info=patient,
                    processing_summary=proc,
                    classification=classification,
                    segmentation_metrics=seg_metrics,
                    severity_assessment=severity,
                    original_image_path=str(Path(orig_p).resolve()) if orig_p else None,
                    heatmap_image_path=str(Path(heat_p).resolve()) if heat_p else None,
                    overlay_image_path=str(Path(over_p).resolve()) if over_p else None,
                    segmentation_mask_path=str(Path(mask_p).resolve()) if mask_p else None,
                    comparison_image_path=str(Path(comp_p).resolve()) if comp_p else None,
                    clinical_insight=insight
                )
                
                # Create parent directory for PDF if it doesn't exist
                if pdf_path_obj:
                    logger.info(f"Creating parent directories and generating PDF report at: {pdf_path_obj}")
                    pdf_path_obj.parent.mkdir(parents=True, exist_ok=True)
                    pdf_gen = ReportLabPDFGenerator()
                    pdf_gen.generate_pdf(report, str(pdf_path_obj))
                    
                    if pdf_path_obj.is_file():
                        logger.info(f"Successfully regenerated PDF and verified: {pdf_path_obj}")
                        return str(pdf_path_obj)
                    else:
                        logger.error(f"PDF regeneration completed but file does not exist at: {pdf_path_obj}")
            except Exception as e:
                logger.error(f"Failed to regenerate PDF on the fly: {e}", exc_info=True)
        else:
            if not json_path_obj:
                logger.error("JSON report path is null.")
            elif not json_path_obj.is_file():
                logger.error(f"JSON report file not found on disk at: {json_path_obj}")
                
        return str(pdf_path_obj) if pdf_path_obj else None
    finally:
        conn.close()
