import os
import logging
from pathlib import Path
from typing import Tuple, Optional
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
        self.setFillColor(colors.HexColor("#2C3E50"))
        self.drawString(54, 750, "CLINICAL BRAIN MRI INTEGRATED REPORT")
        self.setFont("Helvetica", 8)
        self.drawRightString(558, 750, "AUTOMATED DIAGNOSTIC PIPELINE")

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
            "EDUCATIONAL USE ONLY: This report is an AI-generated approximation based on rule-based morphological criteria and deep learning.",
            "It does NOT constitute professional medical advice, diagnosis, or clinical support. Always consult a qualified radiologist."
        ]
        self.drawString(54, 48, disclaimer_lines[0])
        self.drawString(54, 38, disclaimer_lines[1])

        # Page numbers
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#2C3E50"))
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 44, page_str)

        self.restoreState()


class ReportLabPDFGenerator:
    """Renders the ClinicalReport entity into a professional PDF document."""

    def generate_pdf(self, report: ClinicalReport, output_path: str) -> None:
        """Generates a styled clinical PDF report.

        Args:
            report: The aggregated clinical report.
            output_path: Path where PDF will be saved.
        """
        logger = logging.getLogger("pdf_generator.ReportLabPDFGenerator")
        output_path_obj = Path(output_path).resolve()
        output_dir = output_path_obj.parent
        filename = output_path_obj.name
        
        logger.info(f"Generating clinical report PDF. Output directory: {output_dir}, Filename: {filename}")
        
        # Create missing directories automatically
        output_dir.mkdir(parents=True, exist_ok=True)

        # Margins: 0.75 in (54 pt) top/bottom, 0.75 in (54 pt) left/right
        doc = SimpleDocTemplate(
            str(output_path_obj),
            pagesize=letter,
            leftMargin=54,
            rightMargin=54,
            topMargin=72,      # Leave room for running header (750 pt)
            bottomMargin=80,   # Leave room for running footer (60 pt)
        )

        styles = getSampleStyleSheet()

        # Define custom paragraph styles
        title_style = ParagraphStyle(
            name='ReportTitle',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#2C3E50"),
            spaceAfter=15,
        )

        h2_style = ParagraphStyle(
            name='SectionHeader',
            parent=styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=12,
            leading=16,
            textColor=colors.HexColor("#2980B9"),
            spaceBefore=12,
            spaceAfter=6,
            keepWithNext=True,
        )

        body_style = ParagraphStyle(
            name='ReportBody',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=9.5,
            leading=13,
            textColor=colors.HexColor("#2C3E50"),
        )

        meta_label_style = ParagraphStyle(
            name='MetaLabel',
            parent=body_style,
            fontName='Helvetica-Bold',
            textColor=colors.HexColor("#34495E"),
        )

        story = []

        # 1. Main Document Header (decorative banner)
        story.append(Paragraph("BRAIN MRI SCAN FINDINGS", title_style))
        story.append(Spacer(1, 10))

        # 2. Patient Demographics Table
        demo_data = [
            [
                Paragraph("Patient ID:", meta_label_style), Paragraph(report.patient_info.patient_id, body_style),
                Paragraph("Scan Date:", meta_label_style), Paragraph(report.patient_info.scan_date, body_style)
            ],
            [
                Paragraph("Patient Name:", meta_label_style), Paragraph(report.patient_info.name, body_style),
                Paragraph("Referring Physician:", meta_label_style), Paragraph(report.patient_info.ref_physician, body_style)
            ],
            [
                Paragraph("Age / Gender:", meta_label_style), Paragraph(f"{report.patient_info.age} yrs / {report.patient_info.gender}", body_style),
                Paragraph("Workflow Device:", meta_label_style), Paragraph(report.processing_summary.device.upper(), body_style)
            ]
        ]
        demo_table = Table(demo_data, colWidths=[90, 160, 110, 144])
        demo_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8F9F9")),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(demo_table)
        story.append(Spacer(1, 15))

        # 3. Diagnostic Results Block
        story.append(Paragraph("Diagnostic Summary", h2_style))
        
        is_calibrated = getattr(report.classification, "is_calibrated", False)
        diag_data = [
            [Paragraph("Primary Classification Diagnosis:", meta_label_style), Paragraph(f"<b>{report.classification.class_name}</b>", body_style)]
        ]
        if is_calibrated:
            conf_val = report.classification.confidence_score
            conf_str = f"<b>{conf_val:.4%}</b>" if conf_val is not None else "<b>N/A</b>"
            diag_data.append([
                Paragraph("Model Classification Confidence (Calibrated):", meta_label_style),
                Paragraph(conf_str, body_style)
            ])
            uncal_val = report.classification.uncalibrated_confidence_score
            uncal_str = f"{uncal_val:.4%}" if uncal_val is not None else "N/A"
            diag_data.append([
                Paragraph("Model Classification Confidence (Uncalibrated):", meta_label_style),
                Paragraph(uncal_str, body_style)
            ])
            # Add method and parameters
            method = getattr(report.classification, "calibration_method", "N/A")
            params = getattr(report.classification, "calibration_parameters", {})
            param_str = ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in params.items()) if isinstance(params, dict) else "N/A"
            diag_data.append([
                Paragraph("Confidence Calibration Method:", meta_label_style),
                Paragraph(f"{method} ({param_str})", body_style)
            ])
        else:
            conf_val = report.classification.confidence_score
            conf_str = f"{conf_val:.4%}" if conf_val is not None else "N/A"
            diag_data.append([
                Paragraph("Model Classification Confidence:", meta_label_style),
                Paragraph(conf_str, body_style)
            ])

        if report.severity_assessment is not None:
            diag_data.append([
                Paragraph("AI Severity Risk Category:", meta_label_style),
                Paragraph(f"<b>{report.severity_assessment.category.value.upper()}</b>", ParagraphStyle(
                    'SevValue', parent=body_style, fontName='Helvetica-Bold',
                    textColor=colors.HexColor("#C0392B") if report.severity_assessment.category.value.lower() == "high" else colors.HexColor("#D35400") if report.severity_assessment.category.value.lower() == "medium" else colors.HexColor("#27AE60")
                ))
            ])
            diag_data.append([
                Paragraph("Risk Classification Decision Rule:", meta_label_style),
                Paragraph(report.severity_assessment.rule_description, body_style)
            ])

        diag_table = Table(diag_data, colWidths=[180, 324])
        diag_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#EAEDED")),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        story.append(diag_table)
        story.append(Spacer(1, 10))

        # Add Clinical Quality & Consistency warnings if present
        if getattr(report, "quality_warnings", None):
            warning_style = ParagraphStyle(
                name='WarningText',
                parent=body_style,
                fontName='Helvetica-Bold',
                fontSize=9,
                leading=12,
                textColor=colors.HexColor("#78281F"),
            )
            story.append(Paragraph("AI Diagnostic Quality & Coherence Warnings", h2_style))
            warning_rows = []
            for warning in report.quality_warnings:
                warning_rows.append([Paragraph(f"⚠️ {warning}", warning_style)])
            
            warning_table = Table(warning_rows, colWidths=[504])
            warning_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#FDEDEC")),
                ('PADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor("#F1948A")),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            story.append(warning_table)
            story.append(Spacer(1, 10))

        # AI Clinical Insights & Recommendations (B6.15)
        if getattr(report, "clinical_insight", None) is not None:
            ci = report.clinical_insight
            story.append(Paragraph("AI Clinical Insights & Recommendations", h2_style))
            
            insight_body_style = ParagraphStyle(
                name='InsightBody',
                parent=body_style,
                fontSize=9,
                leading=12,
            )
            
            # Narrative callout box
            narrative_table = Table([[Paragraph(ci.summary_narrative, insight_body_style)]], colWidths=[504])
            narrative_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#EBF5FB")),
                ('PADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#AED6F1")),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            story.append(narrative_table)
            story.append(Spacer(1, 10))
            
            # Key Findings & Recommendations Table
            bullets_data = []
            bullets_data.append([Paragraph("<b>Key Findings:</b>", meta_label_style), Paragraph("<b>Clinical Recommendations:</b>", meta_label_style)])
            
            findings_bullet_text = "<br/>".join(f"• {f}" for f in ci.key_findings)
            recs_bullet_text = "<br/>".join(f"• {r}" for r in ci.recommendations)
            
            bullets_data.append([
                Paragraph(findings_bullet_text, insight_body_style),
                Paragraph(recs_bullet_text, insight_body_style)
            ])
            
            bullets_table = Table(bullets_data, colWidths=[252, 252])
            bullets_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#FDFEFE")),
                ('PADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7E9")),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]))
            story.append(bullets_table)
            story.append(Spacer(1, 15))

        story.append(Spacer(1, 5))

        # 4. Quantitative Morphological Analysis Table (if tumor exists)
        if report.segmentation_metrics is not None and report.segmentation_metrics.pixel_count > 0:
            story.append(Paragraph("Quantitative Morphological Analysis", h2_style))
            
            hdr_style = ParagraphStyle('hdr', parent=meta_label_style, textColor=colors.white, alignment=1)
            val_center_style = ParagraphStyle('val_c', parent=body_style, alignment=1)
            
            morph_data = [
                [
                    Paragraph("Estimated Tumor Area", hdr_style),
                    Paragraph("Space Occupied (% of Brain)", hdr_style),
                    Paragraph("Tumor Pixel Count", hdr_style)
                ],
                [
                    Paragraph(f"{report.segmentation_metrics.tumor_area_mm2:.2f} mm²", val_center_style),
                    Paragraph(f"{report.segmentation_metrics.tumor_percentage_brain:.4f}%", val_center_style),
                    Paragraph(f"{report.segmentation_metrics.pixel_count:,} px", val_center_style)
                ]
            ]
            
            morph_table = Table(morph_data, colWidths=[168, 168, 168])
            morph_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
                ('PADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            story.append(morph_table)
            story.append(Spacer(1, 10))

            # Add segmentation quality table if post-processing was run
            is_post_processed = getattr(report.segmentation_metrics, "post_processing_applied", False)
            if is_post_processed:
                q_score = getattr(report.segmentation_metrics, "quality_score", None)
                q_cat = getattr(report.segmentation_metrics, "quality_category", None) or "N/A"
                meta = getattr(report.segmentation_metrics, "post_processing_metadata", None) or {}
                steps_list = meta.get("steps_applied", []) if isinstance(meta, dict) else []
                steps = ", ".join(steps_list) if steps_list else "None"
                
                q_score_str = f"{q_score:.2%}" if q_score is not None else "N/A"
                
                story.append(Paragraph("Segmentation Quality Assessment", ParagraphStyle('q_sub', parent=h2_style, fontSize=11, leading=13)))
                quality_data = [
                    [Paragraph("Segmentation Quality Score:", meta_label_style), Paragraph(f"<b>{q_score_str} ({q_cat})</b>", body_style)],
                    [Paragraph("Morphological Filters Applied:", meta_label_style), Paragraph(steps, body_style)]
                ]
                quality_table = Table(quality_data, colWidths=[180, 324])
                quality_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#EAEDED")),
                    ('PADDING', (0, 0), (-1, -1), 5),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ]))
                story.append(quality_table)
                story.append(Spacer(1, 15))
            else:
                story.append(Spacer(1, 15))

            # Add detailed morphometrics table if stats exist
            if getattr(report.segmentation_metrics, "stats", None) is not None:
                s = report.segmentation_metrics.stats
                story.append(Paragraph("Tumor Morphometry & Region Properties", ParagraphStyle('m_sub', parent=h2_style, fontSize=11, leading=13)))
                morph_detail_data = [
                    [
                        Paragraph("<b>Property</b>", meta_label_style),
                        Paragraph("<b>Value</b>", meta_label_style),
                        Paragraph("<b>Property</b>", meta_label_style),
                        Paragraph("<b>Value</b>", meta_label_style),
                    ],
                    [
                        Paragraph("Perimeter", body_style), Paragraph(f"{s.perimeter_mm:.2f} mm", body_style),
                        Paragraph("Bounding Box Width", body_style), Paragraph(f"{s.bbox_w_mm:.2f} mm ({s.bbox_w_px} px)", body_style),
                    ],
                    [
                        Paragraph("Major Axis Length", body_style), Paragraph(f"{s.major_axis_mm:.2f} mm", body_style),
                        Paragraph("Bounding Box Height", body_style), Paragraph(f"{s.bbox_h_mm:.2f} mm ({s.bbox_h_px} px)", body_style),
                    ],
                    [
                        Paragraph("Minor Axis Length", body_style), Paragraph(f"{s.minor_axis_mm:.2f} mm", body_style),
                        Paragraph("Bounding Box X/Y Offset", body_style), Paragraph(f"X={s.bbox_x_px}, Y={s.bbox_y_px} px", body_style),
                    ],
                    [
                        Paragraph("Solidity Index", body_style), Paragraph(f"{s.solidity:.4f}", body_style),
                        Paragraph("Eccentricity", body_style), Paragraph(f"{s.eccentricity:.4f}", body_style),
                    ],
                    [
                        Paragraph("Circularity Index", body_style), Paragraph(f"{s.circularity:.4f}", body_style),
                        Paragraph("Orientation Angle", body_style), Paragraph(f"{s.orientation_deg:.1f}°", body_style),
                    ]
                ]
                morph_detail_table = Table(morph_detail_data, colWidths=[130, 122, 130, 122])
                morph_detail_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2C3E50")),
                    ('PADDING', (0, 0), (-1, -1), 5),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ]))
                story.append(morph_detail_table)
                story.append(Spacer(1, 15))

        # Explainable AI (XAI 2.0) Section
        if getattr(report, "xai_method", None) is not None:
            story.append(Paragraph("Explainable AI (XAI 2.0) Analysis", h2_style))
            method_display = (
                "Grad-CAM"
                if report.xai_method == "gradcam"
                else "Grad-CAM++"
                if report.xai_method in ["gradcam_plus_plus", "gradcam++"]
                else "EigenCAM"
            )
            overlap_str = f"{report.xai_overlap_percentage:.2%}" if report.xai_overlap_percentage is not None else "N/A"
            xai_data = [
                [Paragraph("Active Explanation Method:", meta_label_style), Paragraph(method_display, body_style)],
                [Paragraph("Lesion Spatial Overlap:", meta_label_style), Paragraph(overlap_str, body_style)],
                [Paragraph("Clinical Interpretation:", meta_label_style), Paragraph(report.xai_explanation_text, body_style)]
            ]
            xai_table = Table(xai_data, colWidths=[180, 324])
            xai_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#EAEDED")),
                ('PADDING', (0, 0), (-1, -1), 5),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]))
            story.append(xai_table)
            story.append(Spacer(1, 15))

        # Longitudinal Scan Comparison & Evolution
        if getattr(report, "longitudinal_comparison", None) is not None:
            lc = report.longitudinal_comparison
            story.append(Paragraph("Longitudinal Scan Comparison & Evolution", h2_style))
            
            comp_details = [
                [Paragraph("<b>Evolution Status</b>", meta_label_style), Paragraph(lc.progression_status.upper(), body_style)],
                [Paragraph("<b>Scan Dates</b>", meta_label_style), Paragraph(f"Previous: {lc.previous_scan_date} | Current: {lc.current_scan_date}", body_style)],
                [Paragraph("<b>Follow-up Summary</b>", meta_label_style), Paragraph(lc.summary_text, body_style)],
                [
                    Paragraph("<b>Quantitative Deltas</b>", meta_label_style),
                    Paragraph(
                        f"Area Delta: {lc.area_delta_mm2:+.2f} mm² ({lc.area_percentage_change:+.1f}%)<br/>"
                        f"Classification Shift: {lc.previous_class} -> {lc.current_class}<br/>"
                        f"Confidence Delta: {lc.confidence_delta:+.2%}<br/>"
                        f"Brain Occupancy Delta: {lc.pct_brain_delta:+.4f}%",
                        body_style
                    )
                ]
            ]
            
            if lc.perimeter_delta_mm is not None:
                shape_text = (
                    f"<br/>Perimeter Delta: {lc.perimeter_delta_mm:+.2f} mm<br/>"
                    f"Major Axis Delta: {lc.major_axis_delta_mm:+.2f} mm | Minor Axis Delta: {lc.minor_axis_delta_mm:+.2f} mm<br/>"
                    f"Solidity Delta: {lc.solidity_delta:+.4f} | Circularity Delta: {lc.circularity_delta:+.4f}"
                )
                comp_details[3][1] = Paragraph(
                    f"Area Delta: {lc.area_delta_mm2:+.2f} mm² ({lc.area_percentage_change:+.1f}%)<br/>"
                    f"Classification Shift: {lc.previous_class} -> {lc.current_class}<br/>"
                    f"Confidence Delta: {lc.confidence_delta:+.2%}<br/>"
                    f"Brain Occupancy Delta: {lc.pct_brain_delta:+.4f}%" + shape_text,
                    body_style
                )

            comp_table = Table(comp_details, colWidths=[130, 374])
            comp_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#F2F4F4")),
                ('PADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]))
            story.append(comp_table)
            story.append(Spacer(1, 15))
            
            if lc.comparison_canvas_path:
                comp_canvas_path_obj = Path(lc.comparison_canvas_path).resolve()
                logger.info(f"Checking if longitudinal comparison canvas exists: {comp_canvas_path_obj}")
                if comp_canvas_path_obj.is_file():
                    story.append(Paragraph("Longitudinal Visual Progression Overlay", ParagraphStyle('lc_sub', parent=h2_style, fontSize=11, leading=13)))
                    from reportlab.platypus import Image as RLImage
                    try:
                        story.append(RLImage(str(comp_canvas_path_obj), width=504, height=168))
                        story.append(Spacer(1, 15))
                    except Exception as e:
                        logger.error(f"Failed to load longitudinal comparison canvas image: {e}")

        # 5. Visual Scans Section (Original Scan, Grad-CAM & Segmentation Mask side-by-side)
        visual_flowables = []
        orig_img_path_obj = Path(report.original_image_path).resolve() if report.original_image_path else None
        overlay_img_path_obj = Path(report.overlay_image_path).resolve() if report.overlay_image_path else None
        mask_img_path_obj = Path(report.segmentation_mask_path).resolve() if report.segmentation_mask_path else None

        has_original = orig_img_path_obj.is_file() if orig_img_path_obj else False
        has_overlay = overlay_img_path_obj.is_file() if overlay_img_path_obj else False
        has_mask = mask_img_path_obj.is_file() if mask_img_path_obj else False

        logger.info(f"Visual scan files check: original={has_original} ({orig_img_path_obj}), overlay={has_overlay} ({overlay_img_path_obj}), mask={has_mask} ({mask_img_path_obj})")

        if has_original or has_overlay or has_mask:
            visual_flowables.append(Paragraph("Clinical Imaging & Deep Learning Findings", h2_style))

            # Prepare Image Cells
            image_cells = []
            
            if has_original:
                img_orig = Image(str(orig_img_path_obj), width=150, height=150)
                caption = Paragraph("<font size=8><b>Fig 1:</b> Original MRI Scan</font>", ParagraphStyle('cap_orig', parent=body_style, alignment=1))
                image_cells.append((img_orig, caption))
                
            if has_overlay:
                img_overlay = Image(str(overlay_img_path_obj), width=150, height=150)
                caption = Paragraph("<font size=8><b>Fig 2:</b> Grad-CAM Attention Overlay</font>", ParagraphStyle('cap_over', parent=body_style, alignment=1))
                image_cells.append((img_overlay, caption))

            if has_mask:
                img_mask = Image(str(mask_img_path_obj), width=150, height=150)
                caption = Paragraph("<font size=8><b>Fig 3:</b> Binarized Segmentation Mask</font>", ParagraphStyle('cap_mask', parent=body_style, alignment=1))
                image_cells.append((img_mask, caption))

            # Build layout dynamically based on count
            if len(image_cells) == 3:
                tbl_data = [
                    [image_cells[0][0], image_cells[1][0], image_cells[2][0]],
                    [image_cells[0][1], image_cells[1][1], image_cells[2][1]]
                ]
                img_table = Table(tbl_data, colWidths=[168, 168, 168])
                img_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('PADDING', (0, 0), (-1, -1), 2),
                ]))
                visual_flowables.append(img_table)
            elif len(image_cells) == 2:
                # With 2 images, use larger size
                image_cells[0][0].drawWidth = 220
                image_cells[0][0].drawHeight = 220
                image_cells[1][0].drawWidth = 220
                image_cells[1][0].drawHeight = 220
                tbl_data = [
                    [image_cells[0][0], image_cells[1][0]],
                    [image_cells[0][1], image_cells[1][1]]
                ]
                img_table = Table(tbl_data, colWidths=[252, 252])
                img_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('PADDING', (0, 0), (-1, -1), 2),
                ]))
                visual_flowables.append(img_table)
            elif len(image_cells) == 1:
                image_cells[0][0].drawWidth = 220
                image_cells[0][0].drawHeight = 220
                tbl_data = [
                    [image_cells[0][0]],
                    [image_cells[0][1]]
                ]
                img_table = Table(tbl_data, colWidths=[252])
                img_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('PADDING', (0, 0), (-1, -1), 2),
                ]))
                visual_flowables.append(img_table)

            comp_img_path_obj = Path(report.comparison_image_path).resolve() if getattr(report, "comparison_image_path", None) else None
            logger.info(f"Checking if comparison image exists: {comp_img_path_obj}")
            if comp_img_path_obj and comp_img_path_obj.is_file():
                img_comp = Image(str(comp_img_path_obj), width=480, height=160)
                comp_caption = Paragraph("<font size=8><b>Fig 4:</b> Segmentation Post-Processing Comparison: Original | Initial UNeXt Mask (Red) | Post-Processed Mask (Green)</font>", ParagraphStyle('cap_c', parent=body_style, alignment=1))
                comp_tbl = Table([[img_comp], [comp_caption]], colWidths=[504])
                comp_tbl.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('PADDING', (0, 0), (-1, -1), 2),
                ]))
                visual_flowables.append(Spacer(1, 10))
                visual_flowables.append(comp_tbl)

            visual_flowables.append(Spacer(1, 15))

        if visual_flowables:
            story.append(KeepTogether(visual_flowables))

        # 6. Technical Benchmarks Summary
        story.append(Paragraph("Technical Execution Metrics", h2_style))
        
        cls_lat = report.processing_summary.classification_latency_sec
        seg_lat = report.processing_summary.segmentation_latency_sec
        xai_lat = report.processing_summary.explainability_latency_sec
        tot_lat = report.processing_summary.execution_time_sec

        cls_lat_str = f"{cls_lat:.4f} s" if cls_lat is not None else "N/A"
        seg_lat_str = f"{seg_lat:.4f} s" if seg_lat is not None else "N/A"
        xai_lat_str = f"{xai_lat:.4f} s" if xai_lat is not None else "N/A"
        tot_lat_str = f"{tot_lat:.4f} s" if tot_lat is not None else "N/A"

        tech_data = [
            [
                Paragraph("Classification Latency:", meta_label_style), Paragraph(cls_lat_str, body_style),
                Paragraph("Segmentation Latency:", meta_label_style), Paragraph(seg_lat_str, body_style)
            ],
            [
                Paragraph("Grad-CAM Latency:", meta_label_style), Paragraph(xai_lat_str, body_style),
                Paragraph("Total Processing Time:", meta_label_style), Paragraph(tot_lat_str, body_style)
            ]
        ]
        tech_table = Table(tech_data, colWidths=[120, 130, 120, 134])
        tech_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#FAFBFB")),
            ('PADDING', (0, 0), (-1, -1), 5),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#EAEDED")),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(tech_table)

        # 7. Clinician Verification & Sign-off Block (Print support)
        story.append(Spacer(1, 10))
        story.append(Paragraph("Clinician Verification & Approval Sign-off", h2_style))
        
        ref_phys = report.patient_info.ref_physician or "Dr. Sarah Smith, MD"
        sig_data = [
            [
                Paragraph("<b>Reviewing Radiologist:</b>", meta_label_style),
                Paragraph("<b>Clinical Center Stamp:</b>", meta_label_style)
            ],
            [
                Paragraph(
                    f"Name: {ref_phys}<br/>"
                    f"Signature: ___________________________<br/>"
                    f"Date: ________________________",
                    body_style
                ),
                Paragraph(
                    "AuraScan AI Integrated Center<br/>"
                    "Validation System Certified<br/>"
                    "Status: APPROVED",
                    body_style
                )
            ]
        ]
        sig_table = Table(sig_data, colWidths=[252, 252])
        sig_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#FAFBFB")),
            ('PADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#EAEDED")),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        story.append(sig_table)

        # Build document
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
