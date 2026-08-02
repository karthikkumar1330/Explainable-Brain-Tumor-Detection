import os
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
        # Margins: 0.75 in (54 pt) top/bottom, 0.75 in (54 pt) left/right
        doc = SimpleDocTemplate(
            output_path,
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
        diag_data = [
            [Paragraph("Primary Classification Diagnosis:", meta_label_style), Paragraph(f"<b>{report.classification.class_name}</b>", body_style)],
            [Paragraph("Model Classification Confidence:", meta_label_style), Paragraph(f"{report.classification.confidence_score:.4%}", body_style)]
        ]
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
        story.append(Spacer(1, 15))

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
            story.append(Spacer(1, 15))

        # 5. Visual Scans Section (Grad-CAM & Segmentation Mask side-by-side)
        visual_flowables = []
        if (report.overlay_image_path and os.path.exists(report.overlay_image_path)) or \
           (report.segmentation_mask_path and os.path.exists(report.segmentation_mask_path)):

            visual_flowables.append(Paragraph("Clinical Imaging & Deep Learning Findings", h2_style))

            # Prepare Image Cells
            image_cells = []
            
            if report.overlay_image_path and os.path.exists(report.overlay_image_path):
                img_overlay = Image(report.overlay_image_path, width=220, height=220)
                caption = Paragraph("<font size=8><b>Fig 1:</b> Grad-CAM Attention Overlay (EfficientNet-B0)</font>", ParagraphStyle('cap', parent=body_style, alignment=1))
                image_cells.append([img_overlay, caption])

            if report.segmentation_mask_path and os.path.exists(report.segmentation_mask_path):
                img_mask = Image(report.segmentation_mask_path, width=220, height=220)
                caption = Paragraph("<font size=8><b>Fig 2:</b> Binarized Segmentation Mask (UNeXt)</font>", ParagraphStyle('cap', parent=body_style, alignment=1))
                image_cells.append([img_mask, caption])

            # Build side-by-side table
            if len(image_cells) == 2:
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

            visual_flowables.append(Spacer(1, 15))

        if visual_flowables:
            story.append(KeepTogether(visual_flowables))

        # 6. Technical Benchmarks Summary
        story.append(Paragraph("Technical Execution Metrics", h2_style))
        tech_data = [
            [
                Paragraph("Classification Latency:", meta_label_style), Paragraph(f"{report.processing_summary.classification_latency_sec:.4f} s", body_style),
                Paragraph("Segmentation Latency:", meta_label_style), Paragraph(f"{report.processing_summary.segmentation_latency_sec:.4f} s", body_style)
            ],
            [
                Paragraph("Grad-CAM Latency:", meta_label_style), Paragraph(f"{report.processing_summary.explainability_latency_sec:.4f} s", body_style),
                Paragraph("Total Processing Time:", meta_label_style), Paragraph(f"{report.processing_summary.execution_time_sec:.4f} s", body_style)
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

        # Build document
        doc.build(story, canvasmaker=NumberedCanvas)
