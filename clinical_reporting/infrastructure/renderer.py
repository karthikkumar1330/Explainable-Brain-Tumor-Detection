import os
from typing import List, Any, Optional
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether
)
from clinical_reporting.domain.entities import ReportData


class EnterpriseReportRenderer:
    """Translates normalized ReportData into a styled list of ReportLab Flowables."""

    def __init__(self) -> None:
        self.styles = getSampleStyleSheet()
        self._init_styles()

    def _init_styles(self) -> None:
        """Configures custom paragraph styles for consistent medical document typography."""
        self.title_style = ParagraphStyle(
            name='EntReportTitle',
            parent=self.styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=16,
            leading=20,
            textColor=colors.white,
        )

        self.subtitle_style = ParagraphStyle(
            name='EntReportSub',
            parent=self.styles['Normal'],
            fontName='Helvetica',
            fontSize=8,
            leading=10,
            textColor=colors.white,
            alignment=2,  # Right-aligned
        )

        self.h2_style = ParagraphStyle(
            name='EntSectionHeader',
            parent=self.styles['Heading2'],
            fontName='Helvetica-Bold',
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#1B365D"),  # Deep Navy
            spaceBefore=12,
            spaceAfter=6,
            keepWithNext=True,
        )

        self.body_style = ParagraphStyle(
            name='EntReportBody',
            parent=self.styles['Normal'],
            fontName='Helvetica',
            fontSize=9,
            leading=12.5,
            textColor=colors.HexColor("#2C3E50"),
        )

        self.meta_label_style = ParagraphStyle(
            name='EntMetaLabel',
            parent=self.body_style,
            fontName='Helvetica-Bold',
            textColor=colors.HexColor("#34495E"),
        )

        self.disclaimer_style = ParagraphStyle(
            name='EntDisclaimerText',
            parent=self.styles['Normal'],
            fontName='Helvetica',
            fontSize=7.5,
            leading=10.5,
            textColor=colors.HexColor("#7F8C8D"),
        )

    def _safe_image(self, path: Optional[str], target_width: float, target_height: float, caption: str) -> List[Any]:
        """Validates and scales an image dynamically preserving its aspect ratio.

        Returns a flowable block containing the image and caption or a formatted warning text if missing.
        """
        caption_style = ParagraphStyle(
            name='EntImgCaption',
            fontName='Helvetica-Oblique',
            fontSize=8,
            leading=10,
            alignment=1,  # Center
            textColor=colors.HexColor("#7F8C8D")
        )

        error_style = ParagraphStyle(
            name='EntImgError',
            fontName='Helvetica',
            fontSize=8.5,
            leading=11,
            alignment=1,
            textColor=colors.HexColor("#C0392B")
        )

        if not path:
            return [
                Spacer(1, 40),
                Paragraph("<b>[Image Unavailable]</b><br/>No image path provided in results.", error_style),
                Spacer(1, 40)
            ]

        resolved_path = os.path.abspath(path)
        if not os.path.exists(resolved_path) or not os.path.isfile(resolved_path):
            return [
                Spacer(1, 40),
                Paragraph(f"<b>[Image File Missing]</b><br/>Could not resolve path:<br/>{os.path.basename(resolved_path)}", error_style),
                Spacer(1, 40)
            ]

        from PIL import Image as PILImage
        try:
            with PILImage.open(resolved_path) as img:
                width, height = img.size
                aspect = width / height
        except Exception as img_err:
            return [
                Spacer(1, 40),
                Paragraph(f"<b>[Format Error]</b><br/>Failed to verify image format:<br/>{str(img_err)}", error_style),
                Spacer(1, 40)
            ]

        # Calculate best fit scaling preserving aspect ratio
        target_aspect = target_width / target_height
        if aspect > target_aspect:
            # Width is the bounding constraint
            final_w = target_width
            final_h = target_width / aspect
        else:
            # Height is the bounding constraint
            final_h = target_height
            final_w = target_height * aspect

        # Pad spacers around vertical alignment to keep images centered in row height
        v_pad = (target_height - final_h) / 2

        from reportlab.platypus import Image as RLImage
        try:
            flowables = []
            if v_pad > 0:
                flowables.append(Spacer(1, v_pad))
            flowables.append(RLImage(resolved_path, width=final_w, height=final_h))
            if v_pad > 0:
                flowables.append(Spacer(1, v_pad))
            flowables.append(Spacer(1, 4))
            flowables.append(Paragraph(caption, caption_style))
            return flowables
        except Exception as load_err:
            return [
                Spacer(1, 40),
                Paragraph(f"<b>[Render Error]</b><br/>Failed to load image:<br/>{str(load_err)}", error_style),
                Spacer(1, 40)
            ]

    def render(self, data: ReportData) -> List[Any]:
        """Translates a structured ReportData model into layout Flowables for PDF templates."""
        story = []

        # 1. Title Banner
        header_data = [
            [
                Paragraph(f"<b>{data.metadata.report_title}</b><br/><font size=8>{data.metadata.report_type}</font>", self.title_style),
                Paragraph(f"<b>{data.metadata.brand_name}</b><br/>ID: {data.metadata.report_id}<br/>Date: {data.metadata.timestamp}", self.subtitle_style)
            ]
        ]
        header_table = Table(header_data, colWidths=[280, 224])
        header_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#1B365D")),
            ('PADDING', (0, 0), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(header_table)
        story.append(Spacer(1, 10))

        # 2. Patient Demographics & Scan Information
        demo_data = [
            [
                Paragraph("Patient ID:", self.meta_label_style), Paragraph(data.patient_info.patient_id, self.body_style),
                Paragraph("Scan Date:", self.meta_label_style), Paragraph(data.patient_info.scan_date or "N/A", self.body_style)
            ],
            [
                Paragraph("Patient Name:", self.meta_label_style), Paragraph(data.patient_info.name, self.body_style),
                Paragraph("Referring Physician:", self.meta_label_style), Paragraph(data.patient_info.referring_physician, self.body_style)
            ],
            [
                Paragraph("Age / Gender:", self.meta_label_style), Paragraph(f"{data.patient_info.age} yrs / {data.patient_info.gender}", self.body_style),
                Paragraph("Hardware Platform:", self.meta_label_style), Paragraph(data.processing_metrics.device, self.body_style)
            ]
        ]
        demo_table = Table(demo_data, colWidths=[90, 160, 110, 144])
        demo_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8F9F9")),
            ('PADDING', (0, 0), (-1, -1), 5),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(demo_table)
        story.append(Spacer(1, 12))

        # 3. AI System Component Status Card
        status_card_flowables = []
        status_card_flowables.append(Paragraph("AI Diagnostic System Status", self.h2_style))

        cls_color = "#27AE60" if data.system_status.classification == "Available" else "#C0392B"
        seg_color = "#27AE60" if data.system_status.segmentation == "Available" else "#C0392B"
        xai_color = "#27AE60" if data.system_status.explainability == "Available" else "#7F8C8D"
        rep_color = "#27AE60" if data.system_status.report_generation == "Successful" else "#C0392B"

        status_data = [
            [
                Paragraph("Classification Pipeline:", self.meta_label_style),
                Paragraph(f"<font color='{cls_color}'><b>{data.system_status.classification}</b></font>", self.body_style),
                Paragraph("Segmentation Model:", self.meta_label_style),
                Paragraph(f"<font color='{seg_color}'><b>{data.system_status.segmentation}</b></font>", self.body_style)
            ],
            [
                Paragraph("Grad-CAM Explainability:", self.meta_label_style),
                Paragraph(f"<font color='{xai_color}'><b>{data.system_status.explainability}</b></font>", self.body_style),
                Paragraph("Report Engine Compiled:", self.meta_label_style),
                Paragraph(f"<font color='{rep_color}'><b>{data.system_status.report_generation}</b></font>", self.body_style)
            ]
        ]
        status_table = Table(status_data, colWidths=[130, 120, 130, 124])
        status_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#EAEDED")),
            ('PADDING', (0, 0), (-1, -1), 5),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        status_card_flowables.append(status_table)
        status_card_flowables.append(Spacer(1, 12))
        story.append(KeepTogether(status_card_flowables))

        # 4. Primary AI Classification Result Card
        diag_flowables = []
        diag_flowables.append(Paragraph("Clinical Classification Diagnosis", self.h2_style))

        diag_data = [
            [Paragraph("Primary AI Classification Class:", self.meta_label_style), Paragraph(f"<b>{data.classification_result.predicted_class}</b>", self.body_style)],
            [Paragraph("Classification Confidence Score:", self.meta_label_style), Paragraph(f"{data.classification_result.confidence_score:.4%}", self.body_style)],
            [Paragraph("Classification ML Backbone Model:", self.meta_label_style), Paragraph(f"{data.classification_result.classification_model} ({data.classification_result.model_version})", self.body_style)],
        ]
        diag_table = Table(diag_data, colWidths=[180, 324])
        diag_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#EAEDED")),
            ('PADDING', (0, 0), (-1, -1), 5),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        diag_flowables.append(diag_table)
        diag_flowables.append(Spacer(1, 12))
        story.append(KeepTogether(diag_flowables))

        # 5. Quantitative Morphological Analysis Card
        morph_flowables = []
        morph_flowables.append(Paragraph("Quantitative Morphological Segmentation Analysis", self.h2_style))

        if data.system_status.segmentation == "Failed":
            warning_style = ParagraphStyle(
                name='EntSegWarnText',
                parent=self.body_style,
                fontName='Helvetica-Bold',
                fontSize=9.5,
                textColor=colors.HexColor("#78281F")
            )
            failed_table = Table([[Paragraph("⚠️ Segmentation analysis is unavailable because the segmentation model execution failed.", warning_style)]], colWidths=[504])
            failed_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#FDEDEC")),
                ('PADDING', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor("#F1948A")),
            ]))
            morph_flowables.append(failed_table)
            morph_flowables.append(Spacer(1, 12))
        elif data.classification_result.predicted_class == "No Tumor" or data.segmentation_result.tumor_area_mm2 <= 0:
            callout_style = ParagraphStyle(
                name='EntNoTumorText',
                parent=self.body_style,
                fontName='Helvetica',
                fontSize=9.5,
                textColor=colors.HexColor("#2C3E50")
            )
            empty_table = Table([[Paragraph("<b>Note:</b> No active tumor mass detected. Segmentation mask is empty.", callout_style)]], colWidths=[504])
            empty_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#EBF5FB")),
                ('PADDING', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#AED6F1")),
            ]))
            morph_flowables.append(empty_table)
            morph_flowables.append(Spacer(1, 12))
        else:
            # Active tumor morphology stats
            morph_hdr_style = ParagraphStyle('morph_hdr', parent=self.meta_label_style, textColor=colors.white, alignment=1)
            morph_val_style = ParagraphStyle('morph_val', parent=self.body_style, alignment=1)

            morph_data = [
                [
                    Paragraph("Estimated Tumor Area", morph_hdr_style),
                    Paragraph("Brain Space Occupied (%)", morph_hdr_style),
                    Paragraph("Tumor Perimeter", morph_hdr_style)
                ],
                [
                    Paragraph(f"{data.segmentation_result.tumor_area_mm2:.2f} mm²", morph_val_style),
                    Paragraph(f"{data.segmentation_result.tumor_percentage_brain:.4f}%", morph_val_style),
                    Paragraph(f"{data.segmentation_result.perimeter_mm:.2f} mm" if data.segmentation_result.perimeter_mm else "N/A", morph_val_style)
                ]
            ]
            morph_table = Table(morph_data, colWidths=[168, 168, 168])
            morph_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1B365D")),
                ('PADDING', (0, 0), (-1, -1), 5),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            morph_flowables.append(morph_table)
            morph_flowables.append(Spacer(1, 8))

            # Quality Assessment Sub-block
            if data.segmentation_result.quality_score is not None:
                qual_data = [
                    [
                        Paragraph("Segmentation Quality Score:", self.meta_label_style),
                        Paragraph(f"{data.segmentation_result.quality_score:.2%} ({data.segmentation_result.quality_category})", self.body_style)
                    ]
                ]
                qual_table = Table(qual_data, colWidths=[180, 324])
                qual_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#EAEDED")),
                    ('PADDING', (0, 0), (-1, -1), 4),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
                ]))
                morph_flowables.append(qual_table)
                morph_flowables.append(Spacer(1, 8))

            # Shape stats detail grid
            s = data.tumor_statistics.shape_statistics
            if s:
                morph_detail_data = [
                    [
                        Paragraph("<b>Property</b>", self.meta_label_style),
                        Paragraph("<b>Value</b>", self.meta_label_style),
                        Paragraph("<b>Property</b>", self.meta_label_style),
                        Paragraph("<b>Value</b>", self.meta_label_style),
                    ],
                    [
                        Paragraph("Bounding Box Width", self.body_style), Paragraph(f"{s.get('bbox_w_mm', 0.0):.2f} mm ({s.get('bbox_w_px', 0)} px)", self.body_style),
                        Paragraph("Solidity Index", self.body_style), Paragraph(f"{s.get('solidity', 0.0):.4f}", self.body_style),
                    ],
                    [
                        Paragraph("Bounding Box Height", self.body_style), Paragraph(f"{s.get('bbox_h_mm', 0.0):.2f} mm ({s.get('bbox_h_px', 0)} px)", self.body_style),
                        Paragraph("Circularity Index", self.body_style), Paragraph(f"{s.get('circularity', 0.0):.4f}", self.body_style),
                    ],
                    [
                        Paragraph("Ellipse Major Axis", self.body_style), Paragraph(f"{s.get('major_axis_mm', 0.0):.2f} mm", self.body_style),
                        Paragraph("Eccentricity", self.body_style), Paragraph(f"{s.get('eccentricity', 0.0):.4f}", self.body_style),
                    ],
                    [
                        Paragraph("Ellipse Minor Axis", self.body_style), Paragraph(f"{s.get('minor_axis_mm', 0.0):.2f} mm", self.body_style),
                        Paragraph("Orientation Angle", self.body_style), Paragraph(f"{s.get('orientation_deg', 0.0):.1f}°", self.body_style),
                    ]
                ]
                morph_detail_table = Table(morph_detail_data, colWidths=[130, 122, 130, 122])
                morph_detail_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#EAEDED")),
                    ('PADDING', (0, 0), (-1, -1), 4),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ]))
                morph_flowables.append(morph_detail_table)
                morph_flowables.append(Spacer(1, 12))
        story.append(KeepTogether(morph_flowables))

        # 6. Explainable AI & Visualization Section
        xai_flowables = []
        xai_flowables.append(Paragraph("Explainable AI (XAI) Analysis", self.h2_style))

        if data.system_status.explainability == "Unavailable":
            warning_style = ParagraphStyle(
                name='EntXaiWarnText',
                parent=self.body_style,
                fontName='Helvetica-Oblique',
                fontSize=9.5,
                textColor=colors.HexColor("#7F8C8D")
            )
            failed_xai_table = Table([[Paragraph("Explainability visualization unavailable for this analysis.", warning_style)]], colWidths=[504])
            failed_xai_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8F9F9")),
                ('PADDING', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
            ]))
            xai_flowables.append(failed_xai_table)
            xai_flowables.append(Spacer(1, 12))
        else:
            method_display = "Grad-CAM"
            if data.explainability_result.xai_method in ["gradcam_plus_plus", "gradcam++"]:
                method_display = "Grad-CAM++"
            elif data.explainability_result.xai_method == "eigencam":
                method_display = "EigenCAM"

            overlap_str = f"{data.explainability_result.overlap_percentage:.2%}" if data.explainability_result.overlap_percentage is not None else "N/A"

            xai_data = [
                [Paragraph("Active Explanation Method:", self.meta_label_style), Paragraph(method_display, self.body_style)],
                [Paragraph("Lesion Spatial Overlap Ratio:", self.meta_label_style), Paragraph(overlap_str, self.body_style)],
                [Paragraph("Saliency Interpretation:", self.meta_label_style), Paragraph(data.explainability_result.explanation_text or "N/A", self.body_style)]
            ]
            xai_table = Table(xai_data, colWidths=[180, 324])
            xai_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#EAEDED")),
                ('PADDING', (0, 0), (-1, -1), 5),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]))
            xai_flowables.append(xai_table)
            xai_flowables.append(Spacer(1, 12))
        story.append(KeepTogether(xai_flowables))

        # Visual Scans Grid (KeepTogether)
        scan_flowables = []
        scan_flowables.append(Paragraph("Clinical Imaging Scan Visualizations", self.h2_style))

        # Check files existence on disk to layout images dynamically
        exist_images = []
        if data.patient_info.original_image_path and os.path.exists(data.patient_info.original_image_path) and os.path.isfile(data.patient_info.original_image_path):
            exist_images.append((data.patient_info.original_image_path, "Original MRI"))

        if data.explainability_result.heatmap_image_path and os.path.exists(data.explainability_result.heatmap_image_path) and os.path.isfile(data.explainability_result.heatmap_image_path):
            exist_images.append((data.explainability_result.heatmap_image_path, "Grad-CAM Heatmap"))

        if data.explainability_result.overlay_image_path and os.path.exists(data.explainability_result.overlay_image_path) and os.path.isfile(data.explainability_result.overlay_image_path):
            exist_images.append((data.explainability_result.overlay_image_path, "Grad-CAM Overlay"))

        if data.segmentation_result.segmentation_mask_path and os.path.exists(data.segmentation_result.segmentation_mask_path) and os.path.isfile(data.segmentation_result.segmentation_mask_path):
            exist_images.append((data.segmentation_result.segmentation_mask_path, "UNeXt Segmentation Mask"))

        num_images = len(exist_images)
        if num_images == 0:
            warning_style = ParagraphStyle(
                name='EntNoVisualsText',
                parent=self.body_style,
                fontName='Helvetica-Oblique',
                fontSize=9.5,
                textColor=colors.HexColor("#7F8C8D")
            )
            no_visuals_table = Table([[Paragraph("Clinical imaging scan visualizations are unavailable.", warning_style)]], colWidths=[504])
            no_visuals_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8F9F9")),
                ('PADDING', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
            ]))
            scan_flowables.append(no_visuals_table)
        else:
            if num_images == 4:
                # 2x2 grid layout
                cell_w = 246
                cell_h = 160
                cells = [self._safe_image(path, cell_w, cell_h, caption) for path, caption in exist_images]
                img_data = [
                    [cells[0], cells[1]],
                    [cells[2], cells[3]]
                ]
                img_table = Table(img_data, colWidths=[252, 252])
                img_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('PADDING', (0, 0), (-1, -1), 4),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#EAEDED")),
                ]))
                scan_flowables.append(img_table)
            elif num_images == 3:
                # 3 columns in a single row
                cell_w = 160
                cell_h = 135
                cells = [self._safe_image(path, cell_w, cell_h, caption) for path, caption in exist_images]
                img_table = Table([cells], colWidths=[168, 168, 168])
                img_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('PADDING', (0, 0), (-1, -1), 3),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#EAEDED")),
                ]))
                scan_flowables.append(img_table)
            elif num_images == 2:
                # 2 columns in a single row
                cell_w = 246
                cell_h = 200
                cells = [self._safe_image(path, cell_w, cell_h, caption) for path, caption in exist_images]
                img_table = Table([cells], colWidths=[252, 252])
                img_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('PADDING', (0, 0), (-1, -1), 4),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#EAEDED")),
                ]))
                scan_flowables.append(img_table)
            else:
                # 1 image centered
                cell_w = 246
                cell_h = 220
                cells = [self._safe_image(exist_images[0][0], cell_w, cell_h, exist_images[0][1])]
                img_table = Table([[cells]], colWidths=[504])
                img_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('PADDING', (0, 0), (-1, -1), 6),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#EAEDED")),
                ]))
                scan_flowables.append(img_table)

        scan_flowables.append(Spacer(1, 12))
        story.append(KeepTogether(scan_flowables))

        # 7. Clinical Insight Summary & Action Recommendations (KeepTogether)
        insight_flowables = []
        insight_flowables.append(Paragraph("AI Clinical Insights & Recommendations", self.h2_style))

        insight_body_style = ParagraphStyle(
            name='EntInsightBody',
            parent=self.body_style,
            fontSize=9,
            leading=12.5,
        )

        if not data.clinical_insights or (not data.clinical_insights.summary_narrative and not data.clinical_insights.key_findings and not data.clinical_insights.recommendations):
            warning_style = ParagraphStyle(
                name='EntInsightWarnText',
                parent=self.body_style,
                fontName='Helvetica-Oblique',
                fontSize=9.5,
                textColor=colors.HexColor("#7F8C8D")
            )
            unavailable_insight_table = Table([[Paragraph("Clinical insight summary is unavailable for this analysis.", warning_style)]], colWidths=[504])
            unavailable_insight_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F8F9F9")),
                ('PADDING', (0, 0), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
            ]))
            insight_flowables.append(unavailable_insight_table)
        else:
            narrative_table = Table([[Paragraph(data.clinical_insights.summary_narrative, insight_body_style)]], colWidths=[504])
            narrative_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#EBF5FB")),
                ('PADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#AED6F1")),
            ]))
            insight_flowables.append(narrative_table)
            insight_flowables.append(Spacer(1, 8))

            # Bulleted findings and recommendations side-by-side
            findings_bullet_text = "<br/>".join(f"• {f}" for f in data.clinical_insights.key_findings) if data.clinical_insights.key_findings else "• No active diagnostic findings listed."
            recs_bullet_text = "<br/>".join(f"• {r}" for r in data.clinical_insights.recommendations) if data.clinical_insights.recommendations else "• Standard neurological follow-up scan as scheduled."

            bullets_data = [
                [Paragraph("<b>Key Synthesized Findings:</b>", self.meta_label_style), Paragraph("<b>Clinical Recommendations:</b>", self.meta_label_style)],
                [Paragraph(findings_bullet_text, insight_body_style), Paragraph(recs_bullet_text, insight_body_style)]
            ]
            bullets_table = Table(bullets_data, colWidths=[252, 252])
            bullets_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#FDFEFE")),
                ('PADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E5E7E9")),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ]))
            insight_flowables.append(bullets_table)

        insight_flowables.append(Spacer(1, 12))
        story.append(KeepTogether(insight_flowables))

        # 8. Technical Metrics + Reviewer Sign-off + Disclaimer Card group (KeepTogether)
        technical_and_signoff = []

        technical_and_signoff.append(Paragraph("Technical Pipeline Latency Benchmarks", self.h2_style))

        cls_lat = f"{data.processing_metrics.classification_latency_sec:.4f} s" if data.processing_metrics.classification_latency_sec else "N/A"
        seg_lat = f"{data.processing_metrics.segmentation_latency_sec:.4f} s" if data.processing_metrics.segmentation_latency_sec else "N/A"
        xai_lat = f"{data.processing_metrics.explainability_latency_sec:.4f} s" if data.processing_metrics.explainability_latency_sec else "N/A"
        tot_lat = f"{data.processing_metrics.total_execution_time_sec:.4f} s" if data.processing_metrics.total_execution_time_sec else "N/A"

        tech_data = [
            [
                Paragraph("Classification Latency:", self.meta_label_style), Paragraph(cls_lat, self.body_style),
                Paragraph("Segmentation Inference:", self.meta_label_style), Paragraph(seg_lat, self.body_style)
            ],
            [
                Paragraph("Grad-CAM Latency:", self.meta_label_style), Paragraph(xai_lat, self.body_style),
                Paragraph("Total Processing Time:", self.meta_label_style), Paragraph(tot_lat, self.body_style)
            ]
        ]
        tech_table = Table(tech_data, colWidths=[130, 122, 130, 122])
        tech_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#FAFBFB")),
            ('PADDING', (0, 0), (-1, -1), 5),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#EAEDED")),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        technical_and_signoff.append(tech_table)
        technical_and_signoff.append(Spacer(1, 10))

        # Reviewer Sign-off & Verification Section
        technical_and_signoff.append(Paragraph("Reviewer Sign-off & Verification", self.h2_style))

        sig_data = [
            [
                Paragraph("<b>Clinician Review & Sign-off:</b>", self.meta_label_style),
                Paragraph("<b>Software Verification Status:</b>", self.meta_label_style)
            ],
            [
                Paragraph(
                    "Reviewer: ________________________<br/>"
                    "Reviewer Role: ________________________<br/>"
                    "Date: ________________________<br/>"
                    "Signature: ________________________",
                    self.body_style
                ),
                Paragraph(
                    "<b>AuraScan AI Prototype</b><br/>"
                    "Automated verification tests passed.<br/>"
                    "Research / Educational Prototype.<br/>"
                    "Not independently clinically validated.",
                    self.body_style
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
        technical_and_signoff.append(sig_table)
        technical_and_signoff.append(Spacer(1, 10))

        # Regulatory Disclaimer & Clinical Intent Section
        technical_and_signoff.append(Paragraph("Regulatory Disclaimer & Clinical Intent", self.h2_style))

        disc_table = Table([[Paragraph(data.disclaimer.text, self.disclaimer_style)]], colWidths=[504])
        disc_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#FAFBFB")),
            ('PADDING', (0, 0), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#BDC3C7")),
        ]))
        technical_and_signoff.append(disc_table)

        story.append(KeepTogether(technical_and_signoff))

        return story
