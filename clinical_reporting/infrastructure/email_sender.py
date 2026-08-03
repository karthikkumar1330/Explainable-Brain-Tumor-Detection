import os
import logging
from typing import Optional
from clinical_reporting.domain.entities import ClinicalReport


class ClinicalEmailReporter:
    """Formats and exports structured clinical AI email reports with educational disclaimers."""

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger("email_reporter")

    def generate_email_html(self, report: ClinicalReport) -> str:
        """Constructs a beautifully styled HTML email template of clinical findings."""
        ci = report.clinical_insight
        insight_html = ""
        if ci is not None:
            findings_li = "".join(f"<li style='margin-bottom: 6px;'>{f}</li>" for f in ci.key_findings)
            recs_li = "".join(f"<li style='margin-bottom: 6px;'>{r}</li>" for r in ci.recommendations)
            
            insight_html = f"""
            <div style="margin-top: 20px; border: 1px solid #AED6F1; border-radius: 6px; overflow: hidden; font-family: sans-serif;">
                <div style="background-color: #EBF5FB; padding: 12px; font-weight: bold; color: #2980B9;">
                    AI Clinical Insights Summary
                </div>
                <div style="padding: 15px; background-color: #FFFFFF; font-size: 14px; line-height: 1.5; color: #2C3E50;">
                    <p style="margin-top: 0;">{ci.summary_narrative}</p>
                    
                    <div style="display: flex; gap: 20px; margin-top: 15px;">
                        <div style="flex: 1; min-width: 200px;">
                            <strong style="color: #2C3E50;">Key Findings:</strong>
                            <ul style="margin: 8px 0; padding-left: 20px; color: #566573;">{findings_li}</ul>
                        </div>
                        <div style="flex: 1; min-width: 200px;">
                            <strong style="color: #2C3E50;">Clinical Recommendations:</strong>
                            <ul style="margin: 8px 0; padding-left: 20px; color: #566573;">{recs_li}</ul>
                        </div>
                    </div>
                </div>
            </div>
            """

        # Build warnings list
        warnings_html = ""
        if getattr(report, "quality_warnings", None):
            warn_items = "".join(f"<div style='color: #78281F; font-size: 13px; font-weight: bold; margin-bottom: 4px;'>⚠️ {w}</div>" for w in report.quality_warnings)
            warnings_html = f"""
            <div style="background-color: #FDEDEC; border: 1px solid #F1948A; border-radius: 6px; padding: 12px; margin-top: 15px; font-family: sans-serif;">
                <strong style="color: #78281F; font-size: 14px;">Diagnostic Coherence Alerts:</strong>
                <div style="margin-top: 6px;">{warn_items}</div>
            </div>
            """

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Clinical Brain MRI AI Report</title>
</head>
<body style="margin: 0; padding: 20px; background-color: #F4F6F7; font-family: Arial, sans-serif;">
    <div style="max-width: 600px; margin: 0 auto; background-color: #FFFFFF; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border: 1px solid #E5E8E8;">
        <!-- Header -->
        <div style="background-color: #2C3E50; padding: 20px; text-align: center; color: #FFFFFF;">
            <h1 style="margin: 0; font-size: 20px; letter-spacing: 1px;">INTEGRATED CLINICAL BRAIN MRI REPORT</h1>
            <span style="font-size: 12px; color: #BDC3C7;">AUTOMATED AI DIAGNOSTIC PIPELINE</span>
        </div>

        <div style="padding: 20px;">
            <!-- Demographics Table -->
            <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 13px;">
                <tr style="background-color: #F8F9F9; border-bottom: 1px solid #E5E8E8;">
                    <td style="padding: 8px; font-weight: bold; color: #34495E; width: 30%;">Patient ID:</td>
                    <td style="padding: 8px; color: #2C3E50;">{report.patient_info.patient_id}</td>
                    <td style="padding: 8px; font-weight: bold; color: #34495E; width: 30%;">Scan Date:</td>
                    <td style="padding: 8px; color: #2C3E50;">{report.patient_info.scan_date}</td>
                </tr>
                <tr style="background-color: #FFFFFF; border-bottom: 1px solid #E5E8E8;">
                    <td style="padding: 8px; font-weight: bold; color: #34495E;">Patient Name:</td>
                    <td style="padding: 8px; color: #2C3E50;">{report.patient_info.name}</td>
                    <td style="padding: 8px; font-weight: bold; color: #34495E;">Ref. Physician:</td>
                    <td style="padding: 8px; color: #2C3E50;">{report.patient_info.ref_physician}</td>
                </tr>
                <tr style="background-color: #F8F9F9;">
                    <td style="padding: 8px; font-weight: bold; color: #34495E;">Age / Gender:</td>
                    <td style="padding: 8px; color: #2C3E50;">{report.patient_info.age} yrs / {report.patient_info.gender}</td>
                    <td style="padding: 8px; font-weight: bold; color: #34495E;">Platform Device:</td>
                    <td style="padding: 8px; color: #2C3E50;">{report.processing_summary.device.upper()}</td>
                </tr>
            </table>

            {warnings_html}

            <!-- Primary Diagnosis -->
            <div style="background-color: #EAEDED; border-left: 6px solid #34495E; padding: 15px; margin-top: 15px; border-radius: 4px;">
                <h3 style="margin: 0; font-size: 14px; color: #34495E;">Primary Classification:</h3>
                <span style="font-size: 18px; font-weight: bold; color: #1A5276;">{report.classification.class_name}</span>
                <br/>
                <span style="font-size: 13px; color: #7F8C8D;">Confidence Score: <strong>{report.classification.confidence_score:.2%}</strong></span>
            </div>

            {insight_html}

            <!-- PDF Notice -->
            <div style="margin-top: 25px; padding: 10px; background-color: #FCF3CF; border: 1px solid #F5CBA7; border-radius: 4px; font-size: 12px; color: #7E5109; text-align: center;">
                The complete visual overlays and diagnostic PDF report are attached to the official record.
            </div>
        </div>

        <!-- Educational Disclaimer Footer -->
        <div style="background-color: #EBEDEF; padding: 15px; border-top: 1px solid #E5E8E8; font-size: 11px; color: #7F8C8D; line-height: 1.4;">
            <strong>EDUCATIONAL DISCLAIMER:</strong> {ci.disclaimer if ci else "For educational use only. Always correlate with licensed physician reviews."}
        </div>
    </div>
</body>
</html>
"""
        return html

    def send_report_email(self, report: ClinicalReport, recipient_email: str, output_dir: str) -> str:
        """Generates, sends (logs), and saves a clinical report email archive to disk."""
        os.makedirs(output_dir, exist_ok=True)
        email_filename = f"{report.patient_info.patient_id}_email_report.html"
        email_path = os.path.join(output_dir, email_filename)

        html_content = self.generate_email_html(report)
        with open(email_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        self.logger.info(
            f"Mock Clinical Email successfully sent to '{recipient_email}'. "
            f"HTML email payload persisted locally for audit: {email_path}"
        )
        return email_path
