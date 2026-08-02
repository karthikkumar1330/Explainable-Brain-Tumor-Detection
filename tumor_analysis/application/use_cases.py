import logging
import numpy as np
from typing import Optional
from tumor_analysis.domain.entities import TumorAnalysisResult, ClinicalReportData, SeverityLevel
from tumor_analysis.domain.interfaces import ITumorAnalyzer


class AnalyzeTumorUseCase:
    """Use case to run tumor area analysis and generate clinical report data."""

    def __init__(
        self,
        analyzer: ITumorAnalyzer,
        logger: logging.Logger,
    ) -> None:
        """Initializes the use case.

        Args:
            analyzer: Implementation of the tumor analyzer interface.
            logger: Logger instance.
        """
        self.analyzer = analyzer
        self.logger = logger

    def execute(
        self,
        mask: np.ndarray,
        patient_id: str,
        tumor_class: str = "Unknown",
        original_image: Optional[np.ndarray] = None,
        pixel_spacing_mm: float = 1.0,
    ) -> ClinicalReportData:
        """Executes the tumor analysis and generates clinical report details.

        Args:
            mask: The segmentation mask (H, W) or (H, W, C).
            patient_id: Identifier for the patient / scan.
            tumor_class: Classification label (e.g. Glioma).
            original_image: Optional original image to compute brain coverage percentage.
            pixel_spacing_mm: The physical spacing factor.

        Returns:
            A ClinicalReportData object.
        """
        self.logger.info(f"Starting tumor area analysis for Patient: {patient_id}")

        try:
            analysis_result = self.analyzer.analyze(
                mask=mask,
                original_image=original_image,
                pixel_spacing_mm=pixel_spacing_mm
            )
            self.logger.info("Tumor area calculations completed successfully.")
        except Exception as e:
            self.logger.error(f"Error during tumor mask analysis: {e}")
            raise e

        # Generate clinical descriptions based on severity and class
        severity = analysis_result.severity_level
        clinical_notes, recommendations = self._generate_clinical_guidelines(
            tumor_class, severity, analysis_result
        )

        return ClinicalReportData(
            patient_id=patient_id,
            tumor_class=tumor_class,
            analysis=analysis_result,
            clinical_notes=clinical_notes,
            recommendations=recommendations
        )

    def _generate_clinical_guidelines(
        self, tumor_class: str, severity: SeverityLevel, result: TumorAnalysisResult
    ) -> tuple[str, str]:
        """Generates patient-specific clinical notes and recommendations based on severity."""
        if severity == SeverityLevel.NORMAL:
            notes = "No active tumor mass detected in the provided segmentation mask."
            recommendations = "Routine follow-up as per standard clinical protocol."
        elif severity == SeverityLevel.LOW:
            notes = (
                f"A small localized tumor mass corresponding to class '{tumor_class}' was identified, "
                f"occupying {result.tumor_percentage_brain:.4f}% of the estimated brain parenchyma."
            )
            recommendations = (
                "1. Schedule a baseline neurological examination.\n"
                "2. Perform a follow-up MRI in 3-6 months to monitor for growth or changes."
            )
        elif severity == SeverityLevel.MEDIUM:
            notes = (
                f"A moderate tumor mass of type '{tumor_class}' was identified, occupying "
                f"{result.tumor_percentage_brain:.4f}% of estimated brain parenchyma. "
                f"Physical tumor area is approximately {result.tumor_area_mm2:.2f} mm²."
            )
            recommendations = (
                "1. Refer to a neuro-oncology specialist for formal evaluation.\n"
                "2. Consider planning standard staging and metabolic profiling."
            )
        elif severity == SeverityLevel.HIGH:
            notes = (
                f"A large tumor mass of type '{tumor_class}' was identified, occupying "
                f"{result.tumor_percentage_brain:.4f}% of estimated brain parenchyma. "
                f"Physical tumor area is approximately {result.tumor_area_mm2:.2f} mm²."
            )
            recommendations = (
                "1. Urgent neurosurgical consultation recommended to evaluate resectability.\n"
                "2. Initiate corticosteroid therapy if symptoms of mass effect or edema are present.\n"
                "3. Multidisciplinary tumor board review."
            )
        else:  # CRITICAL
            notes = (
                f"Critical tumor mass of type '{tumor_class}' detected. "
                f"The tumor occupies a substantial portion ({result.tumor_percentage_brain:.4f}%) of "
                f"the estimated brain parenchyma (Area: {result.tumor_area_mm2:.2f} mm²), indicating "
                f"significant mass effect or extensive progression."
            )
            recommendations = (
                "1. Immediate emergency neurosurgical evaluation.\n"
                "2. Urgent admission for intensive symptom management and therapeutic planning.\n"
                "3. High-resolution MRI scan with contrast."
            )
        return notes, recommendations
