import logging
from clinical_reporting.domain.entities import ClinicalReport
from clinical_reporting.domain.interfaces import IClinicalReportGenerator


class GenerateIntegratedReportUseCase:
    """Use case to orchestrate clinical report generation."""

    def __init__(
        self,
        report_generator: IClinicalReportGenerator,
        logger: logging.Logger,
    ) -> None:
        """Initializes the report generation use case.

        Args:
            report_generator: Generator implementing formatting and file I/O.
            logger: Logger instance.
        """
        self.report_generator = report_generator
        self.logger = logger

    def execute(self, report: ClinicalReport, output_dir: str) -> tuple[str, str]:
        """Orchestrates saving the clinical findings to Markdown and JSON formats.

        Args:
            report: The combined ClinicalReport entity.
            output_dir: Destination folder.

        Returns:
            A tuple of (markdown_file_path, json_file_path).
        """
        self.logger.info(f"Generating integrated clinical report for patient: {report.patient_info.patient_id}")

        try:
            md_path, json_path = self.report_generator.generate(report, output_dir)
            self.logger.info(
                f"Integrated clinical report successfully generated.\n"
                f"  - Markdown: {md_path}\n"
                f"  - JSON: {json_path}"
            )
            return md_path, json_path
        except Exception as e:
            self.logger.error(f"Failed to generate clinical report: {e}")
            raise e
