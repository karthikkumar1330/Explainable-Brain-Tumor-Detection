from abc import ABC, abstractmethod
from typing import Tuple
from clinical_reporting.domain.entities import ClinicalReport


class IClinicalReportGenerator(ABC):
    """Interface for generating formatted clinical reports in multiple formats."""

    @abstractmethod
    def generate(self, report: ClinicalReport, output_dir: str) -> Tuple[str, str, str]:
        """Formats and saves the integrated clinical findings to the disk.

        Args:
            report: The aggregated ClinicalReport entity.
            output_dir: Directory where reports should be written.

        Returns:
            A tuple of (saved_markdown_path, saved_json_path, saved_pdf_path).
        """
        pass
