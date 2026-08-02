from abc import ABC, abstractmethod
from clinical_reporting.domain.entities import ClinicalReport


class IClinicalReportGenerator(ABC):
    """Interface for generating formatted clinical reports in multiple formats."""

    @abstractmethod
    def generate(self, report: ClinicalReport, output_dir: str) -> tuple[str, str]:
        """Formats and saves the integrated clinical findings to the disk.

        Args:
            report: The aggregated ClinicalReport entity.
            output_dir: Directory where reports should be written.

        Returns:
            A tuple of (saved_markdown_path, saved_json_path).
        """
        pass
