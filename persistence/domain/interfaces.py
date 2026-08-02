from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from clinical_reporting.domain.entities import ClinicalReport


class IPersistenceRepository(ABC):
    """Interface for database repository managing report storage and query histories."""

    @abstractmethod
    def initialize_db(self) -> None:
        """Initializes database schema, tables, and indices if they do not exist."""
        pass

    @abstractmethod
    def save_report(self, report: ClinicalReport) -> int:
        """Saves an integrated clinical report to database tables.

        Args:
            report: Aggregated report entity.

        Returns:
            The saved clinical report ID.
        """
        pass

    @abstractmethod
    def get_patient_history(self, patient_id: str) -> List[Dict[str, Any]]:
        """Retrieves history of MRI scan findings for a patient.

        Args:
            patient_id: Target patient identifier.

        Returns:
            A list of dictionary records containing diagnostic details.
        """
        pass

    @abstractmethod
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Calculates dashboard metrics across all persisted patients and scans.

        Returns:
            A dictionary containing distribution metrics.
        """
        pass
