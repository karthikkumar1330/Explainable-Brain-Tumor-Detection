from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from prediction_history.domain.entities import HistorySearchCriteria, PredictionSummary


class IPredictionHistoryRepository(ABC):
    """Interface for repository querying prediction history records and report locations."""

    @abstractmethod
    def search_history(self, criteria: HistorySearchCriteria) -> List[PredictionSummary]:
        """Searches prediction summaries matching the filter criteria.

        Args:
            criteria: Search criteria constraints.

        Returns:
            A list of PredictionSummary dataclasses.
        """
        pass

    @abstractmethod
    def get_report_paths(self, report_id: int) -> Optional[Tuple[str, str, str]]:
        """Retrieves file paths for Markdown, JSON, and PDF reports for a Report ID.

        Args:
            report_id: Database clinical report identifier.

        Returns:
            A tuple of (markdown_path, json_path, pdf_path) if found, otherwise None.
        """
        pass
