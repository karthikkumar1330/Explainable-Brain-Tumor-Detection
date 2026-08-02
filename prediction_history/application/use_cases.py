import os
import shutil
import logging
from typing import List, Optional, Tuple
from prediction_history.domain.entities import HistorySearchCriteria, PredictionSummary
from prediction_history.domain.interfaces import IPredictionHistoryRepository


class SearchHistoryUseCase:
    """Use case to search database records matching demographic or timing criteria."""

    def __init__(self, repository: IPredictionHistoryRepository, logger: logging.Logger) -> None:
        self.repository = repository
        self.logger = logger

    def execute(self, criteria: HistorySearchCriteria) -> List[PredictionSummary]:
        self.logger.info(f"Executing search history with criteria: {criteria}")
        try:
            results = self.repository.search_history(criteria)
            self.logger.info(f"Search complete. Found {len(results)} matching records.")
            return results
        except Exception as e:
            self.logger.error(f"Search history execution failed: {e}")
            raise e


class RetrieveReportUseCase:
    """Use case to locate and copy/export generated report files."""

    def __init__(self, repository: IPredictionHistoryRepository, logger: logging.Logger) -> None:
        self.repository = repository
        self.logger = logger

    def execute(self, report_id: int, export_dir: Optional[str] = None) -> Tuple[str, str, str]:
        self.logger.info(f"Retrieving file paths for Report ID: {report_id}")
        paths = self.repository.get_report_paths(report_id)
        if not paths:
            self.logger.error(f"Clinical report record not found for Report ID: {report_id}")
            raise FileNotFoundError(f"No database records found for Report ID: {report_id}")

        md_path, json_path, pdf_path = paths

        # Verify physical file existence
        missing_files = []
        for path in [md_path, json_path, pdf_path]:
            if not os.path.exists(path):
                missing_files.append(path)

        if missing_files:
            self.logger.warning(f"Some physical report files are missing on disk: {missing_files}")

        # Export if requested
        if export_dir:
            os.makedirs(export_dir, exist_ok=True)
            self.logger.info(f"Exporting report files to: {export_dir}")
            exported_paths = []
            for path in [md_path, json_path, pdf_path]:
                if os.path.exists(path):
                    dest = os.path.join(export_dir, os.path.basename(path))
                    shutil.copy2(path, dest)
                    exported_paths.append(dest)
                else:
                    exported_paths.append("")
            return tuple(exported_paths)

        return md_path, json_path, pdf_path
