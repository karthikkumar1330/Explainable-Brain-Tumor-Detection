import argparse
import sys
import os
import logging
import subprocess
from prediction_history.domain.entities import HistorySearchCriteria
from prediction_history.infrastructure.repository import SQLitePredictionHistoryRepository
from prediction_history.application.use_cases import SearchHistoryUseCase, RetrieveReportUseCase


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the prediction history CLI."""
    parser = argparse.ArgumentParser(
        description="Clinical MRI Prediction History & Report Retrieval Tool"
    )
    # Search Filters
    parser.add_argument(
        "--patient-id",
        type=str,
        default=None,
        help="Search records by Patient ID (partial matching supported)",
    )
    parser.add_argument(
        "--report-id",
        type=int,
        default=None,
        help="Search record by exact Report ID",
    )
    parser.add_argument(
        "--scan-date",
        type=str,
        default=None,
        help="Search records by Scan Acquisition Date (e.g. YYYY-MM-DD)",
    )

    # Retrieval / Actions
    parser.add_argument(
        "--export-report",
        type=int,
        default=None,
        help="Export all report files (MD, JSON, PDF) for the specified Report ID",
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default="outputs/exported_reports",
        help="Target folder for exporting reports",
    )
    parser.add_argument(
        "--open-pdf",
        type=int,
        default=None,
        help="Launch the system default PDF viewer to open the report for the specified Report ID",
    )

    # DB Config
    parser.add_argument(
        "--db-path",
        type=str,
        default="outputs/clinical_reports.db",
        help="Path to the SQLite database file",
    )
    return parser.parse_args()


def display_summaries(summaries) -> None:
    """Prints a clean tabular listing of prediction summaries."""
    if not summaries:
        print("\nNo matching historical records found.")
        return

    print("\n" + "=" * 110)
    print(f"{'REPORT ID':<10} | {'PATIENT ID':<15} | {'PATIENT NAME':<20} | {'SCAN DATE':<12} | {'DIAGNOSIS':<12} | {'SEVERITY':<10} | {'CONFIDENCE':<10}")
    print("=" * 110)
    for s in summaries:
        conf_str = f"{s.confidence_score:.2%}"
        print(f"{s.report_id:<10} | {s.patient_id:<15} | {s.patient_name:<20} | {s.scan_date:<12} | {s.predicted_class:<12} | {s.rule_based_severity:<10} | {conf_str:<10}")
    print("=" * 110 + "\n")


def main() -> None:
    args = parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("prediction_history_cli")

    if not os.path.exists(args.db_path):
        logger.error(f"SQLite database file not found at: {args.db_path}")
        print(f"Error: Database file not found at {args.db_path}. Please run generate_clinical_report.py first.")
        sys.exit(1)

    # Wire Clean Architecture components
    repository = SQLitePredictionHistoryRepository(db_path=args.db_path, logger=logger)
    search_use_case = SearchHistoryUseCase(repository=repository, logger=logger)
    retrieve_use_case = RetrieveReportUseCase(repository=repository, logger=logger)

    # Action 1: Export Reports
    if args.export_report is not None:
        print(f"\nExporting clinical report files for Report ID: {args.export_report}...")
        try:
            md_path, json_path, pdf_path = retrieve_use_case.execute(
                report_id=args.export_report,
                export_dir=args.export_dir
            )
            print("\n" + "=" * 70)
            print("CLINICAL REPORT EXPORT COMPLETE")
            print("=" * 70)
            print(f"Destination Directory : {args.export_dir}")
            print(f"  - Markdown Report   : {md_path}")
            print(f"  - EHR JSON Payload  : {json_path}")
            print(f"  - Clinical PDF      : {pdf_path}")
            print("=" * 70 + "\n")
        except Exception as e:
            logger.error(f"Failed to export reports: {e}")
            sys.exit(1)
        sys.exit(0)

    # Action 2: Open PDF Report in default OS viewer
    if args.open_pdf is not None:
        print(f"\nRetrieving and opening clinical PDF report for Report ID: {args.open_pdf}...")
        try:
            md_path, json_path, pdf_path = retrieve_use_case.execute(report_id=args.open_pdf)
            if not pdf_path or not os.path.exists(pdf_path):
                print(f"Error: PDF report file does not exist physically at: {pdf_path}")
                sys.exit(1)
            
            print(f"Launching default viewer for: {pdf_path}")
            # Windows open file command: os.startfile()
            if hasattr(os, "startfile"):
                os.startfile(pdf_path)
            else:
                # Fallback for other operating systems (macOS / Linux)
                opener = "open" if sys.platform == "darwin" else "xdg-open"
                subprocess.call([opener, pdf_path])
                
            print("PDF launched successfully.\n")
        except Exception as e:
            logger.error(f"Failed to open PDF report: {e}")
            sys.exit(1)
        sys.exit(0)

    # Action 3: General Search & Filter listing
    criteria = HistorySearchCriteria(
        patient_id=args.patient_id,
        report_id=args.report_id,
        scan_date=args.scan_date
    )
    try:
        summaries = search_use_case.execute(criteria)
        display_summaries(summaries)
    except Exception as e:
        logger.error(f"Failed to execute history search: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
