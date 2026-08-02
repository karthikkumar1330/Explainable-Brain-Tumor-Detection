import argparse
import sys
import os
import logging
from persistence.infrastructure.repository import SQLitePersistenceRepository


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for SQLite database queries."""
    parser = argparse.ArgumentParser(
        description="SQLite Database Persistence Analytics Tool"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="outputs/clinical_reports.db",
        help="Path to the SQLite database file",
    )
    parser.add_argument(
        "--patient-id",
        type=str,
        default=None,
        help="Query MRI scan history for a specific Patient ID",
    )
    parser.add_argument(
        "--analytics",
        action="store_true",
        help="Display dashboard analytics aggregated statistics across all patients",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Configure logging to stdout
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("db_query")

    if not os.path.exists(args.db_path):
        logger.error(f"Database file not found at: {args.db_path}")
        print(f"Error: Database file not found at {args.db_path}. Run generate_clinical_report.py first.")
        sys.exit(1)

    # Initialize repository
    repo = SQLitePersistenceRepository(db_path=args.db_path, logger=logger)

    # Mode 1: Query Patient Scan History
    if args.patient_id:
        print("\n" + "=" * 70)
        print(f"MRI SCAN AND DIAGNOSTIC HISTORY FOR PATIENT: {args.patient_id}")
        print("=" * 70)
        try:
            history = repo.get_patient_history(args.patient_id)
            if not history:
                print(f"No records found for patient: {args.patient_id}")
            else:
                # Print Demographics once
                p = history[0]
                print(f"Patient Name : {p['name']}")
                print(f"Age / Gender : {p['age']} years / {p['gender']}")
                print("-" * 70)
                
                # Print Scan History
                for idx, record in enumerate(history, 1):
                    print(f"Scan #{idx}:")
                    print(f"  Acquisition Date  : {record['scan_date']}")
                    print(f"  Referring M.D.    : {record['ref_physician']}")
                    print(f"  Classification    : {record['predicted_class']} (Conf: {record['confidence_score']:.2%})")
                    print(f"  Tumor Area        : {record['tumor_area_mm2']:.2f} mm²")
                    print(f"  AI Severity Level : {record['rule_based_severity']}")
                    print(f"  Saved PDF Report  : {record['pdf_path']}")
                    print(f"  Processed Time    : {record['created_at']}")
                    print("-" * 70)
        except Exception as e:
            logger.error(f"Error querying history: {e}")
            sys.exit(1)

    # Mode 2: Aggregate Dashboard Analytics
    if args.analytics or not args.patient_id:
        print("\n" + "=" * 70)
        print("CLINICAL PIPELINE AGGREGATED DASHBOARD ANALYTICS")
        print("=" * 70)
        try:
            metrics = repo.get_analytics_summary()
            print(f"Total Unique Patients: {metrics['total_patients']}")
            print(f"Total Scans Analyzed : {metrics['total_scans']}")
            print("-" * 70)
            
            print("Classification Distributions:")
            for cls_name, count in metrics["classification_distribution"].items():
                print(f"  - {cls_name:<16}: {count} scans")
            print("-" * 70)

            print("Severity Category Distributions:")
            for sev_name, count in metrics["severity_distribution"].items():
                print(f"  - {sev_name:<16}: {count} scans")
            print("-" * 70)

            print("Average Tumor Spatial Area (Active Tumors):")
            for cls_name, avg_area in metrics["average_tumor_area_mm2"].items():
                print(f"  - {cls_name:<16}: {avg_area:.2f} mm²")
            print("=" * 70 + "\n")
        except Exception as e:
            logger.error(f"Error querying database analytics: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
