import argparse
import sys
import os
import logging
from dashboard.infrastructure.web_server import create_app


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments to boot the clinical dashboard."""
    parser = argparse.ArgumentParser(
        description="AuraScan AI - Brain MRI Web Dashboard Server"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="outputs/clinical_reports.db",
        help="Path to the SQLite database file",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host interface to bind server to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to run the dashboard server on (default: 5000)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run Flask in debug hot-reloading mode",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Configure server logger
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("dashboard_boot")

    if not os.path.exists(args.db_path):
        logger.warning(
            f"SQLite database file not found at: {args.db_path}. "
            f"Initializing a clean database. Run generate_clinical_report.py to populate scan records."
        )

    logger.info(f"Starting AuraScan AI Web Server on http://{args.host}:{args.port}")
    try:
        app = create_app(db_path=args.db_path)
        app.run(host=args.host, port=args.port, debug=args.debug)
    except Exception as e:
        logger.error(f"Failed to start dashboard server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
