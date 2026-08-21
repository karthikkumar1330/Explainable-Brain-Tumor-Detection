from dotenv import load_dotenv
load_dotenv()

import argparse
import sys
import os
import logging
from dashboard.infrastructure.web_server import create_app
from security.application.config import app_config


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
        default=app_config.host,
        help=f"Host interface to bind server to (default: {app_config.host})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=app_config.dashboard_port,
        help=f"Port to run the dashboard server on (default: {app_config.dashboard_port})",
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

    logger.info("Starting AuraScan AI Web Server...")
    logger.info(f" - Configured Host Bind: {args.host}")
    logger.info(f" - Configured Port: {args.port}")
    logger.info(f" - Configured Public Base URL: {app_config.public_base_url}")
    try:
        app = create_app(db_path=args.db_path)
        app.run(host=args.host, port=args.port, debug=args.debug)
    except Exception as e:
        logger.error(f"Failed to start dashboard server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
