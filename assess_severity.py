import argparse
import os
import sys
import logging

# Ensure root directory is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from classification.infrastructure.logging import get_logger
from severity_assessment.domain.entities import SeverityAssessment
from severity_assessment.infrastructure.classifier import RuleBasedSeverityClassifier
from severity_assessment.application.use_cases import AssessSeverityUseCase


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for severity assessment."""
    parser = argparse.ArgumentParser(
        description="Brain Tumor Severity Assessment Tool (Rule-Based)"
    )
    parser.add_argument(
        "--tumor-type",
        type=str,
        required=True,
        help="Classification type of tumor (e.g. Glioma, Meningioma, Pituitary, No Tumor)",
    )
    parser.add_argument(
        "--tumor-area",
        type=float,
        required=True,
        help="Tumor physical area in square millimeters",
    )
    parser.add_argument(
        "--tumor-percentage",
        type=float,
        required=True,
        help="Tumor parenchymal occupancy percentage (0 to 100)",
    )
    return parser.parse_args()


def main() -> None:
    # 1. Parse arguments
    args = parse_args()

    # 2. Setup Logger
    logger = get_logger(
        name="severity_assessment",
        log_dir="logs",
        log_filename="severity_assessment.log",
        level=logging.INFO,
    )

    logger.info("Initializing Severity Assessment Tool...")

    # 3. Instantiate Clean Architecture components
    classifier = RuleBasedSeverityClassifier()
    use_case = AssessSeverityUseCase(classifier=classifier, logger=logger)

    # 4. Execute severity assessment
    try:
        assessment: SeverityAssessment = use_case.execute(
            tumor_type=args.tumor_type,
            tumor_area_mm2=args.tumor_area,
            tumor_percentage=args.tumor_percentage,
        )

        # 5. Display output to console
        print("\n" + "=" * 60)
        print("RULE-BASED AI SEVERITY ASSESSMENT")
        print("=" * 60)
        print(f"Tumor Type           : {args.tumor_type}")
        print(f"Tumor Physical Area  : {args.tumor_area:.2f} mm²")
        print(f"Parenchymal Occupancy: {args.tumor_percentage:.4f}%")
        print("-" * 60)
        print(f"Severity Category    : {assessment.category.value.upper()}")
        print(f"Decision Rule        : {assessment.rule_description}")
        print("-" * 60)
        print(assessment.educational_disclaimer)
        print("=" * 60 + "\n")

    except Exception as e:
        logger.error(f"Severity assessment execution failed: {e}")
        print(f"Error: Severity assessment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
