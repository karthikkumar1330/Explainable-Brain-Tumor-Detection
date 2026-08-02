import argparse
import os
import sys
import cv2
import logging

# Ensure root directory is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from classification.infrastructure.logging import get_logger
from tumor_analysis.domain.entities import ClinicalReportData
from tumor_analysis.infrastructure.analyzer import OpenCVTumorAnalyzer
from tumor_analysis.application.use_cases import AnalyzeTumorUseCase
from tumor_analysis.infrastructure.reporting import save_clinical_report
from severity_assessment.infrastructure.classifier import RuleBasedSeverityClassifier
from severity_assessment.application.use_cases import AssessSeverityUseCase


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for tumor area analysis."""
    parser = argparse.ArgumentParser(
        description="Tumor Area Analysis and Clinical Reporting Tool"
    )
    parser.add_argument(
        "--mask-path",
        type=str,
        required=True,
        help="Path to the predicted binary segmentation mask image",
    )
    parser.add_argument(
        "--image-path",
        type=str,
        default=None,
        help="Path to the original MRI scan image (optional, used for brain size estimation)",
    )
    parser.add_argument(
        "--patient-id",
        type=str,
        default="PATIENT_UNK",
        help="Patient or Scan identifier",
    )
    parser.add_argument(
        "--tumor-class",
        type=str,
        default="Unknown",
        help="Predicted tumor classification label (e.g., Glioma, Meningioma, Pituitary)",
    )
    parser.add_argument(
        "--pixel-spacing",
        type=float,
        default=1.0,
        help="Pixel spacing in millimeters (default: 1.0 mm per pixel)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/clinical_analysis",
        help="Directory to save the clinical reports and analysis outputs",
    )
    return parser.parse_args()


def main() -> None:
    # 1. Parse arguments
    args = parse_args()

    # 2. Setup Logger
    logger = get_logger(
        name="tumor_analysis",
        log_dir="logs",
        log_filename="tumor_analysis.log",
        level=logging.INFO,
    )

    logger.info("Initializing Tumor Area Analysis Tool...")

    # 3. Validate paths
    if not os.path.exists(args.mask_path):
        logger.error(f"Segmentation mask file not found at: {args.mask_path}")
        print(f"Error: Mask file not found at {args.mask_path}")
        sys.exit(1)

    original_image = None
    if args.image_path:
        if not os.path.exists(args.image_path):
            logger.error(f"Original MRI image file not found at: {args.image_path}")
            print(f"Error: Original MRI image file not found at {args.image_path}")
            sys.exit(1)
        original_image = cv2.imread(args.image_path)
        if original_image is None:
            logger.error(f"Could not read original image at: {args.image_path}")
            print(f"Error: Could not read original image at {args.image_path}")
            sys.exit(1)

    # 4. Load segmentation mask
    mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        logger.error(f"Could not read segmentation mask image at: {args.mask_path}")
        print(f"Error: Could not read segmentation mask image at {args.mask_path}")
        sys.exit(1)

    # 5. Wire Clean Architecture components
    analyzer = OpenCVTumorAnalyzer(low_thresh=1.0, med_thresh=5.0, high_thresh=15.0)
    use_case = AnalyzeTumorUseCase(analyzer=analyzer, logger=logger)

    severity_classifier = RuleBasedSeverityClassifier()
    severity_use_case = AssessSeverityUseCase(classifier=severity_classifier, logger=logger)

    # 6. Execute use cases
    try:
        report_data: ClinicalReportData = use_case.execute(
            mask=mask,
            patient_id=args.patient_id,
            tumor_class=args.tumor_class,
            original_image=original_image,
            pixel_spacing_mm=args.pixel_spacing,
        )

        # Perform Rule-Based Severity Assessment
        severity_assessment = severity_use_case.execute(
            tumor_type=report_data.tumor_class,
            tumor_area_mm2=report_data.analysis.tumor_area_mm2,
            tumor_percentage=report_data.analysis.tumor_percentage_brain,
        )

        # 7. Save reports
        base_filename = f"{args.patient_id}_{args.tumor_class.lower().replace(' ', '_')}"
        save_clinical_report(
            report=report_data,
            output_dir=args.output_dir,
            base_filename=base_filename,
            logger=logger,
            severity_assessment=severity_assessment,
        )

        # 8. Print Summary to console
        print("\n" + "=" * 60)
        print("TUMOR AREA ANALYSIS SUMMARY")
        print("=" * 60)
        print(f"Patient ID          : {report_data.patient_id}")
        print(f"Tumor Class         : {report_data.tumor_class}")
        print(f"Parenchymal Severity: {report_data.analysis.severity_level.value}")
        print(f"Tumor Pixel Count   : {report_data.analysis.pixel_count:,} px")
        print(f"Tumor Area          : {report_data.analysis.tumor_area_mm2:.2f} mm²")
        print(f"Tumor % of Image    : {report_data.analysis.tumor_percentage_image:.4f}%")
        print(f"Tumor % of Brain    : {report_data.analysis.tumor_percentage_brain:.4f}%")
        print("-" * 60)
        print(f"AI SEVERITY LEVEL   : {severity_assessment.category.value.upper()}")
        print(f"Decision Rule       : {severity_assessment.rule_description}")
        print("-" * 60)
        print("Clinical Note:")
        print(report_data.clinical_notes)
        print("-" * 60)
        print(severity_assessment.educational_disclaimer)
        print("=" * 60 + "\n")

    except Exception as e:
        logger.error(f"Execution failed: {e}")
        print(f"Error: Analysis execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
