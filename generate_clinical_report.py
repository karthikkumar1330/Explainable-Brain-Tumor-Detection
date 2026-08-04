import argparse
import os
import sys
import time
import datetime
import cv2
import yaml
import numpy as np
import torch
import albumentations as A
import logging

# Ensure root directory is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from classification.config import ClassificationConfig
from classification.infrastructure.models import EfficientNetB0Model, PyTorchModelAdapter
from classification.predict_cls import preprocess_image as preprocess_classification_image
from classification.application.use_cases import PredictUseCase, ExplainPredictionUseCase
from classification.infrastructure.explainability import GradCAMService
from classification.infrastructure.visualization import save_explainability_outputs
from classification.infrastructure.logging import get_logger

from tumor_analysis.infrastructure.analyzer import OpenCVTumorAnalyzer
from tumor_analysis.application.use_cases import AnalyzeTumorUseCase

from severity_assessment.infrastructure.classifier import RuleBasedSeverityClassifier
from severity_assessment.application.use_cases import AssessSeverityUseCase

from clinical_reporting.domain.entities import PatientInfo, ProcessingSummary, ClinicalReport
from clinical_reporting.infrastructure.generator import MarkdownJSONReportGenerator
from clinical_reporting.application.use_cases import GenerateIntegratedReportUseCase


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for the integrated clinical reporting pipeline."""
    parser = argparse.ArgumentParser(
        description="Integrated Clinical Brain MRI Report Generator"
    )
    # Patient Demographics
    parser.add_argument(
        "--image-path",
        type=str,
        required=True,
        help="Path to the patient's original brain MRI scan image file (.png/.tif)",
    )
    parser.add_argument(
        "--patient-id",
        type=str,
        default="PATIENT_001",
        help="Patient identifier",
    )
    parser.add_argument(
        "--patient-name",
        type=str,
        default="Anonymous Patient",
        help="Patient full name",
    )
    parser.add_argument(
        "--patient-age",
        type=int,
        default=45,
        help="Patient age in years",
    )
    parser.add_argument(
        "--patient-gender",
        type=str,
        default="Male",
        help="Patient gender (e.g. Male/Female/Other)",
    )
    parser.add_argument(
        "--ref-physician",
        type=str,
        default="Dr. Sarah Smith",
        help="Referring physician",
    )

    # Models & Checkpoints
    parser.add_argument(
        "--cls-checkpoint",
        type=str,
        default="models/classification/efficientnet_b0_brain_tumor.pth",
        help="Path to the trained classification model (.pth)",
    )
    parser.add_argument(
        "--seg-checkpoint",
        type=str,
        default="models/brain_tumor_unext/model.pth",
        help="Path to the trained UNeXt segmentation model (.pth)",
    )
    parser.add_argument(
        "--seg-config",
        type=str,
        default="models/brain_tumor_unext/config.yml",
        help="Path to UNeXt segmentation config.yml file",
    )

    # Hardware & Parameters
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Target device to run deep learning inference (cuda or cpu)",
    )
    parser.add_argument(
        "--pixel-spacing",
        type=float,
        default=1.0,
        help="MRI physical pixel spacing in millimeters (default: 1.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/clinical_reports",
        help="Destination directory for visual and written reports",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="outputs/clinical_reports.db",
        help="Path to SQLite database file for report persistence",
    )
    return parser.parse_args()


def preprocess_segmentation_image(image_path: str, h: int, w: int) -> torch.Tensor:
    """Loads and preprocesses an image for UNeXt segmentation model using [0, 1] scaling matching BraTS training."""
    img = cv2.imread(image_path)
    if img is None:
        raise IOError(f"Could not load segmentation image at: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (w, h))
    img_tensor = (img_resized.astype(np.float32) / 255.0).transpose(2, 0, 1)  # C, H, W
    return torch.from_numpy(img_tensor).unsqueeze(0)  # 1, C, H, W


def main() -> None:
    # 1. Start timer and load args
    start_total_time = time.time()
    args = parse_args()

    # 2. Setup Logger
    logger = get_logger(
        name="clinical_reporting",
        log_dir="logs",
        log_filename="clinical_reporting.log",
        level=logging.INFO,
    )
    logger.info("Initializing Integrated Clinical Pipeline...")

    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Defaulting to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    # Validate image path
    if not os.path.exists(args.image_path):
        logger.error(f"Input image not found: {args.image_path}")
        print(f"Error: MRI image not found at {args.image_path}")
        sys.exit(1)

    # 3. Predict Classification (EfficientNet-B0)
    logger.info("Step 1: Running Classification model...")
    t0 = time.time()
    try:
        config_cls = ClassificationConfig()
        image_tensor_cls = preprocess_classification_image(args.image_path, config_cls)

        model_cls = EfficientNetB0Model(pretrained=False, num_classes=4)
        model_adapter = PyTorchModelAdapter(model=model_cls, device=device_str)
        model_adapter.load(args.cls_checkpoint)

        predict_use_case = PredictUseCase(model_adapter=model_adapter)
        classification_result = predict_use_case.execute(image_tensor_cls)
        classification_latency = time.time() - t0
        logger.info(f"Classification result: {classification_result.class_name} (Confidence: {classification_result.confidence_score:.4f})")
    except Exception as e:
        logger.error(f"Classification step failed: {e}")
        print(f"Error during classification: {e}")
        sys.exit(1)

    # 4. Generate Explainability Map (Grad-CAM)
    logger.info("Step 2: Generating Grad-CAM explainability overlay...")
    t0 = time.time()
    heatmap_path = None
    overlay_path = None
    try:
        # Chosen target layer: final Conv2d block in features Sequential
        target_layer = model_cls.backbone.features[8]

        explain_service = GradCAMService(
            model=model_cls,
            target_layer=target_layer,
            device=device,
        )
        explain_use_case = ExplainPredictionUseCase(
            predict_use_case=predict_use_case,
            explain_service=explain_service,
            logger=logger,
        )

        _, heatmap = explain_use_case.execute(image_tensor_cls, target_class=classification_result.label)

        # Save outputs
        os.makedirs(args.output_dir, exist_ok=True)
        base_filename = f"{args.patient_id}_gradcam"
        original_image = cv2.imread(args.image_path)
        if original_image is None:
            raise IOError(f"Could not read image for Grad-CAM overlay: {args.image_path}")

        save_explainability_outputs(
            original_image=original_image,
            heatmap=heatmap,
            output_dir=args.output_dir,
            base_filename=base_filename,
            alpha=0.6,
            logger=logger,
        )
        heatmap_path = os.path.join(args.output_dir, f"{base_filename}_heatmap.png")
        overlay_path = os.path.join(args.output_dir, f"{base_filename}_overlay.png")
        explainability_latency = time.time() - t0
    except Exception as e:
        logger.error(f"Explainability step failed: {e}")
        print(f"Error during Grad-CAM generation: {e}")
        explainability_latency = 0.0

    # 5. Run Tumor Segmentation (UNeXt)
    logger.info("Step 3: Running UNeXt Segmentation model...")
    t0 = time.time()
    segmentation_mask_path = None
    segmentation_metrics = None
    try:
        # Load segmentation config
        with open(args.seg_config, "r") as f:
            seg_config = yaml.safe_load(f)

        # Preprocess
        input_tensor_seg = preprocess_segmentation_image(
            args.image_path, seg_config["input_h"], seg_config["input_w"]
        )
        input_tensor_seg = input_tensor_seg.to(device)

        # Instantiate UNeXt model dynamically from archs.py
        import archs
        model_seg = archs.__dict__[seg_config["arch"]](
            num_classes=seg_config["num_classes"],
            input_channels=seg_config["input_channels"],
            deep_supervision=seg_config["deep_supervision"],
        )
        model_seg.load_state_dict(torch.load(args.seg_checkpoint, map_location=device))
        model_seg = model_seg.to(device)
        model_seg.eval()

        # Run inference
        with torch.no_grad():
            output_seg = model_seg(input_tensor_seg)
            if seg_config["deep_supervision"]:
                output_seg = output_seg[-1]
            output_seg = torch.sigmoid(output_seg).squeeze(0).squeeze(0).cpu().numpy()

        # Binarize (using standard 0.5 threshold)
        bin_mask = (output_seg > 0.5).astype(np.uint8)

        # Post-process: connected components area filter (remove blobs < 100 pixels)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            bin_mask, connectivity=8
        )
        filtered_mask = np.zeros_like(bin_mask)
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= 100:
                filtered_mask[labels == label] = 1

        # Resize the mask back to the original image dimensions
        orig_h, orig_w = original_image.shape[:2]
        final_mask = cv2.resize(
            filtered_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
        )

        # Save segmented mask
        segmentation_mask_path = os.path.join(args.output_dir, f"{args.patient_id}_segmentation_mask.jpg")
        cv2.imwrite(segmentation_mask_path, (final_mask * 255).astype(np.uint8))
        logger.info(f"Saved segmentation mask to: {segmentation_mask_path}")
        segmentation_latency = time.time() - t0

        # 6. Run Morphological Analysis
        logger.info("Step 4: Executing Tumor Morphological Analysis...")
        morph_analyzer = OpenCVTumorAnalyzer(low_thresh=1.0, med_thresh=5.0, high_thresh=15.0)
        morph_use_case = AnalyzeTumorUseCase(analyzer=morph_analyzer, logger=logger)
        clinical_data = morph_use_case.execute(
            mask=final_mask,
            patient_id=args.patient_id,
            tumor_class=classification_result.class_name,
            original_image=original_image,
            pixel_spacing_mm=args.pixel_spacing,
        )
        segmentation_metrics = clinical_data.analysis
    except Exception as e:
        logger.error(f"Segmentation/Morphology steps failed: {e}")
        print(f"Error during segmentation: {e}")
        segmentation_latency = 0.0

    # 7. Run AI Severity Risk Assessment (Rule-Based LOW/MEDIUM/HIGH)
    logger.info("Step 5: Executing AI Severity Risk Assessment...")
    severity_assessment = None
    if segmentation_metrics is not None:
        try:
            severity_classifier = RuleBasedSeverityClassifier()
            severity_use_case = AssessSeverityUseCase(classifier=severity_classifier, logger=logger)
            severity_assessment = severity_use_case.execute(
                tumor_type=classification_result.class_name,
                tumor_area_mm2=segmentation_metrics.tumor_area_mm2,
                tumor_percentage=segmentation_metrics.tumor_percentage_brain,
            )
        except Exception as e:
            logger.error(f"Severity assessment failed: {e}")
            print(f"Error during severity risk evaluation: {e}")

    # 8. Generate Integrated Clinical Report
    logger.info("Step 6: Formatting and saving integrated clinical report...")
    try:
        scan_date_str = datetime.date.today().strftime("%Y-%m-%d")
        patient_info = PatientInfo(
            patient_id=args.patient_id,
            name=args.patient_name,
            age=args.patient_age,
            gender=args.patient_gender,
            scan_date=scan_date_str,
            ref_physician=args.ref_physician,
        )

        total_exec_time = time.time() - start_total_time
        processing_summary = ProcessingSummary(
            device=args.device,
            execution_time_sec=total_exec_time,
            classification_model_path=args.cls_checkpoint,
            segmentation_model_path=args.seg_checkpoint,
            classification_latency_sec=classification_latency,
            segmentation_latency_sec=segmentation_latency,
            explainability_latency_sec=explainability_latency,
        )

        clinical_report = ClinicalReport(
            patient_info=patient_info,
            processing_summary=processing_summary,
            classification=classification_result,
            segmentation_metrics=segmentation_metrics,
            severity_assessment=severity_assessment,
            original_image_path=args.image_path,
            heatmap_image_path=heatmap_path,
            overlay_image_path=overlay_path,
            segmentation_mask_path=segmentation_mask_path,
        )

        # Wire generator Clean Architecture components
        generator = MarkdownJSONReportGenerator()
        report_use_case = GenerateIntegratedReportUseCase(report_generator=generator, logger=logger)
        md_file, json_file, pdf_file = report_use_case.execute(report=clinical_report, output_dir=args.output_dir)

        # Step 7: Persist report to SQLite Database
        logger.info("Step 7: Persisting findings to database...")
        from persistence.infrastructure.repository import SQLitePersistenceRepository
        try:
            db_repo = SQLitePersistenceRepository(db_path=args.db_path, logger=logger)
            db_repo.initialize_db()
            db_report_id = db_repo.save_report(clinical_report, output_dir=args.output_dir)
            logger.info(f"Report findings successfully persisted in SQLite (Record ID: {db_report_id}).")
        except Exception as db_err:
            logger.error(f"Failed to persist report to SQLite: {db_err}")
            print(f"Warning: Could not save report to database: {db_err}")

        # 9. Display pipeline complete summary to console
        print("\n" + "=" * 60)
        print("INTEGRATED CLINICAL REPORT GENERATION PIPELINE")
        print("=" * 60)
        print("1. PATIENT DEMOGRAPHICS")
        print(f"   Patient ID        : {patient_info.patient_id}")
        print(f"   Patient Name      : {patient_info.name}")
        print(f"   Age / Gender      : {patient_info.age} years / {patient_info.gender}")
        print(f"   Referring M.D.    : {patient_info.ref_physician}")
        print("-" * 60)
        print("2. DIAGNOSTIC RESULTS")
        print(f"   Classification    : {classification_result.class_name} (Confidence: {classification_result.confidence_score:.4%})")
        if segmentation_metrics is not None:
            print(f"   Tumor Area        : {segmentation_metrics.tumor_area_mm2:.2f} mm² ({segmentation_metrics.pixel_count:,} px)")
            print(f"   Parenchymal Space : {segmentation_metrics.tumor_percentage_brain:.4f}% of brain tissue")
        if severity_assessment is not None:
            print(f"   AI Severity Level : {severity_assessment.category.value.upper()}")
            print(f"   Matched Rule      : {severity_assessment.rule_description}")
        print("-" * 60)
        print("3. CLINICAL EXPLAINABILITY & VISUAL SCANS")
        print(f"   Grad-CAM Overlay  : {overlay_path}")
        print(f"   Tumor Mask        : {segmentation_mask_path}")
        print("-" * 60)
        print("4. TECHNICAL BENCHMARKS")
        print(f"   Hardware Device   : {processing_summary.device.upper()}")
        print(f"   Total Pipeline t  : {processing_summary.execution_time_sec:.4f} seconds")
        print("-" * 60)
        print(severity_assessment.educational_disclaimer if severity_assessment is not None else "")
        print("-" * 60)
        print(f"Reports successfully generated and saved to: {args.output_dir}")
        print(f"  - Markdown Report : {md_file}")
        print(f"  - EHR JSON Payload: {json_file}")
        print(f"  - Clinical PDF    : {pdf_file}")
        print("=" * 60 + "\n")

    except Exception as e:
        logger.error(f"Clinical report generation failed: {e}")
        print(f"Error: Clinical report pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
