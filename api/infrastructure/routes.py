import os
import sqlite3
import shutil
import time
import datetime
import cv2
import yaml
import numpy as np
import torch
import albumentations as A
import logging
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Query
from fastapi.responses import FileResponse, JSONResponse
from contextlib import asynccontextmanager
from typing import Optional, List

from classification.config import ClassificationConfig
from classification.infrastructure.models import EfficientNetB0Model, PyTorchModelAdapter
from classification.predict_cls import preprocess_image as preprocess_classification_image
from classification.application.use_cases import PredictUseCase, ExplainPredictionUseCase
from classification.infrastructure.explainability import GradCAMService
from classification.infrastructure.visualization import save_explainability_outputs

from tumor_analysis.infrastructure.analyzer import OpenCVTumorAnalyzer
from tumor_analysis.application.use_cases import AnalyzeTumorUseCase

from severity_assessment.infrastructure.classifier import RuleBasedSeverityClassifier
from severity_assessment.application.use_cases import AssessSeverityUseCase

from clinical_reporting.domain.entities import PatientInfo, ProcessingSummary, ClinicalReport
from clinical_reporting.infrastructure.generator import MarkdownJSONReportGenerator
from clinical_reporting.application.use_cases import GenerateIntegratedReportUseCase

from persistence.infrastructure.repository import SQLitePersistenceRepository
from prediction_history.infrastructure.repository import SQLitePredictionHistoryRepository
from prediction_history.domain.entities import HistorySearchCriteria

from api.domain.schemas import PatientIntake


# Configure paths
DEFAULT_DB_PATH = "outputs/clinical_reports.db"
CLS_CHECKPOINT = "models/classification/efficientnet_b0_brain_tumor.pth"
SEG_CHECKPOINT = "models/brain_tumor_unext/model.pth"
SEG_CONFIG = "models/brain_tumor_unext/config.yml"
OUTPUT_REPORTS_DIR = "outputs/clinical_reports"
TEMP_UPLOAD_DIR = "outputs/temp_uploads"

# Global Model holders
model_cls = None
predict_use_case = None
model_seg = None
seg_config = None
device = torch.device("cpu")

logger = logging.getLogger("api_routes")


def initialize_api_models():
    """Preloads the deep learning model states into memory."""
    global model_cls, predict_use_case, model_seg, seg_config, device
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"FastAPI API initializing models on device: {device}")

    # Thread tuning for CPU fallback to avoid thrashing
    if device.type == "cpu" and torch.get_num_threads() > 4:
        torch.set_num_threads(4)

    # Load classification pipeline
    try:
        model_cls = EfficientNetB0Model(pretrained=False, num_classes=4)
        model_adapter = PyTorchModelAdapter(model=model_cls, device=str(device))
        model_adapter.load(CLS_CHECKPOINT)
        predict_use_case = PredictUseCase(model_adapter=model_adapter)
        logger.info("Classification model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load classification checkpoint: {e}")

    # Load UNeXt segmentation pipeline
    try:
        with open(SEG_CONFIG, "r") as f:
            seg_config = yaml.safe_load(f)
        
        import archs
        model_seg = archs.__dict__[seg_config["arch"]](
            num_classes=seg_config["num_classes"],
            input_channels=seg_config["input_channels"],
            deep_supervision=seg_config["deep_supervision"],
        )
        model_seg.load_state_dict(torch.load(SEG_CHECKPOINT, map_location=torch.device(device)))
        model_seg = model_seg.to(device)
        model_seg.eval()
        logger.info("UNeXt segmentation model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load segmentation checkpoint: {e}")


def preprocess_segmentation_image(img_bgr: np.ndarray, h: int, w: int) -> torch.Tensor:
    """Preprocesses BGR image slice for segmentation model."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    transform = A.Compose([
        A.Resize(h, w),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    augmented = transform(image=img_rgb)
    img_tensor = augmented['image'].transpose(2, 0, 1)  # C, H, W
    return torch.from_numpy(img_tensor).unsqueeze(0)  # 1, C, H, W


# FastAPI Router
router = APIRouter()


@router.post("/upload")
def upload_mri_file(file: UploadFile = File(...)):
    """API Endpoint: Receives a raw brain MRI image slice upload and validates it."""
    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
    temp_filename = f"upload_{int(time.time())}_{file.filename}"
    temp_filepath = os.path.join(TEMP_UPLOAD_DIR, temp_filename)
    
    try:
        file_bytes = file.file.read()
        with open(temp_filepath, "wb") as f:
            f.write(file_bytes)
        
        # Run MRI Input Validation
        from input_validation.infrastructure.validators import OpenCVMriValidator
        from input_validation.application.use_cases import ValidateMriUploadUseCase
        
        validator = OpenCVMriValidator()
        use_case = ValidateMriUploadUseCase(validator=validator, db_path=DEFAULT_DB_PATH)
        scorecard = use_case.execute(filepath=temp_filepath, file_bytes=file_bytes, filename=file.filename)
        
        if not scorecard.is_valid:
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "MRI Input Validation Failed",
                    "errors": scorecard.errors,
                    "scorecard": scorecard.to_dict()
                }
            )
            
        return {
            "filename": temp_filename,
            "filepath": temp_filepath,
            "scorecard": scorecard.to_dict()
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error handling file upload: {e}")
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        raise HTTPException(status_code=500, detail=f"File upload processing failed: {e}")



@router.post("/classification")
def run_classification(filepath: str):
    """API Endpoint: Runs classification model on the uploaded MRI slice.

    Args:
        filepath: Full path of the cached scan image.
    """
    if not os.path.exists(filepath):
        raise HTTPException(status_code=400, detail="Target MRI file path not found.")
    
    try:
        config_cls = ClassificationConfig()
        image_tensor_cls = preprocess_classification_image(filepath, config_cls)
        result = predict_use_case.execute(image_tensor_cls)
        return {
            "predicted_class": result.class_name,
            "confidence_score": float(result.confidence_score),
            "probabilities": {k: float(v) for k, v in result.probabilities.items()}
        }
    except Exception as e:
        logger.error(f"Classification run error: {e}")
        raise HTTPException(status_code=500, detail=f"Classification process failed: {e}")


@router.post("/segmentation")
def run_segmentation(filepath: str):
    """API Endpoint: Renders brain tumor segmentation mask using UNeXt model."""
    if not os.path.exists(filepath):
        raise HTTPException(status_code=400, detail="Target MRI file path not found.")
    
    try:
        img_bgr = cv2.imread(filepath, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Could not read scan image.")

        # Preprocess and segment
        input_tensor_seg = preprocess_segmentation_image(
            img_bgr, seg_config["input_h"], seg_config["input_w"]
        )
        input_tensor_seg = input_tensor_seg.to(device)
        
        device_type = device.type
        is_autocast_supported = device_type in ["cuda", "cpu"]
        
        with torch.inference_mode():
            if is_autocast_supported:
                dtype = torch.float16 if device_type == "cuda" else torch.bfloat16
                with torch.amp.autocast(device_type=device_type, dtype=dtype):
                    output_seg = model_seg(input_tensor_seg)
            else:
                output_seg = model_seg(input_tensor_seg)
                
            if seg_config["deep_supervision"]:
                output_seg = output_seg[-1]
            output_seg = torch.sigmoid(output_seg).squeeze(0).squeeze(0).cpu().numpy()

        # Resize probability map to native resolution
        orig_h, orig_w = img_bgr.shape[:2]
        prob_map_resized = cv2.resize(output_seg, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        bin_mask_resized = (prob_map_resized > 0.5).astype(np.uint8)

        # Run modular post-processing pipeline
        from segmentation_postprocessing.infrastructure.processors import MedicalImagePostProcessor
        from segmentation_postprocessing.application.use_cases import PostProcessSegmentationUseCase
        from segmentation_postprocessing.infrastructure.visualization import create_segmentation_comparison_image

        post_proc = MedicalImagePostProcessor()
        post_proc_use_case = PostProcessSegmentationUseCase(post_processor=post_proc)
        final_mask, post_proc_meta = post_proc_use_case.execute(bin_mask_resized, prob_map_resized)

        # Save to disk
        os.makedirs(OUTPUT_REPORTS_DIR, exist_ok=True)
        mask_filename = f"mask_{int(time.time())}.jpg"
        mask_path = os.path.join(OUTPUT_REPORTS_DIR, mask_filename)
        cv2.imwrite(mask_path, (final_mask * 255).astype(np.uint8))

        comparison_filename = f"comparison_{int(time.time())}.png"
        comparison_path = os.path.join(OUTPUT_REPORTS_DIR, comparison_filename)
        create_segmentation_comparison_image(
            original_image=img_bgr,
            before_mask=bin_mask_resized,
            after_mask=final_mask,
            output_path=comparison_path
        )

        return {
            "mask_path": mask_path,
            "comparison_path": comparison_path,
            "tumor_pixel_count": int(np.sum(final_mask)),
            "quality_score": float(post_proc_meta["quality_score"]),
            "quality_category": post_proc_meta["quality_category"],
            "post_processing_applied": True,
            "post_processing_metadata": post_proc_meta
        }
    except Exception as e:
        logger.error(f"Segmentation run error: {e}")
        raise HTTPException(status_code=500, detail=f"Segmentation processing failed: {e}")


@router.post("/explainability")
def run_explainability(filepath: str, target_class: int = 1, method: str = "gradcam"):
    """API Endpoint: Runs Explainable AI 2.0 attention heatmap on target classification index."""
    if not os.path.exists(filepath):
        raise HTTPException(status_code=400, detail="Target MRI file path not found.")
    
    try:
        img_bgr = cv2.imread(filepath, cv2.IMREAD_COLOR)
        config_cls = ClassificationConfig()
        image_tensor_cls = preprocess_classification_image(filepath, config_cls)

        from explainable_ai.infrastructure.services import PyTorchXAIEngine
        from explainable_ai.application.use_cases import GenerateExplanationUseCase

        # Map selected XAI method
        xai_param = "gradcam"
        if method.lower() in ["gradcam++", "gradcam_plus_plus"]:
            xai_param = "gradcam_plus_plus"
        elif method.lower() == "eigencam":
            xai_param = "eigencam"

        xai_engine = PyTorchXAIEngine(
            model=model_cls,
            target_layer=model_cls.backbone.features[8],
            device=device
        )
        xai_use_case = GenerateExplanationUseCase(xai_engine=xai_engine)
        xai_result = xai_use_case.execute(
            image_tensor=image_tensor_cls,
            target_class=target_class,
            method=xai_param,
            tumor_mask=None
        )

        os.makedirs(OUTPUT_REPORTS_DIR, exist_ok=True)
        base_cam_name = f"gradcam_{int(time.time())}"
        save_explainability_outputs(
            original_image=img_bgr,
            heatmap=xai_result.heatmap,
            output_dir=OUTPUT_REPORTS_DIR,
            base_filename=base_cam_name,
            alpha=0.6,
            logger=logger
        )
        return {
            "heatmap_path": os.path.join(OUTPUT_REPORTS_DIR, f"{base_cam_name}_heatmap.png"),
            "overlay_path": os.path.join(OUTPUT_REPORTS_DIR, f"{base_cam_name}_overlay.png"),
            "explanation_text": xai_result.explanation_text,
            "quadrant_attention": xai_result.quadrant_attention,
            "method_used": xai_param
        }
    except Exception as e:
        logger.error(f"Explainability hook execution failure: {e}")
        raise HTTPException(status_code=500, detail=f"XAI hook execution failed: {e}")


@router.get("/health")
def get_pipeline_health():
    """API Endpoint: Runs comprehensive system and model health diagnostics and fetches audit telemetry."""
    try:
        from monitoring.infrastructure.health_monitor import PipelineHealthMonitor
        from monitoring.application.use_cases import RunPipelineHealthCheckUseCase
        
        monitor = PipelineHealthMonitor(db_path=DEFAULT_DB_PATH, api_url="http://127.0.0.1:8000/docs")
        use_case = RunPipelineHealthCheckUseCase(monitor=monitor)
        
        report = use_case.execute(
            model_cls=model_cls,
            model_seg=model_seg,
            device=str(device)
        )
        
        db_repo = SQLitePersistenceRepository(db_path=DEFAULT_DB_PATH)
        telemetry = db_repo.get_health_telemetry()
        
        report_dict = report.to_dict()
        report_dict["historical_telemetry"] = telemetry
        return report_dict
    except Exception as e:
        logger.error(f"Health check execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health diagnostics failed: {e}")


@router.post("/report")
def generate_clinical_report_pipeline(filepath: str, intake: PatientIntake):
    """API Endpoint: Runs the complete end-to-end MRI diagnostics report pipeline with validation."""
    if not os.path.exists(filepath):
        raise HTTPException(status_code=400, detail="Target upload MRI file path not found.")
    
    t_endpoint_start = time.time()
    timeline = {}
    timeline["Upload"] = time.time() - t_endpoint_start

    t_start = time.time()
    try:
        # Run MRI Input Validation
        with open(filepath, "rb") as f:
            file_bytes = f.read()
            
        from input_validation.infrastructure.validators import OpenCVMriValidator
        from input_validation.application.use_cases import ValidateMriUploadUseCase
        
        validator = OpenCVMriValidator()
        use_case = ValidateMriUploadUseCase(validator=validator, db_path=DEFAULT_DB_PATH)
        scorecard = use_case.execute(filepath=filepath, file_bytes=file_bytes, filename=os.path.basename(filepath))
        
        timeline["Validation"] = time.time() - t_endpoint_start

        if not scorecard.is_valid:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "MRI Input Validation Failed",
                    "errors": scorecard.errors,
                    "scorecard": scorecard.to_dict()
                }
            )

        img_bgr = cv2.imread(filepath, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Failed to read uploaded image.")

        # 1. Classification (B6.12 Retry & CPU fallback)
        from monitoring.application.pipeline_recovery import PipelineExecutionRecovery
        recovery = PipelineExecutionRecovery(logger=logger)

        t_cls = time.time()
        config_cls = ClassificationConfig()
        image_tensor_cls = preprocess_classification_image(filepath, config_cls)
        
        cls_warnings = []
        active_cls_device = str(device)
        try:
            classification_result = predict_use_case.execute(image_tensor_cls)
        except Exception as e:
            logger.warning(f"Classification failed on {device}: {e}. Retrying with CPU fallback...")
            try:
                model_cls.to("cpu")
                predict_use_case.model_adapter.device = "cpu"
                classification_result = predict_use_case.execute(image_tensor_cls)
                active_cls_device = "cpu"
                cls_warnings.append("Auto-recovery warning: Classification execution failed on GPU. Retried and completed on CPU fallback mode.")
            except Exception as cpu_err:
                logger.critical(f"CPU fallback for classification failed: {cpu_err}")
                raise HTTPException(status_code=500, detail=f"Classification inference failed: {cpu_err}")
        
        cls_latency = time.time() - t_cls
        timeline["Classification"] = time.time() - t_endpoint_start
        timeline["Calibration"] = time.time() - t_endpoint_start

        # 2. Explainability (XAI 2.0)
        t_cam = time.time()
        from explainable_ai.infrastructure.services import PyTorchXAIEngine
        from explainable_ai.application.use_cases import GenerateExplanationUseCase
        from explainable_ai.infrastructure.visualization import overlay_tumor_contour

        # Map selected XAI method
        xai_param = "gradcam"
        if getattr(intake, "xai_method", None) is not None:
            if intake.xai_method.lower() in ["gradcam++", "gradcam_plus_plus"]:
                xai_param = "gradcam_plus_plus"
            elif intake.xai_method.lower() == "eigencam":
                xai_param = "eigencam"

        xai_engine = PyTorchXAIEngine(
            model=model_cls,
            target_layer=model_cls.backbone.features[8],
            device=device
        )
        
        # We generate the raw heatmap first inside a graceful block (B6.12)
        class DummyXaiResult:
            def __init__(self):
                self.explanation_text = "Explanation generation failed due to hook limitations. Degraded gracefully."
                self.overlap_percentage = 0.0
                self.heatmap = np.zeros((img_bgr.shape[0], img_bgr.shape[1]), dtype=np.float32)

        def run_xai():
            heatmap_raw = xai_engine.generate_explanation(
                image_tensor=image_tensor_cls,
                target_class=classification_result.label,
                method=xai_param
            )
            xai_use_case = GenerateExplanationUseCase(xai_engine=xai_engine)
            xai_res = xai_use_case.execute(
                image_tensor=image_tensor_cls,
                target_class=classification_result.label,
                method=xai_param,
                tumor_mask=final_mask
            )
            setattr(xai_res, "heatmap_raw", heatmap_raw)
            return xai_res

        # 3. Segmentation (B6.12 Retry & CPU fallback)
        t_seg = time.time()
        input_tensor_seg = preprocess_segmentation_image(img_bgr, seg_config["input_h"], seg_config["input_w"])
        
        seg_warnings = []
        active_seg_device = str(device)
        try:
            input_tensor_seg_dev = input_tensor_seg.to(device)
            device_type = device.type
            is_autocast_supported = device_type in ["cuda", "cpu"]
            
            with torch.inference_mode():
                if is_autocast_supported:
                    dtype = torch.float16 if device_type == "cuda" else torch.bfloat16
                    with torch.amp.autocast(device_type=device_type, dtype=dtype):
                        output_seg = model_seg(input_tensor_seg_dev)
                else:
                    output_seg = model_seg(input_tensor_seg_dev)
                    
                if seg_config["deep_supervision"]:
                    output_seg = output_seg[-1]
                output_seg = torch.sigmoid(output_seg).squeeze(0).squeeze(0).cpu().numpy()
        except Exception as e:
            logger.warning(f"Segmentation failed on {device}: {e}. Retrying with CPU fallback...")
            try:
                model_seg.to("cpu")
                active_seg_device = "cpu"
                input_tensor_seg_cpu = input_tensor_seg.to("cpu")
                with torch.inference_mode():
                    output_seg = model_seg(input_tensor_seg_cpu)
                    if seg_config["deep_supervision"]:
                        output_seg = output_seg[-1]
                    output_seg = torch.sigmoid(output_seg).squeeze(0).squeeze(0).cpu().numpy()
                seg_warnings.append("Auto-recovery warning: Segmentation execution failed on GPU. Retried and completed on CPU fallback mode.")
            except Exception as cpu_err:
                logger.critical(f"CPU fallback for segmentation failed: {cpu_err}")
                raise HTTPException(status_code=500, detail=f"Segmentation inference failed: {cpu_err}")
                
        bin_mask = (output_seg > 0.5).astype(np.uint8)

        # Resize to original scale so post-processing runs at native resolution
        orig_h, orig_w = img_bgr.shape[:2]
        bin_mask_resized = cv2.resize(bin_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        prob_map_resized = cv2.resize(output_seg, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # Run modular post-processing pipeline
        from segmentation_postprocessing.infrastructure.processors import MedicalImagePostProcessor
        from segmentation_postprocessing.application.use_cases import PostProcessSegmentationUseCase
        from segmentation_postprocessing.infrastructure.visualization import create_segmentation_comparison_image

        post_proc = MedicalImagePostProcessor()
        post_proc_use_case = PostProcessSegmentationUseCase(post_processor=post_proc)
        final_mask, post_proc_meta = post_proc_use_case.execute(bin_mask_resized, prob_map_resized)
        
        seg_latency = time.time() - t_seg
        timeline["Segmentation"] = time.time() - t_endpoint_start

        # Now run XAI using the post-processed segmentation mask
        xai_result, xai_warns = recovery.execute_graceful_stage(
            stage_name="Grad-CAM Explanation Generation",
            stage_fn=run_xai,
            default_fallback_value=DummyXaiResult()
        )
        
        # Extract heatmap
        if hasattr(xai_result, "heatmap_raw"):
            heatmap = xai_result.heatmap_raw
        else:
            heatmap = getattr(xai_result, "heatmap", np.zeros((img_bgr.shape[0], img_bgr.shape[1]), dtype=np.float32))

        cam_latency = time.time() - t_cam
        timeline["GradCAM"] = time.time() - t_endpoint_start

        # Save explanation visualizations with boundary overlays
        os.makedirs(OUTPUT_REPORTS_DIR, exist_ok=True)
        base_cam_name = f"{intake.patient_id}_api_gradcam"
        
        from classification.infrastructure.visualization import overlay_heatmap
        raw_overlay = overlay_heatmap(img_bgr, heatmap, alpha=0.6)
        overlay_with_contour = overlay_tumor_contour(raw_overlay, final_mask)
        
        heatmap_path = os.path.join(OUTPUT_REPORTS_DIR, f"{base_cam_name}_heatmap.png")
        overlay_path = os.path.join(OUTPUT_REPORTS_DIR, f"{base_cam_name}_overlay.png")
        
        heatmap_uint8 = np.uint8(255 * cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0])))
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        cv2.imwrite(heatmap_path, heatmap_color)
        cv2.imwrite(overlay_path, overlay_with_contour)

        # 4. Morphological Analysis
        morph_analyzer = OpenCVTumorAnalyzer(low_thresh=1.0, med_thresh=5.0, high_thresh=15.0)
        morph_use_case = AnalyzeTumorUseCase(analyzer=morph_analyzer, logger=logger)
        
        class DummyClinicalData:
            def __init__(self):
                from tumor_analysis.domain.entities import TumorAnalysisResult
                self.analysis = TumorAnalysisResult(
                    pixel_count=0,
                    tumor_area_mm2=0.0,
                    tumor_percentage_brain=0.0,
                    tumor_percentage_image=0.0,
                    estimated_brain_pixel_count=0,
                    rule_based_severity="LOW",
                    severity_rule_description="Degraded statistics due to fallback.",
                    stats=None
                )

        def run_morph():
            return morph_use_case.execute(
                mask=final_mask,
                patient_id=intake.patient_id,
                tumor_class=classification_result.class_name,
                original_image=img_bgr,
                pixel_spacing_mm=intake.pixel_spacing_mm,
            )
            
        clinical_data, morph_warns = recovery.execute_graceful_stage(
            stage_name="Morphological Stats Extraction",
            stage_fn=run_morph,
            default_fallback_value=DummyClinicalData()
        )
        segmentation_metrics = clinical_data.analysis
        
        # Enrich segmentation_metrics with post-processing details
        from dataclasses import replace
        segmentation_metrics = replace(
            segmentation_metrics,
            quality_score=post_proc_meta["quality_score"],
            quality_category=post_proc_meta["quality_category"],
            post_processing_applied=True,
            post_processing_metadata=post_proc_meta
        )
        
        timeline["Statistics"] = time.time() - t_endpoint_start
        timeline["Comparison"] = time.time() - t_endpoint_start  # API endpoint doesn't evaluate longitudinal comparison

        # 5. Rule-Based Severity Assessment
        severity_classifier = RuleBasedSeverityClassifier()
        severity_use_case = AssessSeverityUseCase(classifier=severity_classifier, logger=logger)
        severity_assessment = severity_use_case.execute(
            tumor_type=classification_result.class_name,
            tumor_area_mm2=segmentation_metrics.tumor_area_mm2,
            tumor_percentage=segmentation_metrics.tumor_percentage_brain,
        )

        # 6. Report generation
        patient_info = PatientInfo(
            patient_id=intake.patient_id,
            name=intake.name,
            age=intake.age,
            gender=intake.gender,
            scan_date=datetime.date.today().strftime("%Y-%m-%d"),
            ref_physician=intake.ref_physician,
        )

        total_exec_time = time.time() - t_start
        processing_summary = ProcessingSummary(
            device=active_cls_device,
            execution_time_sec=total_exec_time,
            classification_model_path=CLS_CHECKPOINT,
            segmentation_model_path=SEG_CHECKPOINT,
            classification_latency_sec=cls_latency,
            segmentation_latency_sec=seg_latency,
            explainability_latency_sec=cam_latency,
        )

        # Run Central Warning Engine Checks (B6.3, B6.4, B6.5, B6.7)
        from monitoring.infrastructure.segmentation_validator import SegmentationValidator
        from monitoring.infrastructure.consistency_checker import ConfidenceConsistencyChecker
        from monitoring.infrastructure.explainability_validator import ExplainabilityValidator
        from monitoring.application.warning_engine import CentralWarningEngine
        
        seg_validator = SegmentationValidator()
        consistency_checker = ConfidenceConsistencyChecker()
        explain_validator = ExplainabilityValidator()
        warning_engine = CentralWarningEngine(
            seg_validator=seg_validator,
            consistency_checker=consistency_checker,
            explain_validator=explain_validator
        )
        
        uncal_conf = getattr(classification_result, "uncalibrated_confidence_score", None)
        is_cal = getattr(classification_result, "is_calibrated", False)
        
        engine_result = warning_engine.collect_warnings(
            input_errors=cls_warnings + seg_warnings + morph_warns,
            predicted_class=classification_result.class_name,
            confidence_score=classification_result.confidence_score,
            is_calibrated=is_cal,
            uncalibrated_confidence=uncal_conf,
            probabilities=classification_result.probabilities,
            mask=final_mask,
            expected_shape=(img_bgr.shape[0], img_bgr.shape[1]),
            tumor_area_mm2=segmentation_metrics.tumor_area_mm2,
            heatmap=heatmap,
            overlap_percentage=xai_result.overlap_percentage,
            pixel_spacing_mm=intake.pixel_spacing_mm,
            brain_pixels=getattr(segmentation_metrics, "estimated_brain_pixel_count", None)
        )
        quality_warnings = engine_result["warnings"]

        # Generate Clinical Insight (B6.15)
        from clinical_insight.application.use_cases import GenerateClinicalInsightUseCase
        insight_use_case = GenerateClinicalInsightUseCase()
        
        solidity_val = None
        circularity_val = None
        if segmentation_metrics and getattr(segmentation_metrics, "stats", None) is not None:
            solidity_val = segmentation_metrics.stats.solidity
            circularity_val = segmentation_metrics.stats.circularity
            
        clinical_insight_res = insight_use_case.execute(
            predicted_class=classification_result.class_name,
            confidence_score=classification_result.confidence_score,
            is_calibrated=is_cal,
            probabilities=classification_result.probabilities,
            tumor_area_mm2=segmentation_metrics.tumor_area_mm2,
            pixel_count=segmentation_metrics.pixel_count,
            solidity=solidity_val,
            circularity=circularity_val,
            xai_method=xai_param,
            xai_overlap_percentage=xai_result.overlap_percentage,
            longitudinal_comparison=None
        )

        clinical_report = ClinicalReport(
            patient_info=patient_info,
            processing_summary=processing_summary,
            classification=classification_result,
            segmentation_metrics=segmentation_metrics,
            severity_assessment=severity_assessment,
            original_image_path=filepath,
            heatmap_image_path=heatmap_path,
            overlay_image_path=overlay_path,
            segmentation_mask_path=mask_path,
            comparison_image_path=comparison_path,
            xai_method=xai_param,
            xai_explanation_text=xai_result.explanation_text,
            xai_overlap_percentage=xai_result.overlap_percentage,
            quality_warnings=quality_warnings,
            clinical_insight=clinical_insight_res,
        )

        timeline["Clinical Report"] = time.time() - t_endpoint_start

        generator = MarkdownJSONReportGenerator()
        report_use_case = GenerateIntegratedReportUseCase(report_generator=generator, logger=logger)
        md_file, json_file, pdf_file = report_use_case.execute(report=clinical_report, output_dir=OUTPUT_REPORTS_DIR)

        timeline["PDF"] = time.time() - t_endpoint_start

        # Run Email Reporter (B6.15)
        from clinical_reporting.infrastructure.email_sender import ClinicalEmailReporter
        email_reporter = ClinicalEmailReporter(logger=logger)
        recipient_email = f"{intake.ref_physician.replace(' ', '_').lower()}@hospital.org"
        email_reporter.send_report_email(
            report=clinical_report,
            recipient_email=recipient_email,
            output_dir=OUTPUT_REPORTS_DIR
        )

        # 7. Database Persistence
        db_repo = SQLitePersistenceRepository(db_path=DEFAULT_DB_PATH)
        db_repo.initialize_db()
        report_db_id = db_repo.save_report(clinical_report, output_dir=OUTPUT_REPORTS_DIR)

        timeline["Database"] = time.time() - t_endpoint_start
        timeline["Completed"] = time.time() - t_endpoint_start

        # Link validation record to prediction ID in DB and Save Audit Log (B6.6)
        pred_id = None
        database_status = "Failed"
        try:
            import json
            conn = db_repo._get_connection()
            row = conn.execute("SELECT prediction_id FROM clinical_reports WHERE id = ?", (report_db_id,)).fetchone()
            if row:
                pred_id = row["prediction_id"]
                db_repo.save_validation_scorecard(
                    file_hash=scorecard.duplicate_check.duplicate_hash,
                    p_hash=validator.compute_perceptual_hash(file_bytes),
                    is_valid=scorecard.is_valid,
                    scorecard_json=json.dumps(scorecard.to_dict()),
                    prediction_id=pred_id
                )
                database_status = "Persisted"
                # Save Timeline Trace (B6.11, B6.14)
                db_repo.save_timeline_trace(pred_id, timeline)
            conn.close()
        except Exception as db_link_err:
            logger.error(f"Failed to link validation scorecard: {db_link_err}")

        # Record AI Audit log entry
        try:
            from monitoring.infrastructure.audit_logger import AuditLogger
            import multiprocessing
            cpu_threads = multiprocessing.cpu_count()
            gpu_active = torch.cuda.is_available() and str(device) != "cpu"
            
            audit_logger = AuditLogger(db_path=DEFAULT_DB_PATH)
            audit_logger.log_execution(
                patient_id=intake.patient_id,
                user=intake.ref_physician or "Dr. System Default",
                model_version_cls=os.path.basename(CLS_CHECKPOINT),
                model_version_seg=os.path.basename(SEG_CHECKPOINT),
                runtime_sec=total_exec_time,
                gpu_active=gpu_active,
                cpu_threads=cpu_threads,
                warnings=quality_warnings,
                errors=[],
                report_status="Generated",
                database_status=database_status,
                prediction_id=pred_id
            )
        except Exception as audit_err:
            logger.error(f"Failed to record AI Audit Log: {audit_err}")

        return {
            "report_id": report_db_id,
            "patient_id": intake.patient_id,
            "diagnosis": classification_result.class_name,
            "confidence": float(classification_result.confidence_score),
            "severity": severity_assessment.category.value,
            "tumor_area_mm2": float(segmentation_metrics.tumor_area_mm2),
            "execution_time_sec": total_exec_time,
            "xai_method": xai_param,
            "xai_explanation": xai_result.explanation_text,
            "xai_overlap_percentage": xai_result.overlap_percentage,
            "quality_warnings": quality_warnings,
            "files": {
                "pdf": pdf_file,
                "json": json_file,
                "markdown": md_file
            }
        }
    except Exception as e:
        logger.error(f"End-to-end report generation pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {e}")


@router.get("/report/{report_id}/pdf")
def serve_report_pdf(report_id: int):
    """Streams the compiled PDF document directly to clients."""
    history_repo = SQLitePredictionHistoryRepository(db_path=DEFAULT_DB_PATH)
    paths = history_repo.get_report_paths(report_id)
    
    if not paths or not paths[2]:
        raise HTTPException(status_code=404, detail="PDF report not found in database records.")
    
    pdf_path = paths[2]
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF file missing on server disk.")
    
    return FileResponse(pdf_path, media_type="application/pdf", filename=os.path.basename(pdf_path))


@router.get("/report/{report_id}/visuals/{visual_type}")
def serve_report_visual(report_id: int, visual_type: str):
    """Streams diagnostic visual maps (heatmap, overlay, or mask)."""
    conn = sqlite3.connect(DEFAULT_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT overlay_path, heatmap_path, mask_path FROM clinical_reports WHERE id = ?;", (report_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Report visual not found.")
        
        if visual_type == "overlay":
            img_path = row["overlay_path"]
        elif visual_type == "heatmap":
            img_path = row["heatmap_path"]
        elif visual_type == "mask":
            img_path = row["mask_path"]
        else:
            raise HTTPException(status_code=400, detail="Invalid visual type. Choose 'overlay', 'heatmap', or 'mask'.")
        
        if not img_path or not os.path.exists(img_path):
            raise HTTPException(status_code=404, detail="Image file missing on server disk.")
        
        media_type = "image/png"
        if img_path.lower().endswith(".jpg") or img_path.lower().endswith(".jpeg"):
            media_type = "image/jpeg"
            
        return FileResponse(img_path, media_type=media_type)
    finally:
        conn.close()


@router.get("/database/history")
def get_prediction_history(patient_id: Optional[str] = Query(None)):
    """API Endpoint: Retrieves scan prediction logs matching patient ID search filters."""
    history_repo = SQLitePredictionHistoryRepository(db_path=DEFAULT_DB_PATH)
    criteria = HistorySearchCriteria(patient_id=patient_id if patient_id else None)
    
    try:
        summaries = history_repo.search_history(criteria)
        results = []
        for s in summaries:
            results.append({
                "report_id": s.report_id,
                "prediction_id": s.prediction_id,
                "patient_id": s.patient_id,
                "patient_name": s.patient_name,
                "scan_date": s.scan_date,
                "predicted_class": s.predicted_class,
                "confidence_score": float(s.confidence_score),
                "tumor_area_mm2": float(s.tumor_area_mm2),
                "rule_based_severity": s.rule_based_severity,
                "created_at": s.created_at,
            })
        return results
    except Exception as e:
        logger.error(f"Error querying database history: {e}")
        raise HTTPException(status_code=500, detail=f"Database query failed: {e}")


@router.get("/dashboard/analytics")
def get_dashboard_telemetry():
    """API Endpoint: returns analytics distribution graphs counts."""
    persistence_repo = SQLitePersistenceRepository(db_path=DEFAULT_DB_PATH)
    try:
        summary = persistence_repo.get_analytics_summary()
        return summary
    except Exception as e:
        logger.error(f"Error compiling analytics widgets: {e}")
        raise HTTPException(status_code=500, detail=f"Analytics telemetry compilation failed: {e}")
