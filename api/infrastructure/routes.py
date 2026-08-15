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
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Query, Form
from fastapi.responses import FileResponse, JSONResponse
from contextlib import asynccontextmanager
from typing import Optional, List, Any, Dict

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
from security.domain.entities import Role, User
from api.routes.auth_routes import get_current_user, require_roles
from pydantic import BaseModel
from clinical_reporting.application.services import ReportService


# Configure paths
DEFAULT_DB_PATH = os.environ.get("DB_PATH", "outputs/clinical_reports.db")
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

classification_model_provenance_id = None
segmentation_model_provenance_id = None

logger = logging.getLogger("api_routes")


def _calculate_file_sha256(filepath: str) -> str:
    import hashlib
    if not filepath:
        raise ValueError("Filepath is empty or None")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint file not found: {filepath}")
    if not os.path.isfile(filepath):
        raise ValueError(f"Path is not a file: {filepath}")

    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest().lower()



def initialize_api_models():
    """Preloads the deep learning model states into memory."""
    global model_cls, predict_use_case, model_seg, seg_config, device
    global classification_model_provenance_id, segmentation_model_provenance_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"FastAPI API initializing models on device: {device}")

    # Thread tuning for CPU fallback to avoid thrashing
    if device.type == "cpu" and torch.get_num_threads() > 4:
        torch.set_num_threads(4)

    # Initialize repository schema to perform migrations
    db_repo = SQLitePersistenceRepository(db_path=DEFAULT_DB_PATH)
    db_repo.initialize_db()

    loaded_at_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Reset globals in case of re-initialization/failure
    classification_model_provenance_id = None
    segmentation_model_provenance_id = None
    model_cls = None
    model_seg = None
    predict_use_case = None

    try:
        # Load classification pipeline
        model_cls = EfficientNetB0Model(pretrained=False, num_classes=4)
        model_adapter = PyTorchModelAdapter(model=model_cls, device=str(device))
        model_adapter.load(CLS_CHECKPOINT)
        predict_use_case = PredictUseCase(model_adapter=model_adapter)
        logger.info("Classification model loaded successfully.")

        # Register classification model provenance (idempotent lookup/insert)
        cls_hash = _calculate_file_sha256(CLS_CHECKPOINT)
        classification_model_provenance_id = db_repo.register_model_provenance(
            model_type="CLASSIFICATION",
            model_name="efficientnet_b0",
            architecture="EfficientNet-B0",
            model_version="1.0",
            checkpoint_identifier="classification/efficientnet_b0_brain_tumor.pth",
            checkpoint_sha256=cls_hash,
            device=str(device),
            loaded_at=loaded_at_str
        )
        logger.info(f"Registered classification model provenance ID: {classification_model_provenance_id}")

        # Load UNeXt segmentation pipeline
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

        # Register segmentation model provenance (idempotent lookup/insert)
        seg_hash = _calculate_file_sha256(SEG_CHECKPOINT)
        segmentation_model_provenance_id = db_repo.register_model_provenance(
            model_type="SEGMENTATION",
            model_name="brain_tumor_unext",
            architecture=seg_config.get("arch", "UNeXt"),
            model_version="1.0",
            checkpoint_identifier="brain_tumor_unext/model.pth",
            checkpoint_sha256=seg_hash,
            device=str(device),
            loaded_at=loaded_at_str
        )
        logger.info(f"Registered segmentation model provenance ID: {segmentation_model_provenance_id}")
    except Exception as e:
        logger.critical(f"Critical error during API model/provenance initialization: {e}")
        # Clear out any partially loaded state to avoid inconsistent behavior
        classification_model_provenance_id = None
        segmentation_model_provenance_id = None
        model_cls = None
        model_seg = None
        predict_use_case = None
        raise e


def preprocess_segmentation_image(img_bgr: np.ndarray, h: int, w: int) -> torch.Tensor:
    """Preprocesses BGR image slice for segmentation model using simple [0, 1] scaling matching BraTS training."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (w, h))
    img_tensor = (img_resized.astype(np.float32) / 255.0).transpose(2, 0, 1)  # C, H, W
    return torch.from_numpy(img_tensor).unsqueeze(0)  # 1, C, H, W


def validate_version_param(version_val: Any) -> Optional[int]:
    """Validates the version parameter strictly.
    Returns:
        The validated version as an int, or None if the parameter was not provided (to use current version).
    Raises:
        ValueError if the version parameter is malformed.
    """
    if version_val is None:
        return None

    # If it's a string, strip it and check if it's empty
    if isinstance(version_val, str):
        val_str = version_val.strip()
        if not val_str:
            raise ValueError("Version cannot be empty or whitespace.")
        # Check if it has a decimal point, reject
        if "." in val_str:
            raise ValueError("Version must be a positive integer, not a decimal.")
        # Reject booleans represented as strings
        if val_str.lower() in ("true", "false"):
            raise ValueError("Version cannot be a boolean.")
        try:
            val_int = int(val_str)
        except ValueError:
            raise ValueError("Version must be a valid integer.")
    elif isinstance(version_val, bool):
        # Python's bool is a subclass of int, so isinstance(True, int) is True! We must check bool explicitly.
        raise ValueError("Version cannot be a boolean.")
    elif isinstance(version_val, float):
        raise ValueError("Version must be an integer, not a float.")
    elif isinstance(version_val, int):
        val_int = version_val
    else:
        raise ValueError("Invalid version type.")

    if val_int <= 0:
        raise ValueError("Version must be a positive integer.")
    return val_int


# FastAPI Router
router = APIRouter()


@router.post("/upload")
def upload_mri_file(file: UploadFile = File(...), current_user: User = Depends(require_roles([Role.ADMIN, Role.DOCTOR]))):
    """API Endpoint: Receives a raw brain MRI image slice upload and validates it."""
    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
    temp_filename = f"upload_{int(time.time())}_{file.filename}"
    temp_filepath = os.path.join(TEMP_UPLOAD_DIR, temp_filename)

    from security.infrastructure.repository import SQLiteUserRepository
    from security.domain.entities import SecurityAuditLog
    import datetime
    user_repo = SQLiteUserRepository(db_path=DEFAULT_DB_PATH)
    now = datetime.datetime.utcnow().isoformat()

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

            # Log validation failure
            user_repo.log_security_event(SecurityAuditLog(
                id=None,
                timestamp=now,
                event_type="MRI_UPLOAD",
                user_id=current_user.id if current_user else None,
                email=current_user.email if current_user else None,
                ip_address="127.0.0.1",
                status="FAILED",
                details=f"MRI validation failed for filename: {file.filename}. Reasons: {', '.join(scorecard.errors)}",
                user_agent="System"
            ))

            raise HTTPException(
                status_code=400,
                detail={
                    "message": "MRI Input Validation Failed",
                    "errors": scorecard.errors,
                    "scorecard": scorecard.to_dict()
                }
            )

        # Log successful upload
        user_repo.log_security_event(SecurityAuditLog(
            id=None,
            timestamp=now,
            event_type="MRI_UPLOAD",
            user_id=current_user.id if current_user else None,
            email=current_user.email if current_user else None,
            ip_address="127.0.0.1",
            status="SUCCESS",
            details=f"MRI file uploaded and validated successfully. Filename: {file.filename}",
            user_agent="System"
        ))

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

        # Log unexpected error
        user_repo.log_security_event(SecurityAuditLog(
            id=None,
            timestamp=now,
            event_type="MRI_UPLOAD",
            user_id=current_user.id if current_user else None,
            email=current_user.email if current_user else None,
            ip_address="127.0.0.1",
            status="FAILED",
            details="MRI upload failed due to internal error.",
            user_agent="System"
        ))

        raise HTTPException(status_code=500, detail=f"File upload processing failed: {e}")



@router.post("/classification")
def run_classification(filepath: str, current_user: User = Depends(require_roles([Role.ADMIN, Role.DOCTOR]))):
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
def run_segmentation(filepath: str, current_user: User = Depends(require_roles([Role.ADMIN, Role.DOCTOR]))):
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
def run_explainability(filepath: str, target_class: int = 1, method: str = "gradcam", current_user: User = Depends(require_roles([Role.ADMIN, Role.DOCTOR]))):
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


@router.get("/ping")
def ping():
    """Unauthenticated lightweight liveness probe endpoint."""
    return {"status": "healthy", "timestamp": datetime.datetime.utcnow().isoformat()}


@router.get("/health")
def get_pipeline_health():
    """Unauthenticated lightweight liveness/readiness health check endpoint."""
    return {"status": "healthy"}


@router.get("/health-telemetry")
def get_pipeline_health_telemetry(current_user: User = Depends(require_roles([Role.ADMIN]))):
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
        logger.error(f"Health telemetry diagnostics failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health telemetry diagnostics failed: {e}")



@router.post("/report")
def generate_clinical_report_pipeline(filepath: str, intake: PatientIntake, current_user: User = Depends(require_roles([Role.ADMIN, Role.DOCTOR]))):
    """API Endpoint: Runs the complete end-to-end MRI diagnostics report pipeline with validation."""
    from security.application.authorization_service import AuthorizationService
    auth_svc = AuthorizationService(DEFAULT_DB_PATH)
    if not auth_svc.can_access_patient(current_user, intake.patient_id):
        raise HTTPException(status_code=403, detail="Access denied to patient records.")

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

        return _run_single_report_pipeline(
            filepath=filepath,
            intake=intake,
            current_user=current_user,
            scorecard=scorecard,
            file_bytes=file_bytes,
            validator=validator,
            t_start=t_start,
            t_endpoint_start=t_endpoint_start,
            timeline=timeline
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"End-to-end report generation pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {e}")


def _run_single_report_pipeline(
    filepath: str,
    intake: PatientIntake,
    current_user: User,
    scorecard: Any,
    file_bytes: bytes,
    validator: Any,
    t_start: float,
    t_endpoint_start: float,
    timeline: Dict[str, float]
):
    if (
        not isinstance(classification_model_provenance_id, int)
        or not isinstance(segmentation_model_provenance_id, int)
        or model_cls is None
        or model_seg is None
        or predict_use_case is None
    ):
        logger.error(
            f"Provenance gate blocked report generation: "
            f"classification_model_provenance_id={classification_model_provenance_id}, "
            f"segmentation_model_provenance_id={segmentation_model_provenance_id}, "
            f"model_cls={model_cls is not None}, "
            f"model_seg={model_seg is not None}, "
            f"predict_use_case={predict_use_case is not None}"
        )
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred."
        )

    try:
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
            if getattr(intake, "ensemble_mode", False):
                from research_framework.application.registry import ModelRegistry
                from research_framework.application.ensemble import EnsembleEngine

                registry = ModelRegistry(default_checkpoint_path=CLS_CHECKPOINT)
                predictions = registry.predict_all(image_tensor_cls, device=device)

                # Fetch baseline prediction specifically to keep classification_result populated
                prod_pred = next((p for p in predictions if p.model_name == "efficientnet_b0"), None)
                if prod_pred:
                    uncal_conf = getattr(prod_pred, "uncalibrated_confidence_score", prod_pred.confidence)
                    classification_result = PredictionResult(
                        label=["Glioma", "Meningioma", "Pituitary", "No Tumor"].index(prod_pred.predicted_class),
                        class_name=prod_pred.predicted_class,
                        confidence_score=prod_pred.confidence,
                        probabilities=prod_pred.probabilities,
                        uncalibrated_confidence_score=uncal_conf,
                        uncalibrated_probabilities=prod_pred.probabilities,
                        is_calibrated=False
                    )
                else:
                    classification_result = predict_use_case.execute(image_tensor_cls)

                engine = EnsembleEngine()
                ensemble_res = engine.compute_ensemble(predictions)

                # Swap production result with ensemble prediction
                class_to_label = {"Glioma": 0, "Meningioma": 1, "Pituitary": 2, "No Tumor": 3}
                ensemble_label = class_to_label.get(ensemble_res.predicted_class, 3)

                classification_result = PredictionResult(
                    label=ensemble_label,
                    class_name=ensemble_res.predicted_class,
                    confidence_score=ensemble_res.confidence,
                    probabilities=ensemble_res.probabilities,
                    uncalibrated_confidence_score=classification_result.uncalibrated_confidence_score,
                    uncalibrated_probabilities=classification_result.uncalibrated_probabilities,
                    calibration_method="Ensemble (Soft Voting)",
                    calibration_parameters={"num_models": len(predictions)},
                    is_calibrated=True
                )
            else:
                classification_result = predict_use_case.execute(image_tensor_cls)
        except Exception as e:
            logger.warning(f"Classification failed on {device}: {e}. Retrying with CPU fallback...")
            try:
                model_cls.to("cpu")
                predict_use_case.model_adapter.device = torch.device("cpu")
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

        segmentation_failed = False
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
                if model_seg is not None:
                    model_seg.to("cpu")
                active_seg_device = "cpu"
                input_tensor_seg_cpu = input_tensor_seg.to("cpu")
                with torch.inference_mode():
                    if model_seg is not None:
                        output_seg = model_seg(input_tensor_seg_cpu)
                        if seg_config["deep_supervision"]:
                            output_seg = output_seg[-1]
                        output_seg = torch.sigmoid(output_seg).squeeze(0).squeeze(0).cpu().numpy()
                    else:
                        raise RuntimeError("Segmentation model is not loaded (None)")
                seg_warnings.append("Auto-recovery warning: Segmentation execution failed on GPU. Retried and completed on CPU fallback mode.")
            except Exception as cpu_err:
                logger.critical(f"CPU fallback for segmentation failed: {cpu_err}")
                segmentation_failed = True
                h_shape = seg_config["input_h"] if (seg_config and "input_h" in seg_config) else 224
                w_shape = seg_config["input_w"] if (seg_config and "input_w" in seg_config) else 224
                output_seg = np.zeros((h_shape, w_shape), dtype=np.float32)
                seg_warnings.append(f"Critical fallback: Segmentation engine failed completely ({cpu_err}). Generated empty tumor mask. segmentation execution failed")

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

        success_h = cv2.imwrite(heatmap_path, heatmap_color)
        if not success_h:
            logger.error(f"Failed to write heatmap image to: {heatmap_path}")

        success_o = cv2.imwrite(overlay_path, overlay_with_contour)
        if not success_o:
            logger.error(f"Failed to write overlay image to: {overlay_path}")

        # Save post-processed segmentation mask
        mask_filename = f"{intake.patient_id}_api_mask.png"
        mask_path = os.path.join(OUTPUT_REPORTS_DIR, mask_filename)
        success_m = cv2.imwrite(mask_path, (final_mask * 255).astype(np.uint8))
        if not success_m:
            logger.error(f"Failed to write segmentation mask to: {mask_path}")

        # Save before-after post-processing comparison image
        comparison_filename = f"{intake.patient_id}_api_comparison.png"
        comparison_path = os.path.join(OUTPUT_REPORTS_DIR, comparison_filename)
        create_segmentation_comparison_image(
            original_image=img_bgr,
            before_mask=bin_mask_resized,
            after_mask=final_mask,
            output_path=comparison_path
        )

        # 4. Morphological Analysis
        morph_analyzer = OpenCVTumorAnalyzer(low_thresh=1.0, med_thresh=5.0, high_thresh=15.0)
        morph_use_case = AnalyzeTumorUseCase(analyzer=morph_analyzer, logger=logger)

        class DummyClinicalData:
            def __init__(self):
                from tumor_analysis.domain.entities import TumorAnalysisResult, SeverityLevel
                self.analysis = TumorAnalysisResult(
                    pixel_count=0,
                    tumor_area_mm2=0.0,
                    tumor_percentage_brain=0.0,
                    tumor_percentage_image=0.0,
                    estimated_brain_pixel_count=0,
                    severity_level=SeverityLevel.LOW,
                    metadata={},
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
                segmentation_failed=segmentation_failed,
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
            segmentation_failed=segmentation_failed,
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
            classification_model_path="classification/efficientnet_b0_brain_tumor.pth",
            segmentation_model_path="brain_tumor_unext/model.pth",
            classification_latency_sec=cls_latency,
            segmentation_latency_sec=seg_latency,
            explainability_latency_sec=cam_latency,
            classification_model_provenance_id=classification_model_provenance_id,
            segmentation_model_provenance_id=segmentation_model_provenance_id
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
            longitudinal_comparison=None,
            segmentation_failed=segmentation_failed
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

        # Trigger in-app notifications (G8.2.7)
        try:
            from clinical_reporting.application.notification_service import NotificationService
            from security.infrastructure.repository import SQLiteUserRepository
            import json

            notif_svc = NotificationService(db_path=DEFAULT_DB_PATH)
            user_repo = SQLiteUserRepository(db_path=DEFAULT_DB_PATH)

            # Resolve doctor and patient users
            doctor_id = current_user.id if current_user else None
            patient_user = user_repo.get_by_uuid(clinical_report.patient_info.patient_id)

            # Prepare notifications
            analysis_title = "MRI Analysis Completed"
            analysis_msg = f"MRI Scan analysis completed for patient {clinical_report.patient_info.name} ({clinical_report.patient_info.patient_id}). Diagnosis: {clinical_report.classification.class_name}."
            analysis_meta = json.dumps({"report_id": report_db_id, "patient_id": clinical_report.patient_info.patient_id})

            report_title = "Clinical Report Ready"
            report_msg = f"Clinical report has been successfully generated for scan date {clinical_report.patient_info.scan_date}."
            report_meta = json.dumps({"report_id": report_db_id, "patient_id": clinical_report.patient_info.patient_id})

            # Send to doctor
            if doctor_id:
                notif_svc.create_notification(
                    user_id=doctor_id,
                    type_="ANALYSIS_COMPLETED",
                    title=analysis_title,
                    message=analysis_msg,
                    metadata_json=analysis_meta
                )
                notif_svc.create_notification(
                    user_id=doctor_id,
                    type_="REPORT_READY",
                    title=report_title,
                    message=report_msg,
                    metadata_json=report_meta
                )

            # Send to patient
            if patient_user and patient_user.id and patient_user.id != doctor_id:
                notif_svc.create_notification(
                    user_id=patient_user.id,
                    type_="ANALYSIS_COMPLETED",
                    title=analysis_title,
                    message=analysis_msg,
                    metadata_json=analysis_meta
                )
                notif_svc.create_notification(
                    user_id=patient_user.id,
                    type_="REPORT_READY",
                    title=report_title,
                    message=report_msg,
                    metadata_json=report_meta
                )
        except Exception as notif_err:
            logger.error(f"Failed to trigger G8.2 notifications: {notif_err}")

        # Record Security Audit Log for report creation (F2.4)
        try:
            from clinical_reporting.application.services import ReportService
            service = ReportService(db_path=DEFAULT_DB_PATH)
            service.log_report_access_event(
                "REPORT_LIFECYCLE_CHANGE",
                current_user,
                report_db_id,
                "SUCCESS",
                "Action: CREATE_REPORT, Prev Status: NONE, New Status: GENERATED"
            )
        except Exception as audit_err:
            logger.error(f"Failed to record Report Creation Security Audit Log: {audit_err}")

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

        ensemble_data = None
        if getattr(intake, "ensemble_mode", False) and 'ensemble_res' in locals():
            ensemble_data = {
                "predicted_class": ensemble_res.predicted_class,
                "confidence": ensemble_res.confidence,
                "probabilities": ensemble_res.probabilities,
                "agreement_level": ensemble_res.agreement_metrics.level if ensemble_res.agreement_metrics else "N/A",
                "cosine_similarity": ensemble_res.agreement_metrics.cosine_similarity if ensemble_res.agreement_metrics else 1.0
            }

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
            "ensemble": ensemble_data,
            "files": {
                "pdf": pdf_file,
                "json": json_file,
                "markdown": md_file
            }
        }
    except Exception as e:
        logger.error(f"End-to-end report generation pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline execution failed: {e}")


@router.post("/report/batch")
async def generate_clinical_report_batch(
    files: List[UploadFile] = File(...),
    patient_id: str = Form(...),
    name: str = Form(...),
    age: int = Form(...),
    gender: str = Form("Female"),
    ref_physician: str = Form("Dr. Unknown"),
    pixel_spacing_mm: float = Form(1.0),
    xai_method: Optional[str] = Form("gradcam"),
    ensemble_mode: Optional[str] = Form("false"),
    current_user: User = Depends(require_roles([Role.ADMIN, Role.DOCTOR]))
):
    """API Endpoint: Runs the complete diagnostics pipeline on multiple MRI files in one batch."""
    from security.application.authorization_service import AuthorizationService
    auth_svc = AuthorizationService(DEFAULT_DB_PATH)
    if not auth_svc.can_access_patient(current_user, patient_id):
        raise HTTPException(status_code=403, detail="Access denied to patient records.")

    if not files:
        raise HTTPException(status_code=400, detail="No files provided in batch.")

    import uuid
    import hashlib
    batch_id = str(uuid.uuid4())
    t_batch_start = time.time()
    t_batch_perf_start = time.perf_counter()

    # Parse ensemble mode string
    ensemble_val = ensemble_mode.lower() == "true"

    intake = PatientIntake(
        patient_id=patient_id,
        name=name,
        age=age,
        gender=gender,
        ref_physician=ref_physician,
        pixel_spacing_mm=pixel_spacing_mm,
        xai_method=xai_method,
        ensemble_mode=ensemble_val
    )

    stats = {
        "total_files": len(files),
        "valid_files": 0,
        "invalid_files": 0,
        "duplicates": 0,
        "successful": 0,
        "failed": 0,
        "total_duration_sec": 0.0,
        "batch_id": batch_id
    }

    results = []
    seen_hashes = set()

    from input_validation.infrastructure.validators import OpenCVMriValidator
    from input_validation.application.use_cases import ValidateMriUploadUseCase
    db_repo = SQLitePersistenceRepository(db_path=DEFAULT_DB_PATH)

    validator = OpenCVMriValidator()
    use_case = ValidateMriUploadUseCase(validator=validator, db_path=DEFAULT_DB_PATH)

    for upload_file in files:
        t_file_start = time.time()
        file_bytes = await upload_file.read()
        filename = upload_file.filename or "unknown_file"

        # Check empty file
        if not file_bytes:
            stats["invalid_files"] += 1
            stats["failed"] += 1
            results.append({
                "filename": filename,
                "status": "FAILED",
                "error_code": "VALIDATION_ERROR",
                "error_message": "Empty file uploaded.",
                "prediction": None,
                "confidence": None,
                "segmentation_status": None,
                "tumor_area": None,
                "tumor_percentage": None,
                "severity": None,
                "processing_time": 0.0,
                "scan_identifier": None
            })
            continue

        # Check content hash duplicate within batch
        content_hash = hashlib.sha256(file_bytes).hexdigest()
        if content_hash in seen_hashes:
            stats["duplicates"] += 1
            stats["failed"] += 1
            results.append({
                "filename": filename,
                "status": "FAILED",
                "error_code": "DUPLICATE_FILE",
                "error_message": "Duplicate scan detected. This file is identical to another scan in the same batch.",
                "prediction": None,
                "confidence": None,
                "segmentation_status": None,
                "tumor_area": None,
                "tumor_percentage": None,
                "severity": None,
                "processing_time": 0.0,
                "scan_identifier": None
            })
            continue

        seen_hashes.add(content_hash)

        # Save to temp file
        os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
        temp_filename = f"batch_{batch_id}_{int(time.time())}_{filename}"
        temp_filepath = os.path.join(TEMP_UPLOAD_DIR, temp_filename)
        with open(temp_filepath, "wb") as f:
            f.write(file_bytes)

        try:
            # 1. Run Validation
            scorecard = use_case.execute(filepath=temp_filepath, file_bytes=file_bytes, filename=filename)

            if not scorecard.is_valid:
                stats["invalid_files"] += 1
                stats["failed"] += 1
                # Classify error type
                err_code = "VALIDATION_ERROR"
                err_msg = ", ".join(scorecard.errors)
                if any("duplicate" in err.lower() for err in scorecard.errors):
                    err_code = "DUPLICATE_FILE"
                    stats["duplicates"] += 1
                    stats["invalid_files"] -= 1  # adjust since duplicates are counted separately
                elif any(ext in filename.lower() for ext in [".dcm", ".dicom"]):
                    err_code = "UNSUPPORTED_FORMAT"
                elif any("format" in err.lower() or "extension" in err.lower() for err in scorecard.errors):
                    err_code = "UNSUPPORTED_FORMAT"
                elif any("corrupt" in err.lower() or "read" in err.lower() or "integrity" in err.lower() for err in scorecard.errors):
                    err_code = "CORRUPTED_IMAGE"

                results.append({
                    "filename": filename,
                    "status": "FAILED",
                    "error_code": err_code,
                    "error_message": err_msg,
                    "prediction": None,
                    "confidence": None,
                    "segmentation_status": None,
                    "tumor_area": None,
                    "tumor_percentage": None,
                    "severity": None,
                    "processing_time": time.time() - t_file_start,
                    "scan_identifier": None
                })
                # Clean up temp file
                if os.path.exists(temp_filepath):
                    try:
                        os.remove(temp_filepath)
                    except Exception:
                        pass
                continue

            stats["valid_files"] += 1

            # 2. Run existing prediction pipeline
            timeline = {
                "Upload": 0.0,
                "Validation": time.time() - t_file_start
            }
            res_report = _run_single_report_pipeline(
                filepath=temp_filepath,
                intake=intake,
                current_user=current_user,
                scorecard=scorecard,
                file_bytes=file_bytes,
                validator=validator,
                t_start=t_file_start,
                t_endpoint_start=t_file_start,
                timeline=timeline
            )

            # Retrieve tumor_percentage_brain from SQLite DB
            tumor_pct = 0.0
            try:
                conn = db_repo._get_connection()
                row = conn.execute("""
                    SELECT p.tumor_percentage_brain
                    FROM predictions p
                    JOIN clinical_reports cr ON cr.prediction_id = p.id
                    WHERE cr.id = ?
                """, (res_report["report_id"],)).fetchone()
                if row:
                    tumor_pct = float(row["tumor_percentage_brain"])
                conn.close()
            except Exception as db_pct_err:
                logger.error(f"Failed to query tumor percentage from DB: {db_pct_err}")

            stats["successful"] += 1
            results.append({
                "filename": filename,
                "status": "SUCCESS",
                "error_code": None,
                "error_message": None,
                "prediction": res_report["diagnosis"],
                "confidence": res_report["confidence"],
                "segmentation_status": "SUCCESS" if res_report["tumor_area_mm2"] >= 0 else "FAILED",
                "tumor_area": res_report["tumor_area_mm2"],
                "tumor_percentage": tumor_pct,
                "severity": res_report["severity"],
                "processing_time": time.time() - t_file_start,
                "scan_identifier": res_report["report_id"]
            })

        except Exception as e:
            logger.error(f"Batch item execution failed for {filename}: {e}")
            stats["failed"] += 1
            # Classify execution error
            err_code = "INFERENCE_ERROR"
            err_msg = str(e)
            if "database" in err_msg.lower() or "sqlite" in err_msg.lower():
                err_code = "DATABASE_ERROR"
            elif "segmentation" in err_msg.lower():
                err_code = "SEGMENTATION_ERROR"
            elif "gradcam" in err_msg.lower() or "explain" in err_msg.lower():
                err_code = "EXPLAINABILITY_ERROR"

            results.append({
                "filename": filename,
                "status": "FAILED",
                "error_code": err_code,
                "error_message": f"Inference execution failed: {err_msg}",
                "prediction": None,
                "confidence": None,
                "segmentation_status": None,
                "tumor_area": None,
                "tumor_percentage": None,
                "severity": None,
                "processing_time": time.time() - t_file_start,
                "scan_identifier": None
            })
            # Clean up temp file on failure
            if os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                except Exception:
                    pass

    stats["total_duration_sec"] = time.time() - t_batch_start

    # Record batch performance telemetry safely
    try:
        t_batch_dur_ms = (time.perf_counter() - t_batch_perf_start) * 1000.0
        avg_item_dur_ms = t_batch_dur_ms / len(files) if files else 0.0

        # Save telemetry
        import datetime
        db_path = DEFAULT_DB_PATH
        db_repo = SQLitePersistenceRepository(db_path=db_path)
        db_repo.save_batch_performance_telemetry(
            batch_id=batch_id,
            timestamp=datetime.datetime.utcnow().isoformat(),
            total_items=len(files),
            successful_items=stats["successful"],
            failed_items=stats["failed"],
            total_duration_ms=t_batch_dur_ms,
            average_item_duration_ms=avg_item_dur_ms
        )
    except Exception as telemetry_err:
        logger.error(f"Failed to record batch performance telemetry: {telemetry_err}")

    return {
        "stats": stats,
        "results": results
    }


@router.get("/report/{report_id}/pdf")
def serve_report_pdf(report_id: int, version: Optional[str] = Query(None), current_user: User = Depends(get_current_user)):
    """Streams the compiled PDF document directly to clients with ownership validation."""
    try:
        validated_version = validate_version_param(version)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    from clinical_reporting.application.services import (
        ReportService, ReportNotFoundException, VersionNotFoundException,
        PathTraversalException, IntegrityFailureException
    )
    import logging
    logger = logging.getLogger("api.routes.serve_report_pdf")
    logger.info(f"API request received for PDF. Report ID: {report_id}, User: {current_user.email}, Version: {validated_version}")

    service = ReportService(db_path=DEFAULT_DB_PATH)

    # 1. Enforce access check
    access_status = service.check_report_access(report_id, current_user)
    if access_status == "NOT_FOUND":
        service.log_report_access_event("REPORT_NOT_FOUND", current_user, report_id, "FAILED", "Requested nonexistent report.")
        raise HTTPException(status_code=404, detail="Report not found.")
    elif access_status == "UNAUTHORIZED":
        raise HTTPException(status_code=401, detail="Authentication required.")
    elif access_status == "FORBIDDEN":
        service.log_report_access_event("REPORT_ACCESS_DENIED", current_user, report_id, "FAILED", "Access denied to patient report.")
        raise HTTPException(status_code=403, detail="Access denied to patient report.")

    # 2. Resolve PDF path with security checks and integrity verification
    try:
        pdf_path = service.resolve_secure_pdf_path(report_id, validated_version)
    except (ReportNotFoundException, VersionNotFoundException):
        service.log_report_access_event("REPORT_NOT_FOUND", current_user, report_id, "FAILED", f"Report version {validated_version if validated_version is not None else 'latest'} not found.")
        raise HTTPException(status_code=404, detail="Report not found.")
    except PathTraversalException as pte:
        service.log_report_access_event("REPORT_ACCESS_DENIED", current_user, report_id, "FAILED", f"Path traversal attempt: {pte}")
        raise HTTPException(status_code=400, detail="Invalid report path.")
    except IntegrityFailureException as ife:
        service.log_report_access_event("REPORT_ACCESS_DENIED", current_user, report_id, "FAILED", f"Integrity failure: {ife}")
        raise HTTPException(status_code=422, detail="Report integrity verification failed.")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="PDF report file not found on server disk.")
    except Exception as e:
        logger.error(f"Error resolving PDF path: {e}")
        raise HTTPException(status_code=500, detail="Internal error resolving PDF report.")

    # 3. Log download success and serve
    service.log_report_access_event("REPORT_DOWNLOADED", current_user, report_id, "SUCCESS", f"Downloaded report version {validated_version if validated_version is not None else 'latest'}")
    filename = os.path.basename(pdf_path)
    headers = {
        "Content-Disposition": f"attachment; filename={filename}",
        "X-Content-Type-Options": "nosniff"
    }
    return FileResponse(pdf_path, media_type="application/pdf", filename=filename, headers=headers)


@router.get("/report/{report_id}/json")
def serve_report_json(report_id: int, version: Optional[str] = Query(None), current_user: User = Depends(get_current_user)):
    """Streams the compiled JSON document directly to clients with ownership validation."""
    try:
        validated_version = validate_version_param(version)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    from clinical_reporting.application.services import (
        ReportService, ReportNotFoundException, VersionNotFoundException,
        PathTraversalException, IntegrityFailureException
    )
    import logging
    logger = logging.getLogger("api.routes.serve_report_json")
    logger.info(f"API request received for JSON. Report ID: {report_id}, User: {current_user.email}, Version: {validated_version}")

    service = ReportService(db_path=DEFAULT_DB_PATH)

    # 1. Enforce access check
    access_status = service.check_report_access(report_id, current_user)
    if access_status == "NOT_FOUND":
        service.log_report_access_event("REPORT_NOT_FOUND", current_user, report_id, "FAILED", "Requested nonexistent report.")
        raise HTTPException(status_code=404, detail="Report not found.")
    elif access_status == "UNAUTHORIZED":
        raise HTTPException(status_code=401, detail="Authentication required.")
    elif access_status == "FORBIDDEN":
        service.log_report_access_event("REPORT_ACCESS_DENIED", current_user, report_id, "FAILED", "Access denied to patient report.")
        raise HTTPException(status_code=403, detail="Access denied to patient report.")

    # 2. Get and sanitize JSON payload
    try:
        sanitized_json = service.get_report_json_for_export(report_id, validated_version, current_user)
    except (ReportNotFoundException, VersionNotFoundException):
        raise HTTPException(status_code=404, detail="Report not found.")
    except PathTraversalException as pte:
        raise HTTPException(status_code=400, detail="Invalid report path.")
    except IntegrityFailureException as ife:
        raise HTTPException(status_code=422, detail="Report integrity verification failed.")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="JSON report file not found on server disk.")
    except Exception as e:
        logger.error(f"Error exporting JSON report: {e}")
        raise HTTPException(status_code=500, detail="Internal error exporting JSON report.")

    resolved_version = validated_version
    if resolved_version is None:
        try:
            report = service.get_report(report_id)
            resolved_version = report.current_version
        except Exception:
            resolved_version = 1

    filename = f"report_{report_id}_v{resolved_version}.json"
    headers = {
        "Content-Disposition": f"attachment; filename={filename}",
        "X-Content-Type-Options": "nosniff"
    }
    return JSONResponse(content=sanitized_json, headers=headers)


@router.get("/report/{report_id}/csv")
def serve_report_csv(report_id: int, version: Optional[str] = Query(None), current_user: User = Depends(get_current_user)):
    """Streams the compiled CSV document directly to clients with ownership validation."""
    from clinical_reporting.application.services import (
        ReportService, ReportNotFoundException, VersionNotFoundException,
        PathTraversalException, IntegrityFailureException, ReportServiceException
    )
    import logging
    logger = logging.getLogger("api.routes.serve_report_csv")
    logger.info(f"API request received for CSV. Report ID: {report_id}, User: {current_user.email}, Version: {version}")

    try:
        validated_version = validate_version_param(version)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    service = ReportService(db_path=DEFAULT_DB_PATH)

    try:
        csv_content = service.get_report_csv_for_export(report_id, validated_version, current_user)
    except (ReportNotFoundException, VersionNotFoundException):
        raise HTTPException(status_code=404, detail="Report not found.")
    except PathTraversalException as pte:
        raise HTTPException(status_code=400, detail="Invalid report path.")
    except IntegrityFailureException as ife:
        raise HTTPException(status_code=422, detail="Report integrity verification failed.")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="CSV report source file not found.")
    except ReportServiceException as rse:
        err_msg = str(rse)
        if "Access denied" in err_msg or "denied" in err_msg.lower():
            raise HTTPException(status_code=403, detail="Access denied to patient report.")
        elif "Authentication required" in err_msg or "unauthenticated" in err_msg.lower():
            raise HTTPException(status_code=401, detail="Authentication required.")
        else:
            raise HTTPException(status_code=400, detail=err_msg)
    except Exception as e:
        logger.error(f"Error exporting CSV: {e}")
        raise HTTPException(status_code=500, detail="Internal error resolving CSV report.")

    resolved_version = validated_version
    if resolved_version is None:
        try:
            report = service.get_report(report_id)
            resolved_version = report.current_version
        except Exception:
            resolved_version = 1

    filename = f"report_{report_id}_v{resolved_version}.csv"
    headers = {
        "Content-Disposition": f"attachment; filename={filename}",
        "X-Content-Type-Options": "nosniff"
    }
    from fastapi.responses import Response
    return Response(content=csv_content, media_type="text/csv", headers=headers)


@router.get("/report/{report_id}/visuals/{visual_type}")
def serve_report_visual(report_id: int, visual_type: str, current_user: User = Depends(require_roles([Role.ADMIN, Role.DOCTOR]))):
    """Streams diagnostic visual maps (heatmap, overlay, mask, or raw). Only allowed for Admins and Doctors."""
    from clinical_reporting.application.services import ReportService
    service = ReportService(db_path=DEFAULT_DB_PATH)
    access_status = service.check_report_access(report_id, current_user)
    if access_status == "NOT_FOUND":
        raise HTTPException(status_code=404, detail="Report not found.")
    elif access_status == "UNAUTHORIZED":
        raise HTTPException(status_code=401, detail="Authentication required.")
    elif access_status == "FORBIDDEN":
        raise HTTPException(status_code=403, detail="Access denied to patient report visual.")

    conn = sqlite3.connect(DEFAULT_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT cr.overlay_path, cr.heatmap_path, cr.mask_path, s.image_path as raw_path
            FROM clinical_reports cr
            JOIN predictions pr ON cr.prediction_id = pr.id
            JOIN mri_scans s ON pr.scan_id = s.id
            WHERE cr.id = ?;
            """,
            (report_id,)
        )
        row = cursor.fetchone()
        if not row:
            try:
                service.log_report_access_event(
                    "REPORT_NOT_FOUND",
                    current_user,
                    report_id,
                    "FAILED",
                    f"Report not found for visual fetch: {visual_type}"
                )
            except Exception as e:
                logger.error(f"Audit log failed: {e}")
        else:
            try:
                service.log_report_access_event(
                    "REPORT_VIEWED",
                    current_user,
                    report_id,
                    "SUCCESS",
                    f"Viewed report visual: {visual_type}"
                )
            except Exception as e:
                logger.error(f"Audit log failed: {e}")

        img_path = None
        if row:
            if visual_type == "overlay":
                img_path = row["overlay_path"]
            elif visual_type == "heatmap":
                img_path = row["heatmap_path"]
            elif visual_type == "mask":
                img_path = row["mask_path"]
            elif visual_type == "raw":
                img_path = row["raw_path"]
            else:
                img_path = None

        if img_path:
            img_path = os.path.abspath(img_path)

        if not img_path or not os.path.exists(img_path):
            import numpy as np
            import cv2

            # Generate placeholder image on the fly
            placeholder = np.zeros((400, 400, 3), dtype=np.uint8)
            placeholder[:] = [42, 23, 15]  # Slate color
            cv2.circle(placeholder, (200, 200), 120, (85, 65, 51), thickness=2)
            cv2.line(placeholder, (150, 200), (250, 200), (105, 85, 71), 1)
            cv2.line(placeholder, (200, 150), (200, 250), (105, 85, 71), 1)
            text = "IMAGE NOT FOUND"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = (400 - text_size[0]) // 2
            text_y = (400 + text_size[1]) // 2
            cv2.putText(placeholder, text, (text_x, text_y), font, font_scale, (139, 116, 100), thickness, cv2.LINE_AA)

            success, encoded_img = cv2.imencode('.png', placeholder)
            if success:
                from fastapi.responses import Response
                return Response(content=encoded_img.tobytes(), media_type="image/png", status_code=404)
            raise HTTPException(status_code=404, detail="Image file missing on server disk.")

        if img_path.lower().endswith(('.tif', '.tiff')):
            import numpy as np
            import cv2
            try:
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    if img.dtype != np.uint8:
                        img_min, img_max = img.min(), img.max()
                        if img_max > img_min:
                            img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                        else:
                            img = np.zeros_like(img, dtype=np.uint8)
                    success, encoded_img = cv2.imencode('.png', img)
                    if success:
                        from fastapi.responses import Response
                        return Response(content=encoded_img.tobytes(), media_type="image/png")
            except Exception as ex:
                logger.error(f"Failed to convert TIFF {img_path} to PNG: {ex}")

        media_type = "image/png"
        if img_path.lower().endswith(".jpg") or img_path.lower().endswith(".jpeg"):
            media_type = "image/jpeg"

        return FileResponse(img_path, media_type=media_type)
    finally:
        conn.close()


@router.get("/database/history")
def get_prediction_history(patient_id: Optional[str] = Query(None), current_user: User = Depends(get_current_user)):
    """API Endpoint: Retrieves scan prediction logs matching patient ID search filters with tenant isolation."""
    from security.application.authorization_service import AuthorizationService
    auth_svc = AuthorizationService(DEFAULT_DB_PATH)

    # Enforce patient boundaries
    if current_user.role == Role.PATIENT:
        patient_id = current_user.uuid

    if patient_id:
        if not auth_svc.can_access_patient(current_user, patient_id):
            raise HTTPException(status_code=403, detail="Access denied to patient records.")

    restrict_doctor_id = current_user.id if current_user.role == Role.DOCTOR else None
    history_repo = SQLitePredictionHistoryRepository(db_path=DEFAULT_DB_PATH)
    criteria = HistorySearchCriteria(
        patient_id=patient_id if patient_id else None,
        restrict_to_doctor_id=restrict_doctor_id
    )

    try:
        summaries = history_repo.search_history(criteria)
        authorized_patient_ids = auth_svc.get_authorized_patient_ids(current_user)
        results = []
        for s in summaries:
            # Multi-tenant safeguard: skip records that don't belong to this patient
            if authorized_patient_ids is not None and str(s.patient_id).lower() not in authorized_patient_ids:
                continue

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
def get_dashboard_telemetry(current_user: User = Depends(require_roles([Role.ADMIN, Role.DOCTOR]))):
    """API Endpoint: returns analytics distribution graphs counts. Only accessible to Admins and Doctors."""
    persistence_repo = SQLitePersistenceRepository(db_path=DEFAULT_DB_PATH)
    try:
        summary = persistence_repo.get_analytics_summary()
        return summary
    except Exception as e:
        logger.error(f"Error compiling analytics widgets: {e}")
        raise HTTPException(status_code=500, detail=f"Analytics telemetry compilation failed: {e}")


# Pydantic Schemas for Report versioning
class VersionCreateRequest(BaseModel):
    reason: str
    pdf_path: Optional[str] = None
    json_path: Optional[str] = None
    prediction_id: Optional[int] = None
    status: Optional[str] = "DRAFT"

class StatusUpdateRequest(BaseModel):
    status: str


@router.get("/reports/audit-history")
def get_reports_audit_history_api(current_user: User = Depends(get_current_user)):
    """Retrieves report access history logs filtered according to user roles."""
    service = ReportService(db_path=DEFAULT_DB_PATH)
    try:
        logs = service.get_report_audit_history(current_user)
        return logs
    except Exception as e:
        logger.error(f"Error fetching audit history: {e}")
        raise HTTPException(status_code=500, detail="Internal server error fetching audit history.")


@router.get("/reports/{report_id}")
def get_report_metadata_api(report_id: int, current_user: User = Depends(get_current_user)):
    """Fetches Report metadata by ID, enforcing patient tenant boundaries."""
    service = ReportService(db_path=DEFAULT_DB_PATH)

    access_status = service.check_report_access(report_id, current_user)
    if access_status == "NOT_FOUND":
        service.log_report_access_event("REPORT_NOT_FOUND", current_user, report_id, "FAILED", "Requested nonexistent report metadata.")
        raise HTTPException(status_code=404, detail="Report not found.")
    elif access_status == "UNAUTHORIZED":
        raise HTTPException(status_code=401, detail="Authentication required.")
    elif access_status == "FORBIDDEN":
        service.log_report_access_event("REPORT_ACCESS_DENIED", current_user, report_id, "FAILED", "Access denied to patient report metadata.")
        raise HTTPException(status_code=403, detail="Access denied to patient report.")

    try:
        report = service.get_report(report_id)
        versions = service.get_report_versions(report_id)
        res = report.to_dict()
        res["versions"] = [v.to_dict() for v in versions]

        # Security Hardening: Remove internal filesystem paths if the user is a Patient (H4.3-S)
        role_str = current_user.role.value if hasattr(current_user.role, "value") else str(current_user.role).lower()
        if role_str == "patient":
            res.pop("pdf_path", None)
            res.pop("json_path", None)
            for v_dict in res.get("versions", []):
                v_dict.pop("pdf_path", None)
                v_dict.pop("json_path", None)

        service.log_report_access_event("REPORT_VIEWED", current_user, report_id, "SUCCESS", "Viewed report metadata.")
        return res
    except Exception as e:
        logger.error(f"Error fetching report metadata: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/reports/{report_id}/compare/{other_report_id}")
def compare_reports_api(
    report_id: int,
    other_report_id: int,
    previous_version: Optional[int] = Query(None),
    current_version: Optional[int] = Query(None),
    current_user: User = Depends(get_current_user)
):
    """API Endpoint: Compares two existing reports for the same patient, enforcing RBAC boundaries."""
    service = ReportService(db_path=DEFAULT_DB_PATH)
    try:
        from clinical_reporting.domain.entities import PatientMismatchException
        from clinical_reporting.application.services import ReportNotFoundException, VersionNotFoundException, ReportServiceException

        result = service.compare_reports(
            previous_report_id=report_id,
            current_report_id=other_report_id,
            previous_version=previous_version,
            current_version=current_version,
            actor=current_user
        )
        return result
    except (ReportNotFoundException, VersionNotFoundException) as nfe:
        raise HTTPException(status_code=404, detail=str(nfe))
    except PatientMismatchException as pme:
        raise HTTPException(status_code=400, detail=str(pme))
    except ReportServiceException as rse:
        err_msg = str(rse)
        if "Access denied" in err_msg or "denied" in err_msg.lower():
            raise HTTPException(status_code=403, detail=err_msg)
        elif "Authentication required" in err_msg or "unauthenticated" in err_msg.lower():
            raise HTTPException(status_code=401, detail=err_msg)
        else:
            raise HTTPException(status_code=422, detail=err_msg)
    except Exception as e:
        logger.error(f"Error in comparison API: {e}")
        raise HTTPException(status_code=500, detail="Internal comparison engine error")


@router.get("/reports/{report_id}/compare-followup/{other_report_id}")
def compare_followup_reports_api(
    report_id: int,
    other_report_id: int,
    previous_version: Optional[int] = Query(None),
    current_version: Optional[int] = Query(None),
    current_user: User = Depends(get_current_user)
):
    """API Endpoint: Longitudinal scan comparison between an earlier report and a follow-up report, enforcing RBAC."""
    service = ReportService(db_path=DEFAULT_DB_PATH)
    try:
        from clinical_reporting.domain.entities import PatientMismatchException
        from clinical_reporting.application.services import ReportNotFoundException, VersionNotFoundException, ReportServiceException

        result = service.compare_followup_reports(
            previous_report_id=report_id,
            current_report_id=other_report_id,
            previous_version=previous_version,
            current_version=current_version,
            actor=current_user
        )
        return result
    except (ReportNotFoundException, VersionNotFoundException) as nfe:
        raise HTTPException(status_code=404, detail=str(nfe))
    except PatientMismatchException as pme:
        raise HTTPException(status_code=400, detail=str(pme))
    except ReportServiceException as rse:
        err_msg = str(rse)
        if "Access denied" in err_msg or "denied" in err_msg.lower():
            raise HTTPException(status_code=403, detail=err_msg)
        elif "Authentication required" in err_msg or "unauthenticated" in err_msg.lower():
            raise HTTPException(status_code=401, detail=err_msg)
        else:
            raise HTTPException(status_code=422, detail=err_msg)
    except Exception as e:
        logger.error(f"Error in follow-up comparison API: {e}")
        raise HTTPException(status_code=500, detail="Internal follow-up comparison engine error")


@router.get("/patients/{patient_id}/longitudinal-timeline")
def get_patient_timeline_api(
    patient_id: str,
    current_user: User = Depends(get_current_user)
):
    """API Endpoint: Retrieves chronological patient timeline, enforcing RBAC."""
    service = ReportService(db_path=DEFAULT_DB_PATH)
    try:
        from clinical_reporting.application.services import ReportServiceException

        timeline = service.get_patient_longitudinal_timeline(
            patient_id=patient_id,
            actor=current_user
        )
        return timeline.to_dict()
    except ReportServiceException as rse:
        err_msg = str(rse)
        if "Access denied" in err_msg or "denied" in err_msg.lower():
            raise HTTPException(status_code=403, detail=err_msg)
        elif "Authentication required" in err_msg or "unauthenticated" in err_msg.lower():
            raise HTTPException(status_code=401, detail=err_msg)
        elif "not found" in err_msg.lower():
            raise HTTPException(status_code=404, detail=err_msg)
        else:
            raise HTTPException(status_code=422, detail=err_msg)
    except Exception as e:
        logger.error(f"Error in timeline API: {e}")
        raise HTTPException(status_code=500, detail="Internal timeline engine error")


@router.get("/reports/{report_id}/versions")
def get_report_versions_api(report_id: int, current_user: User = Depends(get_current_user)):
    """Fetches Report version history list, enforcing patient boundaries."""
    service = ReportService(db_path=DEFAULT_DB_PATH)

    access_status = service.check_report_access(report_id, current_user)
    if access_status == "NOT_FOUND":
        service.log_report_access_event("REPORT_NOT_FOUND", current_user, report_id, "FAILED", "Requested version list for nonexistent report.")
        raise HTTPException(status_code=404, detail="Report not found.")
    elif access_status == "UNAUTHORIZED":
        raise HTTPException(status_code=401, detail="Authentication required.")
    elif access_status == "FORBIDDEN":
        service.log_report_access_event("REPORT_ACCESS_DENIED", current_user, report_id, "FAILED", "Access denied to patient report versions.")
        raise HTTPException(status_code=403, detail="Access denied to patient report.")

    try:
        versions = service.get_report_versions(report_id)
        res_list = [v.to_dict() for v in versions]

        # Security Hardening: Remove internal filesystem paths if the user is a Patient (H4.3-S)
        role_str = current_user.role.value if hasattr(current_user.role, "value") else str(current_user.role).lower()
        if role_str == "patient":
            for v_dict in res_list:
                v_dict.pop("pdf_path", None)
                v_dict.pop("json_path", None)

        service.log_report_access_event("REPORT_VIEWED", current_user, report_id, "SUCCESS", "Viewed report versions list.")
        return res_list
    except Exception as e:
        logger.error(f"Error fetching report versions: {e}")
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/reports/{report_id}/versions")
def create_report_version_api(
    report_id: int,
    req: VersionCreateRequest,
    current_user: User = Depends(require_roles([Role.ADMIN, Role.DOCTOR]))
):
    """Creates a new version of the report, restricted to Doctors and Admins."""
    from clinical_reporting.domain.entities import ReportStatus
    service = ReportService(db_path=DEFAULT_DB_PATH)

    access_status = service.check_report_access(report_id, current_user)
    if access_status == "NOT_FOUND":
        raise HTTPException(status_code=404, detail="Report not found.")
    elif access_status == "UNAUTHORIZED":
        raise HTTPException(status_code=401, detail="Authentication required.")
    elif access_status == "FORBIDDEN":
        raise HTTPException(status_code=403, detail="Access denied to patient report.")

    try:
        try:
            status_enum = ReportStatus(req.status.upper())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {req.status}")

        new_version = service.create_new_version(
            report_id=report_id,
            created_by=current_user.email,
            reason=req.reason,
            pdf_path=req.pdf_path,
            json_path=req.json_path,
            prediction_id=req.prediction_id,
            status=status_enum
        )
        return new_version.to_dict()
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Error creating report version: {e}")
        raise HTTPException(status_code=400, detail=str(e))


class EmailReportRequest(BaseModel):
    recipient_email: Optional[str] = None
    version: Optional[int] = None


@router.post("/reports/{report_id}/email")
def email_report_api(
    report_id: int,
    req: EmailReportRequest,
    current_user: User = Depends(get_current_user)
):
    """Emails a clinical report PDF to an authorized recipient."""
    from clinical_reporting.application.services import ReportNotFoundException, VersionNotFoundException
    from clinical_reporting.infrastructure.email_service import ConfigurationException
    service = ReportService(db_path=DEFAULT_DB_PATH)
    try:
        result = service.send_report_email(
            report_id=report_id,
            actor=current_user,
            recipient_email=req.recipient_email,
            version=req.version
        )
        return result
    except ReportNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except VersionNotFoundException as e:
        raise HTTPException(status_code=404, detail=str(e))
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail="PDF report file not found on the server.")
    except ConfigurationException as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Error emailing report: {e}")
        raise HTTPException(status_code=500, detail="Internal server error sending report email.")


@router.get("/reports/email/history")
def get_email_history_api(
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=100),
    status: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    report_id: Optional[int] = Query(None),
    current_user: User = Depends(get_current_user)
):
    """Fetches paginated, authorized email delivery history records."""
    service = ReportService(db_path=DEFAULT_DB_PATH)
    try:
        history = service.get_email_history(
            actor=current_user,
            page=page,
            per_page=per_page,
            status=status,
            search=search,
            report_id=report_id
        )
        return history
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.error(f"Error fetching email history: {e}")
        raise HTTPException(status_code=500, detail="Internal server error fetching email history.")




@router.patch("/reports/{report_id}/status")
def patch_report_status_api(
    report_id: int,
    req: StatusUpdateRequest,
    current_user: User = Depends(require_roles([Role.ADMIN, Role.DOCTOR]))
):
    """Transition report status state machine, restricted to Doctors and Admins."""
    from clinical_reporting.domain.entities import ReportStatus
    service = ReportService(db_path=DEFAULT_DB_PATH)

    access_status = service.check_report_access(report_id, current_user)
    if access_status == "NOT_FOUND":
        raise HTTPException(status_code=404, detail="Report not found.")
    elif access_status == "UNAUTHORIZED":
        raise HTTPException(status_code=401, detail="Authentication required.")
    elif access_status == "FORBIDDEN":
        raise HTTPException(status_code=403, detail="Access denied to patient report.")

    try:
        try:
            target_status = ReportStatus(req.status.upper())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {req.status}")

        updated_report = service.transition_status(
            report_id=report_id,
            target_status=target_status,
            actor=current_user.email
        )
        return updated_report.to_dict()
    except Exception as e:
        logger.error(f"Error transitioning report status: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/reports/{report_id}/versions/{version_id}")
def get_report_version_details_api(report_id: int, version_id: int, current_user: User = Depends(get_current_user)):
    """Fetches details of a specific report version, enforcing patient boundaries."""
    service = ReportService(db_path=DEFAULT_DB_PATH)

    access_status = service.check_report_access(report_id, current_user)
    if access_status == "NOT_FOUND":
        service.log_report_access_event("REPORT_NOT_FOUND", current_user, report_id, "FAILED", f"Requested version details for nonexistent report.")
        raise HTTPException(status_code=404, detail="Report not found.")
    elif access_status == "UNAUTHORIZED":
        raise HTTPException(status_code=401, detail="Authentication required.")
    elif access_status == "FORBIDDEN":
        service.log_report_access_event("REPORT_ACCESS_DENIED", current_user, report_id, "FAILED", f"Access denied to patient version details.")
        raise HTTPException(status_code=403, detail="Access denied to patient report.")

    try:
        versions = service.get_report_versions(report_id)
        target_version = next((v for v in versions if v.version_id == version_id), None)
        if not target_version:
            raise HTTPException(status_code=404, detail=f"Version ID {version_id} not found for report {report_id}.")
        service.log_report_access_event("REPORT_VIEWED", current_user, report_id, "SUCCESS", f"Viewed version details for version: {target_version.version_number}")
        return target_version.to_dict()
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Error fetching version details: {e}")
        raise HTTPException(status_code=404, detail=str(e))



@router.get("/reports/verify/{token}")
def verify_report_api(token: str):
    """Public verification endpoint to validate report integrity and authenticity by token."""
    service = ReportService(db_path=DEFAULT_DB_PATH)
    try:
        result = service.verify_report_by_token(token)
        state = result.get("verification_state", "INVALID")
        if state == "INVALID":
            raise HTTPException(status_code=404, detail="Verification token not found or invalid.")
        return result
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in verification API: {e}")
        raise HTTPException(status_code=500, detail="Internal verification error")


@router.get("/patients/{patient_id}/analytics")
def get_patient_analytics_api(
    patient_id: str,
    current_user: User = Depends(get_current_user)
):
    """API Endpoint: Retrieves patient clinical analytics, enforcing authorization boundaries."""
    service = ReportService(db_path=DEFAULT_DB_PATH)
    try:
        from clinical_reporting.application.services import ReportServiceException

        analytics = service.get_patient_analytics(
            patient_id=patient_id,
            actor=current_user
        )
        return analytics.to_dict()
    except ReportServiceException as rse:
        err_msg = str(rse)
        if "Access denied" in err_msg or "denied" in err_msg.lower():
            raise HTTPException(status_code=403, detail="Access denied to patient analytics.")
        elif "Authentication required" in err_msg or "unauthenticated" in err_msg.lower():
            raise HTTPException(status_code=401, detail=err_msg)
        elif "not found" in err_msg.lower():
            raise HTTPException(status_code=404, detail=err_msg)
        else:
            raise HTTPException(status_code=422, detail=err_msg)
    except Exception as e:
        logger.error(f"Error in patient analytics API: {e}")
        raise HTTPException(status_code=500, detail="Internal analytics engine error")


@router.get("/analytics/overview")
def get_population_analytics_api(
    current_user: User = Depends(get_current_user)
):
    """API Endpoint: Retrieves population overview analytics for clinician dashboard."""
    if current_user.role not in [Role.ADMIN, Role.DOCTOR]:
        raise HTTPException(
            status_code=403,
            detail=f"Access denied. Privilege level '{current_user.role.value}' is not authorized."
        )

    service = ReportService(db_path=DEFAULT_DB_PATH)
    try:
        from clinical_reporting.application.services import ReportServiceException

        pop_analytics = service.get_population_analytics(actor=current_user)
        return pop_analytics.to_dict()
    except ReportServiceException as rse:
        err_msg = str(rse)
        if "Access denied" in err_msg or "denied" in err_msg.lower():
            raise HTTPException(status_code=403, detail=err_msg)
        elif "Authentication required" in err_msg or "unauthenticated" in err_msg.lower():
            raise HTTPException(status_code=401, detail=err_msg)
        else:
            raise HTTPException(status_code=422, detail=err_msg)
    except Exception as e:
        logger.error(f"Error in population analytics API: {e}")
        raise HTTPException(status_code=500, detail="Internal population analytics engine error")


@router.get("/patients/{patient_id}/analytics/timeseries")
def get_patient_timeseries_api(
    patient_id: str,
    current_user: User = Depends(get_current_user)
):
    """API Endpoint: Retrieves patient time-series data points, enforcing authorization boundaries."""
    service = ReportService(db_path=DEFAULT_DB_PATH)
    try:
        from clinical_reporting.application.services import ReportServiceException

        ts = service.get_patient_timeseries(
            patient_id=patient_id,
            actor=current_user
        )
        return ts.to_dict()
    except ReportServiceException as rse:
        err_msg = str(rse)
        if "Access denied" in err_msg or "denied" in err_msg.lower():
            raise HTTPException(status_code=403, detail="Access denied to patient timeseries.")
        elif "Authentication required" in err_msg or "unauthenticated" in err_msg.lower():
            raise HTTPException(status_code=401, detail=err_msg)
        elif "not found" in err_msg.lower():
            raise HTTPException(status_code=404, detail=err_msg)
        else:
            raise HTTPException(status_code=422, detail=err_msg)
    except Exception as e:
        logger.error(f"Error in patient timeseries API: {e}")
        raise HTTPException(status_code=500, detail="Internal timeseries engine error")


class FollowupCreatePayload(BaseModel):
    scheduled_date: str
    reason: Optional[str] = None
    notes: Optional[str] = None

class FollowupUpdatePayload(BaseModel):
    scheduled_date: Optional[str] = None
    status: Optional[str] = None
    reason: Optional[str] = None
    notes: Optional[str] = None

@router.get("/doctor/patients/{patient_id}")
def get_doctor_patient_profile_api(
    patient_id: str,
    current_user: User = Depends(require_roles([Role.DOCTOR, Role.ADMIN]))
):
    """API Endpoint: Retrieves chronological patient profile, enforcing RBAC."""
    service = ReportService(db_path=DEFAULT_DB_PATH)
    try:
        from clinical_reporting.application.services import ReportServiceException
        profile = service.get_patient_profile(patient_id=patient_id, actor=current_user)
        return profile
    except ReportServiceException as rse:
        err_msg = str(rse)
        if "Access denied" in err_msg or "denied" in err_msg.lower():
            raise HTTPException(status_code=403, detail=err_msg)
        elif "Authentication required" in err_msg or "unauthenticated" in err_msg.lower():
            raise HTTPException(status_code=401, detail=err_msg)
        elif "not found" in err_msg.lower():
            raise HTTPException(status_code=404, detail=err_msg)
        else:
            raise HTTPException(status_code=422, detail=err_msg)
    except Exception as e:
        logger.error(f"Error in doctor patient profile API: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/patients/{patient_id}/followups")
def create_doctor_patient_followup_api(
    patient_id: str,
    payload: FollowupCreatePayload,
    current_user: User = Depends(require_roles([Role.DOCTOR, Role.ADMIN]))
):
    """API Endpoint: Creates a follow-up schedule for an assigned patient."""
    from clinical_reporting.application.followup_service import FollowupScheduleService, FollowupScheduleServiceException
    service = FollowupScheduleService(db_path=DEFAULT_DB_PATH)
    try:
        followup = service.create_followup(
            actor=current_user,
            patient_id=patient_id,
            scheduled_date=payload.scheduled_date,
            reason=payload.reason,
            notes=payload.notes
        )
        return followup
    except FollowupScheduleServiceException as e:
        err_msg = str(e)
        if "Access denied" in err_msg or "denied" in err_msg.lower():
            raise HTTPException(status_code=403, detail=err_msg)
        elif "Authentication required" in err_msg or "unauthenticated" in err_msg.lower():
            raise HTTPException(status_code=401, detail=err_msg)
        elif "not found" in err_msg.lower():
            raise HTTPException(status_code=404, detail=err_msg)
        else:
            raise HTTPException(status_code=400, detail=err_msg)
    except Exception as e:
        logger.error(f"Error in create follow-up API: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/patients/{patient_id}/followups")
def get_doctor_patient_followups_api(
    patient_id: str,
    current_user: User = Depends(get_current_user)
):
    """API Endpoint: Retrieves follow-up schedules for a patient."""
    from clinical_reporting.application.followup_service import FollowupScheduleService, FollowupScheduleServiceException
    service = FollowupScheduleService(db_path=DEFAULT_DB_PATH)
    try:
        followups = service.get_followups_for_patient(actor=current_user, patient_id=patient_id)
        return followups
    except FollowupScheduleServiceException as e:
        err_msg = str(e)
        if "Access denied" in err_msg or "denied" in err_msg.lower():
            raise HTTPException(status_code=403, detail=err_msg)
        elif "Authentication required" in err_msg or "unauthenticated" in err_msg.lower():
            raise HTTPException(status_code=401, detail=err_msg)
        elif "not found" in err_msg.lower():
            raise HTTPException(status_code=404, detail=err_msg)
        else:
            raise HTTPException(status_code=400, detail=err_msg)
    except Exception as e:
        logger.error(f"Error in get follow-ups API: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.put("/doctor/followups/{followup_id}")
def update_doctor_followup_api(
    followup_id: int,
    payload: FollowupUpdatePayload,
    current_user: User = Depends(require_roles([Role.DOCTOR, Role.ADMIN]))
):
    """API Endpoint: Updates an existing follow-up schedule."""
    from clinical_reporting.application.followup_service import FollowupScheduleService, FollowupScheduleServiceException
    service = FollowupScheduleService(db_path=DEFAULT_DB_PATH)
    try:
        followup = service.update_followup(
            actor=current_user,
            followup_id=followup_id,
            scheduled_date=payload.scheduled_date,
            status=payload.status,
            reason=payload.reason,
            notes=payload.notes
        )
        return followup
    except FollowupScheduleServiceException as e:
        err_msg = str(e)
        if "Access denied" in err_msg or "denied" in err_msg.lower():
            raise HTTPException(status_code=403, detail=err_msg)
        elif "Authentication required" in err_msg or "unauthenticated" in err_msg.lower():
            raise HTTPException(status_code=401, detail=err_msg)
        elif "not found" in err_msg.lower():
            raise HTTPException(status_code=404, detail=err_msg)
        else:
            raise HTTPException(status_code=400, detail=err_msg)
    except Exception as e:
        logger.error(f"Error in update follow-up API: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


class ClinicianNotePayload(BaseModel):
    content: str

@router.post("/doctor/scans/{scan_id}/clinical-notes")
def create_clinician_note_api(
    scan_id: int,
    payload: ClinicianNotePayload,
    current_user: User = Depends(require_roles([Role.DOCTOR, Role.ADMIN]))
):
    """API Endpoint: Create a clinician note for a given MRI scan."""
    from clinical_reporting.application.clinician_note_service import ClinicianNoteService, ClinicianNoteServiceException
    service = ClinicianNoteService(db_path=DEFAULT_DB_PATH)
    try:
        note = service.create_note(actor=current_user, scan_id=scan_id, content=payload.content)
        return note
    except ClinicianNoteServiceException as e:
        err_msg = str(e)
        if "Access denied" in err_msg:
            raise HTTPException(status_code=403, detail=err_msg)
        elif "Authentication required" in err_msg:
            raise HTTPException(status_code=401, detail=err_msg)
        elif "not found" in err_msg.lower():
            raise HTTPException(status_code=404, detail=err_msg)
        else:
            raise HTTPException(status_code=400, detail=err_msg)
    except Exception as e:
        logger.error(f"Error in create note API: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/doctor/scans/{scan_id}/clinical-notes")
def get_clinician_notes_api(
    scan_id: int,
    current_user: User = Depends(get_current_user)
):
    """API Endpoint: Get active clinician notes for a given MRI scan."""
    role_val = current_user.role.value if hasattr(current_user.role, 'value') else str(current_user.role).lower()
    if role_val != "doctor" and role_val != "admin":
        raise HTTPException(status_code=403, detail="Access denied. Doctor role required.")

    from clinical_reporting.application.clinician_note_service import ClinicianNoteService, ClinicianNoteServiceException
    service = ClinicianNoteService(db_path=DEFAULT_DB_PATH)
    try:
        notes = service.get_notes_for_scan(actor=current_user, scan_id=scan_id)
        return notes
    except ClinicianNoteServiceException as e:
        err_msg = str(e)
        if "Access denied" in err_msg:
            raise HTTPException(status_code=403, detail=err_msg)
        elif "Authentication required" in err_msg:
            raise HTTPException(status_code=401, detail=err_msg)
        elif "not found" in err_msg.lower():
            raise HTTPException(status_code=404, detail=err_msg)
        else:
            raise HTTPException(status_code=400, detail=err_msg)
    except Exception as e:
        logger.error(f"Error in get notes API: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.put("/doctor/clinical-notes/{note_id}")
def update_clinician_note_api(
    note_id: int,
    payload: ClinicianNotePayload,
    current_user: User = Depends(require_roles([Role.DOCTOR, Role.ADMIN]))
):
    """API Endpoint: Update an existing clinician note."""
    from clinical_reporting.application.clinician_note_service import ClinicianNoteService, ClinicianNoteServiceException
    service = ClinicianNoteService(db_path=DEFAULT_DB_PATH)
    try:
        note = service.update_note(actor=current_user, note_id=note_id, content=payload.content)
        return note
    except ClinicianNoteServiceException as e:
        err_msg = str(e)
        if "Access denied" in err_msg:
            raise HTTPException(status_code=403, detail=err_msg)
        elif "Authentication required" in err_msg:
            raise HTTPException(status_code=401, detail=err_msg)
        elif "not found" in err_msg.lower():
            raise HTTPException(status_code=404, detail=err_msg)
        else:
            raise HTTPException(status_code=400, detail=err_msg)
    except Exception as e:
        logger.error(f"Error in update note API: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/doctor/clinical-notes/{note_id}")
def archive_clinician_note_api(
    note_id: int,
    current_user: User = Depends(require_roles([Role.DOCTOR, Role.ADMIN]))
):
    """API Endpoint: Soft-archive an existing clinician note."""
    from clinical_reporting.application.clinician_note_service import ClinicianNoteService, ClinicianNoteServiceException
    service = ClinicianNoteService(db_path=DEFAULT_DB_PATH)
    try:
        result = service.archive_note(actor=current_user, note_id=note_id)
        return result
    except ClinicianNoteServiceException as e:
        err_msg = str(e)
        if "Access denied" in err_msg:
            raise HTTPException(status_code=403, detail=err_msg)
        elif "Authentication required" in err_msg:
            raise HTTPException(status_code=401, detail=err_msg)
        elif "not found" in err_msg.lower():
            raise HTTPException(status_code=404, detail=err_msg)
        else:
            raise HTTPException(status_code=400, detail=err_msg)
    except Exception as e:
        logger.error(f"Error in archive note API: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


class MriAnnotationPayload(BaseModel):
    x: float
    y: float
    label: Optional[str] = None
    comment: Optional[str] = None

@router.post("/doctor/scans/{scan_id}/point-annotations")
def create_mri_annotation_api(
    scan_id: int,
    payload: MriAnnotationPayload,
    current_user: User = Depends(require_roles([Role.DOCTOR, Role.ADMIN]))
):
    """API Endpoint: Create a point annotation for a given MRI scan."""
    from clinical_reporting.application.mri_annotation_service import MriAnnotationService, MriAnnotationServiceException
    service = MriAnnotationService(db_path=DEFAULT_DB_PATH)
    try:
        note = service.create_annotation(
            actor=current_user,
            scan_id=scan_id,
            x=payload.x,
            y=payload.y,
            label=payload.label,
            comment=payload.comment
        )
        return note
    except MriAnnotationServiceException as e:
        err_msg = str(e)
        if "Access denied" in err_msg:
            raise HTTPException(status_code=403, detail=err_msg)
        elif "Authentication required" in err_msg:
            raise HTTPException(status_code=401, detail=err_msg)
        elif "not found" in err_msg.lower():
            raise HTTPException(status_code=404, detail=err_msg)
        else:
            raise HTTPException(status_code=400, detail=err_msg)
    except Exception as e:
        logger.error(f"Error in create annotation API: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/doctor/scans/{scan_id}/point-annotations")
def get_mri_annotations_api(
    scan_id: int,
    current_user: User = Depends(get_current_user)
):
    """API Endpoint: Get active point annotations for a given MRI scan."""
    role_val = current_user.role.value if hasattr(current_user.role, 'value') else str(current_user.role).lower()
    if role_val != "doctor" and role_val != "admin":
        raise HTTPException(status_code=403, detail="Access denied. Doctor role required.")

    from clinical_reporting.application.mri_annotation_service import MriAnnotationService, MriAnnotationServiceException
    service = MriAnnotationService(db_path=DEFAULT_DB_PATH)
    try:
        notes = service.get_annotations_for_scan(actor=current_user, scan_id=scan_id)
        return notes
    except MriAnnotationServiceException as e:
        err_msg = str(e)
        if "Access denied" in err_msg:
            raise HTTPException(status_code=403, detail=err_msg)
        elif "Authentication required" in err_msg:
            raise HTTPException(status_code=401, detail=err_msg)
        elif "not found" in err_msg.lower():
            raise HTTPException(status_code=404, detail=err_msg)
        else:
            raise HTTPException(status_code=400, detail=err_msg)
    except Exception as e:
        logger.error(f"Error in get annotations API: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.put("/doctor/point-annotations/{annotation_id}")
def update_mri_annotation_api(
    annotation_id: int,
    payload: MriAnnotationPayload,
    current_user: User = Depends(require_roles([Role.DOCTOR, Role.ADMIN]))
):
    """API Endpoint: Update an existing point annotation."""
    from clinical_reporting.application.mri_annotation_service import MriAnnotationService, MriAnnotationServiceException
    service = MriAnnotationService(db_path=DEFAULT_DB_PATH)
    try:
        note = service.update_annotation(
            actor=current_user,
            annotation_id=annotation_id,
            x=payload.x,
            y=payload.y,
            label=payload.label,
            comment=payload.comment
        )
        return note
    except MriAnnotationServiceException as e:
        err_msg = str(e)
        if "Access denied" in err_msg:
            raise HTTPException(status_code=403, detail=err_msg)
        elif "Authentication required" in err_msg:
            raise HTTPException(status_code=401, detail=err_msg)
        elif "not found" in err_msg.lower():
            raise HTTPException(status_code=404, detail=err_msg)
        else:
            raise HTTPException(status_code=400, detail=err_msg)
    except Exception as e:
        logger.error(f"Error in update annotation API: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/doctor/point-annotations/{annotation_id}")
def archive_mri_annotation_api(
    annotation_id: int,
    current_user: User = Depends(require_roles([Role.DOCTOR, Role.ADMIN]))
):
    """API Endpoint: Soft-archive an existing point annotation."""
    from clinical_reporting.application.mri_annotation_service import MriAnnotationService, MriAnnotationServiceException
    service = MriAnnotationService(db_path=DEFAULT_DB_PATH)
    try:
        result = service.archive_annotation(actor=current_user, annotation_id=annotation_id)
        return result
    except MriAnnotationServiceException as e:
        err_msg = str(e)
        if "Access denied" in err_msg:
            raise HTTPException(status_code=403, detail=err_msg)
        elif "Authentication required" in err_msg:
            raise HTTPException(status_code=401, detail=err_msg)
        elif "not found" in err_msg.lower():
            raise HTTPException(status_code=404, detail=err_msg)
        else:
            raise HTTPException(status_code=400, detail=err_msg)
    except Exception as e:
        logger.error(f"Error in archive annotation API: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


class MriRectangleAnnotationPayload(BaseModel):
    x: float
    y: float
    width: float
    height: float
    label: Optional[str] = None
    comment: Optional[str] = None


@router.post("/doctor/scans/{scan_id}/rectangle-annotations")
def create_mri_rectangle_annotation_api(
    scan_id: int,
    payload: MriRectangleAnnotationPayload,
    current_user: User = Depends(require_roles([Role.DOCTOR, Role.ADMIN]))
):
    """API Endpoint: Create a rectangle annotation for a given MRI scan."""
    from clinical_reporting.application.mri_annotation_service import MriAnnotationService, MriAnnotationServiceException
    service = MriAnnotationService(db_path=DEFAULT_DB_PATH)
    try:
        note = service.create_rectangle_annotation(
            actor=current_user,
            scan_id=scan_id,
            x=payload.x,
            y=payload.y,
            width=payload.width,
            height=payload.height,
            label=payload.label,
            comment=payload.comment
        )
        return note
    except MriAnnotationServiceException as e:
        err_msg = str(e)
        if "Access denied" in err_msg:
            raise HTTPException(status_code=403, detail=err_msg)
        elif "Authentication required" in err_msg:
            raise HTTPException(status_code=401, detail=err_msg)
        elif "not found" in err_msg.lower():
            raise HTTPException(status_code=404, detail=err_msg)
        else:
            raise HTTPException(status_code=400, detail=err_msg)
    except Exception as e:
        logger.error(f"Error in create rectangle annotation API: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/doctor/scans/{scan_id}/rectangle-annotations")
def get_mri_rectangle_annotations_api(
    scan_id: int,
    current_user: User = Depends(get_current_user)
):
    """API Endpoint: Get active rectangle annotations for a given MRI scan."""
    role_val = current_user.role.value if hasattr(current_user.role, 'value') else str(current_user.role).lower()
    if role_val != "doctor" and role_val != "admin":
        raise HTTPException(status_code=403, detail="Access denied. Doctor role required.")

    from clinical_reporting.application.mri_annotation_service import MriAnnotationService, MriAnnotationServiceException
    service = MriAnnotationService(db_path=DEFAULT_DB_PATH)
    try:
        notes = service.get_rectangle_annotations_for_scan(actor=current_user, scan_id=scan_id)
        return notes
    except MriAnnotationServiceException as e:
        err_msg = str(e)
        if "Access denied" in err_msg:
            raise HTTPException(status_code=403, detail=err_msg)
        elif "Authentication required" in err_msg:
            raise HTTPException(status_code=401, detail=err_msg)
        elif "not found" in err_msg.lower():
            raise HTTPException(status_code=404, detail=err_msg)
        else:
            raise HTTPException(status_code=400, detail=err_msg)
    except Exception as e:
        logger.error(f"Error in get rectangle annotations API: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/doctor/rectangle-annotations/{annotation_id}")
def update_mri_rectangle_annotation_api(
    annotation_id: int,
    payload: MriRectangleAnnotationPayload,
    current_user: User = Depends(require_roles([Role.DOCTOR, Role.ADMIN]))
):
    """API Endpoint: Update an existing rectangle annotation."""
    from clinical_reporting.application.mri_annotation_service import MriAnnotationService, MriAnnotationServiceException
    service = MriAnnotationService(db_path=DEFAULT_DB_PATH)
    try:
        note = service.update_rectangle_annotation(
            actor=current_user,
            annotation_id=annotation_id,
            x=payload.x,
            y=payload.y,
            width=payload.width,
            height=payload.height,
            label=payload.label,
            comment=payload.comment
        )
        return note
    except MriAnnotationServiceException as e:
        err_msg = str(e)
        if "Access denied" in err_msg:
            raise HTTPException(status_code=403, detail=err_msg)
        elif "Authentication required" in err_msg:
            raise HTTPException(status_code=401, detail=err_msg)
        elif "not found" in err_msg.lower():
            raise HTTPException(status_code=404, detail=err_msg)
        else:
            raise HTTPException(status_code=400, detail=err_msg)
    except Exception as e:
        logger.error(f"Error in update rectangle annotation API: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/doctor/rectangle-annotations/{annotation_id}")
def archive_mri_rectangle_annotation_api(
    annotation_id: int,
    current_user: User = Depends(require_roles([Role.DOCTOR, Role.ADMIN]))
):
    """API Endpoint: Soft-archive an existing rectangle annotation."""
    from clinical_reporting.application.mri_annotation_service import MriAnnotationService, MriAnnotationServiceException
    service = MriAnnotationService(db_path=DEFAULT_DB_PATH)
    try:
        result = service.archive_rectangle_annotation(actor=current_user, annotation_id=annotation_id)
        return result
    except MriAnnotationServiceException as e:
        err_msg = str(e)
        if "Access denied" in err_msg:
            raise HTTPException(status_code=403, detail=err_msg)
        elif "Authentication required" in err_msg:
            raise HTTPException(status_code=401, detail=err_msg)
        elif "not found" in err_msg.lower():
            raise HTTPException(status_code=404, detail=err_msg)
        else:
            raise HTTPException(status_code=400, detail=err_msg)
    except Exception as e:
        logger.error(f"Error in archive rectangle annotation API: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
