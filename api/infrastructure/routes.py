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
    """API Endpoint: Receives a raw brain MRI image slice upload.

    Returns:
        Local path where the uploaded file is temporarily cached.
    """
    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
    temp_filename = f"upload_{int(time.time())}_{file.filename}"
    temp_filepath = os.path.join(TEMP_UPLOAD_DIR, temp_filename)
    
    try:
        with open(temp_filepath, "wb") as f:
            shutil.copyfileobj(file.file, f)
        return {"filename": temp_filename, "filepath": temp_filepath}
    except Exception as e:
        logger.error(f"Error handling file upload: {e}")
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
        
        with torch.no_grad():
            output_seg = model_seg(input_tensor_seg)
            if seg_config["deep_supervision"]:
                output_seg = output_seg[-1]
            output_seg = torch.sigmoid(output_seg).squeeze(0).squeeze(0).cpu().numpy()

        # Binarize and connected components filter
        bin_mask = (output_seg > 0.5).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
        filtered_mask = np.zeros_like(bin_mask)
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] >= 100:
                filtered_mask[labels == label] = 1

        # Resize to original scale
        orig_h, orig_w = img_bgr.shape[:2]
        final_mask = cv2.resize(filtered_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # Save to disk
        os.makedirs(OUTPUT_REPORTS_DIR, exist_ok=True)
        mask_filename = f"mask_{int(time.time())}.jpg"
        mask_path = os.path.join(OUTPUT_REPORTS_DIR, mask_filename)
        cv2.imwrite(mask_path, (final_mask * 255).astype(np.uint8))

        return {"mask_path": mask_path, "tumor_pixel_count": int(np.sum(final_mask))}
    except Exception as e:
        logger.error(f"Segmentation run error: {e}")
        raise HTTPException(status_code=500, detail=f"Segmentation processing failed: {e}")


@router.post("/explainability")
def run_explainability(filepath: str, target_class: int = 1):
    """API Endpoint: Runs Grad-CAM attention heatmap on target classification index."""
    if not os.path.exists(filepath):
        raise HTTPException(status_code=400, detail="Target MRI file path not found.")
    
    try:
        img_bgr = cv2.imread(filepath, cv2.IMREAD_COLOR)
        config_cls = ClassificationConfig()
        image_tensor_cls = preprocess_classification_image(filepath, config_cls)

        explain_service = GradCAMService(
            model=model_cls,
            target_layer=model_cls.backbone.features[8],
            device=device
        )
        explain_use_case = ExplainPredictionUseCase(
            predict_use_case=predict_use_case,
            explain_service=explain_service,
            logger=logger
        )
        _, heatmap = explain_use_case.execute(image_tensor_cls, target_class=target_class)

        os.makedirs(OUTPUT_REPORTS_DIR, exist_ok=True)
        base_cam_name = f"gradcam_{int(time.time())}"
        save_explainability_outputs(
            original_image=img_bgr,
            heatmap=heatmap,
            output_dir=OUTPUT_REPORTS_DIR,
            base_filename=base_cam_name,
            alpha=0.6,
            logger=logger
        )
        return {
            "heatmap_path": os.path.join(OUTPUT_REPORTS_DIR, f"{base_cam_name}_heatmap.png"),
            "overlay_path": os.path.join(OUTPUT_REPORTS_DIR, f"{base_cam_name}_overlay.png")
        }
    except Exception as e:
        logger.error(f"Explainability hook execution failure: {e}")
        raise HTTPException(status_code=500, detail=f"Grad-CAM hook execution failed: {e}")


@router.post("/report")
def generate_clinical_report_pipeline(filepath: str, intake: PatientIntake):
    """API Endpoint: Runs the complete end-to-end MRI diagnostics report pipeline."""
    if not os.path.exists(filepath):
        raise HTTPException(status_code=400, detail="Target upload MRI file path not found.")
    
    t_start = time.time()
    try:
        img_bgr = cv2.imread(filepath, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Failed to read uploaded image.")

        # 1. Classification
        t_cls = time.time()
        config_cls = ClassificationConfig()
        image_tensor_cls = preprocess_classification_image(filepath, config_cls)
        classification_result = predict_use_case.execute(image_tensor_cls)
        cls_latency = time.time() - t_cls

        # 2. Explainability
        t_cam = time.time()
        explain_service = GradCAMService(
            model=model_cls,
            target_layer=model_cls.backbone.features[8],
            device=device
        )
        explain_use_case = ExplainPredictionUseCase(
            predict_use_case=predict_use_case,
            explain_service=explain_service,
            logger=logger
        )
        _, heatmap = explain_use_case.execute(image_tensor_cls, target_class=classification_result.label)
        
        base_cam_name = f"{intake.patient_id}_api_gradcam"
        save_explainability_outputs(
            original_image=img_bgr,
            heatmap=heatmap,
            output_dir=OUTPUT_REPORTS_DIR,
            base_filename=base_cam_name,
            alpha=0.6,
            logger=logger
        )
        heatmap_path = os.path.join(OUTPUT_REPORTS_DIR, f"{base_cam_name}_heatmap.png")
        overlay_path = os.path.join(OUTPUT_REPORTS_DIR, f"{base_cam_name}_overlay.png")
        cam_latency = time.time() - t_cam

        # 3. Segmentation
        t_seg = time.time()
        input_tensor_seg = preprocess_segmentation_image(img_bgr, seg_config["input_h"], seg_config["input_w"])
        input_tensor_seg = input_tensor_seg.to(device)
        with torch.no_grad():
            output_seg = model_seg(input_tensor_seg)
            if seg_config["deep_supervision"]:
                output_seg = output_seg[-1]
            output_seg = torch.sigmoid(output_seg).squeeze(0).squeeze(0).cpu().numpy()

        bin_mask = (output_seg > 0.5).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_mask, connectivity=8)
        filtered_mask = np.zeros_like(bin_mask)
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] >= 100:
                filtered_mask[labels == label] = 1

        orig_h, orig_w = img_bgr.shape[:2]
        final_mask = cv2.resize(filtered_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        mask_path = os.path.join(OUTPUT_REPORTS_DIR, f"{intake.patient_id}_api_mask.jpg")
        cv2.imwrite(mask_path, (final_mask * 255).astype(np.uint8))
        seg_latency = time.time() - t_seg

        # 4. Morphological Analysis
        morph_analyzer = OpenCVTumorAnalyzer(low_thresh=1.0, med_thresh=5.0, high_thresh=15.0)
        morph_use_case = AnalyzeTumorUseCase(analyzer=morph_analyzer, logger=logger)
        clinical_data = morph_use_case.execute(
            mask=final_mask,
            patient_id=intake.patient_id,
            tumor_class=classification_result.class_name,
            original_image=img_bgr,
            pixel_spacing_mm=intake.pixel_spacing_mm,
        )
        segmentation_metrics = clinical_data.analysis

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
            device=str(device),
            execution_time_sec=total_exec_time,
            classification_model_path=CLS_CHECKPOINT,
            segmentation_model_path=SEG_CHECKPOINT,
            classification_latency_sec=cls_latency,
            segmentation_latency_sec=seg_latency,
            explainability_latency_sec=cam_latency,
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
        )

        generator = MarkdownJSONReportGenerator()
        report_use_case = GenerateIntegratedReportUseCase(report_generator=generator, logger=logger)
        md_file, json_file, pdf_file = report_use_case.execute(report=clinical_report, output_dir=OUTPUT_REPORTS_DIR)

        # 7. Database Persistence
        db_repo = SQLitePersistenceRepository(db_path=DEFAULT_DB_PATH)
        db_repo.initialize_db()
        report_db_id = db_repo.save_report(clinical_report, output_dir=OUTPUT_REPORTS_DIR)

        return {
            "report_id": report_db_id,
            "patient_id": intake.patient_id,
            "diagnosis": classification_result.class_name,
            "confidence": float(classification_result.confidence_score),
            "severity": severity_assessment.category.value,
            "tumor_area_mm2": float(segmentation_metrics.tumor_area_mm2),
            "execution_time_sec": total_exec_time,
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
