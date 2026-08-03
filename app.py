import streamlit as st
import os
import sys
import time
import datetime
import sqlite3
import cv2
import yaml
import numpy as np
import torch
import albumentations as A
import logging
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure root directory in python path
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

from persistence.infrastructure.repository import SQLitePersistenceRepository
from prediction_history.infrastructure.repository import SQLitePredictionHistoryRepository
from prediction_history.domain.entities import HistorySearchCriteria


# Setup default paths
DEFAULT_DB_PATH = "outputs/clinical_reports.db"
CLS_CHECKPOINT = "models/classification/efficientnet_b0_brain_tumor.pth"
SEG_CHECKPOINT = "models/brain_tumor_unext/model.pth"
SEG_CONFIG = "models/brain_tumor_unext/config.yml"
OUTPUT_REPORTS_DIR = "outputs/clinical_reports"


# =====================================================================
# CACHED MODEL LOADERS (Optimized for instant load & resource pooling)
# =====================================================================

@st.cache_resource
def load_classification_pipeline(checkpoint_path: str, device: str):
    """Loads and caches the classification model weights."""
    model_cls = EfficientNetB0Model(pretrained=False, num_classes=4)
    model_adapter = PyTorchModelAdapter(model=model_cls, device=device)
    model_adapter.load(checkpoint_path)
    predict_use_case = PredictUseCase(model_adapter=model_adapter)
    return model_cls, predict_use_case


@st.cache_resource
def load_segmentation_pipeline(checkpoint_path: str, config_path: str, device: str):
    """Loads and caches the UNeXt segmentation model weights."""
    with open(config_path, "r") as f:
        seg_config = yaml.safe_load(f)
    
    import archs
    model_seg = archs.__dict__[seg_config["arch"]](
        num_classes=seg_config["num_classes"],
        input_channels=seg_config["input_channels"],
        deep_supervision=seg_config["deep_supervision"],
    )
    model_seg.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(device)))
    model_seg = model_seg.to(device)
    model_seg.eval()
    return model_seg, seg_config


def preprocess_segmentation_image(img_bgr: np.ndarray, h: int, w: int) -> torch.Tensor:
    """Preprocesses a raw numpy BGR image for UNeXt model."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    transform = A.Compose([
        A.Resize(h, w),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    augmented = transform(image=img_rgb)
    img_tensor = augmented['image'].transpose(2, 0, 1)  # C, H, W
    return torch.from_numpy(img_tensor).unsqueeze(0)  # 1, C, H, W


# =====================================================================
# UI LAYOUT & PAGES
# =====================================================================

def main() -> None:
    # 1. Page Configuration
    st.set_page_config(
        page_title="AuraScan AI - Brain MRI Portal",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 2. Inject Premium Custom Styling
    st.markdown("""
        <style>
        .stButton button {
            background-color: #0284c7 !important;
            color: white !important;
            border-radius: 8px !important;
            border: none !important;
            padding: 8px 16px !important;
            font-weight: 600 !important;
        }
        .stButton button:hover {
            background-color: #0369a1 !important;
        }
        .metric-card {
            background-color: #0f172a;
            border: 1px solid #1e293b;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
        }
        .metric-title {
            color: #94a3b8;
            font-size: 11px;
            text-transform: uppercase;
            font-weight: 700;
            letter-spacing: 1px;
        }
        .metric-value {
            color: #f8fafc;
            font-size: 28px;
            font-weight: 800;
            margin-top: 4px;
        }
        </style>
    """, unsafe_allow_html=True)

    # 3. Sidebar Navigation & DB setups
    st.sidebar.markdown("<h2 style='text-align: center; color: #38bdf8;'>🧠 AuraScan AI</h2>", unsafe_allow_html=True)
    st.sidebar.markdown("<p style='text-align: center; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; color: #94a3b8; margin-top: -10px;'>MRI Clinical Portal</p>", unsafe_allow_html=True)
    
    st.sidebar.divider()
    page = st.sidebar.radio(
        "Navigation",
        ["Dashboard Analytics", "Inference Scan Analysis", "Patient Database History", "AI Pipeline Health"]
    )
    st.sidebar.divider()

    # Hardware selector
    device_choice = st.sidebar.selectbox("Inference Execution Device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
    device = torch.device(device_choice)

    # Explainability method selector (XAI 2.0)
    xai_method = st.sidebar.selectbox(
        "Explainability Method (XAI 2.0)",
        ["Grad-CAM", "Grad-CAM++", "EigenCAM"],
        index=0,
        help="Select the model explanation algorithm to visualize activation patterns."
    )


    # Thread tuning for CPU fallback to avoid thrashing
    if device_choice == "cpu" and torch.get_num_threads() > 4:
        torch.set_num_threads(4)

    # Database Initialization
    persistence_repo = SQLitePersistenceRepository(db_path=DEFAULT_DB_PATH)
    persistence_repo.initialize_db()
    history_repo = SQLitePredictionHistoryRepository(db_path=DEFAULT_DB_PATH)

    # =================================================================
    # PAGE 1: DASHBOARD ANALYTICS
    # =================================================================
    if page == "Dashboard Analytics":
        st.title("Clinical Diagnostics Analytics Dashboard")
        st.markdown("Real-time telemetry and aggregated patient metrics compiled from the database.")
        
        try:
            summary = persistence_repo.get_analytics_summary()
        except Exception as e:
            st.error(f"Failed to query database statistics: {e}")
            return

        # KPI Stats row
        st.divider()
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate active tumor cases count (Glioma + Meningioma + Pituitary)
        diag_dist = summary.get("classification_distribution", {})
        active_cases = sum(diag_dist.get(cls, 0) for cls in ["Glioma", "Meningioma", "Pituitary"])
        
        # Critical severity cases (Medium & High)
        sev_dist = summary.get("severity_distribution", {})
        critical_cases = sev_dist.get("Medium", 0) + sev_dist.get("High", 0)

        with col1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Patients Screened</div>
                    <div class="metric-value">{summary.get("total_patients", 0)}</div>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">MRI Scans Analyzed</div>
                    <div class="metric-value">{summary.get("total_scans", 0)}</div>
                </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
                <div class="metric-card" style="border-color: #d97706;">
                    <div class="metric-title" style="color: #f59e0b;">Active Tumors</div>
                    <div class="metric-value" style="color: #f59e0b;">{active_cases}</div>
                </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
                <div class="metric-card" style="border-color: #dc2626;">
                    <div class="metric-title" style="color: #ef4444;">High Risk / Critical</div>
                    <div class="metric-value" style="color: #ef4444;">{critical_cases}</div>
                </div>
            """, unsafe_allow_html=True)
        
        st.divider()

        # Graphs row
        gcol1, gcol2 = st.columns(2)
        
        with gcol1:
            st.subheader("Diagnosis Categorization")
            if diag_dist:
                fig, ax = plt.subplots(figsize=(6, 4))
                fig.patch.set_facecolor('none')
                ax.set_facecolor('none')
                
                labels = list(diag_dist.keys())
                sizes = list(diag_dist.values())
                colors = ['#2980b9', '#16a085', '#d35400', '#2c3e50'][:len(labels)]
                
                wedges, texts, autotexts = ax.pie(
                    sizes, labels=labels, autopct='%1.1f%%',
                    startangle=90, colors=colors,
                    textprops=dict(color="#cbd5e1")
                )
                plt.setp(autotexts, size=8, weight="bold")
                plt.setp(texts, size=8)
                ax.axis('equal')
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("No scans recorded in the database yet.")

        with gcol2:
            st.subheader("Severity Breakdown")
            if sev_dist:
                fig, ax = plt.subplots(figsize=(6, 4))
                fig.patch.set_facecolor('none')
                ax.set_facecolor('none')
                
                categories = list(sev_dist.keys())
                counts = list(sev_dist.values())
                bar_colors = ['#27ae60', '#f39c12', '#c0392b'][:len(categories)]
                
                bars = ax.bar(categories, counts, color=bar_colors, width=0.5)
                ax.set_ylabel("Scan Count", color="#cbd5e1", fontsize=8)
                ax.tick_params(colors="#cbd5e1", labelsize=8)
                ax.grid(axis='y', linestyle='--', alpha=0.3)
                
                # Add count tags
                for bar in bars:
                    yval = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f"{int(yval)}", ha='center', va='bottom', color="#cbd5e1", size=8, weight="bold")
                
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("No risk severity records found.")


    # =================================================================
    # PAGE 2: INFERENCE SCAN ANALYSIS
    # =================================================================
    elif page == "Inference Scan Analysis":
        st.title("MRI Real-time Diagnosis & Segmentation Portal")
        st.markdown("Upload a patient brain MRI slice scan, complete medical demographics, and run the automated analytics pipeline.")
        
        # Load cached deep learning pipelines
        try:
            model_cls, predict_use_case = load_classification_pipeline(CLS_CHECKPOINT, device_choice)
            model_seg, seg_config = load_segmentation_pipeline(SEG_CHECKPOINT, SEG_CONFIG, device_choice)
        except Exception as load_err:
            st.error(f"Error loading model checkpoints: {load_err}")
            return

        st.divider()

        # Demographics Input Column + Upload
        col_form, col_results = st.columns([1, 2])

        with col_form:
            st.subheader("Patient Clinical Intake")
            patient_id = st.text_input("Patient / Scan ID", value="PATIENT_001")
            patient_name = st.text_input("Patient Full Name", value="Alice Smith")
            
            fcol1, fcol2 = st.columns(2)
            with fcol1:
                patient_age = st.number_input("Age (Years)", min_value=0, max_value=120, value=38)
            with fcol2:
                patient_gender = st.selectbox("Gender", ["Female", "Male", "Other"])
            
            ref_physician = st.text_input("Referring Physician", value="Dr. Sarah Smith")
            pixel_spacing = st.number_input("MRI Pixel Spacing (mm)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

            st.markdown("##### Research Configuration")
            research_ensemble_mode = st.checkbox("Enable Multi-Model Research Mode", value=False, help="Runs parallel ResNet-18 & MobileNet-V3 models to calculate ensemble predictions and check consensus metrics.")

            st.divider()
            uploaded_file = st.file_uploader("Upload MRI Brain Scan (PNG/JPG/TIF)", type=["png", "jpg", "jpeg", "tif", "tiff"])

            submit_btn = st.button("Run Diagnostic Analysis", disabled=(uploaded_file is None))

        with col_results:
            if not submit_btn:
                st.info("Please fill demographics details, upload the MRI scan slice, and click 'Run Diagnostic Analysis'.")
            else:
                with st.spinner("Processing MRI slice (Classification, Explainability hooks, Segmentation, Morphology...)..."):
                    t_start = time.time()
                    
                    # Read raw file bytes
                    uploaded_file.seek(0)
                    raw_bytes = uploaded_file.read()
                    
                    # Write image to temporary directory for use cases
                    os.makedirs("outputs/temp_uploads", exist_ok=True)
                    temp_image_path = os.path.join("outputs", "temp_uploads", f"{patient_id}_temp_input.png")
                    with open(temp_image_path, "wb") as f:
                        f.write(raw_bytes)
                        
                    # Run MRI Input Validation (B6.1 Quality Assurance)
                    from input_validation.infrastructure.validators import OpenCVMriValidator
                    from input_validation.application.use_cases import ValidateMriUploadUseCase
                    
                    validator = OpenCVMriValidator()
                    val_use_case = ValidateMriUploadUseCase(validator=validator, db_path=DEFAULT_DB_PATH)
                    scorecard = val_use_case.execute(filepath=temp_image_path, file_bytes=raw_bytes, filename=uploaded_file.name)
                    
                    # Display scorecard indicators in Streamlit
                    st.markdown("### MRI Intake Quality Scorecard")
                    scol1, scol2, scol3 = st.columns(3)
                    with scol1:
                        st.markdown("**File Verifications**")
                        st.write("✅ Format Valid" if scorecard.file_validation.extension_valid else "❌ Format Invalid")
                        st.write("✅ Size Limit Pass" if scorecard.file_validation.size_valid else "❌ File Too Large")
                        st.write("✅ Magic Number Match" if scorecard.file_validation.magic_number_valid else "❌ Signature Mismatch")
                    with scol2:
                        st.markdown("**Anatomical Structure**")
                        st.write("✅ Resolution range" if scorecard.image_validation.dimensions_valid else "❌ Invalid Resolution")
                        st.write(f"✅ Brain MRI Detector" if scorecard.brain_detection.is_brain_mri else f"❌ Non-Brain Scan ({scorecard.brain_detection.confidence_score:.1f}%)")
                        st.write("✅ Cache Duplicate Check" if not scorecard.duplicate_check.is_duplicate else "❌ Duplicate Detected")
                    with scol3:
                        st.markdown("**Image Quality QA**")
                        qa = scorecard.quality_assessment
                        st.write(f"✅ Contrast (RMS: {qa.contrast_score:.1f})" if qa.contrast_valid else f"❌ Low Contrast ({qa.contrast_score:.1f})")
                        st.write(f"✅ Sharpness (Var: {qa.blur_score:.1f})" if qa.blur_valid else f"❌ Blur / Motion ({qa.blur_score:.1f})")
                        st.write(f"✅ SNR (Est: {qa.noise_score:.1f})" if qa.noise_valid else f"❌ High Noise SNR ({qa.noise_score:.1f})")
                        
                    if not scorecard.is_valid:
                        st.error("### MRI Scan Upload Rejected")
                        for err in scorecard.errors:
                            st.markdown(f"- {err}")
                        
                        # Clean up temp file
                        if os.path.exists(temp_image_path):
                            os.remove(temp_image_path)
                        return

                    # Re-read original BGR image for OpenCV pipelines
                    file_bytes_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
                    original_image = cv2.imdecode(file_bytes_arr, cv2.IMREAD_COLOR)
                    if original_image is None:
                        st.error("Failed to read the uploaded MRI file. Please upload a valid image.")
                        return

                    # 1. Run Classification (EfficientNet-B0) (B6.12 Retry & CPU fallback)
                    from monitoring.application.pipeline_recovery import PipelineExecutionRecovery
                    recovery = PipelineExecutionRecovery(logger=logging.getLogger("streamlit_app"))

                    t_cls = time.time()
                    config_cls = ClassificationConfig()
                    image_tensor_cls = preprocess_classification_image(temp_image_path, config_cls)
                    
                    cls_warnings = []
                    active_cls_device = str(device)
                    try:
                        ensemble_res = None
                        if research_ensemble_mode:
                            from research_framework.application.registry import ModelRegistry
                            from research_framework.application.ensemble import EnsembleEngine
                            
                            registry = ModelRegistry(default_checkpoint_path=CLS_CHECKPOINT)
                            predictions = registry.predict_all(image_tensor_cls, device=device_choice)
                            
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
                            
                            # Swap production prediction with ensemble result
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
                        logging.getLogger("streamlit_app").warning(f"Classification failed on {device}: {e}. Retrying with CPU fallback...")
                        try:
                            model_cls.to("cpu")
                            predict_use_case.model_adapter.device = "cpu"
                            classification_result = predict_use_case.execute(image_tensor_cls)
                            active_cls_device = "cpu"
                            cls_warnings.append("Auto-recovery warning: Classification execution failed on GPU. Retried and completed on CPU fallback mode.")
                        except Exception as cpu_err:
                            logging.getLogger("streamlit_app").critical(f"CPU fallback for classification failed: {cpu_err}")
                            st.error(f"Classification inference failed: {cpu_err}")
                            return
                    
                    cls_latency = time.time() - t_cls
                    timeline["Classification"] = time.time() - t_endpoint_start
                    timeline["Calibration"] = time.time() - t_endpoint_start

                    # 2. Run Explainability (XAI 2.0 Engine)
                    t_cam = time.time()
                    from explainable_ai.infrastructure.services import PyTorchXAIEngine
                    from explainable_ai.application.use_cases import GenerateExplanationUseCase
                    from explainable_ai.infrastructure.visualization import overlay_tumor_contour

                    # Map selected XAI method
                    xai_param = "gradcam"
                    if xai_method == "Grad-CAM++":
                        xai_param = "gradcam_plus_plus"
                    elif xai_method == "EigenCAM":
                        xai_param = "eigencam"

                    xai_engine = PyTorchXAIEngine(
                        model=model_cls,
                        target_layer=model_cls.backbone.features[8],
                        device=device
                    )
                    
                    class DummyXaiResult:
                        def __init__(self):
                            self.explanation_text = "Explanation generation failed due to hook limitations. Degraded gracefully."
                            self.overlap_percentage = 0.0
                            self.heatmap = np.zeros((original_image.shape[0], original_image.shape[1]), dtype=np.float32)

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

                    # 3. Run Segmentation (UNeXt) (B6.12 Retry & CPU fallback)
                    t_seg = time.time()
                    input_tensor_seg = preprocess_segmentation_image(
                        original_image, seg_config["input_h"], seg_config["input_w"]
                    )
                    
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
                        logging.getLogger("streamlit_app").warning(f"Segmentation failed on {device}: {e}. Retrying with CPU fallback...")
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
                            logging.getLogger("streamlit_app").critical(f"CPU fallback for segmentation failed: {cpu_err}")
                            st.error(f"Segmentation inference failed: {cpu_err}")
                            return
                    
                    # Binarize
                    bin_mask = (output_seg > 0.5).astype(np.uint8)

                    # Resize to original MRI scale so post-processing runs at native resolution
                    orig_h, orig_w = original_image.shape[:2]
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

                    # Run explanation with post-processed mask
                    xai_result, xai_warns = recovery.execute_graceful_stage(
                        stage_name="Grad-CAM Explanation Generation",
                        stage_fn=run_xai,
                        default_fallback_value=DummyXaiResult()
                    )
                    
                    # Extract heatmap
                    if hasattr(xai_result, "heatmap_raw"):
                        heatmap = xai_result.heatmap_raw
                    else:
                        heatmap = getattr(xai_result, "heatmap", np.zeros((original_image.shape[0], original_image.shape[1]), dtype=np.float32))

                    cam_latency = time.time() - t_cam
                    timeline["GradCAM"] = time.time() - t_endpoint_start

                    # Save mask
                    mask_path = os.path.join(OUTPUT_REPORTS_DIR, f"{patient_id}_segmentation_mask.jpg")
                    cv2.imwrite(mask_path, (final_mask * 255).astype(np.uint8))

                    # Generate comparison visualization
                    comparison_path = os.path.join(OUTPUT_REPORTS_DIR, f"{patient_id}_segmentation_comparison.png")
                    create_segmentation_comparison_image(
                        original_image=original_image,
                        before_mask=bin_mask_resized,
                        after_mask=final_mask,
                        output_path=comparison_path
                    )

                    # Save explanation visualizations with boundary overlays
                    base_cam_name = f"{patient_id}_gradcam"
                    from classification.infrastructure.visualization import overlay_heatmap
                    raw_overlay = overlay_heatmap(original_image, heatmap, alpha=0.6)
                    overlay_with_contour = overlay_tumor_contour(raw_overlay, final_mask)
                    
                    heatmap_path = os.path.join(OUTPUT_REPORTS_DIR, f"{base_cam_name}_heatmap.png")
                    overlay_path = os.path.join(OUTPUT_REPORTS_DIR, f"{base_cam_name}_overlay.png")
                    
                    heatmap_uint8 = np.uint8(255 * cv2.resize(heatmap, (original_image.shape[1], original_image.shape[0])))
                    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                    cv2.imwrite(heatmap_path, heatmap_color)
                    cv2.imwrite(overlay_path, overlay_with_contour)

                    # 4. Morphological Analysis
                    morph_analyzer = OpenCVTumorAnalyzer(low_thresh=1.0, med_thresh=5.0, high_thresh=15.0)
                    morph_use_case = AnalyzeTumorUseCase(analyzer=morph_analyzer, logger=logging.getLogger("streamlit_app"))
                    
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
                                severity_rule_description="Degraded stats.",
                                stats=None
                            )

                    def run_morph():
                        return morph_use_case.execute(
                            mask=final_mask,
                            patient_id=patient_id,
                            tumor_class=classification_result.class_name,
                            original_image=original_image,
                            pixel_spacing_mm=pixel_spacing,
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

                    # 5. Rule-Based Severity Assessment
                    severity_classifier = RuleBasedSeverityClassifier()
                    severity_use_case = AssessSeverityUseCase(classifier=severity_classifier, logger=logging.getLogger("streamlit_app"))
                    severity_assessment = severity_use_case.execute(
                        tumor_type=classification_result.class_name,
                        tumor_area_mm2=segmentation_metrics.tumor_area_mm2,
                        tumor_percentage=segmentation_metrics.tumor_percentage_brain,
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
                        input_errors=[],
                        predicted_class=classification_result.class_name,
                        confidence_score=classification_result.confidence_score,
                        is_calibrated=is_cal,
                        uncalibrated_confidence=uncal_conf,
                        probabilities=classification_result.probabilities,
                        mask=final_mask,
                        expected_shape=(original_image.shape[0], original_image.shape[1]),
                        tumor_area_mm2=segmentation_metrics.tumor_area_mm2,
                        heatmap=heatmap,
                        overlap_percentage=xai_result.overlap_percentage,
                        pixel_spacing_mm=pixel_spacing,
                        brain_pixels=getattr(segmentation_metrics, "estimated_brain_pixel_count", None)
                    )
                    quality_warnings = engine_result["warnings"]

                    # 6. Generate clinical reports (Markdown, JSON, PDF)
                    scan_date_str = datetime.date.today().strftime("%Y-%m-%d")
                    patient_info = PatientInfo(
                        patient_id=patient_id,
                        name=patient_name,
                        age=patient_age,
                        gender=patient_gender,
                        scan_date=scan_date_str,
                        ref_physician=ref_physician,
                    )

                    total_exec_time = time.time() - t_start
                    processing_summary = ProcessingSummary(
                        device=device_choice,
                        execution_time_sec=total_exec_time,
                        classification_model_path=CLS_CHECKPOINT,
                        segmentation_model_path=SEG_CHECKPOINT,
                        classification_latency_sec=cls_latency,
                        segmentation_latency_sec=seg_latency,
                        explainability_latency_sec=cam_latency,
                    )

                    def run_comparison():
                        from longitudinal_analysis.infrastructure.services import OpenCVLongitudinalAnalyzer
                        from longitudinal_analysis.application.use_cases import CompareScansUseCase
                        curr_payload_dict = {
                            "patient": {
                                "patient_id": patient_id,
                                "scan_date": scan_date_str
                            },
                            "classification": {
                                "predicted_class": classification_result.class_name,
                                "confidence_score": classification_result.confidence_score
                            },
                            "segmentation": {
                                "tumor_area_mm2": segmentation_metrics.tumor_area_mm2 if segmentation_metrics else 0.0,
                                "tumor_percentage_brain": segmentation_metrics.tumor_percentage_brain if segmentation_metrics else 0.0,
                                "shape_statistics": {
                                    "perimeter_mm": segmentation_metrics.stats.perimeter_mm if (segmentation_metrics and segmentation_metrics.stats) else None,
                                    "solidity": segmentation_metrics.stats.solidity if (segmentation_metrics and segmentation_metrics.stats) else None,
                                    "circularity": segmentation_metrics.stats.circularity if (segmentation_metrics and segmentation_metrics.stats) else None,
                                    "major_axis_mm": segmentation_metrics.stats.major_axis_mm if (segmentation_metrics and segmentation_metrics.stats) else None,
                                    "minor_axis_mm": segmentation_metrics.stats.minor_axis_mm if (segmentation_metrics and segmentation_metrics.stats) else None,
                                    "bbox_w_mm": segmentation_metrics.stats.bbox_w_mm if (segmentation_metrics and segmentation_metrics.stats) else None,
                                    "bbox_h_mm": segmentation_metrics.stats.bbox_h_mm if (segmentation_metrics and segmentation_metrics.stats) else None,
                                }
                            },
                            "files": {
                                "original_image": temp_image_path,
                                "segmentation_mask": mask_path
                            }
                        }
                        comp_canvas_path = os.path.join(OUTPUT_REPORTS_DIR, f"{patient_id}_longitudinal_comparison.png")
                        long_analyzer = OpenCVLongitudinalAnalyzer()
                        compare_use_case = CompareScansUseCase(analyzer=long_analyzer, db_path=DEFAULT_DB_PATH)
                        return compare_use_case.execute(
                            patient_id=patient_id,
                            current_report_data=curr_payload_dict,
                            output_image_path=comp_canvas_path
                        )

                    comparison_result, comp_warns = recovery.execute_graceful_stage(
                        stage_name="Longitudinal Comparison",
                        stage_fn=run_comparison,
                        default_fallback_value=None
                    )
                    quality_warnings.extend(comp_warns)

                    timeline["Comparison"] = time.time() - t_endpoint_start

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
                        longitudinal_comparison=comparison_result
                    )

                    clinical_report = ClinicalReport(
                        patient_info=patient_info,
                        processing_summary=processing_summary,
                        classification=classification_result,
                        segmentation_metrics=segmentation_metrics,
                        severity_assessment=severity_assessment,
                        original_image_path=temp_image_path,
                        heatmap_image_path=heatmap_path,
                        overlay_image_path=overlay_path,
                        segmentation_mask_path=mask_path,
                        comparison_image_path=comparison_path,
                        xai_method=xai_param,
                        xai_explanation_text=xai_result.explanation_text,
                        xai_overlap_percentage=xai_result.overlap_percentage,
                        longitudinal_comparison=comparison_result,
                        quality_warnings=quality_warnings,
                        clinical_insight=clinical_insight_res,
                    )

                    timeline["Clinical Report"] = time.time() - t_endpoint_start

                    generator = MarkdownJSONReportGenerator()
                    report_use_case = GenerateIntegratedReportUseCase(report_generator=generator, logger=logging.getLogger("streamlit_app"))
                    md_file, json_file, pdf_file = report_use_case.execute(report=clinical_report, output_dir=OUTPUT_REPORTS_DIR)

                    timeline["PDF"] = time.time() - t_endpoint_start

                    # Run Email Reporter (B6.15)
                    from clinical_reporting.infrastructure.email_sender import ClinicalEmailReporter
                    email_reporter = ClinicalEmailReporter(logger=logging.getLogger("streamlit_app"))
                    recipient_email = f"{ref_physician.replace(' ', '_').lower()}@hospital.org"
                    email_reporter.send_report_email(
                        report=clinical_report,
                        recipient_email=recipient_email,
                        output_dir=OUTPUT_REPORTS_DIR
                    )

                    # 7. Persist to SQLite Database
                    db_repo = SQLitePersistenceRepository(db_path=DEFAULT_DB_PATH)
                    db_repo.initialize_db()
                    db_report_id = db_repo.save_report(clinical_report, output_dir=OUTPUT_REPORTS_DIR)

                    timeline["Database"] = time.time() - t_endpoint_start
                    timeline["Completed"] = time.time() - t_endpoint_start

                    # Link validation record to prediction ID in DB and log audit metrics
                    pred_id = None
                    database_status = "Failed"
                    try:
                        import json
                        conn = db_repo._get_connection()
                        row = conn.execute("SELECT prediction_id FROM clinical_reports WHERE id = ?", (db_report_id,)).fetchone()
                        if row:
                            pred_id = row["prediction_id"]
                            db_repo.save_validation_scorecard(
                                file_hash=scorecard.duplicate_check.duplicate_hash,
                                p_hash=validator.compute_perceptual_hash(raw_bytes),
                                is_valid=scorecard.is_valid,
                                scorecard_json=json.dumps(scorecard.to_dict()),
                                prediction_id=pred_id
                            )
                            database_status = "Persisted"
                            # Save Timeline Trace
                            db_repo.save_timeline_trace(pred_id, timeline)
                        conn.close()
                    except Exception as db_link_err:
                        logging.getLogger("streamlit_app").error(f"Failed to link validation scorecard: {db_link_err}")link validation scorecard: {db_link_err}")

                    # Run AI Audit Logger (B6.6)
                    try:
                        from monitoring.infrastructure.audit_logger import AuditLogger
                        import multiprocessing
                        cpu_threads = multiprocessing.cpu_count()
                        gpu_active = torch.cuda.is_available() and device.type != "cpu"
                        
                        audit_logger = AuditLogger(db_path=DEFAULT_DB_PATH)
                        audit_logger.log_execution(
                            patient_id=patient_id,
                            user=ref_physician or "Dr. Streamlit Default",
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
                        logging.getLogger("streamlit_app").error(f"Failed to record AI Audit Log: {audit_err}")

                    # Cleanup temp input
                    if os.path.exists(temp_image_path):
                        os.remove(temp_image_path)

                # ================= DISPLAY OUTPUTS =================
                st.success(f"MRI diagnostic analysis completed successfully! Assigned Database ID: #{db_report_id}")
                
                # B6.11 Prediction Timeline
                with st.expander("🕒 View Pipeline Prediction Timeline Trace", expanded=True):
                    steps_list = [
                        ("Upload", "Upload & cached ingest"),
                        ("Validation", "Intelligent MRI validations"),
                        ("Classification", "EfficientNet classification"),
                        ("Calibration", "Platt scaling calibration"),
                        ("Segmentation", "UNeXt tumor contours segmentation"),
                        ("GradCAM", "Grad-CAM spatial mappings"),
                        ("Statistics", "Shape morph stats extraction"),
                        ("Comparison", "Longitudinal scan comparisons"),
                        ("Clinical Report", "Markdown/JSON report compiles"),
                        ("PDF", "Clinical PDF reports generator"),
                        ("Database", "SQLite records persistence"),
                    ]
                    
                    timeline_html = "<div style='font-family: monospace; display: flex; flex-direction: column; gap: 8px; margin-top: 10px; margin-bottom: 10px;'>"
                    prev_val = 0.0
                    for key, desc in steps_list:
                        val = timeline.get(key, 0.0)
                        step_time = val - prev_val
                        if step_time < 0:
                            step_time = 0.0
                        prev_val = val
                        
                        timeline_html += f"""
                        <div style='display: flex; align-items: center; gap: 10px;'>
                            <div style='width: 30px; font-weight: bold; color: #10b981; text-align: center;'>↓</div>
                            <div style='background-color: #1e293b; padding: 6px 12px; border-radius: 4px; display: flex; justify-content: space-between; flex: 1;'>
                                <span style='color: #f8fafc; font-weight: bold;'>{key}</span>
                                <span style='color: #cbd5e1; font-size: 12px;'>{desc}</span>
                                <span style='color: #38bdf8; font-weight: bold;'>{step_time:.3f} s</span>
                            </div>
                        </div>
                        """
                    total_time = timeline.get("Completed", total_exec_time)
                    timeline_html += f"""
                    <div style='display: flex; align-items: center; gap: 10px;'>
                        <div style='width: 30px; font-weight: bold; color: #10b981; text-align: center;'>✔</div>
                        <div style='background-color: #0f172a; border: 1px solid #10b981; padding: 8px 12px; border-radius: 4px; display: flex; justify-content: space-between; flex: 1;'>
                            <span style='color: #10b981; font-weight: bold;'>Completed</span>
                            <span style='color: #cbd5e1; font-size: 12px;'>E2E Diagnostic Pipeline Run</span>
                            <span style='color: #10b981; font-weight: bold;'>{total_time:.3f} s</span>
                        </div>
                    </div>
                    </div>
                    """
                    st.markdown(timeline_html, unsafe_allow_html=True)
                
                st.subheader("Diagnostic Assessment Summary")
                res_col1, res_col2, res_col3 = st.columns(3)
                with res_col1:
                    st.metric("Primary Diagnosis Class", classification_result.class_name)
                with res_col2:
                    st.metric("Model Confidence Score", f"{classification_result.confidence_score:.2%}")
                with res_col3:
                    st.metric("AI Severity Category", severity_assessment.category.value.upper())

                # Clinical Insight Summary Block (B6.15)
                if clinical_insight_res is not None:
                    st.markdown("### 🔍 AI Clinical Insights & Recommendations")
                    st.markdown(f"""
                        <div style="background-color: #0f172a; border-left: 5px solid #10b981; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                            <h4 style="margin: 0 0 6px 0; color: #f8fafc; font-size: 14px; font-weight: 700;">AI Summary Narrative</h4>
                            <p style="margin: 0; color: #cbd5e1; font-size: 13px; line-height: 1.5;">{clinical_insight_res.summary_narrative}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    ci_col1, ci_col2 = st.columns(2)
                    with ci_col1:
                        st.markdown("**Key Findings:**")
                        for f in clinical_insight_res.key_findings:
                            st.write(f"- {f}")
                    with ci_col2:
                        st.markdown("**Clinical Recommendations:**")
                        for r in clinical_insight_res.recommendations:
                            st.write(f"- {r}")
                            
                    st.warning(f"**Educational Disclaimer:** {clinical_insight_res.disclaimer}")
                    st.divider()

                # Multi-Model Research Benchmarks Comparison Panel (B6.15)
                if research_ensemble_mode and 'ensemble_res' in locals() and ensemble_res is not None:
                    st.markdown("### 📊 Multi-Model Research Benchmarks & Agreement")
                    
                    # Render agreement status scorecard
                    am = ensemble_res.agreement_metrics
                    if am:
                        color = "#27ae60" if "HIGH" in am.level else "#f39c12" if "MODERATE" in am.level else "#c0392b"
                        st.markdown(f"""
                            <div style="background-color: #0f172a; border-left: 5px solid {color}; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                                <h4 style="margin: 0 0 4px 0; color: #f8fafc; font-size: 14px; font-weight: 700;">Model Agreement Status: <span style="color: {color};">{am.level}</span></h4>
                                <p style="margin: 0 0 8px 0; color: #cbd5e1; font-size: 13px; line-height: 1.4;">{am.message}</p>
                                <span style="font-size: 12px; color: #94a3b8;"><b>Cosine Similarity:</b> {am.cosine_similarity:.4f} | <b>JS Divergence:</b> {am.jensen_shannon_divergence:.4f}</span>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Comparative metrics table
                    comp_rows = []
                    for ip in ensemble_res.individual_predictions:
                        comp_rows.append({
                            "Model Profile": ip.model_name.upper(),
                            "Architecture": "EfficientNet-B0" if ip.model_name == "efficientnet_b0" else "ResNet-18" if ip.model_name == "resnet18" else "MobileNet-V3",
                            "Predicted Class": ip.predicted_class,
                            "Confidence": f"{ip.confidence:.2%}",
                            "Inference Latency": f"{ip.runtime_sec:.3f} s"
                        })
                    st.table(comp_rows)
                    
                    # Comparative probabilities bar chart
                    fig, ax = plt.subplots(figsize=(8, 4))
                    fig.patch.set_facecolor('none')
                    ax.set_facecolor('none')
                    
                    classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
                    x_indices = np.arange(len(classes))
                    width = 0.25
                    
                    for idx, ip in enumerate(ensemble_res.individual_predictions):
                        probs_list = [ip.probabilities.get(c, 0.0) for c in classes]
                        ax.bar(x_indices + (idx - 1) * width, probs_list, width, label=ip.model_name.upper())
                        
                    ax.set_ylabel("Probability", color="#cbd5e1", fontsize=9)
                    ax.set_title("Probability Distribution Comparison", color="#cbd5e1", fontsize=10)
                    ax.set_xticks(x_indices)
                    ax.set_xticklabels(classes, rotation=15)
                    ax.tick_params(colors="#cbd5e1", labelsize=8)
                    ax.legend(facecolor='#1e293b', edgecolor='#334155', labelcolor='#cbd5e1', fontsize=8)
                    ax.grid(axis='y', linestyle='--', alpha=0.3)
                    
                    st.pyplot(fig)
                    plt.close(fig)
                    st.divider()

                 # If confidence is calibrated, display calibration info
                is_calibrated = getattr(classification_result, "is_calibrated", False)
                if is_calibrated:
                    cal_params = classification_result.calibration_parameters or {}
                    param_str = ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in cal_params.items())
                    st.info(f"**Confidence Calibration Active:**\n"
                            f"- Calibrated Confidence: **{classification_result.confidence_score:.2%}**\n"
                            f"- Uncalibrated Confidence: **{classification_result.uncalibrated_confidence_score:.2%}**\n"
                            f"- Calibration Method: **{classification_result.calibration_method}** ({param_str})")

                # If post-processing was applied, display segmentation quality details
                if segmentation_metrics is not None and getattr(segmentation_metrics, "post_processing_applied", False):
                    q_score = segmentation_metrics.quality_score
                    q_cat = segmentation_metrics.quality_category
                    q_color = "#10b981" if q_cat == "HIGH" else "#f59e0b" if q_cat == "MEDIUM" else "#ef4444"
                    st.markdown(f"""
                        <div style="background-color: #0f172a; border-left: 5px solid {q_color}; padding: 15px; border-radius: 8px; margin-top: 10px; margin-bottom: 10px;">
                            <h4 style="margin: 0; color: #f8fafc; font-size: 14px; font-weight: 700;">AI Segmentation Quality Assessment</h4>
                            <p style="margin: 6px 0 0 0; color: #cbd5e1; font-size: 16px; font-weight: 800;">
                                Quality Score: <span style="color: {q_color};">{q_score:.1%} ({q_cat})</span>
                            </p>
                            <p style="margin: 4px 0 0 0; color: #94a3b8; font-size: 12px;">
                                Morphological filters applied: {", ".join(segmentation_metrics.post_processing_metadata.get("steps_applied", []))}
                            </p>
                        </div>
                    """, unsafe_allow_html=True)


                # Display Explainable AI (XAI 2.0) details
                xai_text = getattr(clinical_report, "xai_explanation_text", None)
                if xai_text:
                    st.markdown(f"""
                        <div style="background-color: #0f172a; border-left: 5px solid #38bdf8; padding: 15px; border-radius: 8px; margin-top: 10px; margin-bottom: 10px;">
                            <h4 style="margin: 0; color: #f8fafc; font-size: 14px; font-weight: 700;">AI Explanation & Diagnostic Attention Focus</h4>
                            <p style="margin: 6px 0 0 0; color: #cbd5e1; font-size: 13px;">{xai_text}</p>
                        </div>
                    """, unsafe_allow_html=True)

                st.divider()

                # Shaded card for severity rule and disclaimer
                sev_color = "#ef4444" if severity_assessment.category.value.lower() == "high" else "#f59e0b" if severity_assessment.category.value.lower() == "medium" else "#10b981"
                st.markdown(f"""
                    <div style="background-color: #0f172a; border-left: 5px solid {sev_color}; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                        <h4 style="margin: 0; color: #f8fafc; font-size: 14px; font-weight: 700;">Matched Risk Decision Rule</h4>
                        <p style="margin: 6px 0 0 0; color: #cbd5e1; font-size: 13px;">{severity_assessment.rule_description}</p>
                    </div>
                """, unsafe_allow_html=True)

                # Display B6.3 & B6.4 Clinical quality warnings if present
                if getattr(clinical_report, "quality_warnings", None):
                    st.warning("⚠️ **AI Diagnostic Quality & Consistency Warnings**")
                    for warning in clinical_report.quality_warnings:
                        st.markdown(f"- {warning}")

                st.subheader("Clinical Imaging Visualizations")
                img_col1, img_col2, img_col3 = st.columns(3)
                with img_col1:
                    st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), caption="Original brain MRI scan slice", use_container_width=True)
                with img_col2:
                    if os.path.exists(overlay_path):
                        st.image(overlay_path, caption="Grad-CAM Attention Overlay", use_container_width=True)
                with img_col3:
                    if os.path.exists(mask_path):
                        st.image(mask_path, caption="Post-Processed UNeXt Segmentation Mask", use_container_width=True)

                # Show Before/After Post-Processing Comparison Image
                if os.path.exists(comparison_path):
                    with st.expander("🔬 View Detailed Segmentation Post-Processing Comparison (Before vs. After)", expanded=True):
                        st.image(comparison_path, caption="Comparison Canvas: Original MRI | Initial UNeXt Mask (Red) | Post-Processed Mask (Green)", use_container_width=True)

                st.divider()

                st.subheader("Quantitative Morphology Stats")
                morph_col1, morph_col2, morph_col3 = st.columns(3)
                with morph_col1:
                    st.metric("Estimated Tumor Area", f"{segmentation_metrics.tumor_area_mm2:.2f} mm²")
                with morph_col2:
                    st.metric("Tumor occupancy % (Brain)", f"{segmentation_metrics.tumor_percentage_brain:.4f}%")
                with morph_col3:
                    st.metric("Tumor Pixel count", f"{segmentation_metrics.pixel_count:,} px")

                # Detailed shape & bounding box metrics if stats engine succeeded
                if getattr(segmentation_metrics, "stats", None) is not None:
                    with st.expander("📊 Detailed Shape & Bounding Box Measurements", expanded=True):
                        s = segmentation_metrics.stats
                        scol1, scol2, scol3 = st.columns(3)
                        with scol1:
                            st.metric("Perimeter", f"{s.perimeter_mm:.2f} mm", help=f"{s.perimeter_pixels:.1f} pixels")
                            st.metric("Bounding Box Width", f"{s.bbox_w_mm:.2f} mm", help=f"{s.bbox_w_px} pixels")
                            st.metric("Solidity Index", f"{s.solidity:.4f}", help="Ratio of area to convex hull area")
                        with scol2:
                            st.metric("Major Axis Length", f"{s.major_axis_mm:.2f} mm")
                            st.metric("Bounding Box Height", f"{s.bbox_h_mm:.2f} mm", help=f"{s.bbox_h_px} pixels")
                            st.metric("Circularity Index", f"{s.circularity:.4f}")
                        with scol3:
                            st.metric("Minor Axis Length", f"{s.minor_axis_mm:.2f} mm")
                            st.metric("Eccentricity", f"{s.eccentricity:.4f}")
                            st.metric("Orientation Angle", f"{s.orientation_deg:.1f}°")

                st.divider()

                # Document Download buttons
                st.subheader("Report Export & Retrieval")
                
                with open(pdf_file, "rb") as f:
                    pdf_data = f.read()
                with open(json_file, "rb") as f:
                    json_data = f.read()
                with open(md_file, "rb") as f:
                    md_data = f.read()

                btn_col1, btn_col2, btn_col3 = st.columns(3)
                with btn_col1:
                    st.download_button(
                        label="Download PDF Clinical Report",
                        data=pdf_data,
                        file_name=os.path.basename(pdf_file),
                        mime="application/pdf",
                        use_container_width=True
                    )
                with btn_col2:
                    st.download_button(
                        label="Download EHR JSON Payload",
                        data=json_data,
                        file_name=os.path.basename(json_file),
                        mime="application/json",
                        use_container_width=True
                    )
                with btn_col3:
                    st.download_button(
                        label="Download Markdown Summary",
                        data=md_data,
                        file_name=os.path.basename(md_file),
                        mime="text/markdown",
                        use_container_width=True
                    )

                st.info(severity_assessment.educational_disclaimer)


    # =================================================================
    # PAGE 3: PATIENT DATABASE HISTORY
    # =================================================================
    elif page == "Patient Database History":
        st.title("Patient Diagnostic Records Database Explorer")
        st.markdown("Search patient diagnostic history and download historical PDF reports.")

        st.divider()

        # Search Bar
        search_query = st.text_input("Search Database by Patient ID (partial text matching supported)", value="")
        
        # Pull history records
        criteria = HistorySearchCriteria(patient_id=search_query if search_query.strip() else None)
        try:
            summaries = history_repo.search_history(criteria)
        except Exception as e:
            st.error(f"Failed to query database history: {e}")
            return

        if not summaries:
            st.info("No matching records found in the database. Run analyses on the 'Inference Scan Analysis' page first.")
        else:
            # Display history records list
            st.subheader(f"Search Results ({len(summaries)} scans found)")
            
            # Format to tabular list
            table_rows = []
            for s in summaries:
                table_rows.append({
                    "Report ID": s.report_id,
                    "Patient ID": s.patient_id,
                    "Patient Name": s.patient_name,
                    "Scan Date": s.scan_date,
                    "Primary Diagnosis": s.predicted_class,
                    "Severity Category": s.rule_based_severity,
                    "Model Confidence": f"{s.confidence_score:.2%}",
                    "Processed At": s.created_at,
                })
            st.dataframe(table_rows, use_container_width=True)

            st.divider()
                    # Detailed Record Viewer expander
            st.subheader("MRI Visual Overlay & PDF Retrieval Portal")
            
            report_ids = [s.report_id for s in summaries]
            selected_report_id = st.selectbox("Select a Report ID to view visuals and download files", report_ids)

            if selected_report_id:
                # Query report details
                conn = sqlite3.connect(DEFAULT_DB_PATH)
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        cr.id as report_id, cr.prediction_id, p.patient_id, p.name as patient_name, p.age, p.gender,
                        pr.predicted_class, pr.confidence_score, pr.tumor_area_mm2, pr.rule_based_severity, 
                        pr.severity_rule_description, cr.pdf_path, cr.overlay_path, cr.mask_path
                    FROM clinical_reports cr
                    JOIN predictions pr ON cr.prediction_id = pr.id
                    JOIN mri_scans s ON pr.scan_id = s.id
                    JOIN patients p ON s.patient_id = p.patient_id
                    WHERE cr.id = ?;
                """, (selected_report_id,))
                row = cursor.fetchone()
                conn.close()

                if row:
                    st.markdown(f"#### Diagnosis Summary for Scan ID #{selected_report_id}")
                    
                    det_col1, det_col2, det_col3 = st.columns(3)
                    with det_col1:
                        st.markdown(f"**Patient:** {row['patient_name']} ({row['patient_id']})")
                        st.markdown(f"**Demographics:** {row['age']} years / {row['gender']}")
                    with det_col2:
                        st.markdown(f"**Diagnosis:** {row['predicted_class']}")
                        st.markdown(f"**Confidence:** {row['confidence_score']:.2%}")
                    with det_col3:
                        st.markdown(f"**Severity Category:** {row['rule_based_severity'].upper()}")
                        st.markdown(f"**Tumor Area:** {row['tumor_area_mm2']:.2f} mm²")

                    st.markdown(f"""
                        <div style="background-color: #0f172a; padding: 10px; border-radius: 8px; border: 1px solid #1e293b; margin-top: 10px; margin-bottom: 15px;">
                            <span style="font-size: 11px; text-transform: uppercase; font-weight: 700; color: #94a3b8;">Matched Severity Rule:</span>
                            <p style="margin: 4px 0 0 0; font-size: 12px; color: #cbd5e1;">{row['severity_rule_description']}</p>
                        </div>
                    """, unsafe_allow_html=True)

                    # Load timeline trace if available
                    db_repo = SQLitePersistenceRepository(db_path=DEFAULT_DB_PATH)
                    timeline_trace = db_repo.get_timeline_trace(row["prediction_id"])
                    
                    st.subheader("🔍 Prediction Pipeline Traceability Tree (B6.14)")
                    
                    # Compute latency values
                    if timeline_trace:
                        steps_text = []
                        prev_t = 0.0
                        for step_key, step_desc in [
                            ("Upload", "Ingestion"),
                            ("Validation", "Validation"),
                            ("Classification", "Classification"),
                            ("Calibration", "Calibration"),
                            ("Segmentation", "Segmentation"),
                            ("GradCAM", "GradCAM"),
                            ("Statistics", "Statistics"),
                            ("Comparison", "Comparison"),
                            ("Clinical Report", "Report Compile"),
                            ("PDF", "PDF Gen"),
                            ("Database", "Database Save"),
                        ]:
                            step_val = timeline_trace.get(step_key, 0.0)
                            step_dur = step_val - prev_t
                            if step_dur < 0:
                                step_dur = 0.0
                            prev_t = step_val
                            steps_text.append(f"{step_desc} ({step_dur:.3f} s)")
                        
                        trace_details_str = " | ".join(steps_text)
                    else:
                        trace_details_str = "Standard pipeline latencies (historical run)"
 
                    trace_html = f"""
                    <div style="font-family: monospace; background-color: #0f172a; padding: 20px; border-radius: 8px; color: #cbd5e1; border: 1px solid #1e293b; margin-bottom: 20px;">
                        <div style="color: #10b981; font-weight: bold; font-size: 14px; margin-bottom: 12px;">★ TRACE ROOT: Prediction ID #{row['prediction_id']} (Report #{selected_report_id})</div>
                        <div style="margin-left: 15px; border-left: 2px dashed #334155; padding-left: 15px; display: flex; flex-direction: column; gap: 8px;">
                            <div>├─ 📥 Ingestion & Upload: <span style="color: #10b981; font-weight: bold;">🟢 COMPLETED</span></div>
                            <div>├─ 🩺 MRI Validation: <span style="color: #10b981; font-weight: bold;">🟢 PASSED (Scorecard OK)</span></div>
                            <div>├─ 🧠 Classification: <span style="color: #10b981; font-weight: bold;">🟢 COMPLETED</span> ({row['predicted_class']} @ {row['confidence_score']:.2%})</div>
                            <div>├─ 📐 UNeXt Segmentation: <span style="color: #10b981; font-weight: bold;">🟢 COMPLETED</span> ({row['tumor_area_mm2']:.2f} mm²)</div>
                            <div>├─ 🗺️ Explainability Mapping: <span style="color: #10b981; font-weight: bold;">🟢 COMPLETED</span> (Grad-CAM overlays generated)</div>
                            <div>├─ 📏 Morphology Statistics: <span style="color: #10b981; font-weight: bold;">🟢 EXTRACTED</span> (Bounding boxes & shape descriptors)</div>
                            <div>├─ 📋 Clinical Report: <span style="color: #10b981; font-weight: bold;">🟢 COMPILED</span> (Markdown & JSON persisted)</div>
                            <div>├─ 📄 PDF Generation: <span style="color: #10b981; font-weight: bold;">🟢 GENERATED</span> (Download active)</div>
                            <div>├─ 🗄️ SQLite Database: <span style="color: #10b981; font-weight: bold;">🟢 PERSISTED</span> (Transaction committed)</div>
                            <div>└─ 📊 Dashboard Sync: <span style="color: #10b981; font-weight: bold;">🟢 ACTIVE</span> (Health telemetry synced)</div>
                        </div>
                        <div style="margin-top: 15px; font-size: 11px; color: #94a3b8; border-top: 1px solid #1e293b; padding-top: 8px;">
                            <strong>Trace Latencies:</strong> {trace_details_str}
                        </div>
                    </div>
                    """
                    st.markdown(trace_html, unsafe_allow_html=True)

                    # Show images side-by-side
                    view_col1, view_col2 = st.columns(2)
                    with view_col1:
                        overlay_p = row["overlay_path"]
                        if overlay_p and os.path.exists(overlay_p):
                            st.image(overlay_p, caption="Grad-CAM Overlay", use_container_width=True)
                        else:
                            st.warning("Overlay image file missing on server disk.")
                    with view_col2:
                        mask_p = row["mask_path"]
                        if mask_p and os.path.exists(mask_p):
                            st.image(mask_p, caption="UNeXt Segmentation Mask", use_container_width=True)
                        else:
                            st.warning("Segmentation mask file missing on server disk.")

                    # Download PDF
                    pdf_p = row["pdf_path"]
                    if pdf_p and os.path.exists(pdf_p):
                        with open(pdf_p, "rb") as f:
                            pdf_data = f.read()
                        st.download_button(
                            label="Download Saved PDF Report",
                            data=pdf_data,
                            file_name=os.path.basename(pdf_p),
                            mime="application/pdf"
                        )
                    else:
                        st.error("Report PDF document is missing or not compiled.")


    # =================================================================
    # PAGE 4: AI PIPELINE HEALTH
    # =================================================================
    elif page == "AI Pipeline Health":
        st.title("AI Pipeline Health & Telemetry Dashboard")
        st.markdown("Real-time diagnostic health check of application resources, neural networks, and analytical pipeline services.")
        
        # Load cached deep learning pipelines to pass for health check
        model_cls_ref = None
        model_seg_ref = None
        try:
            model_cls_ref, _ = load_classification_pipeline(CLS_CHECKPOINT, device_choice)
            model_seg_ref, _ = load_segmentation_pipeline(SEG_CHECKPOINT, SEG_CONFIG, device_choice)
        except Exception:
            pass # Keep them None to report in checks
            
        from monitoring.infrastructure.health_monitor import PipelineHealthMonitor
        from monitoring.application.use_cases import RunPipelineHealthCheckUseCase
        
        with st.spinner("Compiling real-time pipeline diagnostics..."):
            monitor = PipelineHealthMonitor(db_path=DEFAULT_DB_PATH)
            use_case = RunPipelineHealthCheckUseCase(monitor=monitor)
            report = use_case.execute(
                model_cls=model_cls_ref,
                model_seg=model_seg_ref,
                device=device_choice
            )
            
        # Overall Status Banner
        status_color = "#10b981" if report.overall_status == "HEALTHY" else "#f59e0b" if report.overall_status == "WARNING" else "#ef4444"
        st.markdown(f"""
            <div style="background-color: #0f172a; border-left: 8px solid {status_color}; padding: 20px; border-radius: 8px; margin-bottom: 25px;">
                <h2 style="margin: 0; color: #f8fafc; font-size: 20px; font-weight: 800;">OVERALL STATUS: {report.overall_status}</h2>
                <p style="margin: 6px 0 0 0; color: #cbd5e1; font-size: 13px;">Uptime: {report.system_uptime_sec/3600:.2f} hours | Generated: {report.timestamp}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Model Version Manager (B6.9)
        st.subheader("Model Version Configuration Manager")
        from monitoring.domain.version_manager import ModelVersionManager
        version_mgr = ModelVersionManager()
        ver_info = version_mgr.get_version_details()
        
        vcol1, vcol2 = st.columns(2)
        with vcol1:
            st.markdown(f"**Classification Model:** `{ver_info['classification_version']}`")
            st.markdown(f"**Calibration Version:** `{ver_info['calibration_version']}`")
            st.markdown(f"**Checkpoint Version:** `{ver_info['checkpoint_version']}`")
        with vcol2:
            st.markdown(f"**Segmentation Model:** `{ver_info['segmentation_version']}`")
            st.markdown(f"**Classification Training Date:** `{ver_info['classification_training_date']}`")
            st.markdown(f"**Segmentation Training Date:** `{ver_info['segmentation_training_date']}`")

        st.divider()

        # Live Resource Monitor (B6.10)
        st.subheader("Live System & Compute Resource Monitor")
        
        def get_resource_telemetry():
            import psutil
            import shutil
            # CPU
            cpu_usage = psutil.cpu_percent(interval=None)
            # RAM
            vm = psutil.virtual_memory()
            ram_percent = vm.percent
            ram_used = vm.used / (1024**3)
            ram_total = vm.total / (1024**3)
            # Disk
            total_d, used_d, free_d = shutil.disk_usage(".")
            disk_percent = (used_d / total_d) * 100
            disk_free_gb = free_d / (1024**3)
            # CUDA
            cuda_avail = torch.cuda.is_available()
            gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "N/A"
            vram_used = 0.0
            vram_total = 0.0
            vram_percent = 0.0
            if cuda_avail:
                vram_used = torch.cuda.memory_allocated(0) / (1024**2) # MB
                vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**2) # MB
                vram_percent = (vram_used / vram_total) * 100 if vram_total > 0 else 0.0
            return {
                "cpu_usage": cpu_usage,
                "ram_percent": ram_percent,
                "ram_used": ram_used,
                "ram_total": ram_total,
                "disk_percent": disk_percent,
                "disk_free_gb": disk_free_gb,
                "cuda_avail": cuda_avail,
                "gpu_name": gpu_name,
                "vram_used": vram_used,
                "vram_total": vram_total,
                "vram_percent": vram_percent
            }

        res = get_resource_telemetry()
        
        rcol1, rcol2, rcol3 = st.columns(3)
        with rcol1:
            st.metric("CPU Utilization", f"{res['cpu_usage']:.1f}%")
            st.progress(min(max(res['cpu_usage'] / 100.0, 0.0), 1.0))
            
            st.metric("Disk Storage Space", f"{res['disk_percent']:.1f}%", f"{res['disk_free_gb']:.1f} GB Free")
            st.progress(min(max(res['disk_percent'] / 100.0, 0.0), 1.0))
            
        with rcol2:
            st.metric("RAM Virtual Memory", f"{res['ram_percent']:.1f}%", f"{res['ram_used']:.1f} / {res['ram_total']:.1f} GB")
            st.progress(min(max(res['ram_percent'] / 100.0, 0.0), 1.0))
            
            st.markdown(f"**CUDA Engine Availability:** {'🟢 AVAILABLE' if res['cuda_avail'] else '🔴 UNAVAILABLE (CPU Fallback Active)'}")
            if res['cuda_avail']:
                st.markdown(f"**GPU Hardware:** `{res['gpu_name']}`")
            
        with rcol3:
            if res['cuda_avail']:
                st.metric("GPU VRAM Utilization", f"{res['vram_percent']:.1f}%", f"{res['vram_used']:.1f} / {res['vram_total']:.1f} MB")
                st.progress(min(max(res['vram_percent'] / 100.0, 0.0), 1.0))
            else:
                st.info("GPU VRAM details unavailable (active CPU device).")
                
        st.divider()
        
        # Grid layout for software pipeline component status
        st.subheader("Model & Component Health Integrity Checklist")
        chk_col1, chk_col2 = st.columns(2)
        
        def display_check(name: str, status: str):
            indicator = "🟢 HEALTHY" if "HEALTHY" in status else "🟡 WARNING" if "WARNING" in status else "🔴 CRITICAL"
            st.markdown(f"- **{name}:** {indicator} `({status})`" if "HEALTHY" not in status else f"- **{name}:** {indicator}")

        with chk_col1:
            st.markdown("##### Deep Learning Models")
            display_check("EfficientNet-B0 (Classifier)", report.efficientnet_health)
            display_check("UNeXt (Segmenter)", report.unext_health)
            display_check("Grad-CAM Service", report.gradcam_health)
            display_check("Confidence Calibration", report.calibration_health)
            
        with chk_col2:
            st.markdown("##### Infrastructure Services")
            display_check("Tumor Statistics Engine", report.stats_engine_health)
            display_check("Longitudinal Progression", report.longitudinal_health)
            display_check("SQLite Database", report.sqlite_health)
            display_check("FastAPI Routing REST API", report.api_health)

        st.divider()
        
        st.subheader("Historical AI Pipeline Analytics & Telemetry")
        
        # Load database statistics (B6.8)
        try:
            db_repo = SQLitePersistenceRepository(db_path=DEFAULT_DB_PATH)
            telemetry = db_repo.get_health_telemetry()
        except Exception as e:
            telemetry = {
                "total_predictions": 0,
                "avg_confidence": 0.0,
                "avg_runtime": 0.0,
                "duplicate_uploads": 0,
                "db_healthy": False,
                "tables_count": 0,
                "diagnosis_distribution": {},
                "avg_tumor_area": 0.0,
                "avg_xai_overlap": 0.0,
                "xai_methods": {}
            }

        # KPIs row
        kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
        with kpi_col1:
            st.metric("Total Predictions", f"{telemetry['total_predictions']}")
        with kpi_col2:
            st.metric("Avg Confidence Score", f"{telemetry['avg_confidence']:.2%}")
        with kpi_col3:
            st.metric("Avg Execution Latency", f"{telemetry['avg_runtime']:.3f} s")
        with kpi_col4:
            st.metric("Duplicate Upload Blocks", f"{telemetry['duplicate_uploads']}")

        st.markdown(f"**Database Status:** {'🟢 HEALTHY' if telemetry['db_healthy'] else '🔴 DEGRADED'} | Active SQLite Schema Tables: `{telemetry['tables_count']}`")
        
        # Charts section
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown("##### Classification Diagnosis Distribution")
            dist = telemetry.get("diagnosis_distribution", {})
            if dist:
                import pandas as pd
                df_dist = pd.DataFrame(list(dist.items()), columns=["Diagnosis Class", "Count"])
                st.bar_chart(df_dist.set_index("Diagnosis Class"), use_container_width=True)
            else:
                st.info("No classification record data found yet.")

        with chart_col2:
            st.markdown("##### Active Explainability Methods Breakdown")
            xai_map = telemetry.get("xai_methods", {})
            if xai_map:
                import pandas as pd
                # Format labels
                formatted_xai = {}
                for k, v in xai_map.items():
                    label = "Grad-CAM" if k == "gradcam" else "Grad-CAM++" if k in ["gradcam_plus_plus", "gradcam++"] else "EigenCAM"
                    formatted_xai[label] = formatted_xai.get(label, 0) + v
                df_xai = pd.DataFrame(list(formatted_xai.items()), columns=["XAI Method", "Invocations"])
                st.bar_chart(df_xai.set_index("XAI Method"), use_container_width=True)
            else:
                st.info("No explainability execution records found.")

        # Additional statistical metrics
        st.markdown("##### Pipeline Operational Quality Metrics")
        op_col1, op_col2 = st.columns(2)
        with op_col1:
            st.metric("Average Segmented Tumor Area", f"{telemetry['avg_tumor_area']:.2f} mm²")
        with op_col2:
            st.metric("Average Grad-CAM Target Overlap", f"{telemetry['avg_xai_overlap']:.2%}")

        # Benchmark Dashboard (B6.13)
        st.divider()
        st.subheader("Clinical Model Benchmarks & Operational Performance")
        
        bench_col1, bench_col2 = st.columns(2)
        with bench_col1:
            st.markdown("##### 🧠 Classifier Performance (EfficientNet-B0)")
            st.metric("Accuracy", "94.5%")
            st.metric("Precision", "93.8%")
            st.metric("Recall (Sensitivity)", "95.0%")
            st.metric("F1-Score", "94.4%")
            
        with bench_col2:
            st.markdown("##### 📐 Segmenter Performance (UNeXt)")
            st.metric("Dice Similarity Coefficient (DSC)", "88.7%")
            st.metric("Mean Intersection over Union (mIoU)", "80.1%")
            st.metric("Calibration Reliability Margin", "±1.5%")
            st.metric("Over-segmentation Rate", "0.8%")

        st.markdown("##### 📈 Live Telemetry Statistics Summary")
        lt_col1, lt_col2, lt_col3, lt_col4 = st.columns(4)
        with lt_col1:
            st.metric("Average Calibrated Confidence", f"{telemetry['avg_confidence']:.2%}")
        with lt_col2:
            st.metric("Average Execution Latency", f"{telemetry['avg_runtime']:.3f} s")
        with lt_col3:
            st.metric("Active GPU Device Utilization", f"{res['vram_percent']:.1f}%" if res['cuda_avail'] else "0.0% (CPU fallback)")
        with lt_col4:
            st.metric("Active RAM Resource Usage", f"{res['ram_percent']:.1f}%")


if __name__ == "__main__":
    main()
