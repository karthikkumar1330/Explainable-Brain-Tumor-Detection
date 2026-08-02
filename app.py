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
        ["Dashboard Analytics", "Inference Scan Analysis", "Patient Database History"]
    )
    st.sidebar.divider()

    # Hardware selector
    device_choice = st.sidebar.selectbox("Inference Execution Device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
    device = torch.device(device_choice)

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

            st.divider()
            uploaded_file = st.file_uploader("Upload MRI Brain Scan (PNG/JPG/TIF)", type=["png", "jpg", "jpeg", "tif", "tiff"])

            submit_btn = st.button("Run Diagnostic Analysis", disabled=(uploaded_file is None))

        with col_results:
            if not submit_btn:
                st.info("Please fill demographics details, upload the MRI scan slice, and click 'Run Diagnostic Analysis'.")
            else:
                with st.spinner("Processing MRI slice (Classification, Explainability hooks, Segmentation, Morphology...)..."):
                    t_start = time.time()
                    
                    # Convert file bytes to opencv BGR image
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    original_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
                    if original_image is None:
                        st.error("Failed to read the uploaded MRI file. Please upload a valid image.")
                        return
                    
                    # Write image to a temporary file
                    os.makedirs("outputs/temp_uploads", exist_ok=True)
                    temp_image_path = os.path.join("outputs", "temp_uploads", f"{patient_id}_temp_input.png")
                    cv2.imwrite(temp_image_path, original_image)
                    
                    # 1. Run Classification (EfficientNet-B0)
                    t_cls = time.time()
                    config_cls = ClassificationConfig()
                    image_tensor_cls = preprocess_classification_image(temp_image_path, config_cls)
                    classification_result = predict_use_case.execute(image_tensor_cls)
                    cls_latency = time.time() - t_cls

                    # 2. Run Explainability (Grad-CAM)
                    t_cam = time.time()
                    explain_service = GradCAMService(
                        model=model_cls,
                        target_layer=model_cls.backbone.features[8],
                        device=device
                    )
                    explain_use_case = ExplainPredictionUseCase(
                        predict_use_case=predict_use_case,
                        explain_service=explain_service,
                        logger=logging.getLogger("streamlit_app")
                    )
                    _, heatmap = explain_use_case.execute(image_tensor_cls, target_class=classification_result.label)
                    
                    os.makedirs(OUTPUT_REPORTS_DIR, exist_ok=True)
                    base_cam_name = f"{patient_id}_gradcam"
                    save_explainability_outputs(
                        original_image=original_image,
                        heatmap=heatmap,
                        output_dir=OUTPUT_REPORTS_DIR,
                        base_filename=base_cam_name,
                        alpha=0.6,
                        logger=logging.getLogger("streamlit_app")
                    )
                    heatmap_path = os.path.join(OUTPUT_REPORTS_DIR, f"{base_cam_name}_heatmap.png")
                    overlay_path = os.path.join(OUTPUT_REPORTS_DIR, f"{base_cam_name}_overlay.png")
                    cam_latency = time.time() - t_cam

                    # 3. Run Segmentation (UNeXt)
                    t_seg = time.time()
                    input_tensor_seg = preprocess_segmentation_image(
                        original_image, seg_config["input_h"], seg_config["input_w"]
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

                    # Binarize
                    bin_mask = (output_seg > 0.5).astype(np.uint8)

                    # Post-process: connected components area filter >= 100 pixels
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                        bin_mask, connectivity=8
                    )
                    filtered_mask = np.zeros_like(bin_mask)
                    for label in range(1, num_labels):
                        area = stats[label, cv2.CC_STAT_AREA]
                        if area >= 100:
                            filtered_mask[labels == label] = 1

                    # Resize to original MRI size
                    orig_h, orig_w = original_image.shape[:2]
                    final_mask = cv2.resize(
                        filtered_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
                    )

                    # Save mask
                    mask_path = os.path.join(OUTPUT_REPORTS_DIR, f"{patient_id}_segmentation_mask.jpg")
                    cv2.imwrite(mask_path, (final_mask * 255).astype(np.uint8))
                    seg_latency = time.time() - t_seg

                    # 4. Morphological Analysis
                    morph_analyzer = OpenCVTumorAnalyzer(low_thresh=1.0, med_thresh=5.0, high_thresh=15.0)
                    morph_use_case = AnalyzeTumorUseCase(analyzer=morph_analyzer, logger=logging.getLogger("streamlit_app"))
                    clinical_data = morph_use_case.execute(
                        mask=final_mask,
                        patient_id=patient_id,
                        tumor_class=classification_result.class_name,
                        original_image=original_image,
                        pixel_spacing_mm=pixel_spacing,
                    )
                    segmentation_metrics = clinical_data.analysis

                    # 5. Rule-Based Severity Assessment
                    severity_classifier = RuleBasedSeverityClassifier()
                    severity_use_case = AssessSeverityUseCase(classifier=severity_classifier, logger=logging.getLogger("streamlit_app"))
                    severity_assessment = severity_use_case.execute(
                        tumor_type=classification_result.class_name,
                        tumor_area_mm2=segmentation_metrics.tumor_area_mm2,
                        tumor_percentage=segmentation_metrics.tumor_percentage_brain,
                    )

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
                    )

                    generator = MarkdownJSONReportGenerator()
                    report_use_case = GenerateIntegratedReportUseCase(report_generator=generator, logger=logging.getLogger("streamlit_app"))
                    md_file, json_file, pdf_file = report_use_case.execute(report=clinical_report, output_dir=OUTPUT_REPORTS_DIR)

                    # 7. Persist to SQLite Database
                    db_repo = SQLitePersistenceRepository(db_path=DEFAULT_DB_PATH)
                    db_repo.initialize_db()
                    db_report_id = db_repo.save_report(clinical_report, output_dir=OUTPUT_REPORTS_DIR)

                    # Cleanup temp input
                    if os.path.exists(temp_image_path):
                        os.remove(temp_image_path)

                # ================= DISPLAY OUTPUTS =================
                st.success(f"MRI diagnostic analysis completed successfully! Assigned Database ID: #{db_report_id}")
                
                st.subheader("Diagnostic Assessment Summary")
                res_col1, res_col2, res_col3 = st.columns(3)
                with res_col1:
                    st.metric("Primary Diagnosis Class", classification_result.class_name)
                with res_col2:
                    st.metric("Model Confidence Score", f"{classification_result.confidence_score:.2%}")
                with res_col3:
                    st.metric("AI Severity Category", severity_assessment.category.value.upper())

                st.divider()

                # Shaded card for severity rule and disclaimer
                sev_color = "#ef4444" if severity_assessment.category.value.lower() == "high" else "#f59e0b" if severity_assessment.category.value.lower() == "medium" else "#10b981"
                st.markdown(f"""
                    <div style="background-color: #0f172a; border-left: 5px solid {sev_color}; padding: 15px; border-radius: 8px; margin-bottom: 20px;">
                        <h4 style="margin: 0; color: #f8fafc; font-size: 14px; font-weight: 700;">Matched Risk Decision Rule</h4>
                        <p style="margin: 6px 0 0 0; color: #cbd5e1; font-size: 13px;">{severity_assessment.rule_description}</p>
                    </div>
                """, unsafe_allow_html=True)

                st.subheader("Clinical Imaging Visualizations")
                img_col1, img_col2, img_col3 = st.columns(3)
                with img_col1:
                    st.image(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB), caption="Original brain MRI scan slice", use_container_width=True)
                with img_col2:
                    if os.path.exists(overlay_path):
                        st.image(overlay_path, caption="Grad-CAM Attention Overlay", use_container_width=True)
                with img_col3:
                    if os.path.exists(mask_path):
                        st.image(mask_path, caption="Binarized UNeXt Segmentation Mask", use_container_width=True)

                st.divider()

                st.subheader("Quantitative Morphology Stats")
                morph_col1, morph_col2, morph_col3 = st.columns(3)
                with morph_col1:
                    st.metric("Estimated Tumor Area", f"{segmentation_metrics.tumor_area_mm2:.2f} mm²")
                with morph_col2:
                    st.metric("Tumor occupancy % (Brain)", f"{segmentation_metrics.tumor_percentage_brain:.4f}%")
                with morph_col3:
                    st.metric("Tumor Pixel count", f"{segmentation_metrics.pixel_count:,} px")

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
                        cr.id as report_id, p.patient_id, p.name as patient_name, p.age, p.gender,
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


if __name__ == "__main__":
    main()
