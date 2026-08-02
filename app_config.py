import os


# Database configuration
DATABASE_PATH = os.getenv("DATABASE_PATH", "outputs/clinical_reports.db")

# Model checkpoints
CLS_CHECKPOINT_PATH = os.getenv(
    "CLS_CHECKPOINT_PATH", "models/classification/efficientnet_b0_brain_tumor.pth"
)
SEG_CHECKPOINT_PATH = os.getenv("SEG_CHECKPOINT_PATH", "models/brain_tumor_unext/model.pth")
SEG_CONFIG_PATH = os.getenv("SEG_CONFIG_PATH", "models/brain_tumor_unext/config.yml")

# Directory pathways
REPORTS_DIR = os.getenv("REPORTS_DIR", "outputs/clinical_reports")
TEMP_DIR = os.getenv("TEMP_DIR", "outputs/temp_uploads")

# Server ports
REST_PORT = int(os.getenv("REST_PORT", 8000))
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", 8501))
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", 5000))
