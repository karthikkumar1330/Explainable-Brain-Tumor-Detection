import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


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

# ngrok Tunnel Configuration
USE_NGROK = os.getenv("USE_NGROK", "false").lower() == "true"
NGROK_AUTHTOKEN = os.getenv("NGROK_AUTHTOKEN", "")


_base_url = None

def get_base_url() -> str:
    global _base_url
    if _base_url:
        return _base_url
    
    if USE_NGROK:
        try:
            from pyngrok import ngrok
            tunnels = ngrok.get_tunnels()
            for t in tunnels:
                if t.proto == "https" or t.public_url.startswith("https://"):
                    if str(STREAMLIT_PORT) in t.config.get("addr", ""):
                        _base_url = t.public_url
                        return _base_url
        except Exception:
            pass
            
    return os.getenv("BASE_URL", f"http://localhost:{STREAMLIT_PORT}")

def set_base_url(url: str) -> None:
    global _base_url
    _base_url = url

