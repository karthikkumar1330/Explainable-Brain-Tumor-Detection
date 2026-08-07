import os
import streamlit as st
import streamlit.components.v1 as components


def get_css_filepath() -> str:
    """Returns the absolute path to the design_system.css file."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "design_system.css")


def init_theme_state() -> str:
    """Initializes theme state in st.session_state to always be 'dark'."""
    st.session_state["theme"] = "dark"
    if "theme" in st.query_params:
        st.query_params["theme"] = "dark"
    return "dark"

def clean_html(html_str: str) -> str:
    """Strips leading/trailing whitespace and removes line indentation to prevent Markdown code block parsing."""
    if not html_str:
        return ""
    lines = html_str.strip().splitlines()
    return "\n".join(line.strip() for line in lines)


def st_html(html_str: str, container=None) -> None:
    """Renders HTML in Streamlit cleanly without indentation code block bugs."""
    if not html_str:
        return
    target = container if container is not None else st
    clean = clean_html(html_str)
    target.markdown(clean, unsafe_allow_html=True)


def inject_design_system() -> None:
    """Injects the design system CSS and active theme variable overrides instantly."""
    css_path = get_css_filepath()
    
    css_content = ""
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()

    theme_variables = """
    :root, html, body, [data-theme="dark"], .stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"], [data-testid="stHeader"], .main {
        --bg: #0F172A;
        --surface: #1E293B;
        --card: #1E293B;
        --text: #E2E8F0;
        --muted: #64748B;
        --primary: #2196F3;
        --border: #334155;

        --bg-primary: #0F172A;
        --bg-secondary: #1E293B;
        --bg-tertiary: #0F172A;
        --bg-card: #1E293B;
        --bg-card-hover: #24324D;
        
        --border-color: rgba(255, 255, 255, 0.08);
        --border-highlight: #2196F3;

        --text-primary: #E2E8F0;
        --text-secondary: #94A3B8;
        --text-muted: #64748B;
        --text-accent: #2196F3;
        --text-inverse: #FFFFFF;

        --accent-primary: #2196F3;
        --accent-primary-hover: #1e88e5;
        --accent-secondary: #2196F3;
        --accent-glow: none;

        --status-success: #22C55E;
        --status-success-bg: rgba(34, 197, 94, 0.12);
        --status-warning: #F59E0B;
        --status-warning-bg: rgba(245, 158, 11, 0.12);
        --status-danger: #EF4444;
        --status-danger-bg: rgba(239, 68, 68, 0.12);
        --status-info: #2196F3;
        --status-info-bg: rgba(33, 150, 243, 0.12);

        --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.15);
        --shadow-md: 0 2px 4px rgba(0, 0, 0, 0.20);
        --shadow-lg: 0 4px 12px rgba(0, 0, 0, 0.25);

        --auth-glass-bg: rgba(30, 41, 59, 0.75);
        --auth-glass-border: rgba(255, 255, 255, 0.08);
        --auth-glass-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        --auth-badge-bg: rgba(15, 23, 42, 0.55);
        --auth-badge-border: rgba(255, 255, 255, 0.06);
        --auth-badge-hover-bg: rgba(15, 23, 42, 0.75);
        --btn-gradient: linear-gradient(135deg, #2196F3 0%, #1e88e5 100%);
        --btn-gradient-hover: linear-gradient(135deg, #1e88e5 0%, #1565c0 100%);
        --btn-gradient-shadow: 0 4px 14px 0 rgba(33, 150, 243, 0.3);
        
        --radius: 14px;
        --radius-sm: 8px;
        --radius-md: 14px;
        --radius-lg: 14px;
    }
    """
        
    # Inject CSS styles & dynamic theme variables cleanly
    st_html(f"<style>\\n{css_content}\\n{theme_variables}\\n</style>")

