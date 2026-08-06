import os
import streamlit as st


def get_css_filepath() -> str:
    """Returns the absolute path to the design_system.css file."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, "design_system.css")


def init_theme_state() -> str:
    """Initializes theme state in st.session_state if not present. Default is dark mode."""
    if "theme" not in st.session_state:
        st.session_state["theme"] = "dark"
    return st.session_state["theme"]


def toggle_theme() -> None:
    """Toggles between 'dark' and 'light' mode in session state."""
    current = st.session_state.get("theme", "dark")
    st.session_state["theme"] = "light" if current == "dark" else "dark"


def inject_design_system() -> None:
    """Injects the design system CSS and sets theme attribute on page wrapper."""
    theme = init_theme_state()
    css_path = get_css_filepath()
    
    css_content = ""
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
            
    # Inject CSS + JavaScript attribute setter for light/dark theme
    st.markdown(
        f"""
        <style>
        {css_content}
        </style>
        <script>
            document.documentElement.setAttribute('data-theme', '{theme}');
            var appView = document.querySelector('[data-testid="stAppViewContainer"]');
            if (appView) {{
                appView.setAttribute('data-theme', '{theme}');
            }}
        </script>
        """,
        unsafe_allow_html=True
    )
