import os
import streamlit as st


def get_css_filepath() -> str:
    """Returns the absolute path to the design_system.css file (checking ui/ first)."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    primary_path = os.path.join(base_dir, "ui", "design_system.css")
    if os.path.exists(primary_path):
        return primary_path
    fallback_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(fallback_dir, "design_system.css")


def init_theme_state() -> str:
    """Initializes theme state in st.session_state if not present. Preserves user preference via query params or session state."""
    if "theme" not in st.session_state:
        # Check query params for theme preference
        params = st.query_params
        if "theme" in params and params["theme"] in ["light", "dark"]:
            st.session_state["theme"] = params["theme"]
        else:
            st.session_state["theme"] = "dark"
    return st.session_state["theme"]


def set_theme(theme_name: str) -> None:
    """Sets active theme ('light' or 'dark') in session state and query params."""
    theme = "light" if theme_name.lower() == "light" else "dark"
    st.session_state["theme"] = theme
    st.query_params["theme"] = theme


def toggle_theme():
    current = st.session_state.get("theme", "dark")
    new_theme = "light" if current == "dark" else "dark"
    set_theme(new_theme)
    st.rerun()

def inject_design_system() -> None:
    """Injects the design system CSS and sets theme attribute on root and all page wrappers instantly."""
    theme = init_theme_state()
    css_path = get_css_filepath()
    
    css_content = ""
    if os.path.exists(css_path):
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
            
    # Inject CSS + JavaScript attribute setter & localStorage persistence
    st.markdown(
        f"""
        <style>
        {css_content}
        </style>
        <script>
            (function() {{
                var activeTheme = '{theme}';
                try {{
                    localStorage.setItem('aurascan_theme', activeTheme);
                }} catch (e) {{}}
                
                function applyTheme(t) {{
                    document.documentElement.setAttribute('data-theme', t);
                    if (document.body) {{
                        document.body.setAttribute('data-theme', t);
                    }}
                    var targets = [
                        '[data-testid="stAppViewContainer"]',
                        '[data-testid="stSidebar"]',
                        '[data-testid="stHeader"]',
                        '.main'
                    ];
                    targets.forEach(function(selector) {{
                        var els = document.querySelectorAll(selector);
                        els.forEach(function(el) {{
                            el.setAttribute('data-theme', t);
                        }});
                    }});
                }}
                
                applyTheme(activeTheme);
                setTimeout(function() {{ applyTheme(activeTheme); }}, 50);
                setTimeout(function() {{ applyTheme(activeTheme); }}, 200);
            }})();
        </script>
        """,
        unsafe_allow_html=True
    )
