import streamlit as st
from app.design_system.theme import toggle_theme


def render_header(user=None) -> None:
    """Renders the top enterprise SaaS navbar header with theme toggle and user badge."""
    current_theme = st.session_state.get("theme", "dark")
    theme_icon = "☀️ Light Mode" if current_theme == "dark" else "🌙 Dark Mode"
    
    user_html = ""
    if user:
        user_name = user.get("full_name", "User")
        user_role = user.get("role", "doctor").upper()
        user_html = f"""
            <div class="user-badge-pill">
                <span>👤 {user_name}</span>
                <span style="font-size: 10px; opacity: 0.8;">[{user_role}]</span>
            </div>
        """
    else:
        user_html = """
            <div class="user-badge-pill" style="background: rgba(148, 163, 184, 0.12); color: #94a3b8; border-color: rgba(148, 163, 184, 0.2);">
                <span>🔒 Guest Session</span>
            </div>
        """

    header_html = f"""
        <div class="aurora-navbar">
            <div class="aurora-brand">
                <div class="aurora-logo-icon">🧠</div>
                <div>
                    <h1 class="aurora-title">AuraScan AI</h1>
                    <p class="aurora-subtitle">Enterprise Brain MRI Diagnostic SaaS</p>
                </div>
            </div>
            <div class="aurora-nav-actions">
                {user_html}
            </div>
        </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)
    
    # Theme toggle button row
    col_l, col_r = st.columns([6, 1])
    with col_r:
        if st.button(f"{theme_icon}", key="global_theme_toggle_btn", use_container_width=True):
            toggle_theme()
            st.rerun()


def render_toast(message: str, toast_type: str = "success") -> None:
    """Renders a floating toast notification bar."""
    icon_map = {
        "success": "✅",
        "warning": "⚠️",
        "danger": "❌",
        "info": "ℹ️"
    }
    icon = icon_map.get(toast_type, "ℹ️")
    toast_html = f"""
        <div class="toast-bar toast-{toast_type}">
            <span style="font-size: 18px;">{icon}</span>
            <span style="font-weight: 600; font-size: 13px;">{message}</span>
        </div>
    """
    st.markdown(toast_html, unsafe_allow_html=True)


def render_skeleton_loader(title: str = "Running Multi-Stage AI Diagnostic Pipeline...") -> None:
    """Renders an animated skeleton placeholder shimmer for AI inference processing."""
    skeleton_html = f"""
        <div class="skeleton-container" role="status" aria-label="Loading diagnostic results">
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 16px;">
                <div class="skeleton-box" style="width: 24px; height: 24px; border-radius: 50%;"></div>
                <div class="skeleton-box skeleton-title"></div>
            </div>
            <p style="font-size: 12px; color: var(--text-muted); margin-bottom: 16px;">{title}</p>
            <div class="skeleton-box skeleton-text"></div>
            <div class="skeleton-box skeleton-text" style="width: 70%;"></div>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-top: 16px;">
                <div class="skeleton-box skeleton-block"></div>
                <div class="skeleton-box skeleton-block"></div>
                <div class="skeleton-box skeleton-block"></div>
            </div>
        </div>
    """
    st.markdown(skeleton_html, unsafe_allow_html=True)


def render_landing_page() -> None:
    """Renders a high-converting SaaS Product Landing Page."""
    from ui_system.components import render_landing_page as _render_lp
    _render_lp()



def render_metric_card(title: str, value: str, border_color: str = "default") -> None:
    """Renders a SaaS styled metric card."""
    style_attr = f"border-color: {border_color};" if border_color != "default" else ""
    st.markdown(f"""
        <div class="metric-card-saas" style="{style_attr}">
            <div class="metric-card-header">
                <span class="metric-card-title">{title}</span>
                <span class="metric-indicator-dot"></span>
            </div>
            <div class="metric-card-value">{value}</div>
        </div>
    """, unsafe_allow_html=True)
