"""
AuraScan AI Enterprise Design System Package
Provides modern 2026 healthcare SaaS design tokens, theme state management, and reusable UI components.
"""

from ui_system.theme import (
    inject_design_system,
    init_theme_state,
    get_css_filepath,
)

from ui_system.components import (
    render_header,
    render_sidebar_user_footer,
    render_toast,
    render_badge,
    render_modal,
    render_empty_state_preset,
    render_empty_state,
    render_alert_card,
    render_skeleton_loader,
    render_metric_card,
    render_user_profile,
    render_password_strength_meter,
    render_auth_nav_header,
    render_login_view,
    render_register_view,
    render_unauthenticated_app,
)

__all__ = [
    "inject_design_system",
    "init_theme_state",
    "get_css_filepath",
    "render_header",
    "render_sidebar_user_footer",
    "render_toast",
    "render_badge",
    "render_modal",
    "render_empty_state_preset",
    "render_empty_state",
    "render_alert_card",
    "render_skeleton_loader",
    "render_metric_card",
    "render_user_profile",
    "render_password_strength_meter",
    "render_auth_nav_header",
    "render_login_view",
    "render_register_view",
    "render_unauthenticated_app",
]
