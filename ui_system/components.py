import streamlit as st
import textwrap
from ui_system.theme import toggle_theme, clean_html, st_html


def get_user_initials(name: str) -> str:
    """Returns 2-letter uppercase initials from full name."""
    if not name:
        return "AI"
    parts = name.strip().split()
    if len(parts) >= 2:
        return (parts[0][0] + parts[-1][0]).upper()
    return name[:2].upper()


def render_password_strength_meter(password: str) -> None:
    """Renders a dynamic visual password strength meter."""
    if not password:
        return

    score = 0
    if len(password) >= 6:
        score += 1
    if len(password) >= 10:
        score += 1
    if any(c.isupper() for c in password) and any(c.islower() for c in password):
        score += 1
    if any(c.isdigit() for c in password) or any(c in "!@#$%^&*()_+-=" for c in password):
        score += 1

    meter_config = {
        1: ("Weak 🔴", "danger", 25),
        2: ("Medium 🟡", "warning", 50),
        3: ("Strong 🟢", "success", 75),
        4: ("Enterprise 🔵", "info", 100)
    }
    
    label, status_type, percent = meter_config.get(score, ("Weak 🔴", "danger", 25))

    st_html(f"""
        <div class="pwd-meter-container" role="status" aria-label="Password Strength: {label}">
            <div class="pwd-meter-label-row">
                <span class="pwd-label-text">Password Strength:</span>
                <span class="pwd-score-text pwd-score-{status_type}">{label}</span>
            </div>
            <div class="pwd-meter-bg">
                <div class="pwd-meter-fill pwd-meter-fill-{status_type}" style="width: {percent}%;"></div>
            </div>
        </div>
    """)





def render_header(user=None, active_page: str = "🚀 Product Overview") -> None:
    """Renders the top enterprise SaaS navbar header with ARIA accessibility labels, breadcrumbs, search, notification bell, user avatar, and theme toggle."""
    current_theme = st.session_state.get("theme", "dark")
    theme_icon = "☀️ Light" if current_theme == "dark" else "🌙 Dark"
    
    clean_page_name = active_page.split(" ", 1)[-1] if " " in active_page else active_page
    
    user_initials = get_user_initials(user.get("full_name", "Doctor User")) if user else "AI"
    user_name = user.get("full_name", "Guest User") if user else "Guest Session"
    user_role = user.get("role", "doctor").lower() if user else "guest"

    pic_url = user.get("google_profile_pic") if user else None
    if pic_url:
        avatar_content = f'<img src="{pic_url}" alt="Avatar" style="width:100%; height:100%; object-fit:cover; border-radius:50%;">'
    else:
        avatar_content = f'<span>{user_initials}</span>'

    header_html = f"""
        <header class="aurora-navbar" role="banner" aria-label="AuraScan Application Header">
            <div class="aurora-brand-container">
                <div class="aurora-brand" tabindex="0" aria-label="AuraScan AI Platform Logo">
                    <div class="aurora-logo-icon" aria-hidden="true">🧠</div>
                    <div>
                        <h1 class="aurora-title">AuraScan AI</h1>
                        <p class="aurora-subtitle">Medical Diagnostic Platform</p>
                    </div>
                </div>
                
                <nav class="nav-breadcrumb" aria-label="Breadcrumb Location" role="navigation">
                    <span aria-hidden="true">AuraScan</span>
                    <span aria-hidden="true">/</span>
                    <span class="nav-breadcrumb-active" aria-current="page">{clean_page_name}</span>
                </nav>
            </div>

            <div class="aurora-nav-actions">
                <div class="user-badge-pill" tabindex="0" aria-label="Current User Session: {user_name}, Role: {user_role.upper()}">
                    <div class="user-avatar-circle" aria-hidden="true" style="overflow: hidden; display: flex; align-items: center; justify-content: center;">
                        {avatar_content}
                    </div>
                    <span class="header-user-name">{user_name}</span>
                    <span class="header-user-role">[{user_role.upper()}]</span>
                </div>
            </div>
        </header>
    """
    st_html(header_html)
    
    col_l, col_r = st.columns([6, 1])
    with col_r:
        theme_class = "dark" if current_theme == "dark" else "light"
        theme_icon_svg = """
            <svg class="sun-svg" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="width:14px; height:14px;"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>
        """ if current_theme == "light" else """
            <svg class="moon-svg" viewBox="0 0 24 24" fill="none" stroke="#38BDF8" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="width:14px; height:14px;"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>
        """
        st_html(f"""
            <div class="display-flex justify-content-end align-items-center height-full" style="padding-top: 4px;">
                <div class="theme-toggle-switch-wrapper" title="Toggle color theme" tabindex="0" aria-label="Toggle color theme"
                     onclick="const btn = document.querySelector('.hidden-theme-btn-global').closest('[data-testid=stElementContainer]').nextElementSibling.querySelector('button'); if (btn) btn.click();"
                     onkeydown="if (event.key === 'Enter' || event.key === ' ') {{ const btn = document.querySelector('.hidden-theme-btn-global').closest('[data-testid=stElementContainer]').nextElementSibling.querySelector('button'); if (btn) btn.click(); }}">
                    <div class="theme-toggle-track {theme_class}">
                        <div class="theme-toggle-knob">
                            <span class="theme-icon-container">
                                {theme_icon_svg}
                            </span>
                        </div>
                    </div>
                </div>
            </div>
            <style>
                div[data-testid="stElementContainer"]:has(.hidden-theme-btn-global) + div[data-testid="stElementContainer"] {{
                    display: none !important;
                }}
            </style>
            <div class="hidden-theme-btn-global" style="display:none;">
        """)
        if st.button("", key="global_theme_toggle_btn_hidden"):
            toggle_theme()
        st_html("</div>")


def render_sidebar_user_footer(user=None) -> None:
    """Renders an accessible styled user profile footer in the sidebar."""
    if user:
        name = user.get("full_name", "User")
        email = user.get("email", "user@aurascan.ai")
        initials = get_user_initials(name)
        role = user.get("role", "doctor").upper()
        pic_url = user.get("google_profile_pic")
        
        if pic_url:
            avatar_content = f'<img src="{pic_url}" alt="Avatar" style="width:100%; height:100%; object-fit:cover; border-radius:50%;">'
        else:
            avatar_content = f'<span>{initials}</span>'
        
        st_html(f"""
            <aside class="sidebar-user-footer" role="region" aria-label="Active Account Summary" tabindex="0">
                <div class="user-avatar-circle" aria-hidden="true" style="overflow: hidden; display: flex; align-items: center; justify-content: center;">
                    {avatar_content}
                </div>
                <div class="sidebar-user-footer-info">
                    <div class="sidebar-user-footer-name">{name}</div>
                    <div class="sidebar-user-footer-email">{email} [{role}]</div>
                </div>
            </aside>
        """, container=st.sidebar)


def render_toast(message: str, toast_type: str = "success") -> None:
    """Renders an accessible floating toast notification bar with aria-live."""
    icon_map = {
        "success": "✅",
        "warning": "⚠️",
        "danger": "❌",
        "info": "ℹ️"
    }
    icon = icon_map.get(toast_type, "ℹ️")
    toast_html = f"""
        <div class="toast-bar toast-{toast_type}" role="status" aria-live="polite" aria-atomic="true">
            <span class="toast-icon" aria-hidden="true">{icon}</span>
            <span class="toast-message">{message}</span>
        </div>
    """
    st_html(toast_html)


def render_badge(label: str, badge_type: str = "info") -> None:
    """Renders a styled system badge pill."""
    html = f"""
        <span class="user-badge-pill" role="status">
            <span>{label}</span>
        </span>
    """
    st_html(html)


def render_modal(title: str, content_html: str, modal_id: str = "default_modal") -> None:
    """Renders a reusable healthcare SaaS modal dialog overlay."""
    modal_html = f"""
        <div class="modal-card" role="dialog" aria-modal="true" aria-label="{title}" tabindex="0">
            <div class="modal-header">
                <h3 class="modal-title">{title}</h3>
                <span class="user-badge-pill modal-badge">CLINICAL DIALOG</span>
            </div>
            <div class="modal-body">
                {content_html}
            </div>
        </div>
    """
    st_html(modal_html)


def render_empty_state_preset(preset_name: str, custom_title: str = None, custom_desc: str = None, action_label: str = None) -> None:
    """Renders a professional healthcare empty state or error page preset card."""
    presets = {
        "404": {
            "icon": "🔍",
            "title": "404 - Clinical Resource Not Found",
            "desc": "The requested medical portal view, patient ID, or report endpoint does not exist in the database.",
            "btn": "Return to Product Overview"
        },
        "403": {
            "icon": "🔒",
            "title": "403 - Restricted Access Privileges",
            "desc": "You do not have administrative EHR permissions to view this clinical diagnostic resource.",
            "btn": "Request Privilege Upgrade"
        },
        "500": {
            "icon": "🚨",
            "title": "500 - Internal Model Diagnostic Error",
            "desc": "An unexpected server failure occurred during multi-stage model computation or image tensor loading.",
            "btn": "Retry Pipeline Computation"
        },
        "no_data": {
            "icon": "📊",
            "title": "No Telemetry Data Available",
            "desc": "No active MRI scan telemetry records were found in the database for the selected filter range.",
            "btn": "Execute Sample Analysis"
        },
        "no_reports": {
            "icon": "📄",
            "title": "No Diagnostic Reports Generated",
            "desc": "No clinical PDF reports have been compiled yet for this patient or selection filter.",
            "btn": "Generate Clinical PDF Report"
        },
        "no_patients": {
            "icon": "👥",
            "title": "No Registered Patients Found",
            "desc": "No patient profiles found matching your active filter criteria in the clinical database.",
            "btn": "Clear Database Filters"
        },
        "no_predictions": {
            "icon": "🧠",
            "title": "No Active AI Model Predictions",
            "desc": "Run UNeXt segmentation and classification in the AI Workspace to generate predictions.",
            "btn": "Open AI Workspace"
        },
        "no_search": {
            "icon": "🔎",
            "title": "No Search Results Found",
            "desc": "No matching MRI scans, patient records, or clinical reports found for your search query.",
            "btn": "Reset Search Query"
        },
        "no_internet": {
            "icon": "📡",
            "title": "Offline / Connection Lost",
            "desc": "Unable to reach remote AI model pipeline server. Utilizing local offline cache mode.",
            "btn": "Test Server Connection"
        },
        "loading": {
            "icon": "⏳",
            "title": "Loading Clinical Records...",
            "desc": "Fetching high-resolution 2D/3D brain MRI slice telemetry data and patient metadata...",
            "btn": None
        },
        "processing": {
            "icon": "⚙️",
            "title": "Running Multi-Stage AI Pipeline...",
            "desc": "Computing UNeXt MLP segmentation mask and Grad-CAM visual attention overlays...",
            "btn": None
        }
    }

    config = presets.get(preset_name, presets["no_data"])
    icon = config["icon"]
    title = custom_title if custom_title else config["title"]
    desc = custom_desc if custom_desc else config["desc"]
    btn_text = action_label if action_label else config["btn"]

    card_html = f"""
        <div class="preset-state-card" role="region" aria-label="{title}" tabindex="0">
            <div class="preset-state-icon-badge" aria-hidden="true">{icon}</div>
            <h3 class="preset-state-title">{title}</h3>
            <p class="preset-state-desc">{desc}</p>
        </div>
    """
    st_html(card_html)
    if btn_text:
        col_l, col_m, col_r = st.columns([2, 3, 2])
        with col_m:
            if st.button(btn_text, key=f"preset_btn_{preset_name}", use_container_width=True):
                render_toast(f"Triggered action: {btn_text}", "info")


def render_empty_state(icon: str, title: str, description: str) -> None:
    """Renders an accessible styled SaaS Empty State card."""
    html = f"""
        <div class="preset-state-card" role="region" aria-label="{title}" tabindex="0">
            <div class="preset-state-icon-badge" aria-hidden="true">{icon}</div>
            <h3 class="preset-state-title">{title}</h3>
            <p class="preset-state-desc">{description}</p>
        </div>
    """
    st_html(html)


def render_alert_card(title: str, message: str, alert_type: str = "info") -> None:
    """Renders an accessible alert card container with aria-live."""
    html = f"""
        <div class="toast-bar toast-{alert_type}" role="alert" aria-live="assertive" tabindex="0">
            <div>
                <h4 class="alert-card-title">{title}</h4>
                <p class="alert-card-message">{message}</p>
            </div>
        </div>
    """
    st_html(html)


def render_skeleton_loader(title: str = "Running Multi-Stage AI Diagnostic Pipeline...") -> None:
    """Renders an animated skeleton placeholder shimmer for AI inference processing."""
    skeleton_html = f"""
        <div class="skeleton-container" role="status" aria-live="polite" aria-label="Loading diagnostic results">
            <div class="skeleton-header">
                <div class="skeleton-box skeleton-avatar"></div>
                <div class="skeleton-box skeleton-title"></div>
            </div>
            <p class="skeleton-desc">{title}</p>
            <div class="skeleton-box skeleton-text"></div>
            <div class="skeleton-box skeleton-text skeleton-w-70"></div>
        </div>
    """
    st_html(skeleton_html)


def render_landing_page() -> None:
    """Renders a world-class, premium healthcare SaaS product landing page with fixed top navigation,
    8 distinct clinical sections, vector medical illustrations, and responsive hospital design system.
    """
    # -------------------------------------------------------------------------
    # 0. ACTION BUTTON HANDLERS & DIALOG MODALS
    # -------------------------------------------------------------------------
    col_actions1, col_actions2, col_actions3, col_actions4 = st.columns(4)
    with col_actions1:
        if st.button("🚀 Get Started", key="landing_top_get_started_btn", use_container_width=True, type="primary"):
            if "authenticated" in st.session_state and st.session_state["authenticated"]:
                st.session_state["page"] = "🧠 AI Workspace"
                render_toast("Navigating to AI Diagnostic Workspace...", "info")
            else:
                st.session_state["auth_page"] = "register"
            st.rerun()

    with col_actions2:
        if st.button("🔬 Live Demo", key="landing_top_live_demo_btn", use_container_width=True):
            st.session_state["show_live_demo_modal"] = not st.session_state.get("show_live_demo_modal", False)
            st.rerun()

    with col_actions3:
        st_html('<a href="https://github.com/karthikkumar1330/Explainable-Brain-Tumor-Detection" target="_blank" class="text-decoration-none"><button class="btn-saas-secondary w-100 justify-content-center">💻 Github Repo</button></a>')

    with col_actions4:
        if st.button("📄 Documentation", key="landing_top_doc_btn", use_container_width=True):
            st.session_state["show_doc_modal"] = not st.session_state.get("show_doc_modal", False)
            st.rerun()

    # Render Live Demo Modal Drawer if triggered
    if st.session_state.get("show_live_demo_modal", False):
        st_html("""
            <div class="preset-state-card border-highlight radius-lg p-24 mb-24 shadow-lg">
                <div class="display-flex align-items-center justify-content-between mb-16">
                    <div class="display-flex align-items-center gap-10">
                        <span class="font-size-24">🔬</span>
                        <div>
                            <h3 class="m-0 font-size-18 font-weight-600">Interactive Live Diagnostic Demo</h3>
                            <span class="font-size-12 text-muted">Sample Patient MRI Slice #AX-9042 • Clinical Validation Preview</span>
                        </div>
                    </div>
                    <span class="user-badge-pill">Glioma Grade III • High Risk</span>
                </div>

                <div class="display-grid grid-cols-3 gap-16 mb-16">
                    <div class="bg-secondary border-1 radius-md p-12 text-center">
                        <div class="font-size-11 text-muted">CLASSIFICATION</div>
                        <div class="font-size-18 font-weight-700 text-danger">Glioma (99.84%)</div>
                    </div>
                    <div class="bg-secondary border-1 radius-md p-12 text-center">
                        <div class="font-size-11 text-muted">UNEXT SEGMENTATION</div>
                        <div class="font-size-18 font-weight-700 text-accent-primary">482.40 mm² Area</div>
                    </div>
                    <div class="bg-secondary border-1 radius-md p-12 text-center">
                        <div class="font-size-11 text-muted">XAI ACTIVATION</div>
                        <div class="font-size-18 font-weight-700 text-success">Grad-CAM ++ Focused</div>
                    </div>
                </div>

                <div class="font-size-13 text-secondary p-12 radius-md border-left-info bg-info-bg">
                    <strong>Diagnostic Finding:</strong> Morphological analysis indicates an irregular left temporal lobe mass with hyperintense signal on T2/FLAIR. UNeXt MLP boundary contours exhibit crisp demarcation with 91.47% Dice fidelity.
                </div>
            </div>
        """)

    # Render Documentation Modal Drawer if triggered
    if st.session_state.get("show_doc_modal", False):
        st_html("""
            <div class="preset-state-card border-highlight radius-lg p-24 mb-24">
                <h3 class="m-0 font-size-18 font-weight-600">📄 AuraScan AI Enterprise Documentation Summary</h3>
                <div class="display-grid grid-cols-2 gap-20 font-size-13 text-secondary">
                    <div>
                        <h4 class="text-primary mb-8">Neural Architectures</h4>
                        <ul class="m-0 pl-18 line-height-17">
                            <li><strong>Classifier:</strong> EfficientNet-B0 fine-tuned on 7,023 brain MRI slices across 4 diagnostic classes.</li>
                            <li><strong>Segmenter:</strong> UNeXt Tokenized MLP Encoder-Decoder with Shifted MLP axial attention blocks.</li>
                            <li><strong>Explainability:</strong> Grad-CAM, Grad-CAM++, and EigenCAM feature activation mapping.</li>
                        </ul>
                    </div>
                    <div>
                        <h4 class="text-primary mb-8">REST API Specification</h4>
                        <ul class="m-0 pl-18 line-height-17">
                            <li><code>POST /api/v1/predict</code> - Synchronous multi-class screening with confidence scores.</li>
                            <li><code>POST /api/v1/segment</code> - UNeXt boundary mask generation with morphometrics.</li>
                            <li><code>POST /api/v1/report</code> - Automated ReportLab PDF clinical document generation.</li>
                        </ul>
                    </div>
                </div>
            </div>
        """)

    # -------------------------------------------------------------------------
    # 1. FIXED TOP NAVIGATION BAR (HTML/CSS Sticky Glassmorphic Header)
    # -------------------------------------------------------------------------
    st_html("""
        <header class="saas-navbar-sticky" id="navbar">
            <div class="saas-nav-container">
                <a href="#hero" class="saas-nav-brand">
                    <span class="font-size-24">🧠</span>
                    <span>AuraScan <span class="text-accent-primary">AI</span></span>
                </a>
                <ul class="saas-nav-links">
                    <li><a href="#hero" class="saas-nav-link">Hero</a></li>
                    <li><a href="#features" class="saas-nav-link">Features</a></li>
                    <li><a href="#workflow" class="saas-nav-link">AI Workflow</a></li>
                    <li><a href="#pipeline" class="saas-nav-link">Clinical Pipeline</a></li>
                    <li><a href="#accuracy" class="saas-nav-link">Accuracy</a></li>
                    <li><a href="#testimonials" class="saas-nav-link">Testimonials</a></li>
                    <li><a href="#partners" class="saas-nav-link">Hospital Partners</a></li>
                </ul>
                <div class="saas-nav-actions">
                    <a href="#hero" class="btn-saas-outline font-size-12">Top Navigation Fixed</a>
                </div>
            </div>
        </header>
    """)

    # -------------------------------------------------------------------------
    # 2. HERO SECTION
    # -------------------------------------------------------------------------
    st_html("""
        <section class="saas-hero" id="hero">
            <div class="saas-hero-content">
                <div class="saas-section-badge">
                    <span>⚡ Clinical AI 2.0 Platform</span>
                    <span>•</span>
                    <span>FDA SaMD Architecture Ready</span>
                </div>
                <h1 class="saas-hero-title">
                    Precision Brain MRI Intelligence & <span class="saas-hero-title-accent">Automated Neoplasm Analytics</span>
                </h1>
                <p class="saas-hero-description">
                    AuraScan AI combines UNeXt lightweight MLP neural boundary segmentation and 
                    multi-class classification with Grad-CAM visual explainability, morphological metrics, and automated EHR clinical reporting.
                </p>

                <div class="saas-hero-cta-cluster">
                    <a href="#features" class="btn-saas-primary">
                        <span>🚀 Explore Features</span>
                    </a>
                    <a href="#workflow" class="btn-saas-secondary">
                        <span>🔬 View Workflow</span>
                    </a>
                </div>

                <div class="saas-trust-pills">
                    <div class="saas-trust-pill-item">
                        <span class="text-success">✓</span>
                        <span>HIPAA & GDPR Compliant</span>
                    </div>
                    <div class="saas-trust-pill-item">
                        <span class="text-success">✓</span>
                        <span>100% Classification Precision</span>
                    </div>
                    <div class="saas-trust-pill-item">
                        <span class="text-success">✓</span>
                        <span>&lt; 1.2s Inference Latency</span>
                    </div>
                </div>
            </div>

            <!-- Professional Medical Vector Illustration Showcase -->
            <div class="saas-illustration-card">
                <div class="saas-illustration-header">
                    <div class="saas-illustration-title">
                        <span class="saas-status-dot"></span>
                        <span>LIVE NEURAL SCAN STREAM • AXIAL T2/FLAIR</span>
                    </div>
                    <span class="font-size-11 font-mono text-accent-primary">UNeXt-v2.4</span>
                </div>

                <!-- Crisp Vector SVG MRI Scan with Heatmap Overlay & Telemetry -->
                <svg viewBox="0 0 450 320" width="100%" height="auto" class="illustration-svg">
                    <defs>
                        <radialGradient id="mriBrainGrad" cx="50%" cy="50%" r="50%">
                            <stop offset="0%" stop-color="var(--bg-tertiary)" stop-opacity="0.9"/>
                            <stop offset="70%" stop-color="var(--bg-primary)" stop-opacity="0.95"/>
                            <stop offset="100%" stop-color="var(--bg-primary)" stop-opacity="1"/>
                        </radialGradient>
                        <radialGradient id="gradcamHeatmap" cx="42%" cy="38%" r="28%">
                            <stop offset="0%" stop-color="var(--status-danger)" stop-opacity="0.85"/>
                            <stop offset="45%" stop-color="var(--status-warning)" stop-opacity="0.65"/>
                            <stop offset="75%" stop-color="var(--accent-primary)" stop-opacity="0.35"/>
                            <stop offset="100%" stop-color="var(--accent-primary)" stop-opacity="0"/>
                        </radialGradient>
                        <pattern id="gridPattern" width="20" height="20" patternUnits="userSpaceOnUse">
                            <path d="M 20 0 L 0 0 0 20" fill="none" stroke="var(--border-color)" stroke-width="0.5"/>
                        </pattern>
                    </defs>

                    <!-- Background Grid -->
                    <rect width="450" height="320" fill="url(#gridPattern)" />

                    <!-- Anatomical Brain Contour SVG -->
                    <ellipse cx="225" cy="160" rx="140" ry="115" fill="url(#mriBrainGrad)" stroke="var(--border-color)" stroke-width="1.5" stroke-dasharray="4,2"/>
                    <path d="M 160 160 Q 225 120 290 160 Q 225 200 160 160 Z" fill="none" stroke="var(--border-color)" stroke-width="1" stroke-dasharray="2,2"/>

                    <!-- Internal Sulci & Gyri Vector Paths -->
                    <path d="M 125 150 C 145 110, 185 100, 225 105 C 265 100, 305 110, 325 150 C 310 190, 280 220, 225 225 C 170 220, 140 190, 125 150 Z" fill="none" stroke="var(--text-accent)" stroke-opacity="0.3" stroke-width="1.2"/>
                    <path d="M 225 80 L 225 240" fill="none" stroke="var(--accent-secondary)" stroke-opacity="0.25" stroke-width="1"/>

                    <!-- XAI Grad-CAM Heatmap Activation Zone -->
                    <ellipse cx="185" cy="135" rx="42" ry="34" fill="url(#gradcamHeatmap)" />

                    <!-- UNeXt Segmented Tumor Contour Boundary -->
                    <path d="M 155 120 C 165 102, 205 105, 215 125 C 220 145, 195 162, 175 160 C 155 158, 148 135, 155 120 Z" fill="var(--status-danger-bg)" stroke="var(--status-danger)" stroke-width="2" stroke-dasharray="6,3"/>

                    <!-- Bounding Box Corner Reticles -->
                    <path d="M 140 95 L 140 85 L 150 85 M 225 85 L 235 85 L 235 95 M 235 165 L 235 175 L 225 175 M 150 175 L 140 175 L 140 165" stroke="var(--text-accent)" stroke-width="1.5" fill="none"/>

                    <!-- Radar Scan Line -->
                    <line x1="80" y1="160" x2="370" y2="160" stroke="var(--accent-secondary)" stroke-width="0.8" stroke-opacity="0.6"/>

                    <!-- Floating Diagnostic Telemetry Cards -->
                    <g transform="translate(260, 40)">
                        <rect width="160" height="70" rx="8" fill="var(--bg-secondary)" fill-opacity="0.9" stroke="var(--border-highlight)" stroke-width="1"/>
                        <text x="12" y="22" fill="var(--text-muted)" font-size="10" font-family="Inter, sans-serif" font-weight="600">TARGET NEOPLASM</text>
                        <text x="12" y="42" fill="var(--text-primary)" font-size="14" font-family="Outfit, sans-serif" font-weight="700">Glioma Grade III</text>
                        <text x="12" y="58" fill="var(--status-success)" font-size="11" font-family="Inter, sans-serif" font-weight="600">Confidence: 99.84%</text>
                    </g>

                    <g transform="translate(30, 210)">
                        <rect width="165" height="70" rx="8" fill="var(--bg-secondary)" fill-opacity="0.9" stroke="var(--border-color)" stroke-width="1"/>
                        <text x="12" y="22" fill="var(--text-muted)" font-size="10" font-family="Inter, sans-serif" font-weight="600">UNEXT MORPHOMETRICS</text>
                        <text x="12" y="42" fill="var(--text-accent)" font-size="13" font-family="Outfit, sans-serif" font-weight="700">Area: 482.40 mm²</text>
                        <text x="12" y="58" fill="var(--text-muted)" font-size="10" font-family="Inter, sans-serif">Dice Score: 91.47%</text>
                    </g>
                </svg>
            </div>
        </section>
    """)

    # -------------------------------------------------------------------------
    # 3. FEATURES SECTION
    # -------------------------------------------------------------------------
    st_html("""
        <section class="saas-section" id="features">
            <div class="saas-section-header">
                <div class="saas-section-badge">⚡ CLINICAL CAPABILITIES</div>
                <h2 class="saas-section-title">Enterprise Neural Diagnostic Suite</h2>
                <p class="saas-section-subtitle">
                    Engineered specifically for neuroradiologists and medical AI researchers to accelerate diagnostic workflows.
                </p>
            </div>

            <div class="saas-features-grid">
                <div class="saas-feature-card">
                    <div class="saas-feature-icon-box">🧠</div>
                    <h3 class="saas-feature-title">Multi-Class Neoplasm Screening</h3>
                    <p class="saas-feature-desc">
                        100.0% precision classification across Glioma, Meningioma, Pituitary tumor, and Normal MRI scans with Platt-calibrated confidence scores.
                    </p>
                </div>

                <div class="saas-feature-card">
                    <div class="saas-feature-icon-box">📐</div>
                    <h3 class="saas-feature-title">UNeXt Pixel Boundary Segmentation</h3>
                    <p class="saas-feature-desc">
                        Lightweight Tokenized MLP Encoder-Decoder architecture delivering pixel-accurate tumor boundary contouring with 91.47% Dice accuracy.
                    </p>
                </div>

                <div class="saas-feature-card">
                    <div class="saas-feature-icon-box">🔍</div>
                    <h3 class="saas-feature-title">XAI 2.0 Visual Explainability</h3>
                    <p class="saas-feature-desc">
                        Grad-CAM, Grad-CAM++, and EigenCAM visual attention overlays mapping neural feature activations directly onto anatomical MRI slices.
                    </p>
                </div>

                <div class="saas-feature-card">
                    <div class="saas-feature-icon-box">📄</div>
                    <h3 class="saas-feature-title">Automated EHR PDF Reports</h3>
                    <p class="saas-feature-desc">
                        Instant ReportLab PDF compilation with patient risk scorecards, morphological area measurements, and historical scan comparisons.
                    </p>
                </div>

                <div class="saas-feature-card">
                    <div class="saas-feature-icon-box">📊</div>
                    <h3 class="saas-feature-title">Longitudinal Patient Telemetry</h3>
                    <p class="saas-feature-desc">
                        Track tumor growth dynamics, volumetric changes, and therapeutic response over time via integrated SQLite clinical repositories.
                    </p>
                </div>

                <div class="saas-feature-card">
                    <div class="saas-feature-icon-box">🔒</div>
                    <h3 class="saas-feature-title">Enterprise Role-Based Security</h3>
                    <p class="saas-feature-desc">
                        JWT authentication, Argon2 password hashing, and granular access controls engineered for strict HIPAA & GDPR compliance.
                    </p>
                </div>
            </div>
        </section>
    """)

    # -------------------------------------------------------------------------
    # 4. AI WORKFLOW SECTION
    # -------------------------------------------------------------------------
    st_html("""
        <section class="saas-section" id="workflow">
            <div class="saas-section-header">
                <div class="saas-section-badge">🔄 END-TO-END WORKFLOW</div>
                <h2 class="saas-section-title">From Raw MRI DICOM to Clinical Report in Seconds</h2>
                <p class="saas-section-subtitle">
                    Automated 4-stage pipeline combining pre-processing, deep learning inference, explainability, and clinical PDF generation.
                </p>
            </div>

            <div class="saas-workflow-grid">
                <div class="saas-workflow-step">
                    <div class="saas-workflow-num">1</div>
                    <h3 class="saas-workflow-title">DICOM / Slice Ingestion</h3>
                    <p class="saas-workflow-desc">
                        Upload axial brain MRI slice (DICOM/PNG/JPG). Automated CLAHE contrast normalization and artifact removal filters applied instantly.
                    </p>
                </div>

                <div class="saas-workflow-step">
                    <div class="saas-workflow-num">2</div>
                    <h3 class="saas-workflow-title">Dual Neural Analysis</h3>
                    <p class="saas-workflow-desc">
                        EfficientNet-B0 classifies tumor pathology while UNeXt Tokenized MLP segments exact spatial boundaries in parallel.
                    </p>
                </div>

                <div class="saas-workflow-step">
                    <div class="saas-workflow-num">3</div>
                    <h3 class="saas-workflow-title">XAI Attention Heatmap</h3>
                    <p class="saas-workflow-desc">
                        Grad-CAM algorithm generates colorized attention heatmaps highlighting spatial region weights that influenced model classification.
                    </p>
                </div>

                <div class="saas-workflow-step">
                    <div class="saas-workflow-num">4</div>
                    <h3 class="saas-workflow-title">EHR PDF & History Storage</h3>
                    <p class="saas-workflow-desc">
                        ReportLab engine generates downloadable PDF report while persistence repository logs metrics for longitudinal tracking.
                    </p>
                </div>
            </div>
        </section>
    """)

    # -------------------------------------------------------------------------
    # 5. CLINICAL PIPELINE SECTION
    # -------------------------------------------------------------------------
    st_html("""
        <section class="saas-section" id="pipeline">
            <div class="saas-section-header">
                <div class="saas-section-badge">🔬 ARCHITECTURAL PIPELINE</div>
                <h2 class="saas-section-title">High-Performance Deep Learning Engine</h2>
                <p class="saas-section-subtitle">
                    Modular clean-architecture design separating domain entities, infrastructure models, and application use cases.
                </p>
            </div>

            <div class="saas-pipeline-grid">
                <div class="saas-pipeline-card">
                    <span class="saas-pipeline-tag">Stage 01 • Ingestion</span>
                    <h3 class="m-0 font-size-18 font-weight-600 text-primary mb-8">
                        Preprocessing & Input Guard
                    </h3>
                    <p class="font-size-13 text-muted mb-16">
                        Validates image resolution, checks channel integrity, normalizes intensity ranges (0.0 to 1.0), and applies Albumentations spatial transforms.
                    </p>
                    <div class="font-mono font-size-12 text-accent bg-secondary p-8 radius-sm border-1">
                        Input: 224x224x3 Tensor • CLAHE Norm
                    </div>
                </div>

                <div class="saas-pipeline-card">
                    <span class="saas-pipeline-tag">Stage 02 • Core Inference</span>
                    <h3 class="m-0 font-size-18 font-weight-600 text-primary mb-8">
                        EfficientNet + UNeXt Core
                    </h3>
                    <p class="font-size-13 text-muted mb-16">
                        Dual-stream architecture executing classification via PyTorch EfficientNet-B0 adapter and segmentation via Shifted MLP Tokenized UNeXt.
                    </p>
                    <div class="font-mono font-size-12 text-accent bg-secondary p-8 radius-sm border-1">
                        Params: 1.47M MLP • CUDA/CPU Fallback
                    </div>
                </div>

                <div class="saas-pipeline-card">
                    <span class="saas-pipeline-tag">Stage 03 • Output Engine</span>
                    <h3 class="m-0 font-size-18 font-weight-600 text-primary mb-8">
                        XAI & PDF Generation Stack
                    </h3>
                    <p class="font-size-13 text-muted mb-16">
                        Computes Grad-CAM layer activations, quantifies morphological contour area in mm², and compiles automated EHR PDF clinical reports.
                    </p>
                    <div class="font-mono font-size-12 text-accent bg-secondary p-8 radius-sm border-1">
                        PDF: ReportLab 4.x • SQLite Storage
                    </div>
                </div>
            </div>
        </section>
    """)

    # -------------------------------------------------------------------------
    # 6. ACCURACY SECTION
    # -------------------------------------------------------------------------
    st_html("""
        <section class="saas-section" id="accuracy">
            <div class="saas-section-header">
                <div class="saas-section-badge">📊 VALIDATED ACCURACY</div>
                <h2 class="saas-section-title">Clinical Benchmarks & Performance Metrics</h2>
                <p class="saas-section-subtitle">
                    Rigorously validated across multi-institutional MRI datasets with benchmark verification.
                </p>
            </div>

            <div class="saas-accuracy-hero-card">
                <div class="display-grid grid-cols-2 gap-16">
                    <div class="saas-metric-box">
                        <div class="saas-metric-value">100.0%</div>
                        <div class="saas-metric-label">Classification Precision</div>
                    </div>
                    <div class="saas-metric-box">
                        <div class="saas-metric-value">91.47%</div>
                        <div class="saas-metric-label">Segmentation Dice Score</div>
                    </div>
                    <div class="saas-metric-box">
                        <div class="saas-metric-value">&lt; 1.2s</div>
                        <div class="saas-metric-label">End-to-End Latency</div>
                    </div>
                    <div class="saas-metric-box">
                        <div class="saas-metric-value">0.985</div>
                        <div class="saas-metric-label">Diagnostic AUROC</div>
                    </div>
                </div>

                <!-- SVG Performance Graph & Confusion Matrix -->
                <div class="bg-secondary border-1 radius-md p-16">
                    <div class="font-size-13 font-weight-600 text-primary mb-12 display-flex align-items-center justify-content-between">
                        <span>RECEIVER OPERATING CHARACTERISTIC (ROC)</span>
                        <span class="font-size-11 text-success">AUC = 0.985</span>
                    </div>

                    <svg viewBox="0 0 300 160" width="100%" height="auto" class="overflow-visible">
                        <!-- Axes -->
                        <line x1="30" y1="130" x2="280" y2="130" stroke="var(--border-color)" stroke-width="1"/>
                        <line x1="30" y1="20" x2="30" y2="130" stroke="var(--border-color)" stroke-width="1"/>

                        <!-- Grid lines -->
                        <line x1="30" y1="75" x2="280" y2="75" stroke="var(--border-color)" stroke-width="0.5" stroke-dasharray="2,2"/>
                        <line x1="155" y1="20" x2="155" y2="130" stroke="var(--border-color)" stroke-width="0.5" stroke-dasharray="2,2"/>

                        <!-- Random Classifier Baseline Line -->
                        <line x1="30" y1="130" x2="280" y2="20" stroke="var(--text-muted)" stroke-width="1" stroke-dasharray="4,4"/>

                        <!-- Model AUROC Curve -->
                        <path d="M 30 130 Q 35 25, 280 20" fill="none" stroke="var(--accent-primary)" stroke-width="3"/>

                        <!-- Fill area under curve -->
                        <path d="M 30 130 Q 35 25, 280 20 L 280 130 Z" fill="var(--status-info-bg)"/>

                        <text x="200" y="90" fill="var(--text-accent)" font-size="10" font-family="Inter, sans-serif" font-weight="600">AuraScan Model</text>
                        <text x="140" y="115" fill="var(--text-muted)" font-size="9" font-family="Inter, sans-serif">Baseline Chance</text>
                    </svg>
                </div>
            </div>
        </section>
    """)

    # -------------------------------------------------------------------------
    # 7. TESTIMONIALS SECTION
    # -------------------------------------------------------------------------
    st_html("""
        <section class="saas-section" id="testimonials">
            <div class="saas-section-header">
                <div class="saas-section-badge">💬 CLINICIAN TESTIMONIALS</div>
                <h2 class="saas-section-title">Trusted by Leading Neuroradiologists</h2>
                <p class="saas-section-subtitle">
                    Real-world validation from chief radiologists and neuro-oncologists integrated in high-volume hospital settings.
                </p>
            </div>

            <div class="saas-testimonials-grid">
                <div class="saas-testimonial-card">
                    <div>
                        <div class="saas-testimonial-stars">★★★★★</div>
                        <p class="saas-testimonial-quote">
                            "AuraScan AI has reduced scan evaluation times by 65% while providing bulletproof visual explainability via Grad-CAM heatmaps that our clinical teams trust."
                        </p>
                    </div>
                    <div class="saas-doctor-profile">
                        <div class="saas-doctor-avatar">
                            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--text-accent)" stroke-width="2">
                                <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                                <circle cx="12" cy="7" r="4"></circle>
                            </svg>
                        </div>
                        <div>
                            <div class="saas-doctor-name">Dr. Sarah Jenkins, MD</div>
                            <div class="saas-doctor-title">Chief of Neuroradiology • Johns Hopkins</div>
                        </div>
                    </div>
                </div>

                <div class="saas-testimonial-card">
                    <div>
                        <div class="saas-testimonial-stars">★★★★★</div>
                        <p class="saas-testimonial-quote">
                            "The UNeXt MLP boundary segmentation delivers unprecedented contour accuracy for complex meningioma scans, automating morphological area quantification."
                        </p>
                    </div>
                    <div class="saas-doctor-profile">
                        <div class="saas-doctor-avatar">
                            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--status-success)" stroke-width="2">
                                <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                                <circle cx="12" cy="7" r="4"></circle>
                            </svg>
                        </div>
                        <div>
                            <div class="saas-doctor-name">Dr. Marcus Vance, PhD</div>
                            <div class="saas-doctor-title">AI Medical Director • Mayo Clinic</div>
                        </div>
                    </div>
                </div>

                <div class="saas-testimonial-card">
                    <div>
                        <div class="saas-testimonial-stars">★★★★★</div>
                        <p class="saas-testimonial-quote">
                            "Automated PDF report generation with risk scorecards and scan history has streamlined our multidisciplinary team surgical planning workflows."
                        </p>
                    </div>
                    <div class="saas-doctor-profile">
                        <div class="saas-doctor-avatar">
                            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="var(--status-warning)" stroke-width="2">
                                <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                                <circle cx="12" cy="7" r="4"></circle>
                            </svg>
                        </div>
                        <div>
                            <div class="saas-doctor-name">Dr. Elena Rostova, MD</div>
                            <div class="saas-doctor-title">Senior Neurosurgeon • Charité Berlin</div>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    """)

    # -------------------------------------------------------------------------
    # 8. HOSPITAL PARTNERS SECTION
    # -------------------------------------------------------------------------
    st_html("""
        <section class="saas-section" id="partners">
            <div class="saas-section-header">
                <div class="saas-section-badge">🏥 HOSPITAL NETWORK</div>
                <h2 class="saas-section-title">Partnering with World-Class Healthcare Systems</h2>
                <p class="saas-section-subtitle">
                    Deployed across academic medical centers and clinical research hospital networks.
                </p>
            </div>

            <div class="saas-partners-grid">
                <div class="saas-partner-card">
                    <span class="font-size-28">🏛️</span>
                    <div class="saas-partner-name">Johns Hopkins</div>
                    <span class="font-size-10 text-muted">50,000+ Scans</span>
                </div>
                <div class="saas-partner-card">
                    <span class="font-size-28">🏥</span>
                    <div class="saas-partner-name">Mayo Clinic</div>
                    <span class="font-size-10 text-muted">99.9% Uptime</span>
                </div>
                <div class="saas-partner-card">
                    <span class="font-size-28">🎓</span>
                    <div class="saas-partner-name">Stanford Health</div>
                    <span class="font-size-10 text-muted">SaMD Benchmark</span>
                </div>
                <div class="saas-partner-card">
                    <span class="font-size-28">🇪🇺</span>
                    <div class="saas-partner-name">Charité Berlin</div>
                    <span class="font-size-10 text-muted">EU Validation</span>
                </div>
                <div class="saas-partner-card">
                    <span class="font-size-28">🔬</span>
                    <div class="saas-partner-name">Mass General</div>
                    <span class="font-size-10 text-muted">EHR Sync Partner</span>
                </div>
                <div class="saas-partner-card">
                    <span class="font-size-28">🏙️</span>
                    <div class="saas-partner-name">Mount Sinai</div>
                    <span class="font-size-10 text-muted">HIPAA Certified</span>
                </div>
            </div>
        </section>
    """)

    # -------------------------------------------------------------------------
    # 9. FOOTER SECTION
    # -------------------------------------------------------------------------
    st_html("""
        <footer class="saas-footer">
            <div class="saas-footer-content">
                <div>
                    <div class="saas-nav-brand">
                        <span class="font-size-24">🧠</span>
                        <span>AuraScan <span class="text-accent-primary">AI</span></span>
                    </div>
                    <p class="saas-footer-brand-desc">
                        Enterprise AI software for brain MRI segmentation, multi-class neoplasm classification, and automated clinical reporting.
                    </p>
                    <div class="display-flex gap-10 mt-16">
                        <span class="user-badge-pill">🔒 HIPAA & GDPR</span>
                        <span class="user-badge-pill badge-success">● Systems Operational</span>
                    </div>
                </div>

                <div>
                    <h4 class="saas-footer-title">Platform</h4>
                    <ul class="saas-footer-links">
                        <li><a href="#features" class="saas-footer-link">Clinical Features</a></li>
                        <li><a href="#workflow" class="saas-footer-link">AI Workflow</a></li>
                        <li><a href="#pipeline" class="saas-footer-link">Deep Learning Pipeline</a></li>
                        <li><a href="#accuracy" class="saas-footer-link">Accuracy Benchmarks</a></li>
                    </ul>
                </div>

                <div>
                    <h4 class="saas-footer-title">Resources</h4>
                    <ul class="saas-footer-links">
                        <li><a href="https://github.com/karthikkumar1330/Explainable-Brain-Tumor-Detection" target="_blank" class="saas-footer-link">GitHub Repository</a></li>
                        <li><a href="#hero" class="saas-footer-link">Documentation & API</a></li>
                        <li><a href="#testimonials" class="saas-footer-link">Physician Testimonials</a></li>
                        <li><a href="#partners" class="saas-footer-link">Hospital Partners</a></li>
                    </ul>
                </div>

                <div>
                    <h4 class="saas-footer-title">Enterprise Security</h4>
                    <ul class="saas-footer-links">
                        <li><span class="saas-footer-link">JWT & Argon2 Security</span></li>
                        <li><span class="saas-footer-link">Role-Based Access (RBAC)</span></li>
                        <li><span class="saas-footer-link">Audit Logger & Telemetry</span></li>
                        <li><span class="saas-footer-link">ReportLab EHR PDF Engine</span></li>
                    </ul>
                </div>
            </div>

            <div class="saas-footer-bottom">
                <div>© 2026 AuraScan AI Enterprise Medical Systems. All Rights Reserved.</div>
                <div>For clinical decision support & research purposes. Built with PyTorch & Streamlit.</div>
            </div>
        </footer>
    """)



def render_metric_card(title: str, value: str, border_color: str = "default", value_color: str = "default") -> None:
    """Renders an accessible SaaS metric card with optional border and value color overrides."""
    border_attr = f"border-color: {border_color};" if border_color != "default" else ""
    val_attr = f"color: {value_color};" if value_color != "default" else ""

    st_html(f"""
        <div class="metric-card-saas" style="{border_attr}" role="region" aria-label="Metric: {title} is {value}" tabindex="0">
            <div style="font-size: 12px; font-weight: 500; color: var(--text-muted); margin-bottom: 6px;">{title}</div>
            <div style="font-family: 'Poppins', sans-serif; font-size: 22px; font-weight: 600; {val_attr}">{value}</div>
        </div>
    """)


def render_user_profile(user: dict) -> None:
    """Renders the accessible User Profile & Settings management view."""
    st_html("<h2 class=\"m-0 font-weight-600\">⚙️ Account & System Settings</h2>")
    st.markdown("Manage user profile information, security preferences, and system notification rules.")

    st.divider()

    # Initialize settings active sheet key
    if "settings_active_tab" not in st.session_state:
        st.session_state["settings_active_tab"] = "profile"

    # 2-Column Split: left nav, right content
    col_nav, col_content = st.columns([1, 3])

    with col_nav:
        st_html("""
        <div class="settings-nav-header">
            <span class="font-weight-700 text-muted font-size-11 uppercase letter-spacing-05">User Settings</span>
        </div>
        """)
        
        nav_items = [
            ("👤 Profile", "profile"),
            ("🎨 Appearance", "appearance"),
            ("🔔 Notifications", "notifications"),
            ("🔒 Security & 2FA", "security"),
            ("🔑 API Keys", "api_keys"),
            ("🔌 Integrations", "integrations"),
            ("🌐 Language & Locale", "language"),
            ("♿ Accessibility", "accessibility"),
            ("💻 Sessions & Devices", "sessions")
        ]
        
        for label, val in nav_items:
            is_active = (st.session_state["settings_active_tab"] == val)
            btn_type = "primary" if is_active else "secondary"
            if st.button(label, key=f"sett_nav_btn_{val}", use_container_width=True, type=btn_type):
                st.session_state["settings_active_tab"] = val
                st.rerun()

    with col_content:
        active_tab = st.session_state["settings_active_tab"]

        # Tab 1: Profile Details
        if active_tab == "profile":
            st.markdown("### Public Profile")
            st.markdown("---")
            
            pic_url = user.get("google_profile_pic")
            if pic_url:
                avatar_html = f'<img src="{pic_url}" alt="Avatar" style="width:100%; height:100%; object-fit:cover; border-radius:50%;">'
            else:
                avatar_html = f'{user.get("full_name", "U")[:2].upper()}'

            st_html(f"""
            <div class="display-flex align-items-center gap-16 mb-20">
                <div class="settings-profile-avatar" style="overflow: hidden; display: flex; align-items: center; justify-content: center;">
                    {avatar_html}
                </div>
                <div>
                    <h4 class="m-0 text-primary">{user.get("full_name", "User")}</h4>
                    <span class="font-size-12 text-muted">{user.get("role", "doctor").upper()} | {user.get("email", "")}</span>
                </div>
            </div>
            """)

            u_name = st.text_input("Full Name", value=user.get("full_name", "User"), key="prof_name_input")
            u_email = st.text_input("Email Address", value=user.get("email", ""), disabled=True, key="prof_email_input")
            u_bio = st.text_area("Biography", value="Clinical Radiologist / Attending Physician specializing in Neuro-Oncology diagnostics.", key="prof_bio_input")
            u_phone = st.text_input("Work Phone", value="+1 (555) 019-2834", key="prof_phone_input")
            
            if st.button("Save Profile Changes", key="save_prof_changes_btn", type="primary"):
                user["full_name"] = u_name
                st.session_state["user"] = user
                render_toast("Profile details updated successfully!", "success")
                st.rerun()

        # Tab 2: Appearance Configurations
        elif active_tab == "appearance":
            st.markdown("### Appearance Customization")
            st.markdown("---")
            
            st.write("**Interface Theme**")
            current_theme = st.session_state.get("theme", "dark")
            theme_choice = st.radio("Select color contrast scheme", ["Light Mode", "Dark Mode"], index=0 if current_theme == "light" else 1)
            
            if (theme_choice == "Light Mode" and current_theme == "dark") or (theme_choice == "Dark Mode" and current_theme == "light"):
                toggle_theme()
                st.rerun()
                
            st.write("**Font Typography**")
            font_choice = st.selectbox("Dashboard Primary Font Family", ["Outfit (Default)", "Inter", "Source Code Pro (Monospace)"])
            
            st.write("**Brand Accent Color**")
            accent_choice = st.color_picker("Clinical Highlights Base Color", value="#10b981")
            
            if st.button("Apply Appearance Settings", key="save_appearance_btn"):
                render_toast("Appearance style sheets updated successfully!", "success")

        # Tab 3: Notification Preferences
        elif active_tab == "notifications":
            st.markdown("### Notification Preferences")
            st.markdown("---")
            
            st.checkbox("Enable Clinical Email Notifications", value=True, key="notif_email_cb", help="Receive automated diagnostic report PDFs via email")
            st.checkbox("EHR Critical Level Alerts via SMS", value=True, key="notif_sms_cb", help="Direct dispatch of High-Risk case summaries to workstation SMS client")
            st.checkbox("Auto-download PDF Reports after Analysis", value=False, key="notif_auto_pdf_cb", help="Automatically trigger PDF report download upon completed scan execution")
            
            st.divider()
            st.write("**Email Digests frequency**")
            freq = st.selectbox("Frequency", ["Real-time alerts", "Daily summary digest", "Weekly registry summary"])
            
            if st.button("Save Notification Settings", key="save_notif_btn"):
                render_toast("Notification policy preferences updated!", "success")

        # Tab 4: Security & 2FA
        elif active_tab == "security":
            st.markdown("### Security Configuration")
            st.markdown("---")
            
            st.write("**Update Password**")
            curr_pass = render_password_input_with_toggle("Current Password", "sec_curr_pass", help_text="Enter your current password")
            new_pass = render_password_input_with_toggle("New Password", "sec_new_pass", help_text="Enter a new secure password")
            conf_pass = render_password_input_with_toggle("Confirm New Password", "sec_conf_pass", help_text="Re-enter new password to confirm")
            
            if st.button("Update Account Password", key="save_sec_pass_btn"):
                if curr_pass and new_pass:
                    if new_pass == conf_pass:
                        render_toast("Account security credentials successfully updated!", "success")
                    else:
                        st.warning("New password fields do not match.")
                else:
                    st.warning("Please complete all password fields.")
                    
            st.divider()
            st.write("**Two-Factor Authentication (2FA)**")
            st.checkbox("Enable 2-Factor Authentication (2FA)", value=True, key="sec_2fa_cb", help="Require 2-Factor verification code upon account login")
            st.info("Two-Factor Authentication adds an extra layer of clinical security by requiring a verification code from your registered authentication application.")

        # Tab 5: API Access Keys
        elif active_tab == "api_keys":
            st.markdown("### Developer API Access Keys")
            st.markdown("---")
            st.write("Manage clinical REST API authorization headers and PACS interface tokens.")
            
            if "api_keys_list" not in st.session_state:
                st.session_state["api_keys_list"] = [
                    {"name": "PACS_ROUTER_INTEGRATION", "prefix": "aur_live_e9cf...", "created": "2026-08-01"},
                    {"name": "EHR_GATEWAY_SYNC", "prefix": "aur_live_38bf...", "created": "2026-08-03"}
                ]
                
            for k in st.session_state["api_keys_list"]:
                k_col1, k_col2, k_col3 = st.columns([3, 2, 1])
                with k_col1:
                    st.markdown(f"**{k['name']}**")
                    st.markdown(f"`{k['prefix']}` (Created: {k['created']})")
                with k_col2:
                    st_html("""<span class="text-success font-weight-bold">🟢 ACTIVE</span>""")
                with k_col3:
                    if st.button("Revoke", key=f"revoke_key_{k['name']}", type="secondary"):
                        st.session_state["api_keys_list"] = [x for x in st.session_state["api_keys_list"] if x["name"] != k["name"]]
                        render_toast(f"API key '{k['name']}' revoked successfully.", "success")
                        st.rerun()
                        
            st.divider()
            st.write("**Generate New API Access Key**")
            new_key_name = st.text_input("API Key Name / Client Identifier", placeholder="e.g. iPad_Radiologist_App")
            
            if st.button("Generate Token Code 🔑", key="generate_api_key_btn", type="primary"):
                if new_key_name.strip():
                    import uuid
                    new_prefix = f"aur_live_{uuid.uuid4().hex[:12]}..."
                    st.session_state["api_keys_list"].append({
                        "name": new_key_name.strip(),
                        "prefix": new_prefix,
                        "created": "2026-08-05"
                    })
                    render_toast(f"API token '{new_key_name}' successfully provisioned!", "success")
                    st.rerun()
                else:
                    st.warning("Please specify an API key client name identifier.")

        # Tab 6: Integrations Sync status
        elif active_tab == "integrations":
            st.markdown("### Clinical Interface Integrations")
            st.markdown("---")
            st.write("Manage active real-time data connections with hospital Electronic Health Records (EHR) databases.")
            
            epic_conn = st.toggle("Connect to Epic Systems EHR", value=True, key="int_epic_toggle")
            if epic_conn:
                st_html("""<div class="text-success font-size-12 font-weight-600 mb-8">🟢 Live EHR Sync active (Epic Hub, Region 4)</div>""")
            else:
                st_html("""<div class="text-muted font-size-12 mb-8">⚪ Epic connection inactive</div>""")
                
            cerner_conn = st.toggle("Connect to Oracle Cerner EHR", value=False, key="int_cerner_toggle")
            if cerner_conn:
                st_html("""<div class="text-success font-size-12 font-weight-600 mb-8">🟢 Live EHR Sync active (Oracle Cloud Hub)</div>""")
            else:
                st_html("""<div class="text-muted font-size-12 mb-8">⚪ Oracle Cerner connection inactive</div>""")

            fhir_conn = st.toggle("FHIR REST API Gateway (v4.0.1)", value=True, key="int_fhir_toggle")
            if fhir_conn:
                st_html("""<div class="text-success font-size-12 font-weight-600 mb-8">🟢 FHIR JSON sync endpoint enabled (https://fhir.aurascan.local/v4)</div>""")
            else:
                st_html("""<div class="text-muted font-size-12 mb-8">⚪ FHIR REST Gateway disconnected</div>""")

            if st.button("Apply Integration Bridges", key="save_integrations_btn"):
                render_toast("EHR Sync Bridges configured successfully!", "success")

        # Tab 7: Language & Timezone Locale
        elif active_tab == "language":
            st.markdown("### Language & Locale Preferences")
            st.markdown("---")
            
            lang_opt = st.selectbox("Primary System Translation Language", ["English (US)", "Spanish (ES)", "French (FR)", "German (DE)", "Japanese (JP)"])
            timezone_opt = st.selectbox("Clinical Reports Timezone Offset", ["UTC/GMT (Greenwich Mean Time)", "EST (Eastern Standard Time)", "PST (Pacific Standard Time)", "IST (India Standard Time)"])
            
            if st.button("Save Language Settings", key="save_language_btn"):
                render_toast("Language and local preferences saved successfully!", "success")

        # Tab 8: Accessibility standards
        elif active_tab == "accessibility":
            st.markdown("### Accessibility Options")
            st.markdown("---")
            
            font_size_val = st.radio("Dashboard Font Scaling", ["Small (12px)", "Medium (14px - Default)", "Large (16px)", "Extra Large (18px)"])
            high_contrast = st.checkbox("Enable High Contrast Grid Borders", value=False)
            screen_readers = st.checkbox("Inject Hidden Screen Reader Descriptors (ARIA-compliance)", value=True)
            
            if st.button("Save Accessibility Standards", key="save_accessibility_btn"):
                render_toast("Accessibility options applied!", "success")

        # Tab 9: Sessions & Devices audit
        elif active_tab == "sessions":
            st.markdown("### Sessions & Hardware Devices")
            st.markdown("---")
            
            st.write("##### Active User Browser Sessions")
            st_html("""
            <div class="font-mono bg-card p-12 radius-md border-1 mb-16">
                <div class="text-success font-weight-bold">✦ Chrome on Windows 11 (Current Web Session)</div>
                <div class="font-size-11 text-muted">IP Address: 192.168.1.45 | Location: Seattle, USA | Login: Today, 08:30</div>
            </div>
            <div class="font-mono bg-card p-12 radius-md border-1 mb-16">
                <div class="text-secondary font-weight-bold">✦ Safari on Apple iPad Pro (Dr. Sarah's Tablet)</div>
                <div class="font-size-11 text-muted">IP Address: 192.168.1.102 | Location: Neuro-Oncology Suite 4 | Login: Yesterday, 14:15</div>
            </div>
            """)
            
            st.write("##### Registered Hospital Hardware Devices")
            st_html("""
            <div class="font-mono bg-card p-12 radius-md border-1 mb-16">
                <div class="text-success font-weight-bold">✦ PACS Room 3B Axial Diagnostic Display Monitor</div>
                <div class="font-size-11 text-muted">Hardware Node ID: HW_PACS_3B | Status: Connected | Calibration Check: Pass</div>
            </div>
            <div class="font-mono bg-card p-12 radius-md border-1 mb-16">
                <div class="text-success font-weight-bold">✦ Siemens MAGNETOM MRI Scanner Data Link (Node 2)</div>
                <div class="font-size-11 text-muted">Hardware Node ID: SCAN_LINK_MRI_2 | Status: Connected | Calibration Check: Pass</div>
            </div>
            """)


# =====================================================================
# DEDICATED FULL-SCREEN CLINICAL AUTHENTICATION VIEWS (2-COLUMN SPLIT)
# =====================================================================

def render_auth_nav_header() -> None:
    """Renders the top navbar header for unauthenticated portal visitors with a modern animated theme toggle."""
    current_theme = st.session_state.get("theme", "dark")
    theme_class = "dark" if current_theme == "dark" else "light"
    
    theme_icon_svg = """
        <svg class="sun-svg" viewBox="0 0 24 24" fill="none" stroke="#F59E0B" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="width:14px; height:14px;"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>
    """ if current_theme == "light" else """
        <svg class="moon-svg" viewBox="0 0 24 24" fill="none" stroke="#38BDF8" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" style="width:14px; height:14px;"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>
    """

    col_logo, col_nav, col_theme = st.columns([3, 2.5, 1.8])
    with col_logo:
        st_html("""
            <div class="display-flex align-items-center gap-10 py-4">
                <span class="font-size-24">🧠</span>
                <div>
                    <span class="font-size-18 font-weight-600 text-primary">AuraScan AI</span>
                    <span class="font-size-10 text-accent font-weight-500 uppercase ml-6 px-6 py-2 bg-info-bg radius-sm border-1">Clinical SaaS</span>
                </div>
            </div>
        """)
    with col_nav:
        nav_col1, nav_col2 = st.columns(2)
        with nav_col1:
            if st.button("Sign In", key="unauth_nav_login", use_container_width=True):
                st.session_state["auth_page"] = "login"
                st.rerun()
        with nav_col2:
            if st.button("Register", key="unauth_nav_reg", use_container_width=True):
                st.session_state["auth_page"] = "register"
                st.rerun()
    with col_theme:
        st_html(f"""
            <div class="display-flex justify-content-end align-items-center height-full" style="padding-top: 4px;">
                <div class="theme-toggle-switch-wrapper" title="Toggle color theme" tabindex="0" aria-label="Toggle color theme"
                     onclick="const btn = document.querySelector('.hidden-theme-btn-auth').closest('[data-testid=stElementContainer]').nextElementSibling.querySelector('button'); if (btn) btn.click();"
                     onkeydown="if (event.key === 'Enter' || event.key === ' ') {{ const btn = document.querySelector('.hidden-theme-btn-auth').closest('[data-testid=stElementContainer]').nextElementSibling.querySelector('button'); if (btn) btn.click(); }}">
                    <div class="theme-toggle-track {theme_class}">
                        <div class="theme-toggle-knob">
                            <span class="theme-icon-container">
                                {theme_icon_svg}
                            </span>
                        </div>
                    </div>
                </div>
            </div>
            <style>
                div[data-testid="stElementContainer"]:has(.hidden-theme-btn-auth) + div[data-testid="stElementContainer"] {{
                    display: none !important;
                }}
            </style>
            <div class="hidden-theme-btn-auth" style="display:none;">
        """)
        if st.button("", key="unauth_theme_toggle_hidden"):
            toggle_theme()
        st_html("</div>")
    st_html("<hr class=\"hr-divider\">")


def render_password_input_with_toggle(label: str, key_prefix: str, help_text: str = "") -> str:
    """Renders a password input field with an interactive visibility eye toggle button inside the input field."""
    st_html(f'<div class="password-input-wrapper" id="wrapper_{key_prefix}">')
    
    pwd_val = st.text_input(label, type="password", placeholder=" ", key=f"input_pass_{key_prefix}", help=help_text)
    
    st_html(f'''
        <button class="password-toggle-btn" type="button" aria-label="Toggle password visibility"
                onclick="
                    const wrapper = document.getElementById('wrapper_{key_prefix}');
                    const input = wrapper ? wrapper.querySelector('input') : null;
                    if (input) {{
                        const isPwd = input.type === 'password';
                        input.type = isPwd ? 'text' : 'password';
                        this.classList.toggle('visible', !isPwd);
                    }}
                ">
            <svg class="eye-icon eye-closed" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"></path>
                <line x1="1" y1="1" x2="23" y2="23"></line>
            </svg>
            <svg class="eye-icon eye-open" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
                <circle cx="12" cy="12" r="3"></circle>
            </svg>
        </button>
    ''')
    st_html('</div>')

    return pwd_val


def render_auth_left_panel(subtitle: str = "Enterprise Neuro-Imaging & AI Diagnostic Intelligence Suite") -> None:
    """Renders the left column branding, SVG medical illustration, and compliance trust badges."""
    st_html(f"""
        <div class="auth-left-brand-panel">
            <div class="auth-animated-bg"></div>
            <div class="auth-panel-content">
                <div class="mb-20">
                    <div class="auth-brand-header">
                        <span class="auth-logo-icon">🧠</span>
                        <h2 class="auth-logo-title">AuraScan AI</h2>
                    </div>
                    <h3 class="m-0 font-size-18 font-weight-600 text-primary mt-8 mb-4">Welcome to AuraScan AI</h3>
                    <p class="m-0 font-size-13 text-muted line-height-15">
                        {subtitle}
                    </p>
                </div>

                <!-- High-Tech Medical Brain MRI Vector SVG Illustration with Animations -->
                <div class="mri-illustration-container text-center">
                    <svg class="medical-illustration-svg" viewBox="0 0 380 200" width="100%" height="190" xmlns="http://www.w3.org/2000/svg">
                        <defs>
                            <linearGradient id="mriGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                                <stop offset="0%" stop-color="var(--accent-primary)" stop-opacity="0.9"/>
                                <stop offset="50%" stop-color="var(--text-accent)" stop-opacity="0.6"/>
                                <stop offset="100%" stop-color="var(--accent-secondary)" stop-opacity="0.9"/>
                            </linearGradient>
                            <linearGradient id="scanBeamGrad" x1="0%" y1="0%" x2="0%" y2="100%">
                                <stop offset="0%" stop-color="var(--text-accent)" stop-opacity="0"/>
                                <stop offset="50%" stop-color="var(--text-accent)" stop-opacity="0.8"/>
                                <stop offset="100%" stop-color="var(--text-accent)" stop-opacity="0"/>
                            </linearGradient>
                        </defs>
                        <!-- Outer Animated HUD Rings -->
                        <circle class="mri-hud-ring" cx="190" cy="100" r="85" fill="none" stroke="url(#mriGrad)" stroke-width="1.5" stroke-dasharray="6,4" opacity="0.6"/>
                        <circle cx="190" cy="100" r="68" fill="none" stroke="var(--accent-primary)" stroke-width="1" opacity="0.3"/>
                        <!-- Brain Contours -->
                        <path d="M 155 60 C 128 60 118 88 118 108 C 118 130 138 145 160 145 C 175 145 182 135 187 126 C 187 98 187 78 155 60 Z" fill="url(#mriGrad)" opacity="0.85"/>
                        <path d="M 225 60 C 252 60 262 88 262 108 C 262 130 242 145 220 145 C 205 145 198 135 193 126 C 193 98 193 78 225 60 Z" fill="url(#mriGrad)" opacity="0.85"/>
                        <!-- Synapse Connections -->
                        <line x1="145" y1="90" x2="175" y2="110" stroke="var(--text-primary)" stroke-width="1.5" opacity="0.8"/>
                        <line x1="175" y1="110" x2="205" y2="110" stroke="var(--text-primary)" stroke-width="1.5" opacity="0.8"/>
                        <line x1="205" y1="110" x2="235" y2="90" stroke="var(--text-primary)" stroke-width="1.5" opacity="0.8"/>
                        <circle class="mri-synapse-node" cx="145" cy="90" r="4" fill="var(--text-accent)"/>
                        <circle class="mri-synapse-node" cx="175" cy="110" r="5" fill="var(--accent-secondary)"/>
                        <circle class="mri-synapse-node" cx="205" cy="110" r="5" fill="var(--accent-secondary)"/>
                        <circle class="mri-synapse-node" cx="235" cy="90" r="4" fill="var(--text-accent)"/>
                        <circle cx="190" cy="100" r="6" fill="var(--status-success)"/>
                        <rect class="mri-scan-beam" x="105" y="45" width="170" height="3" fill="url(#scanBeamGrad)"/>
                    </svg>
                </div>

                <!-- MRI Diagnostic Feature Badges -->
                <div class="compliance-badge-grid">
                    <div class="compliance-badge-card">
                        <div class="badge-icon-box">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"/>
                                <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"/>
                            </svg>
                        </div>
                        <div>
                            <div class="badge-title">Brain MRI Analysis</div>
                            <div class="badge-desc">Advanced voxel-based tissue mapping</div>
                        </div>
                    </div>

                    <div class="compliance-badge-card">
                        <div class="badge-icon-box">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
                                <polyline points="22 4 12 14.01 9 11.01"/>
                            </svg>
                        </div>
                        <div>
                            <div class="badge-title">AI Powered Diagnosis</div>
                            <div class="badge-desc">Instant confidence-calibrated insights</div>
                        </div>
                    </div>

                    <div class="compliance-badge-card">
                        <div class="badge-icon-box">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                                <line x1="9" y1="3" x2="9" y2="21"/>
                                <line x1="15" y1="3" x2="15" y2="21"/>
                                <line x1="3" y1="9" x2="21" y2="9"/>
                                <line x1="3" y1="15" x2="21" y2="15"/>
                            </svg>
                        </div>
                        <div>
                            <div class="badge-title">UNeXt Segmentation</div>
                            <div class="badge-desc">Shifted MLP tokenized boundary mapping</div>
                        </div>
                    </div>

                    <div class="compliance-badge-card">
                        <div class="badge-icon-box">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                <path d="M12 2L2 7l10 5 10-5-10-5z"/>
                                <path d="M2 17l10 5 10-5"/>
                                <path d="M2 12l10 5 10-5"/>
                            </svg>
                        </div>
                        <div>
                            <div class="badge-title">EfficientNet Classification</div>
                            <div class="badge-desc">Multi-class pathology category identification</div>
                        </div>
                    </div>
                </div>

                <!-- Security Footer -->
                <div class="auth-left-panel-footer">
                    🔒 <strong>Enterprise Security:</strong> 256-Bit TLS Encryption • Zero-Trust Access Control • SOC2 Type II Certified
                </div>
            </div>
        </div>
    """)


def render_login_view(auth_use_cases) -> None:
    """Renders Login view in full-screen 2-column layout with responsive glass card layout."""
    col_left, col_right = st.columns([1.1, 1.0])

    with col_left:
        render_auth_left_panel("AI-powered medical imaging analytics and neurological scan workspace.")

    with col_right:
        st_html("""
            <div class="auth-right-glass-card">
                <div class="text-center mb-24">
                    <h3 class="m-0 font-size-22 font-weight-600 text-primary mb-4">Sign in to AuraScan AI</h3>
                    <p class="m-0 font-size-13 text-muted">Enter your clinical credentials to access your workspace.</p>
                </div>
        """)

        # Error message rendering (persisted in session state)
        login_error_key = "login_error_msg"
        if login_error_key not in st.session_state:
            st.session_state[login_error_key] = None

        if st.session_state[login_error_key]:
            st_html(f"""
                <div class="auth-alert-card auth-alert-danger" role="alert">
                    <span class="alert-icon">⚠️</span>
                    <div class="alert-content">
                        {st.session_state[login_error_key]}
                    </div>
                </div>
            """)
            st.session_state[login_error_key] = None

        l_email = st.text_input("Clinical Email Address", placeholder=" ", key="st_login_email", help="Enter your registered email")
        l_pass = render_password_input_with_toggle("Password", "login_pass", help_text="Enter account password")

        st.checkbox("Remember me", value=True, key="st_login_remember")

        st_html("<div class=\"spacer-16\"></div>")
        if st.button("Sign In 🔑", key="st_login_submit_btn", type="primary", use_container_width=True):
            try:
                remember_me = st.session_state.get("st_login_remember", False)
                res = auth_use_cases.login(l_email, l_pass, remember_me=remember_me)
                
                st.session_state["user"] = res["user"]
                st.session_state["access_token"] = res["access_token"]
                st.session_state["refresh_token"] = res["refresh_token"]
                render_toast(f"Welcome back, {res['user']['full_name']}!", "success")
                
                # Set default landing page based on role and redirect immediately
                user_role = res["user"].get("role", "patient")
                role_val = user_role.value if hasattr(user_role, "value") else str(user_role)
                role_lower = role_val.lower().strip()
                if role_lower == "admin":
                    landing_page = "🔑 Admin Dashboard"
                elif role_lower == "doctor":
                    landing_page = "🩺 Doctor Dashboard"
                else:
                    landing_page = "👤 Patient Dashboard"
                st.session_state["page"] = landing_page
                st.session_state["nav_page"] = landing_page
                st.rerun()
            except Exception as err:
                st.session_state[login_error_key] = str(err)
                st.rerun()

        st_html("""
                <div class="auth-legal-footer mt-24">
                    Trouble signing in or new to the platform?
                </div>
        """)
        col_forgot, col_reg_btn = st.columns(2)
        with col_forgot:
            if st.button("Forgot Password? 🔍", key="login_to_forgot_btn", use_container_width=True):
                st.session_state["auth_page"] = "forgot"
                st.rerun()
        with col_reg_btn:
            if st.button("Create Account ✨", key="login_to_reg_btn", use_container_width=True):
                st.session_state["auth_page"] = "register"
                st.rerun()

        st_html("""
                <div class="auth-legal-footer mt-16">
                    Protected under HIPAA and SaMD regulations.
                </div>
            </div>
        """)


def render_register_view(auth_use_cases) -> None:
    """Renders Register view in full-screen 2-column layout with responsive glass card layout."""
    col_left, col_right = st.columns([1.1, 1.0])

    with col_left:
        render_auth_left_panel("Create a new clinician or patient account to access AI diagnostic tools and neurological reports.")

    with col_right:
        st_html("""
            <div class="auth-right-glass-card">
                <div class="text-center mb-24">
                    <h3 class="m-0 font-size-22 font-weight-600 text-primary mb-4">Create your account</h3>
                    <p class="m-0 font-size-13 text-muted">Register your professional or patient credentials.</p>
                </div>
        """)

        # Error message rendering (persisted in session state)
        reg_error_key = "reg_error_msg"
        if reg_error_key not in st.session_state:
            st.session_state[reg_error_key] = None

        if st.session_state[reg_error_key]:
            st_html(f"""
                <div class="auth-alert-card auth-alert-danger" role="alert">
                    <span class="alert-icon">⚠️</span>
                    <div class="alert-content">
                        {st.session_state[reg_error_key]}
                    </div>
                </div>
            """)
            st.session_state[reg_error_key] = None

        # Role Selection (Patient vs Doctor vs Admin)
        r_role = st.radio(
            "Account Type",
            ["patient", "doctor", "admin"],
            format_func=lambda x: "🏥 Doctor / Clinician" if x == "doctor" else ("🔑 System Administrator" if x == "admin" else "👤 Patient / User"),
            key="st_reg_role",
            horizontal=True
        )

        r_name = st.text_input("Full Name", placeholder=" ", key="st_reg_fullname", help="Enter full legal display name")
        r_email = st.text_input("Clinical / Personal Email Address", placeholder=" ", key="st_reg_email", help="Enter registered email address")

        r_pass = render_password_input_with_toggle("Password", "reg_pass", help_text="Create a secure account password")
        render_password_strength_meter(r_pass)

        r_confirm = render_password_input_with_toggle("Confirm Password", "reg_confirm", help_text="Re-enter password to confirm")

        # Live Password Match Validation
        if r_pass and r_confirm:
            if r_pass == r_confirm:
                st_html('<div class="pwd-match-badge pwd-match-success">✓ Passwords match</div>')
            else:
                st_html('<div class="pwd-match-badge pwd-match-error">❌ Passwords do not match</div>')

        st_html("<div class=\"spacer-16\"></div>")
        if st.button("Register Account", key="st_reg_submit_btn", type="primary", use_container_width=True):
            if not r_name:
                st.session_state[reg_error_key] = "Please enter your full name."
                st.rerun()
            elif not r_email:
                st.session_state[reg_error_key] = "Please enter a valid email address."
                st.rerun()
            elif r_pass != r_confirm:
                st.session_state[reg_error_key] = "Passwords do not match. Please re-enter your password."
                st.rerun()
            else:
                try:
                    res = auth_use_cases.register(r_email, r_pass, r_name, r_role)
                    
                    # Auto-login immediately after successful registration
                    st.session_state["user"] = res["user"]
                    st.session_state["access_token"] = res["access_token"]
                    st.session_state["refresh_token"] = res["refresh_token"]
                    render_toast(f"Account Registered! Welcome, {res['user']['full_name']}!", "success")
                    
                    # Redirect directly to Dashboard
                    role_lower = r_role.lower().strip()
                    if role_lower == "admin":
                        landing_page = "🔑 Admin Dashboard"
                    elif role_lower == "doctor":
                        landing_page = "🩺 Doctor Dashboard"
                    else:
                        landing_page = "👤 Patient Dashboard"
                    st.session_state["page"] = landing_page
                    st.session_state["nav_page"] = landing_page
                    st.rerun()
                except Exception as err:
                    st.session_state[reg_error_key] = str(err)
                    st.rerun()

        st_html("""
                <div class="auth-legal-footer mt-24">
                    Already have an account?
                </div>
        """)
        if st.button("Sign In to Account", key="reg_to_login_btn", use_container_width=True):
            st.session_state[reg_error_key] = None
            st.session_state["auth_page"] = "login"
            st.rerun()

        st_html("""
            </div>
        """)


def render_forgot_password_view(auth_use_cases) -> None:
    """Renders Forgot Password view in full-screen 2-column layout with simulated recovery instructions."""
    col_left, col_right = st.columns([1.1, 1.0])

    with col_left:
        render_auth_left_panel("Securely request password recovery or credentials reset simulation.")

    with col_right:
        st_html("""
            <div class="auth-right-glass-card">
                <div class="text-center mb-24">
                    <h3 class="m-0 font-size-22 font-weight-600 text-primary mb-4">Reset Password</h3>
                    <p class="m-0 font-size-13 text-muted">Enter your clinical email address to receive password recovery instructions.</p>
                </div>
        """)

        # Error message rendering (persisted in session state)
        forgot_error_key = "forgot_error_msg"
        if forgot_error_key not in st.session_state:
            st.session_state[forgot_error_key] = None

        if st.session_state[forgot_error_key]:
            st_html(f"""
                <div class="auth-alert-card auth-alert-danger" role="alert">
                    <span class="alert-icon">⚠️</span>
                    <div class="alert-content">
                        {st.session_state[forgot_error_key]}
                    </div>
                </div>
            """)
            st.session_state[forgot_error_key] = None

        # Email input field
        fp_email = st.text_input("Clinical Email Address", placeholder="name@hospital.org", key="st_forgot_email", help="Enter registered email address")

        # Simulated state flag
        submitted_key = "forgot_pwd_submitted"
        if submitted_key not in st.session_state:
            st.session_state[submitted_key] = False

        st_html("<div class=\"spacer-16\"></div>")
        
        # We only render the email input and send button if the request has not been submitted yet
        if not st.session_state[submitted_key]:
            if st.button("Send Reset Instructions ✉️", key="st_forgot_submit_btn", type="primary", use_container_width=True):
                if not fp_email or "@" not in fp_email or "." not in fp_email:
                    st.session_state[forgot_error_key] = "Please enter a valid clinical email address."
                    st.rerun()
                else:
                    import datetime
                    from security.domain.entities import SecurityAuditLog
                    
                    email_clean = fp_email.lower().strip()
                    user = auth_use_cases.user_repo.get_by_email(email_clean)
                    if not user:
                        st.session_state[forgot_error_key] = "No account found with this email address."
                        st.rerun()
                    else:
                        now = datetime.datetime.utcnow().isoformat()
                        audit_log = SecurityAuditLog(
                            id=None,
                            timestamp=now,
                            event_type="PASSWORD_RESET_REQUEST",
                            user_id=user.id,
                            email=user.email,
                            ip_address="127.0.0.1",
                            status="SUCCESS",
                            details=f"Simulated password reset request logged for {user.email}.",
                            user_agent="Streamlit UI"
                        )
                        auth_use_cases.user_repo.log_security_event(audit_log)
                        
                        # Log simulation output to console
                        print(f"\n[SIMULATION] Password reset request registered for user: {user.email} (ID: {user.id})\n", flush=True)
                        
                        st.session_state[submitted_key] = True
                        st.rerun()

        if st.session_state[submitted_key]:
            st_html(f"""
                <div class="auth-alert-card auth-alert-success" role="alert" style="margin-top: 16px;">
                    <span class="alert-icon">✓</span>
                    <div class="alert-content">
                        <strong class="font-size-14">Request Logged (Simulation Mode)</strong>
                        <p class="m-0 mt-4 text-secondary line-height-14">
                            A simulated password reset token has been registered for <code>{fp_email}</code>.
                        </p>
                        <p class="m-0 mt-8 font-size-11 text-muted">
                            Since SMTP email services are offline, please check your console or contact your system administrator to manually reset your password.
                        </p>
                    </div>
                </div>
            """)
            st_html("<div class=\"spacer-16\"></div>")
            if st.button("Clear Status", key="forgot_clear_btn", use_container_width=True):
                st.session_state[submitted_key] = False
                st.rerun()

        st_html("""
                <div class="auth-legal-footer mt-24">
                    Remember your credentials?
                </div>
        """)
        if st.button("Sign In to Account", key="forgot_to_login_btn", use_container_width=True):
            st.session_state[submitted_key] = False
            st.session_state[forgot_error_key] = None
            st.session_state["auth_page"] = "login"
            st.rerun()

        st_html("""
            </div>
        """)


def render_unauthenticated_app(auth_use_cases) -> None:
    """Renders the complete SaaS unauthenticated experience with full-screen pages."""
    if "auth_page" not in st.session_state or st.session_state["auth_page"] not in ["login", "register", "forgot"]:
        st.session_state["auth_page"] = "login"

    render_auth_nav_header()

    active_page = st.session_state["auth_page"]

    if active_page == "login":
        render_login_view(auth_use_cases)
    elif active_page == "register":
        render_register_view(auth_use_cases)
    elif active_page == "forgot":
        render_forgot_password_view(auth_use_cases)



# =====================================================================
# REUSABLE UI DESIGN SYSTEM COMPONENT PRIMITIVES (25 PRIMITIVES)
# =====================================================================

def render_primary_button(label: str, key: str = None) -> bool:
    """Renders a Primary Button using standard design tokens."""
    return st.button(label, key=key, type="primary")

def render_secondary_button(label: str, key: str = None) -> None:
    """Renders a Secondary Button element."""
    st_html(f'<button class="btn-secondary">{label}</button>')

def render_outlined_button(label: str, key: str = None) -> None:
    """Renders an Outlined Button element."""
    st_html(f'<button class="btn-outlined">{label}</button>')

def render_danger_button(label: str, key: str = None) -> None:
    """Renders a Danger Button element."""
    st_html(f'<button class="btn-danger">{label}</button>')

def render_success_badge(label: str, badge_type: str = "success") -> None:
    """Renders a Success/Status Badge element."""
    st_html(f'<span class="badge-{badge_type}"><span>{label}</span></span>')

def render_input(label: str, placeholder: str = "", value: str = "", key: str = None) -> str:
    """Renders an accessible Input field."""
    return st.text_input(label, value=value, placeholder=placeholder, key=key)

def render_dropdown(label: str, options: list, key: str = None) -> str:
    """Renders a Dropdown Select control."""
    return st.selectbox(label, options=options, key=key)

def render_checkbox(label: str, value: bool = False, key: str = None) -> bool:
    """Renders a Checkbox control."""
    return st.checkbox(label, value=value, key=key)

def render_radio(label: str, options: list, key: str = None) -> str:
    """Renders a Radio option control."""
    return st.radio(label, options=options, key=key)

def render_toggle(label: str, checked: bool = False) -> None:
    """Renders a Toggle Switch primitive."""
    checked_attr = "checked" if checked else ""
    st_html(f'''
        <div class="toggle-container">
            <label class="toggle-switch">
                <input type="checkbox" {checked_attr}>
                <span class="toggle-slider"></span>
            </label>
            <span class="toggle-label">{label}</span>
        </div>
    ''')

def render_tabs(tabs_list: list, active_index: int = 0) -> None:
    """Renders a Tabs segmented control primitive."""
    tabs_html = '<div class="tabs-container">'
    for idx, tab_name in enumerate(tabs_list):
        active_cls = " active" if idx == active_index else ""
        tabs_html += f'<button class="tab-item{active_cls}">{tab_name}</button>'
    tabs_html += '</div>'
    st_html(tabs_html)

def render_card(title: str, content_html: str, footer_html: str = None) -> None:
    """Renders a Simple White (Light) / Dark Slate (Dark) Card container primitive."""
    footer_part = f'<div class="card-footer">{footer_html}</div>' if footer_html else ''
    card_html = f'''
        <div class="card-container">
            <div class="card-header">
                <h3 class="card-title">{title}</h3>
            </div>
            <div class="card-body">
                {content_html}
            </div>
            {footer_part}
        </div>
    '''
    st_html(card_html)

def render_alert(message: str, alert_type: str = "info", title: str = None) -> None:
    """Renders an Alert banner primitive."""
    title_html = f'<strong>{title}</strong> - ' if title else ''
    st_html(f'''
        <div class="alert-banner alert-{alert_type}" role="alert">
            <div>{title_html}{message}</div>
        </div>
    ''')

def render_progress(percent: int = 50) -> None:
    """Renders a Progress Bar primitive."""
    st_html(f'''
        <div class="progress-bar">
            <div class="progress-fill" style="width: {min(max(percent, 0), 100)}%;"></div>
        </div>
    ''')

def render_drawer(title: str, content_html: str) -> None:
    """Renders a Drawer slide-out container primitive."""
    drawer_html = f'''
        <div class="drawer-container">
            <h3 class="drawer-title">{title}</h3>
            <div>{content_html}</div>
        </div>
    '''
    st_html(drawer_html)

def render_popover(content_html: str) -> None:
    """Renders a Popover menu overlay primitive."""
    st_html(f'<div class="popover-container">{content_html}</div>')

def render_tooltip(text: str, tooltip_text: str) -> None:
    """Renders a Tooltip hover primitive."""
    st_html(f'''
        <div class="tooltip-box">
            <span>{text}</span>
            <span class="tooltip-text">{tooltip_text}</span>
        </div>
    ''')

def render_avatar(initials: str = "AI", name: str = "Clinician") -> None:
    """Renders an Avatar circle primitive."""
    st_html(f'''
        <div class="avatar-circle" title="{name}">
            <span>{initials}</span>
        </div>
    ''')

def render_breadcrumb(items: list) -> None:
    """Renders a Breadcrumb navigation primitive."""
    crumbs_html = '<nav class="breadcrumb-nav">'
    for idx, item in enumerate(items):
        active_cls = " active" if idx == len(items) - 1 else ""
        crumbs_html += f'<span class="breadcrumb-item{active_cls}">{item}</span>'
        if idx < len(items) - 1:
            crumbs_html += '<span class="breadcrumb-separator">/</span>'
    crumbs_html += '</nav>'
    st_html(crumbs_html)

def render_responsive_grid(cards_html_list: list, columns: int = 3) -> None:
    """Renders a Responsive Grid layout primitive."""
    grid_cls = f"grid-{columns}col"
    inner_html = "".join(cards_html_list)
    st_html(f'<div class="grid-container {grid_cls}">{inner_html}</div>')

