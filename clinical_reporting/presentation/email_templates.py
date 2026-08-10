import re
from typing import Optional, List, Tuple, Any
import jinja2

# Set up a secure, auto-escaped Jinja2 environment
jinja_env = jinja2.Environment(
    loader=jinja2.BaseLoader(),
    autoescape=jinja2.select_autoescape(['html', 'xml'])
)

BASE_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{{ subject }}</title>
</head>
<body style="margin: 0; padding: 0; background-color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; -webkit-font-smoothing: antialiased;">
  <table border="0" cellpadding="0" cellspacing="0" width="100%" style="background-color: #f8fafc; padding: 24px 16px;">
    <tr>
      <td align="center">
        <!-- Card Container -->
        <table border="0" cellpadding="0" cellspacing="0" width="100%" style="max-width: 600px; background-color: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
          <!-- Header -->
          <tr>
            <td style="background-color: #0f172a; padding: 24px; text-align: center; border-bottom: 3px solid #0284c7;">
              <span style="color: #ffffff; font-size: 22px; font-weight: 700; letter-spacing: 0.5px;">AuraScan AI</span>
            </td>
          </tr>
          <!-- Body Content -->
          <tr>
            <td style="padding: 32px 24px; color: #334155; font-size: 16px; line-height: 1.6;">
              {% if title %}
              <h1 style="color: #0f172a; font-size: 20px; font-weight: 700; margin-top: 0; margin-bottom: 16px;">{{ title }}</h1>
              {% endif %}
              
              {{ content | safe }}
            </td>
          </tr>
          <!-- Footer -->
          <tr>
            <td style="background-color: #f1f5f9; padding: 24px; text-align: center; color: #64748b; font-size: 12px; line-height: 1.5; border-top: 1px solid #e2e8f0;">
              <p style="margin: 0 0 8px 0; font-weight: 600;">AuraScan AI Platform</p>
              <p style="margin: 0 0 12px 0;">This is an automated clinical notification. Please do not reply directly to this email.</p>
              <p style="margin: 0; font-size: 11px; color: #94a3b8;">CONFIDENTIALITY NOTICE: This transmission is intended only for the use of the individual or entity named above. It contains highly sensitive health information.</p>
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>"""

BASE_TEXT_TEMPLATE = """==================================================
AuraScan AI
==================================================

{% if title %}{{ title }}
--------------------------------------------------
{% endif %}
{{ content }}

--------------------------------------------------
AuraScan AI Platform
This is an automated clinical notification. Please do not reply directly to this email.

CONFIDENTIALITY NOTICE: This transmission is intended only for the use of the individual or entity named above. It contains highly sensitive health information."""


def validate_url(url: str) -> str:
    """Strictly validates a URL target to prevent dangerous protocol injection and quote breakouts."""
    if not url:
        return ""
    
    stripped = url.strip()
    lower_url = stripped.lower()
    
    # Reject dangerous protocols
    if lower_url.startswith(("javascript:", "data:", "vbscript:", "file:")):
        raise ValueError(f"Unsafe URL scheme detected: {url}")
        
    # Enforce safe prefix protocols
    if not (lower_url.startswith("http://") or lower_url.startswith("https://") or lower_url.startswith("/")):
        raise ValueError(f"Invalid URL protocol: {url}")
        
    # Prevent basic HTML breakouts
    if '"' in stripped or "'" in stripped or ">" in stripped or "<" in stripped:
        raise ValueError(f"Dangerous characters in URL: {url}")
        
    return stripped


def render_cta_button(cta_text: str, cta_url: str) -> str:
    """Generates an email-compatible CTA button styled with brand colors and validated URL."""
    validated_url = validate_url(cta_url)
    
    # Inline styled table for absolute compatibility in legacy clients
    btn_template = """<table border="0" cellpadding="0" cellspacing="0" style="margin: 24px 0;">
  <tr>
    <td align="center" bgcolor="#0284c7" style="border-radius: 4px;">
      <a href="{{ url }}" target="_blank" style="display: inline-block; padding: 12px 24px; color: #ffffff; font-size: 15px; font-weight: 600; text-decoration: none; border-radius: 4px; border: 1px solid #0284c7;">{{ text }}</a>
    </td>
  </tr>
</table>"""
    t = jinja_env.from_string(btn_template)
    return t.render(text=cta_text, url=validated_url)


def render_info_card(card_items: List[Tuple[str, Any]]) -> str:
    """Generates an email-compatible information/detail card."""
    card_template = """<table border="0" cellpadding="0" cellspacing="0" width="100%" style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; margin: 20px 0; padding: 16px;">
  <tr>
    <td>
      <table border="0" cellpadding="0" cellspacing="0" width="100%">
        {% for label, value in items %}
        <tr>
          <td style="padding: 6px 0; font-weight: 600; color: #475569; width: 140px; font-size: 14px; vertical-align: top;">{{ label }}:</td>
          <td style="padding: 6px 0; color: #334155; font-size: 14px; vertical-align: top;">{{ value }}</td>
        </tr>
        {% endfor %}
      </table>
    </td>
  </tr>
</table>"""
    t = jinja_env.from_string(card_template)
    return t.render(items=card_items)


def render_status_badge(label: str, status_type: str = "info") -> str:
    """Generates a color-coded status badge matching severity indicators."""
    status_type = status_type.lower()
    
    if status_type in ("success", "completed", "low"):
        bg_color = "#dcfce7"
        color = "#15803d"
    elif status_type in ("warning", "moderate"):
        bg_color = "#fef9c3"
        color = "#a16207"
    elif status_type in ("danger", "critical", "high"):
        bg_color = "#fee2e2"
        color = "#b91c1c"
    else:  # info
        bg_color = "#e2e8f0"
        color = "#475569"

    badge_template = """<span style="display: inline-block; padding: 4px 8px; font-size: 12px; font-weight: 600; border-radius: 9999px; text-transform: uppercase; background-color: {{ bg }}; color: {{ col }};">{{ text }}</span>"""
    t = jinja_env.from_string(badge_template)
    return t.render(text=label, bg=bg_color, col=color)


class EmailTemplateRenderer:
    """Presentation layer class for compiling HTML and text emails securely using Jinja2."""

    @staticmethod
    def render_generic_notification(
        subject: str,
        title: Optional[str],
        message_body: str,
        cta_text: Optional[str] = None,
        cta_url: Optional[str] = None,
        card_items: Optional[List[Tuple[str, Any]]] = None,
        badge_label: Optional[str] = None,
        badge_type: str = "info"
    ) -> Tuple[str, str]:
        """Compiles a generic notification layout into (HTML, Plain-Text) bodies."""
        from markupsafe import escape

        # Ensure message body is HTML-escaped and supports double-newline linebreaks
        escaped_body = str(escape(message_body)).replace("\n", "<br>")
        
        html_content = f"<p style='margin: 0 0 16px 0; color: #334155;'>{escaped_body}</p>"
        text_content = message_body

        # 1. Badge layout
        if badge_label:
            html_badge = render_status_badge(badge_label, badge_type)
            html_content += f"<div style='margin-bottom: 20px;'>{html_badge}</div>"
            text_content += f"\n\nStatus: [{badge_label.upper()}]"

        # 2. Info Card layout
        if card_items:
            html_card = render_info_card(card_items)
            html_content += html_card
            
            text_card = "\n\nDetails:"
            for label, value in card_items:
                text_card += f"\n- {label}: {value}"
            text_content += text_card

        # 3. CTA Button layout
        if cta_text and cta_url:
            html_button = render_cta_button(cta_text, cta_url)
            html_content += html_button
            text_content += f"\n\n{cta_text}: {cta_url}"

        # 4. Render final Base HTML
        html_template = jinja_env.from_string(BASE_HTML_TEMPLATE)
        html_body = html_template.render(
            subject=subject,
            title=title,
            content=html_content
        )

        # 5. Render final Base Text
        text_template = jinja_env.from_string(BASE_TEXT_TEMPLATE)
        text_body = text_template.render(
            title=title,
            content=text_content
        )

        return html_body, text_body

    @staticmethod
    def render_email_verification(verification_url: str, user_name: str) -> Tuple[str, str]:
        """Renders the account verification email template."""
        subject = "Verify Your AuraScan AI Account"
        title = "Email Verification Required"
        message_body = (
            f"Dear {user_name},\n\n"
            f"Thank you for registering with AuraScan AI. To complete your registration and "
            f"activate your clinical account, please verify your email address by clicking the link below."
        )
        return EmailTemplateRenderer.render_generic_notification(
            subject=subject,
            title=title,
            message_body=message_body,
            cta_text="Verify Email Address",
            cta_url=verification_url
        )

    @staticmethod
    def render_otp(otp_code: str, user_name: str, expires_in_minutes: int = 5) -> Tuple[str, str]:
        """Renders the secure 2FA/OTP login verification email template."""
        subject = "Your AuraScan AI Verification Code"
        title = "One-Time Password (OTP)"
        message_body = (
            f"Dear {user_name},\n\n"
            f"Your secure one-time verification code is provided below. "
            f"This code will expire in {expires_in_minutes} minutes."
        )
        card_items = [
            ("Verification Code", otp_code),
            ("Expires In", f"{expires_in_minutes} minutes")
        ]
        return EmailTemplateRenderer.render_generic_notification(
            subject=subject,
            title=title,
            message_body=message_body,
            card_items=card_items,
            badge_label="Secure 2FA",
            badge_type="warning"
        )

    @staticmethod
    def render_password_reset(reset_url: str, user_name: str) -> Tuple[str, str]:
        """Renders the password recovery email template."""
        subject = "Reset Your AuraScan AI Password"
        title = "Password Recovery Request"
        message_body = (
            f"Dear {user_name},\n\n"
            f"We received a request to reset your password. If you made this request, please "
            f"click the link below to configure your new credentials. This link will expire shortly."
        )
        return EmailTemplateRenderer.render_generic_notification(
            subject=subject,
            title=title,
            message_body=message_body,
            cta_text="Reset Password",
            cta_url=reset_url
        )

    @staticmethod
    def render_password_reset_confirmation(user_name: str) -> Tuple[str, str]:
        """Renders the password reset success confirmation email template."""
        subject = "AuraScan AI Password Changed Successfully"
        title = "Password Reset Confirmed"
        message_body = (
            f"Dear {user_name},\n\n"
            f"The password for your AuraScan AI account has been successfully changed.\n\n"
            f"If you did not request this change, please contact system administration immediately."
        )
        return EmailTemplateRenderer.render_generic_notification(
            subject=subject,
            title=title,
            message_body=message_body,
            badge_label="Security Update",
            badge_type="success"
        )
