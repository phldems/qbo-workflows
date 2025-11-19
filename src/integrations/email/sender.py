"""Email sender for check processing notifications.

Provides consistent HTML and text layouts for the different notification
types (processed, manual review, duplicate, failed) so every message carries
the same check details and JSON payload when available.
"""

from __future__ import annotations

import json
import mimetypes
import re
import smtplib
import ssl
from email import encoders
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import make_msgid
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from loguru import logger


class EmailSender:
    """Sends notification emails for check processing results."""

    def __init__(
        self,
        smtp_server: str,
        smtp_port: int,
        username: str,
        password: str,
        use_ssl: bool = True,
    ) -> None:
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.use_ssl = use_ssl

        logger.info("Initialized email sender: %s@%s", username, smtp_server)

    def send_success_notification(
        self,
        recipient: str,
        check_info: Dict[str, Any],
        qbo_link: str,
        attach_image: Optional[str] = None,
        json_data: Optional[Dict[str, Any]] = None,
        reply_to_message_id: Optional[str] = None,
        from_address: Optional[str] = None,
        envelope_from: Optional[str] = None,
        original_subject: Optional[str] = None,
    ) -> Optional[str]:
        """Send notification when a check is processed successfully."""

        logger.info("Sending success notification to %s", recipient)
        logger.debug("Success notification payload: %s", check_info)
        subject = _normalize_subject(
            original_subject, check_info, "Processed Successfully"
        )

        callout_html = self._build_callout_html(
            title="Check processed automatically",
            lines=[
                "This check has been added to your QuickBooks Online account.",
            ],
            accent_color="#2ca01c",
        )

        html = self._build_notification_html(
            header_title="✓ Check Processed Successfully",
            header_color="#2ca01c",
            callout_html=callout_html,
            check_info=check_info,
            json_data=json_data,
            cta={"label": "View in QuickBooks Online", "href": qbo_link},
        )

        text = self._build_notification_text(
            intro_lines=[
                "Check Processed Successfully",
                "This check has been automatically processed and added to QuickBooks Online.",
            ],
            check_info=check_info,
            json_data=json_data,
            closing_lines=[f"View in QuickBooks Online: {qbo_link}"],
        )

        return self._send_email(
            recipient,
            subject,
            text,
            html,
            attach_image,
            reply_to_message_id,
            from_address=from_address,
            envelope_from=envelope_from,
        )

    def send_manual_review_notification(
        self,
        recipient: str,
        check_info: Dict[str, Any],
        reason: str,
        attach_image: Optional[str] = None,
        json_data: Optional[Dict[str, Any]] = None,
        reply_to_message_id: Optional[str] = None,
        from_address: Optional[str] = None,
        envelope_from: Optional[str] = None,
        original_subject: Optional[str] = None,
    ) -> Optional[str]:
        """Send notification when a check requires manual review."""

        logger.info("Sending manual review notification to %s", recipient)
        logger.debug(
            "Manual review payload: %s | reason=%s | has_json=%s",
            check_info,
            reason,
            bool(json_data),
        )
        subject = _normalize_subject(
            original_subject, check_info, "Manual Review Required"
        )

        callout_html = self._build_callout_html(
            title="Manual review required",
            lines=[f"Reason: {reason}", "Please review and process in QuickBooks."],
            accent_color="#f39c12",
        )

        html = self._build_notification_html(
            header_title="⚠ Manual Review Required",
            header_color="#f39c12",
            callout_html=callout_html,
            check_info=check_info,
            json_data=json_data,
        )

        text = self._build_notification_text(
            intro_lines=[
                "Manual Review Required",
                f"Reason: {reason}",
                "Please review this check manually in QuickBooks Online.",
            ],
            check_info=check_info,
            json_data=json_data,
        )

        return self._send_email(
            recipient,
            subject,
            text,
            html,
            attach_image,
            reply_to_message_id,
            from_address=from_address,
            envelope_from=envelope_from,
        )

    def send_duplicate_notification(
        self,
        recipient: str,
        check_info: Dict[str, Any],
        json_data: Optional[Dict[str, Any]] = None,
        attach_image: Optional[str] = None,
        reply_to_message_id: Optional[str] = None,
        from_address: Optional[str] = None,
        envelope_from: Optional[str] = None,
        original_subject: Optional[str] = None,
    ) -> Optional[str]:
        """Send notification when a duplicate check is detected."""

        logger.info("Sending duplicate notification to %s", recipient)
        logger.debug("Duplicate notification payload: %s", check_info)
        subject = _normalize_subject(original_subject, check_info, "Duplicate Detected")

        callout_html = self._build_callout_html(
            title="Duplicate check detected",
            lines=[
                "This document was already processed and was moved to the duplicates folder.",
                "No action is required unless you believe this is not a duplicate.",
            ],
            accent_color="#f39c12",
        )

        html = self._build_notification_html(
            header_title="⚠ Duplicate Check Detected",
            header_color="#f39c12",
            callout_html=callout_html,
            check_info=check_info,
            json_data=json_data,
        )

        text = self._build_notification_text(
            intro_lines=[
                "Duplicate Check Detected",
                "This check was previously processed and set aside to prevent duplicates.",
            ],
            check_info=check_info,
            json_data=json_data,
            closing_lines=[
                "If this is not a duplicate, please review the check manually before processing.",
            ],
        )

        return self._send_email(
            recipient,
            subject,
            text,
            html,
            attach_image,
            reply_to_message_id,
            from_address=from_address,
            envelope_from=envelope_from,
        )

    def send_error_notification(
        self,
        recipient: str,
        error_message: str,
        check_info: Optional[Dict[str, Any]] = None,
        context_details: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        reply_to_message_id: Optional[str] = None,
        from_address: Optional[str] = None,
        envelope_from: Optional[str] = None,
        original_subject: Optional[str] = None,
        attach_image: Optional[str] = None,
    ) -> Optional[str]:
        """Send notification when processing fails."""

        logger.info("Sending error notification to %s", recipient)
        logger.debug(
            "Error notification payload: error=%s info=%s context=%s json=%s",
            error_message,
            check_info,
            context_details,
            bool(json_data),
        )
        subject = _normalize_subject(
            original_subject, check_info, "Check Processing Error"
        )

        context_lines = []
        if context_details:
            for key, value in context_details.items():
                context_lines.append(f"{key}: {value}")

        callout_html = self._build_callout_html(
            title="Processing failed",
            lines=[
                f"Error: {error_message}",
                *context_lines,
                "Please investigate and process manually.",
            ],
            accent_color="#e74c3c",
        )

        html = self._build_notification_html(
            header_title="✗ Check Processing Error",
            header_color="#e74c3c",
            callout_html=callout_html,
            check_info=check_info,
            json_data=json_data,
        )

        intro_lines = [
            "Check Processing Error",
            f"Error: {error_message}",
            *context_lines,
        ]

        text = self._build_notification_text(
            intro_lines=intro_lines,
            check_info=check_info,
            json_data=json_data,
            closing_lines=["Please investigate and process manually."],
        )

        return self._send_email(
            recipient,
            subject,
            text,
            html,
            attachment_path=attach_image,
            reply_to_message_id=reply_to_message_id,
            from_address=from_address,
            envelope_from=envelope_from,
        )

    def _send_email(
        self,
        recipient: str,
        subject: str,
        text_body: str,
        html_body: str,
        attachment_path: Optional[str] = None,
        reply_to_message_id: Optional[str] = None,
        from_address: Optional[str] = None,
        envelope_from: Optional[str] = None,
    ) -> Optional[str]:
        """Send the multipart email and return the Message-ID."""

        message = MIMEMultipart("mixed")
        message["Subject"] = subject
        message["From"] = from_address or self.username
        message["To"] = recipient
        message_id = make_msgid()
        message["Message-ID"] = message_id
        logger.debug(
            "Composed email %s → %s subject=%s message_id=%s",
            message["From"],
            recipient,
            subject,
            message_id,
        )

        alternative = MIMEMultipart("alternative")

        if reply_to_message_id:
            self._apply_thread_headers(message, reply_to_message_id)

        alternative.attach(MIMEText(text_body or "", "plain"))
        alternative.attach(MIMEText(html_body or "", "html"))
        message.attach(alternative)

        if attachment_path:
            self._attach_file(message, attachment_path)

        envelope_sender = envelope_from or self.username

        try:
            if self.use_ssl:
                context = ssl.create_default_context()
                with smtplib.SMTP_SSL(
                    self.smtp_server, self.smtp_port, context=context
                ) as server:
                    server.login(self.username, self.password)
                    server.sendmail(envelope_sender, recipient, message.as_string())
            else:
                with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                    server.starttls()
                    server.login(self.username, self.password)
                    server.sendmail(envelope_sender, recipient, message.as_string())

            logger.info("Email sent successfully to %s", recipient)
            logger.debug("SMTP send complete for Message-ID %s", message_id)
        except Exception:  # pragma: no cover - log context before bubbling
            logger.exception("Failed to send email")
            raise
        return message_id

    def _build_notification_html(
        self,
        *,
        header_title: str,
        header_color: str,
        callout_html: str,
        check_info: Optional[Dict[str, Any]],
        json_data: Optional[Dict[str, Any]] = None,
        cta: Optional[Dict[str, str]] = None,
    ) -> str:
        json_section = self._render_json_html(json_data)
        check_details_html = self._build_check_details_html(check_info)
        cta_html = ""
        if cta:
            cta_html = f'<center><a class="button" href="{cta["href"]}">{cta["label"]}</a></center>'

        return f"""
        <html>
          <head>
            <style>
              body {{ font-family: Arial, sans-serif; background-color: #f4f4f4; }}
              .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
              .header {{ background-color: {header_color}; color: white; padding: 20px; text-align: center; }}
              .content {{ background-color: #ffffff; padding: 20px; }}
              .callout {{ background-color: #f9f9f9; border-left: 4px solid {header_color}; padding: 15px; margin-bottom: 20px; }}
              .callout h3 {{ margin-top: 0; }}
              .check-details {{ border-top: 1px solid #e0e0e0; padding-top: 15px; margin-top: 10px; }}
              .detail-row {{ margin: 8px 0; display: flex; justify-content: space-between; }}
              .label {{ font-weight: bold; color: #555; }}
              .value {{ color: #000; }}
              .button {{ display: inline-block; padding: 12px 24px; background-color: {header_color}; color: white; text-decoration: none; border-radius: 4px; margin: 20px 0; }}
              .footer {{ text-align: center; color: #888; font-size: 12px; margin-top: 20px; }}
              pre {{ background: #f4f4f4; padding: 10px; border-radius: 4px; overflow-x: auto; font-size: 11px; }}
            </style>
          </head>
          <body>
            <div class="container">
              <div class="header">
                <h1>{header_title}</h1>
              </div>
              <div class="content">
                {callout_html}
                {check_details_html}
                {cta_html}
                {json_section}
              </div>
              <div class="footer">
                <p>Automated Check Processing System</p>
                <p>This is an automated message.</p>
              </div>
            </div>
          </body>
        </html>
        """

    def _build_notification_text(
        self,
        *,
        intro_lines: Iterable[str],
        check_info: Optional[Dict[str, Any]],
        json_data: Optional[Dict[str, Any]] = None,
        closing_lines: Optional[Iterable[str]] = None,
    ) -> str:
        sections = []

        intro_text = "\n".join(filter(None, (line.strip() for line in intro_lines)))
        if intro_text:
            sections.append(intro_text)

        details_text = self._build_check_details_text(check_info)
        if details_text:
            sections.append(details_text)

        if closing_lines:
            closing = "\n".join(line.strip() for line in closing_lines if line)
            if closing:
                sections.append(closing)

        json_section = self._render_json_text(json_data)
        if json_section:
            sections.append(json_section.strip())

        sections.append("---\nAutomated Check Processing System")
        return "\n\n".join(sections)

    def _build_callout_html(
        self, title: str, lines: Iterable[str], accent_color: str
    ) -> str:
        paragraphs = "".join(f"<p>{line}</p>" for line in lines if line)
        return (
            '<div class="callout" style="border-left-color:'
            + accent_color
            + '">'
            + f"<h3>{title}</h3>"
            + paragraphs
            + "</div>"
        )

    def _build_check_details_html(self, check_info: Optional[Dict[str, Any]]) -> str:
        if not check_info:
            return ""

        confidence = _safe_number(check_info.get("confidence", 0))
        confidence_color = _confidence_color(confidence)

        rows = [
            ("Check Number", check_info.get("check_number", "N/A")),
            ("Amount", self._format_amount(check_info.get("amount"))),
            (
                "Payee",
                check_info.get("payee") or "N/A",
            ),
            (
                "Payor",
                check_info.get("payor") or check_info.get("customer_name") or "N/A",
            ),
            ("Date", check_info.get("date", "N/A")),
            (
                "Confidence Score",
                f'<span style="color:{confidence_color};font-weight:bold">{confidence:.1f}%</span>',
            ),
        ]

        detail_rows = "".join(
            f'<div class="detail-row"><span class="label">{label}:</span>'
            f' <span class="value">{value}</span></div>'
            for label, value in rows
        )

        return (
            f'<div class="check-details"><h3>Check Information</h3>{detail_rows}</div>'
        )

    def _build_check_details_text(self, check_info: Optional[Dict[str, Any]]) -> str:
        if not check_info:
            return ""

        confidence = _safe_number(check_info.get("confidence", 0))
        lines = [
            "Check Information:",
            f"Check Number: {check_info.get('check_number', 'N/A')}",
            f"Amount: {self._format_amount(check_info.get('amount'))}",
            "Payee: " + (check_info.get("payee") or "N/A"),
            "Payor: "
            + (check_info.get("payor") or check_info.get("customer_name") or "N/A"),
            f"Date: {check_info.get('date', 'N/A')}",
            f"Confidence Score: {confidence:.1f}%",
        ]

        return "\n".join(lines)

    def _format_amount(self, amount: Any) -> str:
        if amount in ("", None):
            return "N/A"
        try:
            value = float(amount)
            return f"${value:,.2f}"
        except (TypeError, ValueError):
            return str(amount)

    def _attach_file(self, message: MIMEMultipart, attachment_path: str) -> None:
        path = Path(attachment_path)
        if not path.exists():
            logger.warning("Attachment %s does not exist", attachment_path)
            return

        mime_type, _ = mimetypes.guess_type(str(path))
        if mime_type and mime_type.startswith("image/"):
            with path.open("rb") as file:
                img = MIMEImage(file.read())
                img.add_header("Content-Disposition", "attachment", filename=path.name)
                message.attach(img)
            return

        with path.open("rb") as file:
            maintype, _, subtype = (mime_type or "application/octet-stream").partition(
                "/"
            )
            part = MIMEBase(maintype, subtype or "octet-stream")
            part.set_payload(file.read())
            encoders.encode_base64(part)
            part.add_header(
                "Content-Disposition", f'attachment; filename="{path.name}"'
            )
            if mime_type:
                part.add_header("Content-Type", mime_type)
            message.attach(part)

    def _apply_thread_headers(
        self, message: MIMEMultipart, reply_to_message_id: str
    ) -> None:
        raw = str(reply_to_message_id)
        parts = [p for p in re.split(r"[\s,;]+", raw) if p]
        normalized_parts = []
        for part in parts:
            token = part.strip().strip('"').strip("<>")
            if token:
                normalized_parts.append(f"<{token}>")

        if not normalized_parts:
            return

        message["In-Reply-To"] = normalized_parts[0]
        message["References"] = " ".join(normalized_parts)
        logger.debug(
            "Threading email: In-Reply-To=%s References=%s",
            normalized_parts[0],
            message["References"],
        )

    def _render_json_html(self, json_data: Optional[Dict[str, Any]]) -> str:
        if not json_data:
            return ""

        pretty = json.dumps(json_data, indent=2)
        return (
            '<div class="check-details">'
            "<h3>Extracted Data (JSON)</h3>"
            '<pre style="background:#f4f4f4;padding:10px;border-radius:4px;overflow-x:auto;font-size:11px;">'
            + pretty
            + "</pre></div>"
        )

    def _render_json_text(self, json_data: Optional[Dict[str, Any]]) -> str:
        if not json_data:
            return ""
        return "Extracted Data (JSON):\n" + json.dumps(json_data, indent=2)


def _normalize_subject(
    original_subject: Optional[str],
    check_info: Optional[Dict[str, Any]],
    default_suffix: str,
) -> str:
    if original_subject:
        subject = original_subject.strip()
        return subject if subject.lower().startswith("re:") else f"Re: {subject}"

    if check_info:
        return (
            f"Re: Check #{check_info.get('check_number', 'Unknown')} {default_suffix}"
        )

    return f"Re: {default_suffix}"


def _confidence_color(confidence: float) -> str:
    if confidence >= 90:
        return "#2ca01c"
    if confidence >= 70:
        return "#f39c12"
    return "#e74c3c"


def _safe_number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
