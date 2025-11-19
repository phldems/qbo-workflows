"""
IMAP email monitoring for incoming check images.
Polls inbox and extracts check attachments.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import time
from loguru import logger

try:
    from imap_tools import MailBox, AND

    IMAP_TOOLS_AVAILABLE = True
except ImportError:
    IMAP_TOOLS_AVAILABLE = False
    logger.warning("imap-tools not available")


class EmailMonitor:
    """Monitors email inbox for check images."""

    def __init__(
        self,
        imap_server: str,
        username: str,
        password: str,
        imap_port: int = 993,
        use_ssl: bool = True,
        attachment_dir: str = "./data/temp",
    ):
        """
        Initialize email monitor.

        Args:
            imap_server: IMAP server address
            username: Email username
            password: Email password
            imap_port: IMAP port
            use_ssl: Use SSL connection
            attachment_dir: Directory to save attachments
        """
        if not IMAP_TOOLS_AVAILABLE:
            raise ImportError(
                "imap-tools not available. Install: pip install imap-tools"
            )

        self.imap_server = imap_server
        self.username = username
        self.password = password
        self.imap_port = imap_port
        self.use_ssl = use_ssl
        self.attachment_dir = Path(attachment_dir)
        self.attachment_dir.mkdir(parents=True, exist_ok=True)

        # Allowed attachment types
        self.allowed_extensions = {".jpg", ".jpeg", ".png", ".pdf", ".tiff", ".tif"}
        self.allowed_content_types = {
            "image/jpeg",
            "image/png",
            "image/tiff",
            "application/pdf",
        }

        logger.info(f"Initialized email monitor for {username}@{imap_server}")

    def fetch_unread_checks(self, folder: str = "INBOX") -> List[Dict[str, Any]]:
        """
        Fetch unread emails with check attachments.

        Args:
            folder: Email folder to monitor

        Returns:
            List of check email data
        """
        logger.info(f"Fetching unread checks from {folder}")

        checks = []

        try:
            with MailBox(self.imap_server).login(
                self.username, self.password, initial_folder=folder
            ) as mailbox:

                # Fetch unseen emails
                for msg in mailbox.fetch(AND(seen=False)):
                    logger.debug(f"Processing email: {msg.subject} from {msg.from_}")

                    # Check for attachments
                    if not msg.attachments:
                        continue

                    # Process each attachment
                    for att in msg.attachments:
                        if self._is_check_attachment(att):
                            # Save attachment
                            saved_info = self._save_attachment(att, msg)

                            if saved_info:
                                # Extract Message-ID for email threading and normalize
                                message_id = None
                                headers = getattr(msg, "headers", None)

                                # Primary attempt: look in headers mapping returned by imap-tools
                                if headers:
                                    # imap-tools often normalizes header keys to lowercase
                                    message_id_values = (
                                        headers.get("message-id")
                                        or headers.get("message_id")
                                        or headers.get("Message-ID")
                                    )
                                    if message_id_values:
                                        if isinstance(message_id_values, (list, tuple)):
                                            raw_mid = (
                                                str(message_id_values[0])
                                                if message_id_values
                                                else None
                                            )
                                        else:
                                            raw_mid = str(message_id_values)

                                        if raw_mid:
                                            import re

                                            # Split on whitespace, commas or semicolons to capture multiple IDs
                                            parts = [
                                                p
                                                for p in re.split(r"[\s,;]+", raw_mid)
                                                if p
                                            ]
                                            normalized_parts = []
                                            for p in parts:
                                                token = p.strip().strip('"')
                                                token = token.strip("<>")
                                                if token:
                                                    normalized_parts.append(
                                                        f"<{token}>"
                                                    )

                                            if normalized_parts:
                                                # Store the full normalized chain (space-separated)
                                                message_id = " ".join(normalized_parts)

                                # Fallbacks: imap-tools message object may expose message id via attributes
                                if not message_id:
                                    for attr in (
                                        "message_id",
                                        "message-id",
                                        "msg_id",
                                        "msgid",
                                        "messageId",
                                    ):
                                        val = getattr(msg, attr, None)
                                        if val:
                                            message_id = str(val)
                                            break

                                # Another fallback: some implementations expose the raw email object
                                if not message_id:
                                    obj = getattr(msg, "obj", None) or getattr(
                                        msg, "message", None
                                    )
                                    try:
                                        if obj and hasattr(obj, "get"):
                                            mid = (
                                                obj.get("Message-ID")
                                                or obj.get("Message-Id")
                                                or obj.get("message-id")
                                            )
                                            if mid:
                                                message_id = str(mid)
                                    except Exception:
                                        # Ignore failures here and leave message_id as None
                                        pass

                                check_data = {
                                    "email_uid": msg.uid,
                                    "email_from": msg.from_,
                                    "email_subject": msg.subject,
                                    "email_date": (
                                        msg.date.isoformat() if msg.date else None
                                    ),
                                    "email_message_id": message_id,  # For email threading
                                    "filename": saved_info["filename"],
                                    "filepath": saved_info["filepath"],
                                    "content_type": att.content_type,
                                }

                                if message_id:
                                    logger.info(
                                        f"📧 Captured Message-ID for threading: {message_id}"
                                    )
                                else:
                                    logger.warning(
                                        f"⚠️  No Message-ID found in email headers - threading will not work"
                                    )

                                checks.append(check_data)
                                logger.info(
                                    f"Found check attachment: {saved_info['filename']}"
                                )

            logger.info(f"Found {len(checks)} check(s)")
            return checks

        except Exception as e:
            logger.error(f"Error fetching emails: {e}")
            raise

    def move_email(
        self, email_uid: str, destination_folder: str, source_folder: str = "INBOX"
    ):
        """
        Move email to a different folder.

        Args:
            email_uid: Email UID
            destination_folder: Destination folder name
            source_folder: Source folder name
        """
        logger.info(f"Moving email {email_uid} to {destination_folder}")

        try:
            with MailBox(self.imap_server).login(
                self.username, self.password
            ) as mailbox:

                _, normalized_names, namespace_prefix = self._get_folder_sets(mailbox)

                # Create folder if doesn't exist
                if destination_folder not in normalized_names:
                    mailbox.folder.create(f"{namespace_prefix}{destination_folder}")
                    normalized_names.add(destination_folder)

                # Select source folder
                mailbox.folder.set(f"{namespace_prefix}{source_folder}")

                # Move email
                mailbox.move(email_uid, f"{namespace_prefix}{destination_folder}")

                logger.info(f"Moved email {email_uid} to {destination_folder}")

        except Exception as e:
            logger.error(f"Error moving email: {e}")
            raise

    def move_email_by_message_id(
        self,
        message_id: Optional[str],
        destination_folder: str,
        source_folder: str = "INBOX",
        max_attempts: int = 5,
        retry_delay: float = 2.0,
    ) -> bool:
        """
        Locate an email by Message-ID in a folder and move it elsewhere.

        Useful for archiving automated replies that get delivered back into the
        monitored inbox.
        """
        normalized_id = (message_id or "").strip()
        if not normalized_id:
            logger.warning("No Message-ID provided for archiving sent email copy")
            return False

        # Try a few common representations (<id>, id without brackets, trimmed)
        stripped = normalized_id.strip("<> ")
        search_values = []
        for candidate in {normalized_id, stripped, f"<{stripped}>"}:
            candidate = candidate.strip()
            if candidate:
                search_values.append(candidate)

        if not search_values:
            logger.warning("Unable to normalize Message-ID %s", message_id)
            return False

        logger.debug(
            "Attempting to move sent notification %s from %s to %s (search %s)",
            normalized_id,
            source_folder,
            destination_folder,
            search_values,
        )

        for attempt in range(1, max_attempts + 1):
            try:
                with MailBox(self.imap_server).login(
                    self.username, self.password
                ) as mailbox:
                    _, normalized_names, namespace_prefix = self._get_folder_sets(
                        mailbox
                    )
                    target_full = f"{namespace_prefix}{destination_folder}"
                    if destination_folder not in normalized_names:
                        mailbox.folder.create(target_full)
                        normalized_names.add(destination_folder)

                    source_full = f"{namespace_prefix}{source_folder}"
                    mailbox.folder.set(source_full)

                    for value in search_values:
                        criteria = f'(HEADER Message-ID "{value}")'
                        logger.debug(
                            "Searching for Message-ID %s with criteria %s (attempt %s)",
                            value,
                            criteria,
                            attempt,
                        )
                        matches = list(
                            mailbox.fetch(
                                criteria=criteria,
                                limit=5,
                                mark_seen=False,
                            )
                        )
                        logger.debug(
                            "Found %s candidate replies for Message-ID %s",
                            len(matches),
                            value,
                        )
                        if not matches:
                            continue

                        for msg in matches:
                            logger.debug(
                                "Moving UID %s (subject=%s) for Message-ID %s",
                                msg.uid,
                                getattr(msg, "subject", ""),
                                value,
                            )
                            mailbox.move(msg.uid, target_full)
                            logger.info(
                                "Moved sent notification with UID %s to %s",
                                msg.uid,
                                destination_folder,
                            )
                        return True

            except Exception as exc:
                logger.warning(
                    "Attempt %s failed while moving sent notification %s: %s",
                    attempt,
                    normalized_id,
                    exc,
                )

            if attempt < max_attempts:
                time.sleep(retry_delay)

        logger.warning(
            "Unable to locate sent notification %s in %s after %s attempts",
            normalized_id,
            source_folder,
            max_attempts,
        )
        return False

    def mark_as_read(self, email_uid: str, folder: str = "INBOX"):
        """
        Mark email as read.

        Args:
            email_uid: Email UID
            folder: Folder containing the email
        """
        try:
            with MailBox(self.imap_server).login(
                self.username, self.password, initial_folder=folder
            ) as mailbox:

                mailbox.flag(email_uid, "\\Seen", True)
                logger.debug(f"Marked email {email_uid} as read")

        except Exception as e:
            logger.error(f"Error marking email as read: {e}")
            raise

    def _is_check_attachment(self, attachment) -> bool:
        """
        Check if attachment is a check image.

        Args:
            attachment: Email attachment object

        Returns:
            True if attachment is a check image
        """
        # Check content type
        if attachment.content_type not in self.allowed_content_types:
            return False

        # Check file extension
        filename_lower = attachment.filename.lower()
        ext = os.path.splitext(filename_lower)[1]

        return ext in self.allowed_extensions

    def _save_attachment(self, attachment, msg) -> Optional[Dict[str, str]]:
        """
        Save attachment to disk.

        Args:
            attachment: Email attachment
            msg: Email message

        Returns:
            Dictionary with filename and filepath
        """
        try:
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            original_name = attachment.filename
            name, ext = os.path.splitext(original_name)

            # Sanitize filename
            safe_name = "".join(
                c for c in name if c.isalnum() or c in (" ", "-", "_")
            ).strip()
            filename = f"{timestamp}_{safe_name}{ext}"

            filepath = self.attachment_dir / filename

            # Save file
            with open(filepath, "wb") as f:
                f.write(attachment.payload)

            logger.debug(f"Saved attachment: {filepath}")

            return {"filename": filename, "filepath": str(filepath)}

        except Exception as e:
            logger.error(f"Error saving attachment: {e}")
            return None

    def _get_folder_sets(self, mailbox):
        existing_folders = mailbox.folder.list()
        delimiter = (
            existing_folders[0].delim
            if existing_folders and existing_folders[0].delim
            else "/"
        )
        namespace_prefix = f"INBOX{delimiter}"

        existing_folder_names = {f.name for f in existing_folders}
        normalized_names = set()
        for name in existing_folder_names:
            if name.startswith(namespace_prefix):
                normalized_names.add(name[len(namespace_prefix) :])
            else:
                normalized_names.add(name)

        return existing_folder_names, normalized_names, namespace_prefix

    def create_folders(self, folders: List[str]):
        """
        Create email folders if they don't exist.

        Args:
            folders: List of folder names to create
        """
        logger.info(f"Creating email folders: {folders}")

        try:
            with MailBox(self.imap_server).login(
                self.username, self.password
            ) as mailbox:

                _, normalized_names, namespace_prefix = self._get_folder_sets(mailbox)

                for folder in folders:
                    if folder not in normalized_names:
                        target_name = f"{namespace_prefix}{folder}"
                        mailbox.folder.create(target_name)
                        logger.info(f"Created folder: {folder}")
                        normalized_names.add(folder)
                    else:
                        logger.debug(f"Folder already exists: {folder}")

        except Exception as e:
            logger.error(f"Error creating folders: {e}")
            raise
