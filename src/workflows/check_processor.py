"""
Main check processing workflow.
Orchestrates the entire check intake and recording process.
"""

import json
import cv2
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
from loguru import logger
from datetime import datetime

from src.config.settings import get_settings
from src.utils.database import Database
from src.utils.id_generator import CheckIDGenerator
from src.ocr.vertex_ai_processor import VertexAIProcessor
from src.ocr.confidence_scorer import ConfidenceScorer
from src.ocr.check_detector import CheckDetector
from src.utils.image_utils import ensure_portrait
from src.integrations.quickbooks.auth import QuickBooksAuth
from src.integrations.quickbooks.salesreceipt import QuickBooksSalesReceipt
from src.integrations.quickbooks.rate_limiter import QBOAPIRateLimiter
from src.integrations.email.monitor import EmailMonitor
from src.integrations.email.sender import EmailSender

try:
    import fitz  # PyMuPDF for PDF conversion

    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    logger.warning("PyMuPDF not installed - PDF check processing will be limited")


class CheckProcessor:
    """Main workflow orchestrator for check processing."""

    def __init__(self, settings=None):
        """
        Initialize check processor with all components.

        Args:
            settings: Optional settings override
        """
        self.settings = settings or get_settings()

        # Initialize database
        self.db = Database(self.settings.database_path)

        # Initialize ID generator
        self.id_gen = CheckIDGenerator(self.db)

        # Initialize OCR components
        self.confidence_scorer = ConfidenceScorer()

        # Initialize check detector with all params from .env
        self.check_detector = CheckDetector(
            dpi=self.settings.check_detector_dpi,
            iqr_factor=self.settings.check_detector_iqr_factor,
            outlier_threshold=self.settings.check_detector_outlier_threshold,
            expand_percent=self.settings.check_detector_expand_percent,
            min_check_dimension_px=self.settings.min_check_dimension_px,
            min_component_area_px=self.settings.min_component_area_px,
            micr_line_max_height_px=self.settings.micr_line_max_height_px,
            micr_line_max_gap_px=self.settings.micr_line_max_gap_px,
            base_detection_confidence=self.settings.base_detection_confidence,
        )
        logger.info("Check detector initialized with .env configuration")

        # Initialize Vertex AI OCR processor
        self.ocr_processor = VertexAIProcessor(
            project_id=self.settings.vertex_ai_project_id,
            location=self.settings.vertex_ai_location,
            model_name=self.settings.vertex_ai_model,
            max_retries=self.settings.vertex_ai_retry_attempts,
            retry_delay=self.settings.vertex_ai_retry_delay,
            retry_backoff_multiplier=self.settings.vertex_ai_retry_backoff_multiplier,
        )
        logger.info(f"Using Vertex AI {self.settings.vertex_ai_model} for OCR")

        # Initialize QuickBooks components
        self.qbo_auth = QuickBooksAuth(
            client_id=self.settings.qbo_client_id,
            client_secret=self.settings.qbo_client_secret,
            redirect_uri=self.settings.qbo_redirect_uri,
            environment=self.settings.qbo_environment,
            database=self.db,
        )

        self.qbo_sales = QuickBooksSalesReceipt(
            base_url=self.settings.get_qbo_base_url(), auth_manager=self.qbo_auth
        )

        self.qbo_rate_limiter = QBOAPIRateLimiter()

        # Initialize email components
        self.email_monitor = EmailMonitor(
            imap_server=self.settings.imap_server,
            username=self.settings.email_username,
            password=self.settings.email_password,
            imap_port=self.settings.imap_port,
            use_ssl=self.settings.imap_use_ssl,
            attachment_dir=self.settings.image_temp_dir,
        )

        self.email_sender = EmailSender(
            smtp_server=self.settings.smtp_server,
            smtp_port=self.settings.smtp_port,
            username=self.settings.email_username,
            password=self.settings.email_password,
            use_ssl=self.settings.smtp_use_ssl,
        )

        logger.info("Check processor initialized")

    def process_new_checks(self):
        """Main processing loop - fetch and process new checks from email."""
        logger.info("Starting check processing cycle")

        try:
            # Fetch unread check emails
            checks = self.email_monitor.fetch_unread_checks(
                self.settings.email_inbox_folder
            )

            if not checks:
                logger.info("No new checks found")
                return

            logger.info(f"Processing {len(checks)} check(s)")

            for check_email in checks:
                try:
                    self.process_single_check(check_email)
                except Exception as e:
                    logger.error(f"Error processing check: {e}")
                    # Move to failed folder
                    self.email_monitor.move_email(
                        check_email["email_uid"],
                        self.settings.email_failed_folder,
                        source_folder=self.settings.email_inbox_folder,
                    )
                    self._archive_notification_copies(
                        check_email, self.settings.email_failed_folder
                    )

        except Exception as e:
            logger.error(f"Error in processing cycle: {e}")
            raise

    def process_single_check(self, check_email: Dict[str, Any]):
        """
        Process check(s) through the complete workflow.
        Handles both single-check and multi-check (multi-page or multi-check-per-page) scenarios.

        Args:
            check_email: Email data with check attachment
        """
        image_path = check_email["filepath"]
        email_uid = check_email["email_uid"]

        logger.info(f"Processing check from {image_path}")

        try:
            # Step 0: Load all pages and detect checks
            pages = self._load_all_pages(image_path)

            if not pages:
                raise ValueError(f"Failed to load any pages from {image_path}")

            # Count total checks across all pages
            total_checks_detected = sum(len(page["checks"]) for page in pages)

            logger.info(
                f"Detected {total_checks_detected} check(s) across {len(pages)} page(s)"
            )

            # Strategy: For multi-page files, process EACH PAGE as a separate check
            # Detection is used to decide whether to crop or use full page

            # Scenario 1: Single-page file
            if len(pages) == 1:
                page = pages[0]
                if len(page["checks"]) == 0:
                    # No detection - use full page
                    logger.info("Single page, no checks detected - using full page")
                    self._process_single_check_image(
                        check_email, page["image"], image_path, 0, 0
                    )
                elif len(page["checks"]) == 1:
                    # Single check detected - use cropped image
                    logger.info("Single page, 1 check detected - using cropped image")
                    check_image = page["checks"][0]["image"]
                    self._process_single_check_image(
                        check_email, check_image, image_path, 0, 0
                    )
                else:
                    # Multiple checks on single page
                    logger.info(
                        f"Single page, {len(page['checks'])} checks detected - processing each"
                    )
                    self._process_multiple_checks(check_email, pages, image_path)
                return

            # Scenario 2: Multi-page file - process EACH PAGE (detection helps decide crop vs full)
            logger.info(
                f"Multi-page file ({len(pages)} pages) - processing each page separately"
            )
            self._process_multi_page_file(check_email, pages, image_path)

        except Exception as e:
            logger.error(f"Error processing check: {e}")
            self._store_error(check_email, str(e))
            self.email_monitor.move_email(
                email_uid,
                self.settings.email_failed_folder,
                source_folder=self.settings.email_inbox_folder,
            )
            self._archive_notification_copies(
                check_email, self.settings.email_failed_folder
            )
            raise
        finally:
            self._cleanup_saved_crops(check_email)
            self._cleanup_temp_file(image_path)

    def _extract_and_normalize_ocr_data(
        self, image: np.ndarray
    ) -> tuple[Dict[str, Any], Dict[str, Any], str]:
        """
        Extract check fields from image using OCR and normalize the data.

        Args:
            image: Check image as numpy array

        Returns:
            Tuple of (ocr_result, confidence_result, full_extraction_json)

        Raises:
            RuntimeError: If OCR extraction fails
        """
        logger.debug(
            f"📸 Sending image to Vertex AI (shape: {image.shape}, size: {image.nbytes/1024:.1f}KB)"
        )
        ocr_result = self.ocr_processor.extract_check_fields(image)

        # Surface upstream extraction issues immediately
        vertex_error = ocr_result.get("error")
        if vertex_error:
            raw_preview = (ocr_result.get("raw_text") or "")[:500]
            logger.error(f"❌ Vertex AI extraction error: {vertex_error}")
            if raw_preview:
                logger.debug(f"Vertex AI raw response preview: {raw_preview}")
            raise RuntimeError(f"Vertex AI extraction failed: {vertex_error}")

        logger.debug(
            f"✅ Vertex AI extraction complete: {len(ocr_result.get('raw_text', ''))} chars extracted"
        )

        # Normalize to expected format
        fields = {
            "check_number": ocr_result.get("check_number"),
            "amount": ocr_result.get("amount_numeric"),
            "date": ocr_result.get("date"),
            "payee": ocr_result.get("payee"),
            "memo": ocr_result.get("memo"),
            "routing_number": ocr_result.get("routing_number"),
            "account_number": ocr_result.get("account_number"),
            "payor_address": ocr_result.get("payor_address"),
        }

        payor_display_name = self._extract_payor_display_name(ocr_result)
        normalized_customer_name = payor_display_name or "Unknown"
        fields["customer_name"] = normalized_customer_name
        fields["payor"] = normalized_customer_name
        ocr_result["fields"] = fields

        # Store full OCR extraction for database
        import json as json_module

        full_extraction_json = json_module.dumps(ocr_result, indent=2)

        # Build confidence result
        confidence_result = {
            "overall_confidence": ocr_result.get("confidence", 0),
            "field_confidences": ocr_result.get("field_confidences", {}),
            "recommendation": self._get_recommendation(ocr_result.get("confidence", 0)),
        }

        overall_confidence = confidence_result["overall_confidence"]
        logger.info(f"OCR confidence: {overall_confidence:.1f}%")

        return ocr_result, confidence_result, full_extraction_json

    def _get_notification_from_address(self) -> str:
        """
        Get the from address for notification emails.

        Returns:
            From address in format: check_processor@domain
        """
        try:
            domain = self.settings.email_username.split("@", 1)[1]
        except Exception:
            domain = "phldems.org"
        return f"check_processor@{domain}"

    def _handle_duplicate_check(
        self,
        check_email: Dict[str, Any],
        fields: Dict[str, Any],
        ocr_result: Dict[str, Any],
        confidence_result: Dict[str, Any],
        crop_image_path: Optional[str] = None,
    ) -> bool:
        """
        Check if check is a duplicate and handle accordingly.

        Args:
            check_email: Email data
            fields: Extracted check fields
            ocr_result: Full OCR extraction result
            confidence_result: Confidence scoring result

        Returns:
            True if duplicate was detected and handled, False otherwise
        """
        if not (fields.get("routing_number") and fields.get("check_number")):
            return False

        duplicate = self.db.check_duplicate(
            routing_number=fields["routing_number"],
            account_number=fields.get("account_number", ""),
            check_number=fields["check_number"],
            date=fields.get("date", ""),
        )

        if not duplicate:
            return False

        logger.warning(f"⚠️  Duplicate check detected: {fields['check_number']}")

        # Send duplicate notification with extracted metadata
        attachment_source = crop_image_path or check_email["filepath"]

        try:
            message_id = check_email.get("email_message_id")
            from_addr = self._get_notification_from_address()

            check_info = self._compose_check_info(
                fields,
                confidence_result,
                ocr_result,
                check_email,
            )

            notification_id = self.email_sender.send_duplicate_notification(
                self.settings.notification_recipient,
                check_info,
                json_data=ocr_result,
                attach_image=attachment_source,
                reply_to_message_id=message_id,
                original_subject=check_email.get("email_subject"),
                from_address=from_addr,
                envelope_from=self.settings.email_username,
            )
            self._record_notification_message(check_email, notification_id)
            logger.info(
                f"📧 Duplicate notification sent for check #{fields['check_number']}"
            )
        except Exception as e:
            logger.error(
                f"Failed to send duplicate notification email: {e}",
                exc_info=True,
            )
            logger.warning(
                "Continuing with duplicate folder move despite email failure"
            )

        # Move email to duplicates folder
        self.email_monitor.move_email(
            check_email["email_uid"],
            self.settings.email_duplicates_folder,
            source_folder=self.settings.email_inbox_folder,
        )
        self._archive_notification_copies(
            check_email, self.settings.email_duplicates_folder
        )
        logger.info(f"📋 Duplicate check moved to Duplicates folder")
        return True

    def _evaluate_manual_review_requirements(
        self,
        ocr_result: Dict[str, Any],
        confidence_result: Dict[str, Any],
        fields: Dict[str, Any],
    ) -> Optional[str]:
        """
        Evaluate if check requires manual review and return reason if so.

        Args:
            ocr_result: Full OCR extraction result
            confidence_result: Confidence scoring result
            fields: Extracted check fields

        Returns:
            Reason string if manual review required, None otherwise
        """
        # Check if Vertex AI recommends manual review
        if ocr_result.get("requires_manual_review"):
            reasons = []
            if ocr_result.get("is_post_dated"):
                check_age = ocr_result.get("check_age_days", 0)
                reasons.append(f"post-dated ({abs(check_age)} days in future)")
            if ocr_result.get("is_stale_dated"):
                check_age = ocr_result.get("check_age_days", 0)
                reasons.append(f"stale-dated ({check_age} days old)")
            if not ocr_result.get("amounts_match"):
                reasons.append("amounts mismatch")
            if ocr_result.get("check_condition") == "poor":
                reasons.append("poor condition")

            reason_text = ", ".join(reasons) if reasons else "AI recommendation"
            logger.warning(
                f"Manual review required for check {fields.get('check_number')}: {reason_text}"
            )
            return f"Vertex AI flagged for manual review: {reason_text}"

        # Check confidence threshold
        overall_confidence = confidence_result["overall_confidence"]
        if overall_confidence < self.settings.manual_review_threshold:
            logger.warning(
                f"Low confidence ({overall_confidence:.1f}%), manual review required"
            )
            return (
                f"Low OCR confidence ({overall_confidence:.1f}%) below threshold "
                f"{self.settings.manual_review_threshold:.1f}%"
            )

        return None

    def _process_successful_check(
        self,
        check_email: Dict[str, Any],
        fields: Dict[str, Any],
        ocr_result: Dict[str, Any],
        confidence_result: Dict[str, Any],
        full_extraction_json: str,
        original_file_path: str,
        page_num: int,
        check_index: int,
        crop_image_path: Optional[str] = None,
    ) -> None:
        """
        Process a check that passed all validation checks.

        This includes: ID generation, QBO SalesReceipt creation, image attachment,
        database storage, notification sending, and email movement.

        Args:
            check_email: Email data
            fields: Extracted check fields
            ocr_result: Full OCR extraction result
            confidence_result: Confidence scoring result
            full_extraction_json: JSON string of full extraction
            original_file_path: Path to original check file
            page_num: Page number in multi-page PDF
            check_index: Index of check in batch
            crop_image_path: Optional path to the cropped image for attachments
        """
        email_uid = check_email["email_uid"]

        # Generate IDs
        ids = self.id_gen.generate_ids(fields)
        doc_number = self.id_gen.generate_doc_number(fields)
        fields["doc_number"] = doc_number
        logger.info(
            f"🔑 Generated IDs - check_id: {ids['primary_id']}, DocNumber: {doc_number}"
        )
        logger.debug(f"   shareable_id: {ids['shareable_id']}")

        # Create QBO SalesReceipt
        logger.info(
            f"💼 Creating QuickBooks SalesReceipt for check #{fields.get('check_number')}"
        )
        qbo_result = self._create_salesreceipt(fields, ids, ocr_result)
        logger.info(
            f"✅ QBO SalesReceipt created: ID={qbo_result['salesreceipt_id']}, DocNumber={qbo_result['salesreceipt_number']}"
        )

        # Attach image to SalesReceipt
        attachment_source = crop_image_path or original_file_path
        self._attach_image_to_salesreceipt(
            qbo_result["salesreceipt_id"], attachment_source
        )

        # Store in database
        self._store_check_record(
            check_email,
            ocr_result,
            ids,
            qbo_result,
            confidence_result,
            full_extraction_json,
        )

        # Send success notification
        self._send_success_notification(
            check_email,
            fields,
            qbo_result,
            confidence_result,
            ocr_result,
            check_image_path=crop_image_path,
        )

        # Move email to processed folder (only for first check in batch)
        if check_index == 0:
            self.email_monitor.move_email(
                email_uid,
                self.settings.email_processed_folder,
                source_folder=self.settings.email_inbox_folder,
            )
            self._archive_notification_copies(
                check_email, self.settings.email_processed_folder
            )

        logger.info(
            f"✅ Successfully processed check #{fields.get('check_number')} "
            f"(page {page_num}, index {check_index})"
        )

    def _process_single_check_image(
        self,
        check_email: Dict[str, Any],
        image: np.ndarray,
        original_file_path: str,
        page_num: int = 0,
        check_index: int = 0,
    ):
        """
        Process a single check image through OCR and QuickBooks workflow.

        Args:
            check_email: Email data
            image: Check image as numpy array
            original_file_path: Path to original file
            page_num: Page number (for multi-page PDFs)
            check_index: Check index on page (for multi-check pages)
        """
        email_uid = check_email["email_uid"]
        crop_image_path: Optional[str] = None

        try:
            # Extract and normalize OCR data
            ocr_result, confidence_result, full_extraction_json = (
                self._extract_and_normalize_ocr_data(image)
            )

            fields = ocr_result.get("fields", {})

            crop_image_path = self._save_check_crop_image(
                image, check_email, page_num, check_index
            )

            # Check for duplicates
            if self._handle_duplicate_check(
                check_email,
                fields,
                ocr_result,
                confidence_result,
                crop_image_path=crop_image_path,
            ):
                return

            # Evaluate manual review requirements
            manual_review_reason = self._evaluate_manual_review_requirements(
                ocr_result, confidence_result, fields
            )
            if manual_review_reason:
                confidence_result["recommendation"] = {
                    "action": "manual_review",
                    "reason": manual_review_reason,
                }
                self._handle_manual_review(
                    check_email,
                    ocr_result,
                    confidence_result,
                    full_extraction_json,
                    crop_image_path=crop_image_path,
                )
                return

            # Process successful check
            self._process_successful_check(
                check_email,
                fields,
                ocr_result,
                confidence_result,
                full_extraction_json,
                original_file_path,
                page_num,
                check_index,
                crop_image_path=crop_image_path,
            )

        except Exception as e:
            logger.error(
                f"Error processing check (page {page_num}, index {check_index}): {e}"
            )

            # Store error in database
            self._store_error(check_email, str(e))

            # Send error notification
            message_id = check_email.get("email_message_id")
            error_check_info = None
            error_json = None
            try:
                if "confidence_result" in locals():
                    error_check_info = self._compose_check_info(
                        fields if "fields" in locals() else {},
                        confidence_result,
                        ocr_result if "ocr_result" in locals() else None,
                        check_email,
                    )
            except Exception:
                logger.exception("Failed to compose error check info")
            if "ocr_result" in locals():
                error_json = ocr_result

            context_details = {
                "File": check_email.get("filename"),
                "Page": page_num,
                "Index": check_index,
            }

            notification_id = self.email_sender.send_error_notification(
                self.settings.notification_recipient,
                str(e),
                check_info=error_check_info,
                context_details=context_details,
                json_data=error_json,
                reply_to_message_id=message_id,
                original_subject=check_email.get("email_subject"),
                from_address=f"check_processor@{self.settings.email_username.split('@',1)[1] if '@' in self.settings.email_username else 'phldems.org'}",
                envelope_from=self.settings.email_username,
                attach_image=crop_image_path or check_email.get("filepath"),
            )
            self._record_notification_message(check_email, notification_id)

            raise  # Re-raise to let caller handle email movement

    def _process_multiple_checks(
        self,
        check_email: Dict[str, Any],
        pages: list[Dict[str, Any]],
        original_file_path: str,
    ):
        email_uid = check_email["email_uid"]
        batch_results: list[Dict[str, Any]] = []
        total_processed = 0
        total_failed = 0
        check_global_index = 0

        for page_num, page in enumerate(pages):
            checks = page.get("checks", [])
            if not checks:
                continue

            logger.info(f"Processing {len(checks)} check(s) on page {page_num}")

            for check_index, check_data in enumerate(checks):
                check_image = check_data["image"]
                logger.info(
                    f"Processing check {check_global_index + 1}: "
                    f"page {page_num}, index {check_index}, "
                    f"confidence {check_data['confidence']:.1f}%"
                )

                try:
                    # Delegate to single-check handler so duplicates/manual review
                    # routing stays centralized.
                    self._process_single_check_image(
                        check_email,
                        check_image,
                        original_file_path,
                        page_num,
                        check_global_index,
                    )

                    batch_results.append(
                        {
                            "status": "success",
                            "page": page_num,
                            "index": check_index,
                            "detection_confidence": check_data["confidence"],
                        }
                    )
                    total_processed += 1
                except Exception as e:
                    logger.error(f"Failed to process check {check_global_index}: {e}")
                    batch_results.append(
                        {
                            "status": "failed",
                            "page": page_num,
                            "index": check_index,
                            "error": str(e),
                        }
                    )
                    total_failed += 1
                finally:
                    check_global_index += 1

        # Summary logging
        logger.info(
            f"Batch processing complete: {total_processed} succeeded, "
            f"{total_failed} failed out of {check_global_index} total"
        )

        # Move email based on results
        if total_failed == 0:
            # All succeeded
            self.email_monitor.move_email(
                email_uid,
                self.settings.email_processed_folder,
                source_folder=self.settings.email_inbox_folder,
            )
            self._archive_notification_copies(
                check_email, self.settings.email_processed_folder
            )
        elif total_processed == 0:
            # All failed
            self.email_monitor.move_email(
                email_uid,
                self.settings.email_failed_folder,
                source_folder=self.settings.email_inbox_folder,
            )
            self._archive_notification_copies(
                check_email, self.settings.email_failed_folder
            )
        else:
            # Partial success - move to manual review
            logger.warning(
                f"Partial batch failure: {total_failed}/{check_global_index} failed"
            )
            self.email_monitor.move_email(
                email_uid,
                self.settings.email_manual_review_folder,
                source_folder=self.settings.email_inbox_folder,
            )
            self._archive_notification_copies(
                check_email, self.settings.email_manual_review_folder
            )

        # TODO: Send batch summary email
        # self._send_batch_notification(check_email, batch_results)

    def _process_multi_page_file(
        self,
        check_email: Dict[str, Any],
        pages: list[Dict[str, Any]],
        original_file_path: str,
    ):
        """
        Process multi-page file using group check detection per page.

        Strategy:
        - Apply the same detection logic used for single-page files to every page
        - When multiple checks are found on a page, process each cropped detection
        - When a single check is detected, rely on the cropped image for improved OCR
        - When detection fails, fall back to processing the full page image

        Args:
            check_email: Email data
            pages: List of page dicts with check detections
            original_file_path: Path to original file
        """
        email_uid = check_email["email_uid"]
        batch_results = []
        total_processed = 0
        total_failed = 0
        global_check_index = 0

        logger.info(f"Processing {len(pages)} page(s) with detection-aware strategy")

        for page_idx, page_data in enumerate(pages):
            page_num = page_data["page_num"]
            full_page_image = page_data["image"]
            detected_checks = page_data["checks"]

            try:
                if not detected_checks:
                    logger.info(
                        f"Page {page_num + 1}: No checks detected - using FULL PAGE fallback"
                    )
                    current_index = global_check_index
                    try:
                        self._process_single_check_image(
                            check_email,
                            full_page_image,
                            original_file_path,
                            page_num,
                            current_index,
                        )
                        batch_results.append(
                            {
                                "status": "success",
                                "page": page_num,
                                "check_index": current_index,
                                "used_detection": False,
                            }
                        )
                        total_processed += 1
                    except Exception as e:
                        logger.error(f"Failed to process page {page_num} fallback: {e}")
                        batch_results.append(
                            {
                                "status": "failed",
                                "page": page_num,
                                "check_index": current_index,
                                "error": str(e),
                                "used_detection": False,
                            }
                        )
                        total_failed += 1
                    finally:
                        global_check_index += 1
                    continue

                if len(detected_checks) == 1:
                    check_data = detected_checks[0]
                    logger.info(
                        f"Page {page_num + 1}: Single check detected "
                        f"(confidence {check_data.get('confidence', 0):.1f}%) - using crop"
                    )
                    current_index = global_check_index
                    try:
                        self._process_single_check_image(
                            check_email,
                            check_data["image"],
                            original_file_path,
                            page_num,
                            current_index,
                        )
                        batch_results.append(
                            {
                                "status": "success",
                                "page": page_num,
                                "check_index": current_index,
                                "used_detection": True,
                                "detection_confidence": check_data.get("confidence"),
                            }
                        )
                        total_processed += 1
                    except Exception as e:
                        logger.error(
                            f"Failed to process page {page_num} detection: {e}"
                        )
                        batch_results.append(
                            {
                                "status": "failed",
                                "page": page_num,
                                "check_index": current_index,
                                "error": str(e),
                                "used_detection": True,
                                "detection_confidence": check_data.get("confidence"),
                            }
                        )
                        total_failed += 1
                    finally:
                        global_check_index += 1
                    continue

                logger.info(
                    f"Page {page_num + 1}: {len(detected_checks)} checks detected - processing each crop"
                )
                for check_idx, check_data in enumerate(detected_checks):
                    try:
                        self._process_single_check_image(
                            check_email,
                            check_data["image"],
                            original_file_path,
                            page_num,
                            global_check_index,
                        )
                        batch_results.append(
                            {
                                "status": "success",
                                "page": page_num,
                                "check_index": global_check_index,
                                "page_check_index": check_idx,
                                "used_detection": True,
                                "detection_confidence": check_data.get("confidence"),
                            }
                        )
                        total_processed += 1
                    except Exception as e:
                        logger.error(
                            f"Failed to process page {page_num} check {check_idx}: {e}"
                        )
                        batch_results.append(
                            {
                                "status": "failed",
                                "page": page_num,
                                "check_index": global_check_index,
                                "page_check_index": check_idx,
                                "error": str(e),
                                "used_detection": True,
                                "detection_confidence": check_data.get("confidence"),
                            }
                        )
                        total_failed += 1
                    finally:
                        global_check_index += 1
                continue

            except Exception as e:
                logger.error(f"Failed to process page {page_num}: {e}")
                batch_results.append(
                    {"status": "failed", "page": page_num, "error": str(e)}
                )
                total_failed += 1

        # Summary logging
        logger.info(
            f"Multi-page processing complete: {total_processed} succeeded, "
            f"{total_failed} failed out of {global_check_index} checks"
        )

        # Move email based on results
        if total_failed == 0:
            # All succeeded
            self.email_monitor.move_email(
                email_uid,
                self.settings.email_processed_folder,
                source_folder=self.settings.email_inbox_folder,
            )
            self._archive_notification_copies(
                check_email, self.settings.email_processed_folder
            )
        elif total_processed == 0:
            # All failed
            self.email_monitor.move_email(
                email_uid,
                self.settings.email_failed_folder,
                source_folder=self.settings.email_inbox_folder,
            )
            self._archive_notification_copies(
                check_email, self.settings.email_failed_folder
            )
        else:
            # Partial success - move to manual review
            logger.warning(
                f"Partial batch failure: {total_failed}/{global_check_index} checks failed"
            )
            self.email_monitor.move_email(
                email_uid,
                self.settings.email_manual_review_folder,
                source_folder=self.settings.email_inbox_folder,
            )
            self._archive_notification_copies(
                check_email, self.settings.email_manual_review_folder
            )

        # TODO: Send batch summary email
        # self._send_batch_notification(check_email, batch_results)

    def _create_salesreceipt(
        self,
        check_fields: Dict[str, Any],
        ids: Dict[str, str],
        ocr_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create SalesReceipt in QuickBooks."""
        logger.info("Creating QBO SalesReceipt")

        # Wait for rate limit
        self.qbo_rate_limiter.wait_if_needed()

        try:
            # Find or create customer
            customer_name = check_fields.get("customer_name") or "Unknown"
            customer_id = self.qbo_sales.find_or_create_customer(
                customer_name, check_fields.get("payor_address")
            )

            # Create sales receipt
            sales_receipt = self.qbo_sales.create_sales_receipt(
                check_data={
                    "check_number": check_fields.get("check_number"),
                    "amount": check_fields.get("amount"),
                    "date": check_fields.get("date"),
                    "customer_name": customer_name,
                    "routing_number": check_fields.get("routing_number"),
                    "account_number": check_fields.get("account_number"),
                    "memo": check_fields.get("memo"),
                    "micr_line": ocr_result.get("micr_line"),
                    "doc_number": ids.get("primary_id"),
                    "payor_address": check_fields.get("payor_address"),
                },
                customer_id=customer_id,
                payment_method_id=self.settings.qbo_payment_method_id,
                deposit_account_id=self.settings.qbo_deposit_account_id,
                income_item_id=self.settings.qbo_income_item_id,
            )

            return {
                "salesreceipt_id": sales_receipt["Id"],
                "salesreceipt_number": sales_receipt.get("DocNumber"),
                "customer_id": customer_id,
            }

        finally:
            self.qbo_rate_limiter.request_complete()

    def _attach_image_to_salesreceipt(self, salesreceipt_id: str, image_path: str):
        """Attach check image to SalesReceipt."""
        logger.info(f"Attaching image to SalesReceipt {salesreceipt_id}")

        self.qbo_rate_limiter.wait_if_needed()

        try:
            self.qbo_sales.attach_image(salesreceipt_id, image_path)
        finally:
            self.qbo_rate_limiter.request_complete()

    def _store_check_record(
        self,
        check_email: Dict[str, Any],
        ocr_result: Dict[str, Any],
        ids: Dict[str, str],
        qbo_result: Dict[str, Any],
        confidence_result: Dict[str, Any],
        full_extraction_json: str,
    ):
        """Store check record in database."""
        fields = ocr_result.get("fields", {})

        check_data = {
            "check_id": ids["primary_id"],
            "shareable_id": ids["shareable_id"],
            "check_number": fields.get("check_number"),
            "routing_number": fields.get("routing_number"),
            "account_number": fields.get("account_number"),
            "amount": fields.get("amount"),
            "check_date": fields.get("date"),
            "customer_name": fields.get("customer_name"),
            "customer_id": qbo_result.get("customer_id"),
            "email_uid": check_email["email_uid"],
            "email_from": check_email["email_from"],
            "email_subject": check_email["email_subject"],
            "email_received_at": check_email.get("email_date"),
            "ocr_confidence": confidence_result["overall_confidence"],
            "qbo_salesreceipt_id": qbo_result["salesreceipt_id"],
            "qbo_salesreceipt_number": qbo_result["salesreceipt_number"],
            "qbo_created_at": datetime.utcnow().isoformat(),
            "status": "processed",
            "image_path": check_email["filepath"],
            "image_filename": check_email["filename"],
            "processed_at": datetime.utcnow().isoformat(),
            "metadata": full_extraction_json,  # Full Vertex AI OCR extraction
        }

        self.db.insert_check(check_data)
        self.id_gen.store_mapping(ids["primary_id"], fields)
        self.db.log_event(
            ids["primary_id"], "processed", "Successfully processed check"
        )

        logger.debug(
            f"Stored check record: check_id={ids['primary_id']}, qbo_id={qbo_result['salesreceipt_id']}"
        )

    def _handle_manual_review(
        self,
        check_email: Dict[str, Any],
        ocr_result: Dict[str, Any],
        confidence_result: Dict[str, Any],
        full_extraction_json: str,
        crop_image_path: Optional[str] = None,
    ):
        """
        Handle check that requires manual review.

        NOTE: Manual review checks do NOT go to QuickBooks automatically.
        They are saved to database and moved to ManualReview folder for human verification.
        Staff must manually create the QBO entry after reviewing.
        """
        logger.info(
            "⚠️  Manual review required - check will NOT be auto-posted to QuickBooks"
        )
        logger.info(
            f"   Reason: {confidence_result.get('recommendation', {}).get('reason', 'Unknown')}"
        )

        # Store in database as manual_review status
        fields = ocr_result.get("fields", {})
        ids = self.id_gen.generate_ids(fields)

        check_data = {
            "check_id": ids["primary_id"],
            "shareable_id": ids["shareable_id"],
            "check_number": fields.get("check_number"),
            "routing_number": fields.get("routing_number"),
            "account_number": fields.get("account_number"),
            "amount": fields.get("amount"),
            "check_date": fields.get("date"),
            "customer_name": fields.get("customer_name"),
            "email_uid": check_email["email_uid"],
            "email_from": check_email["email_from"],
            "email_subject": check_email["email_subject"],
            "email_received_at": check_email.get("email_date"),
            "ocr_confidence": confidence_result["overall_confidence"],
            "status": "manual_review",
            "image_path": check_email["filepath"],
            "image_filename": check_email["filename"],
            "processed_at": datetime.utcnow().isoformat(),
            "metadata": full_extraction_json,  # Full Vertex AI OCR extraction
        }

        self.db.insert_check(check_data)
        logger.debug(
            f"Stored manual review record: check_id={ids['primary_id']}, reason={confidence_result['recommendation'].get('reason')}"
        )

        # Send manual review notification (non-critical - don't fail if this errors)
        try:
            # Get Message-ID for threading
            message_id = check_email.get("email_message_id")

            # Build from_address using same domain and send to the checks mailbox so reply appears in-thread
            try:
                domain = self.settings.email_username.split("@", 1)[1]
            except Exception:
                domain = "phldems.org"

            from_addr = f"check_processor@{domain}"
            recipient = self.settings.notification_recipient

            check_info = self._compose_check_info(
                fields, confidence_result, ocr_result, check_email
            )

            recommendation = confidence_result.get("recommendation")
            if isinstance(recommendation, dict):
                reason_text = recommendation.get("reason") or "Manual review required"
            elif isinstance(recommendation, str):
                reason_text = recommendation
            else:
                reason_text = "Manual review required"

            notification_id = self.email_sender.send_manual_review_notification(
                recipient,
                check_info,
                reason_text,
                crop_image_path or check_email["filepath"],
                json_data=ocr_result,
                reply_to_message_id=message_id,
                original_subject=check_email.get("email_subject"),
                from_address=from_addr,
                envelope_from=self.settings.email_username,
            )
            self._record_notification_message(check_email, notification_id)
        except Exception as e:
            logger.error(
                f"Failed to send manual review notification email: {e}", exc_info=True
            )
            logger.warning(
                "Continuing with manual review folder move despite email failure"
            )

        # Move email (this should always succeed)
        self.email_monitor.move_email(
            check_email["email_uid"],
            self.settings.email_manual_review_folder,
            source_folder=self.settings.email_inbox_folder,
        )
        self._archive_notification_copies(
            check_email, self.settings.email_manual_review_folder
        )

        logger.info(
            f"📋 Check #{fields.get('check_number')} moved to manual review folder"
        )

    def _send_success_notification(
        self,
        check_email: Dict[str, Any],
        check_fields: Dict[str, Any],
        qbo_result: Dict[str, Any],
        confidence_result: Dict[str, Any],
        ocr_result: Dict[str, Any] = None,
        check_image_path: Optional[str] = None,
    ):
        """Send success notification email with JSON output, threaded to original email."""
        qbo_link = self.qbo_sales.get_salesreceipt_url(qbo_result["salesreceipt_id"])

        check_info = self._compose_check_info(
            check_fields, confidence_result, ocr_result, check_email
        )

        recipient = self.settings.notification_recipient

        # Include the full OCR payload for metadata context in the email
        json_data = ocr_result if ocr_result else None

        # Get Message-ID for threading
        message_id = check_email.get("email_message_id")
        from_addr = self._get_notification_from_address()

        notification_id = self.email_sender.send_success_notification(
            recipient,
            check_info,
            qbo_link,
            attach_image=check_image_path,
            json_data=json_data,
            reply_to_message_id=message_id,
            original_subject=check_email.get("email_subject"),
            from_address=from_addr,
            envelope_from=self.settings.email_username,
        )
        self._record_notification_message(check_email, notification_id)

    def _store_error(self, check_email: Dict[str, Any], error_message: str):
        """Store error in database."""
        try:
            check_data = {
                "check_id": f"error_{datetime.utcnow().timestamp()}",
                "email_uid": check_email["email_uid"],
                "email_from": check_email["email_from"],
                "status": "failed",
                "error_message": error_message,
                "image_path": check_email.get("filepath"),
            }

            self.db.insert_check(check_data)
        except Exception as e:
            logger.error(f"Failed to store error in database: {e}")

    def _extract_payor_display_name(self, ocr_result: Dict[str, Any]) -> str:
        """
        Build a human-friendly payor display name from OCR output.

        Preference order:
        1. Explicit payor string (if Vertex supplies one)
        2. Structured payor_name.full_name value
        3. Combination of first/middle/last from payor_name
        """
        if not ocr_result:
            return ""

        payor_value = ocr_result.get("payor")
        if isinstance(payor_value, str):
            candidate = payor_value.strip()
            if candidate:
                return candidate

        payor_name = ocr_result.get("payor_name")
        if isinstance(payor_name, dict):
            full_name = payor_name.get("full_name")
            if isinstance(full_name, str):
                full_name = full_name.strip()
                if full_name:
                    return full_name

            parts = [
                payor_name.get("first_name"),
                payor_name.get("middle_name"),
                payor_name.get("last_name"),
            ]
            combined = " ".join(
                part.strip() for part in parts if isinstance(part, str) and part.strip()
            )
            if combined:
                return combined
        elif isinstance(payor_name, str):
            candidate = payor_name.strip()
            if candidate:
                return candidate

        return ""

    def _compose_check_info(
        self,
        fields: Optional[Dict[str, Any]],
        confidence_result: Dict[str, Any],
        ocr_result: Optional[Dict[str, Any]] = None,
        check_email: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Merge available sources to build the payload shown in notification emails."""
        merged_fields: Dict[str, Any] = {}
        if ocr_result and isinstance(ocr_result, dict):
            merged_fields.update(ocr_result.get("fields") or {})
        if fields:
            merged_fields.update(fields)

        payor_context: Dict[str, Any] = (
            ocr_result if isinstance(ocr_result, dict) else {}
        )
        if merged_fields.get("payor_name") and (
            not payor_context or not payor_context.get("payor_name")
        ):
            payor_context = {
                **payor_context,
                "payor_name": merged_fields.get("payor_name"),
            }

        payor = (
            merged_fields.get("customer_name")
            or merged_fields.get("payor")
            or self._extract_payor_display_name(payor_context)
        )
        if (not payor) and check_email:
            from_value = check_email.get("email_from") or ""
            payor = from_value.split("<")[0].strip().strip('"')
        payor = payor or "Unknown"

        amount = merged_fields.get("amount")
        if (amount in (None, "")) and ocr_result:
            amount = ocr_result.get("amount_numeric")
        if amount in (None, ""):
            amount = 0

        payee = merged_fields.get("payee") or (ocr_result or {}).get("payee") or "N/A"
        date_value = (
            merged_fields.get("date") or (ocr_result or {}).get("date") or "N/A"
        )

        composed = {
            "check_number": merged_fields.get("check_number", "N/A"),
            "amount": amount,
            "payee": payee,
            "payor": payor,
            "customer_name": payor,
            "date": date_value,
            "confidence": confidence_result.get("overall_confidence", 0),
        }
        logger.debug(
            "Composed check info=%s | merged_fields=%s | email_from=%s",
            composed,
            merged_fields,
            (check_email or {}).get("email_from") if check_email else None,
        )
        return composed

    def _record_notification_message(
        self, check_email: Dict[str, Any], message_id: Optional[str]
    ) -> None:
        """Keep track of sent notifications so we can archive them later."""
        if not message_id:
            return
        key = "_notification_messages"
        queue = check_email.setdefault(key, [])
        queue.append(message_id)
        logger.debug(
            "Recorded notification Message-ID %s (pending=%s) for email %s",
            message_id,
            len(queue),
            check_email.get("email_uid"),
        )

    def _archive_notification_copies(
        self, check_email: Dict[str, Any], destination_folder: Optional[str]
    ) -> None:
        """
        Move any sent notification emails into the folder where the original
        incoming email ended up so the inbox stays clean.
        """
        if not destination_folder:
            return

        pending = check_email.get("_notification_messages") or []
        if not pending:
            check_email["last_destination_folder"] = destination_folder
            logger.debug(
                "No pending notifications for email %s; destination recorded as %s",
                check_email.get("email_uid"),
                destination_folder,
            )
            return

        logger.info(
            "Archiving %s notification(s) for email %s into %s",
            len(pending),
            check_email.get("email_uid"),
            destination_folder,
        )
        for message_id in pending:
            try:
                moved = self.email_monitor.move_email_by_message_id(
                    message_id,
                    destination_folder,
                    source_folder=self.settings.email_inbox_folder,
                )
                if not moved:
                    logger.warning(
                        "Unable to archive notification %s to %s",
                        message_id,
                        destination_folder,
                    )
            except Exception:
                logger.exception("Error while archiving notification %s", message_id)

        check_email["_notification_messages"] = []
        check_email["last_destination_folder"] = destination_folder
        logger.debug(
            "Finished archiving notifications for email %s to %s",
            check_email.get("email_uid"),
            destination_folder,
        )

    def _get_recommendation(self, confidence: float) -> str:
        """
        Get processing recommendation based on confidence score.

        Args:
            confidence: Overall confidence score (0-100)

        Returns:
            Recommendation string: 'auto_process' or 'manual_review'
        """
        if confidence >= self.settings.ocr_confidence_threshold:
            return "auto_process"
        else:
            return "manual_review"

    def _cleanup_temp_file(self, file_path: str):
        """
        Delete temporary file after processing.

        Args:
            file_path: Path to temporary file
        """
        try:
            path = Path(file_path).resolve()
            temp_dir = Path(self.settings.image_temp_dir).resolve()
            if path.exists() and str(path).startswith(str(temp_dir)):
                path.unlink()
                logger.debug(f"🗑️  Deleted temporary file: {file_path}")
            else:
                logger.debug(
                    f"⚠️  Skipped cleanup (not in temp dir or doesn't exist): {file_path}"
                )
        except Exception as e:
            logger.warning(f"Failed to delete temporary file {file_path}: {e}")

    def _cleanup_saved_crops(self, check_email: Dict[str, Any]) -> None:
        """Delete any temporary crop images recorded on the email payload."""
        crop_paths = check_email.pop("_crop_paths", []) or []
        for crop_path in crop_paths:
            self._cleanup_temp_file(crop_path)

    def _save_check_crop_image(
        self,
        image: np.ndarray,
        check_email: Dict[str, Any],
        page_num: int,
        check_index: int,
    ) -> Optional[str]:
        """
        Persist cropped check image for later attachment use.

        Returns:
            Path to the saved image, or None on failure.
        """
        try:
            crops_dir = Path(self.settings.image_temp_dir) / "crops"
            crops_dir.mkdir(parents=True, exist_ok=True)
            base_name = Path(check_email.get("filename", "check")).stem or "check"
            timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
            filename = f"{base_name}_p{page_num + 1}_idx{check_index}_{timestamp}.png"
            output_path = crops_dir / filename
            success = cv2.imwrite(str(output_path), image)
            if not success:
                logger.warning(
                    "Failed to save cropped image for page %s index %s",
                    page_num,
                    check_index,
                )
                return None
            crop_paths = check_email.setdefault("_crop_paths", [])
            crop_paths.append(str(output_path))
            logger.debug("Saved cropped check image to %s", output_path)
            return str(output_path)
        except Exception:
            logger.exception(
                "Unexpected error while saving cropped image for page %s index %s",
                page_num,
                check_index,
            )
            return None

    def _load_all_pages(self, file_path: str) -> list[Dict[str, Any]]:
        """
        Load all pages from PDF or single image file.

        Args:
            file_path: Path to PDF or image file

        Returns:
            List of dicts with:
                - page_num: Page number (0-indexed)
                - image: Page image as numpy array (RGB format)
                - checks: List of check images if multiple detected on page
        """
        path = Path(file_path)

        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return []

        pages = []

        # Handle PDF files - extract all pages
        if path.suffix.lower() == ".pdf":
            if not PDF_SUPPORT:
                logger.error("PDF file provided but PyMuPDF is not installed")
                return []

            try:
                logger.info(f"Loading PDF: {file_path}")
                pdf_document = fitz.open(file_path)
                num_pages = len(pdf_document)
                logger.info(f"PDF has {num_pages} page(s)")

                for page_num in range(num_pages):
                    page = pdf_document[page_num]

                    # Convert to image at 300 DPI for quality
                    pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))

                    # Convert to numpy array
                    img_data = pix.samples
                    image = np.frombuffer(img_data, dtype=np.uint8).reshape(
                        pix.height, pix.width, pix.n
                    )

                    # Convert RGBA to RGB if needed
                    if pix.n == 4:
                        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

                    image = ensure_portrait(image)

                    logger.info(
                        f"Loaded page {page_num + 1}/{num_pages} (shape: {image.shape})"
                    )

                    # Detect checks on this page
                    detected_checks = self.check_detector.detect_checks(image)

                    pages.append(
                        {
                            "page_num": page_num,
                            "image": image,
                            "checks": detected_checks,
                        }
                    )

                pdf_document.close()
                logger.info(f"Successfully loaded {num_pages} pages from PDF")
                return pages

            except Exception as e:
                logger.error(f"Failed to load PDF pages: {e}")
                return []

        # Handle regular image files (single page)
        try:
            image = cv2.imread(str(file_path))
            if image is None:
                logger.error(f"Failed to load image: {file_path}")
                return []

            # Convert BGR to RGB (OpenCV loads as BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = ensure_portrait(image)
            logger.info(f"Image loaded (shape: {image.shape})")

            # Detect checks on this image
            detected_checks = self.check_detector.detect_checks(image)

            return [{"page_num": 0, "image": image, "checks": detected_checks}]

        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return []

    def _load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load image from file, converting PDF if needed.
        NOTE: Only loads first page of PDF. Use _load_all_pages() for multi-page support.

        Args:
            image_path: Path to image or PDF file

        Returns:
            Image as numpy array (RGB format), or None if failed
        """
        path = Path(image_path)

        if not path.exists():
            logger.error(f"Image file not found: {image_path}")
            return None

        # Handle PDF files
        if path.suffix.lower() == ".pdf":
            if not PDF_SUPPORT:
                logger.error("PDF file provided but PyMuPDF is not installed")
                return None

            try:
                logger.info(f"Converting PDF to image: {image_path}")
                pdf_document = fitz.open(image_path)
                page = pdf_document[0]  # Get first page

                # Convert to image at 300 DPI for quality
                pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))

                # Convert to numpy array
                img_data = pix.samples
                image = np.frombuffer(img_data, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )

                # Convert RGBA to RGB if needed
                if pix.n == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

                image = ensure_portrait(image)

                pdf_document.close()
                logger.info(f"PDF converted to image (shape: {image.shape})")
                return image

            except Exception as e:
                logger.error(f"Failed to convert PDF: {e}")
                return None

        # Handle regular image files
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None

            # Convert BGR to RGB (OpenCV loads as BGR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            logger.info(f"Image loaded (shape: {image.shape})")
            return image

        except Exception as e:
            logger.error(f"Error loading image: {e}")
            return None
