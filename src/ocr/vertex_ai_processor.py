"""
Vertex AI Gemini Pro OCR Processor

This module provides OCR capabilities using Google's Gemini Pro vision model
through the Vertex AI API. It replaces the local OCR engines (TrOCR, PaddleOCR, etc.)
with a cloud-based multimodal LLM approach.
"""

import base64
import json
import re
import time
from pathlib import Path
from typing import Dict, Optional, Any
import numpy as np
from PIL import Image
import io

from google import genai
from google.genai import types

from loguru import logger


class VertexAIProcessor:
    """
    OCR processor using Google Vertex AI Gemini Pro vision model.

    This processor sends check images to Gemini Pro and uses structured prompting
    to extract check fields in a single API call.
    """

    def __init__(
        self,
        project_id: str,
        location: str = "us-central1",
        model_name: str = "gemini-2.5-pro",
        max_retries: int = 2,
        retry_delay: float = 2.0,
        retry_backoff_multiplier: float = 2.0,
    ):
        """
        Initialize the Vertex AI processor.

        Args:
            project_id: Google Cloud project ID
            location: Vertex AI location/region
            model_name: Gemini model to use (default: gemini-2.5-pro)
            max_retries: Number of attempts for Vertex extraction (>=1)
            retry_delay: Initial delay between attempts in seconds
            retry_backoff_multiplier: Factor to multiply delay after each attempt
        """
        self.project_id = project_id
        self.location = location
        self.model_name = model_name
        self.max_retries = max(1, int(max_retries))
        self.retry_delay = max(0.0, float(retry_delay))
        self.retry_backoff_multiplier = (
            float(retry_backoff_multiplier) if retry_backoff_multiplier > 0 else 1.0
        )

        # Initialize Google GenAI client with Vertex AI
        self.client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
        )

        # Define strict JSON schema for comprehensive check extraction
        # Note: google-genai SDK uses "nullable": true instead of type arrays
        self.response_schema = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema={
                "type": "object",
                "properties": {
                    # Basic check information
                    "check_number": {
                        "type": "string",
                        "nullable": True,
                        "description": "Check number from top-right corner (3-4 digits)",
                    },
                    "date": {
                        "type": "string",
                        "nullable": True,
                        "description": "Check date in MM/DD/YYYY format",
                    },
                    # Amount fields
                    "amount_numeric": {
                        "type": "number",
                        "nullable": True,
                        "description": "Numeric dollar amount from box on right side (no $ sign)",
                    },
                    "amount_written": {
                        "type": "string",
                        "nullable": True,
                        "description": "Written dollar amount from amount line (e.g., 'One hundred fifty and 00/100')",
                    },
                    # Payee information
                    "payee": {
                        "type": "string",
                        "nullable": True,
                        "description": "Payee name from 'Pay to the order of' line",
                    },
                    # Payor information
                    "payor_name": {
                        "type": "object",
                        "nullable": True,
                        "properties": {
                            "first_name": {"type": "string", "nullable": True},
                            "middle_name": {"type": "string", "nullable": True},
                            "last_name": {"type": "string", "nullable": True},
                            "is_joint": {"type": "boolean"},
                            "full_name": {"type": "string", "nullable": True},
                        },
                        "required": [
                            "first_name",
                            "middle_name",
                            "last_name",
                            "is_joint",
                            "full_name",
                        ],
                        "description": "Payor name parsed from signature area or printed name",
                    },
                    "payor_address": {
                        "type": "object",
                        "nullable": True,
                        "properties": {
                            "street": {"type": "string", "nullable": True},
                            "city": {"type": "string", "nullable": True},
                            "state": {"type": "string", "nullable": True},
                            "zip": {"type": "string", "nullable": True},
                            "country": {"type": "string", "nullable": True},
                        },
                        "required": ["street", "city", "state", "zip", "country"],
                        "description": "Payor address from top-left of check",
                    },
                    # Bank information
                    "bank_name": {
                        "type": "string",
                        "nullable": True,
                        "description": "Bank name from check header",
                    },
                    "bank_address": {
                        "type": "string",
                        "nullable": True,
                        "description": "Bank address from check header",
                    },
                    # MICR line
                    "micr_line": {
                        "type": "string",
                        "nullable": True,
                        "description": "Complete MICR line from bottom of check (raw text)",
                    },
                    "routing_number": {
                        "type": "string",
                        "nullable": True,
                        "description": "9-digit ABA routing number from MICR line (digits only)",
                    },
                    "account_number": {
                        "type": "string",
                        "nullable": True,
                        "description": "Account number from MICR line (digits only)",
                    },
                    # Memo
                    "memo": {
                        "type": "string",
                        "nullable": True,
                        "description": "Memo line text",
                    },
                    # Signature and verification
                    "signature_present": {
                        "type": "boolean",
                        "description": "Whether a signature is visible on the check",
                    },
                    "signature_matches_payor": {
                        "type": "boolean",
                        "nullable": True,
                        "description": "Whether signature appears to match one of the payor names (null if can't determine)",
                    },
                    # Validation checks
                    "amounts_match": {
                        "type": "boolean",
                        "nullable": True,
                        "description": "Whether numeric and written amounts match (null if either is missing)",
                    },
                    "check_age_days": {
                        "type": "integer",
                        "nullable": True,
                        "description": "Age of check in days from date on check to today (null if date missing)",
                    },
                    "is_stale_dated": {
                        "type": "boolean",
                        "description": "Whether check is older than 180 days",
                    },
                    "is_post_dated": {
                        "type": "boolean",
                        "description": "Whether check date is in the future",
                    },
                    # Check condition
                    "check_condition": {
                        "type": "string",
                        "enum": ["excellent", "good", "fair", "poor"],
                        "description": "Overall physical condition of check (excellent=pristine, good=minor wear, fair=smudged/worn, poor=damaged/illegible)",
                    },
                    "requires_manual_review": {
                        "type": "boolean",
                        "description": "Whether this check requires manual review (low confidence, mismatched amounts, etc.)",
                    },
                    # Raw text and metadata
                    "raw_text": {
                        "type": "string",
                        "description": "All text visible on the check",
                    },
                    "confidence": {
                        "type": "integer",
                        "description": "Overall confidence score 0-100",
                        "minimum": 0,
                        "maximum": 100,
                    },
                    "field_confidences": {
                        "type": "object",
                        "properties": {
                            "check_number": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 100,
                            },
                            "date": {"type": "integer", "minimum": 0, "maximum": 100},
                            "amount_numeric": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 100,
                            },
                            "amount_written": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 100,
                            },
                            "payee": {"type": "integer", "minimum": 0, "maximum": 100},
                            "payor_name": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 100,
                            },
                            "payor_address": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 100,
                            },
                            "bank_name": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 100,
                            },
                            "bank_address": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 100,
                            },
                            "micr_line": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 100,
                            },
                            "routing_number": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 100,
                            },
                            "account_number": {
                                "type": "integer",
                                "minimum": 0,
                                "maximum": 100,
                            },
                            "memo": {"type": "integer", "minimum": 0, "maximum": 100},
                        },
                        "required": [
                            "check_number",
                            "date",
                            "amount_numeric",
                            "amount_written",
                            "payee",
                            "payor_name",
                            "payor_address",
                            "bank_name",
                            "bank_address",
                            "micr_line",
                            "routing_number",
                            "account_number",
                            "memo",
                        ],
                    },
                },
                "required": [
                    "check_number",
                    "date",
                    "amount_numeric",
                    "amount_written",
                    "payee",
                    "payor_name",
                    "payor_address",
                    "bank_name",
                    "bank_address",
                    "micr_line",
                    "routing_number",
                    "account_number",
                    "memo",
                    "signature_present",
                    "signature_matches_payor",
                    "amounts_match",
                    "check_age_days",
                    "is_stale_dated",
                    "is_post_dated",
                    "check_condition",
                    "requires_manual_review",
                    "raw_text",
                    "confidence",
                    "field_confidences",
                ],
            },
        )

        logger.info(
            f"Initialized Vertex AI processor with model {model_name} "
            f"in project {project_id} ({location}) - max retries: {self.max_retries}"
        )

    def _image_to_part(self, image: np.ndarray) -> types.Part:
        """
        Convert numpy image array to GenAI Part object.

        Args:
            image: Image as numpy array (from cv2/preprocessing)

        Returns:
            Part object containing the image
        """
        # Convert numpy array to PIL Image
        if len(image.shape) == 2:  # Grayscale
            pil_image = Image.fromarray(image, mode="L")
        else:  # RGB or BGR
            # Convert BGR to RGB if needed (OpenCV uses BGR)
            if image.shape[2] == 3:
                image = image[:, :, ::-1]
            pil_image = Image.fromarray(image, mode="RGB")

        # Convert to bytes
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        # Create Part object using inline_data
        return types.Part.from_bytes(data=image_bytes, mime_type="image/png")

    def _get_retry_delay(self, attempt_index: int) -> float:
        """
        Calculate the wait time before the next retry attempt.

        Args:
            attempt_index: 1-based attempt counter
        """
        if self.retry_delay <= 0:
            return 0.0
        return self.retry_delay * (self.retry_backoff_multiplier ** (attempt_index - 1))

    def extract_check_fields(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract comprehensive structured check fields using Gemini Pro vision model.

        Args:
            image: Preprocessed check image as numpy array

        Returns:
            Dictionary containing:
                - check_number: Check number
                - date: Check date (MM/DD/YYYY format)
                - amount_numeric: Numeric dollar amount (as float)
                - amount_written: Written dollar amount (as string)
                - payee: Payee name
                - payor_name: Payor name object (first_name, middle_name, last_name, is_joint, full_name)
                - payor_address: Payor address object (street, city, state, zip, country)
                - bank_name: Bank name
                - bank_address: Bank address
                - micr_line: Complete MICR line
                - routing_number: 9-digit ABA routing number
                - account_number: Bank account number
                - memo: Memo line text
                - signature_present: Whether signature is visible
                - signature_matches_payor: Whether signature matches payor
                - amounts_match: Whether numeric and written amounts match
                - check_age_days: Age of check in days
                - is_stale_dated: Whether check is >180 days old
                - is_post_dated: Whether check date is in future
                - check_condition: Physical condition (excellent/good/fair/poor)
                - requires_manual_review: Whether manual review is needed
                - raw_text: Full text extracted from check
                - confidence: Overall confidence score (0-100)
                - field_confidences: Per-field confidence scores
        """
        logger.info("Extracting check fields using Gemini Pro vision model")

        try:
            logger.debug(f"Converting image to Part (shape: {image.shape})")
            # Convert image to Part
            image_part = self._image_to_part(image)
            logger.debug("Image successfully converted to Part")
        except Exception as e:
            logger.error(
                f"Failed to convert image to Part: {type(e).__name__}: {e}",
                exc_info=True,
            )
            return self._get_error_structure("", f"Image conversion failed: {str(e)}")

        # Get today's date for accurate check age calculations
        from datetime import datetime

        today = datetime.now().strftime("%Y-%m-%d")

        # Create comprehensive structured prompt for check extraction
        prompt = f"""You are an expert OCR system specialized in reading bank checks.
Today's date is {today} for reference when calculating check age.
Analyze this check image and extract ALL fields with maximum precision.

CRITICAL RULES:
1. Extract EXACTLY what is written - do not interpret, correct spelling, or assume intent
2. For handwritten fields: transcribe character-by-character, preserve capitalization
3. If a field is blank, illegible, or missing: set it to null
4. Provide honest confidence scores (0-100) based on legibility
5. Return ALL text visible on the check in the raw_text field
6. For nested objects (payor_name, payor_address): all sub-fields are required, use null for missing parts

FIELD EXTRACTION GUIDE:

BASIC CHECK INFORMATION:
- check_number: Top-right corner (usually 3-4 digits, printed) - extract as string
- date: Top-right area near check number - format as MM/DD/YYYY (if year is 2-digit, interpret as 20XX)

AMOUNT FIELDS:
- amount_numeric: Handwritten box on right side ($XXX.XX format) - extract NUMERIC VALUE ONLY (no $ sign) as number
- amount_written: Written amount line (e.g., "One hundred fifty and 00/100 DOLLARS") - extract EXACTLY as written

PAYEE & PAYOR:
- payee: "Pay to the order of" line - extract EXACTLY as written (handwritten)
- payor_name: Parse from top-left printed name area:
  * first_name: First name (null if can't parse)
  * middle_name: Middle name or initial (null if not present)
  * last_name: Last name (null if can't parse)
  * is_joint: true if "and" or "&" appears between names (e.g., "John and Mary Smith")
  * full_name: Complete name as printed (always provide this even if parsing fails)
  * SPECIAL RULE: If joint account, use first name UNLESS signature clearly matches second name
- payor_address: Parse from top-left address area:
  * street: Street address
  * city: City name
  * state: State (2-letter code if possible, e.g., "CA")
  * zip: ZIP code (5 or 9 digits)
  * country: Country (default to "USA" if not shown but address is US format)

BANK INFORMATION:
- bank_name: Bank name from check header/logo area
- bank_address: Bank address if visible on check

MICR LINE (bottom of check in special font):
- micr_line: Complete MICR line as visible (raw text with special characters)
- routing_number: First 9 digits from MICR (format: ⑆123456789⑆) - DIGITS ONLY
- account_number: Account number from MICR (after routing, before check number) - DIGITS ONLY

MEMO:
- memo: Bottom-left "memo" or "for" line - extract verbatim

SIGNATURE & VERIFICATION:
- signature_present: true if signature is visible, false if blank
- signature_matches_payor: true/false if signature appears to match payor name, null if can't determine
- amounts_match: Compare amount_numeric and amount_written - true if they match, false if mismatch, null if either missing
- check_age_days: Calculate days between check date and today (use actual date math), null if date missing
- is_stale_dated: true if check_age_days > 180, false otherwise
- is_post_dated: true if check date is in the future, false otherwise

CHECK CONDITION:
- check_condition: Rate physical condition as:
  * "excellent": Pristine, no wear, all text perfectly clear
  * "good": Minor wear, all text readable
  * "fair": Noticeable wear/smudging, some fields harder to read
  * "poor": Significant damage, torn, water damage, many illegible fields
- requires_manual_review: true if ANY of the following:
  * Overall confidence < 75
  * amounts_match is false
  * Any critical field (amount_numeric, payee, routing_number, account_number) has confidence < 60
  * check_condition is "poor"
  * is_stale_dated is true
  * is_post_dated is true

CONFIDENCE SCORING (per field):
- 90-100: Clearly printed/typed, no ambiguity, standard format
- 75-89: Handwritten but readable with high certainty
- 60-74: Some ambiguity (smudged, unclear characters, non-standard handwriting)
- 40-59: Partially legible, multiple interpretations possible
- 0-39: Illegible, blank, or missing

IMPORTANT: The response will be automatically validated against a strict JSON schema.
All required fields MUST be present (use null for missing/illegible data).
Confidence scores MUST be integers 0-100.
amount_numeric MUST be a number (not string).
Nested objects (payor_name, payor_address) must have all sub-fields present."""

        response_text = ""

        for attempt in range(1, self.max_retries + 1):
            should_retry = attempt < self.max_retries
            try:
                # Generate response with schema enforcement
                logger.debug(
                    f"Calling Vertex AI with model {self.model_name} (attempt {attempt}/{self.max_retries})"
                )
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[prompt, image_part],
                    config=types.GenerateContentConfig(
                        temperature=0.1,  # Low temperature for consistent extraction
                        top_p=0.95,
                        top_k=40,
                        max_output_tokens=4096,  # Increased for comprehensive schema
                        response_mime_type="application/json",
                        response_schema=self.response_schema.response_schema,  # Strict schema enforcement
                    ),
                )

                # Extract JSON from response (guaranteed to match schema)
                response_text = response.text.strip()
                logger.debug(
                    f"Received response from Vertex AI (attempt {attempt}/{self.max_retries}, length: {len(response_text)} chars)"
                )

                # Parse JSON response
                result = json.loads(response_text)

                # Validate and normalize the result
                result = self._normalize_result(result)

                logger.info(
                    f"Successfully extracted check fields with {result['confidence']}% confidence"
                )
                logger.debug(f"Extracted fields: {result}")

                return result

            except json.JSONDecodeError as e:
                logger.error(
                    f"Failed to parse Gemini response as JSON (attempt {attempt}/{self.max_retries}): {e}",
                    exc_info=True,
                )
                logger.error(f"Response text: {response_text}")

                repaired_text = self._attempt_response_repair(response_text)
                if repaired_text:
                    try:
                        logger.debug(
                            "Retrying JSON parse after applying repair heuristic"
                        )
                        result = json.loads(repaired_text)
                        result = self._normalize_result(result)
                        logger.warning(
                            "Recovered malformed JSON from Vertex AI response via repair heuristic"
                        )
                        return result
                    except json.JSONDecodeError as repair_error:
                        logger.error(
                            "Repaired response still invalid JSON: %s",
                            repair_error,
                            exc_info=True,
                        )

                if should_retry:
                    wait_time = self._get_retry_delay(attempt)
                    logger.warning(
                        f"Vertex AI JSON parsing failed - retrying in {wait_time:.1f}s (attempt {attempt + 1}/{self.max_retries})"
                    )
                    if wait_time > 0:
                        time.sleep(wait_time)
                    continue

                # Return a fallback structure with low confidence after final attempt
                return self._get_error_structure(
                    response_text, f"JSON parsing failed: {str(e)}"
                )

            except Exception as e:
                error_msg = f"{type(e).__name__}: {str(e)}"
                logger.error(
                    "Error during Gemini Pro extraction: {}", error_msg, exc_info=True
                )

                if should_retry:
                    wait_time = self._get_retry_delay(attempt)
                    logger.warning(
                        f"Retrying Vertex AI extraction in {wait_time:.1f}s "
                        f"(attempt {attempt + 1}/{self.max_retries}) due to error: {error_msg}"
                    )
                    if wait_time > 0:
                        time.sleep(wait_time)
                    continue

                # Check for common authentication/permission errors
                if (
                    "RefreshError" in type(e).__name__
                    or "invalid_scope" in str(e).lower()
                ):
                    logger.error("⚠️  Google Cloud Authentication Error detected!")
                    logger.error("   This usually means:")
                    logger.error(
                        "   1. Service account doesn't have Vertex AI permissions"
                    )
                    logger.error(
                        "   2. Vertex AI API is not enabled in your GCP project"
                    )
                    logger.error("   3. Service account key is invalid or expired")
                    logger.error(
                        f"   Project: {self.project_id}, Location: {self.location}"
                    )

                # Return error structure after exhausting retries
                return self._get_error_structure("", error_msg)

    def _get_error_structure(self, raw_text: str, error_msg: str) -> Dict[str, Any]:
        """
        Generate a consistent error structure for failed extractions.

        Args:
            raw_text: Raw text to include (if any)
            error_msg: Error message

        Returns:
            Error structure with all required fields
        """
        return {
            "check_number": None,
            "date": None,
            "amount_numeric": None,
            "amount_written": None,
            "payee": None,
            "payor_name": None,
            "payor_address": None,
            "bank_name": None,
            "bank_address": None,
            "micr_line": None,
            "routing_number": None,
            "account_number": None,
            "memo": None,
            "signature_present": False,
            "signature_matches_payor": None,
            "amounts_match": None,
            "check_age_days": None,
            "is_stale_dated": False,
            "is_post_dated": False,
            "check_condition": "poor",
            "requires_manual_review": True,
            "raw_text": raw_text,
            "confidence": 0,
            "field_confidences": {
                "check_number": 0,
                "date": 0,
                "amount_numeric": 0,
                "amount_written": 0,
                "payee": 0,
                "payor_name": 0,
                "payor_address": 0,
                "bank_name": 0,
                "bank_address": 0,
                "micr_line": 0,
                "routing_number": 0,
                "account_number": 0,
                "memo": 0,
            },
            "error": error_msg,
        }

    def _attempt_response_repair(self, response_text: str) -> Optional[str]:
        """
        Try to repair common JSON formatting mistakes returned by the model
        (e.g., missing quotes around keys) so we can still process the response.
        """
        if not response_text:
            return None

        try:
            pattern = re.compile(
                r'(?P<prefix>[{,]\s*)(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*:(?=\s|["-])'
            )

            def _replacer(match: re.Match) -> str:
                prefix = match.group("prefix")
                key = match.group("key")
                return f'{prefix}"{key}":'

            repaired = pattern.sub(_replacer, response_text)
            if repaired != response_text:
                logger.debug("Applied JSON repair heuristic to Vertex response text")
                response_text = repaired

            comma_pattern = re.compile(r'(["}\]0-9truefalsenull])\s*(?="[^"]+":)')

            def _comma_replacer(match: re.Match) -> str:
                return f"{match.group(1)}, "

            repaired = comma_pattern.sub(_comma_replacer, response_text)
            if repaired != response_text:
                logger.debug("Inserted missing commas in Vertex response text")
                response_text = repaired

            return response_text
        except Exception:
            logger.exception(
                "Unexpected error while attempting to repair Vertex response"
            )

        return None

    def _normalize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize and validate the comprehensive extraction result.

        Args:
            result: Raw result from Gemini

        Returns:
            Normalized result dictionary
        """
        # Ensure all required fields exist
        normalized = {
            "check_number": result.get("check_number"),
            "date": result.get("date"),
            "amount_numeric": result.get("amount_numeric"),
            "amount_written": result.get("amount_written"),
            "payee": result.get("payee"),
            "payor_name": result.get("payor_name"),
            "payor_address": result.get("payor_address"),
            "bank_name": result.get("bank_name"),
            "bank_address": result.get("bank_address"),
            "micr_line": result.get("micr_line"),
            "routing_number": result.get("routing_number"),
            "account_number": result.get("account_number"),
            "memo": result.get("memo"),
            "signature_present": result.get("signature_present", False),
            "signature_matches_payor": result.get("signature_matches_payor"),
            "amounts_match": result.get("amounts_match"),
            "check_age_days": result.get("check_age_days"),
            "is_stale_dated": result.get("is_stale_dated", False),
            "is_post_dated": result.get("is_post_dated", False),
            "check_condition": result.get("check_condition", "fair"),
            "requires_manual_review": result.get("requires_manual_review", False),
            "raw_text": result.get("raw_text", ""),
            "confidence": result.get("confidence", 50),
            "field_confidences": result.get("field_confidences", {}),
        }

        # Convert amount_numeric to float if it's a string
        if isinstance(normalized["amount_numeric"], str):
            try:
                # Remove dollar signs and commas
                amount_str = (
                    normalized["amount_numeric"]
                    .replace("$", "")
                    .replace(",", "")
                    .strip()
                )
                normalized["amount_numeric"] = float(amount_str)
            except (ValueError, AttributeError):
                normalized["amount_numeric"] = None

        # Ensure routing_number is string and 9 digits (if present)
        if normalized["routing_number"]:
            routing = str(normalized["routing_number"]).strip()
            # Remove any non-digit characters
            routing = "".join(c for c in routing if c.isdigit())
            normalized["routing_number"] = routing if len(routing) == 9 else None

        # Ensure account_number is string (if present)
        if normalized["account_number"]:
            normalized["account_number"] = str(normalized["account_number"]).strip()

        # Validate payor_name structure
        if normalized["payor_name"] is not None:
            if not isinstance(normalized["payor_name"], dict):
                normalized["payor_name"] = None
            else:
                # Ensure all required sub-fields exist
                payor_defaults = {
                    "first_name": None,
                    "middle_name": None,
                    "last_name": None,
                    "is_joint": False,
                    "full_name": None,
                }
                for key, default in payor_defaults.items():
                    if key not in normalized["payor_name"]:
                        normalized["payor_name"][key] = default

        # Validate payor_address structure
        if normalized["payor_address"] is not None:
            if not isinstance(normalized["payor_address"], dict):
                normalized["payor_address"] = None
            else:
                # Ensure all required sub-fields exist
                address_defaults = {
                    "street": None,
                    "city": None,
                    "state": None,
                    "zip": None,
                    "country": None,
                }
                for key, default in address_defaults.items():
                    if key not in normalized["payor_address"]:
                        normalized["payor_address"][key] = default

        # Ensure field_confidences has all fields
        default_confidences = {
            "check_number": 50,
            "date": 50,
            "amount_numeric": 50,
            "amount_written": 50,
            "payee": 50,
            "payor_name": 50,
            "payor_address": 50,
            "bank_name": 50,
            "bank_address": 50,
            "micr_line": 50,
            "routing_number": 50,
            "account_number": 50,
            "memo": 50,
        }

        for field, default_conf in default_confidences.items():
            if field not in normalized["field_confidences"]:
                normalized["field_confidences"][field] = default_conf

        return normalized

    def extract_text(self, image: np.ndarray) -> str:
        """
        Extract all text from the check image.

        Args:
            image: Preprocessed check image as numpy array

        Returns:
            Full text extracted from the check
        """
        result = self.extract_check_fields(image)
        return result.get("raw_text", "")
