"""
Multi-layer confidence scoring system for OCR results.
Validates and scores check data extraction confidence.
"""

import re
from typing import Dict, Any, Optional, List
from loguru import logger


class ConfidenceScorer:
    """Calculates comprehensive confidence scores for OCR results."""

    def __init__(
        self,
        char_confidence_weight: float = 0.4,
        format_weight: float = 0.2,
        checksum_weight: float = 0.2,
        consistency_weight: float = 0.2,
    ):
        """
        Initialize confidence scorer.

        Args:
            char_confidence_weight: Weight for character-level confidence
            format_weight: Weight for format validation
            checksum_weight: Weight for checksum validation
            consistency_weight: Weight for consistency checks
        """
        self.weights = {
            "char_confidence": char_confidence_weight,
            "format": format_weight,
            "checksum": checksum_weight,
            "consistency": consistency_weight,
        }

        logger.info(f"Initialized confidence scorer with weights: {self.weights}")

    def calculate_overall_confidence(
        self,
        ocr_result: Dict[str, Any],
        check_fields: Optional[Dict[str, Any]] = None,
        multi_run_results: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive confidence score.

        Args:
            ocr_result: OCR result with confidence scores
            check_fields: Extracted check fields
            multi_run_results: Optional results from multiple OCR runs

        Returns:
            Dictionary with detailed confidence breakdown
        """
        scores = {}

        # 1. Character-level confidence (from OCR engine)
        scores["char_confidence"] = self._score_char_confidence(ocr_result)

        # 2. Format validation (does data match expected patterns?)
        scores["format"] = (
            self._score_format_validation(check_fields) if check_fields else 0.0
        )

        # 3. Checksum validation (routing number validation)
        scores["checksum"] = (
            self._score_checksum_validation(check_fields) if check_fields else 0.0
        )

        # 4. Consistency (multiple OCR runs produce same result?)
        scores["consistency"] = (
            self._score_consistency(multi_run_results) if multi_run_results else 0.0
        )

        # Calculate weighted final confidence
        final_confidence = sum(scores[key] * self.weights[key] for key in scores.keys())

        # Field-level confidence
        field_confidences = (
            self._calculate_field_confidences(check_fields) if check_fields else {}
        )

        return {
            "overall_confidence": final_confidence,
            "component_scores": scores,
            "field_confidences": field_confidences,
            "weights": self.weights,
            "recommendation": self._get_recommendation(
                final_confidence, field_confidences
            ),
        }

    def _score_char_confidence(self, ocr_result: Dict[str, Any]) -> float:
        """
        Score based on character-level OCR confidence.

        Args:
            ocr_result: OCR result dictionary

        Returns:
            Character confidence score (0-100)
        """
        if "confidence" in ocr_result:
            return float(ocr_result["confidence"])

        # Calculate from line confidences if available
        if "lines" in ocr_result and ocr_result["lines"]:
            confidences = [
                line.get("confidence", 0)
                for line in ocr_result["lines"]
                if "confidence" in line
            ]

            if confidences:
                return sum(confidences) / len(confidences)

        # No confidence data available
        return 50.0

    def _score_format_validation(self, check_fields: Dict[str, Any]) -> float:
        """
        Score based on field format validation.

        Args:
            check_fields: Extracted check fields

        Returns:
            Format validation score (0-100)
        """
        if not check_fields:
            return 0.0

        total_fields = 0
        valid_fields = 0

        # Validate routing number (9 digits)
        if check_fields.get("routing_number"):
            total_fields += 1
            if self._validate_routing_format(check_fields["routing_number"]):
                valid_fields += 1

        # Validate account number (4-17 digits typically)
        if check_fields.get("account_number"):
            total_fields += 1
            if self._validate_account_format(check_fields["account_number"]):
                valid_fields += 1

        # Validate check number (1-6 digits)
        if check_fields.get("check_number"):
            total_fields += 1
            if self._validate_check_number_format(check_fields["check_number"]):
                valid_fields += 1

        # Validate amount (positive number with cents)
        if check_fields.get("amount"):
            total_fields += 1
            if self._validate_amount_format(check_fields["amount"]):
                valid_fields += 1

        # Validate date
        if check_fields.get("date"):
            total_fields += 1
            if self._validate_date_format(check_fields["date"]):
                valid_fields += 1

        # Calculate percentage
        if total_fields == 0:
            return 0.0

        return (valid_fields / total_fields) * 100.0

    def _score_checksum_validation(self, check_fields: Dict[str, Any]) -> float:
        """
        Score based on routing number checksum validation.

        Args:
            check_fields: Extracted check fields

        Returns:
            Checksum validation score (0-100)
        """
        if not check_fields or not check_fields.get("routing_number"):
            return 0.0

        routing = check_fields["routing_number"]

        if self.validate_routing_checksum(routing):
            return 100.0
        else:
            return 0.0

    def _score_consistency(self, multi_run_results: List[Dict[str, Any]]) -> float:
        """
        Score based on consistency across multiple OCR runs.

        Args:
            multi_run_results: Results from multiple OCR passes

        Returns:
            Consistency score (0-100)
        """
        if not multi_run_results or len(multi_run_results) < 2:
            return 50.0  # Neutral score if no comparison data

        # Compare key fields across runs
        fields_to_compare = [
            "routing_number",
            "account_number",
            "check_number",
            "amount",
        ]

        total_comparisons = 0
        matching_comparisons = 0

        for field in fields_to_compare:
            # Extract field values from all runs
            values = [
                run.get("fields", {}).get(field)
                for run in multi_run_results
                if run.get("fields", {}).get(field) is not None
            ]

            if len(values) >= 2:
                total_comparisons += 1

                # Check if all values match
                if len(set(str(v) for v in values)) == 1:
                    matching_comparisons += 1

        if total_comparisons == 0:
            return 50.0

        return (matching_comparisons / total_comparisons) * 100.0

    def _calculate_field_confidences(
        self, check_fields: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate confidence for individual fields.

        Args:
            check_fields: Extracted check fields

        Returns:
            Dictionary of field-level confidences
        """
        field_confidences = {}

        # Routing number confidence
        if check_fields.get("routing_number"):
            routing = check_fields["routing_number"]
            score = 0.0

            if self._validate_routing_format(routing):
                score += 50.0
            if self.validate_routing_checksum(routing):
                score += 50.0

            field_confidences["routing_number"] = score

        # Account number confidence
        if check_fields.get("account_number"):
            account = check_fields["account_number"]
            if self._validate_account_format(account):
                field_confidences["account_number"] = 80.0
            else:
                field_confidences["account_number"] = 30.0

        # Check number confidence
        if check_fields.get("check_number"):
            check_num = check_fields["check_number"]
            if self._validate_check_number_format(check_num):
                field_confidences["check_number"] = 90.0
            else:
                field_confidences["check_number"] = 40.0

        # Amount confidence
        if check_fields.get("amount"):
            amount = check_fields["amount"]
            if self._validate_amount_format(amount):
                field_confidences["amount"] = 85.0
            else:
                field_confidences["amount"] = 35.0

        # Date confidence
        if check_fields.get("date"):
            date = check_fields["date"]
            if self._validate_date_format(date):
                field_confidences["date"] = 80.0
            else:
                field_confidences["date"] = 30.0

        return field_confidences

    # ===== Format Validators =====

    def _validate_routing_format(self, routing: str) -> bool:
        """Validate routing number format (9 digits)."""
        return bool(re.match(r"^\d{9}$", str(routing)))

    def _validate_account_format(self, account: str) -> bool:
        """Validate account number format (4-17 digits)."""
        return bool(re.match(r"^\d{4,17}$", str(account)))

    def _validate_check_number_format(self, check_num: str) -> bool:
        """Validate check number format (1-6 digits)."""
        return bool(re.match(r"^\d{1,6}$", str(check_num)))

    def _validate_amount_format(self, amount: Any) -> bool:
        """Validate amount format."""
        try:
            value = float(amount)
            return value > 0 and value < 1000000  # Reasonable check amount
        except (ValueError, TypeError):
            return False

    def _validate_date_format(self, date: str) -> bool:
        """Validate date format."""
        # Common formats: MM/DD/YYYY, YYYY-MM-DD, etc.
        patterns = [
            r"^\d{1,2}/\d{1,2}/\d{2,4}$",
            r"^\d{4}-\d{2}-\d{2}$",
            r"^\d{1,2}-\d{1,2}-\d{2,4}$",
        ]

        return any(re.match(pattern, str(date)) for pattern in patterns)

    # ===== Checksum Validation =====

    @staticmethod
    def validate_routing_checksum(routing: str) -> bool:
        """
        Validate ABA routing number using modulo 10 checksum.

        Algorithm:
        (3×(d1+d4+d7) + 7×(d2+d5+d8) + 1×(d3+d6+d9)) mod 10 = 0

        Args:
            routing: 9-digit routing number

        Returns:
            True if checksum is valid
        """
        if not routing or len(str(routing)) != 9:
            return False

        try:
            routing_str = str(routing)
            checksum = (
                3 * (int(routing_str[0]) + int(routing_str[3]) + int(routing_str[6]))
                + 7 * (int(routing_str[1]) + int(routing_str[4]) + int(routing_str[7]))
                + 1 * (int(routing_str[2]) + int(routing_str[5]) + int(routing_str[8]))
            ) % 10

            return checksum == 0

        except (ValueError, IndexError):
            return False

    # ===== Recommendation =====

    def _get_recommendation(
        self, overall_confidence: float, field_confidences: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Get processing recommendation based on confidence scores.

        Args:
            overall_confidence: Overall confidence score
            field_confidences: Individual field confidences

        Returns:
            Recommendation dictionary
        """
        # Check for critical fields with low confidence
        critical_fields = ["routing_number", "amount", "check_number"]
        low_critical_fields = [
            field
            for field in critical_fields
            if field in field_confidences and field_confidences[field] < 70
        ]

        if overall_confidence >= 75:
            if low_critical_fields:
                action = "manual_review"
                reason = f"Low confidence in critical fields: {', '.join(low_critical_fields)}"
            else:
                action = "auto_process"
                reason = "High confidence in all fields"
        else:
            action = "manual_review"
            reason = "Low overall confidence"

        return {
            "action": action,
            "reason": reason,
            "low_confidence_fields": low_critical_fields,
        }

    def should_manual_review(self, confidence_result: Dict[str, Any]) -> bool:
        """
        Determine if manual review is required.

        Args:
            confidence_result: Result from calculate_overall_confidence()

        Returns:
            True if manual review required
        """
        recommendation = confidence_result.get("recommendation", {})
        return recommendation.get("action") == "manual_review"
