"""
Hybrid ID generation system for checks.
Generates both short hash IDs and reversible shareable IDs.
"""

import hashlib
import base64
import json
from typing import Dict, Any
from loguru import logger


class CheckIDGenerator:
    """Generates unique IDs for checks."""

    def __init__(self, database=None):
        """
        Initialize ID generator.

        Args:
            database: Database instance for storing mappings
        """
        self.database = database

    def generate_ids(self, check_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate both primary and shareable IDs.

        Args:
            check_data: Check information

        Returns:
            Dictionary with primary_id and shareable_id
        """
        primary_id = self.generate_hash_id(check_data)
        shareable_id = self.generate_reversible_id(check_data)

        logger.debug(
            f"Generated IDs - primary: {primary_id}, shareable: {shareable_id}"
        )

        return {"primary_id": primary_id, "shareable_id": shareable_id}

    def generate_hash_id(self, check_data: Dict[str, Any]) -> str:
        """
        Generate short hash ID for database primary key.

        Args:
            check_data: Check information

        Returns:
            Short URL-safe hash (12 characters)

        Notes:
            We truncate the SHA256 output to 12 bytes to keep IDs short,
            which reduces the raw cryptographic collision resistance (truncated
            output has a lower birthday-bound). To ensure practical uniqueness
            within the application we check the database for collisions and
            deterministically retry with a small counter appended until a free
            ID is found (or raise after many attempts). If no database is
            provided we return the truncated hash but cannot guarantee uniqueness.
        """
        # Base of identifying fields (deterministic)
        base_fields = (
            f"{check_data.get('routing_number', '')}"
            f"{check_data.get('account_number', '')}"
            f"{check_data.get('check_number', '')}"
            f"{check_data.get('date', '')}"
            f"{check_data.get('amount', '')}"
        )

        # Deterministically try appended counters to resolve any collisions
        counter = 0
        max_attempts = 1000

        while True:
            attempt_input = base_fields if counter == 0 else f"{base_fields}|{counter}"
            hash_full = hashlib.sha256(attempt_input.encode("utf-8")).digest()

            # Take first 12 bytes and encode as URL-safe base64
            hash_short = base64.urlsafe_b64encode(hash_full[:12]).decode().rstrip("=")

            # If no database provided we cannot check for collisions; return result
            if not self.database:
                if counter > 0:
                    logger.warning(
                        "No database to check for collisions; returning truncated hash (possible collision)."
                    )
                return hash_short

            # Check DB for existing mapping; assume get_check_by_mapping returns None if not found
            existing = self.database.get_check_by_mapping(hash_short)
            if not existing:
                return hash_short

            counter += 1
            if counter >= max_attempts:
                raise RuntimeError(
                    "Unable to generate unique hash ID after multiple attempts"
                )

    def generate_reversible_id(self, check_data: Dict[str, Any]) -> str:
        """
        Generate reversible ID for sharing in URLs/emails.

        Args:
            check_data: Check information

        Returns:
            Base64-encoded JSON string
        """
        # Create compact data structure
        id_data = {
            "cn": check_data.get("check_number", ""),  # Check number
            "amt": str(check_data.get("amount", "")),  # Amount
            "dt": check_data.get("date", ""),  # Date
            "cid": check_data.get("customer_id", ""),  # Customer ID
            "rn": (
                check_data.get("routing_number", "")[-4:]
                if check_data.get("routing_number")
                else ""
            ),  # Last 4 of routing
        }

        # Convert to compact JSON
        json_str = json.dumps(id_data, separators=(",", ":"))

        # Base64 encode
        encoded = base64.urlsafe_b64encode(json_str.encode()).decode()

        # Remove padding for shorter URL
        encoded = encoded.rstrip("=")

        return encoded

    def generate_doc_number(self, check_data: Dict[str, Any]) -> str:
        """
        Generate deterministic DocNumber (<=21 chars) for QuickBooks.

        Args:
            check_data: Check information

        Returns:
            Uppercase hexadecimal string (21 chars) derived from key fields
        """
        hash_input = (
            f"{check_data.get('routing_number', '')}|"
            f"{check_data.get('account_number', '')}|"
            f"{check_data.get('check_number', '')}|"
            f"{check_data.get('date', '')}|"
            f"{check_data.get('amount', '')}"
        ).encode("utf-8")

        doc_hash = hashlib.sha1(hash_input).hexdigest().upper()
        return doc_hash[:21]

    def decode_shareable_id(self, encoded_id: str) -> Dict[str, Any]:
        """
        Decode shareable ID back to check data.

        Args:
            encoded_id: Base64-encoded ID

        Returns:
            Dictionary with check data
        """
        try:
            # Add back padding
            padding = 4 - (len(encoded_id) % 4)
            if padding != 4:
                encoded_id += "=" * padding

            # Decode
            json_str = base64.urlsafe_b64decode(encoded_id).decode()
            check_data = json.loads(json_str)

            # Expand field names
            expanded = {
                "check_number": check_data.get("cn"),
                "amount": (
                    float(check_data.get("amt")) if check_data.get("amt") else None
                ),
                "date": check_data.get("dt"),
                "customer_id": check_data.get("cid"),
                "routing_last4": check_data.get("rn"),
            }

            return expanded

        except Exception as e:
            logger.error(f"Failed to decode shareable ID: {e}")
            raise ValueError(f"Invalid shareable ID: {encoded_id}")

    def store_mapping(self, primary_id: str, check_data: Dict[str, Any]):
        """
        Store ID mapping in database.

        Args:
            primary_id: Primary hash ID
            check_data: Full check data
        """
        if not self.database:
            logger.warning("No database available to store mapping")
            return

        mapping_data = {
            "check_id": primary_id,
            "routing_number": check_data.get("routing_number", ""),
            "account_number": check_data.get("account_number", ""),
            "check_number": check_data.get("check_number", ""),
            "amount": check_data.get("amount", 0),
            "date": check_data.get("date", ""),
            "customer_id": check_data.get("customer_id", ""),
        }

        self.database.insert_check_id_mapping(mapping_data)
        logger.debug(f"Stored ID mapping for {primary_id}")

    def lookup_by_id(self, check_id: str) -> Dict[str, Any]:
        """
        Lookup check data by ID.

        Args:
            check_id: Check ID (primary or shareable)

        Returns:
            Check data from database
        """
        if not self.database:
            raise ValueError("Database required for ID lookup")

        # Try primary ID lookup first
        mapping = self.database.get_check_by_mapping(check_id)

        if mapping:
            return mapping

        # Try decoding as shareable ID
        try:
            decoded = self.decode_shareable_id(check_id)
            return decoded
        except:
            raise ValueError(f"Check ID not found: {check_id}")
