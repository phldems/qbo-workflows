"""
QuickBooks Online SalesReceipt creation and management.
Handles creating sales receipts and attaching check images.
"""

import json
import requests
from typing import Dict, Any, Optional
from pathlib import Path
from loguru import logger
from requests_toolbelt.multipart.encoder import MultipartEncoder


class QuickBooksSalesReceipt:
    """Manages QuickBooks SalesReceipt operations."""

    def __init__(self, base_url: str, auth_manager):
        """
        Initialize SalesReceipt manager.

        Args:
            base_url: QBO API base URL
            auth_manager: QuickBooksAuth instance
        """
        self.base_url = base_url
        self.auth = auth_manager
        logger.info(f"Initialized QuickBooks SalesReceipt manager: {base_url}")

    def create_sales_receipt(
        self,
        check_data: Dict[str, Any],
        customer_id: str,
        payment_method_id: str,
        deposit_account_id: str,
        income_item_id: str,
    ) -> Dict[str, Any]:
        """
        Create a SalesReceipt in QuickBooks.

        Args:
            check_data: Check information
            customer_id: QBO Customer ID
            payment_method_id: Payment method ID for checks
            deposit_account_id: Deposit account ID
            income_item_id: Income item/account ID

        Returns:
            Created SalesReceipt data
        """
        realm_id = self.auth.get_realm_id()
        if not realm_id:
            raise ValueError("No Realm ID found. Please authorize first.")

        access_token = self.auth.get_valid_access_token()

        url = f"{self.base_url}/company/{realm_id}/salesreceipt"

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        params = {"minorversion": "70"}

        # Build SalesReceipt object
        sales_receipt = {
            "CustomerRef": {"value": customer_id},
            "TxnDate": self._convert_date_format(
                check_data.get("date")
            ),  # Convert to YYYY-MM-DD
            "DocNumber": check_data.get("doc_number"),
            "PaymentRefNum": check_data.get("check_number"),
            "PaymentMethodRef": {"value": payment_method_id},
            "DepositToAccountRef": {"value": deposit_account_id},
            "Line": [
                {
                    "Amount": check_data.get("amount"),
                    "DetailType": "SalesItemLineDetail",
                    "SalesItemLineDetail": {"ItemRef": {"value": income_item_id}},
                    "Description": f"Check #{check_data.get('check_number')}",
                }
            ],
            "PrivateNote": self._build_private_note(check_data),
        }

        bill_addr = self._build_bill_address(check_data.get("payor_address"))
        if bill_addr:
            sales_receipt["BillAddr"] = bill_addr

        logger.info(
            f"Creating SalesReceipt for check #{check_data.get('check_number')}"
        )

        try:
            response = requests.post(
                url, headers=headers, params=params, json=sales_receipt, timeout=30
            )

            response.raise_for_status()

            result = response.json()
            sales_receipt_data = result.get("SalesReceipt", {})

            logger.info(
                f"Created SalesReceipt ID: {sales_receipt_data.get('Id')} "
                f"(TxnNumber: {sales_receipt_data.get('DocNumber')})"
            )

            return sales_receipt_data

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to create SalesReceipt: {e}")
            if hasattr(e, "response") and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise

    def attach_image(
        self, salesreceipt_id: str, image_path: str, filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Attach check image to SalesReceipt.

        Args:
            salesreceipt_id: SalesReceipt ID to attach to
            image_path: Path to image file
            filename: Optional custom filename

        Returns:
            Attachment data
        """
        realm_id = self.auth.get_realm_id()
        if not realm_id:
            raise ValueError("No Realm ID found")

        access_token = self.auth.get_valid_access_token()

        url = f"{self.base_url}/company/{realm_id}/upload"

        # Determine filename and content type
        if not filename:
            filename = Path(image_path).name

        content_type = self._get_content_type(filename)

        # Prepare metadata
        metadata = {
            "AttachableRef": [
                {"EntityRef": {"type": "SalesReceipt", "value": str(salesreceipt_id)}}
            ],
            "FileName": filename,
            "ContentType": content_type,
        }

        logger.info(f"Attaching {filename} to SalesReceipt {salesreceipt_id}")

        try:
            # Read image file
            with open(image_path, "rb") as f:
                image_data = f.read()

            # Create multipart form data
            multipart_data = MultipartEncoder(
                fields={
                    "file_metadata_0": (
                        "metadata",
                        json.dumps(metadata),
                        "application/json",
                    ),
                    "file_content_0": (filename, image_data, content_type),
                }
            )

            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": multipart_data.content_type,
                "Accept": "application/json",
            }

            params = {"minorversion": "70"}

            response = requests.post(
                url, headers=headers, params=params, data=multipart_data, timeout=60
            )

            response.raise_for_status()

            result = response.json()
            attachment = result.get("AttachableResponse", [{}])[0].get("Attachable", {})

            logger.info(
                f"Successfully attached image (Attachment ID: {attachment.get('Id')})"
            )

            return attachment

        except Exception as e:
            logger.error(f"Failed to attach image: {e}")
            if hasattr(e, "response") and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise

    def find_or_create_customer(
        self, customer_name: str, address: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Find existing customer by name or create new one.

        Args:
            customer_name: Customer display name

        Returns:
            Customer ID
        """
        realm_id = self.auth.get_realm_id()
        access_token = self.auth.get_valid_access_token()

        # Query for existing customer
        query = f"SELECT * FROM Customer WHERE DisplayName = '{customer_name}'"
        url = f"{self.base_url}/company/{realm_id}/query"

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
        }

        params = {"query": query, "minorversion": "70"}

        logger.info(f"Searching for customer: {customer_name}")

        try:
            response = requests.get(url, headers=headers, params=params, timeout=30)
            response.raise_for_status()

            result = response.json()
            customers = result.get("QueryResponse", {}).get("Customer", [])

            if customers:
                customer_id = customers[0]["Id"]
                logger.info(f"Found existing customer: {customer_id}")
                return customer_id

            # Create new customer
            logger.info(f"Creating new customer: {customer_name}")
            return self._create_customer(customer_name, address)

        except Exception as e:
            logger.error(f"Error finding/creating customer: {e}")
            raise

    def _create_customer(
        self, customer_name: str, address: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new customer."""
        realm_id = self.auth.get_realm_id()
        access_token = self.auth.get_valid_access_token()

        url = f"{self.base_url}/company/{realm_id}/customer"

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        params = {"minorversion": "70"}

        # Parse name
        name_parts = customer_name.split()
        given_name = name_parts[0] if name_parts else customer_name
        family_name = name_parts[-1] if len(name_parts) > 1 else ""

        customer_data = {
            "DisplayName": customer_name,
            "GivenName": given_name,
            "FamilyName": family_name,
        }

        bill_addr = self._build_bill_address(address)
        if bill_addr:
            customer_data["BillAddr"] = bill_addr

        response = requests.post(
            url, headers=headers, params=params, json=customer_data, timeout=30
        )

        response.raise_for_status()

        result = response.json()
        customer_id = result["Customer"]["Id"]

        logger.info(f"Created new customer: {customer_id}")
        return customer_id

    def _get_content_type(self, filename: str) -> str:
        """Get MIME content type from filename."""
        ext = Path(filename).suffix.lower()

        content_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".pdf": "application/pdf",
            ".tiff": "image/tiff",
            ".tif": "image/tiff",
        }

        return content_types.get(ext, "application/octet-stream")

    def get_salesreceipt_url(self, salesreceipt_id: str) -> str:
        """
        Get web URL for SalesReceipt.

        Args:
            salesreceipt_id: SalesReceipt ID

        Returns:
            URL to view SalesReceipt in QBO web interface
        """
        realm_id = self.auth.get_realm_id()

        if self.auth.environment == "sandbox":
            base = "https://app.sandbox.qbo.intuit.com"
        else:
            base = "https://app.qbo.intuit.com"

        return f"{base}/app/salesreceipt?txnId={salesreceipt_id}&companyid={realm_id}"

    def _build_private_note(self, check_data: Dict[str, Any]) -> str:
        """
        Build private note from memo and MICR line.

        Args:
            check_data: Check information dictionary

        Returns:
            Formatted private note string
        """
        parts = []

        # Add memo if present
        if check_data.get("memo"):
            parts.append(f"Memo: {check_data['memo']}")

        # Add MICR line if present
        if check_data.get("micr_line"):
            parts.append(f"MICR: {check_data['micr_line']}")

        return "\n".join(parts) if parts else ""

    def _build_bill_address(
        self, payor_address: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Convert OCR payor address structure into the QBO BillAddr schema.
        """
        if not payor_address or not isinstance(payor_address, dict):
            return None

        street = payor_address.get("street")
        city = payor_address.get("city")
        state = payor_address.get("state")
        postal_code = payor_address.get("zip")
        country = payor_address.get("country")

        bill_addr: Dict[str, Any] = {}
        if street:
            bill_addr["Line1"] = street
        if city:
            bill_addr["City"] = city
        if state:
            bill_addr["CountrySubDivisionCode"] = state
        if postal_code:
            bill_addr["PostalCode"] = postal_code
        if country:
            bill_addr["Country"] = country

        return bill_addr or None

    def _convert_date_format(self, date_str: Optional[str]) -> str:
        """
        Convert date from MM/DD/YYYY to YYYY-MM-DD format.

        Args:
            date_str: Date string in MM/DD/YYYY format

        Returns:
            Date string in YYYY-MM-DD format
        """
        from datetime import datetime

        if not date_str:
            return datetime.now().strftime("%Y-%m-%d")

        try:
            # Try parsing MM/DD/YYYY format
            dt = datetime.strptime(date_str, "%m/%d/%Y")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            try:
                # Try parsing YYYY-MM-DD format (already correct)
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                return date_str
            except ValueError:
                # Fallback to current date
                logger.warning(f"Invalid date format: {date_str}, using current date")
                return datetime.now().strftime("%Y-%m-%d")
