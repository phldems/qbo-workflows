"""
Utilities for mapping check extraction data to QuickBooks formats.

This module contains shared functions used by multiple scripts for converting
check data extracted from OCR into QuickBooks-compatible formats.
"""

import json
from datetime import datetime
from typing import Dict, Optional, Any


def convert_date_format(date_str: str) -> str:
    """
    Convert MM/DD/YYYY to YYYY-MM-DD for QuickBooks.

    Args:
        date_str: Date in MM/DD/YYYY format

    Returns:
        Date in YYYY-MM-DD format. Returns current date if parsing fails.
    """
    if not date_str:
        return datetime.now().strftime("%Y-%m-%d")

    try:
        dt = datetime.strptime(date_str, "%m/%d/%Y")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return datetime.now().strftime("%Y-%m-%d")


def build_private_note(check_data: Dict[str, Any], mode: str = "comprehensive") -> str:
    """
    Build private note from check extraction data.

    Args:
        check_data: Extracted check data dictionary
        mode: 'comprehensive' for full details or 'minimal' for memo+MICR only

    Returns:
        Formatted private note string
    """
    if mode == "minimal":
        return _build_minimal_note(check_data)
    return _build_comprehensive_note(check_data)


def _build_minimal_note(check_data: Dict[str, Any]) -> str:
    """Build minimal private note with memo and MICR line only."""
    note_parts = []

    if check_data.get("memo"):
        note_parts.append(f"Memo: {check_data['memo']}")

    if check_data.get("micr_line"):
        note_parts.append(f"MICR: {check_data['micr_line']}")

    return "\n".join(note_parts) if note_parts else ""


def _build_comprehensive_note(check_data: Dict[str, Any]) -> str:
    """Build comprehensive private note with all check details."""
    note_parts = []

    # Header
    check_num = check_data.get("check_number") or "Unknown"
    payor_name = (
        check_data.get("payor_name", {}).get("full_name")
        if check_data.get("payor_name")
        else None
    )

    note_parts.append(
        f"Check #{check_num}" + (f" from {payor_name}" if payor_name else "")
    )
    note_parts.append("")

    # Payor Information
    if check_data.get("payor_name") or check_data.get("payor_address"):
        note_parts.append("Payor Information:")

        if check_data.get("payor_name"):
            pn = check_data["payor_name"]
            note_parts.append(f"Name: {pn.get('full_name') or 'N/A'}")
            if pn.get("is_joint"):
                note_parts.append("  (Joint Account)")

        if check_data.get("payor_address"):
            addr = check_data["payor_address"]
            if addr.get("street"):
                note_parts.append(f"Address: {addr['street']}")
            if addr.get("city") and addr.get("state") and addr.get("zip"):
                note_parts.append(
                    f"City/State/ZIP: {addr['city']}, {addr['state']} {addr['zip']}"
                )
            if addr.get("country") and addr["country"] != "USA":
                note_parts.append(f"Country: {addr['country']}")

        note_parts.append("")

    # Bank Information
    if (
        check_data.get("bank_name")
        or check_data.get("routing_number")
        or check_data.get("account_number")
    ):
        note_parts.append("Bank Information:")

        if check_data.get("bank_name"):
            note_parts.append(f"Bank Name: {check_data['bank_name']}")
        if check_data.get("routing_number"):
            note_parts.append(f"Routing Number: {check_data['routing_number']}")
        if check_data.get("account_number"):
            note_parts.append(f"Account Number: {check_data['account_number']}")

        note_parts.append("")

    # Memo
    if check_data.get("memo"):
        note_parts.append(f"Memo: {check_data['memo']}")
        note_parts.append("")

    # Amount Details
    note_parts.append("Amount Details:")
    note_parts.append(f"Numeric: ${check_data.get('amount_numeric', 0):,.2f}")
    if check_data.get("amount_written"):
        note_parts.append(f"Written: {check_data['amount_written']}")

    amounts_match = check_data.get("amounts_match")
    if amounts_match is True:
        note_parts.append("Amounts Match: ✓")
    elif amounts_match is False:
        note_parts.append("Amounts Match: ✗ WARNING - MISMATCH!")

    note_parts.append("")

    # Validation
    note_parts.append("Validation:")

    if check_data.get("signature_present"):
        note_parts.append("Signature Present: ✓")
    else:
        note_parts.append("Signature Present: ✗")

    if check_data.get("check_condition"):
        note_parts.append(f"Check Condition: {check_data['check_condition']}")

    if check_data.get("check_age_days") is not None:
        age = check_data["check_age_days"]
        if age < 0:
            note_parts.append(f"Check Age: {age} days (POST-DATED)")
        else:
            note_parts.append(f"Check Age: {age} days")

    if check_data.get("is_post_dated"):
        note_parts.append("Post-Dated: ✓ WARNING")

    if check_data.get("is_stale_dated"):
        note_parts.append("Stale-Dated: ✓ WARNING - >180 days old")

    if check_data.get("requires_manual_review"):
        note_parts.append("Requires Manual Review: ✓")

    note_parts.append("")

    # Confidence
    if check_data.get("confidence") is not None:
        note_parts.append(f"Confidence: {check_data['confidence']}%")
        note_parts.append("")

    # Footer
    note_parts.append("Extracted via Vertex AI Gemini Pro")

    return "\n".join(note_parts)


def map_check_to_salesreceipt(
    check_data: Dict[str, Any],
    customer_id: Optional[str] = None,
    payment_method_id: Optional[str] = None,
    deposit_account_id: Optional[str] = None,
    income_item_id: Optional[str] = None,
    note_mode: str = "comprehensive",
) -> Dict[str, Any]:
    """
    Map extracted check data to QuickBooks SalesReceipt payload.

    Args:
        check_data: Extracted check data from Vertex AI
        customer_id: QuickBooks customer ID (if already known)
        payment_method_id: QBO payment method ID
        deposit_account_id: QBO deposit account ID
        income_item_id: QBO income item ID
        note_mode: 'comprehensive' for full details or 'minimal' for memo+MICR only

    Returns:
        Dictionary formatted for QuickBooks SalesReceipt POST

    Raises:
        ValueError: If required IDs are not provided
    """
    if not payment_method_id or not deposit_account_id or not income_item_id:
        raise ValueError(
            "payment_method_id, deposit_account_id, and income_item_id are required"
        )

    # Get amount
    amount = check_data.get("amount_numeric", 0)

    # Build payload
    payload = {
        "DocNumber": check_data.get("check_number")
        or f"CHK-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "TxnDate": convert_date_format(check_data.get("date")),
        "Line": [
            {
                "Amount": amount,
                "DetailType": "SalesItemLineDetail",
                "SalesItemLineDetail": {
                    "ItemRef": {"value": income_item_id},
                    "Qty": 1,
                    "UnitPrice": amount,
                },
                "Description": "Check payment",
            }
        ],
        "TotalAmt": amount,
        "PaymentMethodRef": {"value": payment_method_id},
        "DepositToAccountRef": {"value": deposit_account_id},
        "PrivateNote": build_private_note(check_data, mode=note_mode),
    }

    # Add customer reference
    if customer_id:
        payload["CustomerRef"] = {"value": customer_id}
    else:
        # Use payee name to find/create customer
        payee = check_data.get("payee")
        if payee:
            payload["CustomerRef"] = {"name": payee}
        else:
            # Fallback to payor name
            payor_full_name = (
                check_data.get("payor_name", {}).get("full_name")
                if check_data.get("payor_name")
                else None
            )
            if payor_full_name:
                payload["CustomerRef"] = {"name": payor_full_name}
            else:
                payload["CustomerRef"] = {"name": "Unknown Customer"}

    return payload


def build_email_summary(check_data: Dict[str, Any], settings: Any) -> str:
    """
    Build email summary for user notification.

    Args:
        check_data: Extracted check data dictionary
        settings: Application settings for referencing configured IDs

    Returns:
        Formatted email summary string
    """
    lines = []

    lines.append("=" * 80)
    lines.append("CHECK PROCESSED SUCCESSFULLY")
    lines.append("=" * 80)
    lines.append("")

    # Key Information
    lines.append("Key Information:")
    lines.append("-" * 80)
    lines.append(f"Check Number: {check_data.get('check_number', 'N/A')}")
    lines.append(f"Date: {check_data.get('date', 'N/A')}")
    lines.append(f"Amount: ${check_data.get('amount_numeric', 0):,.2f}")
    lines.append(f"Payee: {check_data.get('payee', 'N/A')}")
    lines.append(f"Payor: {check_data.get('payor_name', {}).get('full_name', 'N/A')}")
    lines.append("")

    # Bank Information
    lines.append("Bank Information:")
    lines.append("-" * 80)
    lines.append(f"Bank: {check_data.get('bank_name', 'N/A')}")
    lines.append(f"Routing: {check_data.get('routing_number', 'N/A')}")
    lines.append(f"Account: {check_data.get('account_number', 'N/A')}")
    lines.append("")

    # Validation Status
    lines.append("Validation Status:")
    lines.append("-" * 80)

    amounts_match = check_data.get("amounts_match")
    if amounts_match is True:
        lines.append("Amounts Match: ✓ Yes")
    elif amounts_match is False:
        lines.append("Amounts Match: ✗ No - REQUIRES REVIEW")
    else:
        lines.append("Amounts Match: Unknown")

    if check_data.get("is_post_dated"):
        lines.append("Post-Dated: ⚠ WARNING - Check is dated in the future")

    if check_data.get("is_stale_dated"):
        lines.append("Stale-Dated: ⚠ WARNING - Check is >180 days old")

    if check_data.get("requires_manual_review"):
        lines.append("Manual Review: ⚠ Required")

    lines.append(f"Confidence: {check_data.get('confidence', 0)}%")
    lines.append("")

    # QuickBooks Status
    lines.append("QuickBooks Status:")
    lines.append("-" * 80)
    lines.append("SalesReceipt: Ready to create")
    lines.append(f"Deposit Account ID: {settings.qbo_deposit_account_id}")
    lines.append(f"Payment Method ID: {settings.qbo_payment_method_id}")
    lines.append("")

    return "\n".join(lines)
