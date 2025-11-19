#!/usr/bin/env python3
"""
Test QuickBooks GET request for SalesReceipt

This script tests retrieving a SalesReceipt object from QuickBooks
using the robust API client with error handling and retry logic.

Usage:
    python tests/manual/test_qbo_get.py <salesreceipt_id>

Example:
    python tests/manual/test_qbo_get.py 851
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# Add src to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.integrations.quickbooks.auth import QuickBooksAuth
from src.integrations.quickbooks.api_client import (
    QuickBooksAPIClient,
    QuickBooksAPIError,
)
from src.utils.database import Database

# Load environment variables
load_dotenv()

# Configuration
CLIENT_ID = os.getenv("QBO_CLIENT_ID")
CLIENT_SECRET = os.getenv("QBO_CLIENT_SECRET")
REDIRECT_URI = os.getenv("QBO_REDIRECT_URI", "http://localhost:8000/oauth/callback")
ENVIRONMENT = os.getenv("QBO_ENVIRONMENT", "production")
REALM_ID = os.getenv("QBO_REALM_ID")
REFRESH_TOKEN = os.getenv("QBO_REFRESH_TOKEN")

# API base URL
if ENVIRONMENT == "sandbox":
    BASE_URL = "https://sandbox-quickbooks.api.intuit.com/v3"
else:
    BASE_URL = "https://quickbooks.api.intuit.com/v3"

# Configure logger to show debug info
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="DEBUG",
)


def get_salesreceipt(salesreceipt_id: str, api_client: QuickBooksAPIClient):
    """Get a SalesReceipt by ID using robust API client"""
    print(f"\n📡 Making GET request to QuickBooks...")
    print(f"Endpoint: /salesreceipt/{salesreceipt_id}")
    print()

    try:
        # Use the robust API client with automatic retries and error handling
        response = api_client.get(f"salesreceipt/{salesreceipt_id}")

        # Extract intuit_tid for troubleshooting
        intuit_tid = response.get("_intuit_tid", "N/A")

        # Get SalesReceipt data
        salesreceipt = response.get("SalesReceipt", {})

        print("=" * 70)
        print("✅ SalesReceipt Retrieved Successfully!")
        print("=" * 70)
        print()
        print(f"intuit_tid: {intuit_tid}")
        print()

        # Display key information
        print("Key Information:")
        print("-" * 70)
        print(f"  ID:              {salesreceipt.get('Id')}")
        print(f"  Doc Number:      {salesreceipt.get('DocNumber')}")
        print(f"  Transaction Date: {salesreceipt.get('TxnDate')}")
        print(f"  Total Amount:    ${salesreceipt.get('TotalAmt', 0):.2f}")

        customer_ref = salesreceipt.get("CustomerRef", {})
        print(f"  Customer Ref:    {customer_ref.get('name', 'N/A')}")

        payment_ref = salesreceipt.get("PaymentMethodRef", {})
        print(f"  Payment Method:  {payment_ref.get('name', 'N/A')}")
        print()

        # Display line items
        if "Line" in salesreceipt:
            print("Line Items:")
            print("-" * 70)
            for i, line in enumerate(salesreceipt["Line"], 1):
                if line.get("DetailType") == "SalesItemLineDetail":
                    detail = line.get("SalesItemLineDetail", {})
                    print(f"  {i}. {line.get('Description', 'N/A')}")
                    print(f"     Amount: ${line.get('Amount', 0):.2f}")
                    print(f"     Qty: {detail.get('Qty', 1)}")
                    print()

        # Display full JSON response
        print()
        print("=" * 70)
        print("Full JSON Response:")
        print("=" * 70)
        print(json.dumps(response, indent=2))
        print()

        return response

    except QuickBooksAPIError as e:
        print("=" * 70)
        print("❌ QuickBooks API Error")
        print("=" * 70)
        print()
        print(f"Error Message:   {str(e)}")
        print(f"Status Code:     {e.status_code}")
        print(f"intuit_tid:      {e.intuit_tid}")
        print(f"Error Code:      {e.error_code}")
        print(f"Error Detail:    {e.error_detail}")
        print()

        if e.status_code == 404:
            print("💡 Troubleshooting:")
            print(f"   - SalesReceipt ID '{salesreceipt_id}' not found")
            print(f"   - Check your REALM_ID: {REALM_ID}")
            print(f"   - Check environment: {ENVIRONMENT}")
        elif e.status_code == 401:
            print("💡 Troubleshooting:")
            print("   - Access token may be invalid or expired")
            print("   - Refresh token may be invalid")
            print("   - Try running: python get_qbo_tokens.py")

        print()
        print("Full error details logged above")
        return None

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logger.exception("Unexpected error occurred")
        return None


def main():
    print("=" * 70)
    print("QuickBooks SalesReceipt GET Test")
    print("=" * 70)
    print()

    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: python test_qbo_get.py <salesreceipt_id>")
        print()
        print("Example:")
        print("  python test_qbo_get.py 851")
        print()
        sys.exit(1)

    salesreceipt_id = sys.argv[1]

    # Check configuration
    if not all([CLIENT_ID, CLIENT_SECRET, REALM_ID, REFRESH_TOKEN]):
        print("❌ Error: Missing required environment variables")
        print()
        print("Please ensure your .env file contains:")
        print("  - QBO_CLIENT_ID")
        print("  - QBO_CLIENT_SECRET")
        print("  - QBO_REALM_ID")
        print("  - QBO_REFRESH_TOKEN")
        print()
        print("Run 'python get_qbo_tokens.py' to get your tokens.")
        sys.exit(1)

    print(f"Configuration:")
    print(f"  Environment:     {ENVIRONMENT}")
    print(f"  Realm ID:        {REALM_ID}")
    print(f"  SalesReceipt ID: {salesreceipt_id}")
    print()

    # Initialize database (for token storage)
    db = Database("./data/checks.db")

    # Store refresh token in database if not already there
    stored_token = db.get_oauth_token("quickbooks")
    if not stored_token or stored_token.get("refresh_token") != REFRESH_TOKEN:
        print("📝 Storing refresh token in database...")
        db.store_oauth_token(
            service="quickbooks",
            access_token="",  # Will be refreshed
            refresh_token=REFRESH_TOKEN,
            expires_at=(
                datetime.now() - timedelta(days=1)
            ).isoformat(),  # Force refresh
            realm_id=REALM_ID,
        )

    # Initialize auth manager
    auth_manager = QuickBooksAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        environment=ENVIRONMENT,
        database=db,
    )

    # Initialize API client with retry logic
    api_client = QuickBooksAPIClient(
        auth_manager=auth_manager,
        base_url=BASE_URL,
        realm_id=REALM_ID,
        max_retries=3,
        retry_delay=1.0,
        timeout=30,
    )

    print("✓ QuickBooks API client initialized")
    print()

    # Get SalesReceipt
    get_salesreceipt(salesreceipt_id, api_client)


if __name__ == "__main__":
    # Need datetime for token storage
    from datetime import datetime, timedelta

    main()
