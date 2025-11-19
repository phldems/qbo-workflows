#!/usr/bin/env python3
"""
List QuickBooks Accounts and Items

This script helps you find the IDs for:
- QBO_DEPOSIT_ACCOUNT_ID (bank accounts like "Undeposited Funds" or "Checking")
- QBO_INCOME_ITEM_ID (income items like "Services" or "Sales")
- QBO_PAYMENT_METHOD_ID (payment methods like "Check" or "Cash")

Usage:
    python scripts/list_qbo_entities.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

# Add src to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.integrations.quickbooks.auth import QuickBooksAuth
from src.integrations.quickbooks.api_client import (
    QuickBooksAPIClient,
    QuickBooksAPIError,
)
from src.utils.database import Database
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Configuration
CLIENT_ID = os.getenv("QBO_CLIENT_ID")
CLIENT_SECRET = os.getenv("QBO_CLIENT_SECRET")
REDIRECT_URI = os.getenv("QBO_REDIRECT_URI", "https://oauth.platform.intuit.com/op/v1")
ENVIRONMENT = os.getenv("QBO_ENVIRONMENT", "production")
REALM_ID = os.getenv("QBO_REALM_ID")
REFRESH_TOKEN = os.getenv("QBO_REFRESH_TOKEN")

# API base URL
if ENVIRONMENT == "sandbox":
    BASE_URL = "https://sandbox-quickbooks.api.intuit.com/v3"
else:
    BASE_URL = "https://quickbooks.api.intuit.com/v3"

# Configure logger
logger.remove()
logger.add(
    sys.stderr,
    format="<level>{level: <8}</level> | <level>{message}</level>",
    level="INFO",
)


def list_accounts(api_client: QuickBooksAPIClient):
    """List all QuickBooks accounts"""
    print("\n" + "=" * 70)
    print("QuickBooks Accounts")
    print("=" * 70)
    print()

    try:
        # Query for all accounts
        response = api_client.get(
            "query", params={"query": "SELECT * FROM Account WHERE Active = true"}
        )

        accounts = response.get("QueryResponse", {}).get("Account", [])

        if not accounts:
            print("No accounts found.")
            return

        # Group accounts by type
        account_types = {}
        for account in accounts:
            account_type = account.get("AccountType", "Unknown")
            if account_type not in account_types:
                account_types[account_type] = []
            account_types[account_type].append(account)

        # Display accounts by type
        for account_type in sorted(account_types.keys()):
            print(f"\n{account_type}:")
            print("-" * 70)

            for account in sorted(
                account_types[account_type], key=lambda x: x.get("Name", "")
            ):
                account_id = account.get("Id")
                name = account.get("Name")
                subaccount = account.get("SubAccount", False)
                balance = account.get("CurrentBalance", 0)

                indent = "  " if subaccount else ""
                print(
                    f"{indent}ID: {account_id:4} | {name:40} | Balance: ${balance:,.2f}"
                )

        # Highlight useful accounts for deposit
        print("\n" + "=" * 70)
        print("💡 Recommended for QBO_DEPOSIT_ACCOUNT_ID:")
        print("=" * 70)

        for account in accounts:
            name_lower = account.get("Name", "").lower()
            account_type = account.get("AccountType", "")

            # Look for common deposit accounts
            if (
                "undeposited" in name_lower
                or "checking" in name_lower
                or "savings" in name_lower
                or account_type == "Bank"
            ):

                account_id = account.get("Id")
                name = account.get("Name")
                print(f"  ID: {account_id:4} | {name} ({account_type})")

    except QuickBooksAPIError as e:
        print(f"❌ Error fetching accounts: {e}")
        logger.error(f"API Error: {e.error_detail}")


def list_items(api_client: QuickBooksAPIClient):
    """List all QuickBooks items (products/services)"""
    print("\n" + "=" * 70)
    print("QuickBooks Items (Products & Services)")
    print("=" * 70)
    print()

    try:
        # Query for all items
        response = api_client.get(
            "query", params={"query": "SELECT * FROM Item WHERE Active = true"}
        )

        items = response.get("QueryResponse", {}).get("Item", [])

        if not items:
            print("No items found.")
            return

        # Group items by type
        item_types = {}
        for item in items:
            item_type = item.get("Type", "Unknown")
            if item_type not in item_types:
                item_types[item_type] = []
            item_types[item_type].append(item)

        # Display items by type
        for item_type in sorted(item_types.keys()):
            print(f"\n{item_type}:")
            print("-" * 70)

            for item in sorted(item_types[item_type], key=lambda x: x.get("Name", "")):
                item_id = item.get("Id")
                name = item.get("Name")
                description = item.get("Description", "")[:40]

                # Get income account reference
                income_account = ""
                if "IncomeAccountRef" in item:
                    income_account = f" → {item['IncomeAccountRef'].get('name', '')}"

                print(f"  ID: {item_id:4} | {name:30} | {description}{income_account}")

        # Highlight useful items for income
        print("\n" + "=" * 70)
        print("💡 Recommended for QBO_INCOME_ITEM_ID:")
        print("=" * 70)
        print("(Look for Service or NonInventory items you use for check payments)")
        print()

        for item in items:
            item_type = item.get("Type", "")

            # Service and NonInventory items are typically used for income
            if item_type in ["Service", "NonInventory"]:
                item_id = item.get("Id")
                name = item.get("Name")
                income_account = ""
                if "IncomeAccountRef" in item:
                    income_account = f" → {item['IncomeAccountRef'].get('name', '')}"

                print(f"  ID: {item_id:4} | {name:30} ({item_type}){income_account}")

    except QuickBooksAPIError as e:
        print(f"❌ Error fetching items: {e}")
        logger.error(f"API Error: {e.error_detail}")


def list_payment_methods(api_client: QuickBooksAPIClient):
    """List all QuickBooks payment methods"""
    print("\n" + "=" * 70)
    print("QuickBooks Payment Methods")
    print("=" * 70)
    print()

    try:
        # Query for all payment methods
        response = api_client.get(
            "query", params={"query": "SELECT * FROM PaymentMethod WHERE Active = true"}
        )

        payment_methods = response.get("QueryResponse", {}).get("PaymentMethod", [])

        if not payment_methods:
            print("No payment methods found.")
            return

        print("Available Payment Methods:")
        print("-" * 70)

        for method in sorted(payment_methods, key=lambda x: x.get("Name", "")):
            method_id = method.get("Id")
            name = method.get("Name")
            method_type = method.get("Type", "N/A")

            print(f"  ID: {method_id:4} | {name:30} | Type: {method_type}")

        # Highlight "Check" payment method
        print("\n" + "=" * 70)
        print("💡 Recommended for QBO_PAYMENT_METHOD_ID:")
        print("=" * 70)

        for method in payment_methods:
            name_lower = method.get("Name", "").lower()

            if "check" in name_lower:
                method_id = method.get("Id")
                name = method.get("Name")
                print(f"  ID: {method_id:4} | {name}")

    except QuickBooksAPIError as e:
        print(f"❌ Error fetching payment methods: {e}")
        logger.error(f"API Error: {e.error_detail}")


def main():
    print("=" * 70)
    print("QuickBooks Entity Lookup")
    print("=" * 70)
    print()

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
    print(f"  Environment: {ENVIRONMENT}")
    print(f"  Realm ID:    {REALM_ID}")
    print()

    # Initialize database
    db = Database("./data/checks.db")

    # Store refresh token in database if not already there
    stored_token = db.get_oauth_token("quickbooks")
    if not stored_token or stored_token.get("refresh_token") != REFRESH_TOKEN:
        print("Storing refresh token in database...")
        db.store_oauth_token(
            service="quickbooks",
            access_token="",
            refresh_token=REFRESH_TOKEN,
            expires_at=(datetime.now() - timedelta(days=1)).isoformat(),
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

    # Initialize API client
    api_client = QuickBooksAPIClient(
        auth_manager=auth_manager,
        base_url=BASE_URL,
        realm_id=REALM_ID,
        max_retries=3,
        retry_delay=1.0,
        timeout=30,
    )

    print("✓ QuickBooks API client initialized")

    # List entities
    list_accounts(api_client)
    list_items(api_client)
    list_payment_methods(api_client)

    # Summary
    print("\n" + "=" * 70)
    print("📝 Next Steps")
    print("=" * 70)
    print()
    print("Add these IDs to your .env file:")
    print()
    print("1. QBO_DEPOSIT_ACCOUNT_ID=<id>   # Where check money goes")
    print("   └─ Choose from Bank accounts (e.g., Undeposited Funds or Checking)")
    print()
    print("2. QBO_INCOME_ITEM_ID=<id>       # What the check payment is for")
    print("   └─ Choose from Service/NonInventory items")
    print()
    print("3. QBO_PAYMENT_METHOD_ID=<id>    # Payment method")
    print("   └─ Usually the 'Check' payment method")
    print()


if __name__ == "__main__":
    main()
