#!/usr/bin/env python3
"""Create QuickBooks SalesReceipt (Test)"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv
from loguru import logger

# Setup path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.integrations.quickbooks.auth import QuickBooksAuth
from src.integrations.quickbooks.salesreceipt import QuickBooksSalesReceipt
from src.utils.database import Database

load_dotenv()

# Config
CLIENT_ID = os.getenv("QBO_CLIENT_ID")
CLIENT_SECRET = os.getenv("QBO_CLIENT_SECRET")
REDIRECT_URI = os.getenv("QBO_REDIRECT_URI", "https://oauth.platform.intuit.com/op/v1")
ENVIRONMENT = os.getenv("QBO_ENVIRONMENT", "production")
REALM_ID = os.getenv("QBO_REALM_ID")
REFRESH_TOKEN = os.getenv("QBO_REFRESH_TOKEN")
PAYMENT_METHOD_ID = os.getenv("QBO_PAYMENT_METHOD_ID")
DEPOSIT_ACCOUNT_ID = os.getenv("QBO_DEPOSIT_ACCOUNT_ID")
INCOME_ITEM_ID = os.getenv("QBO_INCOME_ITEM_ID")

BASE_URL = f"https://{'sandbox-' if ENVIRONMENT == 'sandbox' else ''}quickbooks.api.intuit.com/v3"

logger.remove()
logger.add(
    sys.stderr,
    format="<level>{level: <8}</level> | <level>{message}</level>",
    level="INFO",
)


def main():
    print("=" * 70)
    print("QuickBooks SalesReceipt Creation Test")
    print("=" * 70)
    print()

    # Validate config
    required = {
        "QBO_CLIENT_ID": CLIENT_ID,
        "QBO_CLIENT_SECRET": CLIENT_SECRET,
        "QBO_REALM_ID": REALM_ID,
        "QBO_REFRESH_TOKEN": REFRESH_TOKEN,
        "QBO_PAYMENT_METHOD_ID": PAYMENT_METHOD_ID,
        "QBO_DEPOSIT_ACCOUNT_ID": DEPOSIT_ACCOUNT_ID,
        "QBO_INCOME_ITEM_ID": INCOME_ITEM_ID,
    }

    missing = [k for k, v in required.items() if not v]
    if missing:
        print("❌ Missing environment variables:")
        for var in missing:
            print(f"  - {var}")
        print("\nUpdate .env file. Run: python scripts/list_qbo_entities.py")
        sys.exit(1)

    print(f"Environment: {ENVIRONMENT}")
    print(f"Realm ID: {REALM_ID}")
    print()

    # Initialize
    db = Database("./data/checks.db")

    # Store refresh token if needed
    stored = db.get_oauth_token("quickbooks")
    if not stored or stored.get("refresh_token") != REFRESH_TOKEN:
        db.store_oauth_token(
            service="quickbooks",
            access_token="",
            refresh_token=REFRESH_TOKEN,
            expires_at=(datetime.now() - timedelta(days=1)).isoformat(),
            realm_id=REALM_ID,
        )

    auth = QuickBooksAuth(CLIENT_ID, CLIENT_SECRET, REDIRECT_URI, ENVIRONMENT, db)
    sr_manager = QuickBooksSalesReceipt(BASE_URL, auth)

    print("✓ Initialized\n")

    # Create test SalesReceipt
    test_data = {
        "check_number": f'TEST-{datetime.now().strftime("%Y%m%d-%H%M%S")}',
        "amount": 150.00,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "payee": "Test Customer",
        "memo": "Test check from create_salesreceipt.py",
        "routing_number": "123456789",
        "account_number": "9876543210",
    }

    print("Test Data:")
    for k, v in test_data.items():
        print(f"  {k:20} {v}")
    print()

    try:
        customer_id = sr_manager.find_or_create_customer(test_data["payee"])
        salesreceipt = sr_manager.create_sales_receipt(
            check_data=test_data,
            customer_id=customer_id,
            payment_method_id=PAYMENT_METHOD_ID,
            deposit_account_id=DEPOSIT_ACCOUNT_ID,
            income_item_id=INCOME_ITEM_ID,
        )

        print("=" * 70)
        print("✅ SalesReceipt Created!")
        print("=" * 70)
        print(f"ID: {salesreceipt.get('Id')}")
        print(f"Doc #: {salesreceipt.get('DocNumber')}")
        print(f"Amount: ${salesreceipt.get('TotalAmt', 0):.2f}")
        print(f"Date: {salesreceipt.get('TxnDate')}")
        print(f"\nView: {sr_manager.get_salesreceipt_url(salesreceipt.get('Id'))}\n")

    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
