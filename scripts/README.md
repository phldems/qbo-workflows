# Setup Scripts

This directory contains essential setup and configuration scripts for the QBO Workflows project.

For test and diagnostic scripts, see `tests/README.md`.

## Overview

These scripts help you with initial QuickBooks OAuth setup and configuration. Run them in order during first-time setup.

## Scripts

### 1. `get_qbo_tokens.py`
Get your initial OAuth tokens from QuickBooks.

```bash
python scripts/get_qbo_tokens.py
```

**What it does:**
- Generates QuickBooks authorization URL
- Walks you through OAuth flow
- Exchanges authorization code for access/refresh tokens
- Optionally updates your `.env` file

**When to use:**
- First-time setup
- When refresh token expires (after 100 days of inactivity)
- If you need to re-authorize the application

---

### 2. `list_qbo_entities.py`
List QuickBooks accounts, items, and payment methods to find IDs.

```bash
python scripts/list_qbo_entities.py
```

**What it does:**
- Lists all QuickBooks accounts (for `QBO_DEPOSIT_ACCOUNT_ID`)
- Lists all items/services (for `QBO_INCOME_ITEM_ID`)
- Lists all payment methods (for `QBO_PAYMENT_METHOD_ID`)
- Highlights recommended options for check processing

**When to use:**
- Finding entity IDs for your `.env` file
- When you need to verify account/item names

---

### 3. `create_salesreceipt.py`
Create a test SalesReceipt in QuickBooks.

```bash
python scripts/create_salesreceipt.py
```

**What it does:**
- Creates a test SalesReceipt with sample data
- Verifies your entity IDs are correct
- Shows QuickBooks URL to view the created receipt
- Tests the full create workflow

**When to use:**
- After setting up entity IDs in `.env`
- Verifying the SalesReceipt creation flow works
- Testing before processing real checks

---

## Initial Setup Workflow

Follow these steps for first-time setup:

1. **Configure `.env` file** with your basic credentials
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

2. **Get OAuth tokens**
   ```bash
   python scripts/get_qbo_tokens.py
   ```
   This will generate `QBO_REFRESH_TOKEN` for your `.env` file.

3. **Find entity IDs**
   ```bash
   python scripts/list_qbo_entities.py
   ```
   This will show you the IDs for:
   - `QBO_DEPOSIT_ACCOUNT_ID`
   - `QBO_INCOME_ITEM_ID`
   - `QBO_PAYMENT_METHOD_ID`

4. **Update `.env`** with the entity IDs from step 3

5. **Verify QuickBooks integration**
   ```bash
   python scripts/create_salesreceipt.py
   ```
   This creates a test SalesReceipt to confirm everything works.

6. **Run tests** to verify all integrations
   ```bash
   pytest tests/test_gcp_credentials.py -v  # Verify GCP/Vertex AI
   pytest tests/test_email.py -v            # Verify email (optional)
   ```

## Troubleshooting

### QuickBooks auth issues
Run `get_qbo_tokens.py` to re-authorize

### Can't find entity IDs
Run `list_qbo_entities.py` and look for accounts/items that match your needs

### SalesReceipt creation failing
1. Check entity IDs are correct (run `list_qbo_entities.py`)
2. Verify OAuth token is valid (run `get_qbo_tokens.py`)
3. Check `.env` file has all required variables

## Requirements

All scripts require:
- Virtual environment activated
- `.env` file configured (at minimum: `QBO_CLIENT_ID`, `QBO_CLIENT_SECRET`, `QBO_REALM_ID`)
- Dependencies installed from `requirements.txt`

```bash
# Activate virtual environment
source .venv/bin/activate

# Install dependencies (if not already done)
pip install -r requirements.txt

# Run any script
python scripts/<script_name>.py
```

## Related Documentation

- **Tests**: See `tests/README.md` for diagnostic and integration tests
- **Configuration**: See `.env.example` for all available environment variables
- **Main Application**: See project root README for running the full workflow
