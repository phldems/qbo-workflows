# Tests

This directory contains all automated and manual tests for the QBO Workflows project.

## Quick Start

```bash
# Run all automated unit tests (default)
pytest

# Run all tests including manual/integration tests
pytest -m ""

# Run only manual/diagnostic tests
pytest -m manual -v

# Run specific test file
pytest tests/test_preprocessing.py -v
```

## Test Organization

Tests are organized by type using pytest markers:

### Unit Tests (default)
Automated tests that don't require external dependencies or services. These run by default with `pytest`.

| File | Purpose |
| --- | --- |
| `test_preprocessing.py` | Verifies image preprocessing (noise removal, deskew, normalization) |
| `test_per_group_outlier_filtering.py` | Tests MICR line/contour detection heuristics |
| `test_check_detection_integration.py` | Runs sample PDFs through detection pipeline (requires env vars) |

### Manual/Integration Tests
Tests that connect to external services or require user interaction. These are marked with `@pytest.mark.manual` and `@pytest.mark.integration`.

| File | Purpose | Usage |
| --- | --- | --- |
| `test_vertex_ai.py` | Smoke test Vertex AI Gemini Pro Vision OCR | `python tests/test_vertex_ai.py <check_image>` |
| `test_email.py` | Validate IMAP/SMTP credentials | `pytest tests/test_email.py -m manual -v` |
| `test_qbo_get.py` | Fetch and display a QuickBooks SalesReceipt | `python tests/test_qbo_get.py <salesreceipt_id>` |
| `test_gcp_credentials.py` | Verify GCP service account credentials | `pytest tests/test_gcp_credentials.py -m manual -v` |
| `test_message_id_threading.py` | Check email threading headers | `python tests/test_message_id_threading.py` |

## Running Tests

### Automated Tests Only (CI/CD)
```bash
pytest
```
This runs only unit tests that don't require external services.

### All Tests
```bash
pytest -m ""
```
Runs all tests including those requiring external services.

### By Marker
```bash
# Manual tests only
pytest -m manual -v

# Integration tests only
pytest -m integration -v

# Unit tests only (same as default)
pytest -m "not manual and not integration"
```

### Verbose Output
```bash
pytest -v                    # Verbose
pytest -vv                   # Very verbose
pytest -s                    # Show print statements
pytest -v -s                 # Both
```

### Specific Tests
```bash
# Run specific file
pytest tests/test_preprocessing.py

# Run specific test function
pytest tests/test_email.py::test_imap -v

# Run tests matching pattern
pytest -k "preprocessing" -v
```

## Test Requirements

### Environment Variables
Most tests require environment variables configured in `.env`:

**Core Settings:**
- `VERTEX_AI_PROJECT_ID` - GCP project ID
- `VERTEX_AI_LOCATION` - GCP region (default: us-central1)
- `VERTEX_AI_MODEL` - Gemini model (default: gemini-2.5-pro)
- `GOOGLE_APPLICATION_CREDENTIALS` - Path to GCP service account key

**Email (for email tests):**
- `IMAP_SERVER`, `IMAP_PORT` - IMAP server settings
- `SMTP_SERVER`, `SMTP_PORT` - SMTP server settings
- `EMAIL_USERNAME`, `EMAIL_PASSWORD` - Email credentials
- `EMAIL_INBOX_FOLDER` - Inbox folder name (default: INBOX)

**QuickBooks (for QBO tests):**
- `QBO_CLIENT_ID`, `QBO_CLIENT_SECRET` - OAuth credentials
- `QBO_REALM_ID` - Company ID
- `QBO_REFRESH_TOKEN` - OAuth refresh token
- `QBO_ENVIRONMENT` - 'sandbox' or 'production'
- `QBO_PAYMENT_METHOD_ID`, `QBO_DEPOSIT_ACCOUNT_ID`, `QBO_INCOME_ITEM_ID`

**Check Detection (for integration tests):**
- `SINGLE_CHECK_SAMPLE_PDF` - Path to single check PDF sample
- `MULTI_CHECK_SAMPLE_PDF` - Path to multi-check PDF sample

### Sample Files
Some tests require sample check images. Configure paths in `.env` or the tests will be skipped.

## Manual Test Scripts

Several tests can also be run as standalone scripts for easier diagnostics:

```bash
# Test Vertex AI OCR
python tests/test_vertex_ai.py sample_check.jpg

# Test email connectivity
python tests/test_email.py

# Test GCP credentials
python tests/test_gcp_credentials.py

# Fetch a QuickBooks SalesReceipt
python tests/test_qbo_get.py <salesreceipt_id>

# Test email threading
python tests/test_message_id_threading.py
```

## Test Development

### Adding New Tests

1. **Unit tests**: Add to existing or create new `test_*.py` file
2. **Manual/integration tests**: Mark with `@pytest.mark.manual` and `@pytest.mark.integration`

Example:
```python
import pytest

@pytest.mark.manual
@pytest.mark.integration
def test_my_external_service():
    """Test that connects to external service."""
    # Test implementation
    pass
```

### Available Markers

Defined in `pytest.ini`:
- `@pytest.mark.manual` - Manual/diagnostic test requiring external services
- `@pytest.mark.integration` - Integration test connecting to external services
- `@pytest.mark.unit` - Unit test (default, can be omitted)

## Continuous Integration

The CI pipeline should run:
```bash
pytest
```

This runs only unit tests by default, skipping manual/integration tests that require credentials or external services.

## Troubleshooting

### Tests are skipped
Check if required environment variables are set:
```bash
pytest -v -rs  # Show reason for skipped tests
```

### Import errors
Ensure you're running from the project root:
```bash
cd /path/to/qbo-workflows
pytest
```

### Manual tests don't run
Manual tests are excluded by default. Run with:
```bash
pytest -m manual -v
```

## Related Resources

- **Setup Scripts**: See `scripts/README.md` for OAuth setup and entity ID lookup
- **Production Code**: `src/` contains the main application code
- **Configuration**: `.env.example` shows all available environment variables
