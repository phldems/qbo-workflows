# QBO Workflows – Automated Check Intake

End-to-end automation for getting mailed-in checks into QuickBooks Online. The workflow watches an IMAP inbox for new attachments, detects single or multiple checks inside PDFs or images, extracts structured fields with Vertex AI Gemini, scores confidence, and creates SalesReceipts (with attachments) through the QuickBooks API. Processing history, OAuth tokens, and troubleshooting metadata are stored in SQLite for auditability.

## Highlights
- **Email-first ingestion** – `src/integrations/email` handles polling, attachment filtering, and folder routing for processed/failed/duplicate messages.
- **Vision pipeline built for checks** – `CheckPreprocessor` normalizes images, `CheckDetector` isolates each check using per-group outlier filtering, and the Vertex AI processor extracts 23+ fields in one call.
- **QuickBooks-native records** – `QuickBooksSalesReceipt` builds compliant payloads, attaches cropped check imagery, and respects rate limits via `QBOAPIRateLimiter`.
- **Evidence trail** – `src/utils/database.Database` persists check metadata, OAuth tokens, log pointers, and manual-review decisions; Loguru writes structured logs to `./logs/check_processor.log`.
- **Scriptable tooling** – The `scripts/` folder includes helpers for OAuth, entity lookups, Vertex smoke tests, and end-to-end demonstrations.

## Flow at a Glance
1. **Inbound email** – `EmailMonitor.fetch_unread_checks()` imports attachments (PDF, TIFF, PNG/JPG) and stores them under `data/temp`.
2. **Preprocess & detection** – `CheckProcessor` loads each page via PyMuPDF, normalizes it through `CheckPreprocessor`, and calls `CheckDetector.detect_checks()` to crop individual checks (single page, multi page, or multi check per page supported).
3. **OCR & validation** – Crops feed `VertexAIProcessor.extract_check_fields()` which enforces the JSON schema documented in `VERTEX_SCHEMA.md`. `ConfidenceScorer` evaluates formatting, MICR checksum, and multi-pass consistency to choose auto-process vs manual-review paths.
4. **QuickBooks write** – The workflow chooses/creates a Customer, posts a SalesReceipt, uploads the check image, and records IDs + timestamps in SQLite. Errors route the source email to the configured Failed/Manual folders.
5. **Notifications** – `EmailSender` can acknowledge processed checks, flag manual review, or escalate failures depending on configuration.

## Repository Layout
| Path | Purpose |
| --- | --- |
| `main.py` | CLI entry point that loads settings, configures logging, and drives the polling loop. |
| `src/workflows/check_processor.py` | Orchestrates ingestion → OCR → QuickBooks, including retry and folder routing logic. |
| `src/ocr/` | Preprocessing (`check_preprocessor.py`), detection (`check_detector.py`), confidence scoring, and Vertex AI client. |
| `src/integrations/` | Email monitors/senders plus QuickBooks auth, API client, rate limiter, and SalesReceipt helper. |
| `src/utils/` | SQLite wrapper, ID generator, misc helpers. |
| `scripts/` | OAuth bootstrap and troubleshooting helpers (tokens, entity lookups, demo flows). See `scripts/README.md` for usage. |
| `tests/` | Automated pytest suites plus manual diagnostics (`tests/manual/`). See `tests/README.md` for details. |

## Tech Stack
- Python 3.11+ (container uses 3.13) with `pydantic-settings`, `numpy`, `opencv-python`, `PyMuPDF`, `google-genai`, `requests`, and `loguru`.
- SQLite for persistence; Vertex AI Gemini Pro Vision for OCR; QuickBooks Online API for accounting.
- Docker image ships only runtime deps (no heavy local ML models) and expects credentials via bind-mounted files.

## Setup
- Follow **`SETUP.md`** for prerequisites, `.env` configuration, Vertex AI + QuickBooks provisioning, and optional Docker usage.
- `.env.example` lists every supported setting; the application reads them via `src/config/settings.py`.

## Running the Workflow
### Local Python
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

### Docker Compose
```bash
docker compose up -d --build
docker compose logs -f check-processor
```
Ensure `${GOOGLE_APPLICATION_CREDENTIALS}` in `.env` points to the host JSON key so it can be mounted into `/app/credentials.json`.

## Testing & QA
- Unit/integration suites plus manual diagnostics live under `tests/`. Run everything with `pytest` or invoke the scripts in `tests/manual/` for smoke checks.
- Detection-focused tests (e.g., `tests/test_check_detection_integration.py`) rely on sample PDFs checked into the repo.
- Manual smoke scripts (see `tests/README.md` for details):
  - `python tests/manual/test_vertex_ai.py` verifies Vertex OCR + schema compliance.
  - `python tests/manual/test_email.py` exercises IMAP/SMTP credentials.
  - `python tests/manual/test_qbo_get.py` pulls an existing SalesReceipt via the robust API client.
- Setup helpers such as `python scripts/list_qbo_entities.py` and `python scripts/get_qbo_tokens.py` still live under `scripts/`.

## Operational Notes
- Logs roll over at `10 MB` / 30 days; adjust via `LOG_ROTATION`/`LOG_RETENTION` env vars.
- Database + temp image directories live under `./data`; back them up if you need to preserve processing history.
- Manual-review and failure folders are IMAP-driven—configure them in `.env` to match your email provider.
- Rate limiting: `QBOAPIRateLimiter` enforces guardrails per minute/hour to keep the integration in good standing.
- Vertex schema or detection tweaks should be documented in `VERTEX_SCHEMA.md` and `DETECTION_ALGORITHM.md` respectively to keep ops, ML, and finance teams aligned.
