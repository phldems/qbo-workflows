# Vertex AI Response Schema

`src/ocr/vertex_ai_processor.py` uses the Google `google-genai` SDK with Vertex AI Gemini Pro Vision to extract all check fields in a single call. This document captures the enforced JSON schema plus usage notes so downstream systems (QuickBooks integration, manual review tooling, analytics) stay aligned when the schema evolves.

## Invocation Summary
- The processor calls `genai.Client(..., vertexai=True)` with a strict `GenerateContentConfig` so Gemini must return JSON conforming to the schema.
- Requests include the cropped check image (`image/png`) and a structured prompt that:  
  - Reminds the model to transcribe exactly what appears,  
  - Supplies today’s date for `check_age_days`, and  
  - Explains how to populate nested objects/booleans.
- Response MIME type is `application/json`. Missing/illegible data must be `null` (never empty strings).

## Top-Level Fields
| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `check_number` | string \| null | ✅ | Digits in the top-right corner. Preserve leading zeros. |
| `date` | string \| null | ✅ | MM/DD/YYYY (two-digit years → 20XX). |
| `amount_numeric` | number \| null | ✅ | Numeric amount with cents, no currency symbol. |
| `amount_written` | string \| null | ✅ | Verbatim transcription of the written amount line. |
| `payee` | string \| null | ✅ | Text from “Pay to the order of”. |
| `payor_name` | object (see below) \| null | ✅ | Parsed owner names. All sub-fields required but may be null individually. |
| `payor_address` | object (see below) \| null | ✅ | Street/city/state/zip/country block from top left. |
| `bank_name` | string \| null | ✅ | Bank brand text/logo area. |
| `bank_address` | string \| null | ✅ | Address associated with the bank header. |
| `micr_line` | string \| null | ✅ | Raw MICR line including transit symbols. |
| `routing_number` | string \| null | ✅ | 9-digit ABA number (digits only). |
| `account_number` | string \| null | ✅ | Account number from MICR (digits only). |
| `memo` | string \| null | ✅ | Memo line text. |
| `signature_present` | boolean | ✅ | Whether a signature is visible anywhere on the signature line. |
| `signature_matches_payor` | boolean \| null | ✅ | Whether the signature name appears to match a payor; null if indeterminable. |
| `amounts_match` | boolean \| null | ✅ | Do `amount_numeric` and `amount_written` refer to the same value? |
| `check_age_days` | integer \| null | ✅ | Days from check date to “today” (prompt includes today’s date). |
| `is_stale_dated` | boolean | ✅ | True if age > 180 days. |
| `is_post_dated` | boolean | ✅ | True if date is in the future. |
| `check_condition` | enum(`excellent`,`good`,`fair`,`poor`) | ✅ | Subjective assessment of scan/surface quality. |
| `requires_manual_review` | boolean | ✅ | Signals low confidence, mismatches, or anomalies that need human review. |
| `raw_text` | string | ✅ | Full transcription of *all* visible text (for auditing / re-OCR). |
| `confidence` | integer (0‑100) | ✅ | Overall OCR confidence assigned by Gemini. |
| `field_confidences` | object (see table) | ✅ | Per-field 0‑100 confidences; every listed key is required. |

### `payor_name`
| Sub-field | Type | Notes |
| --- | --- | --- |
| `first_name` | string \| null | Parsed from printed name block. |
| `middle_name` | string \| null | Include initials when present. |
| `last_name` | string \| null | Family name; null if not legible. |
| `is_joint` | boolean | True when text includes “and”/“&” indicating a shared account. |
| `full_name` | string \| null | Raw text for the entire printed name; always populate even when parsing fails. |

### `payor_address`
| Sub-field | Type | Notes |
| --- | --- | --- |
| `street` | string \| null | Street number + name. |
| `city` | string \| null | City or locality. |
| `state` | string \| null | Prefer two-letter codes (e.g., `CA`). |
| `zip` | string \| null | 5 or 9 digits; keep any hyphenation. |
| `country` | string \| null | Default to `USA` when format clearly matches US addresses. |

### `field_confidences`
All keys are required with integer values from 0–100:
`check_number`, `date`, `amount_numeric`, `amount_written`, `payee`, `payor_name`, `payor_address`, `bank_name`, `bank_address`, `micr_line`, `routing_number`, `account_number`, `memo`.

## Validation & Post-Processing
- `ConfidenceScorer` supplements Gemini’s scores with routing-number checksum validation, format checks, and multi-run consistency. Its output drives the transition between automatic posting and manual review (see `Settings.ocr_confidence_threshold` / `manual_review_threshold`).
- `amounts_match`, `is_stale_dated`, and `is_post_dated` are trusted as-is but still logged. If they contradict local calculations, the check is forced into manual review.
- The MICR line is persisted verbatim for compliance while sanitized routing/account numbers feed QuickBooks lookup logic.

## Testing the Schema
- `python tests/manual/test_vertex_ai.py` loads sample images, hits the configured Vertex model, and prints schema-compliant responses.
- When bumping models or schema fields, update both `vertex_ai_processor.py` and this document, then capture before/after responses for QA. Coordinate with finance/ops so downstream consumers (manual-review tools, analytics dashboards) adjust to new fields or nullability rules.

## Extending the Schema
1. Update the `response_schema` definition to include new properties or relaxed requirements.
2. Document the change here with type + rationale.
3. Re-run the smoke test and at least one real check to confirm the model can produce the new fields.
4. Update any consumers (database schema, QuickBooks payload builder, data warehouse pipelines) before shipping.
