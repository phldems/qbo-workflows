# Check Detection Algorithm

This document describes the end-to-end logic used to isolate individual checks before OCR. Implementation lives primarily in `src/ocr/check_preprocessor.py` and `src/ocr/check_detector.py`, with integration glue inside `src/workflows/check_processor.py`.

## Goals
- Accept multi-page PDFs or multi-check-per-page scans and reliably crop each check.
- Preserve MICR lines and handwritten edges without over-expanding background noise.
- Produce confident crops that downstream Vertex AI OCR can parse without additional geometry work.
- Surface enough metadata (bounding boxes, confidence, component counts) for audit logs and testing.

## Pipeline Overview
1. **Load + normalize pages** – `CheckProcessor._load_all_pages()` converts PDFs via PyMuPDF at 300 DPI and passes raw images to the preprocessor.
2. **Preprocess** (`CheckPreprocessor.process`)  
   - RGB/gray normalization & denoise (bilateral filter).  
   - Document edge detection, perspective correction, and deskewing if a quadrilateral is found.  
   - Illumination flattening, adaptive binarization, and morphology-based cleanup to remove background texture.
3. **Detect** (`CheckDetector.detect_checks`)  
   - Connected components (8-way) on Otsu-thresholded images to collect ink blobs while skipping border-touching or extremely small areas.  
   - Gap-detection with IQR thresholding on vertical spacing to split components into provisional groups (candidate checks).  
   - MICR merge heuristic glues short, bottom-dwelling groups (<1.5 in tall, <2 in gaps) back to the preceding group so MICR text is never left behind.  
   - Per-group outlier filtering uses Z-scores (default 2.5) on centroid positions so stray noise in a group is ignored without affecting other checks.  
   - Bounding boxes expand 4 % to reclaim context, enforce a 1 inch minimum width, crop the pixels, and emit metadata (confidence, component counts, outliers removed).  
4. **Routing logic** (`CheckProcessor.process_single_check`)  
   - Single-page files with zero detections fall back to the full page.  
   - One detection ⇒ crop + OCR.  
   - Multiple detections ⇒ iterate each detection (per page) and propagate the originating filename/page index into downstream IDs/logs.  
   - Multi-page PDFs always process page-by-page; detections decide whether we crop or retain the entire page image.

## Key Parameters
| Parameter | Default | Location | Impact |
| --- | --- | --- | --- |
| `dpi` | 300 | `CheckDetector.__init__` | Used for size heuristics (inch conversions, MICR thresholds). |
| `iqr_factor` | 1.5 | `CheckDetector.__init__` | Larger values reduce the number of detected gaps → fewer groups. |
| `outlier_threshold` | 2.5 | `CheckDetector.__init__` | Lowering it removes more stray components but risks cutting off legitimate ink. |
| `expand_percent` | 4.0 | `CheckDetector.__init__` | Adds padding so signatures and edge text are not clipped. |

Tune them cautiously—changes affect downstream OCR confidence and must be reflected in regression tests.

## Testing
- `tests/test_check_detection_integration.py` — Exercises both single-check (`$SINGLE_CHECK_SAMPLE_PDF`) and multi-check (`$MULTI_CHECK_SAMPLE_PDF`) scenarios end to end.
- `tests/test_per_group_outlier_filtering.py` — Standalone harness for validating how the per-group Z-score filtering behaves on sample PDFs.
- Archive artifacts under `archive/detection_tests/` capture historical experiments and can be referenced when comparing algorithms.

When modifying the detector, regenerate crops from representative PDFs, run the Pytest suite, and capture before/after bbox stats so finance ops can review differences.

## Known Limitations & Future Work
- **Low-contrast scans** – Extremely faint checks may fail edge detection; consider adaptive histogram equalization before thresholding.
- **Exotic layouts** – Wallet-sized or stubbed checks that break the vertical stacking assumption may need alternate grouping heuristics.
- **Throughput** – Connected components on very large TIFFs can be slow; batching or downscaling might be required for >300 DPI scans.
- **3D distortions** – Current perspective correction assumes planar scans; heavy folds remain unsolved.