"""
Preprocessing Visualization Test

Shows the actual preprocessing steps visually so we can see what each does.
"""

import cv2
import numpy as np
import fitz
from pathlib import Path
import pytest

from src.utils.sample_files import get_single_check_sample_pdf


def save_preprocessing_stages(image: np.ndarray, output_dir: Path, page_num: int):
    """Save each preprocessing stage to visualize what's happening."""

    # Rotate landscape pages to portrait so downstream stages match production
    if image.shape[1] > image.shape[0]:
        print("  ↻ Rotating landscape page to portrait for preprocessing preview")
        image = np.rot90(image, k=3)  # clockwise

    gray = (
        cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if len(image.shape) == 3
        else image.copy()
    )

    # Stage 1: Original grayscale
    cv2.imwrite(str(output_dir / f"page_{page_num + 1}_01_gray.png"), gray)

    # Stage 2: Bilateral filter (noise reduction, edge preservation)
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    cv2.imwrite(str(output_dir / f"page_{page_num + 1}_02_bilateral.png"), filtered)

    # Stage 3: CLAHE (high contrast)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    cv2.imwrite(str(output_dir / f"page_{page_num + 1}_03_clahe.png"), enhanced)

    # Stage 4: Sharpened
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    cv2.imwrite(str(output_dir / f"page_{page_num + 1}_04_sharpened.png"), sharpened)

    # Stage 5: Adaptive threshold
    thresh = cv2.adaptiveThreshold(
        filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 3
    )
    cv2.imwrite(str(output_dir / f"page_{page_num + 1}_05_adaptive_thresh.png"), thresh)

    # Stage 6: Otsu threshold
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite(str(output_dir / f"page_{page_num + 1}_06_otsu.png"), otsu)

    # Stage 7: Canny edges
    edges = cv2.Canny(gray, 50, 150)
    cv2.imwrite(str(output_dir / f"page_{page_num + 1}_07_canny.png"), edges)

    # Stage 8: Morphological closing (small kernel)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed_small = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_small, iterations=2)
    cv2.imwrite(
        str(output_dir / f"page_{page_num + 1}_08_morph_close_small.png"), closed_small
    )

    # Stage 9: Morphological closing (large kernel)
    kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
    closed_large = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_large, iterations=5)
    cv2.imwrite(
        str(output_dir / f"page_{page_num + 1}_09_morph_close_large.png"), closed_large
    )

    # Stage 10: Horizontal dilation (for MICR lines)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
    horizontal = cv2.dilate(thresh, horizontal_kernel, iterations=2)
    cv2.imwrite(
        str(output_dir / f"page_{page_num + 1}_10_horizontal_dilate.png"), horizontal
    )

    # Stage 11: Vertical dilation (for text columns)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    vertical = cv2.dilate(thresh, vertical_kernel, iterations=2)
    cv2.imwrite(
        str(output_dir / f"page_{page_num + 1}_11_vertical_dilate.png"), vertical
    )

    # Stage 12: Combined horizontal + vertical
    combined = cv2.bitwise_or(horizontal, vertical)
    cv2.imwrite(str(output_dir / f"page_{page_num + 1}_12_combined_hv.png"), combined)

    # Stage 13: Massive closing on combined
    kernel_massive = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    massive_close = cv2.morphologyEx(
        combined, cv2.MORPH_CLOSE, kernel_massive, iterations=10
    )
    cv2.imwrite(
        str(output_dir / f"page_{page_num + 1}_13_massive_close.png"), massive_close
    )

    print(f"  ✅ Saved 13 preprocessing stages for page {page_num + 1}")


def test_preprocessing_visualization():
    """Visualize all preprocessing stages."""

    try:
        pdf_path = str(get_single_check_sample_pdf())
    except Exception as exc:
        pytest.skip(f"Single-check sample PDF not configured: {exc}")

    print(f"\n{'='*80}")
    print(f"PREPROCESSING VISUALIZATION TEST")
    print(f"{'='*80}")
    print(f"File: {pdf_path}\n")

    # Create output directory
    output_dir = Path("preprocessing_stages")
    output_dir.mkdir(exist_ok=True)
    print(f"📁 Saving stages to: {output_dir.absolute()}\n")

    # Load PDF
    if not Path(pdf_path).exists():
        print(f"❌ File not found: {pdf_path}")
        return

    pdf_document = fitz.open(pdf_path)
    num_pages = len(pdf_document)

    print(f"📄 PDF has {num_pages} page(s)\n")

    for page_num in range(min(num_pages, 2)):  # Limit to first 2 pages for testing
        print(f"Processing page {page_num + 1}...")

        # Convert page to image
        page = pdf_document[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
        img_data = pix.samples
        image = np.frombuffer(img_data, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        if pix.n == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # Save original
        orig_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(output_dir / f"page_{page_num + 1}_00_original.png"), orig_bgr)

        # Save all preprocessing stages
        save_preprocessing_stages(image, output_dir, page_num)

        print()

    pdf_document.close()

    print(f"{'='*80}")
    print("Preprocessing visualization complete!")
    print(f"Review images in: {output_dir.absolute()}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        pdf_path = Path(sys.argv[1])
    else:
        try:
            pdf_path = get_single_check_sample_pdf()
        except Exception as exc:
            print(f"Error resolving single-check PDF: {exc}")
            sys.exit(1)

    test_preprocessing_visualization(str(pdf_path))
