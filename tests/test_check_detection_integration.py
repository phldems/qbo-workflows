"""
Test Check Detection Integration

Verify that the per-group outlier filtering algorithm is working correctly
in the production CheckDetector class.
"""

import sys
import os
import pytest

sys.path.insert(0, ".")

from src.ocr.check_detector import CheckDetector
from src.utils.sample_files import (
    get_multi_check_sample_pdf,
    get_single_check_sample_pdf,
)
from src.utils.image_utils import ensure_portrait
import cv2
import numpy as np
import fitz

from dotenv import load_dotenv
load_dotenv()

@pytest.mark.skipif(
    not os.getenv("SINGLE_CHECK_SAMPLE_PDF"),
    reason="SINGLE_CHECK_SAMPLE_PDF not configured - set in .env to run this test",
)
def test_single_check_detection():
    """Test single check detection using configured sample PDF."""
    print("\n" + "=" * 80)
    print("TEST: Single Check Detection")
    print("=" * 80)

    pdf_path = get_single_check_sample_pdf()

    # Load first page
    pdf_document = fitz.open(str(pdf_path))
    page = pdf_document[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
    img_data = pix.samples
    image = np.frombuffer(img_data, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )
    if pix.n == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    image = ensure_portrait(image)
    pdf_document.close()

    # Initialize detector
    detector = CheckDetector(dpi=300)

    # Detect checks
    checks = detector.detect_checks(image)

    print(f"\n📊 Results:")
    print(f"   Detected: {len(checks)} check(s)")

    assert len(checks) >= 1, "Expected at least 1 check"

    for idx, check in enumerate(checks, 1):
        w, h = check["bbox"][2], check["bbox"][3]
        print(f"\n   Check {idx}:")
        print(f"      Size: {w}x{h}px ({w/300:.1f}x{h/300:.1f}in)")
        print(f"      Confidence: {check['confidence']:.1f}%")
        print(f"      Components: {check['components_used']}")
        print(f"      Outliers removed: {check['outliers_removed']}")


@pytest.mark.skipif(
    not os.getenv("MULTI_CHECK_SAMPLE_PDF"),
    reason="MULTI_CHECK_SAMPLE_PDF not configured - set in .env to run this test",
)
def test_multi_check_detection():
    """Test multi-check detection using configured sample PDF."""
    print("\n" + "=" * 80)
    print("TEST: Multi-Check Detection")
    print("=" * 80)

    pdf_path = get_multi_check_sample_pdf()

    # Load first page
    pdf_document = fitz.open(str(pdf_path))
    page = pdf_document[0]
    pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
    img_data = pix.samples
    image = np.frombuffer(img_data, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n
    )
    if pix.n == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    image = ensure_portrait(image)
    pdf_document.close()

    # Initialize detector
    detector = CheckDetector(dpi=300)

    # Detect checks
    checks = detector.detect_checks(image)

    print(f"\n📊 Results:")
    print(f"   Detected: {len(checks)} check(s)")

    assert len(checks) >= 2, "Expected at least 2 checks for multi-check sample"

    for i, check in enumerate(checks, 1):
        w, h = check["bbox"][2], check["bbox"][3]
        print(f"\n   Check {i}:")
        print(f"      Size: {w}x{h}px ({w/300:.1f}x{h/300:.1f}in)")
        print(f"      Confidence: {check['confidence']:.1f}%")
        print(f"      Components: {check['components_used']}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("CHECK DETECTION INTEGRATION TEST")
    print("=" * 80)

    results = []

    # Run tests
    results.append(("Single Check Detection", test_single_check_detection()))
    results.append(("Multi-Check Detection", test_multi_check_detection()))

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {name}")

    print(f"\n   Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n   🎉 All tests passed!")
    else:
        print(f"\n   ⚠️  {total - passed} test(s) failed")

    print("=" * 80 + "\n")

    sys.exit(0 if passed == total else 1)
