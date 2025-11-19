#!/usr/bin/env python3
"""
Test Vertex AI Gemini Pro Vision

This script tests your Vertex AI setup by processing a sample check image.

Usage:
    pytest tests/test_vertex_ai.py -m manual -v -- <path_to_check_image>
    OR
    python tests/test_vertex_ai.py <path_to_check_image>

Example:
    python tests/test_vertex_ai.py sample_check.jpg
"""

import os
import sys
from pathlib import Path
import json

import cv2
import numpy as np
import pytest
from dotenv import load_dotenv

# Add src to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.ocr.vertex_ai_processor import VertexAIProcessor
from src.ocr.check_preprocessor import CheckPreprocessor

# Load environment variables
load_dotenv()

# Configuration
PROJECT_ID = os.getenv("VERTEX_AI_PROJECT_ID")
LOCATION = os.getenv("VERTEX_AI_LOCATION", "us-central1")
MODEL = os.getenv("VERTEX_AI_MODEL", "gemini-2.5-pro")
CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")


def main():
    print("=" * 70)
    print("Vertex AI Gemini Pro Vision Test")
    print("=" * 70)
    print()

    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: python tests/manual/test_vertex_ai.py <path_to_check_image>")
        print()
        print("Example:")
        print("  python tests/manual/test_vertex_ai.py sample_check.jpg")
        print()
        sys.exit(1)

    image_path = sys.argv[1]

    # Check configuration
    if not PROJECT_ID:
        print("❌ Error: VERTEX_AI_PROJECT_ID not set in .env file")
        sys.exit(1)

    if not CREDENTIALS:
        print("❌ Error: GOOGLE_APPLICATION_CREDENTIALS not set in .env file")
        sys.exit(1)

    if not os.path.exists(CREDENTIALS):
        print(f"❌ Error: Service account key file not found: {CREDENTIALS}")
        sys.exit(1)

    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found: {image_path}")
        sys.exit(1)

    print(f"Configuration:")
    print(f"  Project ID: {PROJECT_ID}")
    print(f"  Location:   {LOCATION}")
    print(f"  Model:      {MODEL}")
    print(f"  Image:      {image_path}")
    print()

    # Load image
    print("📸 Loading image...")
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Could not load image: {image_path}")
        sys.exit(1)

    print(f"✓ Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
    print()

    # Preprocess image
    print("🔧 Preprocessing image...")
    preprocessor = CheckPreprocessor()
    preprocessed = preprocessor.preprocess(image)
    print("✓ Image preprocessed (deskewed, denoised, enhanced)")
    print()

    # Initialize Vertex AI processor
    print("🚀 Initializing Vertex AI...")
    try:
        processor = VertexAIProcessor(
            project_id=PROJECT_ID, location=LOCATION, model_name=MODEL
        )
        print("✓ Vertex AI initialized")
        print()
    except Exception as e:
        print(f"❌ Error initializing Vertex AI: {e}")
        sys.exit(1)

    # Extract check fields
    print("🔍 Extracting check fields with Gemini Pro Vision...")
    print()

    try:
        result = processor.extract_check_fields(preprocessed)

        print("=" * 70)
        print("✅ Check Fields Extracted Successfully!")
        print("=" * 70)
        print()

        # Display extracted fields
        print("Extracted Fields:")
        print("-" * 70)
        print(f"  Check Number:    {result.get('check_number', 'N/A')}")
        print(f"  Amount:          ${result.get('amount', 0):.2f}")
        print(f"  Date:            {result.get('date', 'N/A')}")
        print(f"  Payee:           {result.get('payee', 'N/A')}")
        print(f"  Memo:            {result.get('memo', 'N/A')}")
        print(f"  Routing Number:  {result.get('routing_number', 'N/A')}")
        print(f"  Account Number:  {result.get('account_number', 'N/A')}")
        print()

        # Display confidence scores
        print("Confidence Scores:")
        print("-" * 70)
        print(f"  Overall:         {result.get('confidence', 0):.1f}%")
        print()

        field_confidences = result.get("field_confidences", {})
        if field_confidences:
            print("  Field-level confidences:")
            for field, confidence in field_confidences.items():
                print(f"    {field:18} {confidence:.1f}%")
        print()

        # Display raw OCR text
        print("Raw OCR Text:")
        print("-" * 70)
        raw_text = result.get("raw_text", "")
        if raw_text:
            raw_lines = raw_text.splitlines()
            for line in raw_lines[:10]:  # Show first 10 lines
                print(f"  {line}")
            if len(raw_lines) > 10:
                remaining = len(raw_lines) - 10
                print(f"  ... ({remaining} more lines)")
        else:
            print("  (No raw text available)")
        print()

        # Full JSON response
        print("=" * 70)
        print("Full JSON Response:")
        print("=" * 70)
        print(json.dumps(result, indent=2, default=str))
        print()

        # Summary
        confidence = result.get("confidence", 0)
        if confidence >= 85:
            print("✅ HIGH CONFIDENCE - Ready for automatic processing")
        elif confidence >= 70:
            print("⚠️  MEDIUM CONFIDENCE - May need manual review")
        else:
            print("❌ LOW CONFIDENCE - Requires manual review")

    except Exception as e:
        print("=" * 70)
        print("❌ Error extracting check fields")
        print("=" * 70)
        print()
        print(f"Error: {e}")
        print()
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
