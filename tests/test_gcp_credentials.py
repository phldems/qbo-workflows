#!/usr/bin/env python3
"""
Test Google Cloud Credentials

This script validates your Google Cloud service account credentials
and checks access to Vertex AI.

Usage:
    python tests/manual/test_gcp_credentials.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
PROJECT_ID = os.getenv("VERTEX_AI_PROJECT_ID")
LOCATION = os.getenv("VERTEX_AI_LOCATION", "us-central1")
CREDENTIALS_PATH = os.path.join("./", os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
GEMINI_MODEL = os.getenv("VERTEX_AI_MODEL", "gemini-2.5-pro")


def test_credentials_file():
    """Test if credentials file exists and is readable"""
    print("=" * 70)
    print("Step 1: Checking Credentials File")
    print("=" * 70)
    print()

    assert CREDENTIALS_PATH, (
        "GOOGLE_APPLICATION_CREDENTIALS not set in .env file. "
        "Please add: GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json"
    )

    print(f"Credentials path: {CREDENTIALS_PATH}")
    print()

    assert os.path.exists(CREDENTIALS_PATH), (
        f"Credentials file not found: {CREDENTIALS_PATH}. "
        "Make sure the path is correct and the file exists."
    )

    print("✓ Credentials file exists")
    print()

    # Try to read and validate JSON structure
    import json

    with open(CREDENTIALS_PATH, "r") as f:
        creds = json.load(f)

    required_fields = [
        "type",
        "project_id",
        "private_key_id",
        "private_key",
        "client_email",
    ]
    missing_fields = [field for field in required_fields if field not in creds]

    assert (
        not missing_fields
    ), f"Invalid service account key file. Missing fields: {', '.join(missing_fields)}"

    print("✓ Valid service account key file format")
    print(f"  Service account: {creds.get('client_email')}")
    print(f"  Project ID:      {creds.get('project_id')}")
    print()

    # Check if project_id matches
    if PROJECT_ID and creds.get("project_id") != PROJECT_ID:
        print(f"⚠️  Warning: Project ID mismatch!")
        print(f"  .env VERTEX_AI_PROJECT_ID: {PROJECT_ID}")
        print(f"  Credentials project_id:    {creds.get('project_id')}")
        print()


def test_gcp_auth():
    """Test Google Cloud authentication"""
    print("=" * 70)
    print("Step 2: Testing Google Cloud Authentication")
    print("=" * 70)
    print()

    from google.auth import default

    credentials, project = default()

    print("✓ Successfully authenticated with Google Cloud")
    print(f"  Project: {project}")
    print()


def test_vertex_ai_access():
    """Test Vertex AI API access"""
    print("=" * 70)
    print("Step 3: Testing Vertex AI Access")
    print("=" * 70)
    print()

    assert PROJECT_ID, "VERTEX_AI_PROJECT_ID not set in .env file"

    print(f"Project ID: {PROJECT_ID}")
    print(f"Location:   {LOCATION}")
    print()

    from google import genai
    from google.genai import types

    # Initialize Google GenAI client with Vertex AI
    print("Initializing Vertex AI with google-genai SDK...")
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)
    print("✓ Vertex AI initialized successfully")
    print()

    # Test a simple generation (non-vision)
    print(f"Testing model ({GEMINI_MODEL}) with simple prompt...")
    response = client.models.generate_content(
        model=GEMINI_MODEL, contents="Say 'Hello, Vertex AI is working!'"
    )
    response_text = response.text.strip()
    print(f"✓ Model response: {response_text}")
    print()


def test_vision_capabilities():
    """Test vision model specifically"""
    print("=" * 70)
    print("Step 4: Testing Vision Model Capabilities")
    print("=" * 70)
    print()

    from google import genai
    from google.genai import types
    import base64

    # Create a simple test image (1x1 red pixel PNG)
    test_image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
    test_image_bytes = base64.b64decode(test_image_base64)

    print("Initializing vision model with google-genai SDK...")
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

    print("Testing vision capabilities with sample image...")
    image_part = types.Part.from_bytes(data=test_image_bytes, mime_type="image/png")

    response = client.models.generate_content(
        model=GEMINI_MODEL, contents=["Describe this image briefly.", image_part]
    )

    response_text = response.text.strip()
    print(f"✓ Vision model response: {response_text}")
    print()
    print("✅ Vision model is working correctly!")
    print()


def main():
    print("=" * 70)
    print("Google Cloud Credentials & Vertex AI Test")
    print("=" * 70)
    print()

    results = {
        "credentials_file": False,
        "gcp_auth": False,
        "vertex_ai": False,
        "vision": False,
    }

    # Test credentials file
    results["credentials_file"] = test_credentials_file()
    if not results["credentials_file"]:
        print_summary(results)
        sys.exit(1)

    # Test GCP authentication
    results["gcp_auth"] = test_gcp_auth()
    if not results["gcp_auth"]:
        print_summary(results)
        sys.exit(1)

    # Test Vertex AI access
    results["vertex_ai"] = test_vertex_ai_access()
    if not results["vertex_ai"]:
        print_summary(results)
        sys.exit(1)

    # Test vision capabilities
    results["vision"] = test_vision_capabilities()

    # Print summary
    print_summary(results)

    if all(results.values()):
        print("✅ All tests passed! You're ready to use Vertex AI for check OCR.")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed, but basic Vertex AI access is working.")
        print("You should still be able to process checks.")
        sys.exit(0)


def print_summary(results):
    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    print()

    tests = [
        ("Credentials File", results["credentials_file"]),
        ("GCP Authentication", results["gcp_auth"]),
        ("Vertex AI Access", results["vertex_ai"]),
        ("Vision Model", results["vision"]),
    ]

    for test_name, passed in tests:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {test_name:25} {status}")

    print()


if __name__ == "__main__":
    main()
