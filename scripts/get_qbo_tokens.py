#!/usr/bin/env python3
"""
QuickBooks OAuth Token Helper

This script walks you through getting your initial OAuth tokens from QuickBooks.
Run this ONCE to get your refresh token, then add it to your .env file.

Usage:
    python scripts/get_qbo_tokens.py
"""

import os
import sys
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv
from intuitlib.client import AuthClient
from intuitlib.enums import Scopes

# Load environment variables
load_dotenv()

# Configuration from .env
CLIENT_ID = os.getenv("QBO_CLIENT_ID")
CLIENT_SECRET = os.getenv("QBO_CLIENT_SECRET")
REDIRECT_URI = os.getenv("QBO_REDIRECT_URI", "https://oauth.platform.intuit.com/op/v1")
ENVIRONMENT = os.getenv("QBO_ENVIRONMENT", "production")

if not CLIENT_ID or not CLIENT_SECRET:
    print("❌ Error: QBO_CLIENT_ID and QBO_CLIENT_SECRET must be set in .env file")
    exit(1)


def main():
    print("=" * 70)
    print("QuickBooks OAuth Token Setup")
    print("=" * 70)
    print()

    print(f"Configuration:")
    print(f"  Client ID:     {CLIENT_ID[:10]}...{CLIENT_ID[-4:]}")
    print(f"  Environment:   {ENVIRONMENT}")
    print(f"  Redirect URI:  {REDIRECT_URI}")
    print()

    # Initialize auth client
    auth_client = AuthClient(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        environment=ENVIRONMENT,
        redirect_uri=REDIRECT_URI,
    )

    # Generate authorization URL
    scopes = [Scopes.ACCOUNTING]
    auth_url = auth_client.get_authorization_url(scopes)

    print("Step 1: Authorize the Application")
    print("-" * 70)
    print()
    print("Click the link below to authorize QuickBooks access:")
    print()
    print(f"  {auth_url}")
    print()
    print("This will open QuickBooks in your browser. Follow these steps:")
    print()
    print("  1. Sign in to your QuickBooks account")
    print("  2. Select the company you want to connect")
    print("  3. Click 'Authorize' to grant access")
    print("  4. After authorizing, you'll be redirected to a URL")
    print()
    print("=" * 70)
    print("IMPORTANT: Copy the ENTIRE redirect URL from your browser")
    print("=" * 70)
    print()
    print("The redirect URL will look like this:")
    print(f"  {REDIRECT_URI}?code=YOUR_CODE&realmId=YOUR_REALM_ID&state=...")
    print()
    print("You need to copy the ENTIRE URL including all parameters.")
    print()

    # Get callback URL from user
    print("Step 2: Enter the Redirect URL")
    print("-" * 70)
    print()

    callback_url = input("Paste the redirect URL here and press Enter:\n> ").strip()
    print()

    if not callback_url:
        print("❌ Error: No URL provided")
        return

    # Parse the callback URL
    try:
        parsed = urlparse(callback_url)
        params = parse_qs(parsed.query)

        auth_code = params.get("code", [None])[0]
        realm_id = params.get("realmId", [None])[0]

        if not auth_code:
            print("❌ Error: Could not find 'code' parameter in URL")
            print()
            print(
                "Make sure you copied the entire redirect URL including all parameters."
            )
            return

        if not realm_id:
            print("❌ Error: Could not find 'realmId' parameter in URL")
            print()
            print(
                "Make sure you copied the entire redirect URL including all parameters."
            )
            return

        print("✓ Successfully extracted authorization code and realm ID")
        print(f"  Realm ID: {realm_id}")
        print()

    except Exception as e:
        print(f"❌ Error parsing URL: {e}")
        print()
        print("Make sure you copied the entire redirect URL correctly.")
        return

    # Exchange code for tokens
    print("Step 3: Exchanging authorization code for tokens...")
    print("-" * 70)
    print()

    try:
        # Exchange code for tokens
        auth_client.get_bearer_token(auth_code, realm_id=realm_id)

        access_token = auth_client.access_token
        refresh_token = auth_client.refresh_token

        print("✓ Successfully obtained tokens!")
        print()
        print("=" * 70)
        print("🎉 Setup Complete!")
        print("=" * 70)
        print()
        print("Add these values to your .env file:")
        print()
        print(f"QBO_REALM_ID={realm_id}")
        print(f"QBO_REFRESH_TOKEN={refresh_token}")
        print()
        print("⚠️  IMPORTANT:")
        print("  - Keep your refresh token secret!")
        print("  - Refresh tokens expire after 100 days of inactivity")
        print("  - The access token will be automatically refreshed as needed")
        print()

        # Optionally write to .env file
        response = input(
            "Would you like to automatically update your .env file? (y/n): "
        )
        if response.lower() == "y":
            update_env_file(realm_id, refresh_token)
        else:
            print()
            print("Please manually add the values above to your .env file.")

    except Exception as e:
        print(f"❌ Error exchanging code for tokens: {e}")
        print()
        print("Common issues:")
        print(
            "  - Authorization code may have expired (they're single-use and expire quickly)"
        )
        print("  - Client ID or Client Secret may be incorrect")
        print(
            "  - Redirect URI may not match what's configured in QuickBooks Developer Portal"
        )
        print()
        print("Please try running this script again from the beginning.")


def update_env_file(realm_id: str, refresh_token: str):
    """Update .env file with new values"""
    try:
        env_path = ".env"

        # Read existing .env
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                lines = f.readlines()
        else:
            lines = []

        # Update or add values
        updated_realm = False
        updated_token = False

        for i, line in enumerate(lines):
            if line.startswith("QBO_REALM_ID="):
                lines[i] = f"QBO_REALM_ID={realm_id}\n"
                updated_realm = True
            elif line.startswith("QBO_REFRESH_TOKEN="):
                lines[i] = f"QBO_REFRESH_TOKEN={refresh_token}\n"
                updated_token = True

        # Add if not found
        if not updated_realm:
            lines.append(f"QBO_REALM_ID={realm_id}\n")
        if not updated_token:
            lines.append(f"QBO_REFRESH_TOKEN={refresh_token}\n")

        # Write back
        with open(env_path, "w") as f:
            f.writelines(lines)

        print()
        print("✓ Successfully updated .env file!")
        print()

    except Exception as e:
        print(f"❌ Error updating .env file: {e}")
        print("Please manually add the values to your .env file.")


if __name__ == "__main__":
    main()
