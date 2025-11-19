#!/usr/bin/env python3
"""
Test Email Connection (IMAP/SMTP)

This script tests your email configuration by:
1. Connecting to IMAP server
2. Listing folders
3. Checking for unread messages
4. Optionally sending a test email via SMTP

Usage:
    pytest tests/test_email.py -m manual -v
    OR
    python tests/test_email.py
"""

import os
import sys
from pathlib import Path

import pytest
from dotenv import load_dotenv
from imap_tools import MailBox, AND

# Load environment variables
load_dotenv()

# IMAP Configuration
IMAP_SERVER = os.getenv("IMAP_SERVER")
IMAP_PORT = int(os.getenv("IMAP_PORT", "993"))
IMAP_USE_SSL = os.getenv("IMAP_USE_SSL", "true").lower() == "true"

# SMTP Configuration
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))
SMTP_USE_SSL = os.getenv("SMTP_USE_SSL", "true").lower() == "true"

# Credentials
EMAIL_USERNAME = os.getenv("EMAIL_USERNAME")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

# Folders
INBOX_FOLDER = os.getenv("EMAIL_INBOX_FOLDER", "INBOX")


@pytest.mark.manual
@pytest.mark.integration
def test_imap():
    """Test IMAP connection"""
    print("=" * 70)
    print("Testing IMAP Connection")
    print("=" * 70)
    print()

    assert all(
        [IMAP_SERVER, EMAIL_USERNAME, EMAIL_PASSWORD]
    ), "Missing IMAP configuration in .env file. Required: IMAP_SERVER, EMAIL_USERNAME, EMAIL_PASSWORD"

    print(f"Configuration:")
    print(f"  Server:   {IMAP_SERVER}")
    print(f"  Port:     {IMAP_PORT}")
    print(f"  SSL:      {IMAP_USE_SSL}")
    print(f"  Username: {EMAIL_USERNAME}")
    print()

    print("Connecting to IMAP server...")
    with MailBox(IMAP_SERVER, IMAP_PORT).login(
        EMAIL_USERNAME, EMAIL_PASSWORD
    ) as mailbox:
        print("✓ Successfully connected to IMAP server")
        print()

        # List folders
        print("Available folders:")
        print("-" * 70)
        folders = list(mailbox.folder.list())
        for folder in folders:
            print(f"  - {folder.name}")
        print()

        # Check inbox
        print(f"Checking {INBOX_FOLDER}...")
        mailbox.folder.set(INBOX_FOLDER)

        # Count messages
        all_count = len(list(mailbox.fetch()))
        unread_count = len(list(mailbox.fetch(AND(seen=False))))

        print(f"  Total messages:  {all_count}")
        print(f"  Unread messages: {unread_count}")
        print()

        # Show recent unread messages
        if unread_count > 0:
            print("Recent unread messages:")
            print("-" * 70)
            for i, msg in enumerate(mailbox.fetch(AND(seen=False), limit=5), 1):
                print(f"{i}. From: {msg.from_}")
                print(f"   Subject: {msg.subject}")
                print(f"   Date: {msg.date}")
                print(f"   Attachments: {len(msg.attachments)}")
                print()

        print("✅ IMAP test successful!")


@pytest.mark.manual
@pytest.mark.integration
def test_smtp():
    """Test SMTP connection"""
    print()
    print("=" * 70)
    print("Testing SMTP Connection")
    print("=" * 70)
    print()

    assert all(
        [SMTP_SERVER, EMAIL_USERNAME, EMAIL_PASSWORD]
    ), "Missing SMTP configuration in .env file"

    print(f"Configuration:")
    print(f"  Server:   {SMTP_SERVER}")
    print(f"  Port:     {SMTP_PORT}")
    print(f"  SSL:      {SMTP_USE_SSL}")
    print(f"  Username: {EMAIL_USERNAME}")
    print()

    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart

    print("Connecting to SMTP server...")

    if SMTP_USE_SSL:
        server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT)
    else:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()

    server.login(EMAIL_USERNAME, EMAIL_PASSWORD)
    print("✓ Successfully connected to SMTP server")
    print()

    # Skip interactive prompt during pytest
    import sys

    if "pytest" not in sys.modules:
        # Ask if user wants to send test email
        response = input("Would you like to send a test email? (y/n): ")
        if response.lower() == "y":
            recipient = input("Enter recipient email address: ")

            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = "Test Email from QuickBooks Check Processor"
            msg["From"] = EMAIL_USERNAME
            msg["To"] = recipient

            text = (
                "This is a test email from the QuickBooks Check Processor setup script."
            )
            html = """
            <html>
              <body>
                <h2>Test Email</h2>
                <p>This is a test email from the QuickBooks Check Processor setup script.</p>
                <p>If you received this, your SMTP configuration is working correctly!</p>
              </body>
            </html>
            """

            msg.attach(MIMEText(text, "plain"))
            msg.attach(MIMEText(html, "html"))

            # Send
            print(f"\nSending test email to {recipient}...")
            server.send_message(msg)
            print("✓ Test email sent successfully!")
            print()

    server.quit()
    print("✅ SMTP test successful!")


def main():
    print("=" * 70)
    print("Email Configuration Test")
    print("=" * 70)
    print()

    # Test IMAP
    imap_success = test_imap()

    # Test SMTP
    smtp_success = test_smtp()

    # Summary
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print(f"IMAP: {'✅ PASS' if imap_success else '❌ FAIL'}")
    print(f"SMTP: {'✅ PASS' if smtp_success else '❌ FAIL'}")
    print()

    if imap_success and smtp_success:
        print("✅ All tests passed! Email configuration is working correctly.")
        sys.exit(0)
    else:
        print("❌ Some tests failed. Please check your .env configuration.")
        sys.exit(1)


if __name__ == "__main__":
    main()
