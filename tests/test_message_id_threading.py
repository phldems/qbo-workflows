"""
Test script to verify Message-ID normalization in EmailSender._send_email.
This script monkeypatches smtplib.SMTP_SSL/SMTP to a dummy that prints the constructed
message headers instead of actually sending email.

Run: python tests/manual/test_message_id_threading.py
"""

import sys
import smtplib
from pathlib import Path
from email import message_from_string

# Ensure project root is on sys.path so `src` package is importable
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


# Dummy SMTP classes to avoid network calls
class DummySMTP:
    def __init__(self, server, port, context=None):
        self.server = server
        self.port = port
        self.context = context
        print(f"[DummySMTP] init server={server} port={port} context={context}")

    def __enter__(self):
        print("[DummySMTP] enter")
        return self

    def __exit__(self, exc_type, exc, tb):
        print("[DummySMTP] exit")

    def login(self, username, password):
        print(f"[DummySMTP] login user={username}")

    def sendmail(self, from_addr, to_addrs, msg_str):
        print(f"[DummySMTP] sendmail from={from_addr} to={to_addrs}")
        # Print extracted headers for quick inspection
        msg = message_from_string(msg_str)
        for h in ("Subject", "From", "To", "In-Reply-To", "References"):
            print(f"Header {h}: {msg.get(h)}")
        print("--- Raw message start ---")
        print(msg_str.splitlines()[:20])
        print("--- Raw message end ---\n")

    def send_message(self, msg):
        """Send an email message."""
        print(f"[DummySMTP] send_message to={msg['To']}")

    def quit(self):
        """Close the SMTP connection."""
        print("[DummySMTP] quit")


# Patch smtplib classes
smtplib.SMTP_SSL = DummySMTP
smtplib.SMTP = DummySMTP

from src.integrations.email.sender import EmailSender

sender = EmailSender(
    smtp_server="smtp.example.com",
    smtp_port=465,
    username="bot@example.com",
    password="fakepassword",
    use_ssl=True,
)

recipient = "user@example.com"
subject = "Test Threading"
text = "This is a test"
html = "<p>This is a test</p>"

test_ids = [
    "abc123@example.com",
    "<abc123@example.com>",
    ' "<abc123@example.com>" ',
    " <abc123@example.com> ",
    "<multiple@one> <multiple@two>",
]

for tid in test_ids:
    print(f"\n=== Testing reply_to_message_id: {repr(tid)} ===")
    try:
        sender._send_email(
            recipient, subject, text, html, None, reply_to_message_id=tid
        )
    except Exception as e:
        print(f"Exception while sending: {e}")

print("Test script complete")
