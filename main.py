"""
Main entry point for check processing workflow.
"""

import sys
import time
from datetime import datetime
from loguru import logger
from src.config.settings import get_settings
from src.workflows.check_processor import CheckProcessor
from src.utils.database import Database
from src.integrations.quickbooks.auth import QuickBooksAuth


def setup_logging(settings):
    """Configure logging."""
    logger.remove()  # Remove default handler

    # Console logging
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    )

    # File logging
    logger.add(
        settings.log_file,
        rotation=settings.log_rotation,
        retention=settings.log_retention,
        level=settings.log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
    )

    logger.info("Logging configured")


def main():
    """Main application entry point."""
    # Load settings
    settings = get_settings()

    # Setup logging
    setup_logging(settings)

    logger.info("Starting Check Processing Workflow")
    logger.info(f"Environment: {settings.qbo_environment}")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"Dry run: {settings.dry_run}")

    _ensure_qbo_token_seeded(settings)

    # Initialize processor
    processor = CheckProcessor(settings)

    # Ensure email folders exist
    processor.email_monitor.create_folders(
        [
            settings.email_processed_folder,
            settings.email_failed_folder,
            settings.email_manual_review_folder,
            settings.email_duplicates_folder,
        ]
    )

    if settings.dry_run:
        logger.warning("DRY RUN MODE - No actual changes will be made")

    # Main processing loop
    logger.info(f"Starting polling loop (interval: {settings.imap_poll_interval}s)")

    try:
        while True:
            try:
                processor.process_new_checks()
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.error(f"Error in processing cycle: {e}")

            # Wait before next poll
            time.sleep(settings.imap_poll_interval)

    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        sys.exit(1)


def _ensure_qbo_token_seeded(settings):
    """
    Ensure the QuickBooks refresh token exists in the database.

    If the database lacks a refresh token but the .env provides one, seed it and
    immediately refresh so the stored record contains a valid access token.
    """
    try:
        db = Database(settings.database_path)
    except Exception:
        logger.exception("Unable to initialize database while checking QBO tokens")
        return

    existing = db.get_oauth_token("quickbooks")
    if existing and existing.get("refresh_token"):
        logger.info(
            "QuickBooks token already present in DB (updated %s)",
            existing.get("updated_at"),
        )
        return

    env_refresh = settings.qbo_refresh_token
    if not env_refresh:
        logger.error(
            "No QuickBooks refresh token found in DB or environment. "
            "Run the OAuth helper to authorize this app."
        )
        return

    logger.warning("Seeding QuickBooks refresh token from environment")
    placeholder_exp = datetime.utcnow().isoformat()
    try:
        db.store_oauth_token(
            service="quickbooks",
            access_token="",
            refresh_token=env_refresh,
            expires_at=placeholder_exp,
            realm_id=settings.qbo_realm_id,
        )
    except Exception:
        logger.exception("Failed to seed QuickBooks token into database")
        return

    try:
        qb_auth = QuickBooksAuth(
            client_id=settings.qbo_client_id,
            client_secret=settings.qbo_client_secret,
            redirect_uri=settings.qbo_redirect_uri,
            environment=settings.qbo_environment,
            database=db,
        )
        qb_auth.refresh_access_token(refresh_token=env_refresh)
        logger.info("QuickBooks token seeded and refreshed successfully")
    except Exception:
        logger.exception(
            "Seeded refresh token but failed to refresh access token. "
            "Run the OAuth helper to re-authorize."
        )


if __name__ == "__main__":
    main()
