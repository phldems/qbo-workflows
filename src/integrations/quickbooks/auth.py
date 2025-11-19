"""
QuickBooks Online OAuth 2.0 authentication.
Handles token management and refresh with robust error handling.
"""

import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from loguru import logger

try:
    from intuitlib.client import AuthClient
    from intuitlib.enums import Scopes

    INTUITLIB_AVAILABLE = True
except ImportError:
    INTUITLIB_AVAILABLE = False
    logger.warning("intuitlib not available")


class QuickBooksAuth:
    """Manages QuickBooks OAuth 2.0 authentication."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        redirect_uri: str,
        environment: str = "production",
        database=None,
    ):
        """
        Initialize QuickBooks OAuth client.

        Args:
            client_id: QBO OAuth client ID
            client_secret: QBO OAuth client secret
            redirect_uri: OAuth redirect URI
            environment: 'production' or 'sandbox'
            database: Database instance for token storage
        """
        if not INTUITLIB_AVAILABLE:
            raise ImportError("intuitlib not available. Install: pip install intuitlib")

        self.client_id = client_id
        self.client_secret = client_secret
        self.redirect_uri = redirect_uri
        self.environment = environment
        self.database = database

        self.auth_client = AuthClient(
            client_id=client_id,
            client_secret=client_secret,
            environment=environment,
            redirect_uri=redirect_uri,
        )

        logger.info(f"Initialized QuickBooks OAuth client ({environment})")

    def get_authorization_url(self, state: Optional[str] = None) -> str:
        """
        Get authorization URL for user to grant access.

        Args:
            state: Optional state parameter for security

        Returns:
            Authorization URL
        """
        scopes = [Scopes.ACCOUNTING]  # Access to accounting data

        if state:
            auth_url = self.auth_client.get_authorization_url(scopes, state=state)
        else:
            auth_url = self.auth_client.get_authorization_url(scopes)

        logger.info("Generated authorization URL")
        return auth_url

    def exchange_code_for_tokens(self, auth_code: str, realm_id: str) -> Dict[str, Any]:
        """
        Exchange authorization code for access and refresh tokens.
        This should only be called ONCE during initial authorization.

        Args:
            auth_code: Authorization code from callback
            realm_id: Realm ID (Company ID) from callback

        Returns:
            Dictionary with tokens and metadata
        """
        logger.info(f"Exchanging authorization code for tokens (realm: {realm_id})")

        try:
            self.auth_client.get_bearer_token(auth_code, realm_id=realm_id)

            # Calculate expiration time
            expires_at = datetime.utcnow() + timedelta(seconds=3600)  # 1 hour

            tokens = {
                "access_token": self.auth_client.access_token,
                "refresh_token": self.auth_client.refresh_token,
                "realm_id": realm_id,
                "expires_at": expires_at.isoformat(),
            }

            # Store in database
            if self.database:
                self.database.store_oauth_token(
                    service="quickbooks",
                    access_token=tokens["access_token"],
                    refresh_token=tokens["refresh_token"],
                    expires_at=tokens["expires_at"],
                    realm_id=realm_id,
                )

            logger.info("Successfully exchanged code for tokens")
            return tokens

        except Exception as e:
            logger.error(f"Failed to exchange code for tokens: {e}")
            raise

    def refresh_access_token(
        self,
        refresh_token: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Refresh expired access token using refresh token with retry logic.

        Args:
            refresh_token: Optional refresh token (loaded from DB if not provided)
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (exponential backoff)

        Returns:
            Dictionary with new tokens

        Raises:
            ValueError: If no refresh token available
            Exception: If refresh fails after all retries
        """
        # Load from database if not provided
        if not refresh_token and self.database:
            stored_token = self.database.get_oauth_token("quickbooks")
            if stored_token:
                refresh_token = stored_token["refresh_token"]
            else:
                raise ValueError("No refresh token found in database")

        if not refresh_token:
            raise ValueError("No refresh token provided")

        logger.info("Refreshing access token")

        last_exception = None

        for attempt in range(max_retries):
            try:
                # Attempt to refresh
                self.auth_client.refresh(refresh_token=refresh_token)

                # Calculate new expiration (QuickBooks tokens expire in 1 hour)
                expires_at = datetime.utcnow() + timedelta(seconds=3600)

                tokens = {
                    "access_token": self.auth_client.access_token,
                    "refresh_token": self.auth_client.refresh_token,
                    "expires_at": expires_at.isoformat(),
                }

                # Update database
                if self.database:
                    stored_token = self.database.get_oauth_token("quickbooks")
                    realm_id = stored_token.get("realm_id") if stored_token else None

                    self.database.store_oauth_token(
                        service="quickbooks",
                        access_token=tokens["access_token"],
                        refresh_token=tokens["refresh_token"],
                        expires_at=tokens["expires_at"],
                        realm_id=realm_id,
                    )

                logger.info("Successfully refreshed access token")
                return tokens

            except Exception as e:
                last_exception = e

                # Check if this is a permanent error (invalid refresh token)
                error_str = str(e).lower()
                if (
                    "invalid" in error_str
                    or "revoked" in error_str
                    or "expired" in error_str
                ):
                    logger.error(f"Refresh token is invalid/revoked/expired: {e}")
                    logger.error("You need to re-authorize the application")
                    raise ValueError(
                        "Refresh token is invalid. Please run OAuth flow again to re-authorize."
                    ) from e

                # Retry for transient errors
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2**attempt)  # Exponential backoff
                    logger.warning(
                        f"Failed to refresh token (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(
                        f"Failed to refresh access token after {max_retries} attempts: {e}"
                    )

        # All retries exhausted
        raise Exception(
            f"Failed to refresh access token after {max_retries} attempts"
        ) from last_exception

    def get_valid_access_token(self) -> str:
        """
        Get a valid access token, refreshing if necessary.

        Returns:
            Valid access token

        Raises:
            ValueError: If no tokens found or refresh fails
        """
        if not self.database:
            raise ValueError("Database required for token management")

        # Load stored token
        stored_token = self.database.get_oauth_token("quickbooks")

        if not stored_token:
            raise ValueError("No QuickBooks tokens found. Please authorize first.")

        # Check if token is expired
        expires_at = datetime.fromisoformat(stored_token["expires_at"])
        now = datetime.utcnow()

        # Refresh if expired or expiring soon (within 5 minutes)
        if now >= expires_at - timedelta(minutes=5):
            logger.info("Access token expired or expiring soon, refreshing...")
            tokens = self.refresh_access_token(stored_token["refresh_token"])
            return tokens["access_token"]

        return stored_token["access_token"]

    def get_realm_id(self) -> Optional[str]:
        """
        Get stored realm ID from database.

        Returns:
            Realm ID or None if not found
        """
        if not self.database:
            return None

        stored_token = self.database.get_oauth_token("quickbooks")
        return stored_token.get("realm_id") if stored_token else None

    def revoke_tokens(self):
        """Revoke access and refresh tokens."""
        if not self.database:
            logger.warning("No database available to revoke tokens")
            return

        try:
            stored_token = self.database.get_oauth_token("quickbooks")

            if stored_token and stored_token.get("refresh_token"):
                self.auth_client.revoke(token=stored_token["refresh_token"])
                logger.info("Successfully revoked tokens")

            # Clear from database (you might want to delete or mark as revoked)
            # For now, we'll just log

        except Exception as e:
            logger.error(f"Failed to revoke tokens: {e}")
            raise
