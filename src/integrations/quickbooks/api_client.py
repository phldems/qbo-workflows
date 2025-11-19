"""
QuickBooks API Client with robust error handling.

This module provides a wrapper around QuickBooks API requests with:
- Automatic retry logic with exponential backoff
- intuit_tid capture for troubleshooting
- Comprehensive error logging
- Token refresh handling
"""

import time
import json
from typing import Optional, Dict, Any, Literal
from loguru import logger
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError


class QuickBooksAPIError(Exception):
    """QuickBooks API error with detailed information"""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        intuit_tid: Optional[str] = None,
        error_code: Optional[str] = None,
        error_detail: Optional[str] = None,
        response_body: Optional[str] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.intuit_tid = intuit_tid
        self.error_code = error_code
        self.error_detail = error_detail
        self.response_body = response_body

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for logging"""
        return {
            "message": str(self),
            "status_code": self.status_code,
            "intuit_tid": self.intuit_tid,
            "error_code": self.error_code,
            "error_detail": self.error_detail,
            "response_body": self.response_body,
        }


class QuickBooksAPIClient:
    """
    Robust QuickBooks API client with error handling and retry logic.

    Features:
    - Automatic token refresh
    - Retry logic for transient errors
    - intuit_tid capture for all requests
    - Comprehensive error logging
    """

    def __init__(
        self,
        auth_manager,
        base_url: str,
        realm_id: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: int = 30,
    ):
        """
        Initialize API client.

        Args:
            auth_manager: QuickBooksAuth instance for token management
            base_url: API base URL
            realm_id: QuickBooks company/realm ID
            max_retries: Maximum retry attempts for failed requests
            retry_delay: Initial delay between retries (exponential backoff)
            timeout: Request timeout in seconds
        """
        self.auth_manager = auth_manager
        self.base_url = base_url.rstrip("/")
        self.realm_id = realm_id
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout

    def _get_headers(self, access_token: str) -> Dict[str, str]:
        """Get request headers"""
        return {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "CheckProcessingWorkflow/1.0",
        }

    def _log_error(
        self, method: str, url: str, error: QuickBooksAPIError, attempt: int
    ):
        """Log comprehensive error information"""
        logger.error(
            f"QuickBooks API Error [{method} {url}] (Attempt {attempt})",
            extra={
                "method": method,
                "url": url,
                "attempt": attempt,
                "status_code": error.status_code,
                "intuit_tid": error.intuit_tid,
                "error_code": error.error_code,
                "error_detail": error.error_detail,
                "error_dict": error.to_dict(),
            },
        )

    def _parse_error_response(self, response: requests.Response) -> QuickBooksAPIError:
        """
        Parse QuickBooks error response and extract details.

        Args:
            response: HTTP response object

        Returns:
            QuickBooksAPIError with extracted details
        """
        status_code = response.status_code
        intuit_tid = response.headers.get("intuit_tid", "N/A")

        try:
            error_data = response.json()
        except json.JSONDecodeError:
            error_data = None

        # Extract error details from QuickBooks error format
        error_code = None
        error_detail = None
        error_message = f"HTTP {status_code}"

        if error_data:
            # QuickBooks error format: {"Fault": {"Error": [...]}}
            fault = error_data.get("Fault", {})
            errors = fault.get("Error", [])

            if errors and len(errors) > 0:
                first_error = errors[0]
                error_code = first_error.get("code")
                error_detail = first_error.get("Detail", first_error.get("detail"))
                error_message = first_error.get(
                    "Message", first_error.get("message", error_message)
                )

        return QuickBooksAPIError(
            message=error_message,
            status_code=status_code,
            intuit_tid=intuit_tid,
            error_code=error_code,
            error_detail=error_detail,
            response_body=(
                response.text[:1000] if response.text else None
            ),  # Truncate for logging
        )

    def _should_retry(self, error: QuickBooksAPIError, attempt: int) -> bool:
        """
        Determine if request should be retried.

        Args:
            error: The error that occurred
            attempt: Current attempt number (0-indexed)

        Returns:
            True if should retry, False otherwise
        """
        # Don't retry if max attempts reached
        if attempt >= self.max_retries - 1:
            return False

        # Don't retry for permanent errors
        if error.status_code in [400, 401, 403, 404]:
            return False

        # Retry for transient errors
        # 429: Rate limit
        # 500-599: Server errors
        # Network errors: None status_code
        if error.status_code is None or error.status_code in [429, 500, 502, 503, 504]:
            return True

        return False

    def request(
        self,
        method: Literal["GET", "POST", "PUT", "DELETE"],
        endpoint: str,
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Make API request with retry logic and error handling.

        Args:
            method: HTTP method
            endpoint: API endpoint (relative to company/{realm_id}/)
            data: Request body (for POST/PUT)
            params: Query parameters

        Returns:
            Response data dictionary

        Raises:
            QuickBooksAPIError: If request fails after all retries
        """
        # Build full URL
        url = f"{self.base_url}/company/{self.realm_id}/{endpoint.lstrip('/')}"

        last_error = None

        for attempt in range(self.max_retries):
            try:
                # Get fresh access token (will auto-refresh if needed)
                access_token = self.auth_manager.get_valid_access_token()
                headers = self._get_headers(access_token)

                # Log request
                logger.debug(
                    f"QuickBooks API Request: {method} {url}",
                    extra={
                        "method": method,
                        "url": url,
                        "params": params,
                        "attempt": attempt + 1,
                    },
                )

                # Make request
                response = requests.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=data,
                    params=params,
                    timeout=self.timeout,
                )

                # Capture intuit_tid from response headers
                intuit_tid = response.headers.get("intuit_tid", "N/A")

                # Log response
                logger.debug(
                    f"QuickBooks API Response: {response.status_code}",
                    extra={
                        "method": method,
                        "url": url,
                        "status_code": response.status_code,
                        "intuit_tid": intuit_tid,
                        "attempt": attempt + 1,
                    },
                )

                # Check for success
                if response.ok:
                    # Successful response
                    try:
                        result = response.json()
                        # Add intuit_tid to result for tracking
                        if isinstance(result, dict):
                            result["_intuit_tid"] = intuit_tid
                        return result
                    except json.JSONDecodeError:
                        # Empty response or non-JSON
                        return {"_intuit_tid": intuit_tid}

                # Error response - parse error
                error = self._parse_error_response(response)
                last_error = error

                # Handle 401 Unauthorized - token may have expired
                if response.status_code == 401:
                    logger.warning("Received 401 Unauthorized, token may be expired")
                    # The next iteration will call get_valid_access_token() which will refresh

                # Log error
                self._log_error(method, url, error, attempt + 1)

                # Determine if should retry
                if not self._should_retry(error, attempt):
                    raise error

                # Wait before retry with exponential backoff
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2**attempt)
                    logger.info(
                        f"Retrying in {wait_time}s... (attempt {attempt + 2}/{self.max_retries})"
                    )
                    time.sleep(wait_time)

            except (Timeout, ConnectionError) as e:
                # Network errors
                last_error = QuickBooksAPIError(
                    message=f"Network error: {str(e)}", intuit_tid="N/A"
                )

                logger.error(
                    f"Network error during API request: {e}",
                    extra={
                        "method": method,
                        "url": url,
                        "attempt": attempt + 1,
                        "error_type": type(e).__name__,
                    },
                )

                # Retry network errors
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2**attempt)
                    logger.info(f"Retrying after network error in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise last_error

            except QuickBooksAPIError:
                # Re-raise QuickBooks errors
                raise

            except Exception as e:
                # Unexpected errors
                last_error = QuickBooksAPIError(
                    message=f"Unexpected error: {str(e)}", intuit_tid="N/A"
                )

                logger.error(
                    f"Unexpected error during API request: {e}",
                    exc_info=True,
                    extra={
                        "method": method,
                        "url": url,
                        "attempt": attempt + 1,
                        "error_type": type(e).__name__,
                    },
                )

                raise last_error

        # All retries exhausted
        if last_error:
            raise last_error
        else:
            raise QuickBooksAPIError(
                message=f"Request failed after {self.max_retries} attempts",
                intuit_tid="N/A",
            )

    def get(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make GET request"""
        return self.request("GET", endpoint, params=params)

    def post(self, endpoint: str, data: Dict) -> Dict[str, Any]:
        """Make POST request"""
        return self.request("POST", endpoint, data=data)

    def put(self, endpoint: str, data: Dict) -> Dict[str, Any]:
        """Make PUT request"""
        return self.request("PUT", endpoint, data=data)

    def delete(self, endpoint: str) -> Dict[str, Any]:
        """Make DELETE request"""
        return self.request("DELETE", endpoint)
