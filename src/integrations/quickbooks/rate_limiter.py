"""
Rate limiter for QuickBooks API calls.
Prevents exceeding API rate limits.
"""

import time
from collections import deque
from threading import Lock
from loguru import logger


class QBOAPIRateLimiter:
    """
    Rate limiter for QuickBooks Online API.

    QBO Limits:
    - 500 requests per minute per realmId
    - 10 concurrent requests per second per realmId
    """

    def __init__(
        self,
        max_per_minute: int = 450,  # Buffer below 500 limit
        max_concurrent: int = 8,  # Buffer below 10 limit
    ):
        """
        Initialize rate limiter.

        Args:
            max_per_minute: Maximum requests per minute
            max_concurrent: Maximum concurrent requests
        """
        self.max_per_minute = max_per_minute
        self.max_concurrent = max_concurrent
        self.requests = deque()
        self.lock = Lock()
        self.concurrent_count = 0

        logger.info(
            f"Initialized QBO rate limiter "
            f"(max: {max_per_minute}/min, {max_concurrent} concurrent)"
        )

    def wait_if_needed(self):
        """Wait if rate limit would be exceeded."""
        with self.lock:
            now = time.time()

            # Remove requests older than 1 minute
            while self.requests and self.requests[0] < now - 60:
                self.requests.popleft()

            # Wait if at per-minute rate limit
            if len(self.requests) >= self.max_per_minute:
                sleep_time = 60 - (now - self.requests[0])
                if sleep_time > 0:
                    logger.warning(f"Rate limit reached, sleeping {sleep_time:.2f}s")
                    time.sleep(sleep_time)

                # Clear old requests after sleeping
                now = time.time()
                while self.requests and self.requests[0] < now - 60:
                    self.requests.popleft()

            # Wait if concurrent limit reached
            while self.concurrent_count >= self.max_concurrent:
                time.sleep(0.1)

            # Record this request
            self.requests.append(now)
            self.concurrent_count += 1

    def request_complete(self):
        """Mark a request as complete."""
        with self.lock:
            self.concurrent_count = max(0, self.concurrent_count - 1)
