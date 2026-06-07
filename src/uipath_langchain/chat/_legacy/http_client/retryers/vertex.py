"""Vertex AI (Google GenAI) specific retry strategy implementation."""

from typing import Mapping

import httpx
from google.genai.errors import APIError

from .base import BaseAsyncRetryer, BaseSyncRetryer, RetryProvider

# Vertex-specific exceptions that should always be retried
_VERTEX_RETRY_EXCEPTIONS = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
)


class VertexRetryProvider(RetryProvider):
    """Provider for Vertex AI specific exception handling."""

    def extract_headers_from_exception(
        self, exception: BaseException
    ) -> Mapping[str, str] | None:
        """Extract headers from google.genai APIError response structure."""
        if isinstance(exception, APIError):
            if hasattr(exception.response, "headers"):
                return exception.response.headers
        return None

    def extract_status_code(self, exception: BaseException) -> int | None:
        """Extract HTTP status code from google.genai APIError."""
        if isinstance(exception, APIError):
            return exception.code
        return None

    def get_retry_exceptions(self) -> tuple[type[Exception], ...]:
        """Get Vertex-specific exceptions that should always be retried."""
        return _VERTEX_RETRY_EXCEPTIONS


class VertexRetryer(BaseSyncRetryer):
    """Synchronous retry strategy for Vertex AI with httpx exception handling.

    Handles httpx-based exceptions and response structures from the Google GenAI SDK.

    Args:
        max_retries: Maximum number of retry attempts
        initial: Initial delay for exponential backoff in seconds
        max_delay: Maximum delay between retries in seconds
        logger: Optional logger for retry events
    """

    def get_retry_provider(self) -> RetryProvider:
        return VertexRetryProvider()


class AsyncVertexRetryer(BaseAsyncRetryer):
    """Asynchronous retry strategy for Vertex AI with httpx exception handling.

    Handles httpx-based exceptions and response structures from the Google GenAI SDK.

    Args:
        max_retries: Maximum number of retry attempts
        initial: Initial delay for exponential backoff in seconds
        max_delay: Maximum delay between retries in seconds
        logger: Optional logger for retry events
    """

    def get_retry_provider(self) -> RetryProvider:
        return VertexRetryProvider()
