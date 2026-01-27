"""Base retry strategy with common functionality for all providers.

Provides abstract base classes for provider-specific retry implementations.
Each provider should subclass and implement the abstract methods for
extracting retry information from their specific exception types.
"""

import logging
import random
from abc import ABC, abstractmethod
from typing import Callable, Mapping

from tenacity import (
    AsyncRetrying,
    RetryCallState,
    Retrying,
    stop_after_attempt,
)

RETRYABLE_STATUS_CODES = {408, 429, 502, 503, 504}


class RetryProvider(ABC):
    """Interface for provider-specific retry logic."""

    @abstractmethod
    def extract_headers_from_exception(
        self, exception: BaseException
    ) -> Mapping[str, str] | None:
        """Extract headers from provider-specific exception response structure."""
        pass

    @abstractmethod
    def extract_status_code(self, exception: BaseException) -> int | None:
        """Extract HTTP status code from provider-specific exception."""
        pass

    @abstractmethod
    def get_retry_exceptions(self) -> tuple[type[Exception], ...]:
        """Get the tuple of exception types that should always be retried."""
        pass


def _create_retry_condition(
    retry_provider: RetryProvider,
) -> Callable[[RetryCallState], bool]:
    """Create retry condition for tenacity."""

    def _is_retryable_exception(exception: BaseException) -> bool:
        """Determine if an exception should be retried."""
        retry_on_exceptions = retry_provider.get_retry_exceptions()
        current: BaseException | None = exception

        while current is not None:
            if isinstance(current, retry_on_exceptions):
                return True

            status_code = retry_provider.extract_status_code(current)
            if status_code is not None and status_code in RETRYABLE_STATUS_CODES:
                return True

            # Check for Retry-After header (implies server wants retry)
            headers = retry_provider.extract_headers_from_exception(current)
            if headers:
                retry_after = headers.get("retry-after") or headers.get("Retry-After")
                if retry_after:
                    return True

            current = current.__cause__

        return False

    def retry_condition(retry_state: RetryCallState) -> bool:
        if retry_state.outcome is None:
            return False
        exception = retry_state.outcome.exception()
        if exception is None:
            return False
        return _is_retryable_exception(exception)

    return retry_condition


def _create_wait_strategy(
    retry_provider: RetryProvider,
    initial: float = 5.0,
    max_delay: float = 180.0,
    logger: logging.Logger | None = None,
) -> Callable[[RetryCallState], float]:
    """Create wait strategy honoring Retry-After header with exponential backoff fallback."""

    def _parse_retry_after(header_value: str) -> float | None:
        """Parse Retry-After header value (durations only, not datetimes)."""
        try:
            seconds = float(header_value.strip())
            if seconds < 0:
                return None
            return seconds
        except (ValueError, AttributeError):
            return None

    def _extract_retry_after_header(exception: BaseException) -> float | None:
        """Extract and parse Retry-After header from exception chain."""
        current: BaseException | None = exception
        while current:
            headers = retry_provider.extract_headers_from_exception(current)
            if headers:
                retry_after = headers.get("retry-after") or headers.get("Retry-After")
                if retry_after:
                    parsed = _parse_retry_after(retry_after)
                    if parsed is not None:
                        return parsed
            current = current.__cause__
        return None

    def _exponential_backoff(attempt: int, initial: float) -> float:
        """Calculate exponential backoff with jitter."""
        exponent = attempt - 1
        exponential = initial * (2**exponent)
        jitter = random.uniform(0, 1.0)
        return exponential + jitter

    def wait_strategy(retry_state: RetryCallState) -> float:
        """Calculate wait time based on exception and retry state."""
        if retry_state.outcome is None:
            return initial

        exception = retry_state.outcome.exception()
        if exception is not None:
            retry_after = _extract_retry_after_header(exception)
            if retry_after is not None:
                capped_wait = min(retry_after, max_delay)
                if logger:
                    logger.info(
                        f"Retrying after {retry_after:.1f}s"
                        f"{f' (capped to {capped_wait:.1f}s)' if capped_wait != retry_after else ''}"
                    )
                return capped_wait

        exponential_wait = _exponential_backoff(retry_state.attempt_number, initial)
        capped_wait = min(exponential_wait, max_delay)
        if logger:
            logger.info(
                f"Retrying with exponential backoff after {capped_wait:.1f}s (attempt #{retry_state.attempt_number})"
            )
        return capped_wait

    return wait_strategy


class BaseSyncRetryer(Retrying, ABC):
    """Synchronous retry strategy base class.

    Args:
        max_retries: Maximum number of retry attempts
        initial: Initial delay for exponential backoff in seconds
        max_delay: Maximum delay between retries in seconds
        logger: Optional logger for retry events
    """

    @abstractmethod
    def get_retry_provider(self) -> RetryProvider:
        """Return the provider instance for this retryer."""
        pass

    def __init__(
        self,
        max_retries: int = 5,
        initial: float = 5.0,
        max_delay: float = 120.0,
        logger: logging.Logger | None = None,
    ):
        retry_provider = self.get_retry_provider()

        Retrying.__init__(
            self,
            wait=_create_wait_strategy(retry_provider, initial, max_delay, logger),
            retry=_create_retry_condition(retry_provider),
            stop=stop_after_attempt(max_retries),
            reraise=True,
        )


class BaseAsyncRetryer(AsyncRetrying, ABC):
    """Asynchronous retry strategy base class.

    Args:
        max_retries: Maximum number of retry attempts
        initial: Initial delay for exponential backoff in seconds
        max_delay: Maximum delay between retries in seconds
        logger: Optional logger for retry events
    """

    @abstractmethod
    def get_retry_provider(self) -> RetryProvider:
        """Return the provider instance for this retryer."""
        pass

    def __init__(
        self,
        max_retries: int = 5,
        initial: float = 5.0,
        max_delay: float = 120.0,
        logger: logging.Logger | None = None,
    ):
        retry_provider = self.get_retry_provider()

        AsyncRetrying.__init__(
            self,
            wait=_create_wait_strategy(retry_provider, initial, max_delay, logger),
            retry=_create_retry_condition(retry_provider),
            stop=stop_after_attempt(max_retries),
            reraise=True,
        )
