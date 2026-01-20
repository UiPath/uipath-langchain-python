"""Reusable tenacity retry strategy that honors HTTP Retry-After headers.

Supports only durations in the header, not datetimes.

Uses duck typing to extract retry information from exceptions,
to avoid importing exception types from the different chat SDKs.
"""

import logging
import random
from typing import Callable, Mapping

from tenacity import (
    AsyncRetrying,
    RetryCallState,
    Retrying,
    stop_after_attempt,
)

RETRYABLE_STATUS_CODES = {408, 429, 502, 503, 504}


def _extract_retry_after_header(
    exception: Exception | BaseException,
) -> float | None:
    def _parse_retry_after(header_value: str) -> float | None:
        try:
            seconds = float(header_value.strip())
            if seconds < 0:
                return None
            return seconds
        except (ValueError, AttributeError):
            return None

    def _extract_from_headers(headers: Mapping[str, str]) -> float | None:
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after:
            parsed = _parse_retry_after(retry_after)
            if parsed is not None:
                return parsed
        return None

    current: Exception | BaseException | None = exception
    while current:
        if hasattr(current, "response"):
            response = current.response

            # httpx, google.genai structure
            if hasattr(response, "headers"):
                result = _extract_from_headers(response.headers)
                if result is not None:
                    return result

            # botocore structure
            if isinstance(response, dict) and "ResponseMetadata" in response:
                headers = response.get("ResponseMetadata", {}).get("HTTPHeaders", {})
                result = _extract_from_headers(headers)
                if result is not None:
                    return result

        current = current.__cause__

    return None


def _extract_status_code(exception: Exception | BaseException) -> int | None:
    if hasattr(exception, "response"):
        response = exception.response
        # httpx, google.genai structure
        if hasattr(response, "status_code"):
            return response.status_code
        # botocore structure
        if isinstance(response, dict) and "ResponseMetadata" in response:
            return response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return None


def _is_retryable_exception(
    exception: Exception | BaseException,
    retry_on_exceptions: tuple[type[Exception], ...],
) -> bool:
    current: Exception | BaseException | None = exception
    while current is not None:
        if isinstance(current, retry_on_exceptions):
            # Exception that should always be retried
            return True

        status_code = _extract_status_code(current)
        if status_code is not None and status_code in RETRYABLE_STATUS_CODES:
            return True

        retry_after = _extract_retry_after_header(current)
        if retry_after is not None:
            # Always retry if server requested it
            return True

        current = current.__cause__
    return False


def _create_retry_condition(
    retry_on_exceptions: tuple[type[Exception], ...],
) -> Callable[[RetryCallState], bool]:
    def retry_condition(retry_state: RetryCallState) -> bool:
        if retry_state.outcome is None:
            return False
        exception = retry_state.outcome.exception()
        if exception is None:
            return False
        return _is_retryable_exception(exception, retry_on_exceptions)

    return retry_condition


def _create_retry_after_wait_strategy(
    initial: float = 5.0,
    max_delay: float = 180.0,
    logger: logging.Logger | None = None,
) -> Callable[[RetryCallState], float]:
    """Create a wait strategy that honors the Retry-After header if present, falling back to exponential backoff."""

    def _exponential_backoff(attempt: int, initial: float) -> float:
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


class RetryAfterHeaderStrategy(Retrying):
    """Synchronous retry strategy with Retry-After header support.

    Args:
        retry_on_exceptions: Exception types that should always be retried
        max_retries: Maximum number of retry attempts
        initial: Initial delay for exponential backoff in seconds
        max_delay: Maximum delay between retries in seconds
        logger: Optional logger for retry events
    """

    def __init__(
        self,
        retry_on_exceptions: tuple[type[Exception], ...] = (),
        max_retries: int = 5,
        initial: float = 5.0,
        max_delay: float = 120.0,
        logger: logging.Logger | None = None,
    ):
        super().__init__(
            wait=_create_retry_after_wait_strategy(
                initial=initial,
                max_delay=max_delay,
                logger=logger,
            ),
            retry=_create_retry_condition(retry_on_exceptions),
            stop=stop_after_attempt(max_retries),
            reraise=True,
        )


class AsyncRetryAfterHeaderStrategy(AsyncRetrying):
    """Asynchronous retry strategy with Retry-After header support.

    Args:
        retry_on_exceptions: Exception types that should always be retried
        max_retries: Maximum number of retry attempts
        initial: Initial delay for exponential backoff in seconds
        max_delay: Maximum delay between retries in seconds
        logger: Optional logger for retry events
    """

    def __init__(
        self,
        retry_on_exceptions: tuple[type[Exception], ...] = (),
        max_retries: int = 5,
        initial: float = 5.0,
        max_delay: float = 120.0,
        logger: logging.Logger | None = None,
    ):
        super().__init__(
            wait=_create_retry_after_wait_strategy(
                initial=initial,
                max_delay=max_delay,
                logger=logger,
            ),
            retry=_create_retry_condition(retry_on_exceptions),
            stop=stop_after_attempt(max_retries),
            reraise=True,
        )
