"""AWS Bedrock specific retry strategy implementation."""

from typing import Mapping

import botocore.exceptions

from .base import BaseAsyncRetryer, BaseSyncRetryer, RetryProvider

# Bedrock-specific exceptions that should always be retried
_BEDROCK_RETRY_EXCEPTIONS = (
    botocore.exceptions.ReadTimeoutError,
    botocore.exceptions.ConnectTimeoutError,
    botocore.exceptions.EndpointConnectionError,
)


class BedrockRetryProvider(RetryProvider):
    """Provider for Bedrock specific exception handling."""

    def extract_headers_from_exception(
        self, exception: BaseException
    ) -> Mapping[str, str] | None:
        """Extract headers from botocore response structure."""
        if isinstance(exception, botocore.exceptions.ClientError):
            response = exception.response
            if "ResponseMetadata" in response:
                headers = response["ResponseMetadata"].get("HTTPHeaders", {})
                if headers:
                    return headers
        return None

    def extract_status_code(self, exception: BaseException) -> int | None:
        """Extract HTTP status code from botocore response."""
        if isinstance(exception, botocore.exceptions.ClientError):
            response = exception.response
            if "ResponseMetadata" in response:
                return response["ResponseMetadata"].get("HTTPStatusCode")
        return None

    def get_retry_exceptions(self) -> tuple[type[Exception], ...]:
        """Get Bedrock-specific exceptions that should always be retried."""
        return _BEDROCK_RETRY_EXCEPTIONS


class BedrockRetryer(BaseSyncRetryer):
    """Synchronous retry strategy for AWS Bedrock with botocore exception handling.

    Handles botocore-based exceptions and response structures from the boto3 SDK.

    Args:
        max_retries: Maximum number of retry attempts
        initial: Initial delay for exponential backoff in seconds
        max_delay: Maximum delay between retries in seconds
        logger: Optional logger for retry events
    """

    def get_retry_provider(self) -> RetryProvider:
        return BedrockRetryProvider()


class AsyncBedrockRetryer(BaseAsyncRetryer):
    """Asynchronous retry strategy for AWS Bedrock with botocore exception handling.

    Handles botocore-based exceptions and response structures from the boto3 SDK.

    Args:
        max_retries: Maximum number of retry attempts
        initial: Initial delay for exponential backoff in seconds
        max_delay: Maximum delay between retries in seconds
        logger: Optional logger for retry events
    """

    def get_retry_provider(self) -> RetryProvider:
        return BedrockRetryProvider()
