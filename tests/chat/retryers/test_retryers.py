"""Unit tests for retry-after-aware retry strategy."""

from typing import NoReturn, cast
from unittest.mock import MagicMock, patch

import botocore.exceptions
import httpx
from google.genai import errors as genai_errors
from langchain_google_genai.chat_models import ChatGoogleGenerativeAIError
from tenacity import wait_none

from uipath_langchain.chat.retryers.bedrock import (
    AsyncBedrockRetryer,
)
from uipath_langchain.chat.retryers.vertex import (
    AsyncVertexRetryer,
    VertexRetryer,
)


def raise_boto3_error(
    error_code: str,
    message: str,
    status_code: int = 429,
    headers: dict[str, str] | None = None,
) -> NoReturn:
    """Raise a botocore ClientError for testing."""
    error_response_dict = {
        "Error": {
            "Code": error_code,
            "Message": message,
        },
        "ResponseMetadata": {
            "HTTPStatusCode": status_code,
            "HTTPHeaders": headers or {},
        },
    }
    raise botocore.exceptions.ClientError(
        cast("botocore.exceptions._ClientErrorResponseTypeDef", error_response_dict),
        "test_operation",
    )


def raise_google_genai_error(
    status_code: int = 429,
    message: str = "Resource exhausted",
    headers: dict[str, str] | None = None,
) -> NoReturn:
    """Raise a ChatGoogleGenerativeAIError wrapping google.genai.errors.ClientError."""
    response = MagicMock()
    response.status_code = status_code
    response.headers = headers or {}
    response.text = message

    response_json = {"error": {"message": message}}

    client_error = genai_errors.ClientError(
        code=status_code,
        response_json=response_json,
        response=response,
    )
    genai_error = ChatGoogleGenerativeAIError(message)
    genai_error.__cause__ = client_error
    raise genai_error


class TestVertexRetryStrategy:
    """Tests for VertexRetryStrategy and AsyncVertexRetryStrategy classes."""

    def test_retry_strategy_retries_on_exception(
        self,
    ) -> None:
        call_count = 0

        def failing_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise_google_genai_error(429)
            return "Success"

        retryer = VertexRetryer(max_retries=5)
        retryer.wait = wait_none()

        result = retryer(failing_function)

        assert result == "Success"
        assert call_count == 3  # Failed twice, succeeded on third attempt

    async def test_async_retry_strategy_retries_on_exception(
        self,
    ) -> None:
        call_count = 0

        async def failing_async_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise_google_genai_error(429)
            return "Success"

        retryer = AsyncVertexRetryer(max_retries=5)
        retryer.wait = wait_none()

        result: str = await retryer(failing_async_function)

        assert result == "Success"
        assert call_count == 3  # Failed twice, succeeded on third attempt

    async def test_no_header_but_retryable_status_code_falls_back_to_exponential_backoff(
        self,
    ) -> None:
        call_count = 0
        sleep_delays: list[float] = []

        async def always_failing_async_function():
            nonlocal call_count
            call_count += 1
            raise_google_genai_error(503)

        retryer = AsyncVertexRetryer(max_retries=4, initial=5.0, max_delay=60.0)

        with patch("asyncio.sleep") as mock_sleep:
            mock_sleep.side_effect = lambda delay: sleep_delays.append(delay)
            try:
                await retryer(always_failing_async_function)
            except ChatGoogleGenerativeAIError:
                pass

        assert call_count == 4
        assert len(sleep_delays) == 3

        # Verify exponential growth: 5s, 10s, 20s (with small jitter tolerance)
        assert 5.0 <= sleep_delays[0] <= 6.0
        assert 10.0 <= sleep_delays[1] <= 11.0
        assert 20.0 <= sleep_delays[2] <= 21.0

    async def test_no_status_code_but_retryable_exception_falls_back_to_exponential_backoff(
        self,
    ) -> None:
        call_count = 0
        sleep_delays: list[float] = []

        async def always_failing_async_function():
            nonlocal call_count
            call_count += 1
            raise httpx.TimeoutException("Timeout")

        retryer = AsyncVertexRetryer(
            max_retries=4,
            initial=5.0,
            max_delay=60.0,
        )

        with patch("asyncio.sleep") as mock_sleep:
            mock_sleep.side_effect = lambda delay: sleep_delays.append(delay)
            try:
                await retryer(always_failing_async_function)
            except httpx.TimeoutException:
                pass

        assert call_count == 4
        assert len(sleep_delays) == 3

        # Verify exponential growth: 5s, 10s, 20s (with small jitter tolerance)
        assert 5.0 <= sleep_delays[0] <= 6.0
        assert 10.0 <= sleep_delays[1] <= 11.0
        assert 20.0 <= sleep_delays[2] <= 21.0

    async def test_async_retry_strategy_respects_max_retries(
        self,
    ) -> None:
        """Test that the async retry strategy respects max_retries."""
        call_count = 0

        async def always_failing_async_function():
            nonlocal call_count
            call_count += 1
            raise_google_genai_error(503)

        retryer = AsyncVertexRetryer(max_retries=3)
        retryer.wait = wait_none()

        try:
            await retryer(always_failing_async_function)
            raise AssertionError("Should have raised an exception")
        except ChatGoogleGenerativeAIError:
            pass

        assert call_count == 3  # Should stop after 3 attempts

    async def test_retry_strategy_uses_retry_after_header_with_google_genai(
        self,
    ) -> None:
        call_count = 0
        sleep_delays: list[float] = []

        async def function_with_retry_after():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise_google_genai_error(
                    429,
                    "Resource exhausted",
                    headers={"retry-after": "30"},
                )
            return "Success"

        retryer = AsyncVertexRetryer(max_retries=3)

        with patch("asyncio.sleep") as mock_sleep:
            mock_sleep.side_effect = lambda delay: sleep_delays.append(delay)
            result: str = await retryer(function_with_retry_after)

        assert result == "Success"
        assert call_count == 2
        assert len(sleep_delays) == 1
        assert sleep_delays[0] == 30.0

    async def test_retry_strategy_uses_retry_after_header_with_vertex(self) -> None:
        call_count = 0
        sleep_delays: list[float] = []

        async def function_with_retry_after():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise_google_genai_error(
                    429,
                    "Resource exhausted",
                    headers={"retry-after": "45"},
                )
            return "Success"

        retryer = AsyncVertexRetryer(max_retries=3)

        with patch("asyncio.sleep") as mock_sleep:
            mock_sleep.side_effect = lambda delay: sleep_delays.append(delay)
            result: str = await retryer(function_with_retry_after)

        assert result == "Success"
        assert call_count == 2
        assert len(sleep_delays) == 1
        assert sleep_delays[0] == 45.0


class TestBedrockRetryStrategy:
    """Tests for BedrockRetryer and AsyncBedrockRetryer classes."""

    async def test_retry_strategy_uses_retry_after_header_with_bedrock(self) -> None:
        call_count = 0
        sleep_delays: list[float] = []

        async def function_with_retry_after():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise_boto3_error(
                    "ThrottlingException",
                    "Rate exceeded",
                    headers={"retry-after": "25"},
                )
            return "Success"

        retryer = AsyncBedrockRetryer(max_retries=3)

        with patch("asyncio.sleep") as mock_sleep:
            mock_sleep.side_effect = lambda delay: sleep_delays.append(delay)
            result: str = await retryer(function_with_retry_after)

        assert result == "Success"
        assert call_count == 2
        assert len(sleep_delays) == 1
        assert sleep_delays[0] == 25.0
