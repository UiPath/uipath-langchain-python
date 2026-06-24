"""Tests for normalizing provider HTTP errors into a common shape.

Each provider behind the LLM Gateway raises a different exception type, but all
carry the same gateway body (``{status, detail, ...}``). ``extract_provider_error``
reads the status code + detail off whichever attribute the SDK exposes, walking
the ``__cause__`` chain when LangChain wraps the SDK error.
"""

import pytest
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
)
from uipath_langchain.agent.exceptions.licensing import raise_for_provider_http_error
from uipath_langchain.chat.provider_errors import (
    ProviderError,
    extract_provider_error,
)

_DETAIL = "License not available for LLM usage. You need additional 'AGU'."
_BODY = {
    "title": "License not available",
    "status": 403,
    "detail": _DETAIL,
}


class TestExtractProviderError:
    def test_openai_status_code_and_body(self) -> None:
        class OpenAIError(Exception):
            status_code = 403
            body = _BODY

        result = extract_provider_error(OpenAIError("Forbidden"))
        assert result == ProviderError(status_code=403, detail=_DETAIL)

    def test_bedrock_uipath_api_error_same_shape_as_openai(self) -> None:
        # Bedrock via WrappedBotoClient surfaces as a UiPathAPIError (httpx
        # subclass) exposing OpenAI-style .status_code / .body.
        class UiPathPermissionDeniedError(Exception):
            status_code = 403
            body = _BODY

        result = extract_provider_error(UiPathPermissionDeniedError("Forbidden"))
        assert result == ProviderError(status_code=403, detail=_DETAIL)

    def test_vertex_wrapped_in_langchain_error(self) -> None:
        # google.genai exposes .code + .details; LangChain wraps it in a class
        # that itself exposes nothing, so the fields live on the __cause__.
        class GenAIError(Exception):
            code = 403
            details = _BODY

        class ChatGoogleGenerativeAIError(Exception):
            pass

        try:
            try:
                raise GenAIError("403")
            except GenAIError as cause:
                raise ChatGoogleGenerativeAIError("wrapped") from cause
        except ChatGoogleGenerativeAIError as wrapper:
            result = extract_provider_error(wrapper)

        assert result == ProviderError(status_code=403, detail=_DETAIL)

    def test_botocore_response_dict(self) -> None:
        # Legacy direct boto3 path: botocore.ClientError carries a response dict.
        class ClientError(Exception):
            response = {
                "ResponseMetadata": {"HTTPStatusCode": 403},
                "Error": {"Code": "AccessDenied", "detail": _BODY["detail"]},
            }

        result = extract_provider_error(ClientError("denied"))
        assert result.status_code == 403

    def test_none_returns_empty(self) -> None:
        result = extract_provider_error(None)
        assert result == ProviderError()
        assert not result

    def test_non_int_status_attribute_is_ignored(self) -> None:
        # An unrelated exception that happens to carry a string `code` must not
        # be mistaken for a provider HTTP error.
        class OSLike(Exception):
            code = "ENOENT"

        assert extract_provider_error(OSLike("nope")) == ProviderError()


class TestRaiseForProviderHttpError:
    def test_403_maps_to_license_not_available(self) -> None:
        class OpenAIError(Exception):
            status_code = 403
            body = _BODY

        with pytest.raises(AgentRuntimeError) as exc_info:
            raise_for_provider_http_error(OpenAIError("Forbidden"))

        info = exc_info.value.error_info
        assert info.status == 403
        assert info.category == UiPathErrorCategory.DEPLOYMENT
        assert info.code.endswith(AgentRuntimeErrorCode.LICENSE_NOT_AVAILABLE.value)
        assert _DETAIL in info.detail

    def test_other_status_falls_back_to_http_error_and_str(self) -> None:
        # Non-403 status, and no `detail` in the body → detail falls back to str(e).
        class OpenAIError(Exception):
            status_code = 500
            body: dict[str, str] = {}

        with pytest.raises(AgentRuntimeError) as exc_info:
            raise_for_provider_http_error(OpenAIError("boom"))

        info = exc_info.value.error_info
        assert info.status == 500
        assert info.code.endswith(AgentRuntimeErrorCode.HTTP_ERROR.value)
        assert "boom" in info.detail  # str(e) fallback

    @pytest.mark.parametrize(
        ("status_code", "expected_category"),
        [
            (401, UiPathErrorCategory.DEPLOYMENT),
            (403, UiPathErrorCategory.DEPLOYMENT),
            (408, UiPathErrorCategory.SYSTEM),
            (429, UiPathErrorCategory.SYSTEM),
            (500, UiPathErrorCategory.SYSTEM),
            (503, UiPathErrorCategory.SYSTEM),
            (400, UiPathErrorCategory.UNKNOWN),
            (404, UiPathErrorCategory.UNKNOWN),
        ],
    )
    def test_status_code_maps_to_expected_category(
        self, status_code: int, expected_category: UiPathErrorCategory
    ) -> None:
        class _ProviderError(Exception):
            def __init__(self, status: int) -> None:
                super().__init__("boom")
                self.status_code = status
                self.body: dict[str, str] = {}

        with pytest.raises(AgentRuntimeError) as exc_info:
            raise_for_provider_http_error(_ProviderError(status_code))

        info = exc_info.value.error_info
        assert info.status == status_code
        assert info.category == expected_category

    def test_no_status_does_not_raise(self) -> None:
        # No extractable HTTP status → no-op (the original exception is left to
        # propagate from the caller). Reaching the end without raising is the assert.
        raise_for_provider_http_error(ValueError("unrelated transport error"))
