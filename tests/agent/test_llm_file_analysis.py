"""Tests for the user-facing errors of a rejected file-analysis request.

PC-5223: a Claude model rejecting a CSV with a 400 reached the user as
"Bad Request" / Unknown with "please try again later" -- nothing said which
file failed or that retrying cannot help. The file-analysis mapping names the
files and the fix, and like the generic mapping never quotes the provider's
message.
"""

import httpx
import pytest
from uipath.llm_client import UiPathAPIError, UiPathError, UiPathLLMErrorCode
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
)
from uipath_langchain.agent.exceptions.llm import (
    describe_attachment,
    raise_for_file_analysis_http_error,
    raise_for_llm_client_error,
    raise_for_provider_http_error,
)

_ANTHROPIC_MEDIA_TYPE_400: dict[str, object] = {
    "type": "error",
    "error": {
        "type": "invalid_request_error",
        "message": (
            "messages.0.content.1.document.source.base64.media_type: "
            "Input should be 'application/pdf'"
        ),
    },
}

_FILES = [describe_attachment("Budget.csv", "text/csv")]


def _api_error(status_code: int, body: dict[str, object]) -> UiPathAPIError:
    request = httpx.Request("POST", "https://llm-gateway/completions")
    response = httpx.Response(status_code, request=request, json=body)
    return UiPathAPIError.from_response(response)


def _raised(error: UiPathAPIError, *, all_pdf: bool = False) -> AgentRuntimeError:
    with pytest.raises(AgentRuntimeError) as exc_info:
        raise_for_file_analysis_http_error(error, _FILES, all_pdf=all_pdf)
    return exc_info.value


def test_describe_attachment() -> None:
    assert describe_attachment("a.csv", "text/csv") == "'a.csv' (text/csv)"
    assert describe_attachment("x", "") == "'x' (no file type)"


def test_rejected_file_is_a_user_file_error_naming_the_file() -> None:
    raised = _raised(_api_error(400, _ANTHROPIC_MEDIA_TYPE_400))

    info = raised.error_info
    assert info.code.endswith(AgentRuntimeErrorCode.FILE_ERROR.value)
    assert info.category == UiPathErrorCategory.USER
    assert info.status == 400
    assert info.title == "The model could not read the attached files"
    assert "'Budget.csv' (text/csv)" in info.detail
    assert "Convert the file to PDF or use a different model" in info.detail
    assert "try again later" not in info.detail
    # The provider's own message is never copied to the user.
    assert "media_type" not in info.detail
    assert isinstance(raised.__cause__, UiPathAPIError)


@pytest.mark.parametrize(
    "message",
    [
        "prompt is too long: 1046814 tokens > 1000000 maximum",
        "Input is too long for requested model.",
        "Your input exceeds the context window of this model.",
        "The input token count (1200000) exceeds the maximum number of tokens "
        "allowed (1048576).",
    ],
)
def test_context_window_400_says_the_file_is_too_large(message: str) -> None:
    raised = _raised(_api_error(400, {"error": {"message": message}}))

    info = raised.error_info
    assert info.title == "The attached files are too large for the model"
    assert "'Budget.csv' (text/csv)" in info.detail
    assert "context window" in info.detail
    assert message not in info.detail


def test_gateway_rejection_uses_the_generic_mapping() -> None:
    """A 400 with the gateway's own ProblemDetails is not about the files."""
    raised = _raised(_api_error(400, {"status": 400, "detail": "Model is disabled"}))

    info = raised.error_info
    assert info.code.endswith(AgentRuntimeErrorCode.LLM_PROVIDER_BAD_REQUEST.value)
    assert info.detail == "Model is disabled"


def test_other_statuses_use_the_generic_mapping() -> None:
    raised = _raised(_api_error(500, {"error": {"message": "boom"}}))

    info = raised.error_info
    assert info.code.endswith(AgentRuntimeErrorCode.HTTP_ERROR.value)
    assert info.category == UiPathErrorCategory.SYSTEM


def test_unsupported_mime_type_names_the_attachments() -> None:
    error = UiPathError(
        error_code=UiPathLLMErrorCode.UNSUPPORTED_MIME_TYPE,
        detail="Unsupported MIME type: application/vnd.ms-outlook",
    )

    with pytest.raises(AgentRuntimeError) as exc_info:
        raise_for_llm_client_error(
            error, [describe_attachment("mail.msg", "application/vnd.ms-outlook")]
        )

    detail = exc_info.value.error_info.detail
    assert "'mail.msg' (application/vnd.ms-outlook)" in detail
    assert detail.endswith(
        "Provider detail: Unsupported MIME type: application/vnd.ms-outlook"
    )


def test_unsupported_mime_type_without_attachments_keeps_the_old_wording() -> None:
    error = UiPathError(error_code=UiPathLLMErrorCode.UNSUPPORTED_MIME_TYPE)

    with pytest.raises(AgentRuntimeError) as exc_info:
        raise_for_llm_client_error(error)

    assert exc_info.value.error_info.detail.startswith(
        "The model does not support the file type of this attachment."
    )


def test_rejected_pdfs_are_not_told_to_convert_to_pdf() -> None:
    body: dict[str, object] = {
        "error": {"message": "A maximum of 100 PDF pages may be provided."}
    }
    raised = _raised(_api_error(400, body), all_pdf=True)

    info = raised.error_info
    assert info.title == "The model could not read the attached files"
    assert "Convert the file to PDF" not in info.detail
    assert "password-protected" in info.detail


def test_content_filter_rejection_uses_the_generic_mapping() -> None:
    """Azure OpenAI's content filter is about policy, not the file's format."""
    body: dict[str, object] = {
        "error": {
            "code": "content_filter",
            "message": "The response was filtered due to the prompt triggering "
            "Azure OpenAI's content management policy.",
        }
    }
    raised = _raised(_api_error(400, body))

    assert raised.error_info.code.endswith(
        AgentRuntimeErrorCode.LLM_PROVIDER_BAD_REQUEST.value
    )


@pytest.mark.parametrize("detail", ["", "   "])
def test_blank_gateway_detail_is_no_gateway_rejection(detail: str) -> None:
    raised = _raised(_api_error(400, {"status": 400, "detail": detail}))

    assert raised.error_info.title == "The model could not read the attached files"


def test_generic_mapping_ignores_a_blank_gateway_detail() -> None:
    with pytest.raises(AgentRuntimeError) as exc_info:
        raise_for_provider_http_error(_api_error(400, {"detail": "   "}))

    assert exc_info.value.error_info.detail.startswith(
        "The model provider rejected the request as invalid."
    )
