"""Map normalized LLM-client errors into agent runtime errors."""

from uipath.llm_client import UiPathError, UiPathLLMErrorCode
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
)


def raise_for_llm_client_error(error: UiPathError) -> None:
    """Raise a structured agent error for known LLM-client error codes."""
    if error.error_code == UiPathLLMErrorCode.UNSUPPORTED_MIME_TYPE:
        raise AgentRuntimeError(
            code=AgentRuntimeErrorCode.FILE_ERROR,
            title="Unsupported file attachment format.",
            detail=(
                "The model does not support this attachment's file type. "
                "Remove the attachment or convert it to a supported format."
                + (f" Provider detail: {error.detail}" if error.detail else "")
            ),
            category=UiPathErrorCategory.USER,
        ) from error

    if error.error_code == UiPathLLMErrorCode.EXECUTION_DEADLINE_EXCEEDED:
        raise AgentRuntimeError(
            code=AgentRuntimeErrorCode.EXECUTION_DEADLINE_EXCEEDED,
            title="Agent run reached its execution time limit.",
            detail=(
                error.detail
                or "The run's execution time limit was reached before the LLM call could complete."
            ),
            category=UiPathErrorCategory.SYSTEM,
            should_wrap=False,
        ) from error
