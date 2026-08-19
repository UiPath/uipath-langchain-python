"""Tests for raise_for_llm_client_error."""

import pytest
from uipath.llm_client import UiPathError, UiPathExecutionDeadlineError
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions import AgentRuntimeError, AgentRuntimeErrorCode
from uipath_langchain.agent.exceptions.llm import raise_for_llm_client_error


def test_execution_deadline_maps_to_agent_runtime_error():
    error = UiPathExecutionDeadlineError()

    with pytest.raises(AgentRuntimeError) as exc_info:
        raise_for_llm_client_error(error)

    error_info = exc_info.value.error_info
    assert AgentRuntimeErrorCode.EXECUTION_DEADLINE_EXCEEDED.value in error_info.code
    assert error_info.category == UiPathErrorCategory.SYSTEM
    assert error_info.detail == error.detail
    assert exc_info.value.__cause__ is error


def test_unknown_error_code_does_not_raise():
    raise_for_llm_client_error(UiPathError("boom", error_code=None))
