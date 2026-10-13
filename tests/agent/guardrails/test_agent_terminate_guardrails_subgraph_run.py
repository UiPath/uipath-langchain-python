"""A compiled agent-output guardrail subgraph, run end to end with a validator that
reads files."""

import uuid
from typing import Any

import pytest
from langchain_core.messages import HumanMessage
from uipath.core.guardrails import (
    GuardrailValidationResult,
    GuardrailValidationResultType,
)
from uipath.platform.attachments import Attachment
from uipath.platform.guardrails import BuiltInValidatorGuardrail

from tests.agent.guardrails.test_guardrail_nodes import FakeUiPath
from uipath_langchain.agent.exceptions import AgentRuntimeError, AgentRuntimeErrorCode
from uipath_langchain.agent.guardrails.actions import BlockAction
from uipath_langchain.agent.react.guardrails.guardrails_subgraph import (
    create_agent_terminate_guardrails_subgraph,
)

_IN = "0b6f3a2d-5e4c-4b1a-8f9e-1d2c3b4a5f60"
_OUT = "5d1e2f3a-4b5c-4d6e-8f70-819a2b3c4d5e"


def _protected_code() -> BuiltInValidatorGuardrail:
    return BuiltInValidatorGuardrail.model_validate(
        {
            "$guardrailType": "builtInValidator",
            "id": "ip-code",
            "name": "Protected code",
            "description": "Blocks protected code in the agent's output.",
            "enabledForEvals": True,
            "selector": {"scopes": ["Agent"]},
            "validatorType": "intellectual_property",
            "validatorParameters": [
                {"$parameterType": "enum-list", "id": "ipEntities", "value": ["Code"]},
                {"$parameterType": "enum", "id": "appliesTo", "value": "Both"},
            ],
        }
    )


def _agent_result() -> dict[str, Any]:
    return {
        "summary": "Saved the snippet.",
        "code": {"ID": _OUT, "FullName": "out.py", "MimeType": "text/x-python"},
    }


def _build(monkeypatch, validation: GuardrailValidationResult):
    fake = FakeUiPath(validation)
    monkeypatch.setattr(
        "uipath_langchain.agent.guardrails.guardrail_nodes.UiPath", lambda: fake
    )
    run = create_agent_terminate_guardrails_subgraph(
        terminate_node=("terminate", lambda _state: _agent_result()),
        guardrails=[(_protected_code(), BlockAction("protected code"))],
    )
    return run, fake


def _state() -> dict[str, Any]:
    return {
        "messages": [HumanMessage("Save the code from the notes.")],
        "inner_state": {
            "job_attachments": {
                _IN: Attachment(
                    id=uuid.UUID(_IN), full_name="in.pdf", mime_type="application/pdf"
                ),
                _OUT: Attachment(
                    id=uuid.UUID(_OUT), full_name="out.py", mime_type="text/x-python"
                ),
            }
        },
    }


@pytest.mark.asyncio
async def test_passing_check_returns_the_result_and_judged_only_the_output_file(
    monkeypatch,
):
    run, fake = _build(
        monkeypatch,
        GuardrailValidationResult(
            result=GuardrailValidationResultType.PASSED, reason="clean"
        ),
    )

    result = await run(_state())

    assert result == _agent_result()
    assert fake.guardrails.call_count == 1
    assert fake.guardrails.last_text == str(_agent_result())
    assert [a.file_name for a in fake.guardrails.last_attachments] == ["out.py"]


@pytest.mark.asyncio
async def test_failing_check_blocks_and_names_the_flagged_output_file(monkeypatch):
    run, fake = _build(
        monkeypatch,
        GuardrailValidationResult.model_validate(
            {
                "result": GuardrailValidationResultType.VALIDATION_FAILED,
                "reason": "protected code",
                "flaggedAttachmentIds": [_OUT],
            }
        ),
    )

    with pytest.raises(AgentRuntimeError) as raised:
        await run(_state())

    assert raised.value.error_info.code == AgentRuntimeError.full_code(
        AgentRuntimeErrorCode.TERMINATION_GUARDRAIL_VIOLATION
    )
    assert "Flagged files: out.py" in raised.value.error_info.detail
    assert [a.file_name for a in fake.guardrails.last_attachments] == ["out.py"]
