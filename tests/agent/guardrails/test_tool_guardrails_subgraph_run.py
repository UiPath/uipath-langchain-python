"""A compiled tool-guardrail subgraph, run end to end with a judge that reads files."""

import json
import uuid
from typing import Any

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel
from uipath.core.guardrails import GuardrailValidationResultType
from uipath.platform.attachments import Attachment
from uipath.platform.guardrails import BuiltInValidatorGuardrail

from tests.agent.guardrails.test_guardrail_nodes import _patch_uipath
from uipath_langchain.agent.exceptions import AgentRuntimeError, AgentRuntimeErrorCode
from uipath_langchain.agent.guardrails.actions import BlockAction
from uipath_langchain.agent.react.guardrails.guardrails_subgraph import (
    create_tool_guardrails_subgraph,
)
from uipath_langchain.agent.tools.tool_node import UiPathToolNode

_UUID = "7f2c1e44-0b3a-4a1e-9d55-2f9a1c3b8e10"
_TOOL = "analyze_files"


class _Args(BaseModel):
    attachment: dict[str, Any]
    question: str


def _judge() -> BuiltInValidatorGuardrail:
    return BuiltInValidatorGuardrail.model_validate(
        {
            "$guardrailType": "builtInValidator",
            "id": "judge-files",
            "name": "Judge files",
            "description": "Blocks poisoned files before a tool reads them.",
            "enabledForEvals": True,
            "selector": {"scopes": ["Tool"], "matchNames": [_TOOL]},
            "validatorType": "llm_as_judge",
            "validatorParameters": [],
        }
    )


def _build(monkeypatch, *, result: GuardrailValidationResultType):
    calls: list[dict[str, Any]] = []

    async def analyze(attachment: dict[str, Any], question: str) -> str:
        calls.append({"attachment": attachment, "question": question})
        return "The file lists 12 tickets."

    tool = StructuredTool.from_function(
        coroutine=analyze, name=_TOOL, description="Reads a file.", args_schema=_Args
    )
    graph = create_tool_guardrails_subgraph(
        tool_node=(_TOOL, UiPathToolNode(tool)),
        guardrails=[(_judge(), BlockAction("unsafe file"))],
    )
    fake = _patch_uipath(monkeypatch, result=result, reason="judged")
    return graph, fake, calls


def _input() -> dict[str, Any]:
    args = {"attachment": {"ID": _UUID}, "question": "summarize"}
    return {
        "messages": [
            AIMessage(
                content="", tool_calls=[{"name": _TOOL, "args": args, "id": "c1"}]
            )
        ],
        "inner_state": {
            "job_attachments": {
                _UUID: Attachment(
                    id=uuid.UUID(_UUID), full_name="Tickets.csv", mime_type="text/csv"
                )
            }
        },
    }


@pytest.mark.asyncio
async def test_pre_execution_judge_blocks_before_the_tool_runs(monkeypatch):
    """The judge saw the referenced file and the tool implementation never ran."""
    graph, fake, calls = _build(
        monkeypatch, result=GuardrailValidationResultType.VALIDATION_FAILED
    )

    payload = _input()

    with pytest.raises(AgentRuntimeError) as raised:
        await graph.ainvoke(payload)

    assert raised.value.error_info.code == AgentRuntimeError.full_code(
        AgentRuntimeErrorCode.TERMINATION_GUARDRAIL_VIOLATION
    )
    assert calls == []
    assert fake.guardrails.call_count == 1
    assert [a.id for a in fake.guardrails.last_attachments] == [_UUID]
    assert json.loads(fake.guardrails.last_text)["attachment"] == {"ID": _UUID}


@pytest.mark.asyncio
async def test_passing_judge_lets_the_tool_run_and_judges_its_result(monkeypatch):
    graph, fake, calls = _build(
        monkeypatch, result=GuardrailValidationResultType.PASSED
    )

    final = await graph.ainvoke(_input())

    assert [call["question"] for call in calls] == ["summarize"]
    assert fake.guardrails.call_count == 2
    assert isinstance(final["messages"][-1], ToolMessage)
    # The post-execution judge read the plain-text result and no file with it.
    assert fake.guardrails.last_text == "The file lists 12 tickets."
    assert fake.guardrails.last_attachments == []
