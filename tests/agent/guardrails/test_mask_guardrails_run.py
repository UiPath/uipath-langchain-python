"""Whole agent runs with Mask guardrails, checkpointed between steps."""

import json
import types
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.content import create_tool_call
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from pydantic import BaseModel
from uipath.core.guardrails import (
    GuardrailDetection,
    GuardrailValidationResult,
    GuardrailValidationResultType,
)
from uipath.platform.guardrails import BuiltInValidatorGuardrail

from uipath_langchain.agent.guardrails.actions import MaskAction
from uipath_langchain.agent.react.agent import create_agent
from uipath_langchain.agent.react.jsonschema_pydantic_converter import create_model
from uipath_langchain.agent.react.types import InnerAgentGraphState

_EMAIL = "jane@example.com"
_TOOL = "send_email"

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {"email": {"type": "string"}},
    "required": ["email"],
}
_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
}


class _EmailArgs(BaseModel):
    to: str
    body: str


class _Detector:
    """Flags the email address wherever it appears in the validated text."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    def evaluate_guardrail(
        self,
        text,
        guardrail,
        *,
        attachments=None,
        termination_mode=None,
        scope=None,
        execution_stage=None,
        action_type=None,
    ):
        self.texts.append(text)
        if _EMAIL not in text:
            return GuardrailValidationResult(
                result=GuardrailValidationResultType.PASSED, reason="No PII detected"
            )
        return GuardrailValidationResult(
            result=GuardrailValidationResultType.VALIDATION_FAILED,
            reason="PII detected",
            detections=[
                GuardrailDetection(
                    category="Email",
                    text=_EMAIL,
                    offset=text.index(_EMAIL),
                    length=len(_EMAIL),
                )
            ],
        )


class _StubChatAnthropic(ChatAnthropic):
    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)


def _scripted_model(*responses: AIMessage) -> Any:
    model: Any = _StubChatAnthropic.model_construct(model="claude-opus-5-5")
    model.model_details = {}
    model.bind_tools = Mock(return_value=model)
    model.ainvoke = AsyncMock(side_effect=list(responses))
    return model


def _call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(
        content="", tool_calls=[create_tool_call(name=name, args=args, id=call_id)]
    )


def _mask_guardrail(scope: str) -> BuiltInValidatorGuardrail:
    selector: dict[str, Any] = {"scopes": [scope]}
    if scope == "Tool":
        selector["matchNames"] = [_TOOL]
    return BuiltInValidatorGuardrail.model_validate(
        {
            "$guardrailType": "builtInValidator",
            "id": f"mask-{scope.lower()}",
            "name": f"Mask {scope} PII",
            "description": "Masks PII.",
            "enabledForEvals": True,
            "selector": selector,
            "validatorType": "pii_detection",
            "validatorParameters": [
                {"$parameterType": "enum-list", "id": "entities", "value": ["Email"]}
            ],
        }
    )


@pytest.fixture
def detector(monkeypatch) -> _Detector:
    detector = _Detector()
    monkeypatch.setattr(
        "uipath_langchain.agent.guardrails.guardrail_nodes.UiPath",
        lambda: types.SimpleNamespace(guardrails=detector),
    )
    return detector


def _agent(model: Any, scope: str, received: list[dict[str, Any]]) -> Any:
    async def send_email(to: str, body: str) -> str:
        received.append({"to": to, "body": body})
        return f"Sent to {to}."

    tool = StructuredTool.from_function(
        coroutine=send_email,
        name=_TOOL,
        description="Sends an email.",
        args_schema=_EmailArgs,
    )

    def messages(state: Any) -> list[SystemMessage | HumanMessage]:
        return [
            SystemMessage(content="You send emails."),
            HumanMessage(content=f"Send a welcome note to {state.email}."),
        ]

    return create_agent(
        model,
        [tool],
        messages,
        input_schema=create_model(_INPUT_SCHEMA),
        output_schema=create_model(_OUTPUT_SCHEMA),
        guardrails=[(_mask_guardrail(scope), MaskAction())],
    ).compile(checkpointer=InMemorySaver())


def _llm_inputs(model: Any) -> str:
    return json.dumps(
        [
            [message.content for message in call.args[0]]
            for call in model.ainvoke.call_args_list
        ]
    )


@pytest.mark.asyncio
async def test_agent_scope_hides_the_values_from_the_llm_and_the_tools(detector):
    model = _scripted_model(
        _call(_TOOL, {"to": "[Email_1]", "body": "Welcome!"}, "c1"),
        _call("end_execution", {"summary": "Welcomed [Email_1]."}, "c2"),
    )
    received: list[dict[str, Any]] = []
    graph = _agent(model, "Agent", received)
    config = {"configurable": {"thread_id": "agent-scope"}}

    result = await graph.ainvoke({"email": _EMAIL}, config)

    assert result["summary"] == f"Welcomed {_EMAIL}."
    assert received == [{"to": "[Email_1]", "body": "Welcome!"}]
    assert _EMAIL not in _llm_inputs(model)
    assert detector.texts == [f"Send a welcome note to {_EMAIL}."]
    snapshot = await graph.aget_state(config)
    assert snapshot.values["inner_state"].masked_values == {"[Email_1]": _EMAIL}


@pytest.mark.asyncio
async def test_tool_scope_hides_the_values_from_the_tool_only(detector):
    model = _scripted_model(
        _call(_TOOL, {"to": _EMAIL, "body": "Welcome!"}, "c1"),
        _call("end_execution", {"summary": "Done."}, "c2"),
    )
    received: list[dict[str, Any]] = []
    graph = _agent(model, "Tool", received)
    config = {"configurable": {"thread_id": "tool-scope"}}

    result = await graph.ainvoke({"email": _EMAIL}, config)

    assert result["summary"] == "Done."
    assert received == [{"to": "[Email_1]", "body": "Welcome!"}]
    snapshot = await graph.aget_state(config)
    messages = snapshot.values["messages"]
    tool_result = next(m for m in messages if isinstance(m, ToolMessage))
    # The LLM gets the tool's result with the value put back.
    assert tool_result.content == f"Sent to {_EMAIL}."
    assert snapshot.values["inner_state"].masked_values == {"[Email_1]": _EMAIL}


def test_masked_values_round_trip_through_the_checkpoint_serializer(caplog):
    state = InnerAgentGraphState(
        masked_values={"[Email_1]": _EMAIL, "[Person_1]": "José García"}
    )
    serde = JsonPlusSerializer()

    with caplog.at_level("WARNING"):
        restored = serde.loads_typed(serde.dumps_typed(state))

    assert restored == state
    # Only the state model itself is reported; its values are plain types.
    assert all("InnerAgentGraphState" in r.getMessage() for r in caplog.records)
