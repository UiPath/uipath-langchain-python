"""MaskAction run inside compiled guardrail subgraphs, with a PII detector stand-in."""

import json
import types
import uuid
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel
from uipath.core.guardrails import (
    GuardrailDetection,
    GuardrailValidationErrorType,
    GuardrailValidationResult,
    GuardrailValidationResultType,
)
from uipath.platform.attachments import Attachment
from uipath.platform.errors import EnrichedException
from uipath.platform.guardrails import BuiltInValidatorGuardrail, GuardrailScope

from uipath_langchain.agent.exceptions import AgentRuntimeError, AgentRuntimeErrorCode
from uipath_langchain.agent.guardrails import masking
from uipath_langchain.agent.guardrails.actions import MaskAction
from uipath_langchain.agent.guardrails.types import ExecutionStage
from uipath_langchain.agent.react.guardrails.guardrails_subgraph import (
    create_agent_init_guardrails_subgraph,
    create_agent_terminate_guardrails_subgraph,
    create_tool_guardrails_subgraph,
)
from uipath_langchain.agent.react.init_node import create_init_node
from uipath_langchain.agent.tools.tool_node import UiPathToolNode

_TOOL = "send_email"
_FILE_ID = "7f2c1e44-0b3a-4a1e-9d55-2f9a1c3b8e10"


class _PiiDetector:
    """Finds the configured texts in the validated input, like the PII validator."""

    def __init__(
        self,
        pii: dict[str, str] | None = None,
        *,
        result: GuardrailValidationResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.pii = pii or {}
        self.result = result
        self.error = error
        self.calls: list[dict[str, Any]] = []

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
        self.calls.append(
            {
                "text": text,
                "attachments": attachments,
                "scope": scope,
                "execution_stage": execution_stage,
                "action_type": action_type,
            }
        )
        if self.error is not None:
            raise self.error
        if self.result is not None:
            return self.result
        detections = [
            GuardrailDetection(
                category=category,
                text=value,
                offset=text.find(value),
                length=len(value),
                confidence_score=0.9,
            )
            for value, category in self.pii.items()
            if value in text
        ]
        if not detections:
            return GuardrailValidationResult(
                result=GuardrailValidationResultType.PASSED, reason="No PII detected"
            )
        return GuardrailValidationResult(
            result=GuardrailValidationResultType.VALIDATION_FAILED,
            reason="PII detected",
            detections=detections,
        )


_TAG = "7f3a9c"


@pytest.fixture(autouse=True)
def _fixed_run_tag(monkeypatch):
    monkeypatch.setattr(masking.secrets, "token_hex", lambda _n: _TAG)


def _patch(monkeypatch, detector: _PiiDetector) -> _PiiDetector:
    monkeypatch.setattr(
        "uipath_langchain.agent.guardrails.guardrail_nodes.UiPath",
        lambda: types.SimpleNamespace(guardrails=detector),
    )
    return detector


def _pii_guardrail(scope: str, *, detection_scope: str = "Text"):
    selector: dict[str, Any] = {"scopes": [scope]}
    if scope == "Tool":
        selector["matchNames"] = [_TOOL]
    return BuiltInValidatorGuardrail.model_validate(
        {
            "$guardrailType": "builtInValidator",
            "id": "mask-pii",
            "name": "Mask PII",
            "description": "Masks PII before the call.",
            "enabledForEvals": True,
            "selector": selector,
            "validatorType": "pii_detection",
            "validatorParameters": [
                {
                    "$parameterType": "enum-list",
                    "id": "entities",
                    "value": ["Email", "Person", "Address"],
                },
                {
                    "$parameterType": "enum",
                    "id": "detectionScope",
                    "value": detection_scope,
                },
            ],
        }
    )


class _EmailArgs(BaseModel):
    to: str
    body: str


def _tool_graph(guardrail=None):
    received: list[dict[str, Any]] = []

    async def send_email(to: str, body: str) -> str:
        received.append({"to": to, "body": body})
        return json.dumps({"status": "sent", "to": to, "preview": body})

    tool = StructuredTool.from_function(
        coroutine=send_email,
        name=_TOOL,
        description="Sends an email.",
        args_schema=_EmailArgs,
    )
    graph = create_tool_guardrails_subgraph(
        tool_node=(_TOOL, UiPathToolNode(tool)),
        guardrails=[(guardrail or _pii_guardrail("Tool"), MaskAction())],
    )
    return graph, received


def _tool_call_input(
    args: dict[str, Any], masked_values: dict[str, str] | None = None
) -> dict[str, Any]:
    return {
        "messages": [
            AIMessage(
                content="", tool_calls=[{"name": _TOOL, "args": args, "id": "c1"}]
            )
        ],
        "inner_state": {
            "masked_values": masked_values or {},
            "mask_run_tag": _TAG if masked_values else None,
        },
    }


class TestToolScope:
    @pytest.mark.asyncio
    async def test_the_tool_sees_placeholders_and_its_result_gets_the_values(
        self, monkeypatch
    ):
        detector = _patch(
            monkeypatch,
            _PiiDetector({"jane@example.com": "Email", "José García": "Person"}),
        )
        graph, received = _tool_graph()

        final = await graph.ainvoke(
            _tool_call_input(
                {"to": "jane@example.com", "body": "Hi José García, from Jane"}
            )
        )

        assert received == [
            {
                "to": "[Email-Tool-7f3a9c-1]",
                "body": "Hi [Person-Tool-7f3a9c-1], from Jane",
            }
        ]
        assert final["messages"][0].tool_calls[0]["args"] == received[0]
        result = final["messages"][-1]
        assert isinstance(result, ToolMessage) and isinstance(result.content, str)
        assert json.loads(result.content) == {
            "status": "sent",
            "to": "jane@example.com",
            "preview": "Hi José García, from Jane",
        }
        assert final["inner_state"].masked_values == {
            "[Email-Tool-7f3a9c-1]": "jane@example.com",
            "[Person-Tool-7f3a9c-1]": "José García",
        }
        # Checked once, before the call, as a text-only Mask check.
        assert len(detector.calls) == 1
        call = detector.calls[0]
        assert (call["scope"], call["execution_stage"], call["action_type"]) == (
            GuardrailScope.TOOL,
            "preExecution",
            "Mask",
        )
        assert call["attachments"] is None
        # Non-ASCII characters reach the detector as they are, not as \u escapes.
        assert "José García" in call["text"]

    @pytest.mark.asyncio
    async def test_a_detection_with_json_escapes_is_matched_to_its_value(
        self, monkeypatch
    ):
        # The payload is JSON, so the detector sees the line break as "\n".
        _patch(monkeypatch, _PiiDetector({"1 Main St\\nSpringfield": "Address"}))
        graph, received = _tool_graph()

        await graph.ainvoke(
            _tool_call_input({"to": "x", "body": "Ship to 1 Main St\nSpringfield"})
        )

        assert received == [{"to": "x", "body": "Ship to [Address-Tool-7f3a9c-1]"}]

    @pytest.mark.asyncio
    async def test_a_value_masked_earlier_in_the_run_keeps_its_placeholder(
        self, monkeypatch
    ):
        _patch(
            monkeypatch,
            _PiiDetector({"jane@example.com": "Email", "bob@example.com": "Email"}),
        )
        graph, received = _tool_graph()

        final = await graph.ainvoke(
            _tool_call_input(
                {"to": "bob@example.com", "body": "cc jane@example.com"},
                masked_values={"[Email-Tool-7f3a9c-1]": "jane@example.com"},
            )
        )

        assert received == [
            {"to": "[Email-Tool-7f3a9c-2]", "body": "cc [Email-Tool-7f3a9c-1]"}
        ]
        assert final["inner_state"].masked_values == {
            "[Email-Tool-7f3a9c-1]": "jane@example.com",
            "[Email-Tool-7f3a9c-2]": "bob@example.com",
        }
        # Both placeholders were written by this mask, so both come back.
        assert json.loads(final["messages"][-1].content) == {
            "status": "sent",
            "to": "bob@example.com",
            "preview": "cc jane@example.com",
        }

    @pytest.mark.asyncio
    async def test_a_placeholder_from_elsewhere_in_the_run_is_not_restored(
        self, monkeypatch
    ):
        """An agent-scope placeholder the LLM passes and the tool echoes stays masked:
        the LLM reads the tool's result and must never get the value behind it."""
        _patch(monkeypatch, _PiiDetector({"bob@example.com": "Email"}))
        graph, received = _tool_graph()

        final = await graph.ainvoke(
            _tool_call_input(
                {"to": "bob@example.com", "body": "cc [Email-Tool-7f3a9c-1]"},
                masked_values={"[Email-Tool-7f3a9c-1]": "jane@example.com"},
            )
        )

        assert received == [
            {"to": "[Email-Tool-7f3a9c-2]", "body": "cc [Email-Tool-7f3a9c-1]"}
        ]
        assert json.loads(final["messages"][-1].content) == {
            "status": "sent",
            "to": "bob@example.com",
            "preview": "cc [Email-Tool-7f3a9c-1]",
        }
        # The call's record is dropped once its result is restored.
        assert final["inner_state"].masked_tool_calls == {}

    @pytest.mark.asyncio
    async def test_a_clean_call_runs_unmasked_and_restores_nothing(self, monkeypatch):
        detector = _patch(monkeypatch, _PiiDetector({}))
        graph, received = _tool_graph()

        final = await graph.ainvoke(
            _tool_call_input(
                {"to": "[Email-Tool-7f3a9c-1]", "body": "hello"},
                masked_values={"[Email-Tool-7f3a9c-1]": "jane@example.com"},
            )
        )

        assert received == [{"to": "[Email-Tool-7f3a9c-1]", "body": "hello"}]
        assert (
            json.loads(final["messages"][-1].content)["to"] == "[Email-Tool-7f3a9c-1]"
        )
        assert len(detector.calls) == 1

    @pytest.mark.asyncio
    async def test_the_mask_records_what_it_wrote_into_the_call(self, monkeypatch):
        _patch(
            monkeypatch,
            _PiiDetector({"jane@example.com": "Email", "José García": "Person"}),
        )
        guardrail = _pii_guardrail("Tool")
        graph = create_tool_guardrails_subgraph(
            tool_node=(_TOOL, lambda _state: {}),
            guardrails=[(guardrail, MaskAction())],
        )

        final = await graph.ainvoke(
            _tool_call_input({"to": "jane@example.com", "body": "José García"})
        )

        # The stand-in tool returned no result, so the record is still there.
        assert final["inner_state"].masked_tool_calls == {
            f"{guardrail.id}:c1": ["[Email-Tool-7f3a9c-1]", "[Person-Tool-7f3a9c-1]"]
        }
        # Counts for the action span, never the values.
        assert final["inner_state"].guardrail_masked_entities == {
            "Email": 1,
            "Person": 1,
        }

    @pytest.mark.asyncio
    async def test_the_output_is_restored_without_a_check(self, monkeypatch):
        """After the call the guardrail is not evaluated: no node exists for it."""
        _patch(monkeypatch, _PiiDetector({}))
        graph, _ = _tool_graph()

        assert "tool_post_execution_mask_pii" not in graph.nodes
        assert "tool_post_execution_mask_pii_restore" in graph.nodes
        assert "tool_pre_execution_mask_pii_mask" in graph.nodes

    @pytest.mark.asyncio
    async def test_metadata_counts_the_masked_categories_never_the_values(
        self, monkeypatch
    ):
        _patch(
            monkeypatch,
            _PiiDetector({"jane@example.com": "Email", "José García": "Person"}),
        )
        mask_node = MaskAction().action_node(
            guardrail=_pii_guardrail("Tool"),
            scope=GuardrailScope.TOOL,
            execution_stage=ExecutionStage.PRE_EXECUTION,
            guarded_component_name=_TOOL,
        )[1]
        graph, _ = _tool_graph()

        await graph.ainvoke(
            _tool_call_input({"to": "jane@example.com", "body": "José García"})
        )
        mask_metadata = graph.builder.nodes["tool_pre_execution_mask_pii_mask"].metadata
        restore_metadata = graph.builder.nodes[
            "tool_post_execution_mask_pii_restore"
        ].metadata

        assert mask_metadata["masked_entities"] == {"Email": 1, "Person": 1}
        assert mask_metadata["node_type"] == "guardrail_action"
        assert mask_metadata["action_type"] == "Mask"
        assert restore_metadata["restored_entities"] == {"Email": 1, "Person": 1}
        assert restore_metadata["node_type"] == "guardrail_action"
        for metadata in (mask_metadata, restore_metadata, mask_node.__metadata__):
            serialized = json.dumps(
                {k: v for k, v in metadata.items() if k != "guardrail"}, default=str
            )
            assert "jane@example.com" not in serialized
            assert "García" not in serialized


class _Profile(BaseModel):
    name: str
    age: int


class _AgentInput(BaseModel):
    email: str
    profile: _Profile


class TestAgentScope:
    def _init_graph(self):
        def messages(state: Any) -> list[SystemMessage | HumanMessage]:
            return [
                SystemMessage(f"Reply to {state.email} only."),
                HumanMessage(f"Write to {state.email} about {state.profile.name}."),
            ]

        return create_agent_init_guardrails_subgraph(
            ("guarded-init", create_init_node(messages, _AgentInput)),
            [(_pii_guardrail("Agent", detection_scope="Both"), MaskAction())],
            input_schema=_AgentInput,
        )

    @pytest.mark.asyncio
    async def test_the_agent_input_is_masked_before_the_run(self, monkeypatch):
        detector = _patch(
            monkeypatch,
            _PiiDetector({"jane@example.com": "Email", "Jane Doe": "Person"}),
        )
        graph = self._init_graph()

        final = await graph.ainvoke(
            {
                "email": "jane@example.com",
                "profile": {"name": "Jane Doe", "age": 41},
                "inner_state": {
                    "job_attachments": {
                        _FILE_ID: Attachment(
                            id=uuid.UUID(_FILE_ID),
                            full_name="cv.pdf",
                            mime_type="application/pdf",
                        )
                    }
                },
            }
        )

        assert [m.content for m in final["messages"]] == [
            "Reply to [Email-Agent-7f3a9c-1] only.",
            "Write to [Email-Agent-7f3a9c-1] about [Person-Agent-7f3a9c-1].",
        ]
        assert final["email"] == "[Email-Agent-7f3a9c-1]"
        profile = final["profile"]
        assert _Profile.model_validate(profile).model_dump() == {
            "name": "[Person-Agent-7f3a9c-1]",
            "age": 41,
        }
        assert final["inner_state"].masked_values == {
            "[Email-Agent-7f3a9c-1]": "jane@example.com",
            "[Person-Agent-7f3a9c-1]": "Jane Doe",
        }
        call = detector.calls[0]
        assert call["text"] == "Write to jane@example.com about Jane Doe."
        assert (call["scope"], call["action_type"]) == (GuardrailScope.AGENT, "Mask")
        # The guardrail inspects files, but a mask check never sends them.
        assert call["attachments"] is None

    @pytest.mark.asyncio
    async def test_the_agent_output_gets_the_values_back(self, monkeypatch):
        detector = _patch(monkeypatch, _PiiDetector({"jane@example.com": "Email"}))
        run = create_agent_terminate_guardrails_subgraph(
            terminate_node=(
                "terminate",
                lambda _state: {
                    "summary": "Wrote to [Email-Agent-7f3a9c-1] about [Person-Agent-7f3a9c-1].",
                    "recipients": ["[Email-Agent-7f3a9c-1]", "[Email-Agent-7f3a9c-7]"],
                },
            ),
            guardrails=[(_pii_guardrail("Agent"), MaskAction())],
        )

        result = await run(
            {
                "messages": [HumanMessage("Write to [Email-Agent-7f3a9c-1].")],
                "inner_state": {
                    "masked_values": {
                        "[Email-Agent-7f3a9c-1]": "jane@example.com",
                        "[Person-Agent-7f3a9c-1]": "Jane Doe",
                    },
                    "mask_run_tag": _TAG,
                },
            }
        )

        assert result == {
            "summary": "Wrote to jane@example.com about Jane Doe.",
            "recipients": ["jane@example.com", "[Email-Agent-7f3a9c-7]"],
        }
        assert detector.calls == []


class TestCannotMask:
    @pytest.mark.asyncio
    async def test_a_check_that_could_not_run_stops_the_call(self, monkeypatch):
        _patch(
            monkeypatch,
            _PiiDetector(
                result=GuardrailValidationResult(
                    result=GuardrailValidationResultType.VALIDATION_FAILED,
                    reason="Azure is down",
                    error_type=GuardrailValidationErrorType.PROVIDER_ERROR,
                )
            ),
        )
        graph, received = _tool_graph()

        with pytest.raises(AgentRuntimeError) as raised:
            await graph.ainvoke(_tool_call_input({"to": "a@b.co", "body": "x"}))

        assert raised.value.error_info.code == AgentRuntimeError.full_code(
            AgentRuntimeErrorCode.TERMINATION_GUARDRAIL_ERROR
        )
        assert "provider_error" in raised.value.error_info.detail
        assert received == []

    @pytest.mark.asyncio
    async def test_a_violation_without_detections_stops_the_call(self, monkeypatch):
        _patch(
            monkeypatch,
            _PiiDetector(
                result=GuardrailValidationResult(
                    result=GuardrailValidationResultType.VALIDATION_FAILED,
                    reason="PII detected",
                    detections=[
                        # From a file: no position in the input.
                        GuardrailDetection(category="Email", text="a@b.co"),
                    ],
                )
            ),
        )
        graph, received = _tool_graph()

        with pytest.raises(AgentRuntimeError) as raised:
            await graph.ainvoke(_tool_call_input({"to": "a@b.co", "body": "x"}))

        assert raised.value.error_info.code == AgentRuntimeError.full_code(
            AgentRuntimeErrorCode.TERMINATION_GUARDRAIL_ERROR
        )
        assert received == []

    @pytest.mark.asyncio
    async def test_a_value_that_cannot_be_located_stops_the_call(self, monkeypatch):
        _patch(
            monkeypatch,
            _PiiDetector(
                result=GuardrailValidationResult(
                    result=GuardrailValidationResultType.VALIDATION_FAILED,
                    reason="PII detected",
                    detections=[
                        GuardrailDetection(
                            category="Email", text="a@b.co", offset=7, length=6
                        ),
                        GuardrailDetection(
                            category="Person", text="Zed Q", offset=30, length=5
                        ),
                    ],
                )
            ),
        )
        graph, received = _tool_graph()

        with pytest.raises(AgentRuntimeError) as raised:
            await graph.ainvoke(_tool_call_input({"to": "a@b.co", "body": "x"}))

        assert "found Person" in raised.value.error_info.detail
        assert "Zed Q" not in raised.value.error_info.detail
        assert received == []

    @pytest.mark.asyncio
    async def test_a_rejected_mask_check_is_raised_not_retried(self, monkeypatch):
        response = httpx.Response(
            400, request=httpx.Request("POST", "https://x/validate"), text="no mask"
        )
        rejection = EnrichedException(
            httpx.HTTPStatusError("400", request=response.request, response=response)
        )
        detector = _patch(monkeypatch, _PiiDetector(error=rejection))
        graph, received = _tool_graph()

        with pytest.raises(EnrichedException):
            await graph.ainvoke(_tool_call_input({"to": "a@b.co", "body": "x"}))

        assert len(detector.calls) == 1
        assert received == []


class TestMaskAction:
    def test_action_declares_a_text_only_mask_check(self):
        action = MaskAction()

        assert action.action_type == "Mask"
        assert action.text_only is True

    def test_only_the_post_execution_stage_skips_the_check(self):
        guardrail = _pii_guardrail("Tool")
        action = MaskAction()

        pre = action.unconditional_node(
            guardrail=guardrail,
            scope=GuardrailScope.TOOL,
            execution_stage=ExecutionStage.PRE_EXECUTION,
            guarded_component_name=_TOOL,
        )
        post = action.unconditional_node(
            guardrail=guardrail,
            scope=GuardrailScope.TOOL,
            execution_stage=ExecutionStage.POST_EXECUTION,
            guarded_component_name=_TOOL,
        )

        assert pre is None
        assert post is not None
        assert post[0] == "tool_post_execution_mask_pii_restore"

    @pytest.mark.asyncio
    async def test_llm_scope_is_refused(self):
        from uipath_langchain.agent.react.types import AgentGuardrailsGraphState

        _, node = MaskAction().action_node(
            guardrail=_pii_guardrail("Agent"),
            scope=GuardrailScope.LLM,
            execution_stage=ExecutionStage.PRE_EXECUTION,
            guarded_component_name="llm",
        )

        with pytest.raises(AgentRuntimeError):
            await node(AgentGuardrailsGraphState())
