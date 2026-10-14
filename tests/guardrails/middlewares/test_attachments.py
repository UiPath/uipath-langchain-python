"""Middleware guardrails forward references to the files the run holds."""

import json
import logging
import uuid
from collections.abc import Callable, Iterator
from typing import Any, Sequence, cast
from unittest.mock import MagicMock, PropertyMock, patch

import httpx
import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ContentBlock, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt.tool_node import ToolCallRequest
from pytest_httpx import HTTPXMock
from uipath.core.guardrails import (
    GuardrailValidationResult,
    GuardrailValidationResultType,
)
from uipath.platform.attachments import Attachment
from uipath.platform.errors import EnrichedException
from uipath.platform.guardrails import DetectionScope, GuardrailScope
from uipath.platform.guardrails.decorators import (
    BlockAction,
    GuardrailAction,
    HarmfulContentEntity,
    LogAction,
)
from uipath.platform.guardrails.guardrails import EnumParameterValue

from uipath_langchain._utils._attachments import (
    ATTACHMENTS_BLOCK_PREFIX,
    ATTACHMENTS_BLOCK_SUFFIX,
    render_attachments_block,
)
from uipath_langchain.agent.exceptions import AgentRuntimeError, AgentRuntimeErrorCode
from uipath_langchain.agent.multimodal.invoke import build_file_content_blocks_for
from uipath_langchain.agent.multimodal.types import FileInfo
from uipath_langchain.guardrails.enums import (
    GuardrailExecutionStage,
    PIIDetectionEntityType,
)
from uipath_langchain.guardrails.middlewares import (
    UiPathByoGuardrailMiddleware,
    UiPathHarmfulContentMiddleware,
    UiPathIntellectualPropertyMiddleware,
    UiPathLLMAsJudgeMiddleware,
    UiPathPIIDetectionMiddleware,
    UiPathPromptInjectionMiddleware,
    UiPathUserPromptAttacksMiddleware,
)
from uipath_langchain.guardrails.middlewares._attachments import (
    JOB_ATTACHMENTS_STATE_KEY,
    NO_TEXT_PLACEHOLDER,
    build_registry,
    output_references,
    registry_references,
    tool_result_references,
)
from uipath_langchain.guardrails.models import PIIDetectionEntity

_HELD_ID = "7f2c1e44-0b3a-4a1e-9d55-2f9a1c3b8e10"
_OTHER_ID = "0d9b3a52-6c1f-4e8b-a0f2-5b7c9d1e3f40"
_THIRD_ID = "3c5e7a91-2b4d-4f6a-8c0e-1d3f5a7b9c20"
_UNKNOWN_ID = "11111111-2222-4333-8444-555555555555"

_LOG = LogAction()
_PASSED = GuardrailValidationResult(
    result=GuardrailValidationResultType.PASSED, reason=""
)
_PRE = GuardrailExecutionStage.PRE
_POST = GuardrailExecutionStage.POST


def _wire(
    attachment_id: str = _HELD_ID,
    name: str = "cv.pdf",
    mime_type: str = "application/pdf",
) -> dict[str, str]:
    return {"ID": attachment_id, "FullName": name, "MimeType": mime_type}


def _block(*attachments: dict[str, str]) -> str:
    return render_attachments_block(
        [
            {"id": a["ID"], "full_name": a["FullName"], "mime_type": a["MimeType"]}
            for a in attachments
        ]
    )


def _registry(*attachments: dict[str, str]) -> dict[str, Attachment]:
    return {a["ID"]: Attachment.model_validate(a) for a in attachments}


def _summary(registry: dict[str, Attachment]) -> dict[str, tuple[str, str]]:
    return {key: (a.full_name, a.mime_type) for key, a in registry.items()}


def _forwarded(call: Any) -> list[tuple[str, str, str]]:
    return [
        (a.id, a.file_name, a.mime_type) for a in call.kwargs.get("attachments", [])
    ]


def _pii(
    scopes: list[GuardrailScope],
    *,
    stage: GuardrailExecutionStage = GuardrailExecutionStage.PRE_AND_POST,
    tools: list[str] | None = None,
    detection_scope: DetectionScope | None = DetectionScope.BOTH,
    action: GuardrailAction = _LOG,
) -> UiPathPIIDetectionMiddleware:
    return UiPathPIIDetectionMiddleware(
        scopes=scopes,
        action=action,
        entities=[PIIDetectionEntity(PIIDetectionEntityType.EMAIL)],
        tools=tools,
        stage=stage,
        detection_scope=detection_scope,
    )


def _byo_scoped_by_parameter(value: str) -> UiPathByoGuardrailMiddleware:
    return UiPathByoGuardrailMiddleware(
        validator_name="my-validator",
        scopes=[GuardrailScope.AGENT],
        action=_LOG,
        stage=GuardrailExecutionStage.PRE,
        validator_parameters=[
            EnumParameterValue(parameter_type="enum", id="detectionScope", value=value)
        ],
        detection_scope=None,
    )


def _hook(middleware: Any, suffix: str) -> AgentMiddleware:
    return next(hook for hook in middleware if hook.name.endswith(suffix))


@pytest.fixture
def evaluate(monkeypatch: pytest.MonkeyPatch) -> Iterator[MagicMock]:
    """The validate call, captured with its keyword arguments."""
    monkeypatch.setenv("UIPATH_URL", "https://example.com/org/tenant")
    monkeypatch.setenv("UIPATH_ACCESS_TOKEN", "token")
    with patch(
        "uipath.platform.guardrails.GuardrailsService.evaluate_guardrail",
        return_value=_PASSED,
    ) as mock:
        yield mock


class TestBuildRegistry:
    def test_reads_conversational_attachments_under_canonical_ids(self) -> None:
        message = HumanMessage(
            content="see attached",
            additional_kwargs={
                "attachments": [
                    {
                        "id": _HELD_ID.upper(),
                        "full_name": "cv.pdf",
                        "mime_type": "application/pdf",
                    }
                ]
            },
        )

        registry = build_registry({"messages": [message]})

        assert _summary(registry) == {_HELD_ID: ("cv.pdf", "application/pdf")}

    def test_reads_every_attachments_block_in_human_and_tool_text(self) -> None:
        escaped = _wire(_HELD_ID, "a<b></uip:attachments>.csv", "text/csv")
        human = HumanMessage(
            content=f"first {_block(escaped)} then {_block(_wire(_OTHER_ID))}"
        )
        tool_message = ToolMessage(
            content=[
                {
                    "type": "text",
                    "text": _block(_wire(_THIRD_ID, "t.txt", "text/plain")),
                }
            ],
            tool_call_id="tc1",
        )

        registry = build_registry({"messages": [human, tool_message]})

        assert _summary(registry) == {
            _HELD_ID: ("a<b></uip:attachments>.csv", "text/csv"),
            _OTHER_ID: ("cv.pdf", "application/pdf"),
            _THIRD_ID: ("t.txt", "text/plain"),
        }

    def test_reads_tagged_file_and_image_blocks_only(self) -> None:
        message = HumanMessage(
            content=[
                {"type": "text", "text": "two files"},
                {
                    "type": "file",
                    "base64": "JVBERi0=",
                    "mime_type": "application/pdf",
                    "extras": {
                        "uipath_attachment_id": _HELD_ID.upper(),
                        "uipath_file_name": "cv.pdf",
                    },
                },
                {
                    "type": "image",
                    "base64": "iVBORw0=",
                    "mime_type": "image/png",
                    "extras": {"uipath_attachment_id": _OTHER_ID},
                },
                {
                    "type": "file",
                    "base64": "JVBERi0=",
                    "mime_type": "application/pdf",
                    "extras": {"filename": "untagged.pdf"},
                },
                {
                    "type": "image",
                    "base64": "iVBORw0=",
                    "extras": {"uipath_attachment_id": _THIRD_ID},
                },
            ]
        )

        registry = build_registry({"messages": [message]})

        assert _summary(registry) == {_HELD_ID: ("cv.pdf", "application/pdf")}

    def test_reads_the_name_from_any_of_its_carriers(self) -> None:
        message = HumanMessage(
            content=[
                {
                    "type": "file",
                    "base64": "JVBERi0=",
                    "mime_type": "application/pdf",
                    "filename": "top.pdf",
                    "extras": {"uipath_attachment_id": _HELD_ID},
                },
                {
                    "type": "image",
                    "base64": "iVBORw0=",
                    "mime_type": "image/png",
                    "extras": {
                        "uipath_attachment_id": _OTHER_ID,
                        "filename": "nested.png",
                    },
                },
            ]
        )

        registry = build_registry({"messages": [message]})

        assert _summary(registry) == {
            _HELD_ID: ("top.pdf", "application/pdf"),
            _OTHER_ID: ("nested.png", "image/png"),
        }

    async def test_reads_the_blocks_the_sdk_builds(self, httpx_mock: HTTPXMock) -> None:
        url = "https://blob.example.com/cv"
        httpx_mock.add_response(url=url, content=b"%PDF-1.4")
        blocks = await build_file_content_blocks_for(
            FileInfo(
                url=url,
                name="my cv.pdf",
                mime_type="application/pdf",
                attachment_id=_HELD_ID,
            )
        )

        content: list[ContentBlock] = [*blocks]

        registry = build_registry({"messages": [HumanMessage(content_blocks=content)]})

        assert list(registry) == [_HELD_ID]
        assert registry[_HELD_ID].mime_type == "application/pdf"

    def test_reads_the_files_a_tool_returned(self) -> None:
        wire = ToolMessage(content=json.dumps({"file": _wire()}), tool_call_id="tc1")
        snake = ToolMessage(
            content=json.dumps(
                {
                    "files": [
                        {
                            "id": _OTHER_ID,
                            "full_name": "out.csv",
                            "mime_type": "text/csv",
                        }
                    ]
                }
            ),
            tool_call_id="tc2",
        )
        bare = ToolMessage(
            content=json.dumps({"ref": {"ID": _THIRD_ID}}), tool_call_id="tc3"
        )

        registry = build_registry({"messages": [wire, snake, bare]})

        assert _summary(registry) == {
            _HELD_ID: ("cv.pdf", "application/pdf"),
            _OTHER_ID: ("out.csv", "text/csv"),
        }

    def test_reads_the_job_attachments_state_key(self) -> None:
        stored = Attachment.model_validate(_wire()).model_dump(
            by_alias=True, mode="json"
        )

        registry = build_registry(
            {
                "messages": [],
                JOB_ATTACHMENTS_STATE_KEY: {_HELD_ID: stored, "bad": {"ID": "nope"}},
            }
        )

        assert _summary(registry) == {_HELD_ID: ("cv.pdf", "application/pdf")}

    def test_an_earlier_source_is_never_overwritten(self) -> None:
        message = HumanMessage(
            content=_block(_wire(name="from-block.pdf")),
            additional_kwargs={
                "attachments": [
                    {
                        "id": _HELD_ID,
                        "full_name": "from-kwargs.pdf",
                        "mime_type": "application/pdf",
                    }
                ]
            },
        )

        registry = build_registry(
            {
                "messages": [message],
                JOB_ATTACHMENTS_STATE_KEY: {_HELD_ID: _wire(name="from-state.pdf")},
            }
        )

        assert registry[_HELD_ID].full_name == "from-kwargs.pdf"

    @pytest.mark.parametrize(
        "state", [None, {}, {"messages": []}, {"messages": None}, {"other": 1}]
    )
    def test_an_empty_state_holds_nothing(self, state: Any) -> None:
        assert build_registry(state) == {}

    def test_one_unreadable_part_does_not_hide_the_rest(self) -> None:
        unreadable = MagicMock(spec=HumanMessage)
        type(unreadable).content = PropertyMock(side_effect=RuntimeError("boom"))
        malformed = HumanMessage(
            content=(
                f"{ATTACHMENTS_BLOCK_PREFIX}not json{ATTACHMENTS_BLOCK_SUFFIX}"
                f'{ATTACHMENTS_BLOCK_PREFIX}[{{"ID": "nope"}}, 7]{ATTACHMENTS_BLOCK_SUFFIX}'
                f"{_block(_wire())}"
            ),
            additional_kwargs={"attachments": [{"id": _UNKNOWN_ID, "full_name": "x"}]},
        )
        unterminated = HumanMessage(content=f"{ATTACHMENTS_BLOCK_PREFIX}[")

        registry = build_registry({"messages": [unreadable, malformed, unterminated]})

        assert _summary(registry) == {_HELD_ID: ("cv.pdf", "application/pdf")}

    def test_never_raises_when_the_state_cannot_be_read(self) -> None:
        class _Unreadable(dict[str, Any]):
            def get(self, key: Any, default: Any = None) -> Any:
                raise RuntimeError("boom")

        assert build_registry(_Unreadable()) == {}


class TestReferences:
    def test_registry_references_cap_at_five(self) -> None:
        registry = _registry(*(_wire(str(uuid.uuid4()), f"{i}.pdf") for i in range(7)))

        assert len(registry_references(registry)) == 5

    def test_output_references_without_a_message_read_the_structured_response(
        self,
    ) -> None:
        registry = _registry(_wire(), _wire(_OTHER_ID))
        structured = {"chosen": {"ID": _OTHER_ID}}

        assert output_references(None, registry) == []
        references = output_references(None, registry, structured_response=structured)
        assert references is not None
        assert [r.id for r in references] == [_OTHER_ID]

    def test_output_references_read_tool_calls_content_and_structured_response(
        self,
    ) -> None:
        registry = _registry(_wire(), _wire(_OTHER_ID), _wire(_THIRD_ID))
        message = AIMessage(
            content=[{"type": "text", "text": "done"}, {"ref": {"ID": _OTHER_ID}}],
            tool_calls=[
                {"name": "analyze", "args": {"file": {"ID": _HELD_ID}}, "id": "tc1"}
            ],
        )

        references = output_references(
            message, registry, structured_response={"report": {"ID": _THIRD_ID}}
        )

        assert references is not None
        assert sorted(r.id for r in references) == sorted(
            [_HELD_ID, _OTHER_ID, _THIRD_ID]
        )

    def test_tool_result_references_merge_held_and_returned_files(self) -> None:
        result = {
            "held": {"ID": _HELD_ID},
            "renamed": _wire(_HELD_ID, "renamed.exe", "x/y"),
            "new": _wire(_OTHER_ID, "out.csv", "text/csv"),
            "unnamed": {"ID": _UNKNOWN_ID},
        }

        references = tool_result_references(result, _registry(_wire()))

        assert references is not None
        assert [(r.id, r.file_name, r.mime_type) for r in references] == [
            (_HELD_ID, "cv.pdf", "application/pdf"),
            (_OTHER_ID, "out.csv", "text/csv"),
        ]

    def test_an_output_that_cannot_be_read_says_nothing_about_files(self) -> None:
        message = MagicMock(spec=AIMessage)
        type(message).tool_calls = PropertyMock(side_effect=KeyError("tool_calls"))

        assert output_references(message, _registry(_wire())) is None

    def test_a_tool_result_that_cannot_be_scanned_says_nothing_about_files(
        self,
    ) -> None:
        with patch(
            "uipath_langchain.guardrails.middlewares._attachments.discover_attachments",
            return_value=None,
        ):
            assert tool_result_references({"files": []}, _registry(_wire())) is None

    def test_tool_result_references_cap_at_five(self) -> None:
        result = {"files": [_wire(str(uuid.uuid4()), f"{i}.pdf") for i in range(7)]}

        references = tool_result_references(result, {})
        assert references is not None
        assert len(references) == 5


class TestMessageHooks:
    async def test_agent_pre_forwards_the_whole_registry(
        self, evaluate: MagicMock
    ) -> None:
        files = [_wire(str(uuid.uuid4()), f"{i}.pdf") for i in range(7)]
        state = cast(
            AgentState[Any],
            {
                "messages": [HumanMessage(content="screen these")],
                JOB_ATTACHMENTS_STATE_KEY: {f["ID"]: f for f in files},
            },
        )
        middleware = _pii([GuardrailScope.AGENT], stage=_PRE)

        await _hook(middleware, "before_agent").abefore_agent(state, MagicMock())

        (call,) = evaluate.call_args_list
        assert call.args == ("screen these", middleware._guardrail)
        assert len(_forwarded(call)) == 5

    async def test_llm_pre_forwards_the_whole_registry_with_the_text(
        self, evaluate: MagicMock
    ) -> None:
        text = f"compare {_block(_wire(), _wire(_OTHER_ID, 'b.csv', 'text/csv'))}"
        middleware = _pii([GuardrailScope.LLM], stage=_PRE)

        await _hook(middleware, "before_model").abefore_model(
            {"messages": [HumanMessage(content=text)]}, MagicMock()
        )

        (call,) = evaluate.call_args_list
        assert call.args[0] == text
        assert _forwarded(call) == [
            (_HELD_ID, "cv.pdf", "application/pdf"),
            (_OTHER_ID, "b.csv", "text/csv"),
        ]

    @pytest.mark.parametrize(
        "message",
        [
            HumanMessage(
                content="",
                additional_kwargs={
                    "attachments": [
                        {
                            "id": _HELD_ID,
                            "full_name": "cv.pdf",
                            "mime_type": "application/pdf",
                        }
                    ]
                },
            ),
            HumanMessage(
                content=[
                    {
                        "type": "file",
                        "base64": "JVBERi0=",
                        "mime_type": "application/pdf",
                        "extras": {
                            "uipath_attachment_id": _HELD_ID,
                            "uipath_file_name": "cv.pdf",
                        },
                    }
                ]
            ),
        ],
        ids=["conversational", "content-block"],
    )
    async def test_files_without_text_evaluate_the_placeholder(
        self, evaluate: MagicMock, message: HumanMessage
    ) -> None:
        middleware = _pii([GuardrailScope.AGENT], stage=_PRE)

        await _hook(middleware, "before_agent").abefore_agent(
            {"messages": [message]}, MagicMock()
        )

        (call,) = evaluate.call_args_list
        assert call.args[0] == NO_TEXT_PLACEHOLDER
        assert _forwarded(call) == [(_HELD_ID, "cv.pdf", "application/pdf")]

    async def test_an_attachments_block_is_evaluated_as_text(
        self, evaluate: MagicMock
    ) -> None:
        block = _block(_wire())
        middleware = _pii([GuardrailScope.AGENT], stage=_PRE)

        await _hook(middleware, "before_agent").abefore_agent(
            {"messages": [HumanMessage(content=block)]}, MagicMock()
        )

        (call,) = evaluate.call_args_list
        assert call.args[0] == block
        assert _forwarded(call) == [(_HELD_ID, "cv.pdf", "application/pdf")]

    async def test_no_text_and_no_files_skips_the_evaluation(
        self, evaluate: MagicMock
    ) -> None:
        middleware = _pii([GuardrailScope.AGENT], stage=_PRE)

        await _hook(middleware, "before_agent").abefore_agent(
            {"messages": [HumanMessage(content="")]}, MagicMock()
        )

        evaluate.assert_not_called()

    async def test_a_text_scope_parameter_skips_a_message_of_files_only(
        self, evaluate: MagicMock
    ) -> None:
        middleware = _byo_scoped_by_parameter("Text")
        tagged = {
            "type": "file",
            "base64": "JVBERi0=",
            "mime_type": "application/pdf",
            "extras": {
                "uipath_attachment_id": _HELD_ID,
                "uipath_file_name": "cv.pdf",
            },
        }

        await _hook(middleware, "before_agent").abefore_agent(
            {"messages": [HumanMessage(content=[tagged])]}, MagicMock()
        )

        evaluate.assert_not_called()

    async def test_llm_post_forwards_only_the_files_the_response_references(
        self, evaluate: MagicMock
    ) -> None:
        state: AgentState[Any] = {
            "messages": [
                HumanMessage(content=f"compare {_block(_wire(), _wire(_OTHER_ID))}"),
                AIMessage(
                    content="Analyzing the second file.",
                    tool_calls=[
                        {
                            "name": "analyze",
                            "args": {"attachments": [{"ID": _OTHER_ID}]},
                            "id": "tc1",
                        }
                    ],
                ),
            ]
        }
        middleware = _pii([GuardrailScope.LLM], stage=_POST)

        await _hook(middleware, "after_model").aafter_model(state, MagicMock())

        (call,) = evaluate.call_args_list
        assert call.args[0] == "Analyzing the second file."
        assert _forwarded(call) == [(_OTHER_ID, "cv.pdf", "application/pdf")]

    async def test_agent_post_forwards_the_files_the_structured_response_references(
        self, evaluate: MagicMock
    ) -> None:
        state: AgentState[Any] = {
            "messages": [
                HumanMessage(content=f"pick one {_block(_wire(), _wire(_OTHER_ID))}"),
                AIMessage(content="Picked."),
            ],
            "structured_response": {"chosen": {"ID": _HELD_ID}},
        }
        middleware = _pii([GuardrailScope.AGENT], stage=_POST)

        await _hook(middleware, "after_agent").aafter_agent(state, MagicMock())

        (call,) = evaluate.call_args_list
        assert call.args[0] == "Picked."
        assert _forwarded(call) == [(_HELD_ID, "cv.pdf", "application/pdf")]

    async def test_agent_post_without_references_sends_an_empty_attachment_list(
        self, evaluate: MagicMock
    ) -> None:
        state: AgentState[Any] = {
            "messages": [
                HumanMessage(content=f"summarise {_block(_wire())}"),
                AIMessage(content="A strong candidate."),
            ]
        }
        middleware = _pii([GuardrailScope.AGENT], stage=_POST)

        await _hook(middleware, "after_agent").aafter_agent(state, MagicMock())

        (call,) = evaluate.call_args_list
        assert call.kwargs == {"attachments": []}

    async def test_text_scope_never_forwards(self, evaluate: MagicMock) -> None:
        middleware = _pii(
            [GuardrailScope.AGENT], stage=_PRE, detection_scope=DetectionScope.TEXT
        )

        await _hook(middleware, "before_agent").abefore_agent(
            {"messages": [HumanMessage(content=f"hi {_block(_wire())}")]},
            MagicMock(),
        )

        (call,) = evaluate.call_args_list
        assert call.kwargs == {}
        assert [
            (p.id, p.value)
            for p in call.args[1].validator_parameters
            if p.id == "detectionScope"
        ] == [("detectionScope", "Text")]

    async def test_default_scope_never_forwards(self, evaluate: MagicMock) -> None:
        middleware = _FACTORIES["llm_as_judge"](stage=_PRE)

        await _hook(middleware, "before_agent").abefore_agent(
            {"messages": [HumanMessage(content=f"hi {_block(_wire())}")]},
            MagicMock(),
        )

        (call,) = evaluate.call_args_list
        assert call.kwargs == {}
        assert all(p.id != "detectionScope" for p in call.args[1].validator_parameters)

    async def test_a_text_scope_parameter_never_forwards(
        self, evaluate: MagicMock
    ) -> None:
        middleware = _byo_scoped_by_parameter("Text")

        await _hook(middleware, "before_agent").abefore_agent(
            {"messages": [HumanMessage(content=f"hi {_block(_wire())}")]},
            MagicMock(),
        )

        (call,) = evaluate.call_args_list
        assert call.kwargs == {}

    @pytest.mark.parametrize(
        ("detection_scope", "kwargs"),
        [
            (DetectionScope.FILES, {"attachments": []}),
            (DetectionScope.BOTH, {"attachments": []}),
            (None, {}),
            (DetectionScope.TEXT, {}),
        ],
        ids=["files", "both", "default", "text"],
    )
    async def test_no_files_sends_an_empty_list_only_when_files_were_requested(
        self,
        evaluate: MagicMock,
        detection_scope: DetectionScope | None,
        kwargs: dict[str, Any],
    ) -> None:
        middleware = _pii(
            [GuardrailScope.AGENT], stage=_PRE, detection_scope=detection_scope
        )

        await _hook(middleware, "before_agent").abefore_agent(
            {"messages": [HumanMessage(content="screen this")]}, MagicMock()
        )

        (call,) = evaluate.call_args_list
        assert call.args == ("screen this", middleware._guardrail)
        assert call.kwargs == kwargs

    async def test_a_files_scope_parameter_sends_an_empty_list(
        self, evaluate: MagicMock
    ) -> None:
        middleware = _byo_scoped_by_parameter("Files")

        await _hook(middleware, "before_agent").abefore_agent(
            {"messages": [HumanMessage(content="screen this")]}, MagicMock()
        )

        (call,) = evaluate.call_args_list
        assert call.kwargs == {"attachments": []}

    async def test_a_poisoned_state_evaluates_without_files(
        self, evaluate: MagicMock
    ) -> None:
        middleware = _pii([GuardrailScope.AGENT], stage=_PRE)

        with patch(
            "uipath_langchain.guardrails.middlewares._base.build_registry",
            side_effect=RuntimeError("poisoned"),
        ):
            await _hook(middleware, "before_agent").abefore_agent(
                {"messages": [HumanMessage(content=f"hi {_block(_wire())}")]},
                MagicMock(),
            )

        (call,) = evaluate.call_args_list
        assert call.args[0].startswith("hi ")
        # Says nothing about files: an explicit [] would let a Files-only guardrail
        # pass without judging the text.
        assert call.kwargs == {}


class TestToolHook:
    @staticmethod
    def _request(args: dict[str, Any], state: Any) -> ToolCallRequest:
        tool_call: Any = {"id": "tc1", "name": "analyze", "args": args}
        return ToolCallRequest(
            tool_call=tool_call, tool=MagicMock(), state=state, runtime=MagicMock()
        )

    @staticmethod
    def _returning(content: str) -> Callable[[ToolCallRequest], Any]:
        async def handler(request: ToolCallRequest) -> ToolMessage:
            return ToolMessage(content=content, tool_call_id="tc1")

        return handler

    @staticmethod
    def _held_state() -> dict[str, Any]:
        return {"messages": [HumanMessage(content=f"read {_block(_wire())}")]}

    async def _run(
        self,
        stage: GuardrailExecutionStage,
        args: dict[str, Any],
        state: Any,
        result: str = "done",
    ) -> None:
        middleware = _pii([GuardrailScope.TOOL], stage=stage, tools=["analyze"])
        await _hook(middleware, "wrap_tool_call").awrap_tool_call(
            self._request(args, state), self._returning(result)
        )

    async def test_pre_forwards_a_held_file_with_the_registry_name_and_type(
        self, evaluate: MagicMock
    ) -> None:
        args = {"document": _wire(_HELD_ID.upper(), "renamed.exe", "x/y")}

        await self._run(_PRE, args, self._held_state())

        (call,) = evaluate.call_args_list
        assert call.args[0] == args
        assert _forwarded(call) == [(_HELD_ID, "cv.pdf", "application/pdf")]

    async def test_pre_skips_an_id_the_run_never_held(
        self, evaluate: MagicMock
    ) -> None:
        await self._run(
            _PRE, {"document": _wire(_UNKNOWN_ID, "secret.pdf")}, self._held_state()
        )

        (call,) = evaluate.call_args_list
        assert call.kwargs == {"attachments": []}

    async def test_pre_without_a_mention_sends_an_empty_attachment_list(
        self, evaluate: MagicMock
    ) -> None:
        await self._run(_PRE, {"question": "summarise"}, self._held_state())

        (call,) = evaluate.call_args_list
        assert call.kwargs == {"attachments": []}

    async def test_pre_tolerates_a_request_without_state(
        self, evaluate: MagicMock
    ) -> None:
        await self._run(_PRE, {"document": {"ID": _HELD_ID}}, None)

        (call,) = evaluate.call_args_list
        assert call.kwargs == {"attachments": []}

    async def test_post_forwards_a_file_the_tool_returned(
        self, evaluate: MagicMock
    ) -> None:
        returned = _wire(_OTHER_ID, "report.pdf")

        await self._run(
            _POST, {"q": "x"}, self._held_state(), json.dumps({"report": returned})
        )

        (call,) = evaluate.call_args_list
        assert call.args[0] == {"report": returned}
        assert _forwarded(call) == [(_OTHER_ID, "report.pdf", "application/pdf")]

    async def test_post_plain_text_result_sends_an_empty_attachment_list(
        self, evaluate: MagicMock
    ) -> None:
        await self._run(_POST, {"q": "x"}, self._held_state(), "all good")

        (call,) = evaluate.call_args_list
        assert call.args[0] == {"output": "all good"}
        assert call.kwargs == {"attachments": []}


def _rejection(status: int) -> EnrichedException:
    request = httpx.Request("POST", "https://example.com/validate")
    response = httpx.Response(status, request=request, text="rejected")
    return EnrichedException(
        httpx.HTTPStatusError(str(status), request=request, response=response)
    )


def _reject_files_then(
    verdict: GuardrailValidationResult,
) -> Callable[..., GuardrailValidationResult]:
    def evaluate(*args: Any, **kwargs: Any) -> GuardrailValidationResult:
        if kwargs.get("attachments"):
            raise _rejection(400)
        return verdict

    return evaluate


_FAILED = GuardrailValidationResult(
    result=GuardrailValidationResultType.VALIDATION_FAILED, reason="email found"
)
_MESSAGE_HOOKS = [
    pytest.param(GuardrailScope.AGENT, "before_agent", "abefore_agent", id="agent"),
    pytest.param(GuardrailScope.LLM, "before_model", "abefore_model", id="llm"),
]


class TestAttachmentRejection:
    """A validate call whose attachment references the backend rejects (HTTP 400)
    is re-evaluated on the text alone, so a file never disables the guardrail."""

    @pytest.mark.parametrize(("scope", "suffix", "method"), _MESSAGE_HOOKS)
    async def test_message_hook_applies_the_text_only_verdict(
        self,
        evaluate: MagicMock,
        caplog: pytest.LogCaptureFixture,
        scope: GuardrailScope,
        suffix: str,
        method: str,
    ) -> None:
        evaluate.side_effect = _reject_files_then(_FAILED)
        text = f"mail me at a@b.com {_block(_wire())}"
        middleware = _pii([scope], stage=_PRE, action=BlockAction())

        with pytest.raises(AgentRuntimeError) as exc_info:
            await getattr(_hook(middleware, suffix), method)(
                {"messages": [HumanMessage(content=text)]}, MagicMock()
            )

        assert exc_info.value.error_info.code == AgentRuntimeError.full_code(
            AgentRuntimeErrorCode.TERMINATION_GUARDRAIL_VIOLATION
        )
        with_files, text_only = evaluate.call_args_list
        assert _forwarded(with_files) == [(_HELD_ID, "cv.pdf", "application/pdf")]
        assert text_only.args == (text, middleware._guardrail)
        assert text_only.kwargs == {}
        assert "rejected the attachment references" in caplog.text

    @pytest.mark.parametrize(
        ("status", "content"),
        [
            pytest.param(400, "mail me at a@b.com", id="400-without-files"),
            pytest.param(500, f"mail me {_block(_wire())}", id="500-with-files"),
        ],
    )
    async def test_message_hook_hands_other_failures_to_the_error_handler(
        self,
        evaluate: MagicMock,
        caplog: pytest.LogCaptureFixture,
        status: int,
        content: str,
    ) -> None:
        evaluate.side_effect = _rejection(status)
        middleware = _pii([GuardrailScope.LLM], stage=_PRE, action=BlockAction())

        with caplog.at_level(logging.ERROR):
            await _hook(middleware, "before_model").abefore_model(
                {"messages": [HumanMessage(content=content)]}, MagicMock()
            )

        assert evaluate.call_count == 1
        (record,) = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert record.getMessage() == f"Error evaluating guardrail '{middleware._name}'"
        assert record.exc_info is not None
        assert isinstance(record.exc_info[1], EnrichedException)
        assert record.exc_info[1].status_code == status

    @pytest.mark.parametrize(
        ("stage", "args", "result"),
        [
            pytest.param(_PRE, {"document": _wire()}, "done", id="pre"),
            pytest.param(
                _POST,
                {"q": "x"},
                json.dumps({"report": _wire(_OTHER_ID, "report.pdf")}),
                id="post",
            ),
        ],
    )
    async def test_tool_hook_applies_the_text_only_verdict(
        self,
        evaluate: MagicMock,
        stage: GuardrailExecutionStage,
        args: dict[str, Any],
        result: str,
    ) -> None:
        evaluate.side_effect = _reject_files_then(_FAILED)
        middleware = _pii(
            [GuardrailScope.TOOL], stage=stage, tools=["analyze"], action=BlockAction()
        )

        with pytest.raises(AgentRuntimeError):
            await _hook(middleware, "wrap_tool_call").awrap_tool_call(
                TestToolHook._request(args, TestToolHook._held_state()),
                TestToolHook._returning(result),
            )

        with_files, text_only = evaluate.call_args_list
        assert with_files.kwargs["attachments"]
        assert text_only.args == (with_files.args[0], middleware._guardrail)
        assert text_only.kwargs == {}

    async def test_tool_hook_hands_a_400_without_files_to_the_error_handler(
        self, evaluate: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        evaluate.side_effect = _rejection(400)
        middleware = _pii(
            [GuardrailScope.TOOL], stage=_PRE, tools=["analyze"], action=BlockAction()
        )

        with caplog.at_level(logging.ERROR):
            result = await _hook(middleware, "wrap_tool_call").awrap_tool_call(
                TestToolHook._request({"q": "x"}, TestToolHook._held_state()),
                TestToolHook._returning("done"),
            )

        assert isinstance(result, ToolMessage)
        assert result.content == "done"
        assert evaluate.call_count == 1
        (record,) = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert record.exc_info is not None
        assert isinstance(record.exc_info[1], EnrichedException)


_FACTORIES: dict[str, Callable[..., Any]] = {
    "llm_as_judge": lambda **kwargs: UiPathLLMAsJudgeMiddleware(
        scopes=[GuardrailScope.AGENT],
        action=_LOG,
        guardrail_text="No financial advice.",
        model="gpt-4o-2024-08-06",
        **kwargs,
    ),
    "pii_detection": lambda **kwargs: UiPathPIIDetectionMiddleware(
        scopes=[GuardrailScope.AGENT],
        action=_LOG,
        entities=[PIIDetectionEntity(PIIDetectionEntityType.EMAIL)],
        **kwargs,
    ),
    "harmful_content": lambda **kwargs: UiPathHarmfulContentMiddleware(
        scopes=[GuardrailScope.AGENT],
        action=_LOG,
        entities=[HarmfulContentEntity("Hate")],
        **kwargs,
    ),
    "prompt_injection": lambda **kwargs: UiPathPromptInjectionMiddleware(
        action=_LOG, **kwargs
    ),
    "user_prompt_attacks": lambda **kwargs: UiPathUserPromptAttacksMiddleware(
        action=_LOG, **kwargs
    ),
    "intellectual_property": lambda **kwargs: UiPathIntellectualPropertyMiddleware(
        scopes=[GuardrailScope.AGENT], action=_LOG, entities=["Text"], **kwargs
    ),
    "byo": lambda **kwargs: UiPathByoGuardrailMiddleware(
        validator_name="my-validator",
        scopes=[GuardrailScope.AGENT],
        action=_LOG,
        **kwargs,
    ),
}


def _scope_parameters(middleware: Any) -> list[tuple[str, Any]]:
    return [
        (p.id, p.value)
        for p in middleware._guardrail.validator_parameters
        if p.id.lower() == "detectionscope"
    ]


@pytest.mark.parametrize("factory", list(_FACTORIES.values()), ids=list(_FACTORIES))
class TestDetectionScope:
    def test_none_sends_no_parameter(self, factory: Callable[..., Any]) -> None:
        middleware = factory()

        assert middleware.detection_scope is None
        assert _scope_parameters(middleware) == []
        assert (
            middleware._guardrail.validator_parameters
            == middleware._create_guardrail().validator_parameters
        )

    @pytest.mark.parametrize(
        "value,expected",
        [
            (DetectionScope.TEXT, DetectionScope.TEXT),
            (DetectionScope.BOTH, DetectionScope.BOTH),
            ("Files", DetectionScope.FILES),
        ],
    )
    def test_sends_the_scope(
        self, factory: Callable[..., Any], value: Any, expected: DetectionScope
    ) -> None:
        middleware = factory(detection_scope=value)

        assert middleware.detection_scope is expected
        assert _scope_parameters(middleware) == [("detectionScope", expected.value)]
        (parameter,) = [
            p
            for p in middleware._guardrail.validator_parameters
            if p.id == "detectionScope"
        ]
        assert isinstance(parameter, EnumParameterValue)
        assert parameter.parameter_type == "enum"

    @pytest.mark.parametrize("value", ["files", "", 123])
    def test_rejects_anything_else(
        self, factory: Callable[..., Any], value: Any
    ) -> None:
        with pytest.raises(
            ValueError, match="detection_scope must be a DetectionScope member"
        ):
            factory(detection_scope=value)


class _Model(GenericFakeChatModel):
    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "_Model":
        return self


@tool
def analyze(document: dict[str, Any]) -> str:
    """Analyze a document."""
    return "It is a CV."


class TestAgentRun:
    @staticmethod
    def _agent(middleware: Any) -> Any:
        model = _Model(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "analyze",
                                "args": {"document": {"ID": _HELD_ID}},
                                "id": "tc1",
                            }
                        ],
                    ),
                    AIMessage(content="The CV is fine."),
                ]
            )
        )
        return create_agent(model=model, tools=[analyze], middleware=[*middleware])

    async def test_a_run_without_files_sends_text_only_calls(
        self, evaluate: MagicMock
    ) -> None:
        middleware = _pii(
            [GuardrailScope.AGENT, GuardrailScope.LLM, GuardrailScope.TOOL],
            tools=["analyze"],
            detection_scope=None,
        )

        await self._agent(middleware).ainvoke(
            {"messages": [HumanMessage(content="Is this a CV?")]}
        )

        assert len(evaluate.call_args_list) == 7
        for call in evaluate.call_args_list:
            assert len(call.args) == 2
            assert call.args[1] is middleware._guardrail
            assert call.kwargs == {}

    async def test_a_run_forwards_the_files_each_hook_may_see(
        self, evaluate: MagicMock
    ) -> None:
        text = f"Is this a CV? {_block(_wire(), _wire(_OTHER_ID, 'b.csv', 'text/csv'))}"
        middleware = _pii(
            [GuardrailScope.AGENT, GuardrailScope.LLM, GuardrailScope.TOOL],
            tools=["analyze"],
        )

        await self._agent(middleware).ainvoke(
            {"messages": [HumanMessage(content=text)]}
        )

        seen = [
            (call.args[0], [a[0] for a in _forwarded(call)])
            for call in evaluate.call_args_list
        ]
        assert seen[0] == (text, [_HELD_ID, _OTHER_ID])
        assert ({"document": {"ID": _HELD_ID}}, [_HELD_ID]) in seen
        assert seen[-1] == ("The CV is fine.", [])
