"""Decorated LangChain targets forward the attachment references each call sees."""

import copy
import json
from collections.abc import Callable, Iterator, Sequence
from typing import Annotated, Any
from unittest.mock import MagicMock, patch

import pytest
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool, StructuredTool, tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command
from typing_extensions import TypedDict
from uipath.core.guardrails import (
    GuardrailValidationResult,
    GuardrailValidationResultType,
)
from uipath.platform.attachments import Attachment
from uipath.platform.guardrails import GuardrailAttachment
from uipath.platform.guardrails.guardrails import EnumParameterValue

from uipath_langchain._utils._attachments import render_attachments_block
from uipath_langchain.agent.exceptions import AgentRuntimeError, AgentRuntimeErrorCode
from uipath_langchain.guardrails import (
    BlockAction,
    ByoValidator,
    DetectionScope,
    GuardrailAction,
    GuardrailExecutionStage,
    LogAction,
    PIIDetectionEntity,
    PIIDetectionEntityType,
    PIIValidator,
    guardrail,
)
from uipath_langchain.guardrails._langchain_adapter import (
    LangChainGuardrailAdapter,
    _apply_agent_input_guardrail,
    _apply_agent_output_guardrail,
    _apply_llm_post,
    _apply_llm_pre,
    _wrap_tool_with_guardrail,
)
from uipath_langchain.guardrails.middlewares._attachments import (
    JOB_ATTACHMENTS_STATE_KEY,
    NO_TEXT_PLACEHOLDER,
)

_HELD_ID = "7f2c1e44-0b3a-4a1e-9d55-2f9a1c3b8e10"
_OTHER_ID = "0d9b3a52-6c1f-4e8b-a0f2-5b7c9d1e3f40"
_TYPED_ID = "3c5e7a91-2b4d-4f6a-8c0e-1d3f5a7b9c20"
_UNKNOWN_ID = "11111111-2222-4333-8444-555555555555"
_DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

_PRE = GuardrailExecutionStage.PRE
_POST = GuardrailExecutionStage.POST
_PASSED = GuardrailValidationResult(
    result=GuardrailValidationResultType.PASSED, reason=""
)
_FAILED = GuardrailValidationResult(
    result=GuardrailValidationResultType.VALIDATION_FAILED, reason="email found"
)


def _wire(
    attachment_id: str = _HELD_ID,
    name: str = "cv.pdf",
    mime_type: str = "application/pdf",
) -> dict[str, str]:
    return {"ID": attachment_id, "FullName": name, "MimeType": mime_type}


def _reference(
    attachment_id: str = _HELD_ID,
    name: str = "cv.pdf",
    mime_type: str = "application/pdf",
) -> GuardrailAttachment:
    return GuardrailAttachment(id=attachment_id, file_name=name, mime_type=mime_type)


def _file_block() -> dict[str, Any]:
    return {
        "type": "file",
        "base64": "JVBERi0=",
        "mime_type": "application/pdf",
        "extras": {"uipath_attachment_id": _HELD_ID, "uipath_file_name": "cv.pdf"},
    }


def _block(*attachments: dict[str, str]) -> str:
    return render_attachments_block(
        [
            {"id": a["ID"], "full_name": a["FullName"], "mime_type": a["MimeType"]}
            for a in attachments
        ]
    )


_HELD = _wire()
_OTHER = _wire(_OTHER_ID, "notes.txt", "text/plain")
_TYPED = Attachment.model_validate(_wire(_TYPED_ID, "resume.docx", _DOCX))


class _Replace(GuardrailAction):
    """An action that rewrites the flagged text."""

    def handle_validation_result(
        self, result: GuardrailValidationResult, data: Any, guardrail_name: str
    ) -> str:
        return "MODIFIED"


def _pii(
    stage: GuardrailExecutionStage,
    detection_scope: DetectionScope | None = DetectionScope.BOTH,
    action: GuardrailAction | None = None,
) -> Callable[[Any], Any]:
    decorator: Callable[[Any], Any] = guardrail(
        validator=PIIValidator(
            entities=[PIIDetectionEntity(PIIDetectionEntityType.EMAIL)]
        ),
        action=action or LogAction(),
        name="PII",
        stage=stage,
        detection_scope=detection_scope,
    )
    return decorator


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


def _only_call(evaluate: MagicMock) -> Any:
    (call,) = evaluate.call_args_list
    return call


def _detection_scope(call: Any) -> Any:
    _, built_in = call.args
    return next(
        parameter.value
        for parameter in built_in.validator_parameters
        if parameter.id == "detectionScope"
    )


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------


def _analyze_files_tool() -> BaseTool:
    @tool
    def analyze_files(attachments: list[dict[str, Any]], analysisTask: str) -> str:
        """Analyze the attached files."""
        return "analyzed"

    return analyze_files


def _report_tool(as_command: bool) -> BaseTool:
    @tool
    def make_report(topic: str) -> Any:
        """Write a report file about the topic."""
        content = json.dumps({"report": _wire(_TYPED_ID, "report.docx", _DOCX)})
        if as_command:
            return Command(
                update={
                    "messages": [ToolMessage(content=content, tool_call_id="call-1")]
                }
            )
        return content

    return make_report


def _report_call() -> dict[str, Any]:
    return {
        "name": "make_report",
        "args": {"topic": "hiring"},
        "id": "call-1",
        "type": "tool_call",
    }


class _FakeChatModel(BaseChatModel):
    reply: AIMessage

    @property
    def _llm_type(self) -> str:
        return "fake"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self.reply)])


def _tool_call_reply(
    attachment_id: str, content: str | list[str | dict[Any, Any]] = ""
) -> AIMessage:
    return AIMessage(
        content=content,
        tool_calls=[
            {
                "name": "analyze_files",
                "args": {"attachments": [{"ID": attachment_id}]},
                "id": "call-1",
            }
        ],
    )


class _AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    cv: Attachment
    report: Attachment
    structured_response: dict[str, Any]


class _InvokableStateGraph(StateGraph):  # type: ignore[type-arg]
    """A builder that runs itself, the shape the StateGraph wrap guards."""

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        return self.compile().invoke(input, config, **kwargs)

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        return await self.compile().ainvoke(input, config, **kwargs)


def _agent(kind: str, decorate: Callable[[Any], Any], update: dict[str, Any]) -> Any:
    def agent(state: _AgentState) -> dict[str, Any]:
        return update

    builder: Any = (
        _InvokableStateGraph(_AgentState)
        if kind == "stategraph"
        else StateGraph(_AgentState)
    )
    builder.add_node("agent", agent)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)
    return decorate(builder if kind == "stategraph" else builder.compile())


_AGENT_KINDS = pytest.mark.parametrize("kind", ["compiled", "stategraph"])


# ---------------------------------------------------------------------------
# Tool scope
# ---------------------------------------------------------------------------


class TestToolScope:
    def test_pre_forwards_the_files_the_arguments_describe(
        self, evaluate: MagicMock
    ) -> None:
        analyze = _pii(_PRE)(_analyze_files_tool())

        analyze.invoke({"attachments": [_HELD], "analysisTask": "Summarise the CV"})

        call = _only_call(evaluate)
        assert call.args[0] == {
            "attachments": [_HELD],
            "analysisTask": "Summarise the CV",
        }
        assert call.kwargs["attachments"] == [_reference()]

    def test_pre_sends_no_attachments_for_a_bare_id(self, evaluate: MagicMock) -> None:
        analyze = _pii(_PRE)(_analyze_files_tool())

        analyze.invoke(
            {"attachments": [{"ID": _HELD_ID}], "analysisTask": "Summarise the CV"}
        )

        assert _only_call(evaluate).kwargs == {"attachments": []}

    @pytest.mark.parametrize("as_command", [False, True], ids=["message", "command"])
    def test_post_forwards_the_files_the_result_describes(
        self, evaluate: MagicMock, as_command: bool
    ) -> None:
        make_report = _pii(_POST)(_report_tool(as_command))

        make_report.invoke(_report_call())

        call = _only_call(evaluate)
        assert call.kwargs["attachments"] == [
            _reference(_TYPED_ID, "report.docx", _DOCX)
        ]


# ---------------------------------------------------------------------------
# Tool scope through ainvoke
# ---------------------------------------------------------------------------


def _coroutine_tool(name: str, result: str, ran: list[dict[str, Any]]) -> BaseTool:
    async def run(**kwargs: Any) -> str:
        ran.append(kwargs)
        return result

    return StructuredTool.from_function(
        coroutine=run,
        name=name,
        description="Runs only as a coroutine.",
        args_schema={
            "type": "object",
            "properties": {
                "attachments": {"type": "array", "items": {"type": "object"}},
                "analysisTask": {"type": "string"},
                "topic": {"type": "string"},
            },
        },
    )


def _analyze_args() -> dict[str, Any]:
    return {"attachments": [_HELD], "analysisTask": "Summarise the CV"}


def _assert_blocked(exc_info: pytest.ExceptionInfo[AgentRuntimeError]) -> None:
    assert exc_info.value.error_info.code == AgentRuntimeError.full_code(
        AgentRuntimeErrorCode.TERMINATION_GUARDRAIL_VIOLATION
    )


class _ToolCallingModel(GenericFakeChatModel):
    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "_ToolCallingModel":
        return self


class TestToolAinvoke:
    async def test_pre_blocks_a_coroutine_only_tool(self, evaluate: MagicMock) -> None:
        evaluate.return_value = _FAILED
        ran: list[dict[str, Any]] = []
        analyze = _pii(_PRE, action=BlockAction())(
            _coroutine_tool("analyze_files", "analyzed", ran)
        )

        with pytest.raises(AgentRuntimeError) as exc_info:
            await analyze.ainvoke(_analyze_args())

        _assert_blocked(exc_info)
        assert ran == []
        call = _only_call(evaluate)
        assert call.args[0] == _analyze_args()
        assert call.kwargs["attachments"] == [_reference()]

    async def test_post_sees_a_coroutine_only_tool_result(
        self, evaluate: MagicMock
    ) -> None:
        report = json.dumps({"report": _wire(_TYPED_ID, "report.docx", _DOCX)})
        make_report = _pii(_POST)(_coroutine_tool("make_report", report, []))

        result = await make_report.ainvoke(_report_call())

        assert result.content == report
        call = _only_call(evaluate)
        assert call.args[0] == json.loads(report)
        assert call.kwargs["attachments"] == [
            _reference(_TYPED_ID, "report.docx", _DOCX)
        ]

    @pytest.mark.parametrize("method", ["invoke", "ainvoke"])
    @pytest.mark.parametrize("guardrails", [1, 2], ids=["one", "stacked"])
    async def test_a_sync_tool_is_checked_once_per_stage(
        self, evaluate: MagicMock, method: str, guardrails: int
    ) -> None:
        analyze = _analyze_files_tool()
        for _ in range(guardrails):
            analyze = _pii(GuardrailExecutionStage.PRE_AND_POST)(analyze)

        if method == "invoke":
            analyze.invoke(_analyze_args())
        else:
            await analyze.ainvoke(_analyze_args())

        calls = evaluate.call_args_list
        checked = [_analyze_args()] * guardrails + [{"output": "analyzed"}] * guardrails
        assert [call.args[0] for call in calls] == checked
        assert calls[0].kwargs["attachments"] == [_reference()]

    async def test_a_tool_that_invokes_itself_is_checked_on_every_call(
        self, evaluate: MagicMock
    ) -> None:
        guarded: list[BaseTool] = []

        @tool
        def countdown(n: int) -> str:
            """Count down to zero through the guarded tool."""
            return guarded[0].invoke({"n": n - 1}) if n else "done"

        guarded.append(_pii(_PRE)(countdown))

        await guarded[0].ainvoke({"n": 1})

        assert [call.args[0] for call in evaluate.call_args_list] == [
            {"n": 1},
            {"n": 0},
        ]

    async def test_create_agent_blocks_a_decorated_coroutine_tool_at_pre(
        self, evaluate: MagicMock
    ) -> None:
        evaluate.return_value = _FAILED
        ran: list[dict[str, Any]] = []
        analyze = _pii(_PRE, action=BlockAction())(
            _coroutine_tool("analyze_files", "analyzed", ran)
        )
        model = _ToolCallingModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "analyze_files",
                                "args": _analyze_args(),
                                "id": "call-1",
                            }
                        ],
                    ),
                    AIMessage(content="It is a CV."),
                ]
            )
        )
        agent = create_agent(model=model, tools=[analyze])

        with pytest.raises(AgentRuntimeError) as exc_info:
            await agent.ainvoke({"messages": [HumanMessage(f"Check {_block(_HELD)}")]})

        _assert_blocked(exc_info)
        assert ran == []
        call = _only_call(evaluate)
        assert call.args[0] == _analyze_args()
        assert call.kwargs["attachments"] == [_reference()]


# ---------------------------------------------------------------------------
# LLM scope
# ---------------------------------------------------------------------------


class TestLLMScope:
    def test_pre_forwards_every_file_the_messages_hold(
        self, evaluate: MagicMock
    ) -> None:
        llm = _pii(_PRE)(_FakeChatModel(reply=AIMessage(content="ok")))
        prompt = f"Compare these files {_block(_HELD, _OTHER)}"

        llm.invoke([SystemMessage(content="You review CVs."), HumanMessage(prompt)])

        call = _only_call(evaluate)
        assert call.args[0] == prompt
        assert call.kwargs["attachments"] == [
            _reference(),
            _reference(_OTHER_ID, "notes.txt", "text/plain"),
        ]

    def test_pre_evaluates_a_block_only_message_as_its_text(
        self, evaluate: MagicMock
    ) -> None:
        llm = _pii(_PRE)(_FakeChatModel(reply=AIMessage(content="ok")))
        block = _block(_HELD)

        llm.invoke([HumanMessage(content=block)])

        call = _only_call(evaluate)
        assert call.args[0] == block
        assert call.kwargs["attachments"] == [_reference()]

    def test_pre_evaluates_the_placeholder_for_a_file_without_text(
        self, evaluate: MagicMock
    ) -> None:
        evaluate.return_value = _FAILED
        llm = _pii(_PRE, action=_Replace())(
            _FakeChatModel(reply=AIMessage(content="ok"))
        )
        file_block = _file_block()
        message = HumanMessage(content=[file_block])

        llm.invoke([message])

        call = _only_call(evaluate)
        assert call.args[0] == NO_TEXT_PLACEHOLDER
        assert call.kwargs["attachments"] == [_reference()]
        assert message.content == [file_block]

    def test_post_forwards_only_the_held_file_a_tool_call_names(
        self, evaluate: MagicMock
    ) -> None:
        evaluate.return_value = _FAILED
        llm = _pii(_POST, action=_Replace())(
            _FakeChatModel(reply=_tool_call_reply(_HELD_ID))
        )

        response = llm.invoke([HumanMessage(f"Check {_block(_HELD, _OTHER)}")])

        call = _only_call(evaluate)
        assert call.args[0] == NO_TEXT_PLACEHOLDER
        assert call.kwargs["attachments"] == [_reference()]
        assert response.content == ""

    def test_post_ignores_an_id_the_run_does_not_hold(
        self, evaluate: MagicMock
    ) -> None:
        llm = _pii(_POST)(
            _FakeChatModel(reply=_tool_call_reply(_UNKNOWN_ID, content="Checked"))
        )

        llm.invoke([HumanMessage(f"Check {_block(_HELD)}")])

        call = _only_call(evaluate)
        assert call.args[0] == "Checked"
        assert call.kwargs == {"attachments": []}

    def test_post_evaluates_the_text_of_list_content(self, evaluate: MagicMock) -> None:
        evaluate.return_value = _FAILED
        content: list[str | dict[Any, Any]] = [
            {"type": "text", "text": "Checking the CV."},
            {
                "type": "tool_use",
                "id": "call-1",
                "name": "analyze_files",
                "input": {"attachments": [{"ID": _HELD_ID}]},
            },
        ]
        llm = _pii(_POST, action=_Replace())(
            _FakeChatModel(reply=_tool_call_reply(_HELD_ID, copy.deepcopy(content)))
        )

        response = llm.invoke([HumanMessage(f"Check {_block(_HELD, _OTHER)}")])

        call = _only_call(evaluate)
        assert call.args[0] == "Checking the CV."
        assert call.kwargs["attachments"] == [_reference()]
        assert response.content == content

    def test_post_without_text_or_held_files_is_not_evaluated(
        self, evaluate: MagicMock
    ) -> None:
        llm = _pii(_POST)(_FakeChatModel(reply=_tool_call_reply(_UNKNOWN_ID)))

        llm.invoke([HumanMessage(f"Check {_block(_HELD)}")])

        evaluate.assert_not_called()


# ---------------------------------------------------------------------------
# Agent scope
# ---------------------------------------------------------------------------


class TestAgentScope:
    @_AGENT_KINDS
    def test_input_forwards_typed_attachments_with_the_block_files(
        self, evaluate: MagicMock, kind: str
    ) -> None:
        prompt = f"Screen this candidate {_block(_HELD)}"
        agent = _agent(kind, _pii(_PRE), {"messages": [AIMessage(content="Done")]})

        agent.invoke({"messages": [HumanMessage(prompt)], "cv": _TYPED})

        call = _only_call(evaluate)
        assert call.args[0] == prompt
        assert call.kwargs["attachments"] == [
            _reference(),
            _reference(_TYPED_ID, "resume.docx", _DOCX),
        ]

    @_AGENT_KINDS
    async def test_async_input_forwards_typed_attachments(
        self, evaluate: MagicMock, kind: str
    ) -> None:
        agent = _agent(kind, _pii(_PRE), {"messages": [AIMessage(content="Done")]})

        await agent.ainvoke({"messages": [HumanMessage("Screen")], "cv": _TYPED})

        assert _only_call(evaluate).kwargs["attachments"] == [
            _reference(_TYPED_ID, "resume.docx", _DOCX)
        ]

    @_AGENT_KINDS
    def test_output_forwards_the_held_file_the_structured_response_names(
        self, evaluate: MagicMock, kind: str
    ) -> None:
        agent = _agent(
            kind,
            _pii(_POST),
            {
                "messages": [AIMessage(content="Reviewed")],
                "structured_response": {"reviewed": {"ID": _HELD_ID}},
            },
        )

        agent.invoke({"messages": [HumanMessage(f"Review {_block(_HELD, _OTHER)}")]})

        call = _only_call(evaluate)
        assert call.args[0] == "Reviewed"
        assert call.kwargs["attachments"] == [_reference()]

    @_AGENT_KINDS
    def test_output_forwards_a_typed_attachment_the_agent_returns(
        self, evaluate: MagicMock, kind: str
    ) -> None:
        report = Attachment.model_validate(_wire(_OTHER_ID, "report.pdf"))
        agent = _agent(
            kind,
            _pii(_POST),
            {"messages": [AIMessage(content="Here it is")], "report": report},
        )

        agent.invoke({"messages": [HumanMessage("Write the report")]})

        assert _only_call(evaluate).kwargs["attachments"] == [
            _reference(_OTHER_ID, "report.pdf")
        ]

    @pytest.mark.parametrize(
        ("stage", "message", "update", "typed", "expected"),
        [
            pytest.param(
                _PRE,
                HumanMessage(content=[_file_block()]),
                {"messages": [AIMessage(content="Done")]},
                {"cv": _TYPED},
                [_reference(), _reference(_TYPED_ID, "resume.docx", _DOCX)],
                id="input",
            ),
            pytest.param(
                _POST,
                HumanMessage("Write the report"),
                {
                    "messages": [AIMessage(content="")],
                    "report": Attachment.model_validate(_wire(_OTHER_ID, "report.pdf")),
                },
                {},
                [_reference(_OTHER_ID, "report.pdf")],
                id="output",
            ),
        ],
    )
    def test_files_without_text_evaluate_the_placeholder_and_keep_the_messages(
        self,
        evaluate: MagicMock,
        stage: GuardrailExecutionStage,
        message: HumanMessage,
        update: dict[str, Any],
        typed: dict[str, Any],
        expected: list[GuardrailAttachment],
    ) -> None:
        evaluate.return_value = _FAILED
        contents = [copy.deepcopy(m.content) for m in [message, *update["messages"]]]
        agent = _agent("compiled", _pii(stage, action=_Replace()), update)

        output = agent.invoke({"messages": [message], **typed})

        call = _only_call(evaluate)
        assert call.args[0] == NO_TEXT_PLACEHOLDER
        assert call.kwargs["attachments"] == expected
        assert [m.content for m in output["messages"]] == contents

    def test_output_does_not_forward_the_job_attachments_registry(self) -> None:
        received: list[dict[str, Any]] = []

        def evaluator(*args: Any, **kwargs: Any) -> GuardrailValidationResult:
            received.append(kwargs)
            return _PASSED

        _apply_agent_output_guardrail(
            {
                "messages": [AIMessage(content="Done")],
                JOB_ATTACHMENTS_STATE_KEY: {_HELD_ID: _HELD},
            },
            evaluator,
            LogAction(),
            "PII",
        )

        assert received == [{}]


# ---------------------------------------------------------------------------
# detectionScope=Text keeps every call site text-only
# ---------------------------------------------------------------------------


def _run_tool_pre(detection_scope: DetectionScope) -> None:
    _pii(_PRE, detection_scope)(_analyze_files_tool()).invoke(
        {"attachments": [_HELD], "analysisTask": "Summarise a@b.com"}
    )


def _run_tool_post(detection_scope: DetectionScope) -> None:
    _pii(_POST, detection_scope)(_report_tool(as_command=False)).invoke(_report_call())


def _run_llm_pre(detection_scope: DetectionScope) -> None:
    llm = _pii(_PRE, detection_scope)(_FakeChatModel(reply=AIMessage(content="ok")))
    llm.invoke([HumanMessage(f"Review {_block(_HELD)}")])


def _run_llm_post(detection_scope: DetectionScope) -> None:
    llm = _pii(_POST, detection_scope)(
        _FakeChatModel(reply=_tool_call_reply(_HELD_ID, content="Checking"))
    )
    llm.invoke([HumanMessage(f"Review {_block(_HELD)}")])


def _run_agent_input(detection_scope: DetectionScope) -> None:
    agent = _agent(
        "compiled",
        _pii(_PRE, detection_scope),
        {"messages": [AIMessage(content="Done")]},
    )
    agent.invoke({"messages": [HumanMessage("Screen")], "cv": _TYPED})


def _run_agent_output(detection_scope: DetectionScope) -> None:
    agent = _agent(
        "compiled",
        _pii(_POST, detection_scope),
        {
            "messages": [AIMessage(content="Reviewed")],
            "structured_response": {"reviewed": {"ID": _HELD_ID}},
        },
    )
    agent.invoke({"messages": [HumanMessage(f"Review {_block(_HELD)}")]})


_CALL_SITES = pytest.mark.parametrize(
    "run",
    [
        _run_tool_pre,
        _run_tool_post,
        _run_llm_pre,
        _run_llm_post,
        _run_agent_input,
        _run_agent_output,
    ],
    ids=["tool-pre", "tool-post", "llm-pre", "llm-post", "agent-input", "agent-output"],
)


class TestDetectionScope:
    @_CALL_SITES
    def test_text_scope_sends_no_attachments(
        self, evaluate: MagicMock, run: Callable[[DetectionScope], None]
    ) -> None:
        run(DetectionScope.TEXT)

        call = _only_call(evaluate)
        assert "attachments" not in call.kwargs
        assert _detection_scope(call) == "Text"

    @_CALL_SITES
    def test_both_scope_sends_the_attachments(
        self, evaluate: MagicMock, run: Callable[[DetectionScope], None]
    ) -> None:
        run(DetectionScope.BOTH)

        call = _only_call(evaluate)
        assert call.kwargs["attachments"]
        assert _detection_scope(call) == "Both"


# ---------------------------------------------------------------------------
# Files without text
# ---------------------------------------------------------------------------


def _files_only_llm_pre(detection_scope: DetectionScope | None) -> None:
    llm = _pii(_PRE, detection_scope)(_FakeChatModel(reply=AIMessage(content="ok")))
    llm.invoke([HumanMessage(content=[_file_block()])])


def _files_only_llm_post(detection_scope: DetectionScope | None) -> None:
    llm = _pii(_POST, detection_scope)(_FakeChatModel(reply=_tool_call_reply(_HELD_ID)))
    llm.invoke([HumanMessage(f"Review {_block(_HELD)}")])


def _files_only_agent_input(detection_scope: DetectionScope | None) -> None:
    agent = _agent(
        "compiled",
        _pii(_PRE, detection_scope),
        {"messages": [AIMessage(content="Done")]},
    )
    agent.invoke({"messages": [HumanMessage(content=[_file_block()])], "cv": _TYPED})


def _files_only_agent_output(detection_scope: DetectionScope | None) -> None:
    agent = _agent(
        "compiled",
        _pii(_POST, detection_scope),
        {
            "messages": [AIMessage(content="")],
            "structured_response": {"reviewed": {"ID": _HELD_ID}},
        },
    )
    agent.invoke({"messages": [HumanMessage(f"Review {_block(_HELD)}")]})


_FILES_ONLY_SITES = pytest.mark.parametrize(
    "run",
    [
        _files_only_llm_pre,
        _files_only_llm_post,
        _files_only_agent_input,
        _files_only_agent_output,
    ],
    ids=["llm-pre", "llm-post", "agent-input", "agent-output"],
)


class TestFilesWithoutText:
    @_FILES_ONLY_SITES
    @pytest.mark.parametrize(
        "detection_scope", [DetectionScope.TEXT, None], ids=["text", "default"]
    )
    def test_text_scope_is_not_evaluated(
        self,
        evaluate: MagicMock,
        run: Callable[[DetectionScope | None], None],
        detection_scope: DetectionScope | None,
    ) -> None:
        run(detection_scope)

        evaluate.assert_not_called()

    @_FILES_ONLY_SITES
    @pytest.mark.parametrize(
        "detection_scope",
        [DetectionScope.BOTH, DetectionScope.FILES],
        ids=["both", "files"],
    )
    def test_a_scope_that_includes_files_evaluates_the_placeholder(
        self,
        evaluate: MagicMock,
        run: Callable[[DetectionScope | None], None],
        detection_scope: DetectionScope | None,
    ) -> None:
        run(detection_scope)

        call = _only_call(evaluate)
        assert call.args[0] == NO_TEXT_PLACEHOLDER
        assert call.kwargs["attachments"]

    def test_a_text_scope_parameter_is_not_evaluated(self, evaluate: MagicMock) -> None:
        validator = ByoValidator(
            "byog-judge",
            parameters=[
                EnumParameterValue(
                    parameter_type="enum", id="detectionScope", value="Text"
                )
            ],
        )
        llm = guardrail(validator=validator, action=LogAction(), stage=_POST)(
            _FakeChatModel(reply=_tool_call_reply(_HELD_ID))
        )

        llm.invoke([HumanMessage(f"Review {_block(_HELD)}")])

        evaluate.assert_not_called()

    @pytest.mark.parametrize(
        ("detection_scope", "forwards"),
        [(DetectionScope.TEXT, False), (DetectionScope.BOTH, True), (None, False)],
        ids=["text", "both", "default-is-text-only"],
    )
    def test_the_decorator_evaluator_says_whether_it_forwards_files(
        self, detection_scope: DetectionScope | None, forwards: bool
    ) -> None:
        wrap = LangChainGuardrailAdapter.wrap
        with patch.object(
            LangChainGuardrailAdapter, "wrap", autospec=True, side_effect=wrap
        ) as spy:
            _pii(_PRE, detection_scope)(_analyze_files_tool())

        evaluator = spy.call_args.args[2]
        assert evaluator.forwards_attachments is forwards

    def test_an_evaluator_that_does_not_say_still_gets_the_files(self) -> None:
        received: list[tuple[Any, dict[str, Any]]] = []

        def evaluator(data: Any, *args: Any, **kwargs: Any) -> Any:
            received.append((data, kwargs))
            return _PASSED

        _apply_llm_pre(
            [HumanMessage(content=[_file_block()])], evaluator, LogAction(), "PII"
        )

        assert received == [(NO_TEXT_PLACEHOLDER, {"attachments": [_reference()]})]


# ---------------------------------------------------------------------------
# Evaluators written before files were supported
# ---------------------------------------------------------------------------


class _PositionalEvaluator:
    """An evaluator with the four positional parameters and no ``attachments``."""

    def __init__(self) -> None:
        self.data: list[Any] = []

    def __call__(
        self,
        data: str | dict[str, Any],
        stage: GuardrailExecutionStage,
        input_data: dict[str, Any] | None,
        output_data: dict[str, Any] | None,
    ) -> GuardrailValidationResult:
        self.data.append(data)
        return _PASSED


class TestPositionalEvaluator:
    def test_tool_pre_and_post(self) -> None:
        @tool
        def echo(text: str) -> str:
            """Echo the text back."""
            return f"echo: {text}"

        evaluator = _PositionalEvaluator()
        guarded = _wrap_tool_with_guardrail(
            echo,
            evaluator,  # type: ignore[arg-type]
            LogAction(),
            "PII",
            GuardrailExecutionStage.PRE_AND_POST,
        )

        guarded.invoke({"text": "hi"})

        assert evaluator.data == [{"text": "hi"}, {"output": "echo: hi"}]

    def test_llm_pre(self) -> None:
        evaluator = _PositionalEvaluator()

        _apply_llm_pre(
            [HumanMessage("hi")],
            evaluator,  # type: ignore[arg-type]
            LogAction(),
            "PII",
        )

        assert evaluator.data == ["hi"]

    def test_llm_post(self) -> None:
        evaluator = _PositionalEvaluator()

        _apply_llm_post(
            AIMessage(content="answer"),
            evaluator,  # type: ignore[arg-type]
            LogAction(),
            "PII",
            input_messages=[HumanMessage("hi")],
        )

        assert evaluator.data == ["answer"]

    def test_agent_input(self) -> None:
        evaluator = _PositionalEvaluator()

        _apply_agent_input_guardrail(
            {"messages": [HumanMessage("hi")]},
            evaluator,  # type: ignore[arg-type]
            LogAction(),
            "PII",
        )

        assert evaluator.data == ["hi"]

    def test_agent_output(self) -> None:
        evaluator = _PositionalEvaluator()

        _apply_agent_output_guardrail(
            {"messages": [HumanMessage("hi"), AIMessage(content="answer")]},
            evaluator,  # type: ignore[arg-type]
            LogAction(),
            "PII",
        )

        assert evaluator.data == ["answer"]
