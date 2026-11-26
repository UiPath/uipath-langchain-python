"""LLM-scope checks run inside the model node, around the model call."""

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentState,
    ExtendedModelResponse,
    wrap_model_call,
)
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.types import Command, interrupt
from uipath.core.guardrails import (
    GuardrailScope,
    GuardrailValidationResult,
    GuardrailValidationResultType,
)
from uipath.platform.guardrails.decorators import LogAction

from uipath_langchain.agent.exceptions import AgentRuntimeError
from uipath_langchain.guardrails import EscalateAction, GuardrailExecutionStage
from uipath_langchain.guardrails import escalate_action as escalate_module
from uipath_langchain.guardrails.enums import PIIDetectionEntityType
from uipath_langchain.guardrails.middlewares import UiPathPIIDetectionMiddleware, _base
from uipath_langchain.guardrails.models import GuardrailAction, PIIDetectionEntity

_FAILED = GuardrailValidationResult(
    result=GuardrailValidationResultType.VALIDATION_FAILED, reason="found"
)
_PASSED = GuardrailValidationResult(
    result=GuardrailValidationResultType.PASSED, reason=""
)


class _CountingModel(GenericFakeChatModel):
    """Answers ``answer 1``, ``answer 2``, ... and counts its calls."""

    calls: int = 0

    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        # The async path runs this too, so each model call counts once.
        self.calls += 1
        return super()._generate(*args, **kwargs)


def _model() -> _CountingModel:
    return _CountingModel(messages=iter(AIMessage(f"answer {i}") for i in range(1, 10)))


def _pii(stage: GuardrailExecutionStage, action: GuardrailAction) -> list[Any]:
    return list(
        UiPathPIIDetectionMiddleware(
            scopes=[GuardrailScope.LLM],
            action=action,
            entities=[PIIDetectionEntity(PIIDetectionEntityType.EMAIL)],
            stage=stage,
        )
    )


def _escalate() -> EscalateAction:
    return EscalateAction(app_name="app", app_folder_path="folder")


@pytest.fixture(autouse=True)
def _no_platform_config() -> Iterator[None]:
    config = SimpleNamespace(tenant_name=None, base_url=None)
    with patch.object(escalate_module, "UiPathConfig", config):
        yield


def _approve(reviewed: str, field: str) -> Command[Any]:
    return Command(resume={"action": "Approve", "data": {field: reviewed}})


def _texts(messages: list[BaseMessage]) -> list[str]:
    return [str(m.content) for m in messages]


class TestChecks:
    async def test_input_and_output_are_checked_around_one_model_call(self) -> None:
        model = _model()
        checked: list[str] = []

        def evaluate(self: Any, data: Any, *args: Any, **kwargs: Any) -> Any:
            checked.append(data)
            return _PASSED

        with patch.object(
            _base.BuiltInGuardrailMiddlewareMixin, "_evaluate_guardrail", evaluate
        ):
            agent = create_agent(
                model,
                middleware=_pii(GuardrailExecutionStage.PRE_AND_POST, LogAction()),
            )
            await agent.ainvoke({"messages": [HumanMessage("mail a@b.com")]})

        assert checked == ["mail a@b.com", "answer 1"]
        assert model.calls == 1

    async def test_an_output_escalation_resumes_on_the_same_answer(self) -> None:
        model = _model()
        agent = create_agent(
            model,
            middleware=_pii(GuardrailExecutionStage.POST, _escalate()),
            checkpointer=InMemorySaver(),
        )
        config: Any = {"configurable": {"thread_id": "post"}}

        with patch.object(
            _base.BuiltInGuardrailMiddlewareMixin,
            "_evaluate_guardrail",
            return_value=_FAILED,
        ):
            first = await agent.ainvoke(
                {"messages": [HumanMessage("mail a@b.com")]}, config
            )
            [pending] = first["__interrupt__"]
            resumed = await agent.ainvoke(
                _approve("reviewed answer", "ReviewedOutputs"), config
            )

        assert '"answer 1"' in pending.value.data["Outputs"]
        # The resumed node replays the saved answer: the model ran once and the
        # reviewer's edit lands on the answer they reviewed.
        assert model.calls == 1
        assert _texts(resumed["messages"]) == ["mail a@b.com", "reviewed answer"]

    async def test_an_input_escalation_calls_the_model_after_review(self) -> None:
        model = _model()
        agent = create_agent(
            model,
            middleware=_pii(GuardrailExecutionStage.PRE, _escalate()),
            checkpointer=InMemorySaver(),
        )
        config: Any = {"configurable": {"thread_id": "pre"}}

        with patch.object(
            _base.BuiltInGuardrailMiddlewareMixin,
            "_evaluate_guardrail",
            return_value=_FAILED,
        ):
            first = await agent.ainvoke(
                {"messages": [HumanMessage("mail a@b.com")]}, config
            )
            assert model.calls == 0
            assert first["__interrupt__"]
            resumed = await agent.ainvoke(
                _approve("mail [redacted]", "ReviewedInputs"), config
            )

        assert model.calls == 1
        assert _texts(resumed["messages"]) == ["mail [redacted]", "answer 1"]

    async def test_two_output_escalations_still_call_the_model_once(self) -> None:
        model = _model()
        agent = create_agent(
            model,
            middleware=[
                *_pii(GuardrailExecutionStage.POST, _escalate()),
                *UiPathPIIDetectionMiddleware(
                    name="Second PII",
                    scopes=[GuardrailScope.LLM],
                    action=_escalate(),
                    entities=[PIIDetectionEntity(PIIDetectionEntityType.EMAIL)],
                    stage=GuardrailExecutionStage.POST,
                ),
            ],
            checkpointer=InMemorySaver(),
        )
        config: Any = {"configurable": {"thread_id": "two"}}

        with patch.object(
            _base.BuiltInGuardrailMiddlewareMixin,
            "_evaluate_guardrail",
            return_value=_FAILED,
        ):
            await agent.ainvoke({"messages": [HumanMessage("mail a@b.com")]}, config)
            second = await agent.ainvoke(
                _approve("first review", "ReviewedOutputs"), config
            )
            assert second["__interrupt__"]
            done = await agent.ainvoke(
                _approve("second review", "ReviewedOutputs"), config
            )

        assert model.calls == 1
        assert _texts(done["messages"])[-1] == "second review"

    async def test_an_inner_input_edit_survives_an_outer_output_escalation(
        self,
    ) -> None:
        # The outer guardrail escalates on the output, so its model call is
        # checkpointed; that call also runs the inner guardrail's input check.
        model = _model()
        agent = create_agent(
            model,
            middleware=[
                *_pii(GuardrailExecutionStage.POST, _escalate()),
                *UiPathPIIDetectionMiddleware(
                    name="Input PII",
                    scopes=[GuardrailScope.LLM],
                    action=_escalate(),
                    entities=[PIIDetectionEntity(PIIDetectionEntityType.EMAIL)],
                    stage=GuardrailExecutionStage.PRE,
                ),
            ],
            checkpointer=InMemorySaver(),
        )
        config: Any = {"configurable": {"thread_id": "mixed"}}

        with patch.object(
            _base.BuiltInGuardrailMiddlewareMixin,
            "_evaluate_guardrail",
            return_value=_FAILED,
        ):
            await agent.ainvoke({"messages": [HumanMessage("mail a@b.com")]}, config)
            output_review = await agent.ainvoke(
                _approve("mail [redacted]", "ReviewedInputs"), config
            )
            assert output_review["__interrupt__"]
            done = await agent.ainvoke(
                _approve("reviewed answer", "ReviewedOutputs"), config
            )

        # The replayed call restores the input the model saw, not the original.
        assert model.calls == 1
        assert _texts(done["messages"]) == ["mail [redacted]", "reviewed answer"]


class TestHookNames:
    async def test_input_and_output_guardrails_with_one_name_build_an_agent(
        self,
    ) -> None:
        # Both use the default name; the stage keeps their hooks distinct.
        model = _model()
        checked: list[str] = []

        def evaluate(self: Any, data: Any, *args: Any, **kwargs: Any) -> Any:
            checked.append(data)
            return _PASSED

        with patch.object(
            _base.BuiltInGuardrailMiddlewareMixin, "_evaluate_guardrail", evaluate
        ):
            agent = create_agent(
                model,
                middleware=[
                    *_pii(GuardrailExecutionStage.PRE, LogAction()),
                    *_pii(GuardrailExecutionStage.POST, LogAction()),
                ],
            )
            await agent.ainvoke({"messages": [HumanMessage("mail a@b.com")]})

        assert checked == ["mail a@b.com", "answer 1"]


@wrap_model_call
async def _copy_messages(request: Any, handler: Any) -> Any:
    """Pass the model copies of the messages, as a middleware adding metadata does."""
    copies = [
        m.model_copy(update={"additional_kwargs": {"seen": True}})
        for m in request.messages
    ]
    return await handler(request.override(messages=copies))


class TestInputEdits:
    async def test_an_input_review_edits_the_saved_message_behind_a_copy(
        self,
    ) -> None:
        model = _model()
        agent = create_agent(
            model,
            middleware=[
                _copy_messages,
                *_pii(GuardrailExecutionStage.PRE, _escalate()),
            ],
            checkpointer=InMemorySaver(),
        )
        config: Any = {"configurable": {"thread_id": "copy"}}

        with patch.object(
            _base.BuiltInGuardrailMiddlewareMixin,
            "_evaluate_guardrail",
            return_value=_FAILED,
        ):
            await agent.ainvoke({"messages": [HumanMessage("meet Tuesday")]}, config)
            await agent.ainvoke(_approve("meet Wednesday", "ReviewedInputs"), config)

        saved = (await agent.aget_state(config)).values["messages"]
        assert _texts(saved) == ["meet Wednesday", "answer 1"]


class _ReviewAction(GuardrailAction):
    """A custom action that pauses the run and uses the reviewer's text."""

    def __init__(self) -> None:
        self.reviewed: list[str] = []

    def handle_validation_result(
        self, result: Any, data: Any, guardrail_name: str
    ) -> Any:
        self.reviewed.append(data)
        return interrupt({"review": data})


def _failing_output(checked: list[str]) -> Any:
    """Record what is checked; only the model's answers fail."""

    def evaluate(self: Any, data: Any, *args: Any, **kwargs: Any) -> Any:
        checked.append(data)
        return _FAILED if str(data).startswith("answer") else _PASSED

    return evaluate


class TestOutputActionResume:
    """An output check's action runs after the model node, so resuming it
    re-runs only the action."""

    async def test_a_custom_pausing_action_commits_the_answer_it_reviewed(
        self,
    ) -> None:
        model = _model()
        action = _ReviewAction()
        agent = create_agent(
            model,
            middleware=_pii(GuardrailExecutionStage.POST, action),
            checkpointer=InMemorySaver(),
        )
        config: Any = {"configurable": {"thread_id": "custom"}}

        with patch.object(
            _base.BuiltInGuardrailMiddlewareMixin,
            "_evaluate_guardrail",
            return_value=_FAILED,
        ):
            first = await agent.ainvoke({"messages": [HumanMessage("hi")]}, config)
            [pending] = first["__interrupt__"]
            done = await agent.ainvoke(Command(resume="answer 1, reviewed"), config)

        assert pending.value == {"review": "answer 1"}
        assert model.calls == 1
        assert _texts(done["messages"]) == ["hi", "answer 1, reviewed"]

    async def test_resuming_does_not_check_the_input_again(self) -> None:
        model = _model()
        checked: list[str] = []
        agent = create_agent(
            model,
            middleware=_pii(GuardrailExecutionStage.PRE_AND_POST, _escalate()),
            checkpointer=InMemorySaver(),
        )
        config: Any = {"configurable": {"thread_id": "both"}}

        with patch.object(
            _base.BuiltInGuardrailMiddlewareMixin,
            "_evaluate_guardrail",
            _failing_output(checked),
        ):
            await agent.ainvoke({"messages": [HumanMessage("hi")]}, config)
            done = await agent.ainvoke(
                _approve("reviewed answer", "ReviewedOutputs"), config
            )

        assert checked == ["hi", "answer 1"]
        assert model.calls == 1
        assert _texts(done["messages"]) == ["hi", "reviewed answer"]

    async def test_an_inner_middleware_state_update_survives_the_resume(
        self,
    ) -> None:
        @wrap_model_call
        async def add_note(request: Any, handler: Any) -> Any:
            response = await handler(request)
            return ExtendedModelResponse(
                model_response=response,
                command=Command(update={"messages": [HumanMessage("note")]}),
            )

        model = _model()
        config: Any = {"configurable": {"thread_id": "note"}}
        async with AsyncSqliteSaver.from_conn_string(":memory:") as saver:
            agent = create_agent(
                model,
                middleware=[*_pii(GuardrailExecutionStage.POST, _escalate()), add_note],
                checkpointer=saver,
            )
            with patch.object(
                _base.BuiltInGuardrailMiddlewareMixin,
                "_evaluate_guardrail",
                return_value=_FAILED,
            ):
                await agent.ainvoke({"messages": [HumanMessage("hi")]}, config)
                done = await agent.ainvoke(
                    _approve("reviewed answer", "ReviewedOutputs"), config
                )

        assert model.calls == 1
        assert _texts(done["messages"]) == ["hi", "reviewed answer", "note"]

    async def test_an_agent_nested_in_a_parent_graph_resumes_on_the_same_answer(
        self,
    ) -> None:
        model = _model()
        agent = create_agent(
            model, middleware=_pii(GuardrailExecutionStage.POST, _escalate())
        )
        parent = (
            StateGraph(AgentState)
            .add_node("agent", agent)
            .add_edge(START, "agent")
            .compile(checkpointer=InMemorySaver())
        )
        config: Any = {"configurable": {"thread_id": "nested"}}

        with patch.object(
            _base.BuiltInGuardrailMiddlewareMixin,
            "_evaluate_guardrail",
            return_value=_FAILED,
        ):
            first = await parent.ainvoke({"messages": [HumanMessage("hi")]}, config)
            assert first["__interrupt__"]
            done = await parent.ainvoke(
                _approve("reviewed answer", "ReviewedOutputs"), config
            )

        assert model.calls == 1
        assert _texts(done["messages"]) == ["hi", "reviewed answer"]


class _PausingLog(LogAction):
    """A Log subclass that pauses the run and uses the reviewer's text."""

    def handle_validation_result(
        self, result: Any, data: Any, guardrail_name: str
    ) -> Any:
        return interrupt({"review": data})


class _Capitalize(GuardrailAction):
    """A custom action that edits the answer without pausing."""

    def handle_validation_result(
        self, result: Any, data: Any, guardrail_name: str
    ) -> Any:
        return str(data).capitalize()


def _llm_pii(name: str, action: GuardrailAction) -> list[Any]:
    return list(
        UiPathPIIDetectionMiddleware(
            name=name,
            scopes=[GuardrailScope.LLM],
            action=action,
            entities=[PIIDetectionEntity(PIIDetectionEntityType.EMAIL)],
            stage=GuardrailExecutionStage.POST,
        )
    )


class TestOutputActionPlacement:
    async def test_a_pausing_log_subclass_commits_the_answer_it_reviewed(
        self,
    ) -> None:
        model = _model()
        agent = create_agent(
            model,
            middleware=_pii(GuardrailExecutionStage.POST, _PausingLog()),
            checkpointer=InMemorySaver(),
        )
        config: Any = {"configurable": {"thread_id": "log-subclass"}}

        with patch.object(
            _base.BuiltInGuardrailMiddlewareMixin,
            "_evaluate_guardrail",
            return_value=_FAILED,
        ):
            first = await agent.ainvoke({"messages": [HumanMessage("hi")]}, config)
            [pending] = first["__interrupt__"]
            done = await agent.ainvoke(Command(resume="answer 1, reviewed"), config)

        assert pending.value == {"review": "answer 1"}
        assert model.calls == 1
        assert _texts(done["messages"]) == ["hi", "answer 1, reviewed"]

    async def test_a_later_answer_does_not_reopen_a_rejected_escalation(
        self,
    ) -> None:
        model = _model()
        agent = create_agent(
            model,
            middleware=_pii(GuardrailExecutionStage.POST, _escalate()),
            checkpointer=InMemorySaver(),
        )
        config: Any = {"configurable": {"thread_id": "rejected"}}

        def evaluate(self: Any, data: Any, *args: Any, **kwargs: Any) -> Any:
            return _FAILED if data == "answer 1" else _PASSED

        with patch.object(
            _base.BuiltInGuardrailMiddlewareMixin, "_evaluate_guardrail", evaluate
        ):
            first = await agent.ainvoke({"messages": [HumanMessage("a day")]}, config)
            assert first["__interrupt__"]
            reject: Command[Any] = Command(
                resume={"action": "Reject", "data": {"Reason": "no"}}
            )
            with pytest.raises(AgentRuntimeError):
                await agent.ainvoke(reject, config)
            again = await agent.ainvoke(
                {"messages": [HumanMessage("another day")]}, config
            )

        # The new answer passed; the rejected one's action is not run again.
        assert "__interrupt__" not in again
        assert _texts(again["messages"])[-1] == "answer 2"

    async def test_an_outer_check_sees_the_answer_an_inner_action_edited(
        self,
    ) -> None:
        model = _model()
        audited: list[str] = []

        def evaluate(self: Any, data: Any, *args: Any, **kwargs: Any) -> Any:
            if self._name == "Audit":
                audited.append(data)
                return _PASSED
            return _FAILED if data == "answer 1" else _PASSED

        agent = create_agent(
            model,
            middleware=[
                *_llm_pii("Audit", LogAction()),
                *_llm_pii("Formatter", _Capitalize()),
            ],
        )
        with patch.object(
            _base.BuiltInGuardrailMiddlewareMixin, "_evaluate_guardrail", evaluate
        ):
            done = await agent.ainvoke({"messages": [HumanMessage("hi")]})

        # The audit passes, but it still runs after the formatter's edit.
        assert audited == ["Answer 1"]
        assert _texts(done["messages"]) == ["hi", "Answer 1"]
