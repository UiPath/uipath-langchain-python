"""LLM-scope checks run inside the model node, around the model call."""

from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from uipath.core.guardrails import (
    GuardrailScope,
    GuardrailValidationResult,
    GuardrailValidationResultType,
)
from uipath.platform.guardrails.decorators import LogAction

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
