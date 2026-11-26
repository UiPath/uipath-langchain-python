"""Run a guardrail middleware's LLM hook around a fake model call."""

import asyncio
from collections.abc import Iterable, Mapping
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage
from uipath.core.guardrails import GuardrailScope

from uipath_langchain.guardrails.middlewares import _base

WRAP_MODEL_CALL = "_wrap_model_call"


def model_hook(middleware: Iterable[Any]) -> Any:
    """The middleware's single ``wrap_model_call`` hook."""
    [hook] = [h for h in middleware if h.name.endswith(WRAP_MODEL_CALL)]
    return hook


def model_request(
    messages: list[Any] | None = None, state: Mapping[str, Any] | None = None
) -> Any:
    """The parts of a ``ModelRequest`` the LLM hook reads."""
    messages = messages if messages is not None else [HumanMessage("input")]
    return SimpleNamespace(
        messages=messages, state=state if state is not None else {"messages": messages}
    )


async def answer(request: Any) -> AIMessage:
    """A model handler that always answers ``output``."""
    return AIMessage("output")


async def call_model_hook(
    middleware: Iterable[Any],
    state: Mapping[str, Any],
    reply: AIMessage | None = None,
) -> Any:
    """Run the LLM hook on ``state``'s messages; the model answers ``reply``."""

    async def handler(request: Any) -> AIMessage:
        return reply if reply is not None else AIMessage("output")

    request = model_request(list(state["messages"]), state)
    return await model_hook(middleware).awrap_model_call(request, handler)


def llm_stages(middleware: Iterable[Any]) -> list[str]:
    """The stages (``PRE`` / ``POST``) the LLM hook checks, in order."""
    stages: list[str] = []

    def record(
        self: Any, messages: Any, scope: Any = None, stage: Any = None, **_: Any
    ):
        assert scope == GuardrailScope.LLM
        stages.append(stage.name)

    with patch.object(_base.BuiltInGuardrailMiddlewareMixin, "_check_messages", record):
        asyncio.run(model_hook(middleware).awrap_model_call(model_request(), answer))
    return stages
