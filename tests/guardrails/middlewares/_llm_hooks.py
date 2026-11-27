"""Run a guardrail middleware's LLM hook around a fake model call."""

import asyncio
from collections.abc import Iterable, Mapping
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from langchain.agents.middleware import AgentMiddleware
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


def _has(hook: Any, method: str) -> bool:
    return getattr(type(hook), method) is not getattr(AgentMiddleware, method)


async def run_model_hook(hook: Any, state: Mapping[str, Any], handler: Any) -> Any:
    """Run the hook's nodes as the agent does: before the model, around, after."""
    state = dict(state)
    if _has(hook, "abefore_model"):
        update = await hook.abefore_model(state, None) or {}
        edited = {m.id: m for m in update.get("messages", [])}
        state["messages"] = [edited.get(m.id, m) for m in state["messages"]]
    result = await hook.awrap_model_call(
        model_request(list(state["messages"]), state), handler
    )
    if _has(hook, "aafter_model"):
        command = getattr(result, "command", None)
        update = command.update if command is not None else {}
        reply = _base._last_ai_message(getattr(result, "model_response", result))
        await hook.aafter_model(
            {**state, **update, "messages": [*state["messages"], reply]}, None
        )
    return result


async def call_model_hook(
    middleware: Iterable[Any],
    state: Mapping[str, Any],
    reply: AIMessage | None = None,
) -> Any:
    """Run the LLM hook on ``state``'s messages; the model answers ``reply``."""

    async def handler(request: Any) -> AIMessage:
        return reply if reply is not None else AIMessage("output")

    return await run_model_hook(model_hook(middleware), state, handler)


def llm_stages(middleware: Iterable[Any]) -> list[str]:
    """The stages (``PRE`` / ``POST``) the LLM hook checks, in order."""
    stages: list[str] = []

    def record(
        self: Any, messages: Any, scope: Any = None, stage: Any = None, **_: Any
    ) -> None:
        assert scope == GuardrailScope.LLM
        stages.append(stage.name)

    def record_output(self: Any, *args: Any, **kwargs: Any) -> None:
        stages.append("POST")

    with (
        patch.object(_base.BuiltInGuardrailMiddlewareMixin, "_check_messages", record),
        patch.object(
            _base.BuiltInGuardrailMiddlewareMixin, "_failed_output_check", record_output
        ),
    ):
        asyncio.run(
            run_model_hook(
                model_hook(middleware), {"messages": [HumanMessage("input")]}, answer
            )
        )
    return stages
