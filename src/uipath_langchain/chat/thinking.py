"""Anthropic thinking/reasoning knowledge shared across transports.

Where the thinking config lives depends on the transport: native ChatAnthropic exposes a
`thinking` attribute, Bedrock Invoke nests it under `model_kwargs`, Bedrock Converse under
`additional_model_request_fields`. These helpers hide that so the payload handlers and the
ReAct loop don't each reimplement it.
"""

from typing import Any

from langchain_core.language_models import BaseChatModel

_REASONING_BLOCK_TYPES = frozenset(
    {"reasoning_content", "reasoning", "thinking", "redacted_thinking"}
)


def is_reasoning_block(block: Any) -> bool:
    """True if a message content block is provider reasoning (thinking) output."""
    return isinstance(block, dict) and block.get("type") in _REASONING_BLOCK_TYPES


def thinking_rejects_forced_tool_choice(model: Any) -> bool:
    """True if forcing a tool call is incompatible with this model's active thinking.

    Anthropic models (native + Bedrock) reject a forced tool_choice while extended or
    adaptive thinking is on, so callers downgrade forcing to 'auto' and rely on the
    thinking-off extraction retry (agent/react/forced_extraction.py). OpenAI/Gemini
    reasoning tolerate forcing, so they return False. `{"type": "disabled"}` is thinking
    off. Extend here if another provider shows the same conflict.
    """
    for thinking in _thinking_configs(model):
        if isinstance(thinking.get("type"), str):
            return thinking["type"] != "disabled"
    return False


def strip_thinking(model: BaseChatModel) -> BaseChatModel:
    """Copy of the model with thinking config stripped, so forcing is honored.

    Adaptive thinking also carries an `output_config` effort knob on Converse; drop it too.
    """
    updates: dict[str, object] = {}
    request_fields = getattr(model, "additional_model_request_fields", None)
    if isinstance(request_fields, dict) and (
        "thinking" in request_fields or "output_config" in request_fields
    ):
        updates["additional_model_request_fields"] = {
            k: v
            for k, v in request_fields.items()
            if k not in ("thinking", "output_config")
        }
    model_kwargs = getattr(model, "model_kwargs", None)
    if isinstance(model_kwargs, dict) and "thinking" in model_kwargs:
        updates["model_kwargs"] = {
            k: v for k, v in model_kwargs.items() if k != "thinking"
        }
    if getattr(model, "thinking", None) is not None:
        updates["thinking"] = None
    if not updates:
        return model
    try:
        return model.model_copy(update=updates)
    except Exception:
        return model


def _thinking_configs(model: Any) -> list[dict[str, Any]]:
    """The thinking dicts set on a model, read from every transport's location."""
    invoke = getattr(model, "model_kwargs", None) or {}
    converse = getattr(model, "additional_model_request_fields", None) or {}
    candidates = (
        getattr(model, "thinking", None),
        invoke.get("thinking") if isinstance(invoke, dict) else None,
        converse.get("thinking") if isinstance(converse, dict) else None,
    )
    return [c for c in candidates if isinstance(c, dict)]
