"""Bedrock-backed chat classes.

Ports the legacy ``BedrockModels.anthropic_claude_haiku_4_5`` ``model_name``
default to the upstream ``uipath_langchain_client`` classes so
``UiPathChatBedrock()`` (no args) works.

``UiPathChatBedrock`` and ``UiPathChatAnthropicBedrock`` take the default via
a pydantic ``default_factory`` on the upstream ``model_name`` field — their
upstream before-validators tolerate a missing ``model_id``.

``UiPathChatBedrockConverse`` is the odd one out: its upstream
``ChatBedrockConverse.set_disable_streaming`` before-validator reads ``model``
/ ``model_id`` from the raw input dict and fires BEFORE pydantic applies the
field ``default_factory``. We wrap it in a thin subclass with our own
before-validator that injects the default into the raw input when the caller
passed no model kwarg.
"""

from typing import Any

from pydantic import model_validator
from uipath_langchain_client.clients.bedrock.chat_models import (
    UiPathChatAnthropicBedrock,
    UiPathChatBedrock,
)
from uipath_langchain_client.clients.bedrock.chat_models import (
    UiPathChatBedrockConverse as _UpstreamUiPathChatBedrockConverse,
)

DEFAULT_MODEL_NAME = "anthropic.claude-haiku-4-5-20251001-v1:0"


def _default_factory() -> str:
    return DEFAULT_MODEL_NAME


for _cls in (UiPathChatBedrock, UiPathChatAnthropicBedrock):
    _cls.model_fields["model_name"].default_factory = _default_factory
    _cls.model_rebuild(force=True)


class UiPathChatBedrockConverse(_UpstreamUiPathChatBedrockConverse):
    """Subclass that injects the Bedrock default when no model is passed."""

    @model_validator(mode="before")
    @classmethod
    def _inject_default_model(cls, values: Any) -> Any:
        if isinstance(values, dict) and not any(
            k in values for k in ("model", "model_id", "model_name")
        ):
            values = {**values, "model": _default_factory()}
        return values


__all__ = [
    "UiPathChatBedrock",
    "UiPathChatBedrockConverse",
    "UiPathChatAnthropicBedrock",
]
