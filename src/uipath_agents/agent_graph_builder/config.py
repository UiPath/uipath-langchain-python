"""Configuration for agent graph builder."""

import json
import logging

from pydantic import BaseModel, ConfigDict, Field

from .._config import get_flags

logger = logging.getLogger(__name__)

_FF_MODEL_SETTINGS = "ModelConfigurableSettings"
_DEFAULT_THINKING_MESSAGES_LIMIT = 0


class _ModelConfig(BaseModel):
    """Configuration settings for a specific model from feature flags."""

    model_config = ConfigDict(extra="allow")

    name: str = Field(description="Model name")
    allowedOptionalTools: int = Field(description="Number of allowed optional tools")


def get_thinking_messages_limit(model_name: str) -> int:
    """Get thinking messages limit for a model from feature flags.

    Controls consecutive LLM responses without tool calls before forcing tool usage.

    Args:
        model_name: The model name to look up

    Returns:
        Limit value (0 = force immediate tool calling)
    """
    try:
        flags = get_flags([_FF_MODEL_SETTINGS])
        model_configs_json = flags.get(_FF_MODEL_SETTINGS)

        if not model_configs_json:
            return _DEFAULT_THINKING_MESSAGES_LIMIT

        models = [_ModelConfig(**config) for config in json.loads(model_configs_json)]

        for model in models:
            if model.name == model_name:
                return model.allowedOptionalTools

        return _DEFAULT_THINKING_MESSAGES_LIMIT
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse model config JSON: {e}")
        return _DEFAULT_THINKING_MESSAGES_LIMIT
    except Exception as e:
        logger.warning(f"Failed to fetch feature flags: {e}")
        return _DEFAULT_THINKING_MESSAGES_LIMIT
