"""Configuration for agent graph builder."""

import logging

from pydantic import BaseModel, Field, ValidationError

from .._config import get_flags

logger = logging.getLogger(__name__)

_FF_MODEL_SETTINGS = "ModelConfigurableSettings"


class _ModelConfig(BaseModel):
    """Configuration settings for a specific model from feature flags."""

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
        model_configs_raw = flags.get(_FF_MODEL_SETTINGS)

        if not model_configs_raw:
            return 0

        model_configs = [_ModelConfig(**config) for config in model_configs_raw]

        for entry in model_configs:
            if entry.name == model_name:
                return entry.allowedOptionalTools

        return 0
    except (ValidationError, TypeError, ValueError) as e:
        logger.warning(f"Invalid model config structure: {e}")
        return 0
    except Exception as e:
        logger.warning(f"Failed to fetch feature flags: {e}")
        return 0
