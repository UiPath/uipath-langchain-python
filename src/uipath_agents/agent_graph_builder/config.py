"""Configuration for agent graph builder."""

import json
import logging
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field
from uipath.agent.models.agent import LowCodeAgentDefinition
from uipath.runtime import UiPathRuntimeContext

from .._config import get_flags

logger = logging.getLogger(__name__)

_FF_MODEL_SETTINGS = "ModelConfigurableSettings"
_DEFAULT_THINKING_MESSAGES_LIMIT = 0


def is_deep_agent_enabled(agent_definition: LowCodeAgentDefinition) -> bool:
    """Check if deep agent mode is enabled via settings.deepAgent.enabled."""
    deep_agent_setting = getattr(agent_definition.settings, "deepAgent", None)
    return bool(deep_agent_setting and deep_agent_setting.get("enabled", False))


class AgentExecutionType(StrEnum):
    """The type of execution for an agent run."""

    PLAYGROUND = "playground"
    RUNTIME = "runtime"
    EVAL = "eval"
    UNKNOWN = "unknown"


def get_execution_type(context: UiPathRuntimeContext) -> AgentExecutionType:
    """Get execution type from runtime context command.

    Args:
        context: Runtime context containing the command

    Returns:
        AgentExecutionType corresponding to the command
    """
    match context.command:
        case "run":
            return AgentExecutionType.RUNTIME
        case "debug":
            return AgentExecutionType.PLAYGROUND
        case "dev":
            return AgentExecutionType.PLAYGROUND
        case "eval":
            return AgentExecutionType.EVAL
        case _:
            return AgentExecutionType.UNKNOWN


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
