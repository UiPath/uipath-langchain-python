from langchain_core.language_models import BaseChatModel
from uipath.platform.common import UiPathConfig
from uipath_langchain.chat.chat_model_factory import get_chat_model

from uipath_agents.agent_graph_builder.config import AgentExecutionType


def _get_agenthub_config(
    execution_type: AgentExecutionType, *, is_conversational: bool = False
) -> str:
    """Map the execution type to the AgentHub config.

    When isDebug is set in internalArguments (Maestro debug calls an Agent),
    return the playground variant for licensing purposes. The agent is
    deployed to a solution debug folder (command="run") but the parent
    Orchestrator job is a debug session — use the developer's LLM call debug
    quota instead of requiring consumables (such as Agent Units).
    """
    if UiPathConfig.is_rooted_to_debug_job:
        return (
            "conversationalagentsplayground"
            if is_conversational
            else "agentsplayground"
        )

    match execution_type:
        case AgentExecutionType.PLAYGROUND:
            return (
                "conversationalagentsplayground"
                if is_conversational
                else "agentsplayground"
            )
        case AgentExecutionType.RUNTIME:
            return (
                "conversationalagentsruntime" if is_conversational else "agentsruntime"
            )
        case AgentExecutionType.EVAL:
            return "agentsevals"
        case AgentExecutionType.UNKNOWN:
            return "unknown"


def create_llm(
    model: str,
    temperature: float,
    max_tokens: int,
    execution_type: AgentExecutionType,
    byo_connection_id: str | None = None,
    disable_streaming: bool = True,
    is_conversational: bool = False,
) -> BaseChatModel:
    """Create an LLM instance via the UiPath LLM Gateway passthrough API.

    Args:
        model: The model name (e.g., "gpt-4o", "claude-3-sonnet").
        temperature: Sampling temperature for response generation.
        max_tokens: Maximum number of tokens in the response.
        execution_type: The agent execution context (playground, runtime, or eval).
        byo_connection_id: Optional Integration Service connection ID for
            bring-your-own-model configurations.
        disable_streaming: Whether to disable streaming responses. Defaults to
            True because the UiPath LLM Gateway does not support streaming
            when PII masking is enabled.
        is_conversational: Whether this is a conversational agent. Controls
            the AgentHub config header to skip per-LLM-call billing.

    Returns:
        A configured BaseChatModel instance for the specified provider.
    """
    agenthub_config = _get_agenthub_config(
        execution_type, is_conversational=is_conversational
    )

    return get_chat_model(
        model,
        temperature,
        max_tokens,
        agenthub_config,
        byo_connection_id,
        disable_streaming=disable_streaming,
    )
