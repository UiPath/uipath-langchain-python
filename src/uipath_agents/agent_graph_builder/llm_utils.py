from langchain_core.language_models import BaseChatModel
from uipath_langchain.chat.chat_model_factory import get_chat_model

from uipath_agents.agent_graph_builder.config import AgentExecutionType


def _get_agenthub_config(execution_type: AgentExecutionType) -> str:
    """Map the execution type to the AgentHub config."""
    match execution_type:
        case AgentExecutionType.PLAYGROUND:
            return "agentsplayground"
        case AgentExecutionType.RUNTIME:
            return "agentsruntime"
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

    Returns:
        A configured BaseChatModel instance for the specified provider.
    """
    agenthub_config = _get_agenthub_config(execution_type)

    return get_chat_model(
        model,
        temperature,
        max_tokens,
        agenthub_config,
        byo_connection_id,
        disable_streaming=disable_streaming,
    )
