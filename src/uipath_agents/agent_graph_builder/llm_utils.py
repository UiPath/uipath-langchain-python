from enum import StrEnum

from langchain_core.language_models import BaseChatModel

from uipath_agents.agent_graph_builder.config import AgentExecutionType


class LLMProvider(StrEnum):
    OPENAI = "openai"
    BEDROCK = "bedrock"
    VERTEX = "vertex"


def detect_provider(model_name: str) -> LLMProvider:
    """Detect LLM provider from model name.

    Model name conventions:
    - OpenAI: contains "gpt" (e.g., gpt-4o-2024-08-06, gpt-5-mini-2025-08-07)
    - Bedrock/Claude: contains "anthropic" or "claude" (e.g., anthropic.claude-haiku-4-5)
    - Vertex/Gemini: contains "gemini" (e.g., gemini-2.5-flash)
    """
    model_lower = model_name.lower()

    if "gpt" in model_lower:
        return LLMProvider.OPENAI
    elif "anthropic" in model_lower or "claude" in model_lower:
        return LLMProvider.BEDROCK
    elif "gemini" in model_lower:
        return LLMProvider.VERTEX

    raise ValueError(
        f"Unknown model provider for: {model_name}. "
        "Model name must contain 'gpt', 'anthropic', 'claude', or 'gemini'."
    )


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


def _create_openai_llm(
    model: str,
    temperature: float,
    max_tokens: int,
    agenthub_config: str,
    byo_connection_id: str | None = None,
) -> BaseChatModel:
    """Create UiPathChatOpenAI for OpenAI models via passthrough."""
    from uipath_langchain.chat.openai import UiPathChatOpenAI

    azure_open_ai_latest_api_version = "2025-04-01-preview"

    return UiPathChatOpenAI(
        model_name=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_version=azure_open_ai_latest_api_version,
        use_responses_api=True,
        agenthub_config=agenthub_config,
        byo_connection_id=byo_connection_id,
    )


def _create_bedrock_llm(
    model: str,
    temperature: float,
    max_tokens: int,
    agenthub_config: str,
    byo_connection_id: str | None = None,
) -> BaseChatModel:
    """Create UiPathChatBedrockConverse for Claude models via passthrough."""
    from uipath_langchain.chat.bedrock import UiPathChatBedrockConverse

    return UiPathChatBedrockConverse(
        model_name=model,
        temperature=temperature,
        max_tokens=max_tokens,
        agenthub_config=agenthub_config,
        byo_connection_id=byo_connection_id,
    )


def _create_vertex_llm(
    model: str,
    temperature: float,
    max_tokens: int | None,
    agenthub_config: str,
    byo_connection_id: str | None = None,
) -> BaseChatModel:
    """Create UiPathChatVertex for Gemini models via passthrough."""
    from uipath_langchain.chat.vertex import UiPathChatVertex

    return UiPathChatVertex(
        model_name=model,
        temperature=temperature,
        max_tokens=max_tokens,
        agenthub_config=agenthub_config,
        byo_connection_id=byo_connection_id,
    )


def create_llm(
    model: str,
    temperature: float,
    max_tokens: int,
    execution_type: AgentExecutionType,
    byo_connection_id: str | None = None,
) -> BaseChatModel:
    """Create and configure LLM instance using passthrough API.

    Automatically selects the appropriate passthrough API based on model name:
    - "gpt" -> UiPathChatOpenAI
    - "anthropic" or "claude" -> UiPathChatBedrockConverse
    - "gemini" -> UiPathChatVertex
    """
    provider = detect_provider(model)
    agenthub_config = _get_agenthub_config(execution_type)

    match provider:
        case LLMProvider.OPENAI:
            return _create_openai_llm(
                model, temperature, max_tokens, agenthub_config, byo_connection_id
            )
        case LLMProvider.BEDROCK:
            return _create_bedrock_llm(
                model, temperature, max_tokens, agenthub_config, byo_connection_id
            )
        case LLMProvider.VERTEX:
            return _create_vertex_llm(
                model, temperature, max_tokens, agenthub_config, byo_connection_id
            )
