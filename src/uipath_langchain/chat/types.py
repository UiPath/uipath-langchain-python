from enum import StrEnum
from typing import Protocol, runtime_checkable


class LLMProvider(StrEnum):
    """LLM provider/vendor identifier."""

    OPENAI = "OpenAi"
    BEDROCK = "AwsBedrock"
    VERTEX = "VertexAi"


class APIFlavor(StrEnum):
    """API flavor for LLM communication."""

    OPENAI_RESPONSES = "OpenAIResponses"
    OPENAI_COMPLETIONS = "OpenAiChatCompletions"
    AWS_BEDROCK_CONVERSE = "AwsBedrockConverse"
    AWS_BEDROCK_INVOKE = "AwsBedrockInvoke"
    VERTEX_GEMINI_GENERATE_CONTENT = "GeminiGenerateContent"
    VERTEX_ANTHROPIC_CLAUDE = "AnthropicClaude"


@runtime_checkable
class UiPathPassthroughChatModel(Protocol):
    """Protocol for UiPath chat models with provider and flavor information.

    All UiPath chat model classes (UiPathChatOpenAI, UiPathChatBedrock,
    UiPathChatBedrockConverse, UiPathChatVertex, UiPathChat, UiPathAzureChatOpenAI)
    implement this protocol.
    """

    @property
    def llm_provider(self) -> LLMProvider:
        """The LLM provider for this model."""
        ...

    @property
    def api_flavor(self) -> APIFlavor:
        """The API flavor for this model."""
        ...
