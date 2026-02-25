from enum import StrEnum


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
