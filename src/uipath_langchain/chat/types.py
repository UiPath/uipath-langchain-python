from enum import StrEnum


class LLMProvider(StrEnum):
    """LLM provider/vendor identifier."""

    OPENAI = "openai"
    BEDROCK = "awsbedrock"
    VERTEX = "vertexai"


class APIFlavor(StrEnum):
    """API flavor for LLM communication."""

    OPENAI_RESPONSES = "responses"
    OPENAI_COMPLETIONS = "chat-completions"
    AWS_BEDROCK_CONVERSE = "converse"
    AWS_BEDROCK_INVOKE = "invoke"
    VERTEX_GEMINI_GENERATE_CONTENT = "generate-content"
    VERTEX_ANTHROPIC_CLAUDE = "anthropic-claude"
