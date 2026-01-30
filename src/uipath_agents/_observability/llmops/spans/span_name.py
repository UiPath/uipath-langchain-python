"""Span type identifiers and human-readable span names for tracing."""

__all__ = [
    "SpanName",
    "INNER_STATE_KEY",
    "GUARDRAIL_VALIDATION_RESULT_KEY",
    "GUARDRAIL_VALIDATION_DETAILS_KEY",
]

INNER_STATE_KEY = "inner_state"
GUARDRAIL_VALIDATION_RESULT_KEY = "guardrail_validation_result"
GUARDRAIL_VALIDATION_DETAILS_KEY = "guardrail_validation_details"


class SpanName:
    """Human-readable span names for display."""

    LLM_CALL = "LLM call"
    MODEL_RUN = "Model run"
    AGENT_OUTPUT = "Agent output"
    REVIEW_TASK = "Review task"

    # Guardrail container span names
    AGENT_PRE_GUARDRAILS = "Agent input guardrail check"
    AGENT_POST_GUARDRAILS = "Agent output guardrail check"
    LLM_PRE_GUARDRAILS = "LLM input guardrail check"
    LLM_POST_GUARDRAILS = "LLM output guardrail check"
    TOOL_PRE_GUARDRAILS = "Tool input guardrail check"
    TOOL_POST_GUARDRAILS = "Tool output guardrail check"

    @staticmethod
    def agent_run(agent_name: str, is_conversational: bool = False) -> str:
        if is_conversational:
            return f"Conversational agent run - {agent_name}"
        return f"Agent run - {agent_name}"

    @staticmethod
    def tool_call(tool_name: str) -> str:
        return f"Tool call - {tool_name}"

    @staticmethod
    def guardrail(guardrail_name: str) -> str:
        return f"Guardrail - {guardrail_name}"

    @staticmethod
    def guardrails_container(scope: str, stage: str) -> str:
        """Get container span name for a guardrails phase.

        Args:
            scope: "agent", "llm", or "tool"
            stage: "pre" or "post"

        Returns:
            Human-readable container span name
        """
        scope_map = {"agent": "Agent", "llm": "LLM", "tool": "Tool"}
        return f"{scope_map.get(scope, scope.capitalize())} {stage} guardrails"
