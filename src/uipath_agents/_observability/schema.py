"""Span types and names matching UiPath Agents schema.

This schema matches the C# Agents implementation for consistent
span structure across all UiPath agent implementations.
"""

from enum import Enum

INNER_STATE_KEY = "inner_state"
GUARDRAIL_VALIDATION_RESULT_KEY = "guardrail_validation_result"


class SpanType(str, Enum):
    """Span types matching C# Agents schema."""

    AGENT_RUN = "agentRun"
    AGENT_OUTPUT = "agentOutput"

    # LLM spans (nested: LLM call -> Model run)
    COMPLETION = "completion"  # Outer LLM iteration wrapper
    LLM_CALL = "llmCall"  # Inner actual API call

    # Tool spans
    TOOL_CALL = "toolCall"  # Generic tool call
    PROCESS_TOOL = "processTool"  # UiPath Process invocation
    INTEGRATION_TOOL = "integrationTool"  # UiPath Integration connector
    CONTEXT_GROUNDING_TOOL = "contextGroundingTool"  # RAG/context tool
    MCP_TOOL = "mcpTool"  # Model Context Protocol tool

    LLM_PRE_GUARDRAILS = "llmPreGuardrails"
    LLM_POST_GUARDRAILS = "llmPostGuardrails"
    TOOL_PRE_GUARDRAILS = "toolPreGuardrails"
    TOOL_POST_GUARDRAILS = "toolPostGuardrails"
    AGENT_OUTPUT_PRE_GUARDRAILS = "agentOutputPreGuardrails"
    AGENT_OUTPUT_POST_GUARDRAILS = "agentOutputPostGuardrails"


class SpanName:
    """Human-readable span names for display."""

    LLM_CALL = "LLM call"
    MODEL_RUN = "Model run"
    AGENT_OUTPUT = "Agent output"

    # Guardrail container span names (matching C# Temporal)
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
