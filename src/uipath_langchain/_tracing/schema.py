"""Span types and names matching Temporal implementation schema.

This module defines the span types and naming conventions that match
the Temporal implementation backend.
"""

from enum import Enum


class SpanType(str, Enum):
    """Span types matching Temporal implementation.

    These string values must match exactly for trace compatibility.
    """

    AGENT_RUN = "agentRun"
    COMPLETION = "completion"  # Outer "LLM call" span
    LLM_CALL = "llmCall"  # Inner "Model run" span (actual API call)
    AGENT_OUTPUT = "agentOutput"

    # Tool call types
    TOOL_CALL = "toolCall"
    PROCESS_TOOL = "processTool"
    INTEGRATION_TOOL = "integrationTool"
    CONTEXT_GROUNDING_TOOL = "contextGroundingTool"
    MCP_TOOL = "mcpTool"

    # Guardrail types - scope + stage combinations
    TOOL_PRE_GUARDRAILS = "toolPreGuardrails"
    TOOL_POST_GUARDRAILS = "toolPostGuardrails"
    LLM_PRE_GUARDRAILS = "llmPreGuardrails"
    LLM_POST_GUARDRAILS = "llmPostGuardrails"
    AGENT_PRE_GUARDRAILS = "agentPreGuardrails"
    AGENT_POST_GUARDRAILS = "agentPostGuardrails"
    # Individual guardrail evaluation
    GUARDRAIL_EVALUATION = "guardrailEvaluation"
    TOOL_GUARDRAIL_EVALUATION = "toolGuardrailEvaluation"


class SpanName:
    """Span names matching Temporal implementation.

    These display names are what users see in trace viewers.
    """

    @staticmethod
    def agent_run(agent_name: str, is_conversational: bool = False) -> str:
        """Generate agent run span name.

        Args:
            agent_name: Name of the agent
            is_conversational: Whether this is a conversational agent

        Returns:
            Span name like "Agent run - MyAgent" or "Conversational agent run - MyAgent"
        """
        prefix = "Conversational agent run" if is_conversational else "Agent run"
        return f"{prefix} - {agent_name}"

    LLM_CALL = "LLM call"  # Outer completion span name
    MODEL_RUN = "Model run"  # Inner llmCall span name (actual API call)
    AGENT_OUTPUT = "Agent output"

    @staticmethod
    def tool_call(tool_name: str) -> str:
        """Generate tool call span name.

        Args:
            tool_name: Name of the tool being called

        Returns:
            Span name like "Tool call - MyTool"
        """
        return f"Tool call - {tool_name}"

    # Guardrail span names
    TOOL_PRE_GUARDRAILS = "Tool input guardrail check"
    TOOL_POST_GUARDRAILS = "Tool output guardrail check"
    LLM_PRE_GUARDRAILS = "LLM input guardrail check"
    LLM_POST_GUARDRAILS = "LLM output guardrail check"
    AGENT_PRE_GUARDRAILS = "Agent input guardrail check"
    AGENT_POST_GUARDRAILS = "Agent output guardrail check"

    @staticmethod
    def guardrail_evaluation(guardrail_name: str) -> str:
        """Generate guardrail evaluation span name.

        Args:
            guardrail_name: Name of the guardrail being evaluated

        Returns:
            The guardrail name
        """
        return guardrail_name
