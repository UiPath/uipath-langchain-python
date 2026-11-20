"""Guardrail span types and display names."""


class SpanType:
    """Values of a guardrail span's ``span_type``."""

    GUARDRAIL_ACTION = "guardrailAction"
    GUARDRAIL_EVALUATION = "guardrailEvaluation"

    AGENT_PRE_GUARDRAILS = "agentPreGuardrails"
    AGENT_POST_GUARDRAILS = "agentPostGuardrails"
    LLM_PRE_GUARDRAILS = "llmPreGuardrails"
    LLM_POST_GUARDRAILS = "llmPostGuardrails"
    TOOL_PRE_GUARDRAILS = "toolPreGuardrails"
    TOOL_POST_GUARDRAILS = "toolPostGuardrails"


class SpanName:
    """Display names of the stage containers."""

    AGENT_PRE_GUARDRAILS = "Agent input guardrail check"
    AGENT_POST_GUARDRAILS = "Agent output guardrail check"
    LLM_PRE_GUARDRAILS = "LLM input guardrail check"
    LLM_POST_GUARDRAILS = "LLM output guardrail check"
    TOOL_PRE_GUARDRAILS = "Tool input guardrail check"
    TOOL_POST_GUARDRAILS = "Tool output guardrail check"
