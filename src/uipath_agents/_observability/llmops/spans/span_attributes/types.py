"""Span type constants for agent execution tracing."""

import os
from enum import IntEnum
from functools import cache
from typing import Optional

# --- Environment Variable Names ---
ENV_UIPATH_IS_DEBUG = "UIPATH_IS_DEBUG"
ENV_UIPATH_PROCESS_VERSION = "UIPATH_PROCESS_VERSION"


class ExecutionType(IntEnum):
    """Execution context for agent tracing.

    Debug (0): Studio debug, playground, local development
    Runtime (1): Production job environment
    """

    DEBUG = 0
    RUNTIME = 1


@cache
def get_execution_type() -> int:
    """Get execution type from environment.

    Cached for process lifetime to avoid repeated environment variable reads.

    Returns:
        ExecutionType.DEBUG (0) if UIPATH_IS_DEBUG=true
        ExecutionType.RUNTIME (1) if UIPATH_IS_DEBUG=false or not set
    """
    env_value = os.getenv(ENV_UIPATH_IS_DEBUG, "").lower()
    if env_value == "true":
        return ExecutionType.DEBUG
    return ExecutionType.RUNTIME


@cache
def get_agent_version() -> Optional[str]:
    """Get agent version from environment.

    Cached for process lifetime to avoid repeated environment variable reads.
    """
    return os.getenv(ENV_UIPATH_PROCESS_VERSION) or None


class SpanType:
    """Standard span type identifiers for agent execution traces."""

    # Core types
    AGENT_RUN = "agentRun"
    COMPLETION = "completion"
    LLM_CALL = "llmCall"
    TOOL_CALL = "toolCall"
    TOOL_EXECUTION = "toolExecution"
    AGENT_OUTPUT = "agentOutput"
    AGENT_INPUT = "agentInput"
    ACTION_CENTER_TOOL = "actionCenterTool"

    # Tool types
    PROCESS_TOOL = "processTool"
    AGENT_TOOL = "agentTool"
    API_WORKFLOW_TOOL = "apiWorkflowTool"
    AGENTIC_PROCESS_TOOL = "agenticProcessTool"
    INTEGRATION_TOOL = "integrationTool"
    CONTEXT_GROUNDING_TOOL = "contextGroundingTool"
    ESCALATION_TOOL = "escalationTool"
    MCP_TOOL = "mcpTool"
    MOCK_TOOL = "mockTool"
    IXP_TOOL = "ixpTool"
    INTERNAL_TOOL = "internalTool"

    # Guardrail types
    TOOL_PRE_GUARDRAILS = "toolPreGuardrails"
    TOOL_POST_GUARDRAILS = "toolPostGuardrails"
    LLM_PRE_GUARDRAILS = "llmPreGuardrails"
    LLM_POST_GUARDRAILS = "llmPostGuardrails"
    AGENT_PRE_GUARDRAILS = "agentPreGuardrails"
    AGENT_POST_GUARDRAILS = "agentPostGuardrails"
    GUARDRAIL_EVALUATION = "guardrailEvaluation"
    GUARDRAIL_ESCALATION = "guardrailEscalation"
    TOOL_GUARDRAIL_EVALUATION = "toolGuardrailEvaluation"
    TOOL_GUARDRAIL_ESCALATION = "toolGuardrailEscalation"

    # Governance types
    TOOL_PRE_GOVERNANCE = "toolPreGovernance"
    TOOL_POST_GOVERNANCE = "toolPostGovernance"
    PRE_GOVERNANCE = "preGovernance"
    POST_GOVERNANCE = "postGovernance"
    GOVERNANCE_ESCALATION = "governanceEscalation"

    # MCP types
    MCP_SESSION_START = "mcpSessionStart"
    MCP_SESSION_STOP = "mcpSessionStop"

    # Eval types
    EVAL_SET_RUN = "evalSetRun"
    EVAL = "eval"
    EVAL_ASSERTIONS = "evalAssertions"
    EVAL_OUTPUT = "evalOutput"

    # Other types
    PLANNING = "planning"
    VALIDATION = "validation"
    OUTPUT_CORRECTION = "outputCorrection"
    SIMULATED_TOOL = "simulatedTool"
    SIMULATED_INPUT = "simulatedInput"
    AGENT_MEMORY_LOOKUP = "agentMemoryLookup"
    APPLY_DYNAMIC_FEW_SHOT = "applyDynamicFewShot"
