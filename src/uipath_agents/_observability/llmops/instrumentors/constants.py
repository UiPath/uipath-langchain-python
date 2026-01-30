"""Constants for observability module.

Guardrail scopes, stages, actions, and patterns used across handlers.
"""

import re


class GuardrailScope:
    """Scope where guardrail is applied."""

    AGENT = "agent"
    LLM = "llm"
    TOOL = "tool"


class GuardrailStage:
    """Execution stage for guardrail (before or after operation)."""

    PRE = "pre"
    POST = "post"


class GuardrailAction:
    """Action taken when guardrail validation fails."""

    SKIP = "Skip"
    LOG = "Log"
    BLOCK = "Block"
    ESCALATE = "Escalate"
    FILTER = "Filter"


# Pattern: {scope}_{stage}_execution_{guardrail_name}
GUARDRAIL_NODE_PATTERN = re.compile(r"^(agent|llm|tool)_(pre|post)_execution_(.+)$")

# Map action suffix to action name (fallback for nodes without metadata)
ACTION_SUFFIX_TO_NAME = {
    "_log": GuardrailAction.LOG,
    "_block": GuardrailAction.BLOCK,
    "_hitl": GuardrailAction.ESCALATE,
    "_filter": GuardrailAction.FILTER,
}

# Map action_type metadata string to GuardrailAction
ACTION_TYPE_TO_ACTION = {
    "Log": GuardrailAction.LOG,
    "Block": GuardrailAction.BLOCK,
    "Escalate": GuardrailAction.ESCALATE,
    "Filter": GuardrailAction.FILTER,
}
