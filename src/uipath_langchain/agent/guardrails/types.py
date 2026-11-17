from dataclasses import dataclass
from enum import Enum

from uipath.platform.guardrails import GuardrailScope


class ExecutionStage(str, Enum):
    """Execution stage enumeration."""

    PRE_EXECUTION = "preExecution"
    POST_EXECUTION = "postExecution"


@dataclass(frozen=True)
class GuardrailEnforcementContext:
    """What an agent's guardrail check guards, when, and what the agent does on a violation."""

    scope: GuardrailScope
    execution_stage: ExecutionStage
    action_type: str
