"""UiPath Guardrails for LangChain agents.

Platform guardrail decorators plus LangChain/LangGraph adapter auto-registration.
"""

from uipath.agent.models.agent import AgentGuardrailSeverityLevel
from uipath.core.guardrails import GuardrailScope
from uipath.platform.guardrails.decorators import (
    BlockAction,
    CustomValidator,
    GuardrailAction,
    GuardrailBlockException,
    GuardrailExecutionStage,
    GuardrailTargetAdapter,
    GuardrailValidatorBase,
    LogAction,
    LoggingSeverityLevel,
    PIIDetectionEntity,
    PIIDetectionEntityType,
    PIIValidator,
    PromptInjectionValidator,
    RuleFunction,
    guardrail,
    register_guardrail_adapter,
)

from ._langchain_adapter import LangChainGuardrailAdapter
from .middlewares import (
    UiPathDeterministicGuardrailMiddleware,
    UiPathPIIDetectionMiddleware,
    UiPathPromptInjectionMiddleware,
)

# Auto-register the LangChain adapter so @guardrail knows how to wrap
# BaseTool, BaseChatModel, StateGraph, and CompiledStateGraph.
register_guardrail_adapter(LangChainGuardrailAdapter())

__all__ = [
    # Decorator
    "guardrail",
    # Validators
    "GuardrailValidatorBase",
    "PIIValidator",
    "PromptInjectionValidator",
    "CustomValidator",
    "RuleFunction",
    # Models & enums
    "PIIDetectionEntity",
    "PIIDetectionEntityType",
    "GuardrailExecutionStage",
    "GuardrailScope",
    "GuardrailAction",
    # Actions
    "LogAction",
    "BlockAction",
    "LoggingSeverityLevel",
    # Exception
    "GuardrailBlockException",
    # Adapter registry
    "GuardrailTargetAdapter",
    "register_guardrail_adapter",
    # Middlewares (unchanged)
    "UiPathPIIDetectionMiddleware",
    "UiPathPromptInjectionMiddleware",
    "UiPathDeterministicGuardrailMiddleware",
    # Re-exports for backwards compat
    "AgentGuardrailSeverityLevel",
]
