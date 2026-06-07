"""UiPath Guardrails for LangChain agents.

Platform guardrail decorators plus LangChain/LangGraph adapter auto-registration.
"""

from uipath.agent.models.agent import AgentGuardrailSeverityLevel
from uipath.platform.guardrails.decorators import (
    BlockAction,
    CustomValidator,
    GuardrailAction,
    GuardrailBlockException,
    GuardrailExclude,
    GuardrailExecutionStage,
    GuardrailTargetAdapter,
    GuardrailValidatorBase,
    HarmfulContentEntity,
    HarmfulContentEntityType,
    HarmfulContentValidator,
    IntellectualPropertyEntityType,
    IntellectualPropertyValidator,
    LogAction,
    LoggingSeverityLevel,
    PIIDetectionEntity,
    PIIDetectionEntityType,
    PIIValidator,
    PromptInjectionValidator,
    RuleFunction,
    UserPromptAttacksValidator,
    guardrail,
    register_guardrail_adapter,
)

from ._langchain_adapter import LangChainGuardrailAdapter
from .middlewares import (
    UiPathDeterministicGuardrailMiddleware,
    UiPathHarmfulContentMiddleware,
    UiPathIntellectualPropertyMiddleware,
    UiPathPIIDetectionMiddleware,
    UiPathPromptInjectionMiddleware,
    UiPathUserPromptAttacksMiddleware,
)

# Auto-register the LangChain adapter so @guardrail knows how to wrap
# BaseTool, BaseChatModel, StateGraph, and CompiledStateGraph.
register_guardrail_adapter(LangChainGuardrailAdapter())

__all__ = [
    # Decorator
    "guardrail",
    # Validators
    "GuardrailValidatorBase",
    "HarmfulContentValidator",
    "IntellectualPropertyValidator",
    "PIIValidator",
    "PromptInjectionValidator",
    "UserPromptAttacksValidator",
    "CustomValidator",
    "RuleFunction",
    # Models & enums
    "HarmfulContentEntity",
    "HarmfulContentEntityType",
    "IntellectualPropertyEntityType",
    "PIIDetectionEntity",
    "PIIDetectionEntityType",
    "GuardrailExecutionStage",
    "GuardrailAction",
    # Actions
    "LogAction",
    "BlockAction",
    "LoggingSeverityLevel",
    # Exception
    "GuardrailBlockException",
    # Exclude marker
    "GuardrailExclude",
    # Adapter registry
    "GuardrailTargetAdapter",
    "register_guardrail_adapter",
    # Middlewares
    "UiPathHarmfulContentMiddleware",
    "UiPathIntellectualPropertyMiddleware",
    "UiPathPIIDetectionMiddleware",
    "UiPathPromptInjectionMiddleware",
    "UiPathUserPromptAttacksMiddleware",
    "UiPathDeterministicGuardrailMiddleware",
    # Re-exports for backwards compat
    "AgentGuardrailSeverityLevel",
]
