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
    LLMAsJudgeValidator,
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
from .escalate_action import EscalateAction
from .middlewares import (
    UiPathDeterministicGuardrailMiddleware,
    UiPathHarmfulContentMiddleware,
    UiPathIntellectualPropertyMiddleware,
    UiPathLLMAsJudgeMiddleware,
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
    "LLMAsJudgeValidator",
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
    "EscalateAction",
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
    "UiPathLLMAsJudgeMiddleware",
    "UiPathPIIDetectionMiddleware",
    "UiPathPromptInjectionMiddleware",
    "UiPathUserPromptAttacksMiddleware",
    "UiPathDeterministicGuardrailMiddleware",
    # Re-exports for backwards compat
    "AgentGuardrailSeverityLevel",
]
