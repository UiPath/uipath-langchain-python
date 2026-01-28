"""PII detection guardrail middleware."""

import logging
from typing import Any, Sequence
from uuid import uuid4

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    after_agent,
    after_model,
    before_agent,
    before_model,
    wrap_tool_call,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command
from uipath.core.guardrails import (
    GuardrailSelector,
    GuardrailValidationResult,
    GuardrailValidationResultType,
)
from uipath.platform import UiPath
from uipath.platform.guardrails import (
    BuiltInValidatorGuardrail,
    EnumListParameterValue,
    GuardrailScope,
    MapEnumParameterValue,
)

from ..models import GuardrailAction, PIIDetectionEntity
from ._utils import (
    create_modified_tool_request,
    extract_text_from_messages,
    sanitize_tool_name,
)

logger = logging.getLogger(__name__)


class UiPathPIIDetectionMiddleware:
    """Middleware for PII detection using UiPath guardrails.

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain_core.tools import tool
        from uipath_langchain.guardrails import (
            UiPathPIIDetectionMiddleware,
            PIIDetectionEntity,
            PIIDetectionEntityType,
            LogAction,
            GuardrailScope,
        )
        from uipath_langchain.guardrails.actions import LoggingSeverityLevel

        @tool
        def analyze_joke_syntax(joke: str) -> str:
            \"\"\"Analyze the syntax of a joke.\"\"\"
            return f"Words: {len(joke.split())}"

        # PII detection for Agent and LLM scopes
        middleware_agent_llm = UiPathPIIDetectionMiddleware(
            scopes=[GuardrailScope.AGENT, GuardrailScope.LLM],
            action=LogAction(severity_level=LoggingSeverityLevel.WARNING),
            entities=[
                PIIDetectionEntity(PIIDetectionEntityType.EMAIL, 0.5),
                PIIDetectionEntity(PIIDetectionEntityType.ADDRESS, 0.7),
            ],
        )

        # PII detection for specific tools (using tool reference directly)
        middleware_tool = UiPathPIIDetectionMiddleware(
            scopes=[GuardrailScope.TOOL],
            action=LogAction(severity_level=LoggingSeverityLevel.WARNING),
            entities=[PIIDetectionEntity(PIIDetectionEntityType.EMAIL, 0.5)],
            tools=[analyze_joke_syntax],
        )

        agent = create_agent(
            model=llm,
            tools=[analyze_joke_syntax],
            middleware=[*middleware_agent_llm, *middleware_tool],
        )
        ```

    Args:
        scopes: List of scopes where the guardrail applies (Agent, LLM, Tool)
        action: Action to take when PII is detected (LogAction or BlockAction)
        entities: List of PII entities to detect with their thresholds
        tools: Required when TOOL scope is specified. List of tool names or tool objects
            to apply guardrail to. Must contain at least one tool.
            Can be a mix of strings (tool names) or BaseTool objects.
            If TOOL scope is not specified, this parameter is ignored.
        name: Optional name for the guardrail (defaults to "PII Detection")
        description: Optional description for the guardrail
    """

    def __init__(
        self,
        scopes: Sequence[GuardrailScope],
        action: GuardrailAction,
        entities: Sequence[PIIDetectionEntity],
        *,
        tools: Sequence[str | BaseTool] | None = None,
        name: str = "PII Detection",
        description: str | None = None,
    ):
        """Initialize PII detection guardrail middleware."""
        if not scopes:
            raise ValueError("At least one scope must be specified")
        if not entities:
            raise ValueError("At least one entity must be specified")
        if not isinstance(action, GuardrailAction):
            raise ValueError("action must be an instance of GuardrailAction")

        self._tool_names: list[str] | None = None
        if tools is not None:
            tool_name_list = []
            for tool_or_name in tools:
                if isinstance(tool_or_name, BaseTool):
                    tool_name_list.append(sanitize_tool_name(tool_or_name.name))
                elif isinstance(tool_or_name, str):
                    tool_name_list.append(sanitize_tool_name(tool_or_name))
                else:
                    raise ValueError(
                        f"tool_names must contain strings or BaseTool objects, got {type(tool_or_name)}"
                    )
            self._tool_names = tool_name_list

        scopes_list = list(scopes)
        if GuardrailScope.TOOL in scopes_list:
            if self._tool_names is None or len(self._tool_names) == 0:
                raise ValueError(
                    "Tool scope is specified but tool_names is None or empty. "
                    "Tool scope guardrails require at least one tool name to be specified. "
                    "Please provide tool_names when using GuardrailScope.TOOL."
                )

        self.scopes = scopes_list
        self.action = action
        self.entities = list(entities)
        self._name = name
        self._description = (
            description
            or f"Detects PII entities: {', '.join(e.name for e in entities)}"
        )

        self._guardrail = self._create_guardrail()
        self._uipath: UiPath | None = None
        self._middleware_instances = self._create_middleware_instances()

    def _create_middleware_instances(self) -> list[AgentMiddleware]:
        """Create middleware instances from decorated functions."""
        instances = []
        middleware_instance = self
        guardrail_name = self._name.replace(" ", "_")

        if GuardrailScope.AGENT in self.scopes:

            async def _before_agent_func(
                state: AgentState[Any], runtime: Runtime
            ) -> None:
                messages = state.get("messages", [])
                middleware_instance._check_messages(list(messages))

            _before_agent_func.__name__ = f"{guardrail_name}_before_agent"
            _before_agent = before_agent(_before_agent_func)
            instances.append(_before_agent)

            async def _after_agent_func(
                state: AgentState[Any], runtime: Runtime
            ) -> None:
                messages = state.get("messages", [])
                middleware_instance._check_messages(list(messages))

            _after_agent_func.__name__ = f"{guardrail_name}_after_agent"
            _after_agent = after_agent(_after_agent_func)
            instances.append(_after_agent)

        if GuardrailScope.LLM in self.scopes:

            async def _before_model_func(
                state: AgentState[Any], runtime: Runtime
            ) -> None:
                messages = state.get("messages", [])
                middleware_instance._check_messages(list(messages))

            _before_model_func.__name__ = f"{guardrail_name}_before_model"
            _before_model = before_model(_before_model_func)
            instances.append(_before_model)

            async def _after_model_func(
                state: AgentState[Any], runtime: Runtime
            ) -> None:
                messages = state.get("messages", [])
                ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
                if ai_messages:
                    middleware_instance._check_messages([ai_messages[-1]])

            _after_model_func.__name__ = f"{guardrail_name}_after_model"
            _after_model = after_model(_after_model_func)
            instances.append(_after_model)

        if GuardrailScope.TOOL in self.scopes:

            async def _wrap_tool_call_func(
                request: ToolCallRequest,
                handler: Any,
            ) -> ToolMessage | Command[Any]:
                tool_call = request.tool_call
                tool_name = tool_call.get("name", "")
                sanitized_tool_name = sanitize_tool_name(tool_name)

                if (
                    middleware_instance._tool_names is None
                    or sanitized_tool_name not in middleware_instance._tool_names
                ):
                    return await handler(request)

                input_data = middleware_instance._extract_tool_input_data(request)

                try:
                    result = middleware_instance._evaluate_guardrail(input_data)
                    modified_input = middleware_instance._handle_validation_result(
                        result, input_data
                    )
                    if modified_input is not None and isinstance(modified_input, dict):
                        request = create_modified_tool_request(request, modified_input)
                except Exception as e:
                    logger.error(
                        f"Error evaluating PII guardrail for tool '{tool_name}': {e}",
                        exc_info=True,
                    )

                return await handler(request)

            _wrap_tool_call_func.__name__ = f"{guardrail_name}_wrap_tool_call"
            _wrap_tool_call = wrap_tool_call(_wrap_tool_call_func)  # type: ignore[call-overload]
            instances.append(_wrap_tool_call)

        return instances

    def __iter__(self):
        """Make the class iterable to return middleware instances."""
        return iter(self._middleware_instances)

    def _create_guardrail(self) -> BuiltInValidatorGuardrail:
        """Create BuiltInValidatorGuardrail from configuration."""
        entity_names = [entity.name for entity in self.entities]
        entity_thresholds = {entity.name: entity.threshold for entity in self.entities}

        validator_parameters = [
            EnumListParameterValue(
                parameter_type="enum-list",
                id="entities",
                value=entity_names,
            ),
            MapEnumParameterValue(
                parameter_type="map-enum",
                id="entityThresholds",
                value=entity_thresholds,
            ),
        ]

        selector_kwargs: dict[str, Any] = {"scopes": self.scopes}
        if GuardrailScope.TOOL in self.scopes:
            selector_kwargs["match_names"] = self._tool_names

        return BuiltInValidatorGuardrail(
            id=str(uuid4()),
            name=self._name,
            description=self._description,
            enabled_for_evals=True,
            selector=GuardrailSelector(**selector_kwargs),
            guardrail_type="builtInValidator",
            validator_type="pii_detection",
            validator_parameters=validator_parameters,
        )

    def _get_uipath(self) -> UiPath:
        """Get or create UiPath instance."""
        if self._uipath is None:
            self._uipath = UiPath()
        return self._uipath

    def _evaluate_guardrail(
        self, input_data: str | dict[str, Any]
    ) -> GuardrailValidationResult:
        """Evaluate guardrail against input data."""
        uipath = self._get_uipath()
        return uipath.guardrails.evaluate_guardrail(input_data, self._guardrail)

    def _handle_validation_result(
        self, result: GuardrailValidationResult, input_data: str | dict[str, Any]
    ) -> str | dict[str, Any] | None:
        """Handle guardrail validation result."""
        if result.result == GuardrailValidationResultType.VALIDATION_FAILED:
            return self.action.handle_validation_result(result, input_data, self._name)
        return None

    def _extract_tool_input_data(
        self, request: ToolCallRequest
    ) -> str | dict[str, Any]:
        """Extract tool input data from ToolCallRequest for guardrail evaluation."""
        tool_call = request.tool_call
        args = tool_call.get("args", {})
        if isinstance(args, dict):
            return args
        return str(args)

    def _check_messages(self, messages: list[BaseMessage]) -> None:
        """Check messages for PII and update with modified content if needed."""
        if not messages:
            return

        text = extract_text_from_messages(messages)
        if not text:
            return

        try:
            result = self._evaluate_guardrail(text)
            modified_text = self._handle_validation_result(result, text)
            if (
                modified_text is not None
                and isinstance(modified_text, str)
                and modified_text != text
            ):
                for msg in messages:
                    if isinstance(msg, (HumanMessage, AIMessage)):
                        if isinstance(msg.content, str) and text in msg.content:
                            msg.content = msg.content.replace(text, modified_text, 1)
                            break
        except Exception as e:
            logger.error(f"Error evaluating PII guardrail: {e}", exc_info=True)
