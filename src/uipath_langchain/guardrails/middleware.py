"""UiPath guardrails middleware for LangChain agents."""

import ast
import inspect
import json
import logging
import re
from typing import Any, Callable, Sequence
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
from uipath.platform.guardrails.guardrails import NumberParameterValue

from .enums import GuardrailExecutionStage
from .models import Entity, GuardrailAction

logger = logging.getLogger(__name__)


def _sanitize_tool_name(name: str) -> str:
    """Sanitize tool name for LLM compatibility (alphanumeric, underscore, hyphen only, max 64 chars)."""
    trim_whitespaces = "_".join(name.split())
    sanitized_tool_name = re.sub(r"[^a-zA-Z0-9_-]", "", trim_whitespaces)
    sanitized_tool_name = sanitized_tool_name[:64]
    return sanitized_tool_name


def _create_modified_tool_request(
    request: ToolCallRequest,
    modified_args: dict[str, Any],
) -> ToolCallRequest:
    """Create a new ToolCallRequest with modified args.

    Args:
        request: Original tool call request
        modified_args: Modified tool arguments (dict)

    Returns:
        New ToolCallRequest with modified args
    """
    from copy import deepcopy
    from dataclasses import replace

    # Create a deep copy of the tool_call
    modified_tool_call = deepcopy(request.tool_call)
    modified_tool_call["args"] = modified_args

    # Try to create new request using dataclass replace if ToolCallRequest is a dataclass
    try:
        return replace(request, tool_call=modified_tool_call)
    except (TypeError, AttributeError):
        # Fallback: create new ToolCallRequest instance with all required fields
        return ToolCallRequest(
            tool_call=modified_tool_call,
            tool=request.tool,
            state=request.state,
            runtime=request.runtime,
        )


def _create_modified_tool_result(
    result: ToolMessage | Command[Any],
    modified_output: dict[str, Any] | str,
) -> ToolMessage | Command[Any]:
    """Create a new ToolMessage or Command with modified output.

    Args:
        result: Original tool execution result
        modified_output: Modified tool output (dict or str)

    Returns:
        New ToolMessage or Command with modified output
    """
    from copy import deepcopy

    # Format modified output - convert dict to JSON string if needed
    if isinstance(modified_output, dict):
        import json

        formatted_output = json.dumps(modified_output)
    else:
        formatted_output = modified_output

    if isinstance(result, Command):
        # For Command, extract messages from update and modify the ToolMessage
        update = result.update if hasattr(result, "update") else {}
        messages = update.get("messages", []) if isinstance(update, dict) else []
        if messages and isinstance(messages[0], ToolMessage):
            # Create new ToolMessage with modified content
            modified_message = deepcopy(messages[0])
            modified_message.content = formatted_output

            # Create new Command with modified message
            new_update: dict[str, Any] = (
                dict(deepcopy(update)) if isinstance(update, dict) else {}
            )
            new_update["messages"] = [modified_message]
            return Command(update=new_update)
        return result
    elif isinstance(result, ToolMessage):
        # Create new ToolMessage with modified content
        modified_message = deepcopy(result)
        modified_message.content = formatted_output
        return modified_message
    else:
        return result


def _extract_text_from_messages(messages: list[BaseMessage]) -> str:
    """Extract text content from messages."""
    text_parts = []
    for msg in messages:
        if isinstance(msg, (HumanMessage, AIMessage)):
            if isinstance(msg.content, str):
                text_parts.append(msg.content)
            elif isinstance(msg.content, list):
                # Handle multimodal content
                for part in msg.content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
    return "\n".join(text_parts)


class UiPathPIIDetectionMiddleware:
    """Middleware for PII detection using UiPath guardrails.

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain_core.tools import tool
        from uipath_langchain.guardrails import (
            UiPathPIIDetectionMiddleware, Entity, LogAction, GuardrailScope, PIIDetectionEntity
        )

        @tool
        def analyze_joke_syntax(joke: str) -> str:
            \"\"\"Analyze the syntax of a joke.\"\"\"
            return f"Words: {len(joke.split())}"

        # PII detection for Agent and LLM scopes
        middleware_agent_llm = UiPathPIIDetectionMiddleware(
            scopes=[GuardrailScope.AGENT, GuardrailScope.LLM],
            action=LogAction(severityLevel=AgentGuardrailSeverityLevel.WARNING),
            entities=[
                Entity(PIIDetectionEntity.EMAIL, 0.5),
                Entity(PIIDetectionEntity.ADDRESS, 0.7),
            ],
        )

        # PII detection for specific tools (using tool reference directly)
        middleware_tool = UiPathPIIDetectionMiddleware(
            scopes=[GuardrailScope.TOOL],
            action=LogAction(severityLevel=AgentGuardrailSeverityLevel.WARNING),
            entities=[Entity(PIIDetectionEntity.EMAIL, 0.5)],
            tool_names=[analyze_joke_syntax],  # Pass tool object directly
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
        tool_names: Optional list of tool names or tool objects to apply guardrail to.
            If None and TOOL scope is specified, applies to all tools.
            If provided, only applies to the specified tools.
            Can be a mix of strings (tool names) or BaseTool objects.
        name: Optional name for the guardrail (defaults to "PII Detection")
        description: Optional description for the guardrail
    """

    def __init__(
        self,
        scopes: Sequence[GuardrailScope],
        action: GuardrailAction,
        entities: Sequence[Entity],
        *,
        tool_names: Sequence[str | BaseTool] | None = None,
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

        # Process tool_names: extract names from tool objects and sanitize
        self._tool_names: list[str] | None = None
        if tool_names is not None:
            tool_name_list = []
            for tool_or_name in tool_names:
                if isinstance(tool_or_name, BaseTool):
                    tool_name_list.append(_sanitize_tool_name(tool_or_name.name))
                elif isinstance(tool_or_name, str):
                    tool_name_list.append(_sanitize_tool_name(tool_or_name))
                else:
                    raise ValueError(
                        f"tool_names must contain strings or BaseTool objects, got {type(tool_or_name)}"
                    )
            self._tool_names = tool_name_list

        # Handle TOOL scope: only filter it out if tool_names is not provided
        scopes_list = list(scopes)
        if GuardrailScope.TOOL in scopes_list:
            if self._tool_names is None:
                logger.warning(
                    f"Tool scope is specified but tool_names is None. "
                    f"Tool scope guardrails require tool_names to be specified. "
                    f"Ignoring Tool scope for middleware '{name}'."
                )
                scopes_list = [s for s in scopes_list if s != GuardrailScope.TOOL]
                if not scopes_list:
                    raise ValueError(
                        "At least one supported scope (Agent, LLM, or Tool with tool_names) must be specified."
                    )
            # If tool_names is provided, keep TOOL scope

        self.scopes = scopes_list
        self.action = action
        self.entities = list(entities)
        self._name = name
        self._description = (
            description
            or f"Detects PII entities: {', '.join(e.name for e in entities)}"
        )

        # Convert to BuiltInValidatorGuardrail
        self._guardrail = self._create_guardrail()
        self._uipath: UiPath | None = None  # Lazy initialization

        # Create middleware instances from decorated functions
        self._middleware_instances = self._create_middleware_instances()

    def _create_middleware_instances(self) -> list[AgentMiddleware]:
        """Create middleware instances from decorated functions."""
        instances = []
        middleware_instance = self  # Capture self for closure
        guardrail_name = self._name.replace(
            " ", "_"
        )  # Sanitize name for function names

        # Create before_agent middleware if AGENT scope is enabled
        if GuardrailScope.AGENT in self.scopes:
            # Create function with unique name
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

        # Create before_model/after_model middleware if LLM scope is enabled
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

        # Create wrap_tool_call middleware if TOOL scope is enabled
        if GuardrailScope.TOOL in self.scopes:

            async def _wrap_tool_call_func(
                request: ToolCallRequest,
                handler: Any,
            ) -> ToolMessage | Command[Any]:
                """Wrap tool call to check for PII before execution."""
                # Get tool name from request
                tool_call = request.tool_call
                tool_name = tool_call.get("name", "")
                sanitized_tool_name = _sanitize_tool_name(tool_name)

                # Check if we should apply guardrail to this tool
                if middleware_instance._tool_names is not None:
                    # Only check specified tools
                    if sanitized_tool_name not in middleware_instance._tool_names:
                        # Tool not in list, proceed without checking
                        return await handler(request)
                # If tool_names is None, check all tools

                # Extract tool input data
                input_data = middleware_instance._extract_tool_input_data(request)

                # Evaluate guardrail
                try:
                    result = middleware_instance._evaluate_guardrail(input_data)
                    # Get modified input from action
                    modified_input = middleware_instance._handle_validation_result(
                        result, input_data
                    )
                    # If BlockAction, exception already raised in _handle_validation_result

                    # Apply modification if action returned modified data
                    if modified_input is not None and isinstance(modified_input, dict):
                        # Create new request with modified args
                        request = _create_modified_tool_request(request, modified_input)
                except Exception as e:
                    logger.error(
                        f"Error evaluating PII guardrail for tool '{tool_name}': {e}",
                        exc_info=True,
                    )
                    # Continue with tool execution even if guardrail evaluation fails

                # Proceed with tool execution
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
        # Extract entity names and thresholds
        entity_names = [entity.name for entity in self.entities]
        entity_thresholds = {entity.name: entity.threshold for entity in self.entities}

        # Create validator parameters
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

        # Create selector with match_names if tool_names is provided and TOOL scope is enabled
        selector_kwargs: dict[str, Any] = {"scopes": self.scopes}
        if GuardrailScope.TOOL in self.scopes and self._tool_names is not None:
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
        """Handle guardrail validation result.

        Returns:
            Modified data from the action, or None if no modification.
        """
        from uipath.core.guardrails import GuardrailValidationResultType

        if result.result == GuardrailValidationResultType.VALIDATION_FAILED:
            return self.action.handle_validation_result(result, input_data, self._name)
        return None

    def _extract_tool_input_data(
        self, request: ToolCallRequest
    ) -> str | dict[str, Any]:
        """Extract tool input data from ToolCallRequest for guardrail evaluation.

        Args:
            request: The tool call request containing tool call information

        Returns:
            Tool arguments as dict or string representation
        """
        tool_call = request.tool_call
        args = tool_call.get("args", {})

        # Return as dict if it's already a dict, otherwise convert to string
        if isinstance(args, dict):
            return args
        else:
            # Convert to string representation
            return str(args)

    def _check_messages(self, messages: list[BaseMessage]) -> None:
        """Check messages for PII and update with modified content if needed."""
        if not messages:
            return

        text = _extract_text_from_messages(messages)
        if not text:
            return

        try:
            result = self._evaluate_guardrail(text)
            # Get modified text from action
            modified_text = self._handle_validation_result(result, text)
            # If BlockAction, exception already raised in _handle_validation_result

            # Apply modification if action returned modified text
            if (
                modified_text is not None
                and isinstance(modified_text, str)
                and modified_text != text
            ):
                # Update message content in place
                # Find the message(s) that contain the text and update them
                for msg in messages:
                    if isinstance(msg, (HumanMessage, AIMessage)):
                        if isinstance(msg.content, str) and text in msg.content:
                            # Replace the extracted text with modified text
                            msg.content = msg.content.replace(
                                text, modified_text, 1
                            )  # Replace first occurrence
                            break
        except Exception as e:
            logger.error(f"Error evaluating PII guardrail: {e}", exc_info=True)


class UiPathPromptInjectionMiddleware:
    """Middleware for prompt injection detection using UiPath guardrails.

    Example:
        ```python
        from uipath_langchain.guardrails import UiPathPromptInjectionMiddleware, LogAction, GuardrailScope

        middleware = UiPathPromptInjectionMiddleware(
            scopes=[GuardrailScope.LLM],
            action=LogAction(severityLevel=AgentGuardrailSeverityLevel.WARNING),
            threshold=0.5,
        )
        ```

    Args:
        scopes: List of scopes where the guardrail applies. Only LLM scope is supported.
        action: Action to take when prompt injection is detected (LogAction or BlockAction)
        threshold: Detection threshold (0.0 to 1.0)
        name: Optional name for the guardrail (defaults to "Prompt Injection Detection")
        description: Optional description for the guardrail
    """

    def __init__(
        self,
        scopes: Sequence[GuardrailScope],
        action: GuardrailAction,
        threshold: float = 0.5,
        *,
        name: str = "Prompt Injection Detection",
        description: str | None = None,
    ):
        """Initialize prompt injection detection guardrail middleware."""
        if not scopes:
            raise ValueError("At least one scope must be specified")
        if not isinstance(action, GuardrailAction):
            raise ValueError("action must be an instance of GuardrailAction")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")

        # Prompt injection detection only supports LLM scope
        scopes_list = list(scopes)
        invalid_scopes = [s for s in scopes_list if s != GuardrailScope.LLM]
        if invalid_scopes:
            invalid_scope_names = [s.name for s in invalid_scopes]
            raise ValueError(
                f"Prompt injection detection only supports LLM scope. "
                f"Invalid scopes provided: {invalid_scope_names}. "
                f"Please use scopes=[GuardrailScope.LLM]."
            )

        if GuardrailScope.LLM not in scopes_list:
            raise ValueError(
                "Prompt injection detection requires LLM scope. "
                "Please use scopes=[GuardrailScope.LLM]."
            )

        self.scopes = [GuardrailScope.LLM]
        self.action = action
        self.threshold = threshold
        self._name = name
        self._description = (
            description
            or f"Detects prompt injection attempts with threshold {threshold}"
        )

        # Convert to BuiltInValidatorGuardrail
        self._guardrail = self._create_guardrail()
        self._uipath: UiPath | None = None  # Lazy initialization

        # Create middleware instances from decorated functions
        self._middleware_instances = self._create_middleware_instances()

    def _create_middleware_instances(self) -> list[AgentMiddleware]:
        """Create middleware instances from decorated functions."""
        instances = []
        middleware_instance = self  # Capture self for closure
        guardrail_name = self._name.replace(
            " ", "_"
        )  # Sanitize name for function names

        # Prompt injection detection only supports LLM scope
        # Create before_model/after_model middleware
        async def _before_model_func(state: AgentState[Any], runtime: Runtime) -> None:
            messages = state.get("messages", [])
            middleware_instance._check_messages(list(messages))

        _before_model_func.__name__ = f"{guardrail_name}_before_model"
        _before_model = before_model(_before_model_func)
        instances.append(_before_model)

        async def _after_model_func(state: AgentState[Any], runtime: Runtime) -> None:
            messages = state.get("messages", [])
            ai_messages = [msg for msg in messages if isinstance(msg, AIMessage)]
            if ai_messages:
                middleware_instance._check_messages([ai_messages[-1]])

        _after_model_func.__name__ = f"{guardrail_name}_after_model"
        _after_model = after_model(_after_model_func)
        instances.append(_after_model)

        return instances

    def __iter__(self):
        """Make the class iterable to return middleware instances."""
        return iter(self._middleware_instances)

    def _create_guardrail(self) -> BuiltInValidatorGuardrail:
        """Create BuiltInValidatorGuardrail from configuration."""
        # Create validator parameters
        validator_parameters = [
            NumberParameterValue(
                parameter_type="number",
                id="threshold",
                value=self.threshold,
            ),
        ]

        return BuiltInValidatorGuardrail(
            id=str(uuid4()),
            name=self._name,
            description=self._description,
            enabled_for_evals=True,
            selector=GuardrailSelector(scopes=self.scopes),
            guardrail_type="builtInValidator",
            validator_type="prompt_injection",
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
        """Handle guardrail validation result.

        Returns:
            Modified data from the action, or None if no modification.
        """
        from uipath.core.guardrails import GuardrailValidationResultType

        if result.result == GuardrailValidationResultType.VALIDATION_FAILED:
            return self.action.handle_validation_result(result, input_data, self._name)
        return None

    def _check_messages(self, messages: list[BaseMessage]) -> None:
        """Check messages for prompt injection and update with modified content if needed."""
        if not messages:
            return

        text = _extract_text_from_messages(messages)
        if not text:
            return

        try:
            result = self._evaluate_guardrail(text)
            # Get modified text from action
            modified_text = self._handle_validation_result(result, text)
            # If BlockAction, exception already raised in _handle_validation_result

            # Apply modification if action returned modified text
            if (
                modified_text is not None
                and isinstance(modified_text, str)
                and modified_text != text
            ):
                # Update message content in place
                # Find the message(s) that contain the text and update them
                for msg in messages:
                    if isinstance(msg, (HumanMessage, AIMessage)):
                        if isinstance(msg.content, str) and text in msg.content:
                            # Replace the extracted text with modified text
                            msg.content = msg.content.replace(
                                text, modified_text, 1
                            )  # Replace first occurrence
                            break
        except Exception as e:
            logger.error(
                f"Error evaluating prompt injection guardrail: {e}", exc_info=True
            )


# Type alias for rule functions
RuleFunction = (
    Callable[[dict[str, Any]], bool] | Callable[[dict[str, Any], dict[str, Any]], bool]
)


class UiPathDeterministicGuardrailMiddleware:
    """Middleware for deterministic guardrails using custom rule functions.

    This middleware allows developers to define lambda-like functions for tool-level validation.
    The functions receive the actual tool input/output arguments and return a boolean indicating
    if a violation is detected.

    Example:
        ```python
        from uipath_langchain.guardrails import (
            UiPathDeterministicGuardrailMiddleware,
            GuardrailExecutionStage,
            BlockAction,
        )

        # Using lambda functions with PRE stage (input validation only)
        deterministic_guardrail = UiPathDeterministicGuardrailMiddleware(
            tool_names=[analyze_joke_syntax],
            rules=[
                lambda input_data: "forbidden" in input_data.get("joke", "").lower(),
                lambda input_data: len(input_data.get("joke", "")) > 1000,
            ],
            action=BlockAction(),
            stage=GuardrailExecutionStage.PRE,
            name="Joke Content Validator",
        )

        # Empty rules means always pass (no validation)
        always_pass_guardrail = UiPathDeterministicGuardrailMiddleware(
            tool_names=[analyze_joke_syntax],
            rules=[],  # No validation - always passes
            action=BlockAction(),
            stage=GuardrailExecutionStage.PRE_AND_POST,
        )

        agent = create_agent(
            model=llm,
            tools=[analyze_joke_syntax],
            middleware=[*deterministic_guardrail],
        )
        ```

    Args:
        tool_names: List of tool names or tool objects to apply guardrail to.
            Can be a mix of strings (tool names) or BaseTool objects.
        rules: List of callable functions that receive tool input/output data.
            - Functions with 1 parameter: `Callable[[dict[str, Any]], bool]` for input-only validation
            - Functions with 2 parameters: `Callable[[dict[str, Any], dict[str, Any]], bool]` for input+output validation
            - Functions return `True` if violation detected, `False` otherwise
            - Empty list `[]` means always pass (no validation)
            - When multiple rules are provided, ALL rules must detect violations for the guardrail to trigger.
              If ANY rule passes (returns False), the guardrail passes.
        action: Action to take when violation is detected (LogAction or BlockAction)
        stage: Execution stage for the guardrail (required). Options:
            - GuardrailExecutionStage.PRE: Only validate tool input (pre-execution)
            - GuardrailExecutionStage.POST: Only validate tool output (post-execution)
            - GuardrailExecutionStage.PRE_AND_POST: Validate both input and output
        name: Optional name for the guardrail (defaults to "Deterministic Guardrail")
        description: Optional description for the guardrail
    """

    def __init__(
        self,
        tool_names: Sequence[str | BaseTool],
        rules: Sequence[RuleFunction],
        action: GuardrailAction,
        stage: GuardrailExecutionStage,
        *,
        name: str = "Deterministic Guardrail",
        description: str | None = None,
    ):
        """Initialize deterministic guardrail middleware."""
        if not tool_names:
            raise ValueError("At least one tool name must be specified")
        if not isinstance(action, GuardrailAction):
            raise ValueError("action must be an instance of GuardrailAction")
        if not isinstance(stage, GuardrailExecutionStage):
            raise ValueError(
                f"stage must be an instance of GuardrailExecutionStage, got {type(stage)}"
            )

        # Validate rules (must be callable)
        for i, rule in enumerate(rules):
            if not callable(rule):
                raise ValueError(f"Rule {i + 1} must be callable, got {type(rule)}")
            # Validate parameter count
            sig = inspect.signature(rule)
            param_count = len(sig.parameters)
            if param_count not in (1, 2):
                raise ValueError(
                    f"Rule {i + 1} must have 1 or 2 parameters, got {param_count}"
                )

        # Process tool_names: extract names from tool objects and sanitize
        tool_name_list = []
        for tool_or_name in tool_names:
            if isinstance(tool_or_name, BaseTool):
                tool_name_list.append(_sanitize_tool_name(tool_or_name.name))
            elif isinstance(tool_or_name, str):
                tool_name_list.append(_sanitize_tool_name(tool_or_name))
            else:
                raise ValueError(
                    f"tool_names must contain strings or BaseTool objects, got {type(tool_or_name)}"
                )
        self._tool_names = set(tool_name_list)

        self.rules = list(rules)
        self.action = action
        self._stage = stage
        self._name = name
        self._description = description or "Deterministic guardrail with custom rules"

        # Create middleware instances from decorated functions
        self._middleware_instances = self._create_middleware_instances()

    def _create_middleware_instances(self) -> list[AgentMiddleware]:
        """Create middleware instances from decorated functions."""
        instances = []
        middleware_instance = self  # Capture self for closure
        guardrail_name = self._name.replace(
            " ", "_"
        )  # Sanitize name for function names

        # Deterministic guardrails only support TOOL scope
        # Create wrap_tool_call middleware
        async def _wrap_tool_call_func(
            request: ToolCallRequest,
            handler: Any,
        ) -> ToolMessage | Command[Any]:
            """Wrap tool call to evaluate deterministic rules."""
            # Get tool name from request
            tool_call = request.tool_call
            tool_name = tool_call.get("name", "")
            sanitized_tool_name = _sanitize_tool_name(tool_name)

            # Check if we should apply guardrail to this tool
            if sanitized_tool_name not in middleware_instance._tool_names:
                # Tool not in list, proceed without checking
                return await handler(request)

            # Extract tool input data
            input_data = middleware_instance._extract_tool_input_data(request)

            # Evaluate rules at PRE stage if stage allows PRE execution
            if middleware_instance._stage in (
                GuardrailExecutionStage.PRE,
                GuardrailExecutionStage.PRE_AND_POST,
            ):
                # At PRE stage, evaluate all rules with input_data
                # 1-parameter rules receive input_data (tool input)
                # 2-parameter rules are skipped (no output_data available yet)
                if middleware_instance.rules:
                    result = middleware_instance._evaluate_rules(
                        middleware_instance.rules, input_data, None, None
                    )
                    if result.result == GuardrailValidationResultType.VALIDATION_FAILED:
                        # Get modified input from action
                        modified_input = middleware_instance._handle_validation_result(
                            result, input_data
                        )
                        # If BlockAction, exception already raised in _handle_validation_result

                        # Apply modification if action returned modified data
                        if modified_input is not None and isinstance(
                            modified_input, dict
                        ):
                            # Create new request with modified args
                            request = _create_modified_tool_request(
                                request, modified_input
                            )
                            # Update input_data to use filtered version for POST stage rules
                            input_data = modified_input

            # Execute tool
            tool_result = await handler(request)

            # Evaluate rules at POST stage if stage allows POST execution
            if middleware_instance._stage in (
                GuardrailExecutionStage.POST,
                GuardrailExecutionStage.PRE_AND_POST,
            ):
                output_data = middleware_instance._extract_tool_output_data(tool_result)
                # At POST stage, evaluate all rules
                # 1-parameter rules receive output_data (tool output)
                # 2-parameter rules receive both input_data and output_data
                if middleware_instance.rules:
                    result = middleware_instance._evaluate_rules(
                        middleware_instance.rules,
                        output_data,  # For 1-param rules at POST, this is the output
                        output_data,  # For 2-param rules, this is the output
                        input_data,  # For 2-param rules, this is the original input
                    )
                    if result.result == GuardrailValidationResultType.VALIDATION_FAILED:
                        # For output validation, pass just the output_data to the action
                        # The action should return the modified output (dict or str)
                        modified_output = middleware_instance._handle_validation_result(
                            result, output_data
                        )
                        # If BlockAction, exception already raised in _handle_validation_result

                        # Apply modification if action returned modified data
                        if modified_output is not None:
                            # Create new result with modified output
                            tool_result = _create_modified_tool_result(
                                tool_result, modified_output
                            )

            return tool_result

        _wrap_tool_call_func.__name__ = f"{guardrail_name}_wrap_tool_call"
        _wrap_tool_call = wrap_tool_call(_wrap_tool_call_func)  # type: ignore[call-overload]
        instances.append(_wrap_tool_call)

        return instances

    def __iter__(self):
        """Make the class iterable to return middleware instances."""
        return iter(self._middleware_instances)

    def _evaluate_rules(
        self,
        rules: Sequence[RuleFunction],
        data_for_1param: dict[str, Any],
        output_data: dict[str, Any] | None = None,
        input_data: dict[str, Any] | None = None,
    ) -> GuardrailValidationResult:
        """Evaluate all rules and return validation result.

        The guardrail triggers (VALIDATION_FAILED) only if ALL evaluated rules detect violations.
        If ANY rule passes (returns False), the guardrail passes.

        Args:
            rules: List of rule functions to evaluate
            data_for_1param: Data to pass to 1-parameter rules.
                At PRE stage: tool input data
                At POST stage: tool output data
            output_data: Tool output data (for 2-parameter rules)
            input_data: Tool input data (for 2-parameter rules at POST stage)
        """
        if not rules:
            # Empty rules means always pass
            return GuardrailValidationResult(
                result=GuardrailValidationResultType.PASSED,
                reason="No rules to evaluate",
            )

        violations = []
        passed_rules = []
        evaluated_count = 0

        for i, rule in enumerate(rules):
            sig = inspect.signature(rule)
            param_count = len(sig.parameters)

            try:
                if param_count == 1:
                    # 1-parameter rules receive data_for_1param
                    # At PRE: this is input_data
                    # At POST: this is output_data
                    rule_1param: Callable[[dict[str, Any]], bool] = rule  # type: ignore[assignment]
                    violation = rule_1param(data_for_1param)
                    evaluated_count += 1
                elif param_count == 2:
                    # 2-parameter rules need both input and output
                    if output_data is None or input_data is None:
                        # Skip 2-parameter rules if we don't have both inputs
                        continue
                    rule_2param: Callable[[dict[str, Any], dict[str, Any]], bool] = rule  # type: ignore[assignment]
                    violation = rule_2param(input_data, output_data)
                    evaluated_count += 1
                else:
                    raise ValueError(
                        f"Rule function must have 1 or 2 parameters, got {param_count}"
                    )

                if violation:
                    violations.append(f"Rule {i + 1} detected violation")
                else:
                    passed_rules.append(f"Rule {i + 1}")
            except Exception as e:
                # Treat exceptions as violations (fail-safe)
                logger.error(f"Error in rule function {i + 1}: {e}", exc_info=True)
                violations.append(f"Rule {i + 1} raised exception: {str(e)}")
                evaluated_count += 1

        # If no rules were evaluated (e.g., only 2-parameter rules without both inputs), pass
        if evaluated_count == 0:
            return GuardrailValidationResult(
                result=GuardrailValidationResultType.PASSED,
                reason="No applicable rules to evaluate",
            )

        # Guardrail triggers only if ALL evaluated rules detected violations
        # If any rule passed, the guardrail passes
        if passed_rules:
            return GuardrailValidationResult(
                result=GuardrailValidationResultType.PASSED,
                reason=f"Rules passed: {', '.join(passed_rules)}",
            )

        # All evaluated rules detected violations
        return GuardrailValidationResult(
            result=GuardrailValidationResultType.VALIDATION_FAILED,
            reason="; ".join(violations),
        )

    def _extract_tool_input_data(self, request: ToolCallRequest) -> dict[str, Any]:
        """Extract tool input data from ToolCallRequest for rule evaluation.

        Args:
            request: The tool call request containing tool call information

        Returns:
            Tool arguments as dict
        """
        tool_call = request.tool_call
        args = tool_call.get("args", {})

        # Return as dict if it's already a dict, otherwise convert
        if isinstance(args, dict):
            return args
        else:
            # Convert to dict representation
            return {"args": args}

    def _extract_tool_output_data(
        self, result: ToolMessage | Command[Any]
    ) -> dict[str, Any]:
        """Extract tool output data from handler result.

        Args:
            result: The tool execution result (ToolMessage or Command)

        Returns:
            Tool output as dict
        """
        if isinstance(result, Command):
            # Extract from Command update if available
            update = result.update if hasattr(result, "update") else {}
            messages = update.get("messages", []) if isinstance(update, dict) else []
            if messages and isinstance(messages[0], ToolMessage):
                content = messages[0].content
            else:
                return {}
        elif isinstance(result, ToolMessage):
            content = result.content
        else:
            return {}

        # Parse content to dict (similar to _extract_tool_output_data in guardrails/utils.py)
        if isinstance(content, dict):
            return content
        elif isinstance(content, str):
            try:
                parsed = json.loads(content)
                return parsed if isinstance(parsed, dict) else {"output": parsed}
            except json.JSONDecodeError:
                try:
                    parsed = ast.literal_eval(content)
                    return parsed if isinstance(parsed, dict) else {"output": parsed}
                except (ValueError, SyntaxError):
                    return {"output": content}
        else:
            return {"output": content}

    def _handle_validation_result(
        self, result: GuardrailValidationResult, input_data: str | dict[str, Any]
    ) -> str | dict[str, Any] | None:
        """Handle guardrail validation result.

        Returns:
            Modified data from the action, or None if no modification.
        """
        if result.result == GuardrailValidationResultType.VALIDATION_FAILED:
            return self.action.handle_validation_result(result, input_data, self._name)
        return None
