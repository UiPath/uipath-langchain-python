"""Deterministic guardrail middleware."""

import ast
import inspect
import json
import logging
from typing import Any, Callable, Sequence, cast

from langchain.agents.middleware import AgentMiddleware, wrap_tool_call
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command
from uipath.core.guardrails import (
    GuardrailValidationResult,
    GuardrailValidationResultType,
)

from ..enums import GuardrailExecutionStage
from ..models import GuardrailAction
from ._utils import (
    create_modified_tool_request,
    create_modified_tool_result,
    sanitize_tool_name,
)

logger = logging.getLogger(__name__)

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
            tools=[analyze_joke_syntax],
            rules=[
                lambda input_data: "forbidden" in input_data.get("joke", "").lower(),
                lambda input_data: len(input_data.get("joke", "")) > 1000,
            ],
            action=BlockAction(),
            stage=GuardrailExecutionStage.PRE,
            name="Joke Content Validator",
        )

        # Empty rules means always apply action (transformation)
        always_apply_guardrail = UiPathDeterministicGuardrailMiddleware(
            tools=[analyze_joke_syntax],
            rules=[],
            action=CustomFilterAction(...),
            stage=GuardrailExecutionStage.POST,
        )

        agent = create_agent(
            model=llm,
            tools=[analyze_joke_syntax],
            middleware=[*deterministic_guardrail],
        )
        ```

    Args:
        tools: List of tool names or tool objects to apply guardrail to.
            Can be a mix of strings (tool names) or BaseTool objects.
        rules: List of callable functions that receive tool input/output data.
            - Functions with 1 parameter: `Callable[[dict[str, Any]], bool]` for input-only validation
            - Functions with 2 parameters: `Callable[[dict[str, Any], dict[str, Any]], bool]` for input+output validation
            - Functions return `True` if violation detected, `False` otherwise
            - Empty list `[]` means always apply action (no validation, just transformation)
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
        tools: Sequence[str | BaseTool],
        rules: Sequence[RuleFunction],
        action: GuardrailAction,
        stage: GuardrailExecutionStage,
        *,
        name: str = "Deterministic Guardrail",
        description: str | None = None,
    ):
        """Initialize deterministic guardrail middleware."""
        if not tools:
            raise ValueError("At least one tool name must be specified")
        if not isinstance(action, GuardrailAction):
            raise ValueError("action must be an instance of GuardrailAction")
        if not isinstance(stage, GuardrailExecutionStage):
            raise ValueError(
                f"stage must be an instance of GuardrailExecutionStage, got {type(stage)}"
            )

        for i, rule in enumerate(rules):
            if not callable(rule):
                raise ValueError(f"Rule {i + 1} must be callable, got {type(rule)}")
            sig = inspect.signature(rule)
            param_count = len(sig.parameters)
            if param_count not in (1, 2):
                raise ValueError(
                    f"Rule {i + 1} must have 1 or 2 parameters, got {param_count}"
                )

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
        self._tool_names = set(tool_name_list)

        self.rules = list(rules)
        self.action = action
        self._stage = stage
        self._name = name
        self._description = description or "Deterministic guardrail with custom rules"

        self._middleware_instances = self._create_middleware_instances()

    def _create_middleware_instances(self) -> list[AgentMiddleware]:
        """Create middleware instances from decorated functions."""
        instances = []
        middleware_instance = self
        guardrail_name = self._name.replace(" ", "_")

        async def _wrap_tool_call_func(
            request: ToolCallRequest,
            handler: Any,
        ) -> ToolMessage | Command[Any]:
            tool_call = request.tool_call
            tool_name = tool_call.get("name", "")
            sanitized_tool_name = sanitize_tool_name(tool_name)

            if sanitized_tool_name not in middleware_instance._tool_names:
                return await handler(request)

            input_data = middleware_instance._extract_tool_input_data(request)

            if middleware_instance._stage in (
                GuardrailExecutionStage.PRE,
                GuardrailExecutionStage.PRE_AND_POST,
            ):
                result = middleware_instance._evaluate_and_apply_modifications(
                    GuardrailExecutionStage.PRE,
                    middleware_instance.rules,
                    input_data,
                    None,
                    input_data,
                    request=request,
                )
                modified_request, modified_input_data, _ = result
                if modified_request is not None:
                    request = modified_request
                if modified_input_data is not None:
                    input_data = modified_input_data

            tool_result = await handler(request)

            if middleware_instance._stage in (
                GuardrailExecutionStage.POST,
                GuardrailExecutionStage.PRE_AND_POST,
            ):
                output_data = middleware_instance._extract_tool_output_data(tool_result)
                _, _, tool_result = (
                    middleware_instance._evaluate_and_apply_modifications(
                        GuardrailExecutionStage.POST,
                        middleware_instance.rules,
                        input_data,
                        output_data,
                        output_data,
                        tool_result=tool_result,
                    )
                )

            return tool_result

        _wrap_tool_call_func.__name__ = f"{guardrail_name}_wrap_tool_call"
        _wrap_tool_call = wrap_tool_call(_wrap_tool_call_func)  # type: ignore[call-overload]
        instances.append(_wrap_tool_call)

        return instances

    def __iter__(self):
        """Make the class iterable to return middleware instances."""
        return iter(self._middleware_instances)

    def _evaluate_and_apply_modifications(
        self,
        stage: GuardrailExecutionStage,
        rules: Sequence[RuleFunction],
        input_data: dict[str, Any] | None,
        output_data: dict[str, Any] | None,
        data_for_action: dict[str, Any] | str,
        request: ToolCallRequest | None = None,
        tool_result: ToolMessage | Command[Any] | None = None,
    ) -> tuple[
        ToolCallRequest | None, dict[str, Any] | None, ToolMessage | Command[Any] | None
    ]:
        """Evaluate rules and apply modifications if validation fails."""
        if not rules:
            result = GuardrailValidationResult(
                result=GuardrailValidationResultType.VALIDATION_FAILED,
                reason="Empty rules - always apply action",
            )
        else:
            result = self._evaluate_rules(rules, stage, input_data, output_data)
            if result.result != GuardrailValidationResultType.VALIDATION_FAILED:
                return request, input_data, tool_result

        modified_data = self._handle_validation_result(result, data_for_action)

        if stage == GuardrailExecutionStage.PRE:
            if modified_data is not None and isinstance(modified_data, dict):
                if request is None:
                    return request, input_data, tool_result
                modified_request = create_modified_tool_request(request, modified_data)
                return modified_request, modified_data, tool_result
        else:
            if modified_data is not None:
                if tool_result is None:
                    return request, input_data, tool_result
                modified_result = create_modified_tool_result(
                    tool_result, modified_data
                )
                return request, input_data, modified_result

        return request, input_data, tool_result

    def _evaluate_rules(
        self,
        rules: Sequence[RuleFunction],
        stage: GuardrailExecutionStage,
        input_data: dict[str, Any] | None,
        output_data: dict[str, Any] | None,
    ) -> GuardrailValidationResult:
        """Evaluate all rules and return validation result."""
        if not rules:
            return GuardrailValidationResult(
                result=GuardrailValidationResultType.VALIDATION_FAILED,
                reason="Empty rules - always apply action",
            )

        violations = []
        passed_rules = []
        evaluated_count = 0

        for i, rule in enumerate(rules):
            try:
                if stage == GuardrailExecutionStage.PRE:
                    if input_data is None:
                        continue
                    sig = inspect.signature(rule)
                    param_count = len(sig.parameters)
                    if param_count == 1:
                        rule_1arg = cast(Callable[[dict[str, Any]], bool], rule)
                        violation = rule_1arg(input_data)
                    else:
                        continue
                    evaluated_count += 1
                else:
                    if output_data is None:
                        continue
                    sig = inspect.signature(rule)
                    param_count = len(sig.parameters)
                    if param_count == 2 and input_data is not None:
                        rule_2arg = cast(
                            Callable[[dict[str, Any], dict[str, Any]], bool], rule
                        )
                        violation = rule_2arg(input_data, output_data)
                    elif param_count == 1:
                        rule_1arg = cast(Callable[[dict[str, Any]], bool], rule)
                        violation = rule_1arg(output_data)
                    else:
                        continue
                    evaluated_count += 1

                if violation:
                    violations.append(f"Rule {i + 1} detected violation")
                else:
                    passed_rules.append(f"Rule {i + 1}")
            except Exception as e:
                logger.error(f"Error in rule function {i + 1}: {e}", exc_info=True)
                violations.append(f"Rule {i + 1} raised exception: {str(e)}")
                evaluated_count += 1

        if evaluated_count == 0:
            return GuardrailValidationResult(
                result=GuardrailValidationResultType.PASSED,
                reason="No applicable rules to evaluate",
            )

        if passed_rules:
            return GuardrailValidationResult(
                result=GuardrailValidationResultType.PASSED,
                reason=f"Rules passed: {', '.join(passed_rules)}",
            )

        return GuardrailValidationResult(
            result=GuardrailValidationResultType.VALIDATION_FAILED,
            reason="; ".join(violations),
        )

    def _extract_tool_input_data(self, request: ToolCallRequest) -> dict[str, Any]:
        """Extract tool input data from ToolCallRequest for rule evaluation."""
        tool_call = request.tool_call
        args = tool_call.get("args", {})
        if isinstance(args, dict):
            return args
        return {"args": args}

    def _extract_tool_output_data(
        self, result: ToolMessage | Command[Any]
    ) -> dict[str, Any]:
        """Extract tool output data from handler result."""
        if isinstance(result, Command):
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
        """Handle guardrail validation result."""
        if result.result == GuardrailValidationResultType.VALIDATION_FAILED:
            return self.action.handle_validation_result(result, input_data, self._name)
        return None
