"""Base class for built-in UiPath guardrail middlewares."""

import ast
import asyncio
import json
import logging
from typing import Any

from langchain.agents.middleware import AgentMiddleware, wrap_tool_call
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command
from uipath.core.guardrails import (
    GuardrailValidationResult,
    GuardrailValidationResultType,
)
from uipath.platform import UiPath
from uipath.platform.guardrails import BuiltInValidatorGuardrail
from uipath.platform.guardrails.decorators import GuardrailExecutionStage
from uipath.platform.guardrails.decorators._exceptions import GuardrailBlockException

from uipath_langchain.agent.exceptions import AgentRuntimeError

from ..models import GuardrailAction
from ._utils import (
    convert_block_exception,
    create_modified_tool_request,
    create_modified_tool_result,
    extract_text_from_messages,
    sanitize_tool_name,
)

logger = logging.getLogger(__name__)


class BuiltInGuardrailMiddlewareMixin:
    """Mixin providing shared evaluation logic for built-in guardrail middlewares.

    Subclasses must set:
        _guardrail (BuiltInValidatorGuardrail): The guardrail configuration.
        _name (str): The guardrail name used in log messages.
        action (GuardrailAction): The action to take on violation.
        _tool_names (list[str] | None): Tool names to intercept; None skips all tools.
        _tool_stage (GuardrailExecutionStage): PRE, POST, or PRE_AND_POST evaluation.
    """

    _guardrail: BuiltInValidatorGuardrail
    _name: str
    action: GuardrailAction
    _tool_names: list[str] | None = None
    _tool_stage: GuardrailExecutionStage = GuardrailExecutionStage.PRE_AND_POST
    _uipath: UiPath | None = None

    def _get_uipath(self) -> UiPath:
        """Get or create UiPath instance."""
        if self._uipath is None:
            self._uipath = UiPath()
        return self._uipath

    def _evaluate_guardrail(
        self, input_data: str | dict[str, Any]
    ) -> GuardrailValidationResult:
        """Evaluate the guardrail against input data via the UiPath API."""
        uipath = self._get_uipath()
        return uipath.guardrails.evaluate_guardrail(input_data, self._guardrail)

    def _handle_validation_result(
        self, result: GuardrailValidationResult, input_data: str | dict[str, Any]
    ) -> str | dict[str, Any] | None:
        """Delegate to the action when a violation is detected."""
        if result.result == GuardrailValidationResultType.VALIDATION_FAILED:
            return self.action.handle_validation_result(result, input_data, self._name)
        return None

    def _extract_tool_output_data(
        self, result: ToolMessage | Command[Any]
    ) -> dict[str, Any]:
        """Extract tool output data from handler result for POST-stage evaluation."""
        if isinstance(result, Command):
            update = result.update if hasattr(result, "update") else {}
            messages = update.get("messages", []) if isinstance(update, dict) else []
            tool_msg = next((m for m in messages if isinstance(m, ToolMessage)), None)
            content = tool_msg.content if tool_msg is not None else None
        elif isinstance(result, ToolMessage):
            content = result.content
        else:
            return {}

        if content is None:
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
        return {"output": content}

    def _extract_tool_input_data(
        self, request: ToolCallRequest
    ) -> str | dict[str, Any]:
        """Extract tool input data from ToolCallRequest for guardrail evaluation."""
        tool_call = request.tool_call
        args = tool_call.get("args", {})
        if isinstance(args, dict):
            return args
        return str(args)

    async def _run_tool_guardrail(
        self, request: ToolCallRequest, handler: Any
    ) -> ToolMessage | Command[Any]:
        """Execute PRE guardrail check, run tool handler, then POST guardrail check."""
        tool_call = request.tool_call
        tool_name = tool_call.get("name", "")
        sanitized_tool_name = sanitize_tool_name(tool_name)

        if self._tool_names is None or sanitized_tool_name not in self._tool_names:
            return await handler(request)

        if self._tool_stage in (
            GuardrailExecutionStage.PRE,
            GuardrailExecutionStage.PRE_AND_POST,
        ):
            input_data = self._extract_tool_input_data(request)
            try:
                result = await asyncio.to_thread(self._evaluate_guardrail, input_data)
                modified_input = self._handle_validation_result(result, input_data)
                if modified_input is not None:
                    request = create_modified_tool_request(request, modified_input)
            except GuardrailBlockException as exc:
                raise convert_block_exception(exc) from exc
            except AgentRuntimeError:
                raise
            except Exception as e:
                logger.error(
                    f"Error evaluating '{self._name}' guardrail (PRE)"
                    f" for tool '{tool_name}': {e}",
                    exc_info=True,
                )

        tool_result = await handler(request)

        if self._tool_stage in (
            GuardrailExecutionStage.POST,
            GuardrailExecutionStage.PRE_AND_POST,
        ):
            output_data = self._extract_tool_output_data(tool_result)
            if output_data:
                try:
                    result = await asyncio.to_thread(
                        self._evaluate_guardrail, output_data
                    )
                    modified_output = self._handle_validation_result(
                        result, output_data
                    )
                    if modified_output is not None:
                        tool_result = create_modified_tool_result(
                            tool_result, modified_output
                        )
                except GuardrailBlockException as exc:
                    raise convert_block_exception(exc) from exc
                except AgentRuntimeError:
                    raise
                except Exception as e:
                    logger.error(
                        f"Error evaluating '{self._name}' guardrail (POST)"
                        f" for tool '{tool_name}': {e}",
                        exc_info=True,
                    )

        return tool_result

    def _create_tool_wrap_hook(self, guardrail_name: str) -> AgentMiddleware:
        """Create a wrap_tool_call hook that delegates to _run_tool_guardrail."""
        middleware_instance = self

        async def _wrap_tool_call_func(
            request: ToolCallRequest,
            handler: Any,
        ) -> ToolMessage | Command[Any]:
            return await middleware_instance._run_tool_guardrail(request, handler)

        _wrap_tool_call_func.__name__ = f"{guardrail_name}_wrap_tool_call"
        return wrap_tool_call(_wrap_tool_call_func)  # type: ignore[call-overload]

    def _check_messages(self, messages: list[BaseMessage]) -> None:
        """Evaluate guardrail against message text; apply action on violation."""
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
        except GuardrailBlockException as exc:
            raise convert_block_exception(exc) from exc
        except AgentRuntimeError:
            raise
        except Exception as e:
            logger.error(
                f"Error evaluating guardrail '{self._name}': {e}", exc_info=True
            )
