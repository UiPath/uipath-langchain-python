"""Base class for built-in UiPath guardrail middlewares."""

import ast
import asyncio
import json
import logging
from typing import Any, Sequence

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
from langgraph.errors import GraphBubbleUp
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.runtime import Runtime
from langgraph.types import Command
from uipath.core.guardrails import (
    GuardrailScope,
    GuardrailValidationResult,
    GuardrailValidationResultType,
)
from uipath.platform import UiPath
from uipath.platform.guardrails import BuiltInValidatorGuardrail
from uipath.platform.guardrails.decorators import GuardrailExecutionStage
from uipath.platform.guardrails.decorators._exceptions import GuardrailBlockException

from uipath_langchain.agent.exceptions import AgentRuntimeError

from .._action_context import (
    GuardrailActionContext,
    _action_context,
    component_label,
)
from ..models import GuardrailAction
from ._utils import (
    convert_block_exception,
    create_modified_tool_request,
    create_modified_tool_result,
    extract_text_from_messages,
    sanitize_tool_name,
)

logger = logging.getLogger(__name__)


def _get_tool_message_content(result: ToolMessage | Command[Any]) -> Any:
    """Return the raw content from a ToolMessage or the first ToolMessage in a Command."""
    if isinstance(result, ToolMessage):
        return result.content
    if isinstance(result, Command):
        update = result.update if hasattr(result, "update") else {}
        messages = update.get("messages", []) if isinstance(update, dict) else []
        tool_msg = next((m for m in messages if isinstance(m, ToolMessage)), None)
        return tool_msg.content if tool_msg is not None else None
    return None


def _parse_str_content(content: str) -> dict[str, Any]:
    """Parse a string into a dict, trying JSON then ast.literal_eval."""
    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else {"output": parsed}
    except json.JSONDecodeError:
        pass
    try:
        parsed = ast.literal_eval(content)
        return parsed if isinstance(parsed, dict) else {"output": parsed}
    except (ValueError, SyntaxError):
        return {"output": content}


def _apply_text_modification(
    messages: list[BaseMessage],
    original: str,
    modified: str | dict[str, Any] | None,
) -> None:
    """Substitute ``original`` with ``modified`` in the first matching message.

    No-op unless ``modified`` is a string that differs from ``original``.
    """
    if not isinstance(modified, str) or modified == original:
        return
    for msg in messages:
        if (
            isinstance(msg, (HumanMessage, AIMessage))
            and isinstance(msg.content, str)
            and original in msg.content
        ):
            msg.content = msg.content.replace(original, modified, 1)
            return


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
    scopes: Sequence[GuardrailScope]
    _tool_names: list[str] | None = None
    _tool_stage: GuardrailExecutionStage = GuardrailExecutionStage.PRE_AND_POST
    _uipath: UiPath | None = None

    def _resolve_tool_names(
        self, tools: Sequence[str | BaseTool] | None
    ) -> list[str] | None:
        """Normalize a mix of tool names / ``BaseTool`` objects to sanitized names.

        Shared by the multi-scope built-in middlewares so their tool handling can't
        drift apart. Returns ``None`` when no tools are given.
        """
        if tools is None:
            return None
        names: list[str] = []
        for tool_or_name in tools:
            if isinstance(tool_or_name, BaseTool):
                names.append(sanitize_tool_name(tool_or_name.name))
            elif isinstance(tool_or_name, str):
                names.append(sanitize_tool_name(tool_or_name))
            else:
                raise ValueError(
                    f"tools must contain strings or BaseTool objects, got {type(tool_or_name)}"
                )
        return names

    def _require_tools_for_tool_scope(self, scopes: Sequence[GuardrailScope]) -> None:
        """Ensure a TOOL-scoped guardrail specifies at least one tool."""
        if GuardrailScope.TOOL in scopes and not self._tool_names:
            raise ValueError(
                "Tool scope is specified but tools is None or empty. "
                "Tool scope guardrails require at least one tool to be specified."
            )

    def _build_scope_instances(self, guardrail_name: str) -> list["AgentMiddleware"]:
        """Build hooks for each configured scope: AGENT/LLM message hooks + TOOL wrap.

        Shared by the multi-scope built-in middlewares (PII / harmful content /
        LLM-as-judge) so their scope wiring stays consistent.
        """
        instances: list[AgentMiddleware] = []
        if GuardrailScope.AGENT in self.scopes:
            instances.extend(
                self._build_message_hooks(
                    GuardrailScope.AGENT, self._tool_stage, guardrail_name
                )
            )
        if GuardrailScope.LLM in self.scopes:
            instances.extend(
                self._build_message_hooks(
                    GuardrailScope.LLM, self._tool_stage, guardrail_name
                )
            )
        if GuardrailScope.TOOL in self.scopes:
            instances.append(self._create_tool_wrap_hook(guardrail_name))
        return instances

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
        self,
        result: GuardrailValidationResult,
        input_data: str | dict[str, Any],
        *,
        scope: GuardrailScope | None = None,
        stage: GuardrailExecutionStage | None = None,
        component: str | None = None,
        input_payload: str | None = None,
    ) -> str | dict[str, Any] | None:
        """Delegate to the action when a violation is detected.

        Publishes the guardrail context (scope / stage / component / the
        guardrail's description, plus the original ``input_payload`` on a POST
        output check) for the duration of the action call so context-aware
        actions (e.g. ``EscalateAction``) can read it instead of requiring it to
        be hardcoded.
        """
        if result.result != GuardrailValidationResultType.VALIDATION_FAILED:
            return None
        token = _action_context.set(
            GuardrailActionContext(
                scope=scope,
                execution_stage=stage,
                component=component,
                description=getattr(self._guardrail, "description", None),
                input_payload=input_payload,
            )
        )
        try:
            return self.action.handle_validation_result(result, input_data, self._name)
        finally:
            _action_context.reset(token)

    def _extract_tool_output_data(
        self, result: ToolMessage | Command[Any]
    ) -> dict[str, Any]:
        """Extract tool output data from handler result for POST-stage evaluation."""
        content = _get_tool_message_content(result)
        if content is None:
            return {}
        if isinstance(content, dict):
            return content
        if isinstance(content, str):
            return _parse_str_content(content)
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
                modified_input = self._handle_validation_result(
                    result,
                    input_data,
                    scope=GuardrailScope.TOOL,
                    stage=GuardrailExecutionStage.PRE,
                    component=tool_name,
                )
                if modified_input is not None:
                    request = create_modified_tool_request(request, modified_input)
            except GuardrailBlockException as exc:
                raise convert_block_exception(exc) from exc
            except AgentRuntimeError:
                raise
            except GraphBubbleUp:
                # LangGraph control-flow signals (e.g. interrupt() from an
                # escalation action). Must bubble up so the run can suspend.
                raise
            except Exception:
                logger.exception(
                    f"Error evaluating '{self._name}' guardrail (PRE)"
                    f" for tool '{tool_name}'"
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
                        result,
                        output_data,
                        scope=GuardrailScope.TOOL,
                        stage=GuardrailExecutionStage.POST,
                        component=tool_name,
                        input_payload=json.dumps(
                            self._extract_tool_input_data(request)
                        ),
                    )
                    if modified_output is not None:
                        tool_result = create_modified_tool_result(
                            tool_result, modified_output
                        )
                except GuardrailBlockException as exc:
                    raise convert_block_exception(exc) from exc
                except AgentRuntimeError:
                    raise
                except GraphBubbleUp:
                    # LangGraph control-flow signals (e.g. interrupt() from an
                    # escalation action). Must bubble up so the run can suspend.
                    raise
                except Exception:
                    logger.exception(
                        f"Error evaluating '{self._name}' guardrail (POST)"
                        f" for tool '{tool_name}'"
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
        return wrap_tool_call(_wrap_tool_call_func)

    def _build_message_hooks(
        self,
        scope: GuardrailScope,
        stage: GuardrailExecutionStage,
        guardrail_name: str,
    ) -> list[AgentMiddleware]:
        """Build stage-gated before/after message hooks for an AGENT or LLM scope.

        ``PRE`` registers only the ``before_*`` hook, ``POST`` only the
        ``after_*`` hook, and ``PRE_AND_POST`` both — so a guardrail validates
        (and acts, e.g. escalates) at a single checkpoint instead of twice per
        run. Shared by the message-based middlewares (PII / harmful content /
        intellectual property) so their hook wiring can't drift apart.
        """
        include_pre = stage in (
            GuardrailExecutionStage.PRE,
            GuardrailExecutionStage.PRE_AND_POST,
        )
        include_post = stage in (
            GuardrailExecutionStage.POST,
            GuardrailExecutionStage.PRE_AND_POST,
        )
        mw = self
        hooks: list[AgentMiddleware] = []

        if scope == GuardrailScope.AGENT:
            if include_pre:

                async def _before_agent_func(
                    state: AgentState[Any], runtime: Runtime
                ) -> None:
                    messages = state.get("messages", [])
                    mw._check_messages(
                        list(messages),
                        scope=GuardrailScope.AGENT,
                        stage=GuardrailExecutionStage.PRE,
                    )

                _before_agent_func.__name__ = f"{guardrail_name}_before_agent"
                hooks.append(before_agent(_before_agent_func))

            if include_post:

                async def _after_agent_func(
                    state: AgentState[Any], runtime: Runtime
                ) -> None:
                    # POST validates the agent's OUTPUT — the final AI message —
                    # not the whole conversation, so the flagged content maps back
                    # to a single message (an escalation's ReviewedOutputs edit can
                    # be applied) and the original input is carried separately as
                    # input_text. Mirrors the LLM-scope after_model behavior.
                    messages = state.get("messages", [])
                    ai_messages = [m for m in messages if isinstance(m, AIMessage)]
                    if ai_messages:
                        mw._check_messages(
                            [ai_messages[-1]],
                            scope=GuardrailScope.AGENT,
                            stage=GuardrailExecutionStage.POST,
                            input_text=mw._last_input_text(messages),
                        )

                _after_agent_func.__name__ = f"{guardrail_name}_after_agent"
                hooks.append(after_agent(_after_agent_func))

        elif scope == GuardrailScope.LLM:
            if include_pre:

                async def _before_model_func(
                    state: AgentState[Any], runtime: Runtime
                ) -> None:
                    messages = state.get("messages", [])
                    mw._check_messages(
                        list(messages),
                        scope=GuardrailScope.LLM,
                        stage=GuardrailExecutionStage.PRE,
                    )

                _before_model_func.__name__ = f"{guardrail_name}_before_model"
                hooks.append(before_model(_before_model_func))

            if include_post:

                async def _after_model_func(
                    state: AgentState[Any], runtime: Runtime
                ) -> None:
                    messages = state.get("messages", [])
                    ai_messages = [m for m in messages if isinstance(m, AIMessage)]
                    if ai_messages:
                        mw._check_messages(
                            [ai_messages[-1]],
                            scope=GuardrailScope.LLM,
                            stage=GuardrailExecutionStage.POST,
                            input_text=mw._last_input_text(messages),
                        )

                _after_model_func.__name__ = f"{guardrail_name}_after_model"
                hooks.append(after_model(_after_model_func))

        return hooks

    def _last_input_text(self, messages: Sequence[BaseMessage]) -> str | None:
        """Return the last HumanMessage text — the input for a POST check.

        Used by ``after_*`` hooks to supply the original input alongside the
        flagged output when escalating an output (POST) violation.
        """
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                return extract_text_from_messages([msg]) or None
        return None

    def _check_messages(
        self,
        messages: list[BaseMessage],
        scope: GuardrailScope | None = None,
        stage: GuardrailExecutionStage | None = None,
        input_text: str | None = None,
    ) -> None:
        """Evaluate guardrail against message text; apply action on violation.

        ``input_text`` is the original input for a POST (output) check — the
        message that produced the flagged output — so an escalation can show it
        as ``Inputs`` alongside the flagged ``Outputs``.
        """
        if not messages:
            return

        text = extract_text_from_messages(messages)
        if not text:
            return

        try:
            result = self._evaluate_guardrail(text)
            modified_text = self._handle_validation_result(
                result,
                text,
                scope=scope,
                stage=stage,
                component=component_label(scope),
                input_payload=json.dumps(input_text) if input_text else None,
            )
            _apply_text_modification(messages, text, modified_text)
        except GuardrailBlockException as exc:
            raise convert_block_exception(exc) from exc
        except AgentRuntimeError:
            raise
        except GraphBubbleUp:
            # LangGraph control-flow signals (e.g. interrupt() from an
            # escalation action). Must bubble up so the run can suspend.
            raise
        except Exception:
            logger.exception(f"Error evaluating guardrail '{self._name}'")
