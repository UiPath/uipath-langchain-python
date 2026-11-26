"""Base class for built-in UiPath guardrail middlewares."""

import ast
import asyncio
import json
import logging
import threading
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Generator, Mapping
from contextlib import contextmanager
from typing import Annotated, Any, Literal, NotRequired, Sequence, cast
from uuid import uuid4

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ExtendedModelResponse,
    ModelRequest,
    ModelResponse,
    after_agent,
    before_agent,
    wrap_tool_call,
)
from langchain.agents.middleware.types import PrivateStateAttr
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.config import get_config
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
from uipath.platform.errors import EnrichedException
from uipath.platform.guardrails import (
    DETECTION_SCOPE_PARAMETER_ID,
    BuiltInValidatorGuardrail,
    DetectionScope,
    GuardrailAttachment,
    detection_scope_includes_files,
    referenced_attachments,
)
from uipath.platform.guardrails.decorators import (
    GuardrailExecutionStage,
    LogAction,
)
from uipath.platform.guardrails.decorators._exceptions import GuardrailBlockException

from uipath_langchain.agent.exceptions import (
    AgentRuntimeError,
    centralized_guardrail_error,
)

from .._action_context import (
    GuardrailActionContext,
    _action_context,
    component_label,
)
from ..models import GuardrailAction
from ._attachments import (
    NO_TEXT_PLACEHOLDER,
    build_registry,
    output_references,
    registry_references,
    tool_result_references,
)
from ._enforcement import EnforcementContext
from ._grouping import (
    guardrail_stage,
    guardrails_in_use,
    register_guardrail_node,
)
from ._spans import guardrail_action_span
from ._utils import (
    convert_block_exception,
    create_modified_tool_request,
    create_modified_tool_result,
    extract_text_from_messages,
    sanitize_tool_name,
)

logger = logging.getLogger(__name__)


_OUTPUT_ACTIONS_KEY = "uipath_guardrail_output_actions"


def _merge_output_actions(
    current: dict[str, Any] | None, update: dict[str, Any] | None
) -> dict[str, Any]:
    """Merge pending output actions by hook name; a ``None`` entry removes one."""
    merged = {**(current or {}), **(update or {})}
    return {name: action for name, action in merged.items() if action is not None}


class _OutputActionState(AgentState[Any]):
    """Agent state with the output actions waiting for their node."""

    uipath_guardrail_output_actions: NotRequired[
        Annotated[dict[str, Any], PrivateStateAttr, _merge_output_actions]
    ]


ModelHandler = Callable[[ModelRequest], Awaitable[Any]]


class _PendingAnswers:
    """Message ids of answers with an action left for later: outer checks wait.

    By message id, which copies of the reply keep. Not a ContextVar: LangChain
    runs each hook in a copied context. Bounded, in case a node never runs.
    """

    def __init__(self, size: int = 1024) -> None:
        self._ids: OrderedDict[str, None] = OrderedDict()
        self._size = size
        self._lock = threading.Lock()

    def add(self, message_id: str) -> None:
        with self._lock:
            self._ids[message_id] = None
            self._ids.move_to_end(message_id)
            while len(self._ids) > self._size:
                self._ids.popitem(last=False)

    def discard(self, message_id: str) -> None:
        with self._lock:
            self._ids.pop(message_id, None)

    def __contains__(self, message_id: object) -> bool:
        with self._lock:
            return message_id in self._ids


_pending_answers = _PendingAnswers()


def _pending_key(message_id: str) -> str:
    """``message_id`` within the run's thread, so other runs don't share it."""
    try:
        thread = get_config().get("configurable", {}).get("thread_id")
    except RuntimeError:
        thread = None
    return f"{thread}:{message_id}"


_HOOK_STAGES = {
    (True, True): "pre_and_post",
    (True, False): "pre",
    (False, True): "post",
}


StateHook = Callable[[Mapping[str, Any]], dict[str, Any] | None]


class _ModelGuardrailHook(AgentMiddleware):
    """One guardrail's checks around the model call."""

    def __init__(
        self,
        name: str,
        wrap: Callable[[ModelRequest, ModelHandler], Awaitable[Any]],
        before: StateHook | None = None,
        after: StateHook | None = None,
    ) -> None:
        super().__init__()
        self._hook_name = name
        self._wrap = wrap
        self._before = before
        self._after = after

    @property
    def name(self) -> str:
        """The hook's name, unique per guardrail and stage."""
        return self._hook_name

    async def awrap_model_call(self, request: ModelRequest, handler: Any) -> Any:
        """Check the model's input and output around ``handler``."""
        return await self._wrap(request, handler)


class _WithInputCheck(_ModelGuardrailHook):
    """An input check whose action may pause the run, in the node before the model."""

    async def abefore_model(
        self, state: AgentState[Any], runtime: Runtime
    ) -> dict[str, Any] | None:
        """Check the input; resuming a paused action re-runs only this node."""
        assert self._before is not None
        return self._before(state)


class _WithOutputAction(_ModelGuardrailHook):
    """Output actions that may pause the run, in the node after the model."""

    state_schema = _OutputActionState

    async def aafter_model(
        self, state: AgentState[Any], runtime: Runtime
    ) -> dict[str, Any] | None:
        """Run the action the output check left for this node, if any."""
        assert self._after is not None
        return self._after(state)


class _WithInputCheckAndOutputAction(_WithInputCheck, _WithOutputAction):
    """Both of the above."""


def _with_output_action(
    response: Any, hook_name: str, action: dict[str, Any] | None
) -> ExtendedModelResponse[Any]:
    """``response`` with ``hook_name``'s pending output action set (``None`` clears)."""
    if not isinstance(response, ModelResponse):
        response = ModelResponse(result=[response])
    return ExtendedModelResponse(
        model_response=response,
        command=Command(update={_OUTPUT_ACTIONS_KEY: {hook_name: action}}),
    )


def _last_ai_message(response: Any) -> AIMessage | None:
    """The model's answer in a ``wrap_model_call`` handler's response."""
    if isinstance(response, AIMessage):
        return response
    messages = getattr(response, "result", None) or []
    return next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)


#: ``$guardrailType`` discriminator for built-in validator guardrails. Typed as
#: a ``Literal`` so it satisfies ``BuiltInValidatorGuardrail.guardrail_type``.
BUILT_IN_VALIDATOR_GUARDRAIL_TYPE: Literal["builtInValidator"] = "builtInValidator"

_DETECTION_SCOPE_VALUES = frozenset(member.value for member in DetectionScope)


def _declares_detection_scope(guardrail: BuiltInValidatorGuardrail) -> bool:
    try:
        return any(
            parameter.id.lower() == DETECTION_SCOPE_PARAMETER_ID.lower()
            and isinstance(parameter.value, str)
            and bool(parameter.value.strip())
            for parameter in guardrail.validator_parameters
        )
    except Exception:
        return False


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

    Subclasses may set:
        detection_scope (DetectionScope | None): Which content the guardrail
            evaluates, set through ``_resolve_detection_scope``; None sends no
            ``detectionScope`` parameter.
    """

    _guardrail: BuiltInValidatorGuardrail
    _name: str
    action: GuardrailAction
    scopes: Sequence[GuardrailScope]
    _tool_names: list[str] | None = None
    _tool_stage: GuardrailExecutionStage = GuardrailExecutionStage.PRE_AND_POST
    _uipath: UiPath | None = None
    detection_scope: DetectionScope | None = None

    @staticmethod
    def _resolve_detection_scope(
        value: DetectionScope | str | None,
    ) -> DetectionScope | None:
        """Validate a ``detection_scope`` constructor argument.

        Raises:
            ValueError: When ``value`` is neither None, a ``DetectionScope`` member nor
                one of its exact wire values.
        """
        if value is None or isinstance(value, DetectionScope):
            return value
        if isinstance(value, str) and value in _DETECTION_SCOPE_VALUES:
            return DetectionScope(value)
        raise ValueError(
            "detection_scope must be a DetectionScope member or one of 'Text', 'Files', 'Both'"
        )

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

    def _enforcement(
        self,
        scope: GuardrailScope | None,
        stage: GuardrailExecutionStage | None,
    ) -> EnforcementContext:
        """Where a check of this guardrail runs and what it does on a violation."""
        return EnforcementContext(
            scope=scope,
            stage=stage,
            action=self.action,
            guardrail_name=self._name,
            guardrail_type=self._guardrail.validator_type,
        )

    def _evaluate_guardrail(
        self,
        input_data: str | dict[str, Any],
        attachments: list[GuardrailAttachment] | None = None,
        *,
        enforcement: EnforcementContext | None = None,
    ) -> GuardrailValidationResult:
        """Evaluate the guardrail against input data via the UiPath API.

        ``attachments`` is sent when there is at least one and the scope includes
        files. An empty list, meaning files apply and there are none, is sent only
        when the guardrail asks for files (``Files`` or ``Both``); ``None`` says
        nothing about files, so a text-only evaluation sends the same request as
        before files were supported. When the service
        rejects the attachment references (HTTP 400), the input is evaluated again
        without them, so a file never disables the guardrail. A check the
        organization's AI Trust Layer policy blocked is not retried: it stops the run.
        ``enforcement`` adds the check's ``scope``, ``execution_stage`` and
        ``action_type`` to the request.
        """
        uipath = self._get_uipath()
        options = enforcement.evaluation_options() if enforcement else {}

        def evaluate(**files: Any) -> GuardrailValidationResult:
            try:
                return uipath.guardrails.evaluate_guardrail(
                    input_data, self._guardrail, **files, **options
                )
            except EnrichedException as exc:
                if (blocked := centralized_guardrail_error(exc)) is not None:
                    raise blocked from exc
                raise

        files_apply = detection_scope_includes_files(self._guardrail)
        files_requested = files_apply and (
            self.detection_scope in (DetectionScope.FILES, DetectionScope.BOTH)
            or _declares_detection_scope(self._guardrail)
        )
        if not files_apply or attachments is None:
            return evaluate()
        if not attachments:
            return evaluate(attachments=[]) if files_requested else evaluate()
        try:
            return evaluate(attachments=attachments)
        except EnrichedException as exc:
            if exc.status_code != 400:
                raise
            logger.warning(
                "Guardrail '%s' rejected the attachment references (HTTP 400); "
                "re-evaluating without attachments.",
                self._name,
            )
            return evaluate()

    def _attachments_for(
        self,
        *,
        scope: GuardrailScope,
        stage: GuardrailExecutionStage,
        state: Mapping[str, Any] | None,
        tool_args: Any = None,
        tool_result: Any = None,
    ) -> list[GuardrailAttachment] | None:
        """Return the attachment references one evaluation forwards.

        An agent or LLM input check forwards every file the run holds; an output
        check only the held files the output names. A tool call forwards the held
        files its arguments name, and its result also the files the tool returned.
        ``None`` says nothing about files: the guardrail is scoped to text, by
        ``detection_scope`` or by a ``detectionScope`` validator parameter, or the
        state cannot be read, so the text is evaluated as before instead of failing.
        """
        if (
            self.detection_scope is DetectionScope.TEXT
            or not detection_scope_includes_files(self._guardrail)
        ):
            return None
        try:
            registry = build_registry(state)
            if scope == GuardrailScope.TOOL:
                if stage == GuardrailExecutionStage.PRE:
                    return referenced_attachments(tool_args, registry)
                return tool_result_references(tool_result, registry)
            if stage == GuardrailExecutionStage.PRE:
                return registry_references(registry)
            state = state or {}
            output = next(
                (
                    message
                    for message in reversed(state.get("messages") or [])
                    if isinstance(message, AIMessage)
                ),
                None,
            )
            if scope == GuardrailScope.AGENT:
                return output_references(
                    output,
                    registry,
                    structured_response=state.get("structured_response"),
                )
            return output_references(output, registry)
        except Exception:
            logger.debug(
                "Could not collect attachments for guardrail '%s'; evaluating "
                "without files.",
                self._name,
                exc_info=True,
            )
            return None

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
            with guardrail_action_span(
                self._name,
                self.action,
                result,
                enforcement=self._enforcement(scope, stage),
            ):
                return self.action.handle_validation_result(
                    result, input_data, self._name
                )
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
            attachments = self._attachments_for(
                scope=GuardrailScope.TOOL,
                stage=GuardrailExecutionStage.PRE,
                state=getattr(request, "state", None),
                tool_args=input_data,
            )
            try:
                with guardrail_stage(GuardrailScope.TOOL, GuardrailExecutionStage.PRE):
                    result = await asyncio.to_thread(
                        self._evaluate_guardrail,
                        input_data,
                        attachments,
                        enforcement=self._enforcement(
                            GuardrailScope.TOOL, GuardrailExecutionStage.PRE
                        ),
                    )
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
                attachments = self._attachments_for(
                    scope=GuardrailScope.TOOL,
                    stage=GuardrailExecutionStage.POST,
                    state=getattr(request, "state", None),
                    tool_result=output_data,
                )
                try:
                    with guardrail_stage(
                        GuardrailScope.TOOL, GuardrailExecutionStage.POST
                    ):
                        result = await asyncio.to_thread(
                            self._evaluate_guardrail,
                            output_data,
                            attachments,
                            enforcement=self._enforcement(
                                GuardrailScope.TOOL, GuardrailExecutionStage.POST
                            ),
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
        guardrails_in_use()
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
        """Build stage-gated message hooks for an AGENT or LLM scope.

        AGENT: ``PRE`` registers only ``before_agent``, ``POST`` only
        ``after_agent``, and ``PRE_AND_POST`` both. LLM: one model hook that
        checks the input, the output or both. Shared by the message-based
        middlewares (PII / harmful content / intellectual property).
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
                        attachments=mw._attachments_for(
                            scope=GuardrailScope.AGENT,
                            stage=GuardrailExecutionStage.PRE,
                            state=state,
                        ),
                    )

                _before_agent_func.__name__ = f"{guardrail_name}_before_agent"
                register_guardrail_node(
                    f"{_before_agent_func.__name__}.before_agent",
                    GuardrailScope.AGENT,
                    GuardrailExecutionStage.PRE,
                )
                hooks.append(before_agent(_before_agent_func))

            if include_post:

                async def _after_agent_func(
                    state: AgentState[Any], runtime: Runtime
                ) -> None:
                    # POST validates the agent's OUTPUT — the final AI message —
                    # not the whole conversation, so the flagged content maps back
                    # to a single message (an escalation's ReviewedOutputs edit can
                    # be applied) and the original input is carried separately as
                    # input_text. Mirrors the LLM-scope output check.
                    messages = state.get("messages", [])
                    ai_messages = [m for m in messages if isinstance(m, AIMessage)]
                    if ai_messages:
                        mw._check_messages(
                            [ai_messages[-1]],
                            scope=GuardrailScope.AGENT,
                            stage=GuardrailExecutionStage.POST,
                            input_text=mw._last_input_text(messages),
                            attachments=mw._attachments_for(
                                scope=GuardrailScope.AGENT,
                                stage=GuardrailExecutionStage.POST,
                                state=state,
                            ),
                        )

                _after_agent_func.__name__ = f"{guardrail_name}_after_agent"
                register_guardrail_node(
                    f"{_after_agent_func.__name__}.after_agent",
                    GuardrailScope.AGENT,
                    GuardrailExecutionStage.POST,
                )
                hooks.append(after_agent(_after_agent_func))

        elif scope == GuardrailScope.LLM:
            hooks.append(
                self._create_model_wrap_hook(guardrail_name, include_pre, include_post)
            )

        return hooks

    def _create_model_wrap_hook(
        self, guardrail_name: str, include_pre: bool, include_post: bool
    ) -> AgentMiddleware:
        """Create the hook that checks the model's input and output in the model node.

        Only the built-in Log acts there. Any other action may pause the run or
        raise, and inside the model call a retry or fallback middleware could
        catch that: its input check runs in the hook's ``before_model`` node and
        its output action in the ``after_model`` node.
        """
        guardrails_in_use()
        mw = self
        # The stage keeps same-named PRE and POST guardrails distinct.
        stage = _HOOK_STAGES[(include_pre, include_post)]
        hook_name = f"{guardrail_name}_{stage}_wrap_model_call"
        # Exactly LogAction: a subclass may pause the run.
        outside_model = type(self.action) is not LogAction
        check_before_model = include_pre and outside_model
        act_after_model = include_post and outside_model

        async def _wrap_model_call_func(request: ModelRequest, handler: Any) -> Any:
            if include_pre and not check_before_model:
                mw._check_model_input(request)
            response = await handler(request)
            if not include_post:
                return response
            return mw._check_model_output(
                request, response, hook_name, act_after_model=act_after_model
            )

        before: StateHook | None = None
        after: StateHook | None = None
        if check_before_model:
            register_guardrail_node(
                f"{hook_name}.before_model",
                GuardrailScope.LLM,
                GuardrailExecutionStage.PRE,
            )
            before = mw._run_input_check
        if include_post:
            register_guardrail_node(
                f"{hook_name}.after_model",
                GuardrailScope.LLM,
                GuardrailExecutionStage.POST,
            )

            def after(state: Mapping[str, Any]) -> dict[str, Any] | None:
                return mw._run_output_action(state, hook_name)

        hook_class = {
            (False, False): _ModelGuardrailHook,
            (True, False): _WithInputCheck,
            (False, True): _WithOutputAction,
            (True, True): _WithInputCheckAndOutputAction,
        }[(before is not None, after is not None)]
        return hook_class(hook_name, _wrap_model_call_func, before, after)

    def _check_model_input(self, request: ModelRequest) -> None:
        """Check the messages the model gets (Log only: no edits, no raise)."""
        self._check_messages(
            cast(list[BaseMessage], request.messages),
            scope=GuardrailScope.LLM,
            stage=GuardrailExecutionStage.PRE,
            attachments=self._attachments_for(
                scope=GuardrailScope.LLM,
                stage=GuardrailExecutionStage.PRE,
                state=request.state,
            ),
        )

    def _run_input_check(self, state: Mapping[str, Any]) -> dict[str, Any] | None:
        """Check the saved messages before the model; return the edited ones."""
        messages = list(state.get("messages", []))
        checked = [message.model_copy() for message in messages]
        self._check_messages(
            checked,
            scope=GuardrailScope.LLM,
            stage=GuardrailExecutionStage.PRE,
            attachments=self._attachments_for(
                scope=GuardrailScope.LLM,
                stage=GuardrailExecutionStage.PRE,
                state=state,
            ),
            beside_node=True,
        )
        edited = [
            copy
            for copy, message in zip(checked, messages, strict=True)
            if copy.content != message.content
        ]
        return {"messages": edited} if edited else None

    def _check_model_output(
        self,
        request: ModelRequest,
        response: Any,
        hook_name: str,
        *,
        act_after_model: bool,
    ) -> Any:
        """Check the answer; leave a failed check's action, or the check itself if an
        inner action is pending, for the hook's node."""
        answer = _last_ai_message(response)
        state: Mapping[str, Any] = request.state or {}
        # A new turn's check replaces an earlier one's action (an answer from the
        # model cache can repeat its message id).
        stale = hook_name in (state.get(_OUTPUT_ACTIONS_KEY) or {})
        if answer is None:
            return _with_output_action(response, hook_name, None) if stale else response
        # The question from the state if an earlier middleware trimmed the request.
        input_text = self._last_input_text(request.messages) or self._last_input_text(
            state.get("messages", [])
        )
        attachments = self._attachments_for(
            scope=GuardrailScope.LLM,
            stage=GuardrailExecutionStage.POST,
            # The full history: an earlier middleware may have trimmed the request.
            state={**state, "messages": [*state.get("messages", []), answer]},
        )
        after_inner_action = (
            answer.id is not None and _pending_key(answer.id) in _pending_answers
        )
        violation: dict[str, Any] | None = None
        if after_inner_action:
            violation = {}  # checked in this hook's node, after the inner action
        elif act_after_model:
            violation = self._failed_output_check(answer, attachments)
        else:
            self._check_messages(
                [answer],
                scope=GuardrailScope.LLM,
                stage=GuardrailExecutionStage.POST,
                input_text=input_text,
                attachments=attachments,
            )
        if violation is None:
            return _with_output_action(response, hook_name, None) if stale else response
        if answer.id is None:
            answer.id = str(uuid4())  # so the action's node finds it
        _pending_answers.add(_pending_key(answer.id))
        violation.update(message_id=answer.id, input_text=input_text)
        return _with_output_action(response, hook_name, violation)

    def _failed_output_check(
        self,
        answer: AIMessage,
        attachments: list[GuardrailAttachment] | None,
        *,
        beside_node: bool = False,
    ) -> dict[str, Any] | None:
        """Check the model's answer; if it fails, what its action's node needs."""
        text = extract_text_from_messages([answer])
        if not text and not attachments:
            return None
        payload = text or NO_TEXT_PLACEHOLDER
        scope, stage = GuardrailScope.LLM, GuardrailExecutionStage.POST
        with self._guardrail_errors():
            with guardrail_stage(scope, stage, beside_node=beside_node):
                result = self._evaluate_guardrail(
                    payload, attachments, enforcement=self._enforcement(scope, stage)
                )
            if result.result == GuardrailValidationResultType.VALIDATION_FAILED:
                return {
                    "result": result.model_dump(mode="json"),
                    "payload": payload,
                    "has_text": bool(text),
                }
        return None

    def _run_output_action(
        self, state: Mapping[str, Any], hook_name: str
    ) -> dict[str, Any] | None:
        """Run the output action left for this node and return the state update.

        Dropped if a later model turn replaced its answer.
        """
        violation = (state.get(_OUTPUT_ACTIONS_KEY) or {}).get(hook_name)
        if not violation:
            return None
        done: dict[str, Any] = {_OUTPUT_ACTIONS_KEY: {hook_name: None}}
        _pending_answers.discard(_pending_key(violation["message_id"]))
        answer = next(
            (
                message
                for message in reversed(state.get("messages", []))
                if isinstance(message, AIMessage)
            ),
            None,
        )
        if answer is None or answer.id != violation["message_id"]:
            return done
        violation = self._violation_now(violation, answer, state)
        if violation is None:
            return done
        scope, stage = GuardrailScope.LLM, GuardrailExecutionStage.POST
        payload = violation["payload"]
        input_text = violation.get("input_text")
        reviewed = answer.model_copy()
        with self._guardrail_errors():
            with guardrail_stage(scope, stage, beside_node=True):
                modified_text = self._handle_validation_result(
                    GuardrailValidationResult.model_validate(violation["result"]),
                    payload,
                    scope=scope,
                    stage=stage,
                    component=component_label(scope),
                    input_payload=json.dumps(input_text) if input_text else None,
                )
            if violation.get("has_text"):
                _apply_text_modification([reviewed], payload, modified_text)
        if reviewed.content == answer.content:
            return done
        return {"messages": [reviewed], **done}

    def _violation_now(
        self,
        violation: dict[str, Any],
        answer: AIMessage,
        state: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        """The violation for the answer as it is now; re-checks it if it changed."""
        checked = "result" in violation and not (
            violation.get("has_text")
            and extract_text_from_messages([answer]) != violation["payload"]
        )
        if checked:
            return violation
        rechecked = self._failed_output_check(
            answer,
            self._attachments_for(
                scope=GuardrailScope.LLM,
                stage=GuardrailExecutionStage.POST,
                state=state,
            ),
            beside_node=True,
        )
        return None if rechecked is None else {**violation, **rechecked}

    def _last_input_text(self, messages: Sequence[BaseMessage]) -> str | None:
        """Return the last HumanMessage text — the input for a POST check.

        Used by output (POST) checks to supply the original input alongside the
        flagged output when escalating an output violation.
        """
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                return extract_text_from_messages([msg]) or None
        return None

    @contextmanager
    def _guardrail_errors(self) -> Generator[None]:
        """Let a block, a runtime error or a pause through; log anything else."""
        try:
            yield
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

    def _check_messages(
        self,
        messages: list[BaseMessage],
        scope: GuardrailScope | None = None,
        stage: GuardrailExecutionStage | None = None,
        input_text: str | None = None,
        attachments: list[GuardrailAttachment] | None = None,
        *,
        beside_node: bool = False,
    ) -> None:
        """Evaluate guardrail against message text; apply action on violation.

        ``input_text`` is the original input for a POST (output) check — the
        message that produced the flagged output — so an escalation can show it
        as ``Inputs`` alongside the flagged ``Outputs``. ``attachments`` are the
        file references sent with the text; with files but no text, the
        placeholder ``NO_TEXT_PLACEHOLDER`` is evaluated instead.
        """
        text = extract_text_from_messages(messages)
        if not text and not attachments:
            return
        payload = text or NO_TEXT_PLACEHOLDER

        with self._guardrail_errors():
            with guardrail_stage(scope, stage, beside_node=beside_node):
                result = self._evaluate_guardrail(
                    payload, attachments, enforcement=self._enforcement(scope, stage)
                )
                modified_text = self._handle_validation_result(
                    result,
                    payload,
                    scope=scope,
                    stage=stage,
                    component=component_label(scope),
                    input_payload=json.dumps(input_text) if input_text else None,
                )
            if text:
                _apply_text_modification(messages, text, modified_text)
