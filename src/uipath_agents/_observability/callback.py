"""LangChain callback handler for UiPath span instrumentation.

Injects into LangGraph execution to create UiPath-schema spans
without modifying the uipath-langchain library.

Key feature: Spans are created on START (not just end) for real-time visibility.
"""

import json
import logging
import re
from contextvars import Token
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from opentelemetry import context, trace
from opentelemetry.context import Context
from opentelemetry.trace import Span

from .tracer import UiPathTracer

logger = logging.getLogger(__name__)

# --- Guardrail Constants ---


class GuardrailScope:
    AGENT = "agent"
    LLM = "llm"
    TOOL = "tool"


class GuardrailStage:
    PRE = "pre"
    POST = "post"


class GuardrailAction:
    SKIP = "Skip"
    LOG = "Log"
    BLOCK = "Block"
    ESCALATE = "Escalate"


# Pattern: {scope}_{stage}_execution_{guardrail_name}
GUARDRAIL_NODE_PATTERN = re.compile(r"^(agent|llm|tool)_(pre|post)_execution_(.+)$")

# Map action suffix to action name
ACTION_SUFFIX_TO_NAME = {
    "_log": GuardrailAction.LOG,
    "_block": GuardrailAction.BLOCK,
    "_hitl": GuardrailAction.ESCALATE,
}


class UiPathTracingCallback(BaseCallbackHandler):
    """LangChain callback that creates UiPath-schema OpenTelemetry spans.

    Spans are created immediately on start events (not delayed until end)
    for real-time observability of long-running operations.

    Usage:
        tracer = UiPathTracer()
        callback = UiPathTracingCallback(tracer)
        runtime = SomeRuntime(callbacks=[callback])  # Callback persists on instance

        # Before each execution:
        with tracer.start_agent_run("MyAgent") as agent_span:
            callback.set_agent_span(agent_span)
            await runtime.execute(input)
    """

    def __init__(self, tracer: UiPathTracer) -> None:
        super().__init__()
        self._tracer = tracer
        self._agent_span: Optional[Span] = None
        # LLM spans stored at run_id, model spans at run_id ^ 1 (XOR with 1)
        self._spans: Dict[UUID, Span] = {}
        self._span_context_tokens: Dict[UUID, Token[Context]] = {}
        self._prompts_captured: bool = False
        # Pending interruptible tool spans (for suspend/resume)
        self._pending_tool_name: Optional[str] = None
        self._pending_tool_span: Optional[Span] = None
        self._pending_process_span: Optional[Span] = None
        # Resume mode - skip creating tool spans for first matching tool
        self._resume_tool_name: Optional[str] = None
        # Run IDs for re-invoked tools after resume (no spans created, originals upserted)
        self._reinvoked_tool_run_ids: set[UUID] = set()
        # Guardrail tracking
        self._guardrail_containers: Dict[Tuple[str, str], Span] = {}
        self._guardrail_spans: Dict[UUID, Span] = {}
        self._guardrail_info: Dict[UUID, Tuple[str, str, str]] = {}
        self._current_llm_span: Optional[Span] = None
        self._llm_span_from_guardrail: bool = False
        # Current tool span for tool guardrail parenting
        self._current_tool_span: Optional[Span] = None
        # Pending guardrail actions: node_name -> (span, validation_result)
        # When validation fails, we defer span ending until action node fires
        self._pending_guardrail_actions: Dict[str, Tuple[Span, Optional[str]]] = {}

    def set_agent_span(self, agent_span: Span) -> None:
        self._agent_span = agent_span
        self._spans.clear()
        self._span_context_tokens.clear()
        self._prompts_captured = False
        self._pending_tool_name = None
        self._pending_tool_span = None
        self._pending_process_span = None
        self._guardrail_containers.clear()
        self._guardrail_spans.clear()
        self._guardrail_info.clear()
        self._current_llm_span = None
        self._llm_span_from_guardrail = False
        self._current_tool_span = None
        self._pending_guardrail_actions.clear()

    def set_resume_context(self, tool_name: str) -> None:
        """Set resume context - skip creating spans for this tool on first encounter."""
        self._resume_tool_name = tool_name

    def get_pending_tool_info(
        self,
    ) -> tuple[Optional[str], Optional[Span], Optional[Span]]:
        return (
            self._pending_tool_name,
            self._pending_tool_span,
            self._pending_process_span,
        )

    def _get_span_or_root(self, run_id: Optional[UUID]) -> Optional[Span]:
        if run_id and run_id in self._spans:
            return self._spans[run_id]
        return self._agent_span

    def _model_span_key(self, run_id: UUID) -> UUID:
        """Derive a unique key for the model span from an LLM run_id.

        We create two spans per LLM call: outer LLM span + inner model span.
        Both need storage in _spans dict. XOR with 1 creates a distinct but
        deterministic key, avoiding collision while allowing retrieval at end.
        """
        return UUID(int=run_id.int ^ 1)

    def _start_llm_and_model_spans(
        self,
        run_id: UUID,
        parent_run_id: Optional[UUID],
        serialized: Dict[str, Any],
    ) -> None:
        """Create nested LLM call + model run spans and store in _spans.

        Creates:
        - LLM call span (outer, type: completion) at key run_id
        - Model run span (inner, type: completion) at key _model_span_key(run_id)

        If llmCall was created early by llm_pre guardrails, reuses it and updates attrs.
        """
        self._close_container(GuardrailScope.AGENT, GuardrailStage.PRE)

        model_name = self._extract_model_name(serialized)
        max_tokens, temperature = self._extract_settings(serialized)

        if self._current_llm_span and self._llm_span_from_guardrail:
            llm_span = self._current_llm_span
            settings = {"maxTokens": max_tokens, "temperature": temperature}
            llm_span.set_attribute("settings", json.dumps(settings))
            self._llm_span_from_guardrail = False
        else:
            parent = self._get_span_or_root(parent_run_id)
            llm_span = self._tracer.start_llm_call(
                max_tokens=max_tokens, temperature=temperature, parent_span=parent
            )

        self._current_llm_span = llm_span
        self._spans[run_id] = llm_span

        self._close_container(GuardrailScope.LLM, GuardrailStage.PRE)

        model_span = self._tracer.start_model_run(
            model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            parent_span=llm_span,
        )
        self._spans[self._model_span_key(run_id)] = model_span

    # -------------------------------------------------------------------------
    # LLM Events
    # -------------------------------------------------------------------------

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        try:
            self._start_llm_and_model_spans(run_id, parent_run_id, serialized)
        except Exception:
            logger.exception("Error in on_llm_start callback")

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Also captures interpolated prompts on first LLM call."""
        try:
            self._start_llm_and_model_spans(run_id, parent_run_id, serialized)

            if self._agent_span and not self._prompts_captured:
                self._capture_interpolated_prompts(messages)
        except Exception:
            logger.exception("Error in on_chat_model_start callback")

    def _capture_interpolated_prompts(self, messages: List[List[BaseMessage]]) -> None:
        """Extract and set interpolated prompts on the AgentRun span.

        Overwrites template values ({{input.x}}) with actual interpolated values.
        """
        if not messages or not messages[0]:
            return

        for msg in messages[0]:
            sanitized = self._sanitize_file_data(msg.content)
            content = sanitized if isinstance(sanitized, str) else json.dumps(sanitized)
            if msg.type == "system" and self._agent_span:
                self._agent_span.set_attribute("systemPrompt", content)
            elif msg.type == "human" and self._agent_span:
                self._agent_span.set_attribute("userPrompt", content)

        self._prompts_captured = True

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        try:
            # Close inner span first, then outer
            model_span = self._spans.pop(self._model_span_key(run_id), None)
            if model_span:
                self._set_usage_attributes(model_span, response)
                self._set_tool_calls_attributes(model_span, response)
                self._tracer.end_span_ok(model_span)
            llm_span = self._spans.pop(run_id, None)
            if llm_span:
                self._tracer.end_span_ok(llm_span)

            self._close_container(GuardrailScope.LLM, GuardrailStage.POST)

        except Exception:
            logger.exception("Error in on_llm_end callback")

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        try:
            exc = error if isinstance(error, Exception) else Exception(str(error))
            model_span = self._spans.pop(self._model_span_key(run_id), None)
            if model_span:
                self._tracer.end_span_error(model_span, exc)
            llm_span = self._spans.pop(run_id, None)
            if llm_span:
                self._tracer.end_span_error(llm_span, exc)

            self._close_container(GuardrailScope.LLM, GuardrailStage.POST)

        except Exception:
            logger.exception("Error in on_llm_error callback")

    # -------------------------------------------------------------------------
    # Tool Events
    # -------------------------------------------------------------------------

    def _interruptible_span_key(self, run_id: UUID) -> UUID:
        """Derive a unique key for interruptible tool child span from a tool run_id.

        Similar to _model_span_key, uses XOR to create distinct but
        deterministic key for the child span (escalation or process).
        """
        return UUID(int=run_id.int ^ 2)

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        try:
            tool_name = serialized.get("name", "unknown")
            tool_type = metadata.get("tool_type") if metadata else None
            tool_display_name = metadata.get("display_name") if metadata else None

            # Resume mode: skip duplicate span for re-invoked tool
            if self._resume_tool_name and tool_name == self._resume_tool_name:
                logger.debug("Resume mode: skipping span creation for %s", tool_name)
                self._resume_tool_name = None  # Only skip once
                self._reinvoked_tool_run_ids.add(run_id)
                return

            parent = self._get_span_or_root(parent_run_id)
            span = self._tracer.start_tool_call(tool_name, parent_span=parent)
            self._spans[run_id] = span
            self._current_tool_span = span

            span_context_token = context.attach(trace.set_span_in_context(span))
            self._span_context_tokens[run_id] = span_context_token

            call_id = kwargs.get("tool_call_id")
            if call_id:
                span.set_attribute("callId", call_id)
            if input_str:
                self._set_tool_arguments(span, input_str)

            if tool_type:
                child_span = None
                if tool_type == "escalation" and tool_display_name:
                    child_span = self._tracer.start_escalation_tool(
                        app_name=tool_display_name,
                        parent_span=span,
                    )
                elif tool_type == "process" and tool_display_name:
                    child_span = self._tracer.start_process_tool(
                        process_name=tool_display_name,
                        parent_span=span,
                    )
                elif tool_type == "integration":
                    child_span = self._tracer.start_integration_tool(
                        tool_name=tool_display_name or tool_name,
                        parent_span=span,
                    )

                if child_span:
                    self._spans[self._interruptible_span_key(run_id)] = child_span
                    # Track as pending for suspend scenario (escalation/process only)
                    if tool_type in ("escalation", "process"):
                        self._pending_tool_name = tool_name
                        self._pending_tool_span = span
                        self._pending_process_span = child_span

        except Exception:
            logger.exception("Error in on_tool_start callback")

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        try:
            # Skip re-invoked tool after resume - original spans already upserted
            if run_id in self._reinvoked_tool_run_ids:
                self._reinvoked_tool_run_ids.discard(run_id)
                return

            # Close child span first (inner), then tool span (outer)
            child_span = self._spans.pop(self._interruptible_span_key(run_id), None)
            if child_span:
                self._set_tool_result(child_span, output)
                self._tracer.end_span_ok(child_span)
                if child_span == self._pending_process_span:
                    self._pending_process_span = None

            span_context_token = self._span_context_tokens.pop(run_id, None)
            if span_context_token:
                try:
                    context.detach(span_context_token)
                except ValueError:
                    pass

            span = self._spans.pop(run_id, None)
            if span:
                self._set_tool_result(span, output)
                self._tracer.end_span_ok(span)
                if span == self._pending_tool_span:
                    self._pending_tool_span = None
                    self._pending_tool_name = None
                if span == self._current_tool_span:
                    # Close tool_post container before clearing current tool
                    self._close_container(GuardrailScope.TOOL, GuardrailStage.POST)
                    self._current_tool_span = None

        except Exception:
            logger.exception("Error in on_tool_end callback")

    def _is_graph_interrupt(self, error: BaseException) -> bool:
        """Check if the error is a GraphInterrupt (suspend signal).

        GraphInterrupt is raised by interruptible tools when they need to
        suspend execution. These spans should NOT be closed with ERROR -
        they will be handled by upsert in _handle_suspended().
        """
        error_str = str(error)
        error_type = type(error).__name__
        return error_type == "GraphInterrupt" or error_str.startswith("GraphInterrupt(")

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        try:
            if run_id in self._reinvoked_tool_run_ids:
                self._reinvoked_tool_run_ids.discard(run_id)
                return

            # GraphInterrupt = suspend signal, spans upserted with RUNNING by _handle_suspended()
            if self._is_graph_interrupt(error):
                logger.debug(
                    "GraphInterrupt detected for tool, spans kept open for upsert"
                )
                return

            exc = error if isinstance(error, Exception) else Exception(str(error))

            # Close child span first (inner), then tool span (outer)
            child_span = self._spans.pop(self._interruptible_span_key(run_id), None)
            if child_span:
                self._tracer.end_span_error(child_span, exc)

            span_context_token = self._span_context_tokens.pop(run_id, None)
            if span_context_token:
                try:
                    context.detach(span_context_token)
                except ValueError:
                    pass

            span = self._spans.pop(run_id, None)
            if span:
                self._tracer.end_span_error(span, exc)

        except Exception:
            logger.exception("Error in on_tool_error callback")

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def _sanitize_file_data(self, obj: Any) -> Any:
        """Recursively sanitize content, replacing file data with placeholders."""
        if isinstance(obj, bytes):
            return f"<bytes: {len(obj)} bytes>"
        if isinstance(obj, str):
            if obj.startswith("data:") and ";base64," in obj:
                return "<base64 data omitted>"
            if len(obj) > 1000 and obj.isascii():
                return "<base64 data omitted>"
            return obj
        if isinstance(obj, list):
            return [self._sanitize_file_data(item) for item in obj]
        if isinstance(obj, dict):
            sanitized = {}
            for key, value in obj.items():
                if key in (
                    "data",
                    "bytes",
                    "file_data",
                    "image_url",
                ) and not isinstance(value, dict):
                    if isinstance(value, bytes):
                        sanitized[key] = f"<bytes: {len(value)} bytes>"
                    elif isinstance(value, str) and len(value) > 100:
                        sanitized[key] = "<base64 data omitted>"
                    elif isinstance(value, list):
                        sanitized[key] = self._sanitize_file_data(value)
                    else:
                        sanitized[key] = value
                else:
                    sanitized[key] = self._sanitize_file_data(value)
            return sanitized
        return obj

    def _extract_model_name(self, serialized: Dict[str, Any]) -> str:
        kwargs = serialized.get("kwargs", {})
        return (
            kwargs.get("model_name")
            or kwargs.get("model")
            or serialized.get("name", "unknown")
        )

    def _extract_settings(
        self, serialized: Dict[str, Any]
    ) -> tuple[Optional[int], Optional[float]]:
        """Extract max_tokens and temperature from LLM config.

        Returns:
            Tuple of (max_tokens, temperature)
        """
        kwargs = serialized.get("kwargs", {})
        max_tokens = kwargs.get("max_tokens")
        temperature = kwargs.get("temperature")
        return max_tokens, temperature

    def _set_usage_attributes(self, span: Span, response: Optional[LLMResult]) -> None:
        """Extract and set token usage on span."""
        if not response or not response.llm_output:
            return

        token_usage = response.llm_output.get("token_usage") or response.llm_output.get(
            "usage"
        )
        if not token_usage:
            return

        usage = {
            "completionTokens": token_usage.get("completion_tokens", 0),
            "promptTokens": token_usage.get("prompt_tokens", 0),
            "totalTokens": token_usage.get("total_tokens", 0),
            "isByoExecution": False,
            "llmCalls": 1,
        }
        span.set_attribute("usage", json.dumps(usage))

    def _set_tool_calls_attributes(
        self, span: Span, response: Optional[LLMResult]
    ) -> None:
        """Extract and set tool calls on span."""
        if not response or not response.generations or not response.generations[0]:
            return

        generation = response.generations[0][0]
        message = getattr(generation, "message", None)
        if not message:
            return

        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls:
            return

        formatted_calls = []
        for tc in tool_calls:
            formatted_calls.append(
                {
                    "id": tc.get("id", ""),
                    "name": tc.get("name", ""),
                    "arguments": tc.get("args", {}),
                }
            )
        if formatted_calls:
            span.set_attribute("toolCalls", json.dumps(formatted_calls))

    def _set_tool_arguments(self, span: Span, input_str: str) -> None:
        try:
            args = json.loads(input_str)
            span.set_attribute("arguments", json.dumps(args))
        except (json.JSONDecodeError, TypeError):
            # input_str is not valid JSON, set as raw string
            span.set_attribute("arguments", input_str)

    def _set_tool_result(self, span: Span, output: Any) -> None:
        if output is None:
            return
        if isinstance(output, (dict, list)):
            span.set_attribute("result", json.dumps(output))
        else:
            span.set_attribute("result", str(output))

    # -------------------------------------------------------------------------
    # Chain Events (for Guardrail Nodes)
    # -------------------------------------------------------------------------

    def _parse_guardrail_node(self, node_name: str) -> Optional[Tuple[str, str, str]]:
        """Parse guardrail info from LangGraph node name.

        Args:
            node_name: The langgraph_node name from metadata

        Returns:
            Tuple of (scope, stage, guardrail_name) or None if not a guardrail node
        """
        for suffix in ACTION_SUFFIX_TO_NAME:
            if node_name.endswith(suffix):
                return None

        match = GUARDRAIL_NODE_PATTERN.match(node_name)
        if match:
            scope, stage, guardrail_name = match.groups()
            return (scope, stage, guardrail_name)
        return None

    def _get_or_create_container(
        self, scope: str, stage: str, parent_span: Optional[Span]
    ) -> Span:
        key = (scope, stage)
        if key not in self._guardrail_containers:
            container = self._tracer.start_guardrails_container(
                scope, stage, parent_span
            )
            self._guardrail_containers[key] = container
        return self._guardrail_containers[key]

    def _close_container(self, scope: str, stage: str) -> None:
        key = (scope, stage)
        if key in self._guardrail_containers:
            container = self._guardrail_containers.pop(key)
            self._tracer.end_span_ok(container)

    def _close_previous_phase_containers(
        self, current_scope: str, current_stage: str
    ) -> None:
        """Close containers from previous phases when transitioning.

        Phase order: agent_pre -> llm_pre -> (model) -> llm_post -> tool_pre -> (tool) -> tool_post -> agent_post
        """
        S, T = GuardrailScope, GuardrailStage
        if current_scope == S.LLM and current_stage == T.PRE:
            self._close_container(S.AGENT, T.PRE)
        elif current_scope == S.LLM and current_stage == T.POST:
            self._close_container(S.LLM, T.PRE)
        elif current_scope == S.TOOL and current_stage == T.PRE:
            self._close_container(S.LLM, T.POST)
        elif current_scope == S.TOOL and current_stage == T.POST:
            self._close_container(S.TOOL, T.PRE)
        elif current_scope == S.AGENT and current_stage == T.POST:
            self._close_container(S.LLM, T.POST)
            self._close_container(S.TOOL, T.POST)

    def _check_and_handle_action_node(self, node_name: str) -> bool:
        """Check if node is an action node and end the pending guardrail span.

        Action nodes have the pattern: {evaluation_node_name}_{action}
        where action is one of: log, block, escalate, hitl

        Returns True if this was an action node (caller should skip further processing).
        """
        for suffix, action in ACTION_SUFFIX_TO_NAME.items():
            if node_name.endswith(suffix):
                # Extract evaluation node name by removing the action suffix
                eval_node_name = node_name[: -len(suffix)]
                if eval_node_name in self._pending_guardrail_actions:
                    span, validation_result = self._pending_guardrail_actions.pop(
                        eval_node_name
                    )
                    self._tracer.end_guardrail_evaluation(
                        span,
                        validation_passed=False,
                        validation_result=validation_result,
                        action=action,
                    )
                    return True
        return False

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Detect guardrail nodes and create spans."""
        try:
            if not metadata:
                return

            node_name = metadata.get("langgraph_node", "")
            if not node_name:
                return

            # Check if this is an action node (fires after failed validation)
            if self._check_and_handle_action_node(node_name):
                return

            guardrail_info = self._parse_guardrail_node(node_name)
            if not guardrail_info:
                return

            scope, stage, guardrail_name = guardrail_info
            self._close_previous_phase_containers(scope, stage)

            # Determine parent span based on scope
            if scope == GuardrailScope.LLM:
                if stage == GuardrailStage.PRE:
                    # Create llmCall once per LLM turn (flag cleared by on_llm_start)
                    if not self._llm_span_from_guardrail:
                        self._current_llm_span = self._tracer.start_llm_call(
                            max_tokens=None,
                            temperature=None,
                            parent_span=self._agent_span,
                        )
                        self._llm_span_from_guardrail = True
                parent = self._current_llm_span or self._agent_span
            elif scope == GuardrailScope.TOOL:
                # Tool guardrails are children of the current tool span
                parent = self._current_tool_span or self._agent_span
            else:
                parent = self._get_span_or_root(parent_run_id)

            container = self._get_or_create_container(scope, stage, parent)
            eval_span = self._tracer.start_guardrail_evaluation(
                guardrail_name=guardrail_name,
                scope=scope,
                parent_span=container,
            )
            self._guardrail_spans[run_id] = eval_span
            self._guardrail_info[run_id] = (scope, stage, guardrail_name)

        except Exception:
            logger.exception("Error in on_chain_start callback (guardrail)")

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """End guardrail span with results."""
        try:
            if run_id not in self._guardrail_spans:
                return

            span = self._guardrail_spans.pop(run_id)
            info = self._guardrail_info.pop(run_id, None)

            validation_result = None
            if isinstance(outputs, dict):
                validation_result = outputs.get("guardrail_validation_result")

            validation_passed = validation_result is None

            if validation_passed:
                # Validation passed - end span immediately with "skip"
                self._tracer.end_guardrail_evaluation(
                    span,
                    validation_passed=True,
                    validation_result=None,
                    action=GuardrailAction.SKIP,
                )
            else:
                # Validation failed - defer span ending until action node fires
                # Reconstruct the evaluation node name from guardrail info
                if info:
                    scope, stage, guardrail_name = info
                    eval_node_name = f"{scope}_{stage}_execution_{guardrail_name}"
                    self._pending_guardrail_actions[eval_node_name] = (
                        span,
                        validation_result,
                    )
                else:
                    # Fallback: end with "log" if we can't track the action
                    self._tracer.end_guardrail_evaluation(
                        span,
                        validation_passed=False,
                        validation_result=validation_result,
                        action=GuardrailAction.LOG,
                    )
            # Container closing handled by phase transitions

        except Exception:
            logger.exception("Error in on_chain_end callback (guardrail)")

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Handle guardrail chain errors."""
        try:
            if run_id not in self._guardrail_spans:
                return

            span = self._guardrail_spans.pop(run_id)
            self._guardrail_info.pop(run_id, None)

            exc = error if isinstance(error, Exception) else Exception(str(error))
            self._tracer.end_span_error(span, exc)

        except Exception:
            logger.exception("Error in on_chain_error callback (guardrail)")

    def cleanup(self) -> None:
        for span_context_token in self._span_context_tokens.values():
            try:
                context.detach(span_context_token)
            except Exception:
                pass
        self._span_context_tokens.clear()

        for span in self._spans.values():
            # Skip pending interruptible spans - upserted with RUNNING, completed on resume
            if span is self._pending_tool_span or span is self._pending_process_span:
                continue
            try:
                span.end()
            except Exception:
                pass
        self._spans.clear()
        # Clear pending references without ending spans
        self._pending_tool_span = None
        self._pending_process_span = None
        self._pending_tool_name = None

        for span in self._guardrail_spans.values():
            try:
                span.end()
            except Exception:
                pass
        self._guardrail_spans.clear()
        self._guardrail_info.clear()

        # End any pending guardrail action spans
        for span, _ in self._pending_guardrail_actions.values():
            try:
                span.end()
            except Exception:
                pass
        self._pending_guardrail_actions.clear()

        for span in self._guardrail_containers.values():
            try:
                span.end()
            except Exception:
                pass
        self._guardrail_containers.clear()

        self._current_llm_span = None
        self._current_tool_span = None
