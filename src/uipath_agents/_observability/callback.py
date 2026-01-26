"""LangChain callback handler for UiPath span instrumentation.

Injects into LangGraph execution to create UiPath-schema spans
without modifying the uipath-langchain library.

Key feature: Spans are created on START (not just end) for real-time visibility.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import langchain_core.callbacks
import langchain_core.runnables.config
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from opentelemetry.trace import Span
from uipath.eval.mocks.mockable import MOCKED_ANNOTATION_KEY
from uipath.platform.guardrails import BuiltInValidatorGuardrail

from .schema import (
    GUARDRAIL_VALIDATION_DETAILS_KEY,
    GUARDRAIL_VALIDATION_RESULT_KEY,
    INNER_STATE_KEY,
)
from .telemetry_callback import (
    GUARDRAIL_BLOCKED,
    GUARDRAIL_LOGGED,
    GUARDRAIL_SKIPPED,
    track_event,
)
from .tracer import UiPathTracer

logger = logging.getLogger(__name__)

# --- Global Span Stack Registry ---
_span_stacks: Dict[UUID, List[Span]] = {}


def get_current_run_id() -> Optional[UUID]:
    """Get current run_id from langchain's internal runnable config.

    Reads directly from langchain's context instead of maintaining our own,
    ensuring we're always in sync with the actual execution context.
    """
    config = langchain_core.runnables.config.var_child_runnable_config.get()
    if not isinstance(config, dict):
        return None
    for v in config.values():
        if not isinstance(v, langchain_core.callbacks.BaseCallbackManager):
            continue
        if run_id := v.parent_run_id:
            return run_id
    return None


def _get_current_span() -> Optional[Span]:
    run_id = get_current_run_id()
    if run_id and run_id in _span_stacks:
        stack = _span_stacks[run_id]
        return stack[-1] if stack else None
    return None


def _get_ancestor_spans() -> List[Span]:
    run_id = get_current_run_id()
    if run_id and run_id in _span_stacks:
        return list(_span_stacks[run_id])
    return []


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
    FILTER = "Filter"


# Pattern: {scope}_{stage}_execution_{guardrail_name}
GUARDRAIL_NODE_PATTERN = re.compile(r"^(agent|llm|tool)_(pre|post)_execution_(.+)$")

# Map action suffix to action name
ACTION_SUFFIX_TO_NAME = {
    "_log": GuardrailAction.LOG,
    "_block": GuardrailAction.BLOCK,
    "_hitl": GuardrailAction.ESCALATE,
    "_filter": GuardrailAction.FILTER,
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

    def __init__(
        self,
        tracer: UiPathTracer,
        enriched_properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self._tracer = tracer
        self._enriched_properties = enriched_properties or {}
        self._agent_span: Optional[Span] = None
        # LLM spans stored at run_id, model spans at run_id ^ 1 (XOR with 1)
        self._spans: Dict[UUID, Span] = {}
        self._prompts_captured: bool = False
        # Pending interruptible tool spans (for suspend/resume)
        self._pending_tool_name: Optional[str] = None
        self._pending_tool_span: Optional[Span] = None
        self._pending_process_span: Optional[Span] = None
        # Resume mode - skip creating tool spans for first matching tool
        self._resume_tool_name: Optional[str] = None
        # Run IDs for re-invoked tools after resume (no spans created, originals upserted)
        self._reinvoked_tool_run_ids: set[UUID] = set()
        # Saved span data for resumed tools (to upsert immediately on tool end)
        self._resumed_tool_span_data: Optional[Dict[str, Any]] = None
        self._resumed_process_span_data: Optional[Dict[str, Any]] = None
        self._resumed_trace_id: Optional[str] = None
        self._escalation_run_ids: set[UUID] = set()
        self._process_run_ids: set[UUID] = set()
        self._agent_run_ids: set[UUID] = set()
        # Guardrail tracking
        self._guardrail_containers: Dict[Tuple[str, str], Span] = {}
        self._guardrail_spans: Dict[UUID, Span] = {}
        self._guardrail_info: Dict[UUID, Tuple[str, str, str]] = {}
        self._guardrail_metadata: Dict[UUID, Dict[str, Any]] = {}
        self._current_llm_span: Optional[Span] = None
        self._llm_span_from_guardrail: bool = False
        # Current tool span for tool guardrail parenting
        self._current_tool_span: Optional[Span] = None
        self._tool_span_from_guardrail: bool = False
        # Track when tool ended but waiting for post guardrails
        self._tool_ended_pending_post: bool = False
        # Pending guardrail actions: node_name -> (span, validation_details)
        # When validation fails, we defer span ending until action node fires
        self._pending_guardrail_actions: Dict[str, Tuple[Span, Optional[str]]] = {}

    def set_agent_span(
        self,
        agent_span: Span,
        run_id: UUID,
    ) -> None:
        self._agent_span = agent_span
        self._agent_run_id = run_id
        self._spans.clear()
        self._prompts_captured = False
        self._pending_tool_name = None
        self._pending_tool_span = None
        self._pending_process_span = None
        self._resumed_trace_id = None
        self._resumed_tool_span_data = None
        self._resumed_process_span_data = None
        self._guardrail_containers.clear()
        self._guardrail_spans.clear()
        self._guardrail_info.clear()
        self._guardrail_metadata.clear()
        self._current_llm_span = None
        self._llm_span_from_guardrail = False
        self._current_tool_span = None
        self._tool_span_from_guardrail = False
        self._tool_ended_pending_post = False
        self._pending_guardrail_actions.clear()
        self._escalation_run_ids.clear()
        self._process_run_ids.clear()
        self._agent_run_ids.clear()

        # Clear only this run_id's stack (not all stacks) for parallel agent support
        if run_id in _span_stacks:
            _span_stacks[run_id].clear()
        else:
            _span_stacks[run_id] = []

        _span_stacks[run_id].append(agent_span)

    def set_enriched_properties(
        self,
        enriched_properties: Dict[str, Any],
    ) -> None:
        """Set enriched telemetry properties.

        Args:
            enriched_properties: Dictionary of properties to set for telemetry events.
        """
        self._enriched_properties = enriched_properties

    # --- Span Stack Methods ---

    def _push_span(self, run_id: UUID, span: Span) -> None:
        if run_id not in _span_stacks:
            _span_stacks[run_id] = []
        _span_stacks[run_id].append(span)

    def _pop_span(self, run_id: UUID) -> Optional[Span]:
        if run_id in _span_stacks and _span_stacks[run_id]:
            return _span_stacks[run_id].pop()
        return None

    def cleanup_containers(self) -> None:
        """Close all remaining open guardrail container spans.

        Should be called when agent run completes to ensure all container spans
        are properly ended and exported.
        """
        for key in list(self._guardrail_containers.keys()):
            scope, stage = key
            self._close_container(scope, stage)

    def set_resume_context(
        self,
        tool_name: str,
        trace_id: Optional[str] = None,
        tool_span_data: Optional[Dict[str, Any]] = None,
        process_span_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Set resume context - skip creating spans for this tool on first encounter.

        Also stores span data for immediate upsert when tool completes, fixing
        duration update timing (span duration updates when tool finishes, not
        when outer agent ends).
        """
        self._resume_tool_name = tool_name
        self._resumed_trace_id = trace_id
        self._resumed_tool_span_data = tool_span_data
        self._resumed_process_span_data = process_span_data

    def get_pending_tool_info(
        self,
    ) -> tuple[Optional[str], Optional[Span], Optional[Span]]:
        return (
            self._pending_tool_name,
            self._pending_tool_span,
            self._pending_process_span,
        )

    def resumed_spans_completed(self) -> bool:
        """Check if resumed tool spans were already completed by the callback.

        Returns True if spans were upserted on tool completion, meaning
        _handle_resume_complete can skip redundant upsert.
        """
        # If resume data was cleared, spans were completed
        return (
            self._resumed_trace_id is None
            and self._resumed_tool_span_data is None
            and self._resumed_process_span_data is None
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
        input: Optional[str] = None,
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
            if input:
                llm_span.set_attribute("input", input)
            self._llm_span_from_guardrail = False
        else:
            parent = self._get_span_or_root(parent_run_id)
            llm_span = self._tracer.start_llm_call(
                max_tokens=max_tokens,
                temperature=temperature,
                input=input,
                parent_span=parent,
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
        self._push_span(run_id, model_span)

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
            input_text = prompts[0] if prompts else None
            self._start_llm_and_model_spans(
                run_id, parent_run_id, serialized, input=input_text
            )
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
            # Extract last message as input (matches C# LastOrDefault()?.Text)
            input_text = None
            if messages and messages[0]:
                last_msg = messages[0][-1]
                content = last_msg.content
                input_text = content if isinstance(content, str) else str(content)

            self._start_llm_and_model_spans(
                run_id, parent_run_id, serialized, input=input_text
            )

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
            self._pop_span(run_id)

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
            self._pop_span(run_id)

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

            # Parse arguments and call_id early for typed attributes
            call_id = kwargs.get("tool_call_id")
            arguments = self._parse_tool_arguments(input_str)

            # Map tool_type to toolType attribute value
            tool_type_value = self._get_tool_type_value(tool_type)

            # Check if tool span was created early by tool_pre guardrails
            if self._current_tool_span and self._tool_span_from_guardrail:
                span = self._current_tool_span
                span.update_name(f"Tool call - {tool_name}")
                span.set_attribute("toolName", tool_name)
                span.set_attribute("tool.name", tool_name)
                span.set_attribute("toolType", tool_type_value)
                if call_id:
                    span.set_attribute("callId", call_id)
                if arguments:
                    span.set_attribute("arguments", json.dumps(arguments))
                self._spans[run_id] = span
                self._tool_span_from_guardrail = False
                self._close_container(GuardrailScope.TOOL, GuardrailStage.PRE)
            else:
                parent = self._get_span_or_root(parent_run_id)
                span = self._tracer.start_tool_call(
                    tool_name,
                    tool_type_value=tool_type_value,
                    arguments=arguments,
                    call_id=call_id,
                    parent_span=parent,
                )
                span.set_attribute("tool.name", tool_name)
                self._spans[run_id] = span
                self._current_tool_span = span

            self._push_span(run_id, span)

            if tool_type:
                child_span = None
                if tool_type == "escalation" and tool_display_name:
                    child_span = self._tracer.start_escalation_tool(
                        app_name=tool_display_name,
                        parent_span=span,
                    )
                    self._escalation_run_ids.add(run_id)
                elif tool_type == "agent" and tool_display_name:
                    child_span = self._tracer.start_agent_tool(
                        agent_name=tool_display_name,
                        arguments=arguments,
                        parent_span=span,
                    )
                    self._agent_run_ids.add(run_id)
                elif tool_type == "process" and tool_display_name:
                    child_span = self._tracer.start_process_tool(
                        process_name=tool_display_name,
                        arguments=arguments,
                        parent_span=span,
                    )
                    self._process_run_ids.add(run_id)
                elif tool_type == "integration":
                    child_span = self._tracer.start_integration_tool(
                        tool_name=tool_display_name or tool_name,
                        parent_span=span,
                    )

                if child_span:
                    self._spans[self._interruptible_span_key(run_id)] = child_span
                    self._push_span(run_id, child_span)
                    # Track as pending for suspend scenario (escalation/process/agent)
                    if tool_type in ("escalation", "process", "agent"):
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
            # Handle resumed tool completion - upsert spans immediately with final duration
            if run_id in self._reinvoked_tool_run_ids:
                self._reinvoked_tool_run_ids.discard(run_id)
                self._upsert_resumed_spans_on_completion(output)
                return

            # Close child span first (inner), then tool span (outer)
            child_span = self._spans.pop(self._interruptible_span_key(run_id), None)
            if child_span:
                if hasattr(child_span, "attributes"):
                    if child_span.attributes.get(MOCKED_ANNOTATION_KEY) and hasattr(
                        child_span, "name"
                    ):
                        child_span.update_name(f"Simulated result: {child_span.name}")
                self._pop_span(run_id)  # Pop child
                self._set_tool_result(child_span, output)
                if run_id in self._escalation_run_ids:
                    self._set_escalation_task_info(child_span, output)
                    self._escalation_run_ids.discard(run_id)
                if run_id in self._process_run_ids:
                    self._set_process_job_info(child_span, output)
                    self._process_run_ids.discard(run_id)
                if run_id in self._agent_run_ids:
                    self._set_process_job_info(child_span, output)
                    self._agent_run_ids.discard(run_id)
                self._tracer.end_span_ok(child_span)
                if child_span == self._pending_process_span:
                    self._pending_process_span = None

            span = self._spans.pop(run_id, None)
            if span:
                self._pop_span(run_id)  # Pop tool
                self._set_tool_result(span, output)
                self._tracer.end_span_ok(span)
                if span == self._pending_tool_span:
                    self._pending_tool_span = None
                    self._pending_tool_name = None
                if span == self._current_tool_span:
                    # Mark tool as ended, but keep _current_tool_span for post guardrails
                    # Post guardrails fire AFTER on_tool_end in LangGraph execution order
                    self._tool_ended_pending_post = True

        except Exception:
            logger.exception("Error in on_tool_end callback")

    def _upsert_resumed_spans_on_completion(self, output: Any) -> None:
        """Upsert resumed tool/process spans immediately when tool completes.

        Fixes duration timing: span duration reflects actual tool execution time,
        not time until outer agent ends. Sets Status=OK + EndTime to mark complete.
        """
        if not self._resumed_trace_id:
            return

        # Upsert process span first (inner span)
        if self._resumed_process_span_data:
            # Add result to span data
            if output is not None:
                if isinstance(output, (dict, list)):
                    self._resumed_process_span_data["attributes"]["result"] = (
                        json.dumps(output)
                    )
                else:
                    self._resumed_process_span_data["attributes"]["result"] = str(
                        output
                    )

            self._tracer.upsert_span_complete_by_data(
                trace_id=self._resumed_trace_id,
                span_data=self._resumed_process_span_data,
            )
            logger.debug(
                "Upserted resumed process span %s on tool completion",
                self._resumed_process_span_data.get("name", "unknown"),
            )

        # Upsert tool span (outer span)
        if self._resumed_tool_span_data:
            if output is not None:
                if isinstance(output, (dict, list)):
                    self._resumed_tool_span_data["attributes"]["result"] = json.dumps(
                        output
                    )
                else:
                    self._resumed_tool_span_data["attributes"]["result"] = str(output)

            self._tracer.upsert_span_complete_by_data(
                trace_id=self._resumed_trace_id,
                span_data=self._resumed_tool_span_data,
            )
            logger.debug(
                "Upserted resumed tool span %s on tool completion",
                self._resumed_tool_span_data.get("name", "unknown"),
            )

        # Clear resumed span data
        self._resumed_trace_id = None
        self._resumed_tool_span_data = None
        self._resumed_process_span_data = None

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
                self._pop_span(run_id)  # Pop child
                self._tracer.end_span_error(child_span, exc)

            span = self._spans.pop(run_id, None)
            if span:
                self._pop_span(run_id)  # Pop tool
                self._tracer.end_span_error(span, exc)

        except Exception:
            logger.exception("Error in on_tool_error callback")

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    def _track_guardrail_event(
        self,
        action: str,
        metadata: Optional[Dict[str, Any]] = None,
        extra_props: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Track guardrail telemetry event."""
        props = self._enriched_properties.copy()
        if extra_props:
            props.update(extra_props)
        props["ActionType"] = action
        if metadata:
            props.update(self._build_guardrail_telemetry_props(metadata))

        event_name = None
        if action == GuardrailAction.BLOCK:
            event_name = GUARDRAIL_BLOCKED
        elif action == GuardrailAction.LOG:
            event_name = GUARDRAIL_LOGGED
        elif action == GuardrailAction.SKIP:
            event_name = GUARDRAIL_SKIPPED

        if event_name is None:
            return

        track_event(event_name, props)

    def _build_guardrail_telemetry_props(
        self, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Enrich telemetry properties with guardrail metadata."""
        props: Dict[str, Any] = {}

        guardrail = metadata.get("guardrail")
        scope = metadata.get("scope")
        execution_stage = metadata.get("execution_stage")

        if guardrail:
            props["EnabledForEvals"] = str(guardrail.enabled_for_evals).lower()
            props["GuardrailScopes"] = json.dumps(
                [s.name.title() for s in guardrail.selector.scopes]
            )

            if isinstance(guardrail, BuiltInValidatorGuardrail):
                props["GuardrailType"] = (
                    guardrail.guardrail_type[0].upper() + guardrail.guardrail_type[1:]
                )
                props["ValidatorType"] = "".join(
                    x.title() for x in guardrail.validator_type.split("_")
                )

        if scope:
            props["CurrentScope"] = scope.name.title()

        if execution_stage:
            props["ExecutionStage"] = execution_stage.name.title().replace("_", "")

        return props

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
        """Extract actual model name from LLM serialized data.

        Checks multiple locations where LangChain stores model info:
        - kwargs.model_name, kwargs.model (most common)
        - kwargs.model_id (some providers)
        - kwargs.deployment_name, kwargs.azure_deployment (Azure OpenAI)
        - serialized.id[-1] only if it looks like a model name (not a class)
        """
        kwargs = serialized.get("kwargs", {})

        model = (
            kwargs.get("model_name")
            or kwargs.get("model")
            or kwargs.get("model_id")
            # Azure OpenAI specific
            or kwargs.get("deployment_name")
            or kwargs.get("azure_deployment")
        )
        if model:
            return model

        # Check serialized.id - but only use if it looks like a model name
        # (contains version number, slash, or known model prefix)
        id_list = serialized.get("id", [])
        if id_list:
            last_id = id_list[-1] if isinstance(id_list, list) else str(id_list)
            # Model names typically contain version numbers, slashes, or known prefixes
            if any(
                indicator in str(last_id).lower()
                for indicator in [
                    "gpt",
                    "claude",
                    "gemini",
                    "llama",
                    "mistral",
                    "4o",
                    "3.5",
                    "4-",
                    "/",
                ]
            ):
                return str(last_id)

        return serialized.get("name", "unknown")

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
        if not response:
            return

        token_usage = None

        # Modern path: message.usage_metadata (LangChain 0.2+, always a TypedDict)
        if response.generations and response.generations[0]:
            gen = response.generations[0][0]
            msg = getattr(gen, "message", None)
            if msg and hasattr(msg, "usage_metadata") and msg.usage_metadata:
                um = msg.usage_metadata
                token_usage = {
                    "prompt_tokens": um.get("input_tokens", 0),
                    "completion_tokens": um.get("output_tokens", 0),
                    "total_tokens": um.get("total_tokens", 0),
                }

        # Legacy fallback: llm_output
        if not token_usage and response.llm_output:
            token_usage = response.llm_output.get(
                "token_usage"
            ) or response.llm_output.get("usage")

        if not token_usage:
            return

        usage = {
            "completionTokens": token_usage.get("completion_tokens", 0),
            "promptTokens": token_usage.get("prompt_tokens", 0),
            "totalTokens": token_usage.get("total_tokens", 0),
            "isByoExecution": False,
            # TODO: Placeholder - get from gateway headers via uipath-langchain when available
            "executionDeploymentType": None,
            "isPiiMasked": False,
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

        content = getattr(message, "content", None)
        if content and isinstance(content, str) and content.strip():
            span.set_attribute("explanation", content)

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

    def _parse_tool_arguments(self, input_str: str) -> Optional[Dict[str, Any]]:
        """Parse tool arguments from input string."""
        if not input_str:
            return None
        try:
            args = json.loads(input_str)
            return args if isinstance(args, dict) else None
        except (json.JSONDecodeError, TypeError):
            return None

    def _get_tool_type_value(self, tool_type: Optional[str]) -> str:
        """Map tool_type metadata to toolType attribute value."""
        if tool_type == "agent":
            return "Agent"
        elif tool_type == "process":
            return "Process"
        elif tool_type == "escalation":
            return "ActionCenter"
        else:
            return "Integration"

    def _set_escalation_task_info(self, span: Span, output: Any) -> None:
        """Extract and set task_id/task_url from escalation output."""
        if not isinstance(output, dict):
            return
        task_id = output.get("task_id") or output.get("taskId") or output.get("id")
        if task_id:
            span.set_attribute("taskId", str(task_id))
        task_url = output.get("task_url") or output.get("taskUrl") or output.get("url")
        if task_url:
            span.set_attribute("taskUrl", str(task_url))

    def _set_process_job_info(self, span: Span, output: Any) -> None:
        """Extract and set job_id/job_details_uri from process tool output."""
        if not isinstance(output, dict):
            return
        job_id = output.get("job_id") or output.get("jobId") or output.get("JobId")
        if job_id:
            span.set_attribute("jobId", str(job_id))
        job_uri = (
            output.get("job_details_uri")
            or output.get("jobDetailsUri")
            or output.get("JobDetailsUri")
            or output.get("job_url")
            or output.get("jobUrl")
        )
        if job_uri:
            span.set_attribute("jobDetailsUri", str(job_uri))

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

        # Clear tool span after post guardrails complete
        if (
            scope == GuardrailScope.TOOL
            and stage == GuardrailStage.POST
            and self._tool_ended_pending_post
        ):
            self._current_tool_span = None
            self._tool_ended_pending_post = False

    def _close_previous_phase_containers(
        self, current_scope: str, current_stage: str
    ) -> None:
        """Close containers from previous phases when transitioning.

        Phase order: agent_pre -> llm_pre -> (model) -> llm_post -> tool_pre -> (tool) -> tool_post -> agent_post
        """
        S, T = GuardrailScope, GuardrailStage
        if current_scope == S.LLM and current_stage == T.PRE:
            self._close_container(S.AGENT, T.PRE)
            # New LLM iteration: cleanup any pending tool state from previous iteration
            self._close_container(S.TOOL, T.POST)
        elif current_scope == S.LLM and current_stage == T.POST:
            self._close_container(S.LLM, T.PRE)
        elif current_scope == S.TOOL and current_stage == T.PRE:
            self._close_container(S.LLM, T.POST)
        elif current_scope == S.TOOL and current_stage == T.POST:
            self._close_container(S.TOOL, T.PRE)
        elif current_scope == S.AGENT and current_stage == T.POST:
            self._close_container(S.LLM, T.POST)
            self._close_container(S.TOOL, T.POST)

    def _check_and_handle_action_node(
        self,
        node_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
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
                    span, validation_details = self._pending_guardrail_actions.pop(
                        eval_node_name
                    )

                    # Extract metadata from action node
                    severity_level = None
                    reason = None

                    if metadata:
                        severity_level = metadata.get("severity_level")
                        reason = metadata.get("reason")

                    self._tracer.end_guardrail_evaluation(
                        span,
                        validation_passed=False,
                        validation_result=validation_details,
                        action=action,
                        severity_level=severity_level,
                        reason=reason,
                    )

                    self._track_guardrail_event(action, metadata, {})

                    # If tool_pre guardrail blocks, end the placeholder tool span
                    if (
                        action == GuardrailAction.BLOCK
                        and self._tool_span_from_guardrail
                        and "tool_pre" in eval_node_name
                    ):
                        self._close_container(GuardrailScope.TOOL, GuardrailStage.PRE)
                        if self._current_tool_span:
                            self._current_tool_span.set_attribute(
                                "output", "Blocked by guardrail"
                            )
                            self._current_tool_span.end()
                            self._current_tool_span = None
                        self._tool_span_from_guardrail = False

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
            if self._check_and_handle_action_node(node_name, metadata):
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
                if stage == GuardrailStage.PRE and not self._current_tool_span:
                    self._current_tool_span = self._tracer.start_tool_call(
                        tool_name="Tool call",
                        parent_span=self._agent_span,
                    )
                    self._tool_span_from_guardrail = True
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
            self._guardrail_metadata[run_id] = metadata
            self._push_span(run_id, eval_span)

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

            self._pop_span(run_id)
            span = self._guardrail_spans.pop(run_id)
            info = self._guardrail_info.pop(run_id, None)
            metadata = self._guardrail_metadata.pop(run_id, None)

            validation_result = None
            validation_details = None
            if isinstance(outputs, dict) and INNER_STATE_KEY in outputs:
                validation_result = outputs[INNER_STATE_KEY].get(
                    GUARDRAIL_VALIDATION_RESULT_KEY
                )
                validation_details = outputs[INNER_STATE_KEY].get(
                    GUARDRAIL_VALIDATION_DETAILS_KEY
                )
            elif (
                hasattr(outputs, "update")
                and isinstance(outputs.update, dict)
                and INNER_STATE_KEY in outputs.update
            ):
                validation_result = outputs.update[INNER_STATE_KEY].get(
                    GUARDRAIL_VALIDATION_RESULT_KEY
                )
                validation_details = outputs.update[INNER_STATE_KEY].get(
                    GUARDRAIL_VALIDATION_DETAILS_KEY
                )

            validation_passed = validation_result is True

            if validation_passed:
                # Validation passed - end span immediately with "skip"
                self._tracer.end_guardrail_evaluation(
                    span,
                    validation_passed=True,
                    validation_result=validation_details,
                    action=GuardrailAction.SKIP,
                )

                # Track Guardrail.Skipped event
                base_props = {"validationDetails": validation_details}
                self._track_guardrail_event(GuardrailAction.SKIP, metadata, base_props)
            else:
                # Validation failed - defer span ending until action node fires
                # Reconstruct the evaluation node name from guardrail info
                if info:
                    scope, stage, guardrail_name = info
                    eval_node_name = f"{scope}_{stage}_execution_{guardrail_name}"
                    self._pending_guardrail_actions[eval_node_name] = (
                        span,
                        validation_details,
                    )
                else:
                    # Fallback: end with "log" if we can't track the action
                    self._tracer.end_guardrail_evaluation(
                        span,
                        validation_passed=False,
                        validation_result=validation_details,
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

            self._pop_span(run_id)
            span = self._guardrail_spans.pop(run_id)
            self._guardrail_info.pop(run_id, None)

            exc = error if isinstance(error, Exception) else Exception(str(error))
            self._tracer.end_span_error(span, exc)

        except Exception:
            logger.exception("Error in on_chain_error callback (guardrail)")

    def cleanup(self) -> None:
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

        # Clear resume-related fields
        self._resumed_trace_id = None
        self._resumed_tool_span_data = None
        self._resumed_process_span_data = None

        for span in self._guardrail_spans.values():
            try:
                span.end()
            except Exception:
                pass
        self._guardrail_spans.clear()
        self._guardrail_info.clear()
        self._guardrail_metadata.clear()

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

        # End orphaned placeholder tool span if guardrail created it but tool never started
        if self._tool_span_from_guardrail and self._current_tool_span:
            try:
                self._current_tool_span.end()
            except Exception:
                pass
            self._tool_span_from_guardrail = False

        self._current_llm_span = None
        self._current_tool_span = None
