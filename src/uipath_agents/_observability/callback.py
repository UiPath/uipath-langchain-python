"""LangChain callback handler for UiPath span instrumentation.

Injects into LangGraph execution to create UiPath-schema spans
without modifying the uipath-langchain library.

Key feature: Spans are created on START (not just end) for real-time visibility.
"""

import json
import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import LLMResult
from opentelemetry.trace import Span

from .tracer import UiPathTracer

logger = logging.getLogger(__name__)


class UiPathTracingCallback(BaseCallbackHandler):
    """LangChain callback that creates UiPath-schema OpenTelemetry spans.

    Spans are created immediately on start events (not delayed until end)
    for real-time observability of long-running operations.

    Context handling: Uses set_span_in_context() at span creation only,
    avoiding hazardous attach()/detach() patterns that can corrupt context.

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
        self._prompts_captured: bool = False
        # Pending interruptible tool spans (for suspend/resume)
        self._pending_tool_name: Optional[str] = None
        self._pending_tool_span: Optional[Span] = None
        self._pending_process_span: Optional[Span] = None
        # Resume mode - skip creating tool spans for first matching tool
        self._resume_tool_name: Optional[str] = None

    def set_agent_span(self, agent_span: Span) -> None:
        self._agent_span = agent_span
        self._spans.clear()
        self._prompts_captured = False
        self._pending_tool_name = None
        self._pending_tool_span = None
        self._pending_process_span = None

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
        """
        model_name = self._extract_model_name(serialized)
        max_tokens, temperature = self._extract_settings(serialized)
        parent = self._get_span_or_root(parent_run_id)

        # LLM call: no model (model only on child span)
        llm_span = self._tracer.start_llm_call(
            max_tokens=max_tokens, temperature=temperature, parent_span=parent
        )
        self._spans[run_id] = llm_span

        # Model run: has model and settings
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
            content = (
                msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
            )
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
            # Close model span first (inner), then LLM span (outer)
            model_span = self._spans.pop(self._model_span_key(run_id), None)
            if model_span:
                # Add usage and toolCalls to model span
                self._set_usage_attributes(model_span, response)
                self._set_tool_calls_attributes(model_span, response)
                self._tracer.end_span_ok(model_span)
            llm_span = self._spans.pop(run_id, None)
            if llm_span:
                self._tracer.end_span_ok(llm_span)

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

            # Extract tool_type and display_name from metadata (set by uipath-langchain)
            tool_type = metadata.get("tool_type") if metadata else None
            tool_display_name = metadata.get("display_name") if metadata else None

            # Resume mode: skip span creation for the resumed tool
            if self._resume_tool_name and tool_name == self._resume_tool_name:
                logger.debug("Resume mode: skipping span creation for %s", tool_name)
                self._resume_tool_name = None  # Only skip once
                # Mark this run_id as "skip" so on_tool_end knows not to end spans
                self._spans[run_id] = None  # type: ignore
                return

            parent = self._get_span_or_root(parent_run_id)
            span = self._tracer.start_tool_call(tool_name, parent_span=parent)
            self._spans[run_id] = span

            if tool_type and tool_display_name:
                child_span = None
                if tool_type == "escalation":
                    child_span = self._tracer.start_escalation_tool(
                        app_name=tool_display_name,
                        parent_span=span,
                    )
                elif tool_type == "process":
                    child_span = self._tracer.start_process_tool(
                        process_name=tool_display_name,
                        parent_span=span,
                    )
                if child_span:
                    self._spans[self._interruptible_span_key(run_id)] = child_span
                    # Track as pending for suspend scenario
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
            # Close interruptible child span first, then tool span (parent)
            child_span = self._spans.pop(self._interruptible_span_key(run_id), None)
            if child_span:
                self._tracer.end_span_ok(child_span)
                # Clear pending if this was the pending span
                if child_span == self._pending_process_span:
                    self._pending_process_span = None

            span = self._spans.pop(run_id, None)
            if span:  # None means this was a skipped resume tool
                self._tracer.end_span_ok(span)
                if span == self._pending_tool_span:
                    self._pending_tool_span = None
                    self._pending_tool_name = None

        except Exception:
            logger.exception("Error in on_tool_end callback")

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        try:
            exc = error if isinstance(error, Exception) else Exception(str(error))

            # Close interruptible child span first, then tool span (parent)
            child_span = self._spans.pop(self._interruptible_span_key(run_id), None)
            if child_span:
                self._tracer.end_span_error(child_span, exc)

            span = self._spans.pop(run_id, None)
            if span:
                self._tracer.end_span_error(span, exc)

        except Exception:
            logger.exception("Error in on_tool_error callback")

    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

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

    def cleanup(self) -> None:
        for span in self._spans.values():
            try:
                span.end()
            except Exception:
                pass
        self._spans.clear()
