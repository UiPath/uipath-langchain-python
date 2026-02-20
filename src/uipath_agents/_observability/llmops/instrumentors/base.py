"""Base classes and shared state for span instrumentors."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set, Tuple
from uuid import UUID

from opentelemetry.trace import Span

from ..spans import LlmOpsSpanFactory


@dataclass
class InstrumentationState:
    """Shared mutable state accessed by all span instrumentors.

    Centralizes state that needs to be shared between LLM, Tool, and Guardrail
    instrumentors. Each instrumentor receives a reference to this object.
    """

    # Core tracing
    span_factory: LlmOpsSpanFactory
    agent_span: Optional[Span] = None
    agent_run_id: Optional[UUID] = None

    # Span storage (key -> span)
    spans: Dict[UUID, Span] = field(default_factory=dict)

    # LLM-specific state
    prompts_captured: bool = False
    current_llm_span: Optional[Span] = None
    llm_span_from_guardrail: bool = False  # Track if LLM span was created by guardrails

    # Tool-specific state
    current_tool_span: Optional[Span] = None
    tool_span_from_guardrail: bool = (
        False  # Track if tool span was created by guardrails
    )
    escalation_run_ids: Set[UUID] = field(default_factory=set)
    process_run_ids: Set[UUID] = field(default_factory=set)
    agent_run_ids: Set[UUID] = field(default_factory=set)
    ixp_extraction_run_ids: Set[UUID] = field(default_factory=set)
    vs_escalation_run_ids: Set[UUID] = field(default_factory=set)
    mcp_run_ids: Set[UUID] = field(default_factory=set)
    context_grounding_run_ids: Set[UUID] = field(default_factory=set)
    tool_output_schemas: Dict[UUID, Any] = field(default_factory=dict)

    # Pending interruptible tool spans (for suspend/resume)
    pending_tool_name: Optional[str] = None
    pending_tool_span: Optional[Span] = None
    pending_process_span: Optional[Span] = None

    # Preserved OTEL spans across suspend/resume (survives cleanup/reset)
    # Used by file exporter which needs the real OTEL span object to export
    suspended_tool_span: Optional[Span] = None
    suspended_process_span: Optional[Span] = None

    # Resume state
    resume_tool_name: Optional[str] = None
    reinvoked_tool_run_ids: Set[UUID] = field(default_factory=set)
    resumed_trace_id: Optional[str] = None
    resumed_tool_span_data: Optional[Dict[str, Any]] = None
    resumed_process_span_data: Optional[Dict[str, Any]] = None

    # Guardrail state
    guardrail_containers: Dict[Tuple[str, str], Span] = field(default_factory=dict)
    guardrail_metadata: Dict[UUID, Dict[str, Any]] = field(default_factory=dict)
    upcoming_guardrail_actions_info: Dict[str, Tuple[Span, UUID]] = field(
        default_factory=dict
    )
    pending_hitl_guardrail_span: Optional[Span] = None
    resumed_hitl_guardrail_span_data: Optional[Dict[str, Any]] = None
    resumed_llm_span_data: Optional[Dict[str, Any]] = None
    pending_hitl_guardrail_container_span: Optional[Span] = None
    resumed_hitl_guardrail_container_span_data: Optional[Dict[str, Any]] = None

    # Escalation state
    pending_escalation_span: Optional[Span] = None
    escalate_action_resume_data: Dict[UUID, Dict[str, Any]] = field(
        default_factory=dict
    )
    resumed_escalation_trace_id: Optional[str] = None
    resumed_escalation_span_data: Optional[Dict[str, Any]] = None

    # Enriched properties for telemetry
    enriched_properties: Dict[str, Any] = field(default_factory=dict)

    def reset_for_new_run(
        self, agent_span: Span, run_id: UUID, prompts_captured: bool
    ) -> None:
        """Reset state for a new agent run."""
        self.agent_span = agent_span
        self.agent_run_id = run_id
        self.spans.clear()
        self.prompts_captured = prompts_captured

        # Tool state
        self.pending_tool_name = None
        self.pending_tool_span = None
        self.pending_process_span = None
        self.current_tool_span = None
        self.tool_span_from_guardrail = False
        self.escalation_run_ids.clear()
        self.process_run_ids.clear()
        self.agent_run_ids.clear()
        self.ixp_extraction_run_ids.clear()
        self.vs_escalation_run_ids.clear()
        self.mcp_run_ids.clear()
        self.context_grounding_run_ids.clear()
        self.tool_output_schemas.clear()

        # Resume state
        self.resumed_trace_id = None
        self.resumed_tool_span_data = None
        self.resumed_process_span_data = None

        # Guardrail state
        self.guardrail_containers.clear()
        self.guardrail_metadata.clear()
        self.current_llm_span = None
        self.llm_span_from_guardrail = False
        self.upcoming_guardrail_actions_info.clear()
        self.pending_hitl_guardrail_span = None
        self.resumed_hitl_guardrail_span_data = None
        self.resumed_llm_span_data = None
        self.pending_hitl_guardrail_container_span = None
        self.resumed_hitl_guardrail_container_span_data = None

        # Escalation state
        self.pending_escalation_span = None
        self.escalate_action_resume_data.clear()
        self.resumed_escalation_trace_id = None
        self.resumed_escalation_span_data = None

    def get_span_or_root(self, run_id: Optional[UUID]) -> Optional[Span]:
        """Get span by run_id or fall back to agent span.

        Checks SpanHierarchyManager first to get the most specific (deepest)
        parent span, e.g. Analyze_Files child span instead of Tool call outer.
        """
        from ..callback import _get_current_span

        # Prefer context stack — returns deepest pushed span (e.g. child tool span)
        ctx_span = _get_current_span()
        if ctx_span:
            return ctx_span
        if run_id and run_id in self.spans:
            return self.spans[run_id]
        return self.agent_span


class BaseSpanInstrumentor:
    """Base class for span instrumentors with access to shared state."""

    def __init__(self, state: InstrumentationState) -> None:
        self._state = state

    @property
    def _span_factory(self) -> LlmOpsSpanFactory:
        return self._state.span_factory

    @property
    def _agent_span(self) -> Optional[Span]:
        return self._state.agent_span

    @property
    def _spans(self) -> Dict[UUID, Span]:
        return self._state.spans
