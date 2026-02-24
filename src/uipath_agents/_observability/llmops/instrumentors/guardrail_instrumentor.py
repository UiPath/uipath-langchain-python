"""Guardrail span instrumentor for LLMOps instrumentation."""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from opentelemetry.trace import NonRecordingSpan, Span, SpanContext, TraceFlags
from uipath.core.guardrails import GuardrailScope, UniversalRule
from uipath.platform.guardrails import BuiltInValidatorGuardrail, DeterministicGuardrail
from uipath.tracing import SpanStatus
from uipath_langchain.agent.guardrails.types import ExecutionStage

from ...event_emitter import GuardrailEvent, track_event
from ..span_hierarchy import SpanHierarchyManager
from ..spans.span_name import (
    GUARDRAIL_VALIDATION_DETAILS_KEY,
    GUARDRAIL_VALIDATION_RESULT_KEY,
    INNER_STATE_KEY,
)
from ..spans.spans_schema import to_json_string
from .attribute_helpers import (
    get_tool_type_value,
)
from .base import BaseSpanInstrumentor, InstrumentationState
from .constants import (
    ACTION_SUFFIX_TO_NAME,
    GUARDRAIL_NODE_PATTERN,
    GuardrailAction,
)

logger = logging.getLogger(__name__)


class GuardrailSpanInstrumentor(BaseSpanInstrumentor):
    """Instruments guardrail chain events with spans.

    Guardrail nodes are detected by naming pattern: {scope}_{stage}_execution_{name}
    Creates container spans to group guardrails by scope/stage.

    Span hierarchy:
        AgentRun
        └── GuardrailsContainer (scope_stage)
            └── GuardrailEvaluation
    """

    def __init__(self, state: InstrumentationState) -> None:
        super().__init__(state)

    # --- Container Management ---

    def _is_graph_interrupt(self, error: BaseException) -> bool:
        """Check if the error is a GraphInterrupt (suspend signal)."""
        error_str = str(error)
        error_type = type(error).__name__
        return error_type == "GraphInterrupt" or error_str.startswith("GraphInterrupt(")

    def _get_or_create_guardrails_group_span(
        self,
        scope: str,
        stage: str,
        parent_span: Optional[Span],
    ) -> Span:
        """Get or create a guardrail container span.

        If there's a resumed HITL container span data available, restore it as a
        NonRecordingSpan so that children spans can be created under it.
        """
        key = (scope, stage)
        if key not in self._state.guardrail_containers:
            # Check if we have resumed container span data to restore
            container_data = self._state.resumed_hitl_guardrail_container_span_data
            if container_data and self._state.resumed_escalation_trace_id:
                restored_span_context = SpanContext(
                    trace_id=int(self._state.resumed_escalation_trace_id, 16),
                    span_id=int(container_data["span_id"], 16),
                    is_remote=True,
                    trace_flags=TraceFlags(0x01),  # Sampled
                )
                self._state.guardrail_containers[key] = NonRecordingSpan(
                    restored_span_context
                )
            else:
                self._state.guardrail_containers[key] = (
                    self._span_factory.start_guardrails_container(
                        scope, stage, parent_span
                    )
                )
        return self._state.guardrail_containers[key]

    def close_container(self, scope: str, stage: str) -> None:
        """Close a guardrail container span."""
        key = (scope, stage)
        container_closed = False
        if key in self._state.guardrail_containers:
            container = self._state.guardrail_containers.pop(key)
            # Handle NonRecordingSpan (resumed container) - use upsert_span_complete_by_data
            if isinstance(container, NonRecordingSpan):
                container_data = self._state.resumed_hitl_guardrail_container_span_data
                trace_id = self._state.resumed_escalation_trace_id
                if container_data and trace_id:
                    self._span_factory.upsert_span_complete_by_data(
                        trace_id=trace_id,
                        span_data=container_data,
                    )
            else:
                self._span_factory.end_span_ok(container)
            self._state.resumed_hitl_guardrail_container_span_data = None
            self._state.pending_hitl_guardrail_container_span = None
            container_closed = True

        # Only clear LLM/tool spans if we actually closed a container
        # (prevents premature clearing when on_llm_end/on_tool_end calls this
        # before POST guardrails have run)
        if not container_closed:
            return

        # Clear LLM span after post guardrails complete
        if (
            scope == GuardrailScope.LLM
            and stage == ExecutionStage.POST_EXECUTION
            and self._state.current_llm_span is not None
        ):
            self._state.current_llm_span = None

        # Clear tool span after post guardrails complete
        if (
            scope == GuardrailScope.TOOL
            and stage == ExecutionStage.POST_EXECUTION
            and self._state.current_tool_span is not None
        ):
            self._state.current_tool_span = None

    def cleanup_containers(self) -> None:
        """Close all remaining open guardrail container spans."""
        for key in list(self._state.guardrail_containers.keys()):
            scope, stage = key
            self.close_container(scope, stage)

    def _close_previous_phase_containers(
        self, current_scope: str, current_stage: str
    ) -> None:
        """Close containers from previous phases when transitioning."""
        if (
            current_scope == GuardrailScope.LLM
            and current_stage == ExecutionStage.PRE_EXECUTION
        ):
            self.close_container(GuardrailScope.AGENT, ExecutionStage.PRE_EXECUTION)
            self.close_container(GuardrailScope.TOOL, ExecutionStage.POST_EXECUTION)
        elif (
            current_scope == GuardrailScope.LLM
            and current_stage == ExecutionStage.POST_EXECUTION
        ):
            self.close_container(GuardrailScope.LLM, ExecutionStage.PRE_EXECUTION)
        elif (
            current_scope == GuardrailScope.TOOL
            and current_stage == ExecutionStage.PRE_EXECUTION
        ):
            self.close_container(GuardrailScope.LLM, ExecutionStage.POST_EXECUTION)
        elif (
            current_scope == GuardrailScope.TOOL
            and current_stage == ExecutionStage.POST_EXECUTION
        ):
            self.close_container(GuardrailScope.TOOL, ExecutionStage.PRE_EXECUTION)
        elif (
            current_scope == GuardrailScope.AGENT
            and current_stage == ExecutionStage.POST_EXECUTION
        ):
            self.close_container(GuardrailScope.LLM, ExecutionStage.POST_EXECUTION)
            self.close_container(GuardrailScope.TOOL, ExecutionStage.POST_EXECUTION)

    # --- Node Parsing ---

    def _parse_guardrail_node(self, node_name: str) -> Optional[Tuple[str, str, str]]:
        """Parse guardrail info from LangGraph node name."""
        match = GUARDRAIL_NODE_PATTERN.match(node_name)
        if match:
            scope, stage, guardrail_name = match.groups()
            return (scope, stage, guardrail_name)
        return None

    def _extract_scope_stage(
        self, eval_node_name: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """Extract scope and stage from evaluation node name."""
        match = GUARDRAIL_NODE_PATTERN.match(eval_node_name)
        if match:
            scope, stage, _ = match.groups()
            return (scope, stage)
        return (None, None)

    # --- Action Node Handling ---

    def handle_action_end(
        self,
        run_id: UUID,
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        """Handle guardrail action node completion (success path)."""
        if metadata is None:
            return
        action = metadata.get("action_type")

        node_name = metadata.get("langgraph_node", "")
        if not node_name:
            return

        # Extract evaluation node name by removing the action suffix
        eval_node_name = self._get_guardrail_eval_node_name(node_name)

        # Resume scenario: approved HITL task
        # Because this callback is invoked for both the action node and the create_escalation_task,
        # we need to also filter by the node that has reviewed_by in the metadata
        escalation_data = metadata.get("escalation_data") or {}
        reviewed_by = escalation_data.get("reviewed_by")
        if action == GuardrailAction.ESCALATE and reviewed_by is not None:
            trace_id = self._state.resumed_escalation_trace_id
            eval_span_data = self._state.resumed_hitl_guardrail_span_data

            # Complete Review task span
            self._complete_resumed_escalation_from_outputs(metadata)

            # Complete guardrail eval span
            if trace_id and eval_span_data:
                self._span_factory.upsert_span_complete_by_data(
                    trace_id=trace_id,
                    span_data=eval_span_data,
                )

            self._track_escalation_outcome_event(
                GuardrailEvent.ESCALATION_APPROVED, metadata
            )
            return

        # Normal flow: complete the pending guardrail span
        severity_level = metadata.get("severity_level")
        reason = metadata.get("reason")
        excluded_fields = metadata.get("excluded_fields")
        updated_data = metadata.get("updated_data")

        if eval_node_name not in self._state.upcoming_guardrail_actions_info.keys():
            return

        eval_span, eval_run_id = self._state.upcoming_guardrail_actions_info.pop(
            eval_node_name
        )
        # Remove mapping for action node run id
        SpanHierarchyManager.pop(run_id)
        # Remove mapping for eval node run id
        SpanHierarchyManager.pop(eval_run_id)
        if eval_span:
            self._span_factory.end_guardrail_evaluation(
                eval_span,
                action=action,
                severity_level=severity_level,
                reason=reason,
                excluded_fields=excluded_fields,
                updated_data=updated_data,
            )

        # The Guardrails.Escalated event is logged right after the HITL interruption is done
        if action != GuardrailAction.ESCALATE:
            self._track_guardrail_event(action, metadata)

    def handle_action_error(
        self,
        run_id: UUID,
        metadata: Optional[Dict[str, Any]],
        error: BaseException,
    ) -> None:
        """Handle guardrail action node error (block/reject path)."""
        if metadata is None:
            return
        action = metadata.get("action_type")

        node_name = metadata.get("langgraph_node", "")
        if not node_name:
            return

        # Extract evaluation node name by removing the action suffix
        eval_node_name = self._get_guardrail_eval_node_name(node_name)
        reason = metadata.get("reason")

        error_str = " ".join(str(arg) for arg in error.args)

        if action == GuardrailAction.BLOCK:
            eval_span, eval_run_id = self._state.upcoming_guardrail_actions_info.pop(
                eval_node_name
            )
            # Remove mapping for action node run id
            SpanHierarchyManager.pop(run_id)
            # Remove mapping for eval node run id
            SpanHierarchyManager.pop(eval_run_id)

            # End guardrail span
            if eval_span and error:
                self._span_factory.error_guardrail_evaluation(
                    eval_span, action=action, reason=reason, error=error
                )

            self._track_guardrail_event(action, metadata)

        elif action == GuardrailAction.ESCALATE:
            # Complete Review task span
            self._complete_resumed_escalation_from_outputs(metadata, error_str)

            # Complete guardrail eval span
            eval_span_data = self._state.resumed_hitl_guardrail_span_data
            trace_id = self._state.resumed_escalation_trace_id
            if trace_id and eval_span_data:
                eval_span_data["attributes"]["error"] = str(error_str)
                self._span_factory.upsert_span_complete_by_data(
                    trace_id=trace_id,
                    span_data=eval_span_data,
                    status=SpanStatus.ERROR,
                )

            self._track_escalation_outcome_event(
                GuardrailEvent.ESCALATION_REJECTED, metadata
            )

        # End guardrail group span
        scope = metadata.get("scope")
        execution_stage = metadata.get("execution_stage")

        if scope and execution_stage:
            container_key = (scope, execution_stage)
            if container_key in self._state.guardrail_containers:
                guardrail_group_span = self._state.guardrail_containers.pop(
                    container_key
                )
                if isinstance(guardrail_group_span, NonRecordingSpan):
                    container_data = (
                        self._state.resumed_hitl_guardrail_container_span_data
                    )
                    trace_id = self._state.resumed_escalation_trace_id
                    if container_data and trace_id:
                        container_data["attributes"]["error"] = str(error_str)
                        self._span_factory.upsert_span_complete_by_data(
                            trace_id=trace_id,
                            span_data=container_data,
                            status=SpanStatus.ERROR,
                        )
                else:
                    self._span_factory.end_span_error(guardrail_group_span, error)

        # End LLM call span
        if scope == GuardrailScope.LLM:
            if isinstance(self._state.current_llm_span, NonRecordingSpan):
                llm_span_data = self._state.resumed_llm_span_data
                trace_id = self._state.resumed_escalation_trace_id
                if llm_span_data and trace_id:
                    llm_span_data["attributes"]["error"] = str(error_str)
                    self._span_factory.upsert_span_complete_by_data(
                        trace_id=trace_id,
                        span_data=llm_span_data,
                        status=SpanStatus.ERROR,
                    )
            elif self._state.current_llm_span:
                self._span_factory.end_span_error(self._state.current_llm_span, error)

            self._state.current_llm_span = None
            return

        # End Tool call span
        if scope == GuardrailScope.TOOL:
            # Use upsert_span_complete_by_data for resumed spans because they are NonRecordingSpan that can't record attributes
            if self._state.resumed_tool_span_data and self._state.resumed_trace_id:
                # Add error attributes to the span data
                self._state.resumed_tool_span_data["attributes"]["error"] = error_str
                self._span_factory.upsert_span_complete_by_data(
                    trace_id=self._state.resumed_trace_id,
                    span_data=self._state.resumed_tool_span_data,
                    status=SpanStatus.ERROR,
                )
            elif self._state.current_tool_span:
                self._span_factory.end_span_error(self._state.current_tool_span, error)
                self._state.current_tool_span = None
            return

    def _get_guardrail_eval_node_name(self, node_name: str) -> str:
        eval_node_name = node_name
        for suffix in ACTION_SUFFIX_TO_NAME:
            if node_name.endswith(suffix):
                eval_node_name = node_name[: -len(suffix)]
                break
        return eval_node_name

    # --- Telemetry ---

    def _track_guardrail_event(
        self,
        action: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        validation_details: Optional[str] = None,
    ) -> None:
        """Track guardrail telemetry event."""
        props = self._state.enriched_properties.copy()
        props["ActionType"] = action

        if validation_details:
            reason = self._translate_validation_reason(validation_details)
            props["Reason"] = reason

        if metadata:
            props.update(self._build_guardrail_telemetry_props(metadata))

        event_name = None
        if action == GuardrailAction.BLOCK:
            event_name = GuardrailEvent.BLOCKED
        elif action == GuardrailAction.LOG:
            event_name = GuardrailEvent.LOGGED
        elif action == GuardrailAction.SKIP:
            event_name = GuardrailEvent.SKIPPED
        elif action == GuardrailAction.FILTER:
            event_name = GuardrailEvent.FILTERED
        elif action == GuardrailAction.ESCALATE:
            event_name = GuardrailEvent.ESCALATED

        if event_name is None:
            return

        track_event(event_name, props)

    def _track_escalation_outcome_event(
        self,
        event_name: GuardrailEvent,
        metadata: Dict[str, Any],
    ) -> None:
        """Track escalation outcome telemetry event (approved/rejected)."""
        props = self._state.enriched_properties.copy()
        props["ActionType"] = GuardrailAction.ESCALATE

        escalation_data = metadata.get("escalation_data")
        if escalation_data:
            reviewed_inputs = escalation_data.get("reviewed_inputs")
            reviewed_outputs = escalation_data.get("reviewed_outputs")

            if reviewed_inputs is not None or reviewed_outputs is not None:
                props["WasDataModifiedByReviewer"] = True
            else:
                props["WasDataModifiedByReviewer"] = False

        props.update(self._build_guardrail_telemetry_props(metadata))

        track_event(event_name, props)

    def _build_guardrail_telemetry_props(
        self, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build telemetry properties from guardrail metadata."""
        props: Dict[str, Any] = {}

        guardrail = metadata.get("guardrail")
        scope = metadata.get("scope")
        execution_stage = metadata.get("execution_stage")
        severity_level = metadata.get("severity_level")
        escalation_data = metadata.get("escalation_data")
        tool_type = metadata.get("tool_type")

        if guardrail:
            props["EnabledForEvals"] = str(guardrail.enabled_for_evals).lower()
            props["GuardrailScopes"] = json.dumps(
                [s.name.title() for s in guardrail.selector.scopes]
            )
            props["GuardrailType"] = (
                guardrail.guardrail_type[0].upper() + guardrail.guardrail_type[1:]
            )

            if isinstance(guardrail, BuiltInValidatorGuardrail):
                props["ValidatorType"] = "".join(
                    x.title() for x in guardrail.validator_type.split("_")
                )
            elif isinstance(guardrail, DeterministicGuardrail):
                props["NumberOfRules"] = str(len(guardrail.rules))
                props["RuleDetails"] = json.dumps(
                    self._extract_rule_details(guardrail.rules)
                )

        if scope:
            props["CurrentScope"] = scope.name.title()

        if execution_stage:
            props["ExecutionStage"] = execution_stage.name.title().replace("_", "")

        if severity_level:
            props["SeverityLevel"] = severity_level.title()

        if escalation_data:
            props["RecipientType"] = escalation_data.get("recipient_type")

        if tool_type:
            props["ToolType"] = get_tool_type_value(tool_type)

        return props

    def _extract_rule_details(self, rules: list[Any]) -> list[Dict[str, str]]:
        """Extract rule details for telemetry."""
        rule_details = []
        for rule in rules:
            if isinstance(rule, UniversalRule):
                rule_details.append(
                    {
                        "Type": "Always enforce the guardrail",
                        "ApplyTo": rule.apply_to[0].upper() + rule.apply_to[1:],
                    }
                )
                continue

            operator = self._extract_operator_from_description(rule.rule_description)
            rule_details.append(
                {
                    "Type": rule.rule_type.title(),
                    "FieldSelectorType": rule.field_selector.selector_type.title(),
                    "Operator": operator,
                }
            )
        return rule_details

    def _extract_operator_from_description(
        self, rule_description: Optional[str]
    ) -> str:
        """Extract operator from rule description string."""
        if not rule_description:
            return "Unknown"

        known_operators = [
            "doesNotContain",
            "doesNotEqual",
            "doesNotStartWith",
            "doesNotEndWith",
            "greaterThanOrEqual",
            "lessThanOrEqual",
            "greaterThan",
            "lessThan",
            "startsWith",
            "endsWith",
            "contains",
            "equals",
            "isEmpty",
            "isNotEmpty",
            "matchesRegex",
        ]

        for op in known_operators:
            if f" {op} " in rule_description or rule_description.endswith(f" {op}"):
                return op[0].upper() + op[1:]

        return "Unknown"

    def _translate_validation_reason(self, reason: Optional[str]) -> Optional[str]:
        """Translate validation reason to standardized format."""
        if reason and "didn't match" in reason:
            return "RuleDidNotMeet"
        return reason

    def _get_guardrail_validation_result(
        self, source: dict[str, Any]
    ) -> tuple[Optional[bool], Optional[str]]:
        validation_result: Optional[bool] = None
        validation_details: Optional[str] = None
        if isinstance(source, dict) and INNER_STATE_KEY in source:
            validation_result = source[INNER_STATE_KEY].get(
                GUARDRAIL_VALIDATION_RESULT_KEY
            )
            validation_details = source[INNER_STATE_KEY].get(
                GUARDRAIL_VALIDATION_DETAILS_KEY
            )
        elif (
            hasattr(source, "update")
            and isinstance(source.update, dict)
            and INNER_STATE_KEY in source.update
        ):
            validation_result = source.update[INNER_STATE_KEY].get(
                GUARDRAIL_VALIDATION_RESULT_KEY
            )
            validation_details = source.update[INNER_STATE_KEY].get(
                GUARDRAIL_VALIDATION_DETAILS_KEY
            )
        return validation_result, validation_details

    # --- Chain Event Handlers ---

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
        """Handle chain start event (for guardrail nodes)."""
        try:
            if not metadata or metadata.get("guardrail") is None:
                return

            self._state.guardrail_metadata[run_id] = metadata

            node_name = metadata.get("langgraph_node", "")
            if not node_name:
                return

            scope = metadata.get("scope")
            execution_stage = metadata.get("execution_stage")
            guardrail = metadata.get("guardrail")
            if not guardrail:
                return
            guardrail_name = guardrail.name
            node_type = metadata.get("node_type")

            # Determine parent span based on scope
            parent: Optional[Span] = None
            if scope == GuardrailScope.LLM:
                if (
                    execution_stage == ExecutionStage.PRE_EXECUTION
                    and self._state.current_llm_span is None
                ):
                    self._state.current_llm_span = self._span_factory.start_llm_call(
                        parent_span=self._agent_span,
                    )
                    self._state.llm_span_from_guardrail = True
                parent = self._state.current_llm_span or self._agent_span
            elif scope == GuardrailScope.TOOL:
                if (
                    execution_stage == ExecutionStage.PRE_EXECUTION
                    and self._state.current_tool_span is None
                ):
                    tool_name = metadata.get("tool_name") or "unknown"
                    tool_type = metadata.get("tool_type") or None
                    self._state.current_tool_span = self._span_factory.start_tool_call(
                        tool_name=tool_name,
                        tool_type_value=get_tool_type_value(tool_type),
                        parent_span=self._agent_span,
                    )
                    self._state.tool_span_from_guardrail = True
                # Use existing tool span as parent for PRE_EXECUTION
                if execution_stage == ExecutionStage.PRE_EXECUTION:
                    parent = self._state.current_tool_span
                # For POST execution after resume, current_tool_span will be None, and we have to recreate the context based on the span id and trace id
                elif execution_stage == ExecutionStage.POST_EXECUTION:
                    resumed_data = self._state.resumed_tool_span_data
                    trace_id = self._state.resumed_trace_id
                    if resumed_data and trace_id:
                        parent_span_id = resumed_data.get("span_id")
                        if parent_span_id:
                            span_context = SpanContext(
                                trace_id=int(trace_id, 16),
                                span_id=int(parent_span_id, 16),
                                is_remote=True,
                                trace_flags=TraceFlags(0x01),  # Sampled
                            )
                            parent = NonRecordingSpan(span_context)
                    # Fallback to current tool span for non-resume POST_EXECUTION
                    if parent is None and self._state.current_tool_span is not None:
                        parent = self._state.current_tool_span
            else:
                parent = self._state.get_span_or_root(parent_run_id)

            if node_type == "guardrail_evaluation":
                if scope and execution_stage:
                    self._close_previous_phase_containers(scope, execution_stage)

                    guardrails_group_span = self._get_or_create_guardrails_group_span(
                        scope, execution_stage, parent
                    )
                    rule_descriptions: list[str] = []
                    if isinstance(guardrail, DeterministicGuardrail):
                        rule_descriptions = [
                            rule.rule_description
                            for rule in guardrail.rules
                            if hasattr(rule, "rule_description")
                            and rule.rule_description
                        ]
                    eval_span = self._span_factory.start_guardrail_evaluation(
                        guardrail_name=guardrail_name,
                        guardrail_description=guardrail.description,
                        guardrail_action=metadata.get("action_type") or "unknown",
                        rule_details=rule_descriptions,
                        parent_span=guardrails_group_span,
                    )
                    SpanHierarchyManager.push(run_id, eval_span)
            #  Checking if self._state.upcoming_guardrail_actions_info is not None makes this logic to not apply for resumed nodes
            elif node_type == "guardrail_action":
                action = metadata.get("action_type")

                # On resume escalation task
                if (
                    action == GuardrailAction.ESCALATE
                    and self._state.resumed_hitl_guardrail_container_span_data
                    and scope
                    and execution_stage
                ):
                    self._get_or_create_guardrails_group_span(
                        scope, execution_stage, parent
                    )

                eval_node_name = self._get_guardrail_eval_node_name(node_name)
                if eval_node_name not in self._state.upcoming_guardrail_actions_info:
                    return

                eval_span_tuple = self._state.upcoming_guardrail_actions_info.get(
                    eval_node_name
                )
                if not eval_span_tuple:
                    return
                eval_span, _ = eval_span_tuple
                if not eval_span:
                    return
                # Make sure we store the guardrail span for action node run id, so that we can close them accordingly
                # For Block, we will close them with error on_chain_error flow
                SpanHierarchyManager.push(run_id, eval_span)

                # Create "Review task" child span for HITL escalation
                if action == GuardrailAction.ESCALATE and scope:
                    review_task_span = self._span_factory.start_guardrail_escalation(
                        guardrail_name=guardrail_name,
                        scope=scope,
                        parent_span=eval_span,
                    )
                    self._state.pending_escalation_span = review_task_span
                    self._state.pending_hitl_guardrail_span = eval_span
                    # Store the container span for completion after resume
                    if scope and execution_stage:
                        container_key = (scope, execution_stage)
                        if container_key in self._state.guardrail_containers:
                            self._state.pending_hitl_guardrail_container_span = (
                                self._state.guardrail_containers[container_key]
                            )

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
        """Handle chain end event (for guardrail nodes)."""
        try:
            guardrail_metadata = self._state.guardrail_metadata.pop(run_id, None)
            if guardrail_metadata is None:
                return

            node_name = guardrail_metadata.get("langgraph_node", "")
            if not node_name:
                return

            node_type = guardrail_metadata.get("node_type")
            payload = guardrail_metadata.get("payload")

            if node_type == "guardrail_evaluation":
                validation_result, validation_details = (
                    self._get_guardrail_validation_result(outputs)
                )

                validation_passed = validation_result is True

                if validation_passed:
                    span = SpanHierarchyManager.pop(run_id)
                    if span:
                        self._span_factory.end_guardrail_evaluation(
                            span,
                            validation_result=validation_details,
                            action=GuardrailAction.SKIP,
                            payload=payload,
                        )
                    self._track_guardrail_event(
                        GuardrailAction.SKIP, guardrail_metadata, validation_details
                    )
                else:
                    eval_span = SpanHierarchyManager.current(run_id)
                    # pass the eval span, so that on action nodes we can close the span accordingly
                    if eval_span:
                        self._state.upcoming_guardrail_actions_info[node_name] = (
                            eval_span,
                            run_id,
                        )

                        # Update span attributes and upsert without ending the span
                        self._span_factory.upsert_guardrail_evaluation(
                            eval_span,
                            validation_result=validation_details,
                            payload=payload,
                        )
            elif node_type == "guardrail_action":
                self.handle_action_end(run_id, guardrail_metadata)

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
        """Handle chain error event (for guardrail nodes)."""
        try:
            guardrail_metadata = self._state.guardrail_metadata.get(run_id, None)
            if guardrail_metadata is None:
                return
            # GraphInterrupt = suspend signal, spans kept open, but add taskUrl attribute
            if self._is_graph_interrupt(error):
                self._upsert_suspended_escalation(guardrail_metadata)
                self._track_guardrail_event(
                    GuardrailAction.ESCALATE, guardrail_metadata
                )
                return

            self.handle_action_error(run_id, guardrail_metadata, error)
            return

        except Exception:
            logger.exception("Error in on_chain_error callback (guardrail)")

    # --- Escalation Completion ---

    def _upsert_suspended_escalation(
        self,
        metadata: Dict[str, Any],
    ) -> None:
        review_task_span = self._state.pending_escalation_span
        if not review_task_span:
            return

        escalation_data = metadata.get("escalation_data") or {}
        assigned_to = escalation_data.get("assigned_to")
        task_url = escalation_data.get("task_url")

        if assigned_to:
            review_task_span.set_attribute("assignedTo", assigned_to)
        if task_url:
            review_task_span.set_attribute("taskUrl", task_url)

        self._span_factory.upsert_span_suspended(review_task_span)

    def _complete_resumed_escalation_from_outputs(
        self,
        metadata: Dict[str, Any],
        error_str: Optional[str] = None,
    ) -> None:
        """Complete resumed escalation span using saved data."""
        trace_id = self._state.resumed_escalation_trace_id
        review_task_span_data = self._state.resumed_escalation_span_data
        if not trace_id or not review_task_span_data:
            return

        escalation_data = metadata.get("escalation_data") or {}
        assigned_to = escalation_data.get("assigned_to")
        reviewed_by = escalation_data.get("reviewed_by")
        reviewed_inputs = escalation_data.get("reviewed_inputs")
        reviewed_outputs = escalation_data.get("reviewed_outputs")
        reason = escalation_data.get("reason")

        attrs = dict(review_task_span_data.get("attributes", {}))
        attrs["reviewStatus"] = "Completed"
        attrs["reviewOutcome"] = "Rejected" if error_str else "Approved"
        if assigned_to:
            attrs["assignedTo"] = assigned_to
        if reviewed_by:
            attrs["reviewedBy"] = reviewed_by
        if reason:
            attrs["reason"] = reason
        if reviewed_inputs:
            attrs["reviewedInputs"] = to_json_string(reviewed_inputs)
        if reviewed_outputs:
            attrs["reviewedOutputs"] = to_json_string(reviewed_outputs)
        if error_str:
            attrs["error"] = str(error_str)
        updated_span_data = dict(review_task_span_data)
        updated_span_data["attributes"] = attrs

        self._span_factory.upsert_span_complete_by_data(
            trace_id=trace_id,
            span_data=updated_span_data,
            status=SpanStatus.ERROR if error_str else SpanStatus.OK,
        )
