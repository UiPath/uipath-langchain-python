"""Guardrail span instrumentor for LLMOps instrumentation."""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

from opentelemetry.trace import Span
from uipath.core.guardrails import UniversalRule
from uipath.platform.guardrails import BuiltInValidatorGuardrail, DeterministicGuardrail

from ...event_emitter import GuardrailEvent, track_event
from ..span_hierarchy import SpanHierarchyManager
from ..spans.span_name import (
    GUARDRAIL_VALIDATION_DETAILS_KEY,
    GUARDRAIL_VALIDATION_RESULT_KEY,
    INNER_STATE_KEY,
)
from .base import BaseSpanInstrumentor, InstrumentationState
from .constants import (
    ACTION_SUFFIX_TO_NAME,
    ACTION_TYPE_TO_ACTION,
    GUARDRAIL_NODE_PATTERN,
    GuardrailAction,
    GuardrailScope,
    GuardrailStage,
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

    def _get_or_create_container(
        self, scope: str, stage: str, parent_span: Optional[Span]
    ) -> Span:
        """Get or create a guardrail container span."""
        key = (scope, stage)
        if key not in self._state.guardrail_containers:
            container = self._span_factory.start_guardrails_container(
                scope, stage, parent_span
            )
            self._state.guardrail_containers[key] = container
        return self._state.guardrail_containers[key]

    def close_container(self, scope: str, stage: str) -> None:
        """Close a guardrail container span."""
        key = (scope, stage)
        if key in self._state.guardrail_containers:
            container = self._state.guardrail_containers.pop(key)
            self._span_factory.end_span_ok(container)

        # Clear tool span after post guardrails complete
        if (
            scope == GuardrailScope.TOOL
            and stage == GuardrailStage.POST
            and self._state.tool_ended_pending_post
        ):
            self._state.current_tool_span = None
            self._state.tool_ended_pending_post = False

    def cleanup_containers(self) -> None:
        """Close all remaining open guardrail container spans."""
        for key in list(self._state.guardrail_containers.keys()):
            scope, stage = key
            self.close_container(scope, stage)

    def _close_previous_phase_containers(
        self, current_scope: str, current_stage: str
    ) -> None:
        """Close containers from previous phases when transitioning."""
        S, T = GuardrailScope, GuardrailStage
        if current_scope == S.LLM and current_stage == T.PRE:
            self.close_container(S.AGENT, T.PRE)
            self.close_container(S.TOOL, T.POST)
        elif current_scope == S.LLM and current_stage == T.POST:
            self.close_container(S.LLM, T.PRE)
        elif current_scope == S.TOOL and current_stage == T.PRE:
            self.close_container(S.LLM, T.POST)
        elif current_scope == S.TOOL and current_stage == T.POST:
            self.close_container(S.TOOL, T.PRE)
        elif current_scope == S.AGENT and current_stage == T.POST:
            self.close_container(S.LLM, T.POST)
            self.close_container(S.TOOL, T.POST)

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

    def _check_and_handle_action_node(
        self,
        node_name: str,
        run_id: UUID,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Check if node is an action node and end the pending guardrail span."""
        if not metadata or "action_type" not in metadata:
            return False

        action = ACTION_TYPE_TO_ACTION.get(metadata["action_type"])
        if action is None:
            return False

        # Extract evaluation node name by removing the action suffix
        eval_node_name = node_name
        for suffix in ACTION_SUFFIX_TO_NAME:
            if node_name.endswith(suffix):
                eval_node_name = node_name[: -len(suffix)]
                break

        # Resume scenario: action node but no pending guardrail
        if eval_node_name not in self._state.pending_guardrail_actions:
            if (
                action == GuardrailAction.ESCALATE
                and self._state.resumed_escalation_trace_id
                and self._state.resumed_escalation_span_data
            ):
                self._state.escalate_action_resume_data[run_id] = {
                    "trace_id": self._state.resumed_escalation_trace_id,
                    "span_data": self._state.resumed_escalation_span_data,
                }
                self._state.resumed_escalation_trace_id = None
                self._state.resumed_escalation_span_data = None
            return True

        # Normal flow: complete the pending guardrail span
        span, validation_details = self._state.pending_guardrail_actions.pop(
            eval_node_name
        )

        severity_level = metadata.get("severity_level") if metadata else None
        reason = metadata.get("reason") if metadata else None

        self._span_factory.end_guardrail_evaluation(
            span,
            validation_passed=False,
            validation_result=validation_details,
            action=action,
            severity_level=severity_level,
            reason=reason,
        )

        # Create "Review task" child span for HITL escalation
        if action == GuardrailAction.ESCALATE:
            scope, _ = self._extract_scope_stage(eval_node_name)
            guardrail_name = (
                eval_node_name.split("_execution_")[-1]
                if "_execution_" in eval_node_name
                else eval_node_name
            )
            escalation_span = self._span_factory.start_guardrail_escalation(
                guardrail_name=guardrail_name,
                scope=scope or "agent",
                parent_span=span,
            )
            self._state.pending_escalation_span = escalation_span
            self._state.pending_escalation_info = {
                "guardrail_name": guardrail_name,
                "scope": scope or "agent",
            }
            self._state.escalate_action_run_ids[run_id] = escalation_span

        # If tool_pre guardrail blocks, end the placeholder tool span
        if (
            action == GuardrailAction.BLOCK
            and self._state.tool_span_from_guardrail
            and "tool_pre" in eval_node_name
        ):
            self.close_container(GuardrailScope.TOOL, GuardrailStage.PRE)
            if self._state.current_tool_span:
                self._state.current_tool_span.set_attribute(
                    "output", "Blocked by guardrail"
                )
                self._state.current_tool_span.end()
                self._state.current_tool_span = None
            self._state.tool_span_from_guardrail = False

        return True

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

        if event_name is None:
            return

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

        return props

    def _extract_rule_details(self, rules: List[Any]) -> List[Dict[str, str]]:
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
            if not metadata:
                return

            # Store guardrail metadata for telemetry
            if metadata.get("guardrail") is not None:
                self._state.guardrail_metadata[run_id] = metadata

            node_name = metadata.get("langgraph_node", "")
            if not node_name:
                return

            # Check if this is an action node
            if self._check_and_handle_action_node(node_name, run_id, metadata):
                return

            guardrail_info = self._parse_guardrail_node(node_name)
            if not guardrail_info:
                return

            scope, stage, guardrail_name = guardrail_info
            self._close_previous_phase_containers(scope, stage)

            # Determine parent span based on scope
            if scope == GuardrailScope.LLM:
                if stage == GuardrailStage.PRE:
                    if not self._state.llm_span_from_guardrail:
                        self._state.current_llm_span = (
                            self._span_factory.start_llm_call(
                                max_tokens=None,
                                temperature=None,
                                parent_span=self._agent_span,
                            )
                        )
                        self._state.llm_span_from_guardrail = True
                parent = self._state.current_llm_span or self._agent_span
            elif scope == GuardrailScope.TOOL:
                if stage == GuardrailStage.PRE and not self._state.current_tool_span:
                    self._state.current_tool_span = self._span_factory.start_tool_call(
                        tool_name="Tool call",
                        parent_span=self._agent_span,
                    )
                    self._state.tool_span_from_guardrail = True
                parent = self._state.current_tool_span or self._agent_span
            else:
                parent = self._state.get_span_or_root(parent_run_id)

            container = self._get_or_create_container(scope, stage, parent)
            eval_span = self._span_factory.start_guardrail_evaluation(
                guardrail_name=guardrail_name,
                scope=scope,
                parent_span=container,
            )
            self._state.guardrail_spans[run_id] = eval_span
            self._state.guardrail_info[run_id] = (scope, stage, guardrail_name)
            SpanHierarchyManager.push(run_id, eval_span)

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
            # Handle escalate action nodes
            if run_id in self._state.escalate_action_run_ids:
                self._complete_escalation_from_outputs(run_id, outputs)
                return

            # Handle resume escalate action nodes
            if run_id in self._state.escalate_action_resume_data:
                self._complete_resumed_escalation_from_outputs(run_id, outputs)
                return

            guardrail_metadata = self._state.guardrail_metadata.get(run_id)
            if guardrail_metadata is not None:
                self._track_guardrail_event(
                    guardrail_metadata.get("action_type"),
                    guardrail_metadata,
                )

            if run_id not in self._state.guardrail_spans:
                return

            SpanHierarchyManager.pop(run_id)
            span = self._state.guardrail_spans.pop(run_id)
            info = self._state.guardrail_info.pop(run_id, None)
            metadata = self._state.guardrail_metadata.pop(run_id, None)

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
                self._span_factory.end_guardrail_evaluation(
                    span,
                    validation_passed=True,
                    validation_result=validation_details,
                    action=GuardrailAction.SKIP,
                )
                self._track_guardrail_event(
                    GuardrailAction.SKIP, metadata, validation_details
                )
            else:
                if info:
                    scope, stage, guardrail_name = info
                    eval_node_name = f"{scope}_{stage}_execution_{guardrail_name}"
                    self._state.pending_guardrail_actions[eval_node_name] = (
                        span,
                        validation_details,
                    )
                else:
                    self._span_factory.end_guardrail_evaluation(
                        span,
                        validation_passed=False,
                        validation_result=validation_details,
                        action=GuardrailAction.LOG,
                    )

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
            if run_id not in self._state.guardrail_spans:
                return

            SpanHierarchyManager.pop(run_id)
            span = self._state.guardrail_spans.pop(run_id)
            self._state.guardrail_info.pop(run_id, None)

            exc = error if isinstance(error, Exception) else Exception(str(error))
            self._span_factory.end_span_error(span, exc)

        except Exception:
            logger.exception("Error in on_chain_error callback (guardrail)")

    # --- Escalation Completion ---

    def _complete_escalation_from_outputs(
        self,
        run_id: UUID,
        outputs: Dict[str, Any],
    ) -> None:
        """Complete escalation span using reviewed data from node outputs."""
        escalation_span = self._state.escalate_action_run_ids.pop(run_id, None)
        if not escalation_span:
            return

        reviewed_inputs = None
        reviewed_outputs = None
        reviewed_by = None

        logger.info(
            f"Completing escalation span from outputs: "
            f"reviewed_inputs={reviewed_inputs}, reviewed_outputs={reviewed_outputs}"
        )

        self._span_factory.end_guardrail_escalation(
            escalation_span,
            review_outcome="Approved",
            reviewed_by=reviewed_by,
            reviewed_inputs=reviewed_inputs,
            reviewed_outputs=reviewed_outputs,
        )

        if escalation_span == self._state.pending_escalation_span:
            self._state.pending_escalation_span = None
            self._state.pending_escalation_info = None

    def _complete_resumed_escalation_from_outputs(
        self,
        run_id: UUID,
        outputs: Dict[str, Any],
    ) -> None:
        """Complete resumed escalation span using saved data."""
        resume_data = self._state.escalate_action_resume_data.pop(run_id, None)
        if not resume_data:
            return

        trace_id = resume_data.get("trace_id")
        span_data = resume_data.get("span_data")
        if not trace_id or not span_data:
            return

        reviewed_inputs = None
        reviewed_outputs = None
        reviewed_by = None

        logger.info(
            f"Completing resumed escalation span from outputs: "
            f"reviewed_inputs={reviewed_inputs}, reviewed_outputs={reviewed_outputs}"
        )

        attrs = dict(span_data.get("attributes", {}))
        attrs["reviewStatus"] = "completed"
        attrs["reviewOutcome"] = "Approved"
        if reviewed_by:
            attrs["reviewedBy"] = reviewed_by
        if reviewed_inputs:
            attrs["reviewedInputs"] = (
                json.dumps(reviewed_inputs)
                if isinstance(reviewed_inputs, dict)
                else str(reviewed_inputs)
            )
        if reviewed_outputs:
            attrs["reviewedOutputs"] = (
                json.dumps(reviewed_outputs)
                if isinstance(reviewed_outputs, dict)
                else str(reviewed_outputs)
            )

        updated_span_data = dict(span_data)
        updated_span_data["attributes"] = attrs

        self._span_factory.upsert_span_complete_by_data(
            trace_id=trace_id,
            span_data=updated_span_data,
        )
