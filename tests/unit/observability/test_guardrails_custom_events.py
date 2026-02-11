"""Tests for LlmOpsInstrumentationCallback LangChain callback handler."""

from unittest.mock import MagicMock, patch
from uuid import uuid4

from langgraph.errors import GraphInterrupt
from langgraph.types import Interrupt
from uipath.core.guardrails import (
    AllFieldsSelector,
    FieldReference,
    FieldSource,
    GuardrailScope,
    SpecificFieldsSelector,
)
from uipath_langchain.agent.guardrails.types import ExecutionStage

from uipath_agents._observability.llmops.spans.span_name import (
    GUARDRAIL_VALIDATION_DETAILS_KEY,
    GUARDRAIL_VALIDATION_RESULT_KEY,
    INNER_STATE_KEY,
)

# span_exporter and callback fixture comes from conftest.py


class TestGuardrailTelemetryEvents:
    """Tests for guardrail telemetry event tracking."""

    def test_skip_action_tracks_guardrail_skipped_event(
        self, callback, tracer, span_exporter
    ):
        """When validation passes (Skip), Guardrail.Skipped event should be tracked."""
        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)
            callback.set_enriched_properties({"AgentName": "TestAgent"})

            # Create mock guardrail for metadata
            mock_guardrail = MagicMock()
            mock_guardrail.name = "pii_guard"
            mock_guardrail.description = "PII Guard"
            mock_guardrail.enabled_for_evals = True
            mock_guardrail.guardrail_type = "builtInValidator"
            mock_guardrail.selector = MagicMock()
            mock_guardrail.selector.scopes = [GuardrailScope.AGENT]

            with patch(
                "uipath_agents._observability.llmops.instrumentors.guardrail_instrumentor.track_event"
            ) as mock_track:
                run_id = uuid4()
                callback.on_chain_start(
                    {},
                    {},
                    run_id=run_id,
                    metadata={
                        "langgraph_node": "agent_pre_execution_pii_guard",
                        "guardrail": mock_guardrail,
                        "node_type": "guardrail_evaluation",
                        "scope": GuardrailScope.AGENT,
                        "execution_stage": ExecutionStage.PRE_EXECUTION,
                        "action_type": "Skip",
                    },
                )
                callback.on_chain_end(
                    {
                        INNER_STATE_KEY: {
                            GUARDRAIL_VALIDATION_RESULT_KEY: True,
                            GUARDRAIL_VALIDATION_DETAILS_KEY: "No PII found",
                        }
                    },
                    run_id=run_id,
                )

                # Verify Guardrail.Skipped event was tracked
                mock_track.assert_called_once()
                call_args = mock_track.call_args
                assert call_args[0][0] == "Guardrail.Skipped"
                props = call_args[0][1]
                assert props["ActionType"] == "Skip"
                assert props["AgentName"] == "TestAgent"
                assert props.get("Reason") == "No PII found"
                # ToolType should not be present for non-TOOL scope guardrails
                assert "ToolType" not in props

    def test_block_action_tracks_guardrail_blocked_event(
        self, callback, tracer, span_exporter
    ):
        """When validation fails with Block, Guardrail.Blocked event should be tracked."""
        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)
            callback.set_enriched_properties({"AgentName": "TestAgent"})

            # Create mock guardrail for metadata
            mock_guardrail = MagicMock()
            mock_guardrail.name = "pii_guard"
            mock_guardrail.description = "PII Guard"
            mock_guardrail.enabled_for_evals = True
            mock_guardrail.guardrail_type = "builtInValidator"
            mock_guardrail.selector = MagicMock()
            mock_guardrail.selector.scopes = [GuardrailScope.AGENT]

            with patch(
                "uipath_agents._observability.llmops.instrumentors.guardrail_instrumentor.track_event"
            ) as mock_track:
                run_id = uuid4()
                callback.on_chain_start(
                    {},
                    {},
                    run_id=run_id,
                    metadata={
                        "langgraph_node": "agent_pre_execution_pii_guard",
                        "guardrail": mock_guardrail,
                        "node_type": "guardrail_evaluation",
                        "scope": GuardrailScope.AGENT,
                        "execution_stage": ExecutionStage.PRE_EXECUTION,
                        "action_type": "Block",
                    },
                )
                callback.on_chain_end(
                    {
                        INNER_STATE_KEY: {
                            GUARDRAIL_VALIDATION_RESULT_KEY: False,
                            GUARDRAIL_VALIDATION_DETAILS_KEY: "PII detected",
                        }
                    },
                    run_id=run_id,
                )

                # Action node fires with _block suffix
                action_run_id = uuid4()
                callback.on_chain_start(
                    {},
                    {},
                    run_id=action_run_id,
                    metadata={
                        "langgraph_node": "agent_pre_execution_pii_guard_block",
                        "action_type": "Block",
                        "guardrail": mock_guardrail,
                        "node_type": "guardrail_action",
                        "scope": GuardrailScope.AGENT,
                        "execution_stage": ExecutionStage.PRE_EXECUTION,
                    },
                )
                # Event is tracked on chain end
                callback.on_chain_end({}, run_id=action_run_id)

                # Verify Guardrail.Blocked event was tracked
                mock_track.assert_called_once()
                call_args = mock_track.call_args
                assert call_args[0][0] == "Guardrail.Blocked"
                props = call_args[0][1]
                assert props["ActionType"] == "Block"
                assert props["AgentName"] == "TestAgent"

    def test_log_action_tracks_guardrail_logged_event(
        self, callback, tracer, span_exporter
    ):
        """When validation fails with Log, Guardrail.Logged event should be tracked."""
        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)
            callback.set_enriched_properties({"AgentName": "TestAgent"})

            # Create mock guardrail for metadata
            mock_guardrail = MagicMock()
            mock_guardrail.name = "prompt_injection"
            mock_guardrail.description = "Prompt Injection Guard"
            mock_guardrail.enabled_for_evals = True
            mock_guardrail.guardrail_type = "builtInValidator"
            mock_guardrail.selector = MagicMock()
            mock_guardrail.selector.scopes = [GuardrailScope.LLM]

            with patch(
                "uipath_agents._observability.llmops.instrumentors.guardrail_instrumentor.track_event"
            ) as mock_track:
                run_id = uuid4()
                callback.on_chain_start(
                    {},
                    {},
                    run_id=run_id,
                    metadata={
                        "langgraph_node": "llm_pre_execution_prompt_injection",
                        "guardrail": mock_guardrail,
                        "node_type": "guardrail_evaluation",
                        "scope": GuardrailScope.LLM,
                        "execution_stage": ExecutionStage.PRE_EXECUTION,
                        "action_type": "Log",
                    },
                )
                callback.on_chain_end(
                    {INNER_STATE_KEY: {GUARDRAIL_VALIDATION_RESULT_KEY: False}},
                    run_id=run_id,
                )

                # Action node fires with _log suffix
                action_run_id = uuid4()
                callback.on_chain_start(
                    {},
                    {},
                    run_id=action_run_id,
                    metadata={
                        "langgraph_node": "llm_pre_execution_prompt_injection_log",
                        "severity_level": "Info",
                        "action_type": "Log",
                        "guardrail": mock_guardrail,
                        "node_type": "guardrail_action",
                        "scope": GuardrailScope.LLM,
                        "execution_stage": ExecutionStage.PRE_EXECUTION,
                    },
                )
                # Event is tracked on chain end
                callback.on_chain_end({}, run_id=action_run_id)

                mock_track.assert_called_once()
                call_args = mock_track.call_args
                assert call_args[0][0] == "Guardrail.Logged"
                props = call_args[0][1]
                assert props["ActionType"] == "Log"
                assert props["SeverityLevel"] == "Info"

    def test_filter_action_tracks_guardrail_filtered_event(
        self, callback, tracer, span_exporter
    ):
        """When validation fails with Filter, Guardrail.Filtered event should be tracked."""
        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)
            callback.set_enriched_properties({"AgentName": "TestAgent"})

            # Create mock guardrail for metadata
            mock_guardrail = MagicMock()
            mock_guardrail.name = "pii"
            mock_guardrail.description = "PII Filter"
            mock_guardrail.enabled_for_evals = True
            mock_guardrail.guardrail_type = "builtInValidator"
            mock_guardrail.selector = MagicMock()
            mock_guardrail.selector.scopes = [GuardrailScope.TOOL]

            with patch(
                "uipath_agents._observability.llmops.instrumentors.guardrail_instrumentor.track_event"
            ) as mock_track:
                run_id = uuid4()
                callback.on_chain_start(
                    {},
                    {},
                    run_id=run_id,
                    metadata={
                        "langgraph_node": "tool_post_execution_pii",
                        "guardrail": mock_guardrail,
                        "node_type": "guardrail_evaluation",
                        "scope": GuardrailScope.TOOL,
                        "execution_stage": ExecutionStage.POST_EXECUTION,
                        "action_type": "Filter",
                        "tool_type": "process",
                    },
                )
                callback.on_chain_end(
                    {INNER_STATE_KEY: {GUARDRAIL_VALIDATION_RESULT_KEY: False}},
                    run_id=run_id,
                )

                # Action node fires with _filter suffix
                action_run_id = uuid4()
                callback.on_chain_start(
                    {},
                    {},
                    run_id=action_run_id,
                    metadata={
                        "langgraph_node": "tool_post_execution_pii_filter",
                        "action_type": "Filter",
                        "guardrail": mock_guardrail,
                        "node_type": "guardrail_action",
                        "scope": GuardrailScope.TOOL,
                        "execution_stage": ExecutionStage.POST_EXECUTION,
                        "tool_type": "process",
                    },
                )
                # Event is tracked on chain end
                callback.on_chain_end({}, run_id=action_run_id)

                mock_track.assert_called_once()
                call_args = mock_track.call_args
                assert call_args[0][0] == "Guardrail.Filtered"
                props = call_args[0][1]
                assert props["ActionType"] == "Filter"
                assert props["AgentName"] == "TestAgent"
                assert props["ToolType"] == "Process"

    def test_enriched_properties_included_in_event(
        self, callback, tracer, span_exporter
    ):
        """Enriched properties from runtime should be included in guardrail events."""
        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)
            callback.set_enriched_properties(
                {
                    "AgentName": "MyAgent",
                    "AgentId": "agent-123",
                    "Model": "gpt-4o",
                    "CloudOrganizationId": "org-456",
                }
            )

            # Create mock guardrail for metadata
            mock_guardrail = MagicMock()
            mock_guardrail.name = "guard"
            mock_guardrail.description = "Test Guard"
            mock_guardrail.enabled_for_evals = True
            mock_guardrail.guardrail_type = "builtInValidator"
            mock_guardrail.selector = MagicMock()
            mock_guardrail.selector.scopes = [GuardrailScope.AGENT]

            with patch(
                "uipath_agents._observability.llmops.instrumentors.guardrail_instrumentor.track_event"
            ) as mock_track:
                run_id = uuid4()
                callback.on_chain_start(
                    {},
                    {},
                    run_id=run_id,
                    metadata={
                        "langgraph_node": "agent_pre_execution_guard",
                        "guardrail": mock_guardrail,
                        "node_type": "guardrail_evaluation",
                        "scope": GuardrailScope.AGENT,
                        "execution_stage": ExecutionStage.PRE_EXECUTION,
                        "action_type": "Skip",
                    },
                )
                callback.on_chain_end(
                    {INNER_STATE_KEY: {GUARDRAIL_VALIDATION_RESULT_KEY: True}},
                    run_id=run_id,
                )

                props = mock_track.call_args[0][1]
                assert props["AgentName"] == "MyAgent"
                assert props["AgentId"] == "agent-123"
                assert props["Model"] == "gpt-4o"
                assert props["CloudOrganizationId"] == "org-456"

    def test_builtin_validator_guardrail_metadata_enrichment(
        self, callback, tracer, span_exporter
    ):
        """BuiltInValidatorGuardrail metadata should enrich telemetry properties."""
        from uipath.platform.guardrails import BuiltInValidatorGuardrail

        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)

            # Create mock BuiltInValidatorGuardrail
            mock_guardrail = MagicMock(spec=BuiltInValidatorGuardrail)
            mock_guardrail.name = "pii_guard"
            mock_guardrail.description = "PII Guard"
            mock_guardrail.enabled_for_evals = True
            mock_guardrail.guardrail_type = "builtIn"
            mock_guardrail.validator_type = "pii_detection"
            mock_guardrail.selector = MagicMock()
            mock_guardrail.selector.scopes = [GuardrailScope.AGENT, GuardrailScope.LLM]

            with patch(
                "uipath_agents._observability.llmops.instrumentors.guardrail_instrumentor.track_event"
            ) as mock_track:
                run_id = uuid4()
                callback.on_chain_start(
                    {},
                    {},
                    run_id=run_id,
                    metadata={
                        "langgraph_node": "agent_pre_execution_pii_guard",
                        "guardrail": mock_guardrail,
                        "node_type": "guardrail_evaluation",
                        "scope": GuardrailScope.AGENT,
                        "execution_stage": ExecutionStage.PRE_EXECUTION,
                        "action_type": "Skip",
                    },
                )
                callback.on_chain_end(
                    {INNER_STATE_KEY: {GUARDRAIL_VALIDATION_RESULT_KEY: True}},
                    run_id=run_id,
                )

                props = mock_track.call_args[0][1]
                assert props["EnabledForEvals"] == "true"
                assert props["GuardrailType"] == "BuiltIn"
                assert props["ValidatorType"] == "PiiDetection"
                assert props["CurrentScope"] == "Agent"
                assert "Agent" in props["GuardrailScopes"]
                assert "Llm" in props["GuardrailScopes"]

    def test_escalate_action_tracks_guardrail_escalated_event(
        self, callback, tracer, span_exporter
    ):
        """When escalation triggers a GraphInterrupt, Guardrail.Escalated event should be tracked."""
        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)
            callback.set_enriched_properties({"AgentName": "TestAgent"})

            # Create mock guardrail for metadata
            mock_guardrail = MagicMock()
            mock_guardrail.name = "pii_guard"
            mock_guardrail.description = "PII Guard"
            mock_guardrail.enabled_for_evals = True
            mock_guardrail.guardrail_type = "builtInValidator"
            mock_guardrail.selector = MagicMock()
            mock_guardrail.selector.scopes = [GuardrailScope.AGENT]

            with patch(
                "uipath_agents._observability.llmops.instrumentors.guardrail_instrumentor.track_event"
            ) as mock_track:
                # Evaluation node fires and fails validation
                run_id = uuid4()
                callback.on_chain_start(
                    {},
                    {},
                    run_id=run_id,
                    metadata={
                        "langgraph_node": "agent_pre_execution_pii_guard",
                        "guardrail": mock_guardrail,
                        "node_type": "guardrail_evaluation",
                        "scope": GuardrailScope.AGENT,
                        "execution_stage": ExecutionStage.PRE_EXECUTION,
                        "action_type": "Escalate",
                    },
                )
                callback.on_chain_end(
                    {INNER_STATE_KEY: {GUARDRAIL_VALIDATION_RESULT_KEY: False}},
                    run_id=run_id,
                )

                # Action node fires with _hitl suffix (initial escalation)
                action_run_id = uuid4()
                callback.on_chain_start(
                    {},
                    {},
                    run_id=action_run_id,
                    metadata={
                        "langgraph_node": "agent_pre_execution_pii_guard_hitl",
                        "action_type": "Escalate",
                        "guardrail": mock_guardrail,
                        "node_type": "guardrail_action",
                        "scope": GuardrailScope.AGENT,
                        "execution_stage": ExecutionStage.PRE_EXECUTION,
                    },
                )

                # Event is NOT tracked yet (only after interrupt)
                mock_track.assert_not_called()

                # GraphInterrupt fires — this is when the Escalated event is sent
                callback.on_chain_error(
                    GraphInterrupt([Interrupt(value="Suspended for HITL")]),
                    run_id=action_run_id,
                )

                # Verify Guardrail.Escalated event was tracked after interrupt
                mock_track.assert_called_once()
                call_args = mock_track.call_args
                assert call_args[0][0] == "Guardrail.Escalated"
                props = call_args[0][1]
                assert props["ActionType"] == "Escalate"
                assert props["AgentName"] == "TestAgent"

    def test_escalation_approved_tracks_guardrail_escalation_approved_event(
        self, callback, tracer, span_exporter
    ):
        """When escalation is approved on resume, Guardrail.EscalationApproved event should be tracked."""
        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)
            callback.set_enriched_properties({"AgentName": "TestAgent"})

            # Create mock guardrail
            mock_guardrail = MagicMock()
            mock_guardrail.name = "pii_guard"
            mock_guardrail.description = "PII Guard"
            mock_guardrail.enabled_for_evals = True
            mock_guardrail.guardrail_type = "deterministic"
            mock_guardrail.selector = MagicMock()
            mock_guardrail.selector.scopes = [GuardrailScope.AGENT]

            # Simulate resumed escalation context
            callback._state.resumed_escalation_trace_id = "0123456789abcdef"
            callback._state.resumed_escalation_span_data = {
                "name": "Review task",
                "span_id": "abcd1234",
                "attributes": {
                    "type": "guardrailEscalation",
                    "reviewStatus": "Pending",
                },
            }
            callback._state.resumed_hitl_guardrail_span_data = {
                "name": "pii_guard",
                "span_id": "eval5678",
                "attributes": {},
            }

            with patch(
                "uipath_agents._observability.llmops.instrumentors.guardrail_instrumentor.track_event"
            ) as mock_track:
                # Resume: Escalate action node fires on resume
                action_run_id = uuid4()
                callback.on_chain_start(
                    {},
                    {},
                    run_id=action_run_id,
                    metadata={
                        "langgraph_node": "agent_pre_execution_pii_guard_hitl",
                        "guardrail": mock_guardrail,
                        "node_type": "guardrail_action",
                        "scope": GuardrailScope.AGENT,
                        "execution_stage": ExecutionStage.PRE_EXECUTION,
                        "action_type": "Escalate",
                        "escalation_data": {
                            "reviewed_by": "reviewer@example.com",
                            "recipient_type": "UserEmail",
                            "reviewed_inputs": {"input": "sanitized"},
                        },
                    },
                )

                with patch.object(
                    callback._state.span_factory, "upsert_span_complete_by_data"
                ):
                    # on_chain_end - HITL approved (no error)
                    callback.on_chain_end(
                        {},
                        run_id=action_run_id,
                    )

                # Verify Guardrail.EscalationApproved event was tracked
                mock_track.assert_called_once()
                call_args = mock_track.call_args
                assert call_args[0][0] == "Guardrail.EscalationApproved"
                props = call_args[0][1]
                assert props["ActionType"] == "Escalate"
                assert props["AgentName"] == "TestAgent"
                assert props["RecipientType"] == "UserEmail"
                assert props["WasDataModifiedByReviewer"]

    def test_escalation_rejected_tracks_guardrail_escalation_rejected_event(
        self, callback, tracer, span_exporter
    ):
        """When escalation is rejected on resume, Guardrail.EscalationRejected event should be tracked."""
        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)
            callback.set_enriched_properties({"AgentName": "TestAgent"})

            # Create mock guardrail
            mock_guardrail = MagicMock()
            mock_guardrail.name = "pii_guard"
            mock_guardrail.description = "PII Guard"
            mock_guardrail.enabled_for_evals = True
            mock_guardrail.guardrail_type = "deterministic"
            mock_guardrail.selector = MagicMock()
            mock_guardrail.selector.scopes = [GuardrailScope.AGENT]

            # Simulate resumed escalation context
            callback._state.resumed_escalation_trace_id = "0123456789abcdef"
            callback._state.resumed_escalation_span_data = {
                "name": "Review task",
                "span_id": "abcd1234",
                "attributes": {
                    "type": "guardrailEscalation",
                    "reviewStatus": "Pending",
                },
            }
            callback._state.resumed_hitl_guardrail_span_data = {
                "name": "pii_guard",
                "span_id": "eval5678",
                "attributes": {},
            }

            # Resume: Escalate action node fires on resume
            action_run_id = uuid4()
            callback.on_chain_start(
                {},
                {},
                run_id=action_run_id,
                metadata={
                    "langgraph_node": "agent_pre_execution_pii_guard_hitl",
                    "guardrail": mock_guardrail,
                    "node_type": "guardrail_action",
                    "scope": GuardrailScope.AGENT,
                    "execution_stage": ExecutionStage.PRE_EXECUTION,
                    "action_type": "Escalate",
                    "escalation_data": {
                        "recipient_type": "UserEmail",
                        "reviewed_by": "reviewer@example.com",
                    },
                },
            )

            with patch(
                "uipath_agents._observability.llmops.instrumentors.guardrail_instrumentor.track_event"
            ) as mock_track:
                with patch.object(
                    callback._state.span_factory, "upsert_span_complete_by_data"
                ):
                    # on_chain_error - HITL rejected
                    callback.on_chain_error(
                        Exception("User rejected: Invalid data"),
                        run_id=action_run_id,
                    )

                # Verify Guardrail.EscalationRejected event was tracked
                mock_track.assert_called_once()
                call_args = mock_track.call_args
                assert call_args[0][0] == "Guardrail.EscalationRejected"
                props = call_args[0][1]
                assert props["ActionType"] == "Escalate"
                assert props["AgentName"] == "TestAgent"
                assert props["RecipientType"] == "UserEmail"
                assert not props["WasDataModifiedByReviewer"]


class TestRuleDetailsExtraction:
    """Tests for rule details extraction for guardrail telemetry events."""

    def test_translate_validation_reason_didnt_match(self, callback):
        """Test that 'didn't match' is translated to 'RuleDidNotMeet'."""
        result = callback._guardrail_instrumentor._translate_validation_reason(
            "Field 'name' didn't match the expected pattern"
        )
        assert result == "RuleDidNotMeet"

    def test_translate_validation_reason_no_translation(self, callback):
        """Test that reasons without 'didn't match' are returned as-is."""
        result = callback._guardrail_instrumentor._translate_validation_reason(
            "No PII found"
        )
        assert result == "No PII found"

    def test_translate_validation_reason_none(self, callback):
        """Test that None is returned as-is."""
        result = callback._guardrail_instrumentor._translate_validation_reason(None)
        assert result is None

    def test_extract_operator_from_description_contains(self, callback):
        """Test extraction of 'contains' operator from rule description."""
        result = callback._guardrail_instrumentor._extract_operator_from_description(
            "message.content contains 'forbidden'"
        )
        assert result == "Contains"

    def test_extract_operator_from_description_does_not_contain(self, callback):
        """Test extraction of 'doesNotContain' operator from rule description."""
        result = callback._guardrail_instrumentor._extract_operator_from_description(
            "message.content doesNotContain 'allowed'"
        )
        assert result == "DoesNotContain"

    def test_extract_operator_from_description_greater_than(self, callback):
        """Test extraction of 'greaterThan' operator from rule description."""
        result = callback._guardrail_instrumentor._extract_operator_from_description(
            "data.count greaterThan 10.0"
        )
        assert result == "GreaterThan"

    def test_extract_operator_from_description_greater_than_or_equal(self, callback):
        """Test extraction of 'greaterThanOrEqual' operator from rule description."""
        result = callback._guardrail_instrumentor._extract_operator_from_description(
            "data.value greaterThanOrEqual 5.0"
        )
        assert result == "GreaterThanOrEqual"

    def test_extract_operator_from_description_less_than(self, callback):
        """Test extraction of 'lessThan' operator from rule description."""
        result = callback._guardrail_instrumentor._extract_operator_from_description(
            "data.count lessThan 100"
        )
        assert result == "LessThan"

    def test_extract_operator_from_description_equals(self, callback):
        """Test extraction of 'equals' operator from rule description."""
        result = callback._guardrail_instrumentor._extract_operator_from_description(
            "All fields equals 'test'"
        )
        assert result == "Equals"

    def test_extract_operator_from_description_is_empty(self, callback):
        """Test extraction of 'isEmpty' operator (no value) from rule description."""
        result = callback._guardrail_instrumentor._extract_operator_from_description(
            "message.content isEmpty"
        )
        assert result == "IsEmpty"

    def test_extract_operator_from_description_is_not_empty(self, callback):
        """Test extraction of 'isNotEmpty' operator from rule description."""
        result = callback._guardrail_instrumentor._extract_operator_from_description(
            "message.content isNotEmpty"
        )
        assert result == "IsNotEmpty"

    def test_extract_operator_from_description_starts_with(self, callback):
        """Test extraction of 'startsWith' operator from rule description."""
        result = callback._guardrail_instrumentor._extract_operator_from_description(
            "data.prefix startsWith 'hello'"
        )
        assert result == "StartsWith"

    def test_extract_operator_from_description_ends_with(self, callback):
        """Test extraction of 'endsWith' operator from rule description."""
        result = callback._guardrail_instrumentor._extract_operator_from_description(
            "data.suffix endsWith 'world'"
        )
        assert result == "EndsWith"

    def test_extract_operator_from_description_matches_regex(self, callback):
        """Test extraction of 'matchesRegex' operator from rule description."""
        result = callback._guardrail_instrumentor._extract_operator_from_description(
            "data.pattern matchesRegex '^[0-9]+$'"
        )
        assert result == "MatchesRegex"

    def test_extract_operator_from_description_none(self, callback):
        """Test extraction returns 'Unknown' for None description."""
        result = callback._guardrail_instrumentor._extract_operator_from_description(
            None
        )
        assert result == "Unknown"

    def test_extract_operator_from_description_empty(self, callback):
        """Test extraction returns 'Unknown' for empty description."""
        result = callback._guardrail_instrumentor._extract_operator_from_description("")
        assert result == "Unknown"

    def test_extract_operator_from_description_unknown_operator(self, callback):
        """Test extraction returns 'Unknown' for unrecognized operator."""
        result = callback._guardrail_instrumentor._extract_operator_from_description(
            "field.path unknownOp 'value'"
        )
        assert result == "Unknown"

    def test_extract_rule_details_word_rule(self, callback):
        """Test rule details extraction for word rule."""
        from unittest.mock import MagicMock

        mock_rule = MagicMock()
        mock_rule.rule_type = "word"
        mock_rule.field_selector = SpecificFieldsSelector(
            selector_type="specific",
            fields=[FieldReference(path="age", source=FieldSource.INPUT)],
        )
        mock_rule.rule_description = "message.content contains 'forbidden'"

        result = callback._guardrail_instrumentor._extract_rule_details([mock_rule])

        assert len(result) == 1
        assert result[0]["Type"] == "Word"
        assert result[0]["FieldSelectorType"] == "Specific"
        assert result[0]["Operator"] == "Contains"

    def test_extract_rule_details_number_rule(self, callback):
        """Test rule details extraction for number rule."""
        from unittest.mock import MagicMock

        mock_rule = MagicMock()
        mock_rule.rule_type = "number"
        mock_rule.field_selector = AllFieldsSelector(
            selector_type="all", sources=[FieldSource.OUTPUT]
        )
        mock_rule.field_selector_type = "all"
        mock_rule.rule_description = "All fields greaterThan 10.0"

        result = callback._guardrail_instrumentor._extract_rule_details([mock_rule])

        assert len(result) == 1
        assert result[0]["Type"] == "Number"
        assert result[0]["FieldSelectorType"] == "All"
        assert result[0]["Operator"] == "GreaterThan"

    def test_extract_rule_details_boolean_rule(self, callback):
        """Test rule details extraction for boolean rule."""
        from unittest.mock import MagicMock

        mock_rule = MagicMock()
        mock_rule.rule_type = "boolean"
        mock_rule.field_selector = SpecificFieldsSelector(
            selector_type="specific",
            fields=[FieldReference(path="is_active", source=FieldSource.INPUT)],
        )
        mock_rule.rule_description = "data.is_active equals True"

        result = callback._guardrail_instrumentor._extract_rule_details([mock_rule])

        assert len(result) == 1
        assert result[0]["Type"] == "Boolean"
        assert result[0]["FieldSelectorType"] == "Specific"
        assert result[0]["Operator"] == "Equals"

    def test_extract_rule_details_universal_rule(self, callback):
        """Test rule details extraction for universal rule (always enforce)."""
        from uipath.core.guardrails import ApplyTo, UniversalRule

        mock_rule = UniversalRule(
            rule_type="always",
            apply_to=ApplyTo.INPUT_AND_OUTPUT,
        )

        result = callback._guardrail_instrumentor._extract_rule_details([mock_rule])

        assert len(result) == 1
        assert result[0]["Type"] == "Always enforce the guardrail"
        assert result[0]["ApplyTo"] == "InputAndOutput"

    def test_extract_rule_details_multiple_rules(self, callback):
        """Test rule details extraction for multiple rules."""
        from unittest.mock import MagicMock

        word_rule = MagicMock()
        word_rule.rule_type = "word"
        word_rule.field_selector_type = "specific"
        word_rule.rule_description = "field contains 'test'"

        number_rule = MagicMock()
        number_rule.rule_type = "number"
        number_rule.field_selector_type = "all"
        number_rule.rule_description = "All fields lessThan 100"

        result = callback._guardrail_instrumentor._extract_rule_details(
            [word_rule, number_rule]
        )

        assert len(result) == 2
        assert result[0]["Type"] == "Word"
        assert result[0]["Operator"] == "Contains"
        assert result[1]["Type"] == "Number"
        assert result[1]["Operator"] == "LessThan"


class TestDeterministicGuardrailTelemetry:
    """Tests for DeterministicGuardrail telemetry properties."""

    def test_deterministic_guardrail_includes_rule_details(
        self, callback, tracer, span_exporter
    ):
        """DeterministicGuardrail metadata should include NumberOfRules property."""
        import json

        from uipath.core.guardrails import DeterministicGuardrail

        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)

            # Create mock DeterministicGuardrail with rules
            mock_guardrail = MagicMock(spec=DeterministicGuardrail)
            mock_guardrail.name = "custom_guard"
            mock_guardrail.description = "Custom Guard"
            mock_guardrail.enabled_for_evals = True
            mock_guardrail.guardrail_type = "custom"
            mock_guardrail.selector = MagicMock()
            mock_guardrail.selector.scopes = [GuardrailScope.TOOL]

            # Create mock rules
            mock_rule1 = MagicMock()
            mock_rule1.rule_type = "word"
            mock_rule1.field_selector = SpecificFieldsSelector(
                selector_type="specific",
                fields=[FieldReference(path="status", source=FieldSource.INPUT)],
            )
            mock_rule1.rule_description = "field contains 'test'"

            mock_rule2 = MagicMock()
            mock_rule2.rule_type = "number"
            mock_rule2.field_selector = AllFieldsSelector(
                selector_type="all", sources=[FieldSource.OUTPUT]
            )
            mock_rule2.rule_description = "All fields greaterThan 5"

            mock_guardrail.rules = [mock_rule1, mock_rule2]

            with patch(
                "uipath_agents._observability.llmops.instrumentors.guardrail_instrumentor.track_event"
            ) as mock_track:
                run_id = uuid4()
                callback.on_chain_start(
                    {},
                    {},
                    run_id=run_id,
                    metadata={
                        "langgraph_node": "tool_pre_execution_custom_guard",
                        "guardrail": mock_guardrail,
                        "node_type": "guardrail_evaluation",
                        "scope": GuardrailScope.TOOL,
                        "execution_stage": ExecutionStage.PRE_EXECUTION,
                        "action_type": "Skip",
                    },
                )
                callback.on_chain_end(
                    {INNER_STATE_KEY: {GUARDRAIL_VALIDATION_RESULT_KEY: True}},
                    run_id=run_id,
                )

                props = mock_track.call_args[0][1]
                assert props["NumberOfRules"] == "2"
                assert props["GuardrailType"] == "Custom"

                assert "RuleDetails" in props

                rule_details = json.loads(props["RuleDetails"])
                assert len(rule_details) == 2
                assert rule_details[0]["Type"] == "Word"
                assert rule_details[0]["FieldSelectorType"] == "Specific"
                assert rule_details[0]["Operator"] == "Contains"
                assert rule_details[1]["Type"] == "Number"
                assert rule_details[1]["FieldSelectorType"] == "All"
                assert rule_details[1]["Operator"] == "GreaterThan"

    def test_deterministic_guardrail_with_universal_rule(
        self, callback, tracer, span_exporter
    ):
        """DeterministicGuardrail with UniversalRule should have correct RuleDetails."""
        import json

        from uipath.core.guardrails import (
            ApplyTo,
            DeterministicGuardrail,
            UniversalRule,
        )

        agent_run_id = uuid4()
        with tracer.start_agent_run("TestAgent") as agent_span:
            callback.set_agent_span(agent_span, agent_run_id)

            mock_guardrail = MagicMock(spec=DeterministicGuardrail)
            mock_guardrail.name = "always_guard"
            mock_guardrail.description = "Always Guard"
            mock_guardrail.enabled_for_evals = True
            mock_guardrail.guardrail_type = "custom"
            mock_guardrail.selector = MagicMock()
            mock_guardrail.selector.scopes = [GuardrailScope.TOOL]

            universal_rule = UniversalRule(
                rule_type="always",
                apply_to=ApplyTo.INPUT_AND_OUTPUT,
            )
            mock_guardrail.rules = [universal_rule]

            with patch(
                "uipath_agents._observability.llmops.instrumentors.guardrail_instrumentor.track_event"
            ) as mock_track:
                run_id = uuid4()
                callback.on_chain_start(
                    {},
                    {},
                    run_id=run_id,
                    metadata={
                        "langgraph_node": "tool_pre_execution_always_guard",
                        "guardrail": mock_guardrail,
                        "node_type": "guardrail_evaluation",
                        "scope": GuardrailScope.TOOL,
                        "execution_stage": ExecutionStage.PRE_EXECUTION,
                        "action_type": "Skip",
                    },
                )
                callback.on_chain_end(
                    {INNER_STATE_KEY: {GUARDRAIL_VALIDATION_RESULT_KEY: True}},
                    run_id=run_id,
                )

                props = mock_track.call_args[0][1]
                rule_details = json.loads(props["RuleDetails"])

                assert len(rule_details) == 1
                assert rule_details[0]["Type"] == "Always enforce the guardrail"
                assert rule_details[0]["ApplyTo"] == "InputAndOutput"
