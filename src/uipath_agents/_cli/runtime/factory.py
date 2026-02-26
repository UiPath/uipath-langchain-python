import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.state import CompiledStateGraph, StateGraph
from pydantic import BaseModel, ConfigDict, Field
from uipath._cli._utils._folders import get_personal_workspace_key_async
from uipath.agent.models.agent import AgentDefinition
from uipath.agent.react.conversational_prompts import PromptUserSettings
from uipath.core import UiPathSpanUtils, UiPathTraceManager
from uipath.core.chat import (
    UiPathConversationMessage,
    UiPathConversationMessageData,
)
from uipath.core.tracing import UiPathTraceSettings
from uipath.platform.common import UiPathConfig
from uipath.platform.resume_triggers import UiPathResumeTriggerHandler
from uipath.runtime import (
    UiPathResumableRuntime,
    UiPathRuntimeContext,
    UiPathRuntimeFactorySettings,
    UiPathRuntimeProtocol,
)
from uipath.runtime.base import UiPathDisposableProtocol
from uipath.runtime.errors import UiPathErrorCategory
from uipath.tracing import LlmOpsHttpExporter
from uipath_langchain.agent.exceptions import (
    AgentStartupError,
    AgentStartupErrorCode,
)
from uipath_langchain.runtime.factory import UiPathLangGraphRuntimeFactory
from uipath_langchain.runtime.storage import SqliteResumableStorage

from uipath_agents._errors import ExceptionMapper
from uipath_agents.agent_graph_builder import build_agent_graph
from uipath_agents.agent_graph_builder.config import get_execution_type

from ..._bts.bts_callback import BtsCallback
from ..._bts.bts_runtime import BtsRuntime
from ..._bts.bts_state import BtsState
from ..._bts.bts_storage import SqliteBtsStateStorage
from ..._observability import configure_telemetry, shutdown_telemetry
from ..._observability.event_emitter import TelemetryEventEmitter
from ..._observability.exporters import FilteringSpanExporter
from ..._observability.instrumented_runtime import InstrumentedRuntime
from ..._observability.llmops import (
    LlmOpsInstrumentationCallback,
    LlmOpsSpanFactory,
    SqliteTraceContextStorage,
    is_custom_instrumentation_span,
)
from ..._observability.llmops.callback import _get_ancestor_spans, _get_current_span
from ..._observability.utils import configure_appinsights_cloud_role, setup_otel_env
from ..constants import AGENT_ENTRYPOINT
from ..utils import _prepare_agent_execution_contract, load_agent_configuration
from .reporter import ReporterRuntime
from .runtime import AgentsLangGraphRuntime

load_dotenv()

# Setup OTEL environment variables at module load time
# This must happen before any TracerProvider is created
setup_otel_env()

# Configure cloud role for Application Insights custom events
configure_appinsights_cloud_role()

logger = logging.getLogger(__name__)


class AgentsRuntimeFactory(UiPathLangGraphRuntimeFactory):
    """Factory for creating Agent runtimes from agent.json configuration."""

    def __init__(self, context: UiPathRuntimeContext) -> None:
        """Initialize the factory.

        Args:
            context: UiPathRuntimeContext to use for runtime creation
        """
        logger.info(f"Initializing AgentsRuntimeFactory for command {context.command}")
        _prepare_agent_execution_contract()
        self._disposables: list[UiPathDisposableProtocol] = []
        self._agent_definition: AgentDefinition | None = None

        super().__init__(context)

    def discover_entrypoints(self) -> list[str]:
        """Discover the Agent entrypoint agent.json"""
        return [AGENT_ENTRYPOINT]

    async def new_runtime(
        self, entrypoint: str, runtime_id: str, **kwargs: Any
    ) -> UiPathRuntimeProtocol:
        """Create a new Agent runtime instance.

        Args:
            entrypoint: Agent entrypoint (agent.json)
            runtime_id: Unique identifier for the runtime instance

        Returns:
            Configured runtime instance with compiled graph
        """
        try:
            # Extract settings override from kwargs to pass through method chain
            settings = kwargs.get("settings")

            self._agent_definition = self._load_agent_definition(entrypoint, settings)

            # Get shared memory instance
            memory = await self._get_memory()

            # Pass definition to graph loading
            compiled_graph = await self._resolve_and_compile_graph(
                entrypoint, memory, agent_definition=self._agent_definition, **kwargs
            )

            return await self._create_runtime(
                compiled_graph=compiled_graph,
                runtime_id=runtime_id,
                entrypoint=entrypoint,
                memory=memory,
                agent_definition=self._agent_definition,
            )
        except Exception as e:
            reporter = ReporterRuntime(
                ExceptionMapper.map_config(e),
                agent_definition=self._agent_definition,
            )
            span_factory, callback, event_emitter = self._create_telemetry_components()
            return InstrumentedRuntime(
                reporter,
                span_factory,
                callback,
                self.context,
                event_emitter=event_emitter,
                agent_definition=self._agent_definition,
            )

    async def get_settings(self) -> UiPathRuntimeFactorySettings | None:
        """Return factory settings with low-code specific trace filtering."""
        return UiPathRuntimeFactorySettings(
            trace_settings=UiPathTraceSettings(
                span_filter=lambda span: bool(
                    span.attributes
                    and span.attributes.get("uipath.custom_instrumentation")
                )
            )
        )

    async def dispose(self) -> None:
        """Cleanup factory resources."""
        if self.context.trace_manager:
            self.context.trace_manager.flush_spans()
        shutdown_telemetry()

        for disposable in self._disposables:
            await disposable.dispose()
        self._disposables.clear()

        await super().dispose()

    def _setup_instrumentation(self, trace_manager: UiPathTraceManager | None) -> None:
        """Setup tracing and instrumentation."""
        super()._setup_instrumentation(trace_manager)
        configure_telemetry(trace_manager)
        UiPathSpanUtils.register_current_span_provider(_get_current_span)
        UiPathSpanUtils.register_current_span_ancestors_provider(_get_ancestor_spans)

    def _load_agent_definition(
        self, entrypoint: str, settings: dict[str, Any] | None = None
    ) -> AgentDefinition:
        """Load and prepare the agent definition.

        Args:
            entrypoint: Agent file path (agent.json)
            settings: Optional settings override to apply to agent definition
        Returns:
            Prepared AgentDefinition

        Raises:
            AgentStartupError: If definition cannot be loaded
        """
        try:
            agent_json_path = Path.cwd() / entrypoint
            agent_definition = load_agent_configuration(agent_json_path)

            # Apply settings override if provided
            if settings:
                agent_definition = self._apply_settings_override(
                    agent_definition, settings
                )

            # Low-code Conversational Agents (agent.json with isConversational=true) implicitly are given:
            # - 'messages' and 'uipath__user_settings' input fields
            # - 'uipath__agent_response_messages' output field
            if agent_definition.is_conversational:
                agent_definition.input_schema = (
                    self._get_conversational_agent_input_schema(
                        agent_definition.input_schema
                    )
                )
                agent_definition.output_schema = self._get_conversational_agent_output_schema(
                    # Currently, the output_schema from conversational agent.jsons has just a default 'content' field and is ignored.
                    # When user-defined outputs are supported, pass in agent_definition.output_schema here instead.
                    None
                )

            return agent_definition

        except FileNotFoundError as e:
            raise AgentStartupError(
                AgentStartupErrorCode.FILE_NOT_FOUND,
                "Agent configuration not found",
                f"Agent file '{entrypoint}' not found: {str(e)}",
                UiPathErrorCategory.USER,
            ) from e

    def _apply_settings_override(
        self, agent_definition: AgentDefinition, settings: dict[str, Any]
    ) -> Any:
        """Apply settings override to agent definition.

        Args:
            agent_definition: The loaded agent definition from agent.json
            settings: Settings override dict with keys like 'model', 'temperature', etc.

        Returns:
            Agent definition with settings overridden
        """
        # Get current settings and apply overrides
        current_settings = agent_definition.settings
        override = settings

        # Create updated settings dict
        updated_settings = {
            "engine": override.get("engine", current_settings.engine),
            "model": override.get("model", current_settings.model),
            "maxTokens": override.get("max_tokens", current_settings.max_tokens),
            "temperature": override.get("temperature", current_settings.temperature),
        }

        logger.info(
            f"Applying settings override: model='{updated_settings['model']}', "
            f"temperature={updated_settings['temperature']}"
        )

        # Create a copy of agent_definition with updated settings
        agent_dict = agent_definition.model_dump(by_alias=True)
        agent_dict["settings"] = updated_settings

        # Reconstruct the agent definition
        return AgentDefinition.model_validate(agent_dict)

    def _merge_system_model_into_existing_schema(
        self, existing_schema: dict[str, Any] | None, system_model: type[BaseModel]
    ) -> dict[str, Any]:
        """Helper function to merge an internal model's fields into existing schema, if any. Merges in properties, defs, and required fields from the system model."""
        system_schema = system_model.model_json_schema()

        # If no existing schema, return system schema as-is
        if not existing_schema:
            return system_schema

        schema = deepcopy(existing_schema)

        if "properties" not in schema:
            schema["properties"] = {}

        schema["properties"].update(system_schema["properties"])

        if "$defs" in system_schema:
            if "$defs" not in schema:
                schema["$defs"] = {}
            schema["$defs"].update(system_schema["$defs"])

        if "required" in system_schema:
            system_required = set(system_schema["required"])
            current_required = set(schema.get("required", []))
            schema["required"] = list(current_required | system_required)

        return schema

    def _get_conversational_agent_input_schema(
        self, existing_input_schema: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Generate input schema for conversational agents.

        Merges default low-code conversational-agent system fields (messages, uipath__user_settings)
        into user-defined fields from existing_input_schema
        """

        class DefaultLowCodeConversationalInput(BaseModel):
            messages: list[UiPathConversationMessage] = Field(default=[])
            uipath__user_settings: PromptUserSettings | None = Field(default=None)
            model_config = ConfigDict(extra="allow")

        return self._merge_system_model_into_existing_schema(
            existing_input_schema, DefaultLowCodeConversationalInput
        )

    def _get_conversational_agent_output_schema(
        self, existing_output_schema: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Generate output schema for conversational agents.

        Merges default low-code conversational-agent system fields (uipath__agent_response_messages)
        into user-defined fields from existing_output_schema
        """

        class DefaultLowCodeConversationalOutput(BaseModel):
            uipath__agent_response_messages: list[UiPathConversationMessageData] = (
                Field(default=[])
            )

        return self._merge_system_model_into_existing_schema(
            existing_output_schema, DefaultLowCodeConversationalOutput
        )

    async def _load_graph(
        self, entrypoint: str, **kwargs: Any
    ) -> StateGraph[Any, Any, Any] | CompiledStateGraph[Any, Any, Any, Any]:
        """Load agent graph for the given entrypoint.

        Args:
            entrypoint: Agent file path (agent.json)

        Returns:
            Compiled StateGraph for the agent

        Raises:
            AgentStartupError: If graph cannot be loaded
        """
        agent_definition = cast(AgentDefinition, kwargs.get("agent_definition"))
        graph, disposables = await build_agent_graph(
            agent_definition, execution_type=get_execution_type(self.context)
        )
        self._disposables.extend(disposables)
        return graph

    async def _create_runtime(
        self,
        compiled_graph: CompiledStateGraph[Any, Any, Any, Any],
        runtime_id: str,
        entrypoint: str,
        memory: AsyncSqliteSaver,
        agent_definition: AgentDefinition,
    ) -> UiPathRuntimeProtocol:
        """Create an agent runtime instance from a compiled graph.

        Args:
            compiled_graph: The compiled graph
            runtime_id: Unique identifier for the runtime instance
            entrypoint: Agent entrypoint name
            agent_definition: Pre-loaded agent definition

        Returns:
            Configured runtime stack: Base → Resumable → BTS → Telemetry

        Note:
            Runtime wrapping order:
            1. AgentsLangGraphRuntime (base - handles LangGraph execution)
            2. UiPathResumableRuntime (adds persistence/resume)
            3. BtsRuntime (adds BTS transaction/operation tracking)
            4. InstrumentedRuntime (adds tracing spans + trace context preservation)

            The callback is passed via constructor to the base runtime,
            ensuring it persists across debug/chat re-executions where
            the same runtime instance is executed multiple times.

            Trace context storage is shared between InstrumentedRuntime
            (for trace context preservation) and UiPathResumableRuntime
            (for agent state persistence).
        """
        # Create storage first - shared between telemetry and resumable runtime
        storage = SqliteResumableStorage(memory)
        trace_context_storage = SqliteTraceContextStorage(storage)

        # Only fetch folder_key for local runs (no job_key), not production
        if not UiPathConfig.job_key and not UiPathConfig.folder_key:
            try:
                folder_key = await get_personal_workspace_key_async()
                if folder_key:
                    os.environ["UIPATH_FOLDER_KEY"] = folder_key
            except Exception:
                pass  # Folder key fetch failed, LlmOps tracing may fail

        span_factory, instrumentation_callback, event_emitter = (
            self._create_telemetry_components()
        )

        # --- BTS setup ---
        bts_state = BtsState()
        bts_callback = BtsCallback(bts_state)
        bts_storage = SqliteBtsStateStorage(storage)

        from uipath.platform import UiPath

        try:
            sdk = UiPath()
            bts_state.tracker_service = sdk.automation_tracker
        except Exception:
            logger.warning("Failed to initialize AutomationTrackerService for BTS")

        agent_name = agent_definition.name or "Unknown"

        base_runtime = AgentsLangGraphRuntime(
            graph=compiled_graph,
            runtime_id=runtime_id,
            entrypoint=entrypoint,
            callbacks=[instrumentation_callback, bts_callback],
            agent_definition=agent_definition,
            storage=storage,
        )
        resumable_runtime = self._wrap_in_resumable_runtime(
            base_runtime, storage, runtime_id
        )
        bts_runtime = BtsRuntime(
            delegate=resumable_runtime,
            state=bts_state,
            callback=bts_callback,
            agent_name=agent_name,
            runtime_id=runtime_id,
            bts_storage=bts_storage,
            parent_operation_id=self.context.parent_operation_id,
        )
        instrumented_runtime = InstrumentedRuntime(
            bts_runtime,
            span_factory,
            instrumentation_callback,
            self.context,
            event_emitter=event_emitter,
            agent_definition=agent_definition,
            trace_context_storage=trace_context_storage,
        )

        if not self.context.resume:
            from uipath_agents._services import register_licensing_async

            await register_licensing_async(
                agent_definition, job_key=UiPathConfig.job_key
            )

        return instrumented_runtime

    def _create_telemetry_components(
        self,
    ) -> tuple[LlmOpsSpanFactory, LlmOpsInstrumentationCallback, TelemetryEventEmitter]:
        """Create the telemetry component chain for instrumentation."""
        llmops_exporter = LlmOpsHttpExporter()
        filtered_exporter = FilteringSpanExporter(
            llmops_exporter, filter_fn=is_custom_instrumentation_span
        )
        span_factory = LlmOpsSpanFactory(exporter=filtered_exporter)
        callback = LlmOpsInstrumentationCallback(span_factory)
        event_emitter = TelemetryEventEmitter()
        return span_factory, callback, event_emitter

    def _wrap_in_resumable_runtime(
        self,
        base_runtime: UiPathRuntimeProtocol,
        storage: SqliteResumableStorage,
        runtime_id: str,
    ) -> UiPathResumableRuntime:
        """Wrap a base runtime in UiPathResumableRuntime with storage and trigger manager.

        Args:
            base_runtime: The base runtime to wrap
            storage: Pre-created storage (shared with InstrumentedRuntime)
            runtime_id: Unique identifier for the runtime instance

        Returns:
            UiPathResumableRuntime wrapping the base runtime
        """
        trigger_manager = UiPathResumeTriggerHandler()

        return UiPathResumableRuntime(
            delegate=base_runtime,
            storage=storage,
            trigger_manager=trigger_manager,
            runtime_id=runtime_id,
        )
