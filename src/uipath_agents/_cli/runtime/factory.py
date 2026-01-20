import os
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.state import CompiledStateGraph, StateGraph
from pydantic import BaseModel, Field
from uipath._cli._utils._folders import get_personal_workspace_key_async
from uipath.agent.models.agent import AgentDefinition
from uipath.agent.react.conversational_prompts import PromptUserSettings
from uipath.core import UiPathTraceManager
from uipath.core.chat import UiPathConversationMessage
from uipath.platform.common import UiPathConfig
from uipath.platform.resume_triggers import UiPathResumeTriggerHandler
from uipath.runtime import (
    UiPathResumableRuntime,
    UiPathRuntimeContext,
    UiPathRuntimeProtocol,
)
from uipath.runtime.errors import UiPathErrorCategory
from uipath.tracing import LlmOpsHttpExporter
from uipath_langchain.runtime.errors import LangGraphErrorCode, LangGraphRuntimeError
from uipath_langchain.runtime.factory import UiPathLangGraphRuntimeFactory
from uipath_langchain.runtime.storage import SqliteResumableStorage

from uipath_agents.agent_graph_builder import build_agent_graph
from uipath_agents.agent_graph_builder.config import get_execution_type

from ..._observability import configure_telemetry, shutdown_telemetry
from ..._observability.callback import UiPathTracingCallback
from ..._observability.runtime_wrapper import TelemetryRuntimeWrapper
from ..._observability.span_processor import SourceMarkerProcessor
from ..._observability.sqlite_trace_context_storage import SqliteTraceContextStorage
from ..._observability.telemetry_callback import AppInsightsTelemetryCallback
from ..._observability.tracer import UiPathTracer
from ..constants import AGENT_ENTRYPOINT
from ..utils import _prepare_agent_run_files, load_agent_configuration
from .runtime import AgentsLangGraphRuntime

load_dotenv()


class AgentsRuntimeFactory(UiPathLangGraphRuntimeFactory):
    """Factory for creating Agent runtimes from agent.json configuration."""

    def __init__(self, context: UiPathRuntimeContext) -> None:
        """Initialize the factory.

        Args:
            context: UiPathRuntimeContext to use for runtime creation
        """
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
        agent_definition = self._load_agent_definition(
            entrypoint, kwargs.get("settings")
        )

        # Get shared memory instance
        memory = await self._get_memory()

        # Pass definition to graph loading
        compiled_graph = await self._resolve_and_compile_graph(
            entrypoint, memory, agent_definition=agent_definition, **kwargs
        )

        return await self._create_runtime(
            compiled_graph=compiled_graph,
            runtime_id=runtime_id,
            entrypoint=entrypoint,
            memory=memory,
            agent_definition=agent_definition,
        )

    async def dispose(self) -> None:
        """Cleanup factory resources."""
        if self.context.trace_manager:
            self.context.trace_manager.flush_spans()
        shutdown_telemetry()
        await super().dispose()

    def _setup_instrumentation(self, trace_manager: UiPathTraceManager | None) -> None:
        """Setup tracing and instrumentation."""
        super()._setup_instrumentation(trace_manager)
        if trace_manager:
            trace_manager.tracer_provider.add_span_processor(SourceMarkerProcessor())

        configure_telemetry(trace_manager)

    def _load_agent_definition(
        self, entrypoint: str, settings: dict[str, Any] | None = None
    ) -> AgentDefinition:
        """Load and prepare the agent definition.

        Args:
            entrypoint: Agent file path (agent.json)
            settings: Optional settings to apply to the agent definition
        Returns:
            Prepared AgentDefinition

        Raises:
            LangGraphRuntimeError: If definition cannot be loaded
        """
        try:
            _prepare_agent_run_files()

            agent_json_path = Path.cwd() / entrypoint
            agent_definition = load_agent_configuration(agent_json_path)

            if agent_definition.is_conversational:
                agent_definition.input_schema = (
                    self._get_conversational_agent_input_schema()
                )

            if settings:
                # Use case is evaluation runs where we want to
                # override the agent's default model settings
                model_name = str(settings.get("model_name"))
                if model_name != "same-as-agent":
                    agent_definition.settings.model = model_name
                temperature = settings.get("temperature")
                if temperature not in ["same-as-agent", None]:
                    if isinstance(temperature, (int, float)):
                        agent_definition.settings.temperature = float(temperature)
                    elif isinstance(temperature, str):
                        try:
                            agent_definition.settings.temperature = float(temperature)
                        except ValueError:
                            pass  # Ignore invalid temperature strings

            return agent_definition

        except FileNotFoundError as e:
            raise LangGraphRuntimeError(
                LangGraphErrorCode.GRAPH_NOT_FOUND,
                "Agent configuration not found",
                f"Agent file '{entrypoint}' not found: {str(e)}",
                UiPathErrorCategory.DEPLOYMENT,
            ) from e
        except Exception as e:
            raise LangGraphRuntimeError(
                LangGraphErrorCode.GRAPH_LOAD_ERROR,
                "Failed to load agent configuration",
                f"Unexpected error loading agent '{entrypoint}': {str(e)}",
                UiPathErrorCategory.USER,
            ) from e

    async def _load_graph(
        self, entrypoint: str, **kwargs: Any
    ) -> StateGraph[Any, Any, Any] | CompiledStateGraph[Any, Any, Any, Any]:
        """Load agent graph for the given entrypoint.

        Args:
            entrypoint: Agent file path (agent.json)

        Returns:
            Compiled StateGraph for the agent

        Raises:
            LangGraphRuntimeError: If graph cannot be loaded
        """
        agent_definition = cast(AgentDefinition, kwargs.get("agent_definition"))
        try:
            return await build_agent_graph(
                agent_definition, execution_type=get_execution_type(self.context)
            )
        except Exception as e:
            raise LangGraphRuntimeError(
                LangGraphErrorCode.GRAPH_LOAD_ERROR,
                "Failed to build agent graph",
                f"Unexpected error building agent '{entrypoint}': {str(e)}",
                UiPathErrorCategory.USER,
            ) from e

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
            Configured runtime stack: Base → Telemetry → Resumable

        Note:
            Runtime wrapping order:
            1. AgentsLangGraphRuntime (base - handles LangGraph execution)
            2. TelemetryRuntimeWrapper (adds tracing spans + trace context preservation)
            3. UiPathResumableRuntime (adds persistence/resume)

            The callback is passed via constructor to the base runtime,
            ensuring it persists across debug/chat re-executions where
            the same runtime instance is executed multiple times.

            Trace context storage is shared between TelemetryRuntimeWrapper
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

        llmops_exporter = LlmOpsHttpExporter()
        tracer = UiPathTracer(exporter=llmops_exporter)
        tracing_callback = UiPathTracingCallback(tracer)
        telemetry_callback = AppInsightsTelemetryCallback()

        base_runtime = AgentsLangGraphRuntime(
            graph=compiled_graph,
            runtime_id=runtime_id,
            entrypoint=entrypoint,
            callbacks=[tracing_callback, telemetry_callback],
            agent_definition=agent_definition,
        )
        telemetry_runtime = TelemetryRuntimeWrapper(
            base_runtime,
            tracer,
            tracing_callback,
            self.context,
            telemetry_callback=telemetry_callback,
            agent_definition=agent_definition,
            trace_context_storage=trace_context_storage,
        )
        return await self._wrap_in_resumable_runtime(
            telemetry_runtime, storage, runtime_id
        )

    async def _wrap_in_resumable_runtime(
        self,
        base_runtime: UiPathRuntimeProtocol,
        storage: SqliteResumableStorage,
        runtime_id: str,
    ) -> UiPathResumableRuntime:
        """Wrap a base runtime in UiPathResumableRuntime with storage and trigger manager.

        Args:
            base_runtime: The base runtime to wrap
            storage: Pre-created storage (shared with TelemetryRuntimeWrapper)
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

    def _get_conversational_agent_input_schema(self) -> dict[str, Any]:
        """Gets conversational agent input schema."""
        # Currently conversational agents don't support user defined input schemas, but we have
        # https://uipath.atlassian.net/browse/JAR-9067 to enable. However, for the python runtime, we are also using the
        # input property to provide the input message and userSettings and that usage will conflict with user defined
        # inputs when implementing that feature. There are at least three solutions:
        #
        # 1) make agent builder emit the system defined schema with a nested field containing the user defined schema.
        # Then when CAS starts the job it can include the inputs passed to the start conversation API along with the
        # existing message and user settings inputs. In this case, the schema set here would be put in the agent
        # definition by agent builder and this "fixup" step can be removed.
        #
        # 2) add a new "system_input" property that can be passed when starting/resuming the agent job and restore the
        # input property to being user defined content only. Agent builder would then emit a schema for that property
        # (just like for other agents, rather than special casing as done for option 1). In this case, we don't need the
        # schema set below. The system_input property type could be statically defined, or maybe we want to make it a
        # property bag that is used without a schema.
        #
        # 3) Rename these to __uipath_messages and __uipath_userSettings, and reserve the __uipath_ prefix for system
        # properties.
        #
        # Note that there is an additional "fixup" that is done in the message factory: agent builder is including a
        # vestigial user message in the agent definition ("What is the current date?"). That should be removed at some
        # point as well.

        class ConversationalAgentInput(BaseModel):
            """Input schema for conversational agents."""

            # Defining a BaseModel here to ensure $defs are properly included in the schema
            messages: list[UiPathConversationMessage] = []
            user_settings: PromptUserSettings | None = Field(
                default=None, alias="userSettings"
            )

            model_config = {"extra": "allow"}

        return ConversationalAgentInput.model_json_schema()
