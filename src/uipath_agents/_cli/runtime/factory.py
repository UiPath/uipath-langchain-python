import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langgraph.graph.state import CompiledStateGraph, StateGraph
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
from uipath_agents.agent_graph_builder.config import AgentExecutionType

from ..._observability import configure_telemetry, shutdown_telemetry
from ..._observability.callback import UiPathTracingCallback
from ..._observability.runtime_wrapper import TelemetryRuntimeWrapper
from ..._observability.span_attributes import AgentSpanInfo
from ..._observability.span_processor import SourceMarkerProcessor
from ..._observability.sqlite_trace_context_storage import SqliteTraceContextStorage
from ..._observability.telemetry_callback import AppInsightsTelemetryCallback
from ..._observability.tracer import UiPathTracer
from ..constants import AGENT_ENTRYPOINT
from ..utils import _prepare_agent_run_files, load_agent_configuration
from .runtime import AgentsLangGraphRuntime
from .utils import validate_json_against_json_schema

load_dotenv()


class AgentsRuntimeFactory(UiPathLangGraphRuntimeFactory):
    """Factory for creating Agent runtimes from agent.json configuration."""

    def __init__(self, context: UiPathRuntimeContext) -> None:
        """Initialize the factory.

        Args:
            context: UiPathRuntimeContext to use for runtime creation
        """
        super().__init__(context)
        self._agent_info: AgentSpanInfo | None = None

    def _setup_instrumentation(self, trace_manager: UiPathTraceManager | None) -> None:
        """Setup tracing and instrumentation."""
        super()._setup_instrumentation(trace_manager)
        if trace_manager:
            trace_manager.tracer_provider.add_span_processor(SourceMarkerProcessor())

        configure_telemetry(trace_manager)

    def discover_entrypoints(self) -> list[str]:
        """Discover the Agent entrypoint agent.json"""
        return [AGENT_ENTRYPOINT]

    async def _load_graph(
        self, entrypoint: str, **kwargs: Any
    ) -> StateGraph[Any, Any, Any] | CompiledStateGraph[Any, Any, Any, Any]:
        """Load agent graph for the given entrypoint and validate input against schema.

        Args:
            entrypoint: Agent file path (agent.json)

        Returns:
            Compiled StateGraph for the agent

        Raises:
            LangGraphRuntimeError: If graph cannot be loaded
        """
        try:
            _prepare_agent_run_files()

            agent_json_path = Path.cwd() / entrypoint
            agent_definition = load_agent_configuration(agent_json_path)

            if agent_definition.is_conversational:
                self._fixup_conversational_agent_definition(agent_definition)

            model = None
            max_tokens = None
            temperature = None
            engine = None
            max_iterations = None
            if hasattr(agent_definition, "settings") and agent_definition.settings:
                settings = agent_definition.settings
                model = (
                    str(settings.model)
                    if hasattr(settings, "model") and settings.model
                    else None
                )
                max_tokens = (
                    settings.max_tokens
                    if hasattr(settings, "max_tokens") and settings.max_tokens
                    else None
                )
                temperature = (
                    settings.temperature
                    if hasattr(settings, "temperature") and settings.temperature
                    else None
                )
                engine = (
                    str(settings.engine)
                    if hasattr(settings, "engine") and settings.engine
                    else None
                )
                max_iterations = (
                    settings.max_iterations
                    if hasattr(settings, "max_iterations") and settings.max_iterations
                    else None
                )

            # Store agent info for telemetry (avoids re-loading config later)
            self._agent_info = AgentSpanInfo(
                name=agent_definition.name or "unknown",
                input_schema=agent_definition.input_schema,
                output_schema=agent_definition.output_schema,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                engine=engine,
                max_iterations=max_iterations,
                is_conversational=agent_definition.is_conversational,
            )

            if not self.context.resume:
                # Validate input against schema
                validate_json_against_json_schema(
                    agent_definition.input_schema, self.context.get_input()
                )

            return await build_agent_graph(
                agent_definition, execution_type=self._get_execution_type()
            )

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
                "Failed to load agent graph",
                f"Unexpected error loading agent '{entrypoint}': {str(e)}",
                UiPathErrorCategory.USER,
            ) from e

    def _fixup_conversational_agent_definition(
        self, agent_definition: AgentDefinition
    ) -> None:
        """Fix up a conversational agent definition."""
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
        agent_definition.input_schema = conversational_agent_input_schema

    async def _create_runtime_instance(
        self,
        compiled_graph: CompiledStateGraph[Any, Any, Any, Any],
        runtime_id: str,
        entrypoint: str,
        **kwargs: Any,
    ) -> UiPathRuntimeProtocol:
        """Create an agent runtime instance from a compiled graph.

        Args:
            compiled_graph: The compiled graph
            runtime_id: Unique identifier for the runtime instance
            entrypoint: Agent entrypoint name

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
        memory = await self._get_memory()
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
        )
        telemetry_runtime = TelemetryRuntimeWrapper(
            base_runtime,
            tracer,
            tracing_callback,
            telemetry_callback=telemetry_callback,
            agent_info=self._agent_info,
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

    def _get_execution_type(self) -> AgentExecutionType:
        match self.context.command:
            case "run":
                return AgentExecutionType.RUNTIME
            case "debug":
                return AgentExecutionType.PLAYGROUND
            case "eval":
                return AgentExecutionType.EVAL
            case _:
                return AgentExecutionType.RUNTIME

    async def dispose(self) -> None:
        """Cleanup factory resources."""
        if self.context.trace_manager:
            self.context.trace_manager.flush_spans()
        shutdown_telemetry()
        await super().dispose()


conversational_agent_input_schema: dict[str, Any] = {
    "type": "object",
    "additionalProperties": True,
    "properties": {
        "messages": {
            "type": "array",
            "items": UiPathConversationMessage.model_json_schema(),
        },
        "userSettings": PromptUserSettings.model_json_schema(),
    },
}
