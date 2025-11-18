import logging
import os
from typing import Any, AsyncGenerator, Optional, Sequence
from uuid import uuid4

from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import EmptyInputError, GraphRecursionError, InvalidUpdateError
from langgraph.graph.state import CompiledStateGraph, StateGraph
from langgraph.types import Interrupt, StateSnapshot
from typing_extensions import override
from uipath._cli._runtime._contracts import (
    UiPathBaseRuntime,
    UiPathBreakpointResult,
    UiPathErrorCategory,
    UiPathErrorCode,
    UiPathResumeTrigger,
    UiPathRuntimeContext,
    UiPathRuntimeResult,
    UiPathRuntimeStatus,
)
from uipath._cli.models.runtime_schema import Entrypoint
from uipath._events._events import (
    UiPathAgentMessageEvent,
    UiPathAgentStateEvent,
    UiPathRuntimeEvent,
)

from .._utils._schema import generate_schema_from_graph
from ._exception import LangGraphErrorCode, LangGraphRuntimeError
from ._graph_resolver import AsyncResolver, LangGraphJsonResolver
from ._input import get_graph_input
from ._output import create_and_save_resume_trigger, serialize_output

logger = logging.getLogger(__name__)


class LangGraphRuntime(UiPathBaseRuntime):
    """
    A runtime class implementing the async context manager protocol.
    This allows using the class with 'async with' statements.
    """

    def __init__(
        self,
        context: UiPathRuntimeContext,
        graph_resolver: AsyncResolver,
        memory: AsyncSqliteSaver,
    ):
        super().__init__(context)
        self.context: UiPathRuntimeContext = context
        self.graph_resolver: AsyncResolver = graph_resolver
        self.memory: AsyncSqliteSaver = memory
        self.resume_triggers_table: str = "__uipath_resume_triggers"

    async def execute(self) -> Optional[UiPathRuntimeResult]:
        """Execute the graph with the provided input and configuration."""
        graph = await self.graph_resolver()
        if not graph:
            return None

        try:
            compiled_graph = await self._setup_graph(self.memory, graph)
            graph_input = await self._get_graph_input(self.memory)
            graph_config = self._get_graph_config()

            # Execute without streaming
            graph_output = await compiled_graph.ainvoke(
                graph_input,
                graph_config,
                interrupt_before=self.context.breakpoints,
            )

            # Get final state and create result
            self.context.result = await self._create_runtime_result(
                compiled_graph, graph_config, self.memory, graph_output
            )

            return self.context.result

        except Exception as e:
            raise self._create_runtime_error(e) from e

    async def stream(
        self,
    ) -> AsyncGenerator[UiPathRuntimeEvent | UiPathRuntimeResult, None]:
        """
        Stream graph execution events in real-time.

        Yields UiPath UiPathRuntimeEvent instances (thin wrappers around framework data),
        then yields the final UiPathRuntimeResult as the last item.
        The result is also stored in self.context.result.

        Yields:
            - UiPathAgentMessageEvent: Wraps framework messages (BaseMessage, chunks, etc.)
            - UiPathAgentStateEvent: Wraps framework state updates
            - Final event: UiPathRuntimeResult or UiPathBreakpointResult

        Example:
            async for event in runtime.stream():
                if isinstance(event, UiPathRuntimeResult):
                    # Last event is the result
                    print(f"Final result: {event}")
                elif isinstance(event, UiPathAgentMessageEvent):
                    # Access framework-specific message
                    message = event.payload  # BaseMessage or AIMessageChunk
                    print(f"Message: {message.content}")
                elif isinstance(event, UiPathAgentStateEvent):
                    # Access framework-specific state
                    state = event.payload
                    print(f"Node {event.node_name} updated: {state}")

        Raises:
            LangGraphRuntimeError: If execution fails
        """
        graph = await self.graph_resolver()
        if not graph:
            return

        try:
            compiled_graph = await self._setup_graph(self.memory, graph)
            graph_input = await self._get_graph_input(self.memory)
            graph_config = self._get_graph_config()

            # Track final chunk for result creation
            final_chunk: Optional[dict[Any, Any]] = None

            # Stream events from graph
            async for stream_chunk in compiled_graph.astream(
                graph_input,
                graph_config,
                interrupt_before=self.context.breakpoints,
                stream_mode=["messages", "updates"],
                subgraphs=True,
            ):
                _, chunk_type, data = stream_chunk

                # Emit UiPathAgentMessageEvent for messages
                if chunk_type == "messages":
                    if isinstance(data, tuple):
                        message, _ = data
                        event = UiPathAgentMessageEvent(
                            payload=message,
                            execution_id=self.context.execution_id,
                        )
                        yield event

                # Emit UiPathAgentStateEvent for state updates
                elif chunk_type == "updates":
                    if isinstance(data, dict):
                        final_chunk = data

                        # Emit state update event for each node
                        for node_name, agent_data in data.items():
                            if isinstance(agent_data, dict):
                                state_event = UiPathAgentStateEvent(
                                    payload=serialize_output(agent_data),
                                    node_name=node_name,
                                    execution_id=self.context.execution_id,
                                )
                                yield state_event

            # Extract output from final chunk
            graph_output = self._extract_graph_result(
                final_chunk, compiled_graph.output_channels
            )

            # Get final state and create result
            self.context.result = await self._create_runtime_result(
                compiled_graph, graph_config, self.memory, graph_output
            )

            # Yield the final result as last event
            yield self.context.result

        except Exception as e:
            raise self._create_runtime_error(e) from e

    async def _setup_graph(
        self, memory: AsyncSqliteSaver, graph: StateGraph[Any, Any, Any]
    ) -> CompiledStateGraph[Any, Any, Any]:
        """Setup and compile the graph with memory and interrupts."""
        interrupt_before: list[str] = []
        interrupt_after: list[str] = []

        return graph.compile(
            checkpointer=memory,
            interrupt_before=interrupt_before,
            interrupt_after=interrupt_after,
        )

    def _get_graph_config(self) -> RunnableConfig:
        """Build graph execution configuration."""
        graph_config: RunnableConfig = {
            "configurable": {
                "thread_id": (
                    self.context.execution_id or self.context.job_id or "default"
                )
            },
            "callbacks": [],
        }

        # Add optional config from environment
        recursion_limit = os.environ.get("LANGCHAIN_RECURSION_LIMIT")
        max_concurrency = os.environ.get("LANGCHAIN_MAX_CONCURRENCY")

        if recursion_limit is not None:
            graph_config["recursion_limit"] = int(recursion_limit)
        if max_concurrency is not None:
            graph_config["max_concurrency"] = int(max_concurrency)

        return graph_config

    async def _get_graph_input(self, memory: AsyncSqliteSaver) -> Any:
        """Process and return graph input."""
        return await get_graph_input(
            context=self.context,
            memory=memory,
            resume_triggers_table=self.resume_triggers_table,
        )

    async def _get_graph_state(
        self,
        compiled_graph: CompiledStateGraph[Any, Any, Any],
        graph_config: RunnableConfig,
    ) -> Optional[StateSnapshot]:
        """Get final graph state."""
        try:
            return await compiled_graph.aget_state(graph_config)
        except Exception:
            return None

    def _extract_graph_result(
        self, final_chunk: Any, output_channels: str | Sequence[str]
    ) -> Any:
        """
        Extract the result from a LangGraph output chunk according to the graph's output channels.

        Args:
            final_chunk: The final chunk from graph.astream()
            output_channels: The graph's output channel configuration

        Returns:
            The extracted result according to the graph's output_channels configuration
        """
        # Unwrap from subgraph tuple format if needed
        if isinstance(final_chunk, tuple) and len(final_chunk) == 2:
            final_chunk = final_chunk[1]

        # If the result isn't a dict or graph doesn't define output channels, return as is
        if not isinstance(final_chunk, dict):
            return final_chunk

        # Case 1: Single output channel as string
        if isinstance(output_channels, str):
            return final_chunk.get(output_channels, final_chunk)

        # Case 2: Multiple output channels as sequence
        elif hasattr(output_channels, "__iter__") and not isinstance(
            output_channels, str
        ):
            # Check which channels are present
            available_channels = [ch for ch in output_channels if ch in final_chunk]

            # If no available channels, output may contain the last_node name as key
            unwrapped_final_chunk = {}
            if not available_channels and len(final_chunk) == 1:
                potential_unwrap = next(iter(final_chunk.values()))
                if isinstance(potential_unwrap, dict):
                    unwrapped_final_chunk = potential_unwrap
                    available_channels = [
                        ch for ch in output_channels if ch in unwrapped_final_chunk
                    ]

            if available_channels:
                # Create a dict with the available channels
                return {
                    channel: final_chunk.get(channel)
                    or unwrapped_final_chunk.get(channel)
                    for channel in available_channels
                }

        # Fallback for any other case
        return final_chunk

    def _is_interrupted(self, state: StateSnapshot) -> bool:
        """Check if execution was interrupted (static or dynamic)."""
        # Check for static interrupts (interrupt_before/after)
        if hasattr(state, "next") and state.next:
            return True

        # Check for dynamic interrupts (interrupt() inside node)
        if hasattr(state, "tasks"):
            for task in state.tasks:
                if hasattr(task, "interrupts") and task.interrupts:
                    return True

        return False

    def _get_dynamic_interrupt(self, state: StateSnapshot) -> Optional[Interrupt]:
        """Get the first dynamic interrupt if any."""
        if not hasattr(state, "tasks"):
            return None

        for task in state.tasks:
            if hasattr(task, "interrupts") and task.interrupts:
                for interrupt in task.interrupts:
                    if isinstance(interrupt, Interrupt):
                        return interrupt
        return None

    async def _create_runtime_result(
        self,
        compiled_graph: CompiledStateGraph[Any, Any, Any],
        graph_config: RunnableConfig,
        memory: AsyncSqliteSaver,
        graph_output: Optional[Any],
    ) -> UiPathRuntimeResult:
        """
        Get final graph state and create the execution result.

        Stores the result in self.context.result and self.context.state.

        Args:
            compiled_graph: The compiled graph instance
            graph_config: The graph execution configuration
            memory: The SQLite memory instance
            graph_output: The graph execution output
        """
        # Get the final state
        graph_state = await self._get_graph_state(compiled_graph, graph_config)

        # Check if execution was interrupted (static or dynamic)
        if graph_state and self._is_interrupted(graph_state):
            return await self._create_suspended_result(
                graph_state, memory, graph_output
            )
        else:
            # Normal completion
            return self._create_success_result(graph_output)

    async def _create_suspended_result(
        self,
        graph_state: StateSnapshot,
        graph_memory: AsyncSqliteSaver,
        graph_output: Optional[Any],
    ) -> UiPathRuntimeResult:
        """Create result for suspended execution."""
        # Check if it's a dynamic interrupt
        dynamic_interrupt = self._get_dynamic_interrupt(graph_state)
        resume_trigger: Optional[UiPathResumeTrigger] = None

        if dynamic_interrupt:
            # Dynamic interrupt - create and save resume trigger
            resume_trigger = await create_and_save_resume_trigger(
                interrupt_value=dynamic_interrupt.value,
                memory=graph_memory,
                resume_triggers_table=self.resume_triggers_table,
            )
            output = serialize_output(graph_output)
            return UiPathRuntimeResult(
                output=output,
                status=UiPathRuntimeStatus.SUSPENDED,
                resume=resume_trigger,
            )
        else:
            # Static interrupt (breakpoint)
            return self._create_breakpoint_result(graph_state)

    def _create_breakpoint_result(
        self,
        graph_state: StateSnapshot,
    ) -> UiPathBreakpointResult:
        """Create result for execution paused at a breakpoint."""

        # Get next nodes - these are the nodes that will execute when resumed
        next_nodes = list(graph_state.next)

        # Determine breakpoint type and node
        if next_nodes:
            # Breakpoint is BEFORE these nodes (interrupt_before)
            breakpoint_type = "before"
            breakpoint_node = next_nodes[0]
        else:
            # Breakpoint is AFTER the last executed node (interrupt_after)
            # Get the last executed node from tasks
            breakpoint_type = "after"
            if graph_state.tasks:
                # Tasks contain the nodes that just executed
                # Get the last task's name
                breakpoint_node = graph_state.tasks[-1].name
            else:
                # Fallback if no tasks (shouldn't happen)
                breakpoint_node = "unknown"

        return UiPathBreakpointResult(
            breakpoint_node=breakpoint_node,
            breakpoint_type=breakpoint_type,
            current_state=serialize_output(graph_state.values),
            next_nodes=next_nodes,
        )

    def _create_success_result(self, output: Optional[Any]) -> UiPathRuntimeResult:
        """Create result for successful completion."""
        return UiPathRuntimeResult(
            output=serialize_output(output),
            status=UiPathRuntimeStatus.SUCCESSFUL,
        )

    def _create_runtime_error(self, e: Exception) -> LangGraphRuntimeError:
        """Handle execution errors and create appropriate LangGraphRuntimeError."""
        if isinstance(e, LangGraphRuntimeError):
            return e

        detail = f"Error: {str(e)}"

        if isinstance(e, GraphRecursionError):
            return LangGraphRuntimeError(
                LangGraphErrorCode.GRAPH_LOAD_ERROR,
                "Graph recursion limit exceeded",
                detail,
                UiPathErrorCategory.USER,
            )

        if isinstance(e, InvalidUpdateError):
            return LangGraphRuntimeError(
                LangGraphErrorCode.GRAPH_INVALID_UPDATE,
                str(e),
                detail,
                UiPathErrorCategory.USER,
            )

        if isinstance(e, EmptyInputError):
            return LangGraphRuntimeError(
                LangGraphErrorCode.GRAPH_EMPTY_INPUT,
                "The input data is empty",
                detail,
                UiPathErrorCategory.USER,
            )

        return LangGraphRuntimeError(
            UiPathErrorCode.EXECUTION_ERROR,
            "Graph execution failed",
            detail,
            UiPathErrorCategory.USER,
        )

    async def validate(self) -> None:
        """Validate runtime inputs."""
        pass

    async def cleanup(self) -> None:
        """Cleanup runtime resources."""
        pass


class LangGraphScriptRuntime(LangGraphRuntime):
    """
    Resolves the graph from langgraph.json config file and passes it to the base runtime.
    """

    def __init__(
        self,
        context: UiPathRuntimeContext,
        memory: AsyncSqliteSaver,
        entrypoint: Optional[str] = None,
    ):
        self.resolver = LangGraphJsonResolver(entrypoint=entrypoint)
        super().__init__(context, self.resolver, memory=memory)

    @override
    async def get_entrypoint(self) -> Entrypoint:
        """Get entrypoint for this LangGraph runtime."""
        graph = await self.resolver()
        compiled_graph = graph.compile()
        schema_details = generate_schema_from_graph(compiled_graph)

        return Entrypoint(
            file_path=self.context.entrypoint,  # type: ignore[call-arg]
            unique_id=str(uuid4()),
            type="agent",
            input=schema_details.schema["input"],
            output=schema_details.schema["output"],
        )

    async def cleanup(self) -> None:
        """Cleanup runtime resources including resolver."""
        await super().cleanup()
        await self.resolver.cleanup()
