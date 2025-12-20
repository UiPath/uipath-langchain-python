import logging
import os
from typing import Any, AsyncGenerator
from uuid import uuid4

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables.config import RunnableConfig
from langgraph.errors import EmptyInputError, GraphRecursionError, InvalidUpdateError
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, Interrupt, StateSnapshot
from uipath.runtime import (
    UiPathBreakpointResult,
    UiPathExecuteOptions,
    UiPathRuntimeResult,
    UiPathRuntimeStatus,
    UiPathStreamOptions,
)
from uipath.runtime.errors import UiPathErrorCategory, UiPathErrorCode
from uipath.runtime.events import (
    UiPathRuntimeEvent,
    UiPathRuntimeMessageEvent,
    UiPathRuntimeStateEvent,
)
from uipath.runtime.schema import UiPathRuntimeSchema

from uipath_langchain.chat import UiPathChatMessagesMapper
from uipath_langchain.runtime.errors import LangGraphErrorCode, LangGraphRuntimeError
from uipath_langchain.runtime.schema import get_entrypoints_schema, get_graph_schema

from ._serialize import serialize_output

logger = logging.getLogger(__name__)


class UiPathLangGraphRuntime:
    """
    A runtime class for executing LangGraph graphs within the UiPath framework.
    """

    def __init__(
        self,
        graph: CompiledStateGraph[Any, Any, Any, Any],
        runtime_id: str | None = None,
        entrypoint: str | None = None,
        callbacks: list[BaseCallbackHandler] | None = None,
    ):
        """
        Initialize the runtime.

        Args:
            graph: The CompiledStateGraph to execute
            runtime_id: Unique identifier for this runtime instance
            entrypoint: Optional entrypoint name (for schema generation)
        """
        self.graph: CompiledStateGraph[Any, Any, Any, Any] = graph
        self.runtime_id: str = runtime_id or "default"
        self.entrypoint: str | None = entrypoint
        self.callbacks: list[BaseCallbackHandler] = callbacks or []
        self.chat = UiPathChatMessagesMapper()
        self._middleware_node_names: set[str] = self._detect_middleware_nodes()

    async def execute(
        self,
        input: dict[str, Any] | None = None,
        options: UiPathExecuteOptions | None = None,
    ) -> UiPathRuntimeResult:
        """Execute the graph with the provided input and configuration."""
        try:
            graph_input = await self._get_graph_input(input, options)
            graph_config = self._get_graph_config()

            # Execute without streaming
            graph_output = await self.graph.ainvoke(
                graph_input,
                graph_config,
                interrupt_before=options.breakpoints if options else None,
            )

            # Get final state and create result
            result = await self._create_runtime_result(graph_config, graph_output)

            return result

        except Exception as e:
            raise self._create_runtime_error(e) from e

    async def stream(
        self,
        input: dict[str, Any] | None = None,
        options: UiPathStreamOptions | None = None,
    ) -> AsyncGenerator[UiPathRuntimeEvent, None]:
        """
        Stream graph execution events in real-time.

        Yields UiPath UiPathRuntimeEvent instances (thin wrappers around framework data),
        then yields the final UiPathRuntimeResult as the last item.

        Yields:
            - UiPathRuntimeMessageEvent: Wraps framework messages (BaseMessage, chunks, etc.)
            - UiPathRuntimeStateEvent: Wraps framework state updates
            - Final event: UiPathRuntimeResult or UiPathBreakpointResult

        Example:
            async for event in runtime.stream():
                if isinstance(event, UiPathRuntimeResult):
                    # Last event is the result
                    print(f"Final result: {event}")
                elif isinstance(event, UiPathRuntimeMessageEvent):
                    # Access framework-specific message
                    message = event.payload  # BaseMessage or AIMessageChunk
                    print(f"Message: {message.content}")
                elif isinstance(event, UiPathRuntimeStateEvent):
                    # Access framework-specific state
                    state = event.payload
                    print(f"Node {event.node_name} updated: {state}")

        Raises:
            LangGraphRuntimeError: If execution fails
        """
        try:
            graph_input = await self._get_graph_input(input, options)
            graph_config = self._get_graph_config()

            # Track final chunk for result creation
            final_chunk: dict[Any, Any] | None = None

            # Stream events from graph
            async for stream_chunk in self.graph.astream(
                graph_input,
                graph_config,
                interrupt_before=options.breakpoints if options else None,
                stream_mode=["messages", "updates"],
                subgraphs=True,
            ):
                _, chunk_type, data = stream_chunk

                # Emit UiPathRuntimeMessageEvent for messages
                if chunk_type == "messages":
                    if isinstance(data, tuple):
                        message, _ = data
                        try:
                            events = self.chat.map_event(message)
                        except Exception as e:
                            logger.warning(f"Error mapping message event: {e}")
                            events = None
                        if events:
                            for mapped_event in events:
                                event = UiPathRuntimeMessageEvent(
                                    payload=mapped_event,
                                )
                                yield event

                # Emit UiPathRuntimeStateEvent for state updates
                elif chunk_type == "updates":
                    if isinstance(data, dict):
                        filtered_data = {
                            node_name: agent_data
                            for node_name, agent_data in data.items()
                            if not self._is_middleware_node(node_name)
                        }
                        if filtered_data:
                            final_chunk = filtered_data

                        # Emit state update event for each node
                        for node_name, agent_data in data.items():
                            if isinstance(agent_data, dict):
                                state_event = UiPathRuntimeStateEvent(
                                    payload=serialize_output(agent_data),
                                    node_name=node_name,
                                )
                                yield state_event

            # Extract output from final chunk
            graph_output = self._extract_graph_result(final_chunk)

            # Get final state and create result
            result = await self._create_runtime_result(graph_config, graph_output)

            # Yield the final result as last event
            yield result

        except Exception as e:
            raise self._create_runtime_error(e) from e

    async def get_schema(self) -> UiPathRuntimeSchema:
        """Get schema for this LangGraph runtime."""
        schema_details = get_entrypoints_schema(self.graph)

        return UiPathRuntimeSchema(
            filePath=self.entrypoint,
            uniqueId=str(uuid4()),
            type="agent",
            input=schema_details.schema["input"],
            output=schema_details.schema["output"],
            graph=get_graph_schema(self.graph, xray=1),
        )

    def _get_graph_config(self) -> RunnableConfig:
        """Build graph execution configuration."""
        graph_config: RunnableConfig = {
            "configurable": {"thread_id": self.runtime_id},
            "callbacks": self.callbacks,
        }

        # Add optional config from environment
        recursion_limit = os.environ.get("LANGCHAIN_RECURSION_LIMIT")
        max_concurrency = os.environ.get("LANGCHAIN_MAX_CONCURRENCY")

        if recursion_limit is not None:
            graph_config["recursion_limit"] = int(recursion_limit)
        if max_concurrency is not None:
            graph_config["max_concurrency"] = int(max_concurrency)

        return graph_config

    async def _get_graph_input(
        self,
        input: dict[str, Any] | None,
        options: UiPathExecuteOptions | None,
    ) -> Any:
        """Process and return graph input."""
        graph_input = input or {}
        if isinstance(graph_input, dict):
            messages = graph_input.get("messages", None)
            if messages and isinstance(messages, list):
                graph_input["messages"] = self.chat.map_messages(messages)
        if options and options.resume:
            return Command(resume=graph_input)
        return graph_input

    async def _get_graph_state(
        self,
        graph_config: RunnableConfig,
    ) -> StateSnapshot | None:
        """Get final graph state."""
        try:
            return await self.graph.aget_state(graph_config)
        except Exception:
            return None

    def _extract_graph_result(self, final_chunk: Any) -> Any:
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

        output_channels = self.graph.output_channels

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
        return bool(state.next)

    async def _create_runtime_result(
        self,
        graph_config: RunnableConfig,
        graph_output: Any,
    ) -> UiPathRuntimeResult:
        """
        Get final graph state and create the execution result.

        Args:
            graph_config: The graph execution configuration
            graph_output: The graph execution output
        """
        # Get the final state
        graph_state = await self._get_graph_state(graph_config)

        # Check if execution was interrupted (static or dynamic)
        if graph_state and self._is_interrupted(graph_state):
            return await self._create_suspended_result(graph_state)
        else:
            # Normal completion
            return self._create_success_result(graph_output)

    async def _create_suspended_result(
        self,
        graph_state: StateSnapshot,
    ) -> UiPathRuntimeResult:
        """Create result for suspended execution."""
        interrupt_map: dict[str, Any] = {}

        # Get nodes that are still scheduled to run
        next_nodes = set(graph_state.next) if graph_state.next else set()

        if graph_state.interrupts:
            for interrupt in graph_state.interrupts:
                if isinstance(interrupt, Interrupt):
                    # Find which task this interrupt belongs to
                    for task in graph_state.tasks:
                        if task.interrupts and interrupt in task.interrupts:
                            # Only include if this task's node is still in next
                            if task.name in next_nodes:
                                interrupt_map[interrupt.id] = interrupt.value
                            break

        # If we have dynamic interrupts, return suspended with interrupt map
        # The output is used to create the resume triggers
        if interrupt_map:
            return UiPathRuntimeResult(
                output=interrupt_map,
                status=UiPathRuntimeStatus.SUSPENDED,
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

    def _create_success_result(self, output: Any) -> UiPathRuntimeResult:
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

    def _detect_middleware_nodes(self) -> set[str]:
        """
        Detect middleware nodes by their naming pattern.

        Middleware nodes always contain both:
        1. "Middleware" in the name (by convention)
        2. A dot "." separator (MiddlewareName.hook_name)

        Returns:
            Set of middleware node names
        """
        middleware_nodes: set[str] = set()

        for node_name in self.graph.nodes.keys():
            if "." in node_name and "Middleware" in node_name:
                middleware_nodes.add(node_name)

        return middleware_nodes

    def _is_middleware_node(self, node_name: str) -> bool:
        """Check if a node name represents a middleware node."""
        return node_name in self._middleware_node_names

    async def dispose(self) -> None:
        """Cleanup runtime resources."""
        pass
