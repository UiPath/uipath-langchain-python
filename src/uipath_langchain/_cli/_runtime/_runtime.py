import logging
import os
from typing import Any, Optional, Sequence

from langchain_core.messages import BaseMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import EmptyInputError, GraphRecursionError, InvalidUpdateError
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Interrupt, StateSnapshot
from uipath._cli._runtime._contracts import (
    UiPathBaseRuntime,
    UiPathErrorCategory,
    UiPathResumeTrigger,
    UiPathRuntimeResult,
    UiPathRuntimeStatus,
)

from ._context import LangGraphRuntimeContext
from ._conversation import map_message
from ._exception import LangGraphRuntimeError
from ._graph_resolver import AsyncResolver, LangGraphJsonResolver
from ._input import get_graph_input
from ._output import create_and_save_resume_trigger, serialize_output

logger = logging.getLogger(__name__)


class LangGraphRuntime(UiPathBaseRuntime):
    """
    A runtime class implementing the async context manager protocol.
    This allows using the class with 'async with' statements.
    """

    def __init__(self, context: LangGraphRuntimeContext, graph_resolver: AsyncResolver):
        super().__init__(context)
        self.context: LangGraphRuntimeContext = context
        self.graph_resolver: AsyncResolver = graph_resolver
        self.resume_triggers_table: str = "__uipath_resume_triggers"

    async def execute(self) -> Optional[UiPathRuntimeResult]:
        """
        Execute the graph with the provided input and configuration.

        Returns:
            Dictionary with execution results

        Raises:
            LangGraphRuntimeError: If execution fails
        """
        graph = await self.graph_resolver()
        if not graph:
            return None

        try:
            async with AsyncSqliteSaver.from_conn_string(
                self.state_file_path
            ) as memory:
                self.context.memory = memory

                # Compile the graph with the checkpointer and any debug interrupts
                interrupt_before: list[str] = []
                interrupt_after: list[str] = []

                compiled_graph = graph.compile(
                    checkpointer=self.context.memory,
                    interrupt_before=interrupt_before,
                    interrupt_after=interrupt_after,
                )

                # Process input, handling resume if needed
                graph_input = await get_graph_input(
                    context=self.context,
                    memory=self.context.memory,
                    resume_triggers_table=self.resume_triggers_table,
                )

                # Build graph config
                graph_config: RunnableConfig = self._get_graph_config()
                graph_output: Optional[Any] = None

                # Execute the graph
                if self.context.chat_handler or self.is_debug_run():
                    # Stream mode for debugging or chat
                    graph_output = await self._execute_streaming(
                        compiled_graph, graph_input, graph_config
                    )
                else:
                    # Normal mode
                    graph_output = await compiled_graph.ainvoke(
                        graph_input, graph_config
                    )

                # Get the final state
                graph_state: Optional[StateSnapshot] = None
                try:
                    graph_state = await compiled_graph.aget_state(graph_config)
                except Exception:
                    pass

                # Check if execution was interrupted (static or dynamic)
                if graph_state and self._is_interrupted(graph_state):
                    self.context.result = await self._create_suspended_result(
                        graph_state, self.context.memory, graph_output
                    )
                else:
                    # Normal completion
                    self.context.result = self._create_success_result(graph_output)

                return self.context.result

        except Exception as e:
            raise self._create_runtime_error(e) from e

    async def _execute_streaming(
        self,
        compiled_graph: CompiledStateGraph[Any, Any, Any],
        graph_input: Any,
        graph_config: RunnableConfig,
    ) -> Any:
        """Execute graph in streaming mode for chat/debug."""
        final_chunk: Optional[dict[Any, Any]] = None

        async for stream_chunk in compiled_graph.astream(
            graph_input,
            graph_config,
            stream_mode=["messages", "updates"],
            subgraphs=True,
        ):
            _, chunk_type, data = stream_chunk

            if chunk_type == "messages" and self.context.chat_handler:
                if isinstance(data, tuple):
                    message, _ = data
                    event = map_message(
                        message=message,
                        conversation_id=self.context.execution_id,
                        exchange_id=self.context.execution_id,
                    )
                    if event:
                        self.context.chat_handler.on_event(event)

            elif chunk_type == "updates":
                if isinstance(data, dict):
                    # Print messages if in debug mode
                    for agent_data in data.values():
                        if isinstance(agent_data, dict):
                            messages = agent_data.get("messages", [])
                            if isinstance(messages, list):
                                for message in messages:
                                    if isinstance(message, BaseMessage):
                                        message.pretty_print()
                    final_chunk = data

        return self._extract_graph_result(final_chunk, compiled_graph.output_channels)

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

    async def _create_suspended_result(
        self,
        graph_state: StateSnapshot,
        graph_memory: AsyncSqliteSaver,
        graph_output: Optional[Any],
    ) -> UiPathRuntimeResult:
        """Create result for suspended execution."""
        # Check if it's a dynamic interrupt
        dynamic_interrupt: Optional[Interrupt] = self._get_dynamic_interrupt(
            graph_state
        )
        resume_trigger: Optional[UiPathResumeTrigger] = None

        if dynamic_interrupt:
            # Dynamic interrupt - create and save resume trigger
            resume_trigger = await create_and_save_resume_trigger(
                interrupt_value=dynamic_interrupt.value,
                memory=graph_memory,
                resume_triggers_table=self.resume_triggers_table,
            )
            output = serialize_output(graph_output)
        else:
            # Static interrupt (breakpoint)
            # Output represents the current graph state values
            output = serialize_output(graph_state.values)

        return UiPathRuntimeResult(
            output=output,
            status=UiPathRuntimeStatus.SUSPENDED,
            resume=resume_trigger,
        )

    def _create_success_result(self, output: Optional[Any]) -> UiPathRuntimeResult:
        """Create result for successful completion."""
        return UiPathRuntimeResult(
            output=serialize_output(output),
            status=UiPathRuntimeStatus.SUCCESSFUL,
        )

    def _create_runtime_error(self, e: Exception) -> Exception:
        """Handle execution errors and raise appropriate LangGraphRuntimeError."""
        if isinstance(e, LangGraphRuntimeError):
            return e

        detail = f"Error: {str(e)}"

        if isinstance(e, GraphRecursionError):
            return LangGraphRuntimeError(
                "GRAPH_RECURSION_ERROR",
                "Graph recursion limit exceeded",
                detail,
                UiPathErrorCategory.USER,
            )

        if isinstance(e, InvalidUpdateError):
            return LangGraphRuntimeError(
                "GRAPH_INVALID_UPDATE",
                str(e),
                detail,
                UiPathErrorCategory.USER,
            )

        if isinstance(e, EmptyInputError):
            return LangGraphRuntimeError(
                "GRAPH_EMPTY_INPUT",
                "The input data is empty",
                detail,
                UiPathErrorCategory.USER,
            )

        return LangGraphRuntimeError(
            "EXECUTION_ERROR",
            "Graph execution failed",
            detail,
            UiPathErrorCategory.USER,
        )

    def _get_graph_config(self) -> RunnableConfig:
        graph_config: RunnableConfig = {
            "configurable": {
                "thread_id": (
                    self.context.execution_id or self.context.job_id or "default"
                )
            },
            "callbacks": [],
        }

        # Add optional config
        recursion_limit = os.environ.get("LANGCHAIN_RECURSION_LIMIT", None)
        max_concurrency = os.environ.get("LANGCHAIN_MAX_CONCURRENCY", None)

        if recursion_limit is not None:
            graph_config["recursion_limit"] = int(recursion_limit)
        if max_concurrency is not None:
            graph_config["max_concurrency"] = int(max_concurrency)

        return graph_config

    def _extract_graph_result(self, final_chunk, output_channels: str | Sequence[str]):
        """
        Extract the result from a LangGraph output chunk according to the graph's output channels.

        Args:
            final_chunk: The final chunk from graph.astream()
            graph: The LangGraph instance

        Returns:
            The extracted result according to the graph's output_channels configuration
        """
        # Unwrap from subgraph tuple format if needed
        if isinstance(final_chunk, tuple) and len(final_chunk) == 2:
            final_chunk = final_chunk[
                1
            ]  # Extract data part from (namespace, data) tuple

        # If the result isn't a dict or graph doesn't define output channels, return as is
        if not isinstance(final_chunk, dict):
            return final_chunk

        # Case 1: Single output channel as string
        if isinstance(output_channels, str):
            if output_channels in final_chunk:
                return final_chunk[output_channels]
            else:
                return final_chunk

        # Case 2: Multiple output channels as sequence
        elif hasattr(output_channels, "__iter__") and not isinstance(
            output_channels, str
        ):
            # Check which channels are present
            available_channels = [ch for ch in output_channels if ch in final_chunk]

            # if no available channels, output may contain the last_node name as key
            unwrapped_final_chunk = {}
            if not available_channels:
                if len(final_chunk) == 1 and isinstance(
                    unwrapped_final_chunk := next(iter(final_chunk.values())), dict
                ):
                    available_channels = [
                        ch for ch in output_channels if ch in unwrapped_final_chunk
                    ]

            if available_channels:
                # Create a dict with the available channels
                return {
                    channel: final_chunk.get(channel, None)
                    or unwrapped_final_chunk[channel]
                    for channel in available_channels
                }

        # Fallback for any other case
        return final_chunk

    async def validate(self) -> None:
        pass

    async def cleanup(self):
        pass


class LangGraphScriptRuntime(LangGraphRuntime):
    """
    Resolves the graph from langgraph.json config file and passes it to the base runtime.
    """

    def __init__(
        self, context: LangGraphRuntimeContext, entrypoint: Optional[str] = None
    ):
        self.resolver = LangGraphJsonResolver(entrypoint=entrypoint)
        super().__init__(context, self.resolver)

    async def cleanup(self):
        await super().cleanup()
        await self.resolver.cleanup()
