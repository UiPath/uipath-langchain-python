import logging
import os
from typing import Any, List, Optional, Sequence

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import EmptyInputError, GraphRecursionError, InvalidUpdateError
from uipath._cli._runtime._contracts import (
    UiPathBaseRuntime,
    UiPathErrorCategory,
    UiPathRuntimeResult,
)

from ._context import LangGraphRuntimeContext
from ._conversation import map_message
from ._exception import LangGraphRuntimeError
from ._graph_resolver import AsyncResolver, LangGraphJsonResolver
from ._input import LangGraphInputProcessor
from ._output import LangGraphOutputProcessor

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

                # Compile the graph with the checkpointer
                compiled_graph = graph.compile(checkpointer=self.context.memory)

                # Process input, handling resume if needed
                input_processor = LangGraphInputProcessor(context=self.context)

                processed_input = await input_processor.process()

                callbacks: List[BaseCallbackHandler] = []

                graph_config: RunnableConfig = {
                    "configurable": {
                        "thread_id": (
                            self.context.execution_id
                            or self.context.job_id
                            or "default"
                        )
                    },
                    "callbacks": callbacks,
                }

                recursion_limit = os.environ.get("LANGCHAIN_RECURSION_LIMIT", None)
                max_concurrency = os.environ.get("LANGCHAIN_MAX_CONCURRENCY", None)

                if recursion_limit is not None:
                    graph_config["recursion_limit"] = int(recursion_limit)
                if max_concurrency is not None:
                    graph_config["max_concurrency"] = int(max_concurrency)

                if self.context.chat_handler or self.is_debug_run():
                    final_chunk: Optional[dict[Any, Any]] = None
                    async for stream_chunk in compiled_graph.astream(
                        processed_input,
                        graph_config,
                        stream_mode=["messages", "updates"],
                        subgraphs=True,
                    ):
                        _, chunk_type, data = stream_chunk
                        if chunk_type == "messages":
                            if self.context.chat_handler:
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
                                # data is a dict, e.g. {'agent': {'messages': [...]}}
                                for agent_data in data.values():
                                    if isinstance(agent_data, dict):
                                        messages = agent_data.get("messages", [])
                                        if isinstance(messages, list):
                                            for message in messages:
                                                if isinstance(message, BaseMessage):
                                                    message.pretty_print()
                                final_chunk = data

                    self.context.output = self._extract_graph_result(
                        final_chunk, compiled_graph.output_channels
                    )
                else:
                    # Execute the graph normally at runtime or eval
                    self.context.output = await compiled_graph.ainvoke(
                        processed_input, graph_config
                    )

                # Get the state if available
                try:
                    self.context.state = await compiled_graph.aget_state(graph_config)
                except Exception:
                    pass

                output_processor = await LangGraphOutputProcessor.create(self.context)

                self.context.result = await output_processor.process()

                return self.context.result

        except Exception as e:
            if isinstance(e, LangGraphRuntimeError):
                raise

            detail = f"Error: {str(e)}"

            if isinstance(e, GraphRecursionError):
                raise LangGraphRuntimeError(
                    "GRAPH_RECURSION_ERROR",
                    "Graph recursion limit exceeded",
                    detail,
                    UiPathErrorCategory.USER,
                ) from e

            if isinstance(e, InvalidUpdateError):
                raise LangGraphRuntimeError(
                    "GRAPH_INVALID_UPDATE",
                    str(e),
                    detail,
                    UiPathErrorCategory.USER,
                ) from e

            if isinstance(e, EmptyInputError):
                raise LangGraphRuntimeError(
                    "GRAPH_EMPTY_INPUT",
                    "The input data is empty",
                    detail,
                    UiPathErrorCategory.USER,
                ) from e

            raise LangGraphRuntimeError(
                "EXECUTION_ERROR",
                "Graph execution failed",
                detail,
                UiPathErrorCategory.USER,
            ) from e
        finally:
            pass

    async def validate(self) -> None:
        pass

    async def cleanup(self):
        pass

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
