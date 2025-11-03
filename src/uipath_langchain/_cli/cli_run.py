import asyncio
import logging
import os
import json
from typing import Optional

from openinference.instrumentation.langchain import (
    LangChainInstrumentor,
    get_current_span,
)
from uipath._cli._debug._bridge import ConsoleDebugBridge, UiPathDebugBridge
from uipath._cli._conversational._bridge import get_conversation_bridge
from uipath._cli._conversational._runtime import UiPathConversationRuntime
from uipath._cli._runtime._contracts import (
    UiPathRuntimeFactory,
    UiPathRuntimeResult,
)
from uipath._cli.middlewares import MiddlewareResult
from uipath._events._events import UiPathAgentStateEvent, UiPathAgentMessageEvent
from uipath.tracing import JsonLinesFileExporter, LlmOpsHttpExporter
from uipath.agent.conversation import UiPathConversationMessage
from pydantic import TypeAdapter

from .._tracing import (
    _instrument_traceable_attributes,
)
from ._runtime._exception import LangGraphRuntimeError
from ._runtime._runtime import (  # type: ignore[attr-defined]
    LangGraphRuntimeContext,
    LangGraphScriptRuntime,
)
from ._utils._config import UiPathConfig
from ._utils._graph import LangGraphConfig


logger = logging.getLogger(__name__)

def langgraph_run_middleware(
    entrypoint: Optional[str],
    input: Optional[str],
    resume: bool,
    trace_file: Optional[str] = None,
    **kwargs,
) -> MiddlewareResult:
    """Middleware to handle LangGraph execution"""
    config = LangGraphConfig()
    if not config.exists:
        return MiddlewareResult(
            should_continue=True
        )  # Continue with normal flow if no langgraph.json

    try:

        async def execute():
            context = LangGraphRuntimeContext.with_defaults(**kwargs)
            context.entrypoint = entrypoint
            context.input = input
            context.resume = resume
            context.execution_id = context.job_id or "default"
            _instrument_traceable_attributes()

            # Check if this is a conversational agent
            uipath_config = UiPathConfig()
            is_conversational = False
            if uipath_config.exists:
                is_conversational = uipath_config.is_conversational
                context.is_conversational = is_conversational

                if is_conversational and context.input:
                    try:
                        input_dict = json.loads(context.input)

                        conversation_id = input_dict.get("conversation_id") or input_dict.get("conversationId")
                        exchange_id = input_dict.get("exchange_id") or input_dict.get("exchangeId")

                        # Store IDs in context for reuse in output
                        if conversation_id:
                            context.conversation_id = conversation_id
                        if exchange_id:
                            context.exchange_id = exchange_id

                        context.input_message = TypeAdapter(UiPathConversationMessage).validate_python(input_dict)
                        logger.info(f"Parsed conversational input: message_id={context.input_message.message_id}, conversation_id={conversation_id}, exchange_id={exchange_id}")
                    except Exception as e:
                        logger.warning(f"Failed to parse input as UiPathConversationMessage: {e}. Using as plain JSON.")

            def generate_runtime(
                ctx: LangGraphRuntimeContext,
            ) -> LangGraphScriptRuntime:
                runtime = LangGraphScriptRuntime(ctx, ctx.entrypoint)
                # If not resuming and no job id, delete the previous state file
                if not ctx.resume and ctx.job_id is None:
                    if os.path.exists(runtime.state_file_path):
                        os.remove(runtime.state_file_path)
                return runtime

            runtime_factory = UiPathRuntimeFactory(
                LangGraphScriptRuntime,
                LangGraphRuntimeContext,
                runtime_generator=generate_runtime,
                context_generator=lambda: context,
            )

            runtime_factory.add_instrumentor(LangChainInstrumentor, get_current_span)

            if trace_file:
                runtime_factory.add_span_exporter(JsonLinesFileExporter(trace_file))

            if context.job_id:
                runtime_factory.add_span_exporter(
                    LlmOpsHttpExporter(extra_process_spans=True)
                )

            # Handle conversational agents
            if is_conversational:
                conversation_bridge = get_conversation_bridge(context)
                async with UiPathConversationRuntime.from_conversation_context(
                    context=context,
                    factory=runtime_factory,
                    conversation_bridge=conversation_bridge,
                ) as conversation_runtime:
                    await conversation_runtime.execute()
            # Handle non-conversational agents
            elif context.job_id:
                # Cloud execution - direct runtime execution
                await runtime_factory.execute(context)
            else:
                # Local execution - stream with debug bridge for visibility
                debug_bridge: UiPathDebugBridge = ConsoleDebugBridge()
                await debug_bridge.emit_execution_started(context.execution_id)
                async for event in runtime_factory.stream(context):
                    if isinstance(event, UiPathRuntimeResult):
                        await debug_bridge.emit_execution_completed(event)
                    elif isinstance(event, UiPathAgentStateEvent):
                        await debug_bridge.emit_state_update(event)

        asyncio.run(execute())

        return MiddlewareResult(
            should_continue=False,
            error_message=None,
        )

    except LangGraphRuntimeError as e:
        return MiddlewareResult(
            should_continue=False,
            error_message=e.error_info.detail,
            should_include_stacktrace=True,
        )
    except Exception as e:
        return MiddlewareResult(
            should_continue=False,
            error_message=f"Error: {str(e)}",
            should_include_stacktrace=True,
        )
