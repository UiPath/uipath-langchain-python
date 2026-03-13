"""Voice agent job runtime — config generation, tool execution, and CAS communication."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

import httpx
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.state import CompiledStateGraph
from uipath.agent.models.agent import (
    AgentContextResourceConfig,
    AgentContextRetrievalMode,
    AgentEscalationResourceConfig,
    AgentIntegrationToolResourceConfig,
    AgentIxpExtractionResourceConfig,
    AgentIxpVsEscalationResourceConfig,
    AgentMessageRole,
    AgentProcessToolResourceConfig,
    BaseAgentResourceConfig,
)
from uipath.core.errors import UiPathPendingTriggerError
from uipath.core.triggers import UiPathResumeTrigger
from uipath.platform.resume_triggers import UiPathResumeTriggerHandler
from uipath.runtime import (
    UiPathExecuteOptions,
    UiPathRuntimeEvent,
    UiPathRuntimeResult,
    UiPathRuntimeSchema,
    UiPathRuntimeStatus,
    UiPathStreamNotSupportedError,
    UiPathStreamOptions,
)
from uipath.runtime.context import UiPathRuntimeContext
from uipath_langchain.agent.tools.static_args import (
    apply_static_args,
    resolve_static_args,
)
from uipath_langchain.agent.tools.tool_factory import create_tools_from_resources
from uipath_langchain.agent.tools.utils import (
    sanitize_dict_for_serialization,
    sanitize_tool_name,
)
from uipath_langchain.runtime.runtime import UiPathLangGraphRuntime
from uipath_langchain.runtime.storage import SqliteResumableStorage

from uipath_agents._cli.runtime.factory import AgentsRuntimeFactory
from uipath_agents.voice.graph import build_voice_tool_graph, get_voice_system_prompt

if TYPE_CHECKING:
    from typing import Literal

    from uipath.agent.models.agent import LowCodeAgentDefinition
    from uipath.runtime import UiPathRuntimeProtocol

logger = logging.getLogger(__name__)

# tools used by voice agents are often short and need responses fast
_VOICE_POLL_INTERVAL_SECONDS = 1.0
# timeout is already 60s from CAS cloud side
_VOICE_POLL_TIMEOUT_SECONDS = 60.0


async def post_to_cas(
    ctx: UiPathRuntimeContext,
    endpoint: str,
    data: dict[str, Any],
) -> None:
    """POST data to the CAS voice endpoint."""
    conversation_id = ctx.conversation_id
    if not conversation_id:
        raise RuntimeError(
            "conversation_id is required to POST to CAS but was not set on the runtime context"
        )

    cas_ws_host = os.environ.get("CAS_WEBSOCKET_HOST")
    if cas_ws_host:
        url = f"http://{cas_ws_host}/api/v1/voice/{conversation_id}/{endpoint}"
        logger.warning("CAS_WEBSOCKET_HOST is set. Using URL '%s'.", url)
    else:
        base_url = os.environ.get("UIPATH_URL")
        if not base_url:
            raise RuntimeError(
                "UIPATH_URL environment variable required for voice mode"
            )

        parsed = urlparse(base_url)
        if not parsed.netloc:
            raise RuntimeError(f"Invalid UIPATH_URL format: {base_url}")

        url = (
            f"{base_url.rstrip('/')}/autopilotforeveryone_"
            f"/api/v1/voice/{conversation_id}/{endpoint}"
        )

    headers = {
        "Authorization": f"Bearer {os.environ.get('UIPATH_ACCESS_TOKEN', '')}",
        "X-UiPath-Internal-TenantId": ctx.tenant_id
        or os.environ.get("UIPATH_TENANT_ID", ""),
        "X-UiPath-Internal-AccountId": ctx.org_id
        or os.environ.get("UIPATH_ORGANIZATION_ID", ""),
        "X-UiPath-ConversationId": conversation_id,
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=data, headers=headers)
        response.raise_for_status()


class VoiceLangGraphRuntime(UiPathLangGraphRuntime):
    """Passes input through without chat mapping.

    Voice tool calls supply a synthetic AIMessage directly — the base
    class's UiPathChatMessagesMapper would corrupt it.
    """

    async def _get_graph_input(
        self,
        input: dict[str, Any] | None,
        options: UiPathExecuteOptions | None,
    ) -> Any:
        from langgraph.types import Command

        graph_input = input or {}
        if options and options.resume:
            return Command(resume=graph_input)
        return graph_input


async def _build_voice_runtime(
    compiled_graph: CompiledStateGraph[Any],
    runtime_id: str,
    memory: AsyncSqliteSaver,
    ctx: UiPathRuntimeContext,
    agent_name: str,
) -> tuple[UiPathRuntimeProtocol, SqliteResumableStorage]:
    """Construct the runtime wrapping stack for a voice tool call.

    Same layer order as AgentsRuntimeFactory._create_runtime:
    VoiceLangGraphRuntime -> UiPathResumableRuntime -> BtsRuntime -> InstrumentedRuntime
    """
    storage = SqliteResumableStorage(memory)
    base_runtime = VoiceLangGraphRuntime(
        graph=compiled_graph,
        runtime_id=runtime_id,
    )

    return await AgentsRuntimeFactory.wrap_with_agents_stack(
        base_runtime,
        storage,
        runtime_id,
        ctx,
        agent_name,
        licensed=False,
        trigger_manager=VoicePollingTriggerHandler(),
    )


# Excluded: Internal (requires LLM), MCP (needs separate create_mcp_tools path)
# Disabled in agent builder already
_VOICE_SUPPORTED_RESOURCE_TYPES = (
    AgentIntegrationToolResourceConfig,
    AgentContextResourceConfig,
    AgentProcessToolResourceConfig,
    AgentIxpExtractionResourceConfig,
    AgentEscalationResourceConfig,
    AgentIxpVsEscalationResourceConfig,
)

# These context modes use durable_interrupt and need the full LangGraph reasoning loop
_CONTEXT_MODES_REQUIRING_LANGGRAPH = {
    AgentContextRetrievalMode.DEEP_RAG,
    AgentContextRetrievalMode.BATCH_TRANSFORM,
}


class UnsupportedVoiceToolError(Exception):
    """Raised when an agent has tools that are not supported in voice mode."""


def _filter_voice_resources(
    agent_definition: LowCodeAgentDefinition,
) -> list[BaseAgentResourceConfig]:
    supported: list[BaseAgentResourceConfig] = []
    unsupported_names: list[str] = []
    for r in agent_definition.resources:
        if not r.is_enabled:
            continue
        if not isinstance(r, _VOICE_SUPPORTED_RESOURCE_TYPES):
            unsupported_names.append(getattr(r, "name", "unknown"))
            continue
        if (
            isinstance(r, AgentContextResourceConfig)
            and r.settings is not None
            and r.settings.retrieval_mode in _CONTEXT_MODES_REQUIRING_LANGGRAPH
        ):
            unsupported_names.append(r.name)
            continue
        supported.append(r)
    if unsupported_names:
        names = ", ".join(f"'{n}'" for n in unsupported_names)
        raise UnsupportedVoiceToolError(
            f"The following tools are not supported in voice mode: {names}"
        )
    return supported


async def _create_voice_tools(
    agent_definition: LowCodeAgentDefinition,
) -> list[tuple[BaseTool, BaseAgentResourceConfig | None]]:
    resources = _filter_voice_resources(agent_definition)
    voice_agent_def = agent_definition.model_copy(update={"resources": resources})

    # llm=None safe — we pre-filtered out tools that require an LLM
    tools = await create_tools_from_resources(
        voice_agent_def,
        llm=None,  # type: ignore[arg-type]
    )

    resource_by_sanitized_name = {
        sanitize_tool_name(r.name): r for r in resources if getattr(r, "name", None)
    }

    return [(tool, resource_by_sanitized_name.get(tool.name)) for tool in tools]


class VoicePollingTriggerHandler:
    """Trigger handler that polls read_trigger until the child job completes.

    Voice jobs can't suspend and wait for Orchestrator to resume them,
    so we block here instead of raising UiPathPendingTriggerError.
    """

    def __init__(self) -> None:
        self._delegate = UiPathResumeTriggerHandler()

    async def create_trigger(self, suspend_value: Any) -> UiPathResumeTrigger:
        return await self._delegate.create_trigger(suspend_value)

    async def read_trigger(self, trigger: UiPathResumeTrigger) -> Any | None:
        elapsed = 0.0
        while elapsed < _VOICE_POLL_TIMEOUT_SECONDS:
            try:
                return await self._delegate.read_trigger(trigger)
            except UiPathPendingTriggerError:
                await asyncio.sleep(_VOICE_POLL_INTERVAL_SECONDS)
                elapsed += _VOICE_POLL_INTERVAL_SECONDS
        raise TimeoutError(
            f"Voice tool did not complete within {_VOICE_POLL_TIMEOUT_SECONDS}s"
        )


async def execute_voice_tool_call(
    agent_definition: LowCodeAgentDefinition,
    tool_name: str,
    args: dict[str, Any],
    call_id: str,
    ctx: UiPathRuntimeContext,
    *,
    resume: bool = False,
) -> UiPathRuntimeResult:
    """Execute a single tool call through a stub LangGraph graph.

    Quick tools complete immediately. Process tools with @durable_interrupt
    are polled to completion via VoicePollingTriggerHandler.
    """
    runtime_id = ctx.conversation_id or ctx.job_id or "default"
    agent_name = agent_definition.name or "Voice Agent"
    state_path = ctx.resolved_state_file_path

    async with AsyncSqliteSaver.from_conn_string(state_path) as memory:
        if not tool_name:
            return UiPathRuntimeResult(
                status=UiPathRuntimeStatus.FAULTED,
                output={
                    "error": "Could not determine tool name (missing input and no stored metadata)"
                },
            )

        tool_pairs = await _create_voice_tools(agent_definition)

        tool = None
        resource = None
        for t, r in tool_pairs:
            if t.name == tool_name:
                tool, resource = t, r
                break

        if tool is None:
            return UiPathRuntimeResult(
                status=UiPathRuntimeStatus.FAULTED,
                output={"error": f"Unknown tool: {tool_name}"},
            )

        if resource is not None:
            static_args = resolve_static_args(resource, {})
            sanitized = sanitize_dict_for_serialization(static_args)
            args = apply_static_args(sanitized, args)

        try:
            stub_graph = build_voice_tool_graph(tool)
            compiled_graph = stub_graph.compile(checkpointer=memory)
            runtime, storage = await _build_voice_runtime(
                compiled_graph=compiled_graph,
                runtime_id=runtime_id,
                memory=memory,
                ctx=ctx,
                agent_name=agent_name,
            )

            try:
                input_state: dict[str, Any] | None = None
                if not resume:
                    input_state = {
                        "messages": [
                            AIMessage(
                                content="",
                                tool_calls=[
                                    {
                                        "id": call_id,
                                        "name": tool_name,
                                        "args": args,
                                    }
                                ],
                            )
                        ]
                    }

                result = await runtime.execute(
                    input_state, UiPathExecuteOptions(resume=resume)
                )

                if isinstance(result.output, dict):
                    result.output["__voice_call_id"] = call_id

                return result

            finally:
                await runtime.dispose()

        except Exception as exc:
            logger.exception("Voice tool call failed: %s", tool_name)
            return UiPathRuntimeResult(
                status=UiPathRuntimeStatus.FAULTED,
                output={"error": str(exc)},
            )


def extract_tool_result(result: UiPathRuntimeResult) -> tuple[str, bool]:
    """Extract (result_string, is_error) from a runtime result for POSTing to CAS."""
    if result.status == UiPathRuntimeStatus.FAULTED:
        output = result.output
        if isinstance(output, dict):
            return output.get("error", "Unknown error"), True
        return str(output) if output else "Unknown error", True

    output = result.output
    if isinstance(output, dict):
        messages = output.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("type") == "tool":
                content = msg.get("content", "")
                is_error = msg.get("status") == "error"
                return str(content), is_error

    return str(output) if output else "", False


class VoiceJobRuntime:
    """Runtime that handles voice agent jobs (config and tool-call).

    Dispatches by voice_mode:
    - "config": builds voice config and POSTs it to CAS
    - "toolCall": executes a tool call and POSTs the result to CAS
    """

    def __init__(
        self,
        agent_definition: LowCodeAgentDefinition,
        voice_mode: Literal["config", "toolCall"],
        ctx: UiPathRuntimeContext,
    ) -> None:
        self._agent_definition = agent_definition
        self._voice_mode = voice_mode
        self._ctx = ctx

    async def execute(
        self,
        input: dict[str, Any] | None = None,
        options: UiPathExecuteOptions | None = None,
    ) -> UiPathRuntimeResult:
        if self._voice_mode == "config":
            return await self._execute_get_config()
        elif self._voice_mode == "toolCall":
            return await self._execute_tool_call(input or {}, options)
        else:
            return UiPathRuntimeResult(
                status=UiPathRuntimeStatus.FAULTED,
                output={"error": f"Unknown voice mode: {self._voice_mode!r}"},
            )

    # no-op
    async def stream(
        self,
        input: dict[str, Any] | None = None,
        options: UiPathStreamOptions | None = None,
    ) -> AsyncGenerator[UiPathRuntimeEvent, None]:
        raise UiPathStreamNotSupportedError(
            "VoiceJobRuntime does not support streaming. Use execute() instead."
        )
        yield

    # no-op
    async def get_schema(self) -> UiPathRuntimeSchema:
        return UiPathRuntimeSchema(
            filePath="",
            uniqueId="",
            type="voice",
            input={},
            output={},
        )

    # no-op
    async def dispose(self) -> None:
        pass

    async def _execute_get_config(self) -> UiPathRuntimeResult:
        try:
            resources = _filter_voice_resources(self._agent_definition)

            system_msg = next(
                (
                    m
                    for m in self._agent_definition.messages
                    if m.role == AgentMessageRole.SYSTEM
                ),
                None,
            )
            system_prompt = get_voice_system_prompt(
                system_message=system_msg.content if system_msg else "",
                agent_name=self._agent_definition.name,
            )

            config = {
                "systemInstruction": system_prompt,
                "model": self._agent_definition.settings.model,
                "voiceName": getattr(self._agent_definition.settings, "persona", None)
                or "Aoede",
                "tools": [
                    {
                        "name": sanitize_tool_name(r.name),
                        "description": r.description,
                        "input_schema": getattr(r, "input_schema", None),
                    }
                    for r in resources
                ],
            }
            await post_to_cas(self._ctx, "config", config)
            return UiPathRuntimeResult(status=UiPathRuntimeStatus.SUCCESSFUL)
        except Exception as exc:
            logger.exception("Voice config job failed")
            try:
                await post_to_cas(self._ctx, "config", {"error": str(exc)})
            except Exception:
                logger.warning("Failed to POST config error to CAS", exc_info=True)
            return UiPathRuntimeResult(
                status=UiPathRuntimeStatus.FAULTED,
                output={"error": str(exc)},
            )

    async def _execute_tool_call(
        self,
        input: dict[str, Any],
        options: UiPathExecuteOptions | None,
    ) -> UiPathRuntimeResult:
        call_id: str = ""
        try:
            voice_tool_call = input.get("voiceToolCall", {})
            call_id = voice_tool_call.get("callId", "")
            tool_name: str = voice_tool_call.get("toolName", "")
            args: dict[str, Any] = voice_tool_call.get("args", {})

            is_resume = bool(options and options.resume)

            if not is_resume:
                if not tool_name:
                    error_msg = "voiceToolCall.toolName is required"
                    await self._post_tool_error(call_id, error_msg)
                    return UiPathRuntimeResult(
                        status=UiPathRuntimeStatus.FAULTED,
                        output={"error": error_msg},
                    )
                if not call_id:
                    error_msg = "voiceToolCall.callId is required"
                    return UiPathRuntimeResult(
                        status=UiPathRuntimeStatus.FAULTED,
                        output={"error": error_msg},
                    )

            result = await execute_voice_tool_call(
                self._agent_definition,
                tool_name,
                args,
                call_id,
                self._ctx,
                resume=is_resume,
            )

            # On resume, call_id may be empty — retrieve from stored metadata
            if not call_id:
                call_id = (
                    result.output.get("__voice_call_id", "")
                    if isinstance(result.output, dict)
                    else ""
                )

            if not call_id:
                logger.error("Cannot POST tool result: call_id is empty")
                return UiPathRuntimeResult(
                    status=UiPathRuntimeStatus.FAULTED,
                    output={"error": "Missing call_id for tool result POST"},
                )

            tool_result, is_error = extract_tool_result(result)
            await post_to_cas(
                self._ctx,
                f"tool-result/{call_id}",
                {
                    "result": tool_result,
                    "isError": is_error,
                },
            )
            return result

        except Exception as exc:
            logger.exception("Voice tool call job failed")
            await self._post_tool_error(call_id, str(exc))
            return UiPathRuntimeResult(
                status=UiPathRuntimeStatus.FAULTED,
                output={"error": str(exc)},
            )

    async def _post_tool_error(self, call_id: str, error_message: str) -> None:
        if not call_id:
            logger.warning(
                "Cannot POST tool error to CAS: call_id is empty. Original error: %s",
                error_message,
            )
            return
        try:
            await post_to_cas(
                self._ctx,
                f"tool-result/{call_id}",
                {"error": error_message},
            )
        except Exception:
            logger.warning("Failed to POST tool error to CAS", exc_info=True)
