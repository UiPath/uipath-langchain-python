"""Process tool creation for UiPath process execution."""

from typing import Any

from langchain.tools import BaseTool
from langchain_core.messages import ToolCall
from langchain_core.tools import StructuredTool
from uipath.agent.models.agent import AgentProcessToolResourceConfig, AgentToolType
from uipath.eval.mocks import mockable
from uipath.platform import UiPath
from uipath.platform.common import WaitJob

from uipath_langchain.agent.react.job_attachments import get_job_attachments
from uipath_langchain.agent.react.jsonschema_pydantic_converter import create_model
from uipath_langchain.agent.react.types import AgentGraphState
from uipath_langchain.agent.tools.static_args import handle_static_args
from uipath_langchain.agent.tools.structured_tool_with_argument_properties import (
    StructuredToolWithArgumentProperties,
)
from uipath_langchain.agent.tools.tool_node import (
    ToolWrapperReturnType,
)

from .durable_interrupt import durable_interrupt
from .utils import sanitize_tool_name


def create_process_tool(resource: AgentProcessToolResourceConfig) -> StructuredTool:
    """Uses interrupt() to suspend graph execution until process completes (handled by runtime)."""
    # Import here to avoid circular dependency
    from uipath_langchain.agent.wrappers import get_job_attachment_wrapper

    tool_name: str = sanitize_tool_name(resource.name)
    process_name = resource.properties.process_name
    folder_path = resource.properties.folder_path

    input_model: Any = create_model(resource.input_schema, tool_name=tool_name)
    output_model: Any = create_model(resource.output_schema, tool_name=tool_name)

    _span_context: dict[str, Any] = {}
    _bts_context: dict[str, Any] = {}

    async def process_tool_fn(**kwargs: Any):
        attachments = get_job_attachments(input_model, kwargs)
        input_arguments = input_model.model_validate(kwargs).model_dump(mode="json")

        @mockable(
            name=resource.name,
            description=resource.description,
            input_schema=input_model.model_json_schema(),
            output_schema=output_model.model_json_schema(),
            example_calls=resource.properties.example_calls,
        )
        async def invoke_process(**_tool_kwargs: Any):
            parent_span_id = _span_context.pop("parent_span_id", None)
            parent_operation_id = _bts_context.pop("parent_operation_id", None)

            @durable_interrupt
            async def start_job():
                client = UiPath()
                job = await client.processes.invoke_async(
                    name=process_name,
                    input_arguments=input_arguments,
                    folder_path=folder_path,
                    attachments=attachments,
                    parent_span_id=parent_span_id,
                    parent_operation_id=parent_operation_id,
                )

                if job.key:
                    bts_key = (
                        "wait_for_agent_job_key"
                        if resource.type == AgentToolType.AGENT
                        else "wait_for_job_key"
                    )
                    _bts_context[bts_key] = str(job.key)

                return WaitJob(job=job, process_folder_key=job.folder_key)

            return await start_job()

        return await invoke_process(**kwargs)

    job_attachment_wrapper = get_job_attachment_wrapper(output_type=output_model)

    async def process_tool_wrapper(
        tool: BaseTool,
        call: ToolCall,
        state: AgentGraphState,
    ) -> ToolWrapperReturnType:
        call["args"] = handle_static_args(resource, state, call["args"])
        return await job_attachment_wrapper(tool, call, state)

    tool = StructuredToolWithArgumentProperties(
        name=tool_name,
        description=resource.description,
        args_schema=input_model,
        coroutine=process_tool_fn,
        output_type=output_model,
        metadata={
            "tool_type": resource.type.lower(),
            "display_name": process_name,
            "folder_path": folder_path,
            "args_schema": input_model,
            "output_schema": output_model,
            "_span_context": _span_context,
            "_bts_context": _bts_context,
        },
        argument_properties=resource.argument_properties,
    )
    tool.set_tool_wrappers(awrapper=process_tool_wrapper)

    return tool
