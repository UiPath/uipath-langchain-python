"""Process tool creation for UiPath process execution."""

from typing import Any

from langchain.tools import BaseTool
from langchain_core.messages import ToolCall
from langchain_core.tools import StructuredTool
from langgraph.types import interrupt
from uipath.agent.models.agent import AgentProcessToolResourceConfig
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

from .durable_interrupt import durable_task
from .utils import sanitize_tool_name


def create_process_tool(resource: AgentProcessToolResourceConfig) -> StructuredTool:
    """Uses interrupt() to suspend graph execution until process completes (handled by runtime)."""
    # Import here to avoid circular dependency
    from uipath_langchain.agent.wrappers import get_job_attachment_wrapper

    tool_name: str = sanitize_tool_name(resource.name)
    process_name = resource.properties.process_name
    folder_path = resource.properties.folder_path

    input_model: Any = create_model(resource.input_schema)
    output_model: Any = create_model(resource.output_schema)

    _span_context: dict[str, Any] = {}

    async def process_tool_fn(**kwargs: Any):
        attachments = get_job_attachments(input_model, kwargs)
        input_arguments = input_model.model_validate(kwargs).model_dump(mode="json")

        @mockable(
            name=tool_name.lower(),
            description=resource.description,
            input_schema=input_model.model_json_schema(),
            output_schema=output_model.model_json_schema(),
            example_calls=resource.properties.example_calls,
        )
        async def invoke_process():
            parent_span_id = _span_context.pop("parent_span_id", None)

            @durable_task
            async def start_job():
                client = UiPath()
                job = await client.processes.invoke_async(
                    name=process_name,
                    input_arguments=input_arguments,
                    folder_path=folder_path,
                    attachments=attachments,
                    parent_span_id=parent_span_id,
                )
                return WaitJob(job=job, process_folder_key=job.folder_key)

            return interrupt(await start_job())

        return await invoke_process()

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
        },
        argument_properties=resource.argument_properties,
    )
    tool.set_tool_wrappers(awrapper=process_tool_wrapper)

    return tool
