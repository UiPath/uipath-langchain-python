"""Process tool creation for UiPath process execution."""

import json
from typing import Any

from langchain_core.tools import StructuredTool
from uipath.agent.models.agent import AgentProcessToolResourceConfig, AgentToolType
from uipath.eval.mocks import mockable
from uipath.platform import UiPath
from uipath.platform.common import WaitJobRaw
from uipath.platform.errors import EnrichedException
from uipath.platform.orchestrator import JobState
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain._utils import get_execution_folder_path
from uipath_langchain._utils.durable_interrupt import durable_interrupt
from uipath_langchain.agent.attachments.job_attachments import get_job_attachments
from uipath_langchain.agent.exceptions import raise_for_enriched
from uipath_langchain.agent.react.jsonschema_pydantic_converter import (
    create_model,
    create_output_model,
)
from uipath_langchain.agent.tools.structured_tool_with_argument_properties import (
    StructuredToolWithArgumentProperties,
)

from .utils import sanitize_tool_name

_START_JOBS_404_TEMPLATES: dict[str, str] = {
    "AssociatedProcessNotFound": "Could not find process for tool '{tool}'. Please check if the process is deployed in the configured folder.",
    "AttachmentNotFound": "Could not find an attachment passed to tool '{tool}'. Please check that the attachments provided to the tool still exist.",
}

_START_JOBS_404_FALLBACK_TEMPLATE = "Could not start process for tool '{tool}': an item required to start the job was not found. Server message: {message}"


def _start_jobs_errors(
    e: EnrichedException,
) -> dict[tuple[int, str | None], tuple[str, UiPathErrorCategory]]:
    server_message = (e.error_info.message if e.error_info else None) or ""
    not_found_template = _START_JOBS_404_TEMPLATES.get(
        server_message, _START_JOBS_404_FALLBACK_TEMPLATE
    )
    return {
        (404, "1002"): (
            not_found_template,
            UiPathErrorCategory.DEPLOYMENT,
        ),
        (400, "1100"): (
            "Could not find folder for tool '{tool}'. Please check if the folder exists and is accessible by the robot.",
            UiPathErrorCategory.DEPLOYMENT,
        ),
        (409, None): (
            "Cannot start process for tool '{tool}': {message}",
            UiPathErrorCategory.DEPLOYMENT,
        ),
    }


def create_process_tool(
    resource: AgentProcessToolResourceConfig,
    run_as_me: bool = False,
) -> StructuredTool:
    """Uses interrupt() to suspend graph execution until process completes (handled by runtime)."""
    # Import here to avoid circular dependency
    from uipath_langchain.agent.wrappers import get_job_attachment_wrapper

    tool_name: str = sanitize_tool_name(resource.name)
    process_name = resource.properties.process_name
    folder_path = get_execution_folder_path()

    input_model: Any = create_model(resource.input_schema)
    output_model: Any = create_output_model(resource.output_schema, resource.name)

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
                try:
                    job = await client.processes.invoke_async(
                        name=process_name,
                        input_arguments=input_arguments,
                        folder_path=folder_path,
                        attachments=attachments,
                        parent_span_id=parent_span_id,
                        parent_operation_id=parent_operation_id,
                        run_as_me=True if run_as_me else None,
                    )
                except EnrichedException as e:
                    raise_for_enriched(
                        e,
                        _start_jobs_errors(e),
                        title=f"Failed to execute tool '{resource.name}'",
                        tool=resource.name,
                    )
                    raise

                if job.key:
                    bts_key = (
                        "wait_for_agent_job_key"
                        if resource.type == AgentToolType.AGENT
                        else "wait_for_job_key"
                    )
                    _bts_context[bts_key] = str(job.key)

                return WaitJobRaw(job=job, process_folder_key=job.folder_key)

            job = await start_job()

            if (job.state or "").lower() == JobState.FAULTED:
                error_info = str(job.info or "Unknown error")
                return f"{error_info}"

            client = UiPath()
            output_str = await client.jobs.extract_output_async(job)
            if output_str:
                try:
                    return json.loads(output_str)
                except (json.JSONDecodeError, TypeError):
                    return output_str
            return output_str

        return await invoke_process(**kwargs)

    job_attachment_wrapper = get_job_attachment_wrapper(output_type=output_model)

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
    tool.set_tool_wrappers(awrapper=job_attachment_wrapper)

    return tool
