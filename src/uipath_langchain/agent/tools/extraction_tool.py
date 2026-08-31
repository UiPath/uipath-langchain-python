"""Ixp extraction tool."""

from typing import Any

from langchain.tools import BaseTool
from langchain_core.messages import ToolCall, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.types import Command, interrupt
from uipath.agent.models.agent import AgentIxpExtractionResourceConfig
from uipath.eval.mocks import mockable
from uipath.platform.attachments import Attachment
from uipath.platform.common import DocumentExtraction
from uipath.platform.documents import ExtractionResponseIXP
from uipath.platform.errors import EnrichedException
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain._utils.durable_interrupt import SUSPENDS_RUN
from uipath_langchain.agent.attachments.job_attachments import (
    get_job_attachment_paths,
    get_job_attachments,
    raise_for_job_attachment_error,
)
from uipath_langchain.agent.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
)
from uipath_langchain.agent.react.jsonschema_pydantic_converter import create_model
from uipath_langchain.agent.react.types import AgentGraphState
from uipath_langchain.agent.tools.tool_node import (
    ToolWrapperMixin,
    ToolWrapperReturnType,
)

from .structured_tool_with_output_type import StructuredToolWithOutputType
from .utils import sanitize_tool_name


class StructuredToolWithWrapper(StructuredToolWithOutputType, ToolWrapperMixin):
    pass


def _single_attachment_input_schema() -> dict[str, Any]:
    """Build the one-job-attachment input schema Agent Builder writes."""
    attachment = Attachment.model_json_schema(by_alias=True)
    attachment["required"] = ["ID"]
    attachment["x-uipath-resource-kind"] = "JobAttachment"
    return {
        "type": "object",
        "properties": {
            "attachment": {
                "description": "File to extract data from.",
                "$ref": "#/definitions/job-attachment",
            }
        },
        "required": ["attachment"],
        "definitions": {"job-attachment": attachment},
    }


def _create_input_model(input_schema: dict[str, Any]) -> Any:
    model = create_model(input_schema)
    if get_job_attachment_paths(model):
        return model
    return create_model(_single_attachment_input_schema())


def create_ixp_extraction_tool(
    resource: AgentIxpExtractionResourceConfig,
) -> StructuredTool:
    """Uses interrupt() to suspend graph execution until data is extracted (handled by runtime)."""
    from uipath_langchain.agent.wrappers import resolve_job_attachment_args

    tool_name: str = sanitize_tool_name(resource.name)
    resource_name = resource.name
    project_name = resource.properties.project_name
    version_tag = resource.properties.version_tag

    input_model: Any = _create_input_model(resource.input_schema)

    @mockable(
        name=resource.name,
        description=resource.description,
        input_schema=input_model.model_json_schema(),
        output_schema=ExtractionResponseIXP.model_json_schema(),
        example_calls=resource.properties.example_calls,
    )
    async def extraction_tool_fn(**kwargs: Any) -> dict[str, Any]:
        from uipath.platform import UiPath

        attachments = get_job_attachments(input_model, kwargs)
        if not attachments:
            raise AgentRuntimeError(
                code=AgentRuntimeErrorCode.INVALID_ATTACHMENT_ID,
                title="Missing job attachment",
                detail=(
                    f"Tool '{resource_name}' was called without a job attachment to "
                    f"extract data from."
                ),
                category=UiPathErrorCategory.USER,
            )
        attachment = attachments[0]
        uipath = UiPath()

        # TODO: current workaround. DocumentExtraction model should support attachment_id and use the
        # start_ixp_extraction_from_attachment sdk method once support is added

        try:
            attachment_local_file_path = await uipath.attachments.download_async(
                key=attachment.id, destination_path=attachment.full_name
            )
        except EnrichedException as e:
            raise_for_job_attachment_error(
                e,
                title="Failed to download job attachment",
                attachment_name=attachment.full_name,
                attachment_id=attachment.id,
            )
            raise
        document_extraction_response = interrupt(
            DocumentExtraction(
                project_name=project_name,
                tag=version_tag,
                file_path=attachment_local_file_path,
            )
        )

        return document_extraction_response

    async def extraction_tool_wrapper(
        tool: BaseTool,
        call: ToolCall,
        state: AgentGraphState,
    ) -> ToolWrapperReturnType:
        error = resolve_job_attachment_args(tool, call, state)
        if error is not None:
            return error

        tool_result = await tool.ainvoke(call["args"])
        data_projection = tool_result["dataProjection"]
        # update the state with extraction response for later reuse in ixpVsEscalation

        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=str(data_projection),
                        name=call["name"],
                        tool_call_id=call["id"],
                    )
                ],
                "inner_state": {"tools_storage": {resource_name: tool_result}},
            }
        )

    tool = StructuredToolWithWrapper(
        name=tool_name,
        description=resource.description,
        args_schema=input_model,
        coroutine=extraction_tool_fn,
        output_type=ExtractionResponseIXP,
        metadata={
            "tool_type": "ixp_extraction",
            SUSPENDS_RUN: True,
            "display_name": resource.name,
            "project_name": project_name,
            "version_tag": version_tag,
            "args_schema": input_model,
        },
    )
    tool.set_tool_wrappers(awrapper=extraction_tool_wrapper)

    return tool
