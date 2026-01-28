"""Ixp extraction tool."""

from typing import Any

from langchain.tools import BaseTool
from langchain_core.messages import ToolCall
from langchain_core.tools import StructuredTool
from langgraph.types import interrupt
from uipath.agent.models.agent import AgentIxpExtractionResourceConfig
from uipath.eval.mocks import mockable
from uipath.platform.attachments import Attachment
from uipath.platform.common import DocumentExtraction
from uipath.platform.documents import ExtractionResponseIXP

from uipath_langchain.agent.react.types import AgentGraphState
from uipath_langchain.agent.tools.tool_node import ToolWrapperReturnType

from .integration_tool import StructuredToolWithWrapper
from .utils import sanitize_tool_name


def create_ixp_extraction_tool(
    resource: AgentIxpExtractionResourceConfig,
) -> StructuredTool:
    """Uses interrupt() to suspend graph execution until data is extracted (handled by runtime)."""
    tool_name: str = sanitize_tool_name(resource.name)
    resource_name = resource.name
    project_name = resource.properties.project_name
    version_tag = resource.properties.version_tag

    @mockable(
        name=resource.name,
        description=resource.description,
        input_schema=Attachment.model_json_schema(),
        output_schema=ExtractionResponseIXP.model_json_schema(),
        example_calls=resource.properties.example_calls,
    )
    async def extraction_tool_fn(**kwargs: Any) -> ExtractionResponseIXP:
        from uipath.platform import UiPath

        uipath = UiPath()

        attachment_id = kwargs.get("id")
        attachment_full_name = kwargs.get("full_name")

        # TODO: attachment_mime_type is currently not used anywhere (attachment_full_name will also be obsolete once attachments api is onboarded)
        # should we use them somewhere else? otherwise input_schema should only contain the file id
        # attachment_mime_type = kwargs.get("mime_type")

        # TODO: current workaround. DocumentExtraction model should support attachment_id and use the
        # start_ixp_extraction_from_attachment sdk method once support is added

        attachment_local_file_path = await uipath.attachments.download_async(
            key=attachment_id, destination_path=attachment_full_name
        )
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
        result = await tool.ainvoke(call["args"])

        # store extraction response for later reuse in vsEscalation
        state.inner_state.mappings = {resource_name: result}

        return result

    tool = StructuredToolWithWrapper(
        name=tool_name,
        description=resource.description,
        args_schema=Attachment,
        coroutine=extraction_tool_fn,
        output_type=ExtractionResponseIXP,
    )
    tool.set_tool_wrappers(awrapper=extraction_tool_wrapper)

    return tool
