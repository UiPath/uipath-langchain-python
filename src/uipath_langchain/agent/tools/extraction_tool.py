"""Process tool creation for UiPath process execution."""

from typing import Any

from langchain_core.tools import StructuredTool
from langgraph.types import interrupt
from uipath.agent.models.agent import AgentIxpExtractionResourceConfig
from uipath.eval.mocks import mockable
from uipath.platform.attachments import Attachment
from uipath.platform.common import DocumentExtraction
from uipath.platform.documents import ExtractionResponseIXP

from .structured_tool_with_output_type import StructuredToolWithOutputType
from .utils import sanitize_tool_name


def create_ixp_extraction_tool(
    resource: AgentIxpExtractionResourceConfig,
) -> StructuredTool:
    """Uses interrupt() to suspend graph execution until data is extracted (handled by runtime)."""
    tool_name: str = sanitize_tool_name(resource.name)
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
        return interrupt(
            DocumentExtraction(
                project_name=project_name,
                tag=version_tag,
                file_path=attachment_local_file_path,
            )
        )

    tool = StructuredToolWithOutputType(
        name=tool_name,
        description=resource.description,
        args_schema=Attachment,
        coroutine=extraction_tool_fn,
        output_type=ExtractionResponseIXP,
    )

    return tool
