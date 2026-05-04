"""Ixp extraction tool."""

import uuid
from typing import Any, Optional

from langchain.tools import BaseTool
from langchain_core.messages import ToolCall, ToolMessage
from langchain_core.tools import StructuredTool
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field
from uipath.agent.models.agent import AgentIxpExtractionResourceConfig
from uipath.eval.mocks import mockable
from uipath.platform.common import DocumentExtraction
from uipath.platform.documents import ExtractionResponseIXP

from uipath_langchain.agent.react.types import AgentGraphState
from uipath_langchain.agent.tools.tool_node import (
    ToolWrapperMixin,
    ToolWrapperReturnType,
)

from .structured_tool_with_output_type import StructuredToolWithOutputType
from .utils import sanitize_tool_name


class StructuredToolWithWrapper(StructuredToolWithOutputType, ToolWrapperMixin):
    pass


class ExtractionToolInputSchema(BaseModel):
    """Alias-free mirror of `Attachment` used as the tool's args_schema.

    We don't use `Attachment` directly because its fields carry aliases
    (`id` -> `ID`, `full_name` -> `FullName`, ...) and LangChain mishandles
    aliased fields in two places (see PR #796):

    1. `BaseTool._parse_input()` extracts each field with `getattr(model, key)`,
       where `key` is the alias. For aliases that collide with built-in model
       attributes (e.g. `schema`), this returns the built-in instead of the
       field value, so downstream `kwargs.get("id") / kwargs.get("full_name")`
       came back as `None`.
    2. `tool_call_schema` rebuilds a subset of the model by copying each field
       but drops alias and serialization options, so the rebuilt schema no
       longer matches what the LLM emits.

    Until LangChain fixes both, exposing an alias-free schema with field
    names matching `Attachment`'s python names sidesteps the issue. Keep the
    fields here in sync with `Attachment` — the test
    `test_extraction_tool_has_attachment_input_schema` enforces this.
    """

    id: uuid.UUID
    full_name: str
    mime_type: str
    metadata: Optional[dict[str, Any]] = Field(None)


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
        input_schema=ExtractionToolInputSchema.model_json_schema(),
        output_schema=ExtractionResponseIXP.model_json_schema(),
        example_calls=resource.properties.example_calls,
    )
    async def extraction_tool_fn(**kwargs: Any) -> ExtractionResponseIXP:
        from uipath.platform import UiPath

        attachment = ExtractionToolInputSchema.model_validate(kwargs)
        uipath = UiPath()

        # TODO: current workaround. DocumentExtraction model should support attachment_id and use the
        # start_ixp_extraction_from_attachment sdk method once support is added

        attachment_local_file_path = await uipath.attachments.download_async(
            key=attachment.id, destination_path=attachment.full_name
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
        args_schema=ExtractionToolInputSchema,
        coroutine=extraction_tool_fn,
        output_type=ExtractionResponseIXP,
        metadata={
            "tool_type": "ixp_extraction",
            "display_name": resource.name,
            "project_name": project_name,
            "version_tag": version_tag,
        },
    )
    tool.set_tool_wrappers(awrapper=extraction_tool_wrapper)

    return tool
