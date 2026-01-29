"""Ixp escalation tool."""

from langchain.tools import BaseTool
from langchain_core.messages import ToolCall
from langchain_core.tools import StructuredTool
from langgraph.types import interrupt
from pydantic import BaseModel
from uipath.agent.models.agent import AgentIxpVsEscalationResourceConfig
from uipath.eval.mocks import mockable
from uipath.platform.common import DocumentExtractionValidation
from uipath.platform.documents import (
    ActionPriority,
    ExtractionResponseIXP,
    FieldGroupValueProjection,
)

from uipath_langchain.agent.react.types import AgentGraphState
from uipath_langchain.agent.tools.tool_node import (
    ToolWrapperMixin,
    ToolWrapperReturnType,
)

from .structured_tool_with_output_type import StructuredToolWithOutputType
from .utils import sanitize_tool_name


class StructuredToolWithWrapper(StructuredToolWithOutputType, ToolWrapperMixin):
    pass


def create_ixp_escalation_tool(
    resource: AgentIxpVsEscalationResourceConfig,
) -> StructuredTool:
    """Uses interrupt() to suspend graph execution until data is extracted (handled by runtime)."""
    tool_name: str = sanitize_tool_name(resource.name)
    storage_bucket_name: str = resource.vs_escalation_properties.storage_bucket_name
    storage_bucket_folder_path: str = (
        resource.vs_escalation_properties.storage_bucket_folder_path
    )
    action_priority: ActionPriority = (
        resource.vs_escalation_properties.action_priority or ActionPriority.MEDIUM
    )
    action_title: str = (
        resource.vs_escalation_properties.action_title or "VS Escalation Task"
    )

    ixp_tool_name: str = resource.vs_escalation_properties.ixp_tool_id

    class OutputSchema(BaseModel):
        data: list[FieldGroupValueProjection] | None

    @mockable(
        name=resource.name,
        description=resource.description,
        input_schema={},
        output_schema=OutputSchema.model_json_schema(),
        example_calls=[],
    )
    async def ixp_escalation_tool(
        extraction_result: ExtractionResponseIXP,
    ) -> OutputSchema:
        response = interrupt(
            DocumentExtractionValidation(
                extraction_response=extraction_result,
                action_title=action_title,
                storage_bucket_name=storage_bucket_name,
                action_folder=storage_bucket_folder_path,
                action_priority=action_priority,
            )
        )
        return OutputSchema(data=response["dataProjection"])

    async def ixp_escalation_tool_wrapper(
        tool: BaseTool,
        call: ToolCall,
        state: AgentGraphState,
    ) -> ToolWrapperReturnType:
        extraction_result = state.inner_state.tools_storage.get(ixp_tool_name)
        if not extraction_result:
            raise RuntimeError(
                f"Extraction result not found for {ixp_tool_name} ixp extraction tool."
            )

        call["args"]["extraction_result"] = extraction_result
        return await tool.ainvoke(call["args"])

    tool = StructuredToolWithWrapper(
        name=tool_name,
        description=resource.description,
        args_schema={},
        coroutine=ixp_escalation_tool,
        output_type=OutputSchema,
    )
    tool.set_tool_wrappers(awrapper=ixp_escalation_tool_wrapper)

    return tool
