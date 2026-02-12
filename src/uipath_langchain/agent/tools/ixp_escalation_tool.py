"""Ixp escalation tool."""

from typing import Any

from langchain.tools import BaseTool
from langchain_core.messages import ToolCall
from langchain_core.tools import StructuredTool
from langgraph.types import interrupt
from pydantic import BaseModel
from uipath.agent.models.agent import AgentIxpVsEscalationResourceConfig
from uipath.eval.mocks import mockable
from uipath.platform import UiPath
from uipath.platform.common import WaitDocumentExtractionValidation
from uipath.platform.documents import (
    ActionPriority,
    ExtractionResponseIXP,
    FieldGroupValueProjection,
    StartExtractionValidationResponse,
)
from uipath.runtime.errors import UiPathErrorCode

from uipath_langchain.agent.react.types import AgentGraphState
from uipath_langchain.agent.tools.tool_node import (
    ToolWrapperMixin,
    ToolWrapperReturnType,
)

from ..exceptions import AgentTerminationException
from .durable_interrupt import durable_task
from .structured_tool_with_output_type import StructuredToolWithOutputType
from .utils import (
    resolve_task_title,
    sanitize_dict_for_serialization,
    sanitize_tool_name,
)


class StructuredToolWithWrapper(StructuredToolWithOutputType, ToolWrapperMixin):
    pass


def create_ixp_escalation_tool(
    resource: AgentIxpVsEscalationResourceConfig,
) -> StructuredTool:
    """Uses interrupt() to suspend graph execution until extraction is validated."""
    tool_name: str = sanitize_tool_name(resource.name)
    storage_bucket_name: str = resource.vs_escalation_properties.storage_bucket_name
    storage_bucket_folder_path: str = (
        resource.vs_escalation_properties.storage_bucket_folder_path
    )
    channel = resource.channels[0]
    action_priority = ActionPriority.from_str(channel.priority)
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
    async def ixp_escalation_tool() -> dict[str, Any]:
        if tool.metadata is None:
            raise RuntimeError("Tool metadata is required")
        extraction_result = tool.metadata.get("extraction_result")
        if not isinstance(extraction_result, ExtractionResponseIXP):
            raise RuntimeError("extraction_result not set in metadata")
        task_title = tool.metadata.get("task_title") or "VS Escalation Task"

        @durable_task
        async def start_extraction_validation():
            uipath = UiPath()
            validation_response = (
                await uipath.documents.create_validate_extraction_action_async(
                    extraction_response=extraction_result,
                    action_title=task_title,
                    storage_bucket_name=storage_bucket_name,
                    storage_bucket_directory_path=storage_bucket_folder_path,
                    action_priority=action_priority,
                )
            )

            start_response = StartExtractionValidationResponse(
                operation_id=validation_response.operation_id,
                document_id=extraction_result.extraction_result.document_id,
                project_id=extraction_result.project_id,
                tag=extraction_result.tag,
            )
            task_url = validation_response.action_data.get("taskUrl")

            return WaitDocumentExtractionValidation(
                extraction_validation=start_response,
                task_url=task_url,
            )

        wait_obj = await start_extraction_validation()
        response = interrupt(wait_obj)
        # store validation response in tool metadata to check for exceptions in the wrapper
        tool.metadata["_validation_response"] = response

        return OutputSchema(data=response["dataProjection"]).model_dump(by_alias=True)

    async def ixp_escalation_tool_wrapper(
        tool: BaseTool,
        call: ToolCall,
        state: AgentGraphState,
    ) -> ToolWrapperReturnType:
        if tool.metadata is None:
            raise RuntimeError("Tool metadata is required for task_title resolution")

        tool.metadata["task_title"] = resolve_task_title(
            channel.task_title,
            sanitize_dict_for_serialization(dict(state)),
            default_title="VS Escalation Task",
        )

        extraction_result_dict = state.inner_state.tools_storage.get(ixp_tool_name)
        if not extraction_result_dict:
            raise RuntimeError(
                f"Extraction result not found for {ixp_tool_name} ixp extraction tool."
            )

        tool.metadata["extraction_result"] = ExtractionResponseIXP(
            **extraction_result_dict
        )
        result = await tool.ainvoke(call["args"])

        validation_response = tool.metadata.get("_validation_response", {})
        if validation_response and (
            rejection_details := validation_response["actionData"]["data"].get(
                "documentRejectionDetails"
            )
        ):
            detail_template = "Vs Escalation Exception: {}"
            raise AgentTerminationException(
                code=UiPathErrorCode.EXECUTION_ERROR,
                title="VS Escalation Exception",
                detail=detail_template.format(
                    rejection_details.get("reason")
                    or "The validation was marked as exception."
                ),
            )

        return result

    tool = StructuredToolWithWrapper(
        name=tool_name,
        description=resource.description,
        args_schema={},
        coroutine=ixp_escalation_tool,
        output_type=OutputSchema,
        metadata={
            "tool_type": "vs_escalation",
            "display_name": channel.properties.app_name,
            "channel_type": channel.type,
            "ixp_tool_id": ixp_tool_name,
            "storage_bucket_name": storage_bucket_name,
        },
    )
    tool.set_tool_wrappers(awrapper=ixp_escalation_tool_wrapper)

    return tool
