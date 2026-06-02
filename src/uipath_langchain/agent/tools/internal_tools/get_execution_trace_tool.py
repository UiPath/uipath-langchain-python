"""Internal tool that fetches the execution trace (audit trail) for a PIMs case."""

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import StructuredTool
from uipath.agent.models.agent import AgentInternalToolResourceConfig

from ._pims_read_tool_factory import build_pims_read_tool

PIMS_EXECUTION_TRACE_PATH = (
    "pims_/api/v2/element-executions/case-instances/{instance_id}"
)


def create_get_execution_trace_tool(
    resource: AgentInternalToolResourceConfig, llm: BaseChatModel
) -> StructuredTool:
    """Create the GetExecutionTrace internal tool.

    Returns the audit trail (``elementExecutions[]``, ``traceId``, etc.) from
    the PIMs ``v2/element-executions/case-instances`` endpoint. Shared
    HTTP/auth/folder/error logic lives in :func:`build_pims_read_tool`.
    """
    return build_pims_read_tool(
        resource,
        path_template=PIMS_EXECUTION_TRACE_PATH,
        subject="execution trace",
    )
