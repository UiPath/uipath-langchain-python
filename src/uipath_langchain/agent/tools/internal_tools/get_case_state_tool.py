"""Internal tool that fetches the current state of a PIMs case instance."""

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import StructuredTool
from uipath.agent.models.agent import AgentInternalToolResourceConfig

from ._pims_read_tool_factory import build_pims_read_tool

PIMS_INSTANCE_PATH = "pims_/api/v1/instances/{instance_id}"


def create_get_case_state_tool(
    resource: AgentInternalToolResourceConfig, llm: BaseChatModel
) -> StructuredTool:
    """Create the GetCaseState internal tool.

    Returns the case instance state from the PIMs ``/v1/instances/{id}``
    endpoint. Shared HTTP/auth/folder/error logic lives in
    :func:`build_pims_read_tool`.
    """
    return build_pims_read_tool(
        resource,
        path_template=PIMS_INSTANCE_PATH,
        subject="case state",
    )
