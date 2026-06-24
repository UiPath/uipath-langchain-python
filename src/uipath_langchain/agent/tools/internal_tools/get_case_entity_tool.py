"""Internal tool that fetches the case entity (business object, documents, comments) for a PIMs case."""

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import StructuredTool
from uipath.agent.models.agent import AgentInternalToolResourceConfig

from ._pims_read_tool_factory import build_pims_read_tool

PIMS_CASE_ENTITY_PATH = "pims_/api/v1/cases/{instance_id}/case-json"


def create_get_case_entity_tool(
    resource: AgentInternalToolResourceConfig, llm: BaseChatModel
) -> StructuredTool:
    """Create the GetCaseEntity internal tool.

    Returns the case-json payload from the PIMs ``case-json`` endpoint. Shared
    HTTP/auth/folder/error logic lives in :func:`build_pims_read_tool`.

    Note: as of testing on alpha, ``case-json`` returns the case **definition**
    (id, version, metadata, stages) rather than the business object / linked
    documents / comments. The "entity" naming is provisional pending the CM
    team confirming the right surface.
    """
    return build_pims_read_tool(
        resource,
        path_template=PIMS_CASE_ENTITY_PATH,
        subject="case entity",
    )
