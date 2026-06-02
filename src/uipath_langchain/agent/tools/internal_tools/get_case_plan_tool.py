"""Internal tool that fetches the case plan (stages, tasks, rules) for a PIMs case."""

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import StructuredTool
from uipath.agent.models.agent import AgentInternalToolResourceConfig

from ._pims_read_tool_factory import build_pims_read_tool

PIMS_CASE_PLAN_PATH = "pims_/api/v1/cases/{instance_id}/case-rules"


def create_get_case_plan_tool(
    resource: AgentInternalToolResourceConfig, llm: BaseChatModel
) -> StructuredTool:
    """Create the GetCasePlan internal tool.

    Returns the full case plan from the PIMs ``case-rules`` endpoint: every
    stage and task with entry/exit rules, ``isRequired``, and
    ``shouldRunOnlyOnce``. Shared HTTP/auth/folder/error logic lives in
    :func:`build_pims_read_tool`.
    """
    return build_pims_read_tool(
        resource,
        path_template=PIMS_CASE_PLAN_PATH,
        subject="case plan",
    )
