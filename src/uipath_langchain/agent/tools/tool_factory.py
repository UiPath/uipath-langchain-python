"""Factory functions for creating tools from agent resources."""

from logging import getLogger

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from uipath.agent.models.agent import (
    AgentClientSideToolResourceConfig,
    AgentContextResourceConfig,
    AgentEscalationResourceConfig,
    AgentIntegrationToolResourceConfig,
    AgentInternalToolResourceConfig,
    AgentIxpExtractionResourceConfig,
    AgentIxpVsEscalationResourceConfig,
    AgentProcessToolResourceConfig,
    BaseAgentResourceConfig,
    LowCodeAgentDefinition,
)

from uipath_langchain.chat.hitl import REQUIRE_CONVERSATIONAL_CONFIRMATION

from .client_side_tool import create_client_side_tool
from .context_tool import create_context_tool
from .escalation_tool import create_escalation_tool
from .extraction_tool import create_ixp_extraction_tool
from .integration_tool import create_integration_tool
from .internal_tools import create_internal_tool
from .ixp_escalation_tool import create_ixp_escalation_tool
from .process_tool import create_process_tool

logger = getLogger(__name__)


def _is_user_token() -> bool:
    """Check if the current token is a user token (sub_type == 'user')."""
    try:
        from uipath._cli._utils._common import get_claim_from_token

        sub_type = get_claim_from_token("sub_type")
        logger.info("Token sub_type=%r", sub_type)
        return sub_type == "user"
    except Exception as e:
        logger.info("Token sub_type check failed: %s", e)
        return False


async def create_tools_from_resources(
    agent: LowCodeAgentDefinition, llm: BaseChatModel
) -> list[BaseTool]:

    tools: list[BaseTool] = []
    is_user = _is_user_token()
    run_as_me = agent.is_conversational and is_user
    logger.info(
        "RunAsMe decision: is_conversational=%s, is_user_token=%s, run_as_me=%s",
        agent.is_conversational,
        is_user,
        run_as_me,
    )

    logger.info("Creating tools for agent '%s' from resources", agent.name)

    for resource in agent.resources:
        if not resource.is_enabled:
            logger.info(
                "Skipping disabled resource '%s' of type '%s'",
                resource.name,
                type(resource).__name__,
            )
            continue

        logger.info(
            "Creating tool for resource '%s' of type '%s'",
            resource.name,
            type(resource).__name__,
        )
        tool = await _build_tool_for_resource(
            resource, llm, agent=agent, run_as_me=run_as_me
        )
        if tool is not None:
            if isinstance(tool, list):
                tools.extend(tool)
            else:
                tools.append(tool)

                if agent.is_conversational:
                    props = getattr(resource, "properties", None)
                    if props and getattr(
                        props, REQUIRE_CONVERSATIONAL_CONFIRMATION, False
                    ):
                        if tool.metadata is None:
                            tool.metadata = {}
                        tool.metadata[REQUIRE_CONVERSATIONAL_CONFIRMATION] = True

    return tools


async def _build_tool_for_resource(
    resource: BaseAgentResourceConfig,
    llm: BaseChatModel,
    agent: LowCodeAgentDefinition | None = None,
    run_as_me: bool = False,
) -> BaseTool | list[BaseTool] | None:
    if isinstance(resource, AgentProcessToolResourceConfig):
        return create_process_tool(resource, run_as_me=run_as_me)

    elif isinstance(resource, AgentContextResourceConfig):
        return create_context_tool(resource, llm=llm, agent=agent)

    elif isinstance(resource, AgentEscalationResourceConfig):
        return create_escalation_tool(resource)

    elif isinstance(resource, AgentIntegrationToolResourceConfig):
        return create_integration_tool(resource)

    elif isinstance(resource, AgentInternalToolResourceConfig):
        return create_internal_tool(resource, llm)

    elif isinstance(resource, AgentIxpExtractionResourceConfig):
        return create_ixp_extraction_tool(resource)

    elif isinstance(resource, AgentIxpVsEscalationResourceConfig):
        return create_ixp_escalation_tool(resource)

    elif isinstance(resource, AgentClientSideToolResourceConfig):
        return create_client_side_tool(resource)

    return None
