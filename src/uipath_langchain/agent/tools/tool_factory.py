"""Factory functions for creating tools from agent resources."""

from logging import getLogger

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from uipath.agent.models.agent import (
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

from .context_tool import create_context_tool
from .datafabric_tool import create_datafabric_tools
from .escalation_tool import create_escalation_tool
from .extraction_tool import create_ixp_extraction_tool
from .integration_tool import create_integration_tool
from .internal_tools import create_internal_tool
from .ixp_escalation_tool import create_ixp_escalation_tool
from .process_tool import create_process_tool

logger = getLogger(__name__)


async def create_tools_from_resources(
    agent: LowCodeAgentDefinition, llm: BaseChatModel
) -> list[BaseTool]:
    """Create tools from agent resources including Data Fabric tools.

    Args:
        agent: The agent definition.
        llm: The language model for tool creation.

    Returns:
        List of BaseTool instances.
    """
    tools: list[BaseTool] = []

    logger.info("Creating tools for agent '%s' from resources", agent.name)

    # Register the generic Data Fabric query tool (no fetching/schema here)
    tools.extend(create_datafabric_tools(agent))

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
        tool = await _build_tool_for_resource(resource, llm)
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
    resource: BaseAgentResourceConfig, llm: BaseChatModel
) -> BaseTool | list[BaseTool] | None:
    if isinstance(resource, AgentProcessToolResourceConfig):
        return create_process_tool(resource)

    elif isinstance(resource, AgentContextResourceConfig):
        if resource.is_datafabric:
            logger.info(
                "Skipping Data Fabric context '%s' - handled separately",
                resource.name,
            )
            return None
        return create_context_tool(resource)

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

    return None
