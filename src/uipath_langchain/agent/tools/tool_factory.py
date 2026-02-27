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

from .context_tool import create_context_tool
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
    tools: list[BaseTool] = []

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
        tool = await _build_tool_for_resource(resource, llm)
        if tool is not None:
            # propagate requireConversationalConfirmation to tool metadata (conversational agents only)
            if agent.is_conversational:
                tool_list = tool if isinstance(tool, list) else [tool]
                props = getattr(resource, "properties", None)
                if props and getattr(
                    props, "require_conversational_confirmation", False
                ):
                    # some resources (like mcp) can return a list of tools, so normalize to a list
                    for t in tool_list:
                        if t.metadata is None:
                            t.metadata = {}
                        t.metadata["require_conversational_confirmation"] = True

            if isinstance(tool, list):
                tools.extend(tool)
            else:
                tools.append(tool)

    return tools


async def _build_tool_for_resource(
    resource: BaseAgentResourceConfig, llm: BaseChatModel
) -> BaseTool | list[BaseTool] | None:
    if isinstance(resource, AgentProcessToolResourceConfig):
        return create_process_tool(resource)

    elif isinstance(resource, AgentContextResourceConfig):
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
