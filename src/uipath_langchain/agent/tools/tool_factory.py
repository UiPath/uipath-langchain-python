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
) -> tuple[list[BaseTool], str]:
    """Create tools from agent resources including Data Fabric tools.

    Args:
        agent: The agent definition.
        llm: The language model for tool creation.

    Returns:
        Tuple of (tools, datafabric_schema_context).
    """
    tools: list[BaseTool] = []
    datafabric_schema_context: str = ""

    logger.info("Creating tools for agent '%s' from resources", agent.name)

    # Handle Data Fabric tools first (they need special handling)
    datafabric_tools, schema_context = await create_datafabric_tools(agent)
    tools.extend(datafabric_tools)
    datafabric_schema_context = schema_context

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

    return tools, datafabric_schema_context


async def _build_tool_for_resource(
    resource: BaseAgentResourceConfig, llm: BaseChatModel
) -> BaseTool | list[BaseTool] | None:
    if isinstance(resource, AgentProcessToolResourceConfig):
        return create_process_tool(resource)

    elif isinstance(resource, AgentContextResourceConfig):
        # Skip Data Fabric contexts - handled separately via create_datafabric_tools()
        retrieval_mode = resource.settings.retrieval_mode
        mode_value = retrieval_mode.value if hasattr(retrieval_mode, 'value') else str(retrieval_mode)
        if mode_value.lower() == "datafabric":
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
