"""Factory functions for creating tools from agent resources."""

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool, StructuredTool
from uipath.agent.models.agent import (
    AgentContextResourceConfig,
    AgentEscalationResourceConfig,
    AgentIntegrationToolResourceConfig,
    AgentInternalToolResourceConfig,
    AgentProcessToolResourceConfig,
    BaseAgentResourceConfig,
    LowCodeAgentDefinition,
)

from .context_tool import create_context_tool
from .escalation_tool import create_escalation_tool
from .integration_tool import create_integration_tool
from .internal_tools import create_internal_tool
from .process_tool import create_process_tool


async def create_tools_from_resources(
    agent: LowCodeAgentDefinition, llm: BaseChatModel
) -> list[BaseTool]:
    tools: list[BaseTool] = []

    for resource in agent.resources:
        tool = await _build_tool_for_resource(resource, llm)
        if tool is not None:
            tools.append(tool)

    return tools


async def _build_tool_for_resource(
    resource: BaseAgentResourceConfig, llm: BaseChatModel
) -> StructuredTool | None:
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

    return None
