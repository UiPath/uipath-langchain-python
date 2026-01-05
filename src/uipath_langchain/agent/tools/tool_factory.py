"""Factory functions for creating tools from agent resources."""

from langchain_core.tools import BaseTool, StructuredTool
from uipath.agent.models.agent import (
    AgentContextResourceConfig,
    AgentContextRetrievalMode,
    AgentEscalationResourceConfig,
    AgentIntegrationToolResourceConfig,
    AgentProcessToolResourceConfig,
    BaseAgentResourceConfig,
    LowCodeAgentDefinition,
)

from .context_tool import create_context_tool
from .deeprag_tool import create_deeprag_tool
from .escalation_tool import create_escalation_tool
from .integration_tool import create_integration_tool
from .process_tool import create_process_tool


async def create_tools_from_resources(agent: LowCodeAgentDefinition) -> list[BaseTool]:
    tools: list[BaseTool] = []

    for resource in agent.resources:
        tool = await _build_tool_for_resource(resource)
        if tool is not None:
            tools.append(tool)

    return tools


async def _build_tool_for_resource(
    resource: BaseAgentResourceConfig,
) -> StructuredTool | None:
    if isinstance(resource, AgentProcessToolResourceConfig):
        return create_process_tool(resource)

    elif isinstance(resource, AgentContextResourceConfig):
        # Route to DeepRAG tool if retrieval mode is DEEP_RAG
        if resource.settings.retrieval_mode == AgentContextRetrievalMode.DEEP_RAG:
            return create_deeprag_tool(resource)
        return create_context_tool(resource)

    elif isinstance(resource, AgentEscalationResourceConfig):
        return create_escalation_tool(resource)

    elif isinstance(resource, AgentIntegrationToolResourceConfig):
        return create_integration_tool(resource)

    return None
