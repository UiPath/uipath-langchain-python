"""Tests for tool_factory.py module."""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from langchain_core.language_models import BaseChatModel
from uipath.agent.models.agent import (
    AgentContextQuerySetting,
    AgentContextResourceConfig,
    AgentContextRetrievalMode,
    AgentContextSettings,
    AgentEscalationChannel,
    AgentEscalationChannelProperties,
    AgentEscalationResourceConfig,
    AgentIntegrationToolProperties,
    AgentIntegrationToolResourceConfig,
    AgentInternalAnalyzeFilesToolProperties,
    AgentInternalToolResourceConfig,
    AgentInternalToolType,
    AgentIxpExtractionResourceConfig,
    AgentIxpExtractionToolProperties,
    AgentMcpResourceConfig,
    AgentMcpTool,
    AgentProcessToolProperties,
    AgentProcessToolResourceConfig,
    AgentResourceType,
    AgentSettings,
    AgentToolType,
    LowCodeAgentDefinition,
)
from uipath.platform.connections import Connection

from uipath_langchain.agent.tools.base_uipath_structured_tool import (
    BaseUiPathStructuredTool,
)
from uipath_langchain.agent.tools.tool_factory import (
    _build_tool_for_resource,
    create_tools_from_resources,
)

# Common test data
EMPTY_SCHEMA = {"type": "object", "properties": {}}


@pytest.fixture
def mock_uipath_sdk():
    """Create a mock UiPath SDK."""
    with patch("uipath_langchain.agent.tools.integration_tool.UiPath") as mock:
        mock.return_value = MagicMock()
        # Note: MCP tools no longer call SDK during tool creation.
        # The SDK is called lazily in McpClient._initialize_client() on first tool use.
        yield mock


@pytest.fixture
def process_resource() -> AgentProcessToolResourceConfig:
    """Create a process tool resource config."""
    return AgentProcessToolResourceConfig(
        type=AgentToolType.PROCESS,
        name="test_process",
        description="Test process description",
        input_schema=EMPTY_SCHEMA,
        output_schema=EMPTY_SCHEMA,
        properties=AgentProcessToolProperties(
            process_name="MyProcess",
            folder_path="/Shared/MyFolder",
        ),
    )


@pytest.fixture
def context_resource() -> AgentContextResourceConfig:
    """Create a context tool resource config."""
    return AgentContextResourceConfig(
        resource_type=AgentResourceType.CONTEXT,
        name="test_context",
        description="Test context description",
        index_name="test_index",
        folder_path="/Shared/MyFolder",
        settings=AgentContextSettings(
            retrieval_mode=AgentContextRetrievalMode.SEMANTIC,
            result_count=10,
            query=AgentContextQuerySetting(
                variant="static",
                value="test query",
                description="test description",
            ),
        ),
    )


@pytest.fixture
def escalation_resource() -> AgentEscalationResourceConfig:
    """Create an escalation tool resource config."""
    return AgentEscalationResourceConfig(
        resource_type=AgentResourceType.ESCALATION,
        name="test_escalation",
        description="Test escalation description",
        channels=[
            AgentEscalationChannel(
                name="test_channel",
                type="action_center",
                description="Test channel description",
                task_title="Test Task",
                input_schema=EMPTY_SCHEMA,
                output_schema=EMPTY_SCHEMA,
                properties=AgentEscalationChannelProperties(
                    app_name="TestApp",
                    folder_name="/Shared/MyFolder",
                    app_version=1,
                    resource_key="test-key",
                ),
                recipients=[],
            )
        ],
    )


@pytest.fixture
def integration_resource() -> AgentIntegrationToolResourceConfig:
    """Create an integration tool resource config."""
    return AgentIntegrationToolResourceConfig(
        type=AgentToolType.INTEGRATION,
        name="test_integration",
        description="Test integration description",
        input_schema=EMPTY_SCHEMA,
        output_schema=EMPTY_SCHEMA,
        properties=AgentIntegrationToolProperties(
            method="GET",
            tool_path="/api/test",
            object_name="test_object",
            tool_display_name="Test Tool",
            tool_description="Test tool description",
            connection=Connection(
                id="test-connection-id",
                name="Test Connection",
                element_instance_id=12345,
            ),
            parameters=[],
        ),
    )


@pytest.fixture
def internal_resource() -> AgentInternalToolResourceConfig:
    """Create an internal tool resource config."""
    return AgentInternalToolResourceConfig(
        type=AgentToolType.INTERNAL,
        name="test_internal",
        description="Test internal description",
        input_schema={
            "type": "object",
            "properties": {
                "analysisTask": {"type": "string"},
                "attachments": {"type": "array"},
            },
        },
        output_schema=EMPTY_SCHEMA,
        properties=AgentInternalAnalyzeFilesToolProperties(
            tool_type=AgentInternalToolType.ANALYZE_FILES,
        ),
    )


@pytest.fixture
def ixp_extraction_resource() -> AgentIxpExtractionResourceConfig:
    """Create an IXP extraction tool resource config."""
    return AgentIxpExtractionResourceConfig(
        type=AgentToolType.IXP,
        name="test_extraction",
        description="Test extraction description",
        input_schema=EMPTY_SCHEMA,
        output_schema=EMPTY_SCHEMA,
        properties=AgentIxpExtractionToolProperties(
            project_name="TestProject",
            version_tag="v1.0",
        ),
    )


@pytest.fixture
def mcp_resource() -> AgentMcpResourceConfig:
    """Create an MCP tool resource config with multiple tools."""
    return AgentMcpResourceConfig(
        resource_type=AgentResourceType.MCP,
        name="test_mcp",
        description="Test MCP description",
        slug="test-mcp-slug",
        folder_path="/Shared/MyFolder",
        is_enabled=True,
        available_tools=[
            AgentMcpTool(
                name="tool1",
                description="Tool 1",
                input_schema=EMPTY_SCHEMA,
            ),
            AgentMcpTool(
                name="tool2",
                description="Tool 2",
                input_schema=EMPTY_SCHEMA,
            ),
        ],
    )


@pytest.fixture
def all_resources(
    process_resource,
    context_resource,
    escalation_resource,
    integration_resource,
    internal_resource,
    ixp_extraction_resource,
    mcp_resource,
):
    """Fixture providing all resource types."""
    return [
        process_resource,
        context_resource,
        escalation_resource,
        integration_resource,
        internal_resource,
        ixp_extraction_resource,
        mcp_resource,
    ]


def assert_tool_is_base_uipath(tool):
    """Helper to assert tool is BaseUiPathStructuredTool instance."""
    if isinstance(tool, list):
        for t in tool:
            assert isinstance(t, BaseUiPathStructuredTool)
    else:
        assert tool is not None
        assert isinstance(tool, BaseUiPathStructuredTool)


@pytest.mark.asyncio
class TestCreateToolsFromResources:
    """Test cases for create_tools_from_resources function."""

    async def test_only_enabled_tools_returned(
        self, process_resource, context_resource
    ):
        """Test that only enabled tools are returned from resources."""
        enabled_process_tool = process_resource
        enabled_process_tool.is_enabled = True
        enabled_process_tool.name = "EnabledProcess"

        disabled_context_tool = context_resource
        disabled_context_tool.is_enabled = False

        agent = LowCodeAgentDefinition(
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "object", "properties": {}},
            messages=[],
            settings=Mock(spec=AgentSettings),
            resources=[enabled_process_tool, disabled_context_tool],
        )

        mock_llm = AsyncMock(spec=BaseChatModel)
        tools = await create_tools_from_resources(agent, mock_llm)

        assert len(tools) == 1
        assert tools[0].name == "EnabledProcess"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "resource_fixture",
        [
            "process_resource",
            "context_resource",
            "escalation_resource",
            "integration_resource",
            "internal_resource",
            "ixp_extraction_resource",
            # Note: mcp_resource is excluded because MCP tools are created
            # separately via create_mcp_tools_from_agent, not through
            # _build_tool_for_resource
        ],
    )
    async def test_resource_produces_base_uipath_tool(
        self, resource_fixture, mock_uipath_sdk, request
    ):
        """Test that each resource type produces BaseUiPathStructuredTool instance."""
        resource = request.getfixturevalue(resource_fixture)
        mock_llm = AsyncMock(spec=BaseChatModel)
        tool = await _build_tool_for_resource(resource, mock_llm)
        assert_tool_is_base_uipath(tool)
