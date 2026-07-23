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
    AgentQuickFormChannelProperties,
    AgentQuickFormEscalationChannel,
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
def flow_resource() -> AgentProcessToolResourceConfig:
    """Create a process tool resource config of type Flow."""
    return AgentProcessToolResourceConfig(
        type=AgentToolType.FLOW,
        name="test_flow",
        description="Test flow description",
        input_schema=EMPTY_SCHEMA,
        output_schema=EMPTY_SCHEMA,
        properties=AgentProcessToolProperties(
            process_name="MyFlow",
            folder_path="/Shared/Flows",
        ),
    )


@pytest.fixture
def function_resource() -> AgentProcessToolResourceConfig:
    """Create a process tool resource config of type Function."""
    return AgentProcessToolResourceConfig(
        type=AgentToolType.FUNCTION,
        name="test_function",
        description="Test function description",
        input_schema=EMPTY_SCHEMA,
        output_schema=EMPTY_SCHEMA,
        properties=AgentProcessToolProperties(
            process_name="MyFunction",
            folder_path="/Shared/Functions",
        ),
    )


@pytest.fixture
def context_resource() -> AgentContextResourceConfig:
    """Create a context tool resource config."""
    return AgentContextResourceConfig(
        resource_type=AgentResourceType.CONTEXT,
        context_type="index",
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
                type="actionCenter",
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
def quick_form_resource() -> AgentEscalationResourceConfig:
    """Create a quick-form escalation tool resource config."""
    return AgentEscalationResourceConfig(
        resource_type=AgentResourceType.ESCALATION,
        name="test_quick_form_escalation",
        description="Test quick-form escalation description",
        channels=[
            AgentQuickFormEscalationChannel(
                name="test_channel",
                type="actionCenterQuickForm",
                description="Test quick-form channel",
                task_title="Test Task",
                input_schema=EMPTY_SCHEMA,
                output_schema=EMPTY_SCHEMA,
                properties=AgentQuickFormChannelProperties(
                    form_schema={
                        "schemaId": "00000000-0000-0000-0000-000000000abc",
                        "fields": [{"name": "decision", "type": "string"}],
                        "outcomes": ["approve", "reject"],
                    },
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
            "flow_resource",
            "function_resource",
            "context_resource",
            "escalation_resource",
            "quick_form_resource",
            "integration_resource",
            "internal_resource",
            "ixp_extraction_resource",
            # Note: mcp_resource is excluded because MCP tools are created
            # separately via create_mcp_tools_and_clients, not through
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

    async def test_flow_resource_routes_through_process_tool_path(
        self, flow_resource, mock_uipath_sdk
    ):
        """A Flow-type resource is dispatched via the process_tool factory path."""
        mock_llm = AsyncMock(spec=BaseChatModel)
        with patch(
            "uipath_langchain.agent.tools.tool_factory.create_process_tool"
        ) as mock_create_process_tool:
            mock_create_process_tool.return_value = MagicMock(
                spec=BaseUiPathStructuredTool
            )
            tool = await _build_tool_for_resource(flow_resource, mock_llm)

        mock_create_process_tool.assert_called_once_with(
            flow_resource, conversational_run_as_me=False
        )
        assert tool is not None

    async def test_quick_form_resource_routes_through_escalation_tool_path(
        self, quick_form_resource, mock_uipath_sdk
    ):
        """A QuickForm escalation resource is dispatched via create_escalation_tool."""
        mock_llm = AsyncMock(spec=BaseChatModel)
        with patch(
            "uipath_langchain.agent.tools.tool_factory.create_escalation_tool"
        ) as mock_create_escalation_tool:
            mock_create_escalation_tool.return_value = MagicMock(
                spec=BaseUiPathStructuredTool
            )
            tool = await _build_tool_for_resource(quick_form_resource, mock_llm)

        mock_create_escalation_tool.assert_called_once_with(
            quick_form_resource, agent=None
        )
        assert tool is not None

    async def test_function_resource_routes_through_process_tool_path(
        self, function_resource, mock_uipath_sdk
    ):
        """A Function-type resource is dispatched via the process_tool factory path."""
        mock_llm = AsyncMock(spec=BaseChatModel)
        with patch(
            "uipath_langchain.agent.tools.tool_factory.create_process_tool"
        ) as mock_create_process_tool:
            mock_create_process_tool.return_value = MagicMock(
                spec=BaseUiPathStructuredTool
            )
            tool = await _build_tool_for_resource(function_resource, mock_llm)

        mock_create_process_tool.assert_called_once_with(
            function_resource, conversational_run_as_me=False
        )
        assert tool is not None

    async def test_conversational_run_as_me_forwarded_to_process_tool(
        self, process_resource, mock_uipath_sdk
    ):
        """The dispatcher forwards conversational_run_as_me unchanged.

        The per-type suppression rule (RPA / PROCESS ignores RunAsMe) lives
        inside ``create_process_tool``; see ``test_process_tool.py``.
        """
        process_resource.is_enabled = True
        agent = LowCodeAgentDefinition(
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "object", "properties": {}},
            messages=[],
            settings=Mock(spec=AgentSettings),
            resources=[process_resource],
        )
        mock_llm = AsyncMock(spec=BaseChatModel)
        with patch(
            "uipath_langchain.agent.tools.tool_factory.create_process_tool"
        ) as mock_create_process_tool:
            mock_create_process_tool.return_value = MagicMock(
                spec=BaseUiPathStructuredTool
            )
            await create_tools_from_resources(
                agent, mock_llm, conversational_run_as_me=True
            )

        mock_create_process_tool.assert_called_once_with(
            process_resource, conversational_run_as_me=True
        )

    async def test_conversational_run_as_me_defaults_false_when_not_provided(
        self, process_resource, mock_uipath_sdk
    ):
        """conversational_run_as_me defaults to False when omitted (non-conversational agents)."""
        process_resource.is_enabled = True
        agent = LowCodeAgentDefinition(
            input_schema={"type": "object", "properties": {}},
            output_schema={"type": "object", "properties": {}},
            messages=[],
            settings=Mock(spec=AgentSettings),
            resources=[process_resource],
        )
        mock_llm = AsyncMock(spec=BaseChatModel)
        with patch(
            "uipath_langchain.agent.tools.tool_factory.create_process_tool"
        ) as mock_create_process_tool:
            mock_create_process_tool.return_value = MagicMock(
                spec=BaseUiPathStructuredTool
            )
            await create_tools_from_resources(agent, mock_llm)

        mock_create_process_tool.assert_called_once_with(
            process_resource, conversational_run_as_me=False
        )
