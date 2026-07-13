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
    AgentInternalDeepRagSettings,
    AgentInternalDeepRagToolProperties,
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
    CitationMode,
    DeepRagCitationModeSetting,
    DeepRagFileExtension,
    DeepRagFileExtensionSetting,
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
def deeprag_resource() -> AgentInternalToolResourceConfig:
    """Create a DeepRAG internal tool resource config."""
    return AgentInternalToolResourceConfig(
        type=AgentToolType.INTERNAL,
        name="test_deeprag",
        description="Test DeepRAG description",
        input_schema=EMPTY_SCHEMA,
        output_schema=EMPTY_SCHEMA,
        properties=AgentInternalDeepRagToolProperties(
            tool_type=AgentInternalToolType.DEEP_RAG,
            settings=AgentInternalDeepRagSettings(
                context_type="index",
                query=AgentContextQuerySetting(
                    variant="static",
                    value="test query",
                    description="test description",
                ),
                citation_mode=DeepRagCitationModeSetting(value=CitationMode.INLINE),
                file_extension=DeepRagFileExtensionSetting(
                    value=DeepRagFileExtension.PDF
                ),
            ),
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


# A tool output schema with a dangling $ref (no matching $defs entry) -- the shape
# Studio Web emits for a .NET decimal? output argument. See create_output_model.
_MALFORMED_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "left": {"$ref": "#/$defs/Nullableofdecimal", "type": "object"},
    },
}


def _set_malformed_output_schema(resource):
    """Put a malformed output schema where the given resource type reads it.

    Escalation tools take the output schema from their channel(s); every other
    tool type reads ``resource.output_schema`` directly.
    """
    channels = getattr(resource, "channels", None)
    if channels:
        for channel in channels:
            channel.output_schema = _MALFORMED_OUTPUT_SCHEMA
    else:
        resource.output_schema = _MALFORMED_OUTPUT_SCHEMA


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

        mock_create_process_tool.assert_called_once_with(flow_resource, run_as_me=False)
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
            function_resource, run_as_me=False
        )
        assert tool is not None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "resource_fixture",
        [
            "process_resource",
            "escalation_resource",
            "integration_resource",
            "internal_resource",  # analyze_files
            "deeprag_resource",
        ],
    )
    async def test_malformed_output_schema_is_non_blocking(
        self, resource_fixture, mock_uipath_sdk, request
    ):
        """Each of the 5 tool factories that convert an output schema must tolerate
        a malformed one instead of failing agent startup.

        Output schemas are design-time only (guardrails + eval simulation) and are
        never used during execution, so a dangling ``$ref`` (e.g. a .NET
        ``Nullableofdecimal`` with no ``$defs``) must not raise
        ``AGENT_STARTUP.INVALID_TOOL_CONFIG``. The tool is still built, with its
        output model degraded to an empty schema.
        """
        resource = request.getfixturevalue(resource_fixture)
        _set_malformed_output_schema(resource)

        # Must not raise (this previously threw AGENT_STARTUP.INVALID_TOOL_CONFIG).
        tool = await _build_tool_for_resource(resource, AsyncMock(spec=BaseChatModel))

        assert_tool_is_base_uipath(tool)
        # Output model degraded to an empty schema rather than crashing startup.
        assert tool.output_type.model_json_schema().get("properties", {}) == {}
