"""Tests for graph building functionality."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest
from uipath.agent.models.agent import (
    AgentByomProperties,
    AgentMessage,
    AgentMessageRole,
    AgentSettings,
    LowCodeAgentDefinition,
)

from uipath_agents.agent_graph_builder import build_agent_graph
from uipath_agents.agent_graph_builder.config import AgentExecutionType


def create_test_agent_definition(**overrides: Any) -> LowCodeAgentDefinition:
    """Create a test agent definition."""
    defaults: dict[str, Any] = {
        "id": "test-agent",
        "name": "Test Agent",
        "messages": [
            AgentMessage(role=AgentMessageRole.SYSTEM, content="Test system message"),
            AgentMessage(role=AgentMessageRole.USER, content="Test user message"),
        ],
        "settings": AgentSettings(
            engine="azure_openai",
            model="gpt-4",
            temperature=0.7,
            max_tokens=1024,
        ),
        "input_schema": {"type": "object", "properties": {}},
        "output_schema": {"type": "object", "properties": {}},
        "resources": [],
    }
    defaults.update(overrides)
    return LowCodeAgentDefinition(**defaults)


@pytest.mark.asyncio
class TestBuildAgentGraph:
    """Test building the agent graph."""

    @pytest.fixture(autouse=True)
    def mock_mcp_tools(self):
        """Mock create_mcp_tools_from_agent to avoid SDK initialization."""
        with patch(
            "uipath_agents.agent_graph_builder.graph.create_mcp_tools_from_agent",
            new_callable=AsyncMock,
            return_value=([], []),
        ):
            yield

    async def test_builds_graph_with_minimal_config(self):
        """Test that graph is built with minimal configuration."""
        agent_def = create_test_agent_definition()

        with (
            patch(
                "uipath_agents.agent_graph_builder.graph.create_tools_from_resources",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch("uipath_agents.agent_graph_builder.graph.create_llm") as mock_llm,
            patch(
                "uipath_agents.agent_graph_builder.graph.create_agent"
            ) as mock_create,
            patch(
                "uipath_agents.agent_graph_builder.graph.get_thinking_messages_limit",
                return_value=0,
            ),
        ):
            mock_llm.return_value = MagicMock()
            mock_create.return_value = MagicMock()

            graph, disposables = await build_agent_graph(agent_def)

            assert graph is not None
            assert disposables == []
            mock_create.assert_called_once()

    async def test_passes_llm_settings(self):
        """Test that LLM settings are passed correctly."""
        agent_def = create_test_agent_definition(
            settings=AgentSettings(
                engine="azure_openai",
                model="test-model",
                temperature=0.5,
                max_tokens=2048,
                byom_properties=AgentByomProperties(
                    connection_id="test-connection-id",
                    connector_key="test-connector-key",
                ),
            )
        )

        with (
            patch(
                "uipath_agents.agent_graph_builder.graph.create_tools_from_resources",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch("uipath_agents.agent_graph_builder.graph.create_llm") as mock_llm,
            patch(
                "uipath_agents.agent_graph_builder.graph.create_agent"
            ) as mock_create,
            patch(
                "uipath_agents.agent_graph_builder.graph.get_thinking_messages_limit",
                return_value=0,
            ),
        ):
            mock_llm.return_value = MagicMock()
            mock_create.return_value = MagicMock()

            await build_agent_graph(
                agent_def, execution_type=AgentExecutionType.PLAYGROUND
            )

            mock_llm.assert_called_once_with(
                model="test-model",
                temperature=0.5,
                max_tokens=2048,
                execution_type=AgentExecutionType.PLAYGROUND,
                byo_connection_id="test-connection-id",
            )

    async def test_handles_input_data_dict(self):
        """Test building graph with dict input data."""
        agent_def = create_test_agent_definition(
            messages=[
                AgentMessage(role=AgentMessageRole.SYSTEM, content="Process {{task}}"),
                AgentMessage(role=AgentMessageRole.USER, content="Start"),
            ],
            input_schema={
                "type": "object",
                "properties": {"task": {"type": "string"}},
                "required": ["task"],
            },
        )

        with (
            patch(
                "uipath_agents.agent_graph_builder.graph.create_tools_from_resources",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch("uipath_agents.agent_graph_builder.graph.create_llm") as mock_llm,
            patch(
                "uipath_agents.agent_graph_builder.graph.create_agent"
            ) as mock_create,
            patch(
                "uipath_agents.agent_graph_builder.graph.get_thinking_messages_limit",
                return_value=0,
            ),
        ):
            mock_llm.return_value = MagicMock()
            mock_create.return_value = MagicMock()

            await build_agent_graph(agent_def)

            mock_create.assert_called_once()
            message_factory = mock_create.call_args.kwargs["messages"]
            messages = message_factory({"task": "analysis"})
            assert "analysis" in messages[0].content

    async def test_creates_tools_from_resources(self):
        """Test that tools are created from agent resources."""
        agent_def = create_test_agent_definition(
            resources=[
                {
                    "type": "tool",
                    "name": "test_tool",
                    "description": "A test tool",
                    "$resourceType": "custom",
                }
            ]
        )

        mock_tool = MagicMock()
        with (
            patch(
                "uipath_agents.agent_graph_builder.graph.create_tools_from_resources",
                new_callable=AsyncMock,
                return_value=[mock_tool],
            ) as mock_create_tools,
            patch("uipath_agents.agent_graph_builder.graph.create_llm") as mock_llm,
            patch(
                "uipath_agents.agent_graph_builder.graph.create_agent"
            ) as mock_create_agent,
            patch(
                "uipath_agents.agent_graph_builder.graph.get_thinking_messages_limit",
                return_value=0,
            ),
        ):
            mock_llm.return_value = MagicMock()
            mock_create_agent.return_value = MagicMock()

            await build_agent_graph(agent_def)

            mock_create_tools.assert_called_once()
            mock_create_agent.assert_called_once()
            call_kwargs = mock_create_agent.call_args.kwargs
            assert call_kwargs["tools"] == [mock_tool]

    async def test_passes_thinking_messages_limit_via_config(self):
        """Test that thinking_messages_limit is passed via AgentGraphConfig."""
        agent_def = create_test_agent_definition()

        with (
            patch(
                "uipath_agents.agent_graph_builder.graph.create_tools_from_resources",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch("uipath_agents.agent_graph_builder.graph.create_llm") as mock_llm,
            patch(
                "uipath_agents.agent_graph_builder.graph.create_agent"
            ) as mock_create,
            patch(
                "uipath_agents.agent_graph_builder.graph.get_thinking_messages_limit",
                return_value=5,
            ),
        ):
            mock_llm.return_value = MagicMock()
            mock_create.return_value = MagicMock()

            await build_agent_graph(agent_def)

            mock_create.assert_called_once()
            config = mock_create.call_args.kwargs["config"]
            assert config.thinking_messages_limit == 5

    async def test_message_factory_filters_internal_state_fields(self):
        """Test that message_factory only uses input_schema fields, not internal LangGraph state."""
        agent_def = create_test_agent_definition(
            messages=[
                AgentMessage(role=AgentMessageRole.SYSTEM, content="Topic: {{topic}}"),
                AgentMessage(role=AgentMessageRole.USER, content="Start research"),
            ],
            input_schema={
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"],
            },
        )

        with (
            patch(
                "uipath_agents.agent_graph_builder.graph.create_tools_from_resources",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch("uipath_agents.agent_graph_builder.graph.create_llm") as mock_llm,
            patch(
                "uipath_agents.agent_graph_builder.graph.create_agent"
            ) as mock_create,
            patch(
                "uipath_agents.agent_graph_builder.graph.get_thinking_messages_limit",
                return_value=0,
            ),
        ):
            mock_llm.return_value = MagicMock()
            mock_create.return_value = MagicMock()

            await build_agent_graph(agent_def)

            message_factory = mock_create.call_args.kwargs["messages"]

            # Simulate state with user input + internal LangGraph fields
            state_with_internal_fields: dict[str, Any] = {
                "topic": "mount everest",
                "messages": [],  # Should be filtered out
                "termination": None,  # Should be filtered out
            }

            messages = message_factory(state_with_internal_fields)

            assert "mount everest" in messages[0].content
            # Verify that internal state fields did NOT leak into the messages
            assert "[]" not in messages[0].content  # messages field not interpolated
            assert (
                "None" not in messages[0].content
            )  # termination field not interpolated

    async def test_message_factory_with_nested_objects(self):
        """Test that message_factory handles nested object fields with dot notation."""
        agent_def = create_test_agent_definition(
            messages=[
                AgentMessage(
                    role=AgentMessageRole.SYSTEM,
                    content="User: {{user.name}} ({{user.email}}), Role: {{user.role}}",
                ),
                AgentMessage(role=AgentMessageRole.USER, content="Process request"),
            ],
            input_schema={
                "type": "object",
                "properties": {
                    "user": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "email": {"type": "string"},
                            "role": {"type": "string"},
                        },
                    }
                },
                "required": ["user"],
            },
        )

        with (
            patch(
                "uipath_agents.agent_graph_builder.graph.create_tools_from_resources",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch("uipath_agents.agent_graph_builder.graph.create_llm") as mock_llm,
            patch(
                "uipath_agents.agent_graph_builder.graph.create_agent"
            ) as mock_create,
            patch(
                "uipath_agents.agent_graph_builder.graph.get_thinking_messages_limit",
                return_value=0,
            ),
        ):
            mock_llm.return_value = MagicMock()
            mock_create.return_value = MagicMock()

            await build_agent_graph(agent_def)

            message_factory = mock_create.call_args.kwargs["messages"]

            # Simulate state with nested object + internal fields
            state: dict[str, Any] = {
                "user": {
                    "name": "Alice Johnson",
                    "email": "alice@example.com",
                    "role": "admin",
                },
                "messages": [],
                "termination": None,
            }

            messages = message_factory(state)

            # Verify nested fields are interpolated correctly
            assert "Alice Johnson" in messages[0].content
            assert "alice@example.com" in messages[0].content
            assert "admin" in messages[0].content

    async def test_message_factory_with_array_fields(self):
        """Test that message_factory handles array fields correctly."""
        agent_def = create_test_agent_definition(
            messages=[
                AgentMessage(
                    role=AgentMessageRole.SYSTEM,
                    content="Tags: {{tags}}, Count: {{count}}",
                ),
                AgentMessage(role=AgentMessageRole.USER, content="Process items"),
            ],
            input_schema={
                "type": "object",
                "properties": {
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "count": {"type": "integer"},
                },
                "required": ["tags", "count"],
            },
        )

        with (
            patch(
                "uipath_agents.agent_graph_builder.graph.create_tools_from_resources",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch("uipath_agents.agent_graph_builder.graph.create_llm") as mock_llm,
            patch(
                "uipath_agents.agent_graph_builder.graph.create_agent"
            ) as mock_create,
            patch(
                "uipath_agents.agent_graph_builder.graph.get_thinking_messages_limit",
                return_value=0,
            ),
        ):
            mock_llm.return_value = MagicMock()
            mock_create.return_value = MagicMock()

            await build_agent_graph(agent_def)

            message_factory = mock_create.call_args.kwargs["messages"]

            # State with array field
            state = {
                "tags": ["python", "async", "testing"],
                "count": 42,
                "messages": [],
            }

            messages = message_factory(state)

            # Verify array is serialized as JSON
            assert '["python", "async", "testing"]' in messages[0].content
            assert "42" in messages[0].content

    async def test_message_factory_with_complex_nested_structure(self):
        """Test message_factory with deeply nested objects and arrays."""
        agent_def = create_test_agent_definition(
            messages=[
                AgentMessage(
                    role=AgentMessageRole.SYSTEM,
                    content="Project: {{project.name}}, Owner: {{project.owner.name}}, Teams: {{project.teams}}",
                ),
                AgentMessage(role=AgentMessageRole.USER, content="Analyze project"),
            ],
            input_schema={
                "type": "object",
                "properties": {
                    "project": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "owner": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "id": {"type": "integer"},
                                },
                            },
                            "teams": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                    }
                },
                "required": ["project"],
            },
        )

        with (
            patch(
                "uipath_agents.agent_graph_builder.graph.create_tools_from_resources",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch("uipath_agents.agent_graph_builder.graph.create_llm") as mock_llm,
            patch(
                "uipath_agents.agent_graph_builder.graph.create_agent"
            ) as mock_create,
            patch(
                "uipath_agents.agent_graph_builder.graph.get_thinking_messages_limit",
                return_value=0,
            ),
        ):
            mock_llm.return_value = MagicMock()
            mock_create.return_value = MagicMock()

            await build_agent_graph(agent_def)

            message_factory = mock_create.call_args.kwargs["messages"]

            # Simulate state with deeply nested structure + internal fields
            state: dict[str, Any] = {
                "project": {
                    "name": "Agent Framework",
                    "owner": {"name": "Engineering Team", "id": 101},
                    "teams": ["backend", "frontend", "qa"],
                },
                "messages": [],
                "termination": None,
                "agent_outcome": {"result": "pending"},
            }

            messages = message_factory(state)

            # Verify complex nested interpolation works
            assert "Agent Framework" in messages[0].content
            assert "Engineering Team" in messages[0].content
            assert '["backend", "frontend", "qa"]' in messages[0].content
            # Verify internal state is NOT leaked
            assert "pending" not in messages[0].content

    async def test_message_factory_with_boolean_and_object_values(self):
        """Test message_factory handles boolean and object values correctly."""
        agent_def = create_test_agent_definition(
            messages=[
                AgentMessage(
                    role=AgentMessageRole.SYSTEM,
                    content="Active: {{is_active}}, Config: {{config}}, Count: {{count}}",
                ),
                AgentMessage(role=AgentMessageRole.USER, content="Check status"),
            ],
            input_schema={
                "type": "object",
                "properties": {
                    "is_active": {"type": "boolean"},
                    "config": {
                        "type": "object",
                        "properties": {
                            "timeout": {"type": "integer"},
                            "retry": {"type": "boolean"},
                        },
                    },
                    "count": {"type": "number"},
                },
                "required": ["is_active", "config"],
            },
        )

        with (
            patch(
                "uipath_agents.agent_graph_builder.graph.create_tools_from_resources",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch("uipath_agents.agent_graph_builder.graph.create_llm") as mock_llm,
            patch(
                "uipath_agents.agent_graph_builder.graph.create_agent"
            ) as mock_create,
            patch(
                "uipath_agents.agent_graph_builder.graph.get_thinking_messages_limit",
                return_value=0,
            ),
        ):
            mock_llm.return_value = MagicMock()
            mock_create.return_value = MagicMock()

            await build_agent_graph(agent_def)

            message_factory = mock_create.call_args.kwargs["messages"]

            # State with boolean, object values, and optional field
            state = {
                "is_active": True,
                "config": {"timeout": 30, "retry": False},
                "count": 3.14,
                "messages": [],
            }

            messages = message_factory(state)

            # Verify boolean is serialized as JSON
            assert "true" in messages[0].content
            # Verify object is serialized as JSON
            assert '"timeout": 30' in messages[0].content
            assert '"retry": false' in messages[0].content
            # Verify number is included
            assert "3.14" in messages[0].content

    async def test_message_factory_with_array_of_objects(self):
        """Test message_factory handles arrays of objects correctly."""
        agent_def = create_test_agent_definition(
            messages=[
                AgentMessage(
                    role=AgentMessageRole.SYSTEM,
                    content="Users: {{users}}",
                ),
                AgentMessage(role=AgentMessageRole.USER, content="Process users"),
            ],
            input_schema={
                "type": "object",
                "properties": {
                    "users": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer"},
                                "name": {"type": "string"},
                                "active": {"type": "boolean"},
                            },
                        },
                    }
                },
                "required": ["users"],
            },
        )

        with (
            patch(
                "uipath_agents.agent_graph_builder.graph.create_tools_from_resources",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch("uipath_agents.agent_graph_builder.graph.create_llm") as mock_llm,
            patch(
                "uipath_agents.agent_graph_builder.graph.create_agent"
            ) as mock_create,
            patch(
                "uipath_agents.agent_graph_builder.graph.get_thinking_messages_limit",
                return_value=0,
            ),
        ):
            mock_llm.return_value = MagicMock()
            mock_create.return_value = MagicMock()

            await build_agent_graph(agent_def)

            message_factory = mock_create.call_args.kwargs["messages"]

            # State with array of objects
            state = {
                "users": [
                    {"id": 1, "name": "Alice", "active": True},
                    {"id": 2, "name": "Bob", "active": False},
                ],
                "messages": [],
            }

            messages = message_factory(state)

            # Verify array of objects is serialized as JSON
            assert '"id": 1' in messages[0].content
            assert '"name": "Alice"' in messages[0].content
            assert '"active": true' in messages[0].content
            assert '"name": "Bob"' in messages[0].content
            assert '"active": false' in messages[0].content

    async def test_passes_is_conversational_false_by_default(self):
        """Test that is_conversational=False is passed via AgentGraphConfig by default."""
        agent_def = create_test_agent_definition()

        with (
            patch(
                "uipath_agents.agent_graph_builder.graph.create_tools_from_resources",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch("uipath_agents.agent_graph_builder.graph.create_llm") as mock_llm,
            patch(
                "uipath_agents.agent_graph_builder.graph.create_agent"
            ) as mock_create,
            patch(
                "uipath_agents.agent_graph_builder.graph.get_thinking_messages_limit",
                return_value=0,
            ),
        ):
            mock_llm.return_value = MagicMock()
            mock_create.return_value = MagicMock()

            await build_agent_graph(agent_def)

            mock_create.assert_called_once()
            config = mock_create.call_args.kwargs["config"]
            assert config.is_conversational is False

    async def test_passes_is_conversational_true_for_conversational_agent(self):
        """Test that is_conversational=True is passed via AgentGraphConfig for conversational agents."""
        agent_def = create_test_agent_definition()

        with (
            patch(
                "uipath_agents.agent_graph_builder.graph.create_tools_from_resources",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch("uipath_agents.agent_graph_builder.graph.create_llm") as mock_llm,
            patch(
                "uipath_agents.agent_graph_builder.graph.create_agent"
            ) as mock_create,
            patch(
                "uipath_agents.agent_graph_builder.graph.get_thinking_messages_limit",
                return_value=0,
            ),
            patch.object(
                type(agent_def),
                "is_conversational",
                new_callable=PropertyMock,
                return_value=True,
            ),
        ):
            mock_llm.return_value = MagicMock()
            mock_create.return_value = MagicMock()

            await build_agent_graph(agent_def)

            mock_create.assert_called_once()
            config = mock_create.call_args.kwargs["config"]
            assert config.is_conversational is True

    async def test_conversational_agent_uses_conversational_message_factory(self):
        """Test that conversational agents use the conversational message factory."""
        agent_def = create_test_agent_definition(
            input_schema={
                "type": "object",
                "properties": {
                    "userSettings": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                    }
                },
            },
        )

        with (
            patch(
                "uipath_agents.agent_graph_builder.graph.create_tools_from_resources",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch("uipath_agents.agent_graph_builder.graph.create_llm") as mock_llm,
            patch(
                "uipath_agents.agent_graph_builder.graph.create_agent"
            ) as mock_create,
            patch(
                "uipath_agents.agent_graph_builder.graph.get_thinking_messages_limit",
                return_value=0,
            ),
            patch(
                "uipath_agents.agent_graph_builder.message_utils.get_chat_system_prompt",
                return_value="Conversational system prompt",
            ),
            patch.object(
                type(agent_def),
                "is_conversational",
                new_callable=PropertyMock,
                return_value=True,
            ),
        ):
            mock_llm.return_value = MagicMock()
            mock_create.return_value = MagicMock()

            await build_agent_graph(agent_def)

            mock_create.assert_called_once()
            message_factory = mock_create.call_args.kwargs["messages"]

            # Call the factory - for conversational agents it should return only SystemMessage
            messages = message_factory({"userSettings": {"name": "Test User"}})

            # Conversational message factory returns only a SystemMessage
            # (it relies on init_node to manage user messages)
            assert len(messages) == 1
            from langchain_core.messages import SystemMessage

            assert isinstance(messages[0], SystemMessage)
