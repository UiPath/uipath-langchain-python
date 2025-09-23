import copy
import json

from pydantic import TypeAdapter
from uipath.agent.models.agent import AgentDefinition

from tests.tools.conftest import (
    get_file_path,
    uipath_connection_mock,
    uipath_integration_mock,
    uipath_interrupt_mock,
    uipath_token_mock,
)
from uipath_langchain.tools.preconfigured import safe_extract_tools


class TestSafeExtractTools:
    """Test class for tool extraction."""

    async def test_extraction(self, monkeypatch, httpx_mock, mocker) -> None:
        # Arrange
        monkeypatch.setenv("UIPATH_URL", "https://example.com/")
        monkeypatch.setenv("UIPATH_ACCESS_TOKEN", "1234567890")

        agent_definition = get_file_path("agent_definition.json")
        with open(agent_definition) as agent_definition_file:
            agent_definition_json = json.load(agent_definition_file)
            agent_definition = TypeAdapter(AgentDefinition).validate_python(
                agent_definition_json
            )
        tool_response = {"foo": "bar"}

        # Act
        tools = safe_extract_tools(agent_definition)

        # Assert tools and schema definitions
        assert len(tools) == 3
        assert tools[0].input_schema.model_json_schema().get("properties") == {
            "query": {"description": "Query.", "title": "Query", "type": "string"}
        }
        assert tools[0].OutputType.model_json_schema().get("properties") == {}
        assert tools[1].input_schema.model_json_schema().get("properties") == {
            "name": {"description": "Name.", "title": "Name", "type": "string"}
        }
        assert tools[1].OutputType.model_json_schema().get("properties") == {
            "status": {
                "description": "Escalation status.",
                "title": "Status",
                "type": "string",
            }
        }
        assert tools[2].input_schema.model_json_schema().get("properties") == {
            "name": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
                "default": None,
                "title": "Name",
            }
        }
        assert tools[2].OutputType.model_json_schema().get("properties") == {
            "content": {"description": "Output content", "title": "Content"}
        }

        # Invoke Integration
        mock_connection = copy.deepcopy(
            agent_definition_json["resources"][0]["properties"]["connection"]
        )
        mock_connection["element_instance_id"] = 1  # Updated instance id.

        with (
            uipath_connection_mock(httpx_mock, response=mock_connection),
            uipath_token_mock(httpx_mock, response={"accessToken": "accessToken"}),
            uipath_integration_mock(httpx_mock, tool_response),
        ):
            # Act
            response = await tools[0].ainvoke(input={"query": "query"})

            # Assert
            assert response == tool_response

        # Invoke Escalation
        with uipath_interrupt_mock(mocker, tool_response):
            # Act
            response = await tools[1].ainvoke(input={"name": "name"})

            # Assert
            assert response == tool_response

        # Invoke Escalation
        with uipath_interrupt_mock(mocker, tool_response):
            # Act
            response = await tools[2].ainvoke(input={"name": "name"})

            # Assert
            assert response == tool_response
