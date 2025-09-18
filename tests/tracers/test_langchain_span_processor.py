"""Tests for LangchainSpanProcessor."""

import json

import pytest

from uipath_langchain.tracers.LangchainSpanProcessor import (
    LangchainSpanProcessor,
    unflatten_dict,
)


class TestUnflattenDict:
    """Test the unflatten_dict utility function."""

    def test_simple_unflatten(self):
        """Test basic unflattening functionality."""
        flat_dict = {"user.name": "John", "user.age": 30, "settings.theme": "dark"}

        result = unflatten_dict(flat_dict)

        assert result == {
            "user": {"name": "John", "age": 30},
            "settings": {"theme": "dark"},
        }

    def test_array_unflatten(self):
        """Test unflattening with array indices."""
        flat_dict = {
            "items.0.name": "first",
            "items.0.value": 1,
            "items.1.name": "second",
            "items.1.value": 2,
        }

        result = unflatten_dict(flat_dict)

        expected = {
            "items": [{"name": "first", "value": 1}, {"name": "second", "value": 2}]
        }
        assert result == expected

    def test_nested_arrays(self):
        """Test deeply nested structures with arrays."""
        flat_dict = {
            "llm.messages.0.content": "hello",
            "llm.messages.0.tools.0.name": "tool1",
            "llm.messages.0.tools.1.name": "tool2",
            "llm.provider": "azure",
        }

        result = unflatten_dict(flat_dict)

        expected = {
            "llm": {
                "messages": [
                    {
                        "content": "hello",
                        "tools": [{"name": "tool1"}, {"name": "tool2"}],
                    }
                ],
                "provider": "azure",
            }
        }
        assert result == expected

    def test_sparse_arrays(self):
        """Test arrays with gaps in indices."""
        flat_dict = {"items.0.name": "first", "items.2.name": "third"}

        result = unflatten_dict(flat_dict)

        expected = {"items": [{"name": "first"}, None, {"name": "third"}]}
        assert result == expected

    def test_empty_dict(self):
        """Test with empty dictionary."""
        result = unflatten_dict({})
        assert result == {}

    def test_single_level_keys(self):
        """Test with keys that don't need unflattening."""
        flat_dict = {"name": "value", "number": 42}
        result = unflatten_dict(flat_dict)
        assert result == flat_dict


class TestLangchainSpanProcessor:
    """Test the LangchainSpanProcessor class."""

    def test_init_defaults(self):
        """Test initialization with default parameters."""
        processor = LangchainSpanProcessor()
        assert processor._dump_attributes_as_string is True
        assert processor._unflatten_attributes is True

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        processor = LangchainSpanProcessor(
            dump_attributes_as_string=False, unflatten_attributes=True
        )
        assert processor._dump_attributes_as_string is False
        assert processor._unflatten_attributes is True

    def test_process_span_without_attributes(self):
        """Test processing span without attributes."""
        processor = LangchainSpanProcessor()
        span_data = {"Id": "test-id", "Name": "TestSpan"}

        result = processor.process_span(span_data)
        assert result == span_data

    def test_process_span_with_unflatten_disabled(self):
        """Test processing span with unflattening disabled."""
        processor = LangchainSpanProcessor(
            dump_attributes_as_string=False, unflatten_attributes=False
        )

        attributes = {
            "llm.output_messages.0.role": "assistant",
            "llm.provider": "azure",
            "model": "gpt-4",
        }

        span_data = {"Id": "test-id", "Attributes": json.dumps(attributes)}

        result = processor.process_span(span_data)

        # Should keep flattened structure
        assert result["attributes"]["llm.output_messages.0.role"] == "assistant"
        assert result["attributes"]["llm.provider"] == "azure"
        assert result["attributes"]["model"] == "gpt-4"

    def test_process_span_with_unflatten_enabled(self):
        """Test processing span with unflattening enabled."""
        processor = LangchainSpanProcessor(
            dump_attributes_as_string=False, unflatten_attributes=True
        )

        attributes = {
            "llm.output_messages.0.message.role": "assistant",
            "llm.output_messages.0.message.content": "Hello",
            "llm.output_messages.0.message.tool_calls.0.function.name": "get_time",
            "llm.provider": "azure",
            "model": "gpt-4",
        }

        span_data = {"Id": "test-id", "Attributes": json.dumps(attributes)}

        result = processor.process_span(span_data)

        # Should have nested structure
        attrs = result["attributes"]
        assert attrs["llm"]["output_messages"][0]["message"]["role"] == "assistant"
        assert attrs["llm"]["output_messages"][0]["message"]["content"] == "Hello"
        assert (
            attrs["llm"]["output_messages"][0]["message"]["tool_calls"][0]["function"][
                "name"
            ]
            == "get_time"
        )
        assert attrs["llm"]["provider"] == "azure"
        assert attrs["model"] == "gpt-4"

    def test_process_span_with_unflatten_and_json_output(self):
        """Test processing span with unflattening and JSON string output."""
        processor = LangchainSpanProcessor(
            dump_attributes_as_string=True, unflatten_attributes=True
        )

        attributes = {"llm.provider": "azure", "llm.messages.0.role": "user"}

        span_data = {"Id": "test-id", "Attributes": json.dumps(attributes)}

        result = processor.process_span(span_data)

        # Should be JSON string
        assert isinstance(result["attributes"], str)

        # Parse and verify nested structure
        parsed = json.loads(result["attributes"])
        assert parsed["llm"]["provider"] == "azure"
        assert parsed["llm"]["messages"][0]["role"] == "user"

    def test_attribute_mapping_with_unflatten(self):
        """Test that attribute mapping works with unflattening."""
        processor = LangchainSpanProcessor(
            dump_attributes_as_string=False, unflatten_attributes=True
        )

        attributes = {
            "llm.model_name": "gpt-4",  # Should be mapped to "model"
            "llm.output_messages.0.role": "assistant",
            "input.value": '{"text": "hello"}',  # Should be parsed
        }

        span_data = {"Id": "test-id", "Attributes": json.dumps(attributes)}

        result = processor.process_span(span_data)
        attrs = result["attributes"]

        # Check mapping worked
        assert attrs["model"] == "gpt-4"
        assert attrs["input"] == {"text": "hello"}

        # Check unflattening worked
        assert attrs["llm"]["output_messages"][0]["role"] == "assistant"

    def test_token_usage_processing_with_unflatten(self):
        """Test token usage processing with unflattening."""
        processor = LangchainSpanProcessor(
            dump_attributes_as_string=False, unflatten_attributes=True
        )

        attributes = {
            "llm.token_count.prompt": 100,
            "llm.token_count.completion": 50,
            "llm.token_count.total": 150,
            "llm.provider": "azure",
        }

        span_data = {"Id": "test-id", "Attributes": json.dumps(attributes)}

        result = processor.process_span(span_data)
        attrs = result["attributes"]

        # Check usage structure
        assert attrs["usage"]["promptTokens"] == 100
        assert attrs["usage"]["completionTokens"] == 50
        assert attrs["usage"]["totalTokens"] == 150
        assert attrs["usage"]["isByoExecution"] is False

        # Check unflattening of other attributes
        assert attrs["llm"]["provider"] == "azure"

    def test_unflatten_error_handling(self):
        """Test error handling in unflattening."""
        processor = LangchainSpanProcessor(
            dump_attributes_as_string=False, unflatten_attributes=True
        )

        # Create a scenario that might cause unflattening issues
        # This should be handled gracefully
        attributes = {"normal.key": "value", "llm.provider": "azure"}

        span_data = {"Id": "test-id", "Attributes": json.dumps(attributes)}

        # Should not raise an exception
        result = processor.process_span(span_data)
        assert "attributes" in result

    def test_process_span_with_dict_attributes_unflatten_enabled(self):
        """Test processing span with dictionary attributes and unflattening enabled."""
        processor = LangchainSpanProcessor(
            dump_attributes_as_string=False, unflatten_attributes=True
        )

        # Simulate the real-world case where Attributes is already a dictionary
        attributes = {
            "llm.output_messages.0.message.role": "assistant",
            "llm.output_messages.0.message.tool_calls.0.tool_call.id": "call_123",
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "get_time",
            "llm.provider": "azure",
            "model": "gpt-4",
        }

        span_data = {
            "Id": "test-id",
            "Attributes": attributes,  # Already a dictionary, not a JSON string
        }

        result = processor.process_span(span_data)

        # Should have nested structure
        attrs = result["attributes"]
        assert attrs["llm"]["output_messages"][0]["message"]["role"] == "assistant"
        assert (
            attrs["llm"]["output_messages"][0]["message"]["tool_calls"][0]["tool_call"][
                "id"
            ]
            == "call_123"
        )
        assert (
            attrs["llm"]["output_messages"][0]["message"]["tool_calls"][0]["tool_call"][
                "function"
            ]["name"]
            == "get_time"
        )
        assert attrs["llm"]["provider"] == "azure"
        assert attrs["model"] == "gpt-4"

    def test_real_world_trace_unflatten(self):
        """Test with real-world trace data to verify unflattening works correctly."""
        processor = LangchainSpanProcessor(
            dump_attributes_as_string=False, unflatten_attributes=True
        )

        # Real trace data from user's example (dictionary format)
        real_trace_attributes = {
            "input.mime_type": "application/json",
            "output.mime_type": "application/json",
            "llm.input_messages.0.message.role": "user",
            "llm.input_messages.0.message.content": "You are a helpful assistant with access to various tools. \n    The user is asking about: Weather and Technology\n    \n    Please use the available tools to gather some relevant information. For example:\n    - Check the current time\n    - Generate a random number if relevant\n    - Calculate squares of numbers if needed\n    - Get weather information for any cities mentioned\n    \n    Use at least 2-3 tools to demonstrate their functionality.",
            "llm.output_messages.0.message.role": "assistant",
            "llm.output_messages.0.message.tool_calls.0.tool_call.id": "call_qWaFnNRY8mk2PQjEu0wRLaRd",
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "get_current_time",
            "llm.output_messages.0.message.tool_calls.1.tool_call.id": "call_3ckaPILSv4SmyeufQf1ovA3H",
            "llm.output_messages.0.message.tool_calls.1.tool_call.function.name": "generate_random_number",
            "llm.output_messages.0.message.tool_calls.1.tool_call.function.arguments": '{"min_val": 1, "max_val": 10}',
            "llm.output_messages.0.message.tool_calls.2.tool_call.id": "call_BjaiJ0NHwWs14fMbCyjDElEX",
            "llm.output_messages.0.message.tool_calls.2.tool_call.function.name": "get_weather_info",
            "llm.output_messages.0.message.tool_calls.2.tool_call.function.arguments": '{"city": "San Francisco"}',
            "llm.invocation_parameters": '{"model": "gpt-4o-mini-2024-07-18", "url": "https://alpha.uipath.com/..."}',
            "llm.tools.0.tool.json_schema": '{"type": "function", "function": {"name": "get_current_time", "description": "Get the current date and time.", "parameters": {"properties": {}, "type": "object"}}}',
            "llm.tools.1.tool.json_schema": '{"type": "function", "function": {"name": "generate_random_number", "description": "Generate a random number between min_val and max_val (inclusive).", "parameters": {"properties": {"min_val": {"default": 1, "type": "integer"}, "max_val": {"default": 100, "type": "integer"}}, "type": "object"}}}',
            "llm.tools.2.tool.json_schema": '{"type": "function", "function": {"name": "calculate_square", "description": "Calculate the square of a given number.", "parameters": {"properties": {"number": {"type": "number"}}, "required": ["number"], "type": "object"}}}',
            "llm.tools.3.tool.json_schema": '{"type": "function", "function": {"name": "get_weather_info", "description": "Get mock weather information for a given city.", "parameters": {"properties": {"city": {"type": "string"}}, "required": ["city"], "type": "object"}}}',
            "llm.provider": "azure",
            "llm.system": "openai",
            "session.id": "a879985a-8d39-4f51-94e1-8433423f35db",
            "metadata": '{"thread_id": "a879985a-8d39-4f51-94e1-8433423f35db", "langgraph_step": 1, "langgraph_node": "make_tool_calls"}',
            "model": "gpt-4o-mini-2024-07-18",
            "usage": {
                "promptTokens": 219,
                "completionTokens": 66,
                "totalTokens": 285,
                "isByoExecution": False,
            },
        }

        span_data = {
            "PermissionStatus": 0,
            "Id": "7d137190-348c-4ef2-9b19-165295643b82",
            "TraceId": "81dbeaf2-c2ba-4b1e-95fd-b722f53dc405",
            "ParentId": "f71478d6-f081-4bf6-a942-0944d97ffadb",
            "Name": "UiPathChat",
            "StartTime": "2025-08-26T16:11:17.276Z",
            "EndTime": "2025-08-26T16:11:20.027Z",
            "Attributes": real_trace_attributes,  # Dictionary format (not JSON string)
            "SpanType": "completion",
        }

        # Process the span
        result = processor.process_span(span_data)

        # Verify the trace data structure is preserved
        assert result["Id"] == "7d137190-348c-4ef2-9b19-165295643b82"
        assert result["Name"] == "UiPathChat"
        assert result["SpanType"] == "completion"

        # Verify attributes are unflattened and accessible
        attrs = result["attributes"]
        assert isinstance(attrs, dict)

        # Test LLM provider info
        assert attrs["llm"]["provider"] == "azure"
        assert attrs["llm"]["system"] == "openai"

        # Test input messages
        input_messages = attrs["llm"]["input_messages"]
        assert len(input_messages) == 1
        assert input_messages[0]["message"]["role"] == "user"
        assert "helpful assistant" in input_messages[0]["message"]["content"]

        # Test output messages and tool calls
        output_messages = attrs["llm"]["output_messages"]
        assert len(output_messages) == 1
        assert output_messages[0]["message"]["role"] == "assistant"

        tool_calls = output_messages[0]["message"]["tool_calls"]
        assert len(tool_calls) == 3

        # Verify individual tool calls
        assert tool_calls[0]["tool_call"]["function"]["name"] == "get_current_time"
        assert tool_calls[0]["tool_call"]["id"] == "call_qWaFnNRY8mk2PQjEu0wRLaRd"

        assert (
            tool_calls[1]["tool_call"]["function"]["name"] == "generate_random_number"
        )
        assert (
            tool_calls[1]["tool_call"]["function"]["arguments"]
            == '{"min_val": 1, "max_val": 10}'
        )

        assert tool_calls[2]["tool_call"]["function"]["name"] == "get_weather_info"
        assert (
            tool_calls[2]["tool_call"]["function"]["arguments"]
            == '{"city": "San Francisco"}'
        )

        # Test tools schema
        tools = attrs["llm"]["tools"]
        assert len(tools) == 4

        # Parse and verify tool schemas
        tool_0_schema = json.loads(tools[0]["tool"]["json_schema"])
        assert tool_0_schema["function"]["name"] == "get_current_time"

        tool_1_schema = json.loads(tools[1]["tool"]["json_schema"])
        assert tool_1_schema["function"]["name"] == "generate_random_number"

        # Test session data
        assert attrs["session"]["id"] == "a879985a-8d39-4f51-94e1-8433423f35db"

        # Test metadata
        metadata = json.loads(attrs["metadata"])
        assert metadata["thread_id"] == "a879985a-8d39-4f51-94e1-8433423f35db"
        assert metadata["langgraph_step"] == 1
        assert metadata["langgraph_node"] == "make_tool_calls"

        # Test model and usage info
        assert attrs["model"] == "gpt-4o-mini-2024-07-18"
        assert attrs["usage"]["promptTokens"] == 219
        assert attrs["usage"]["completionTokens"] == 66
        assert attrs["usage"]["totalTokens"] == 285
        assert attrs["usage"]["isByoExecution"] is False

        # Test MIME types
        assert attrs["input"]["mime_type"] == "application/json"
        assert attrs["output"]["mime_type"] == "application/json"

        print("✅ Real-world trace unflattening test passed!")
        print(f"   - Processed {len(real_trace_attributes)} flattened attributes")
        print(f"   - Created nested structure with {len(tool_calls)} tool calls")
        print(f"   - Verified {len(tools)} tool schemas")
        print("   - All nested access patterns work correctly")

    def test_invalid_json_attributes(self):
        """Test handling of invalid JSON in attributes."""
        processor = LangchainSpanProcessor(unflatten_attributes=True)

        span_data = {"Id": "test-id", "Attributes": "invalid json {"}

        # Should handle gracefully and return original span
        # Note: invalid JSON causes the Attributes key to be removed
        result = processor.process_span(span_data)
        assert result["Id"] == "test-id"
        assert "Attributes" not in result  # Attributes key is removed on invalid JSON


# uipath-langchain==0.0.123.dev1001490444
