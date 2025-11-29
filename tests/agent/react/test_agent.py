"""End-to-end tests for create_agent function."""

from typing import Any, Sequence

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool, tool
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

from uipath_langchain.agent.react.agent import create_agent, create_state_with_input
from uipath_langchain.agent.react.exceptions import AgentTerminationException
from uipath_langchain.agent.react.types import AgentGraphConfig, AgentGraphState


class SharedState:
    """Shared state container for mock model instances."""

    def __init__(self, responses: list[AIMessage]):
        self.responses = responses
        self.call_count = 0


class MockChatModel(BaseChatModel):
    """Mock chat model for testing that returns predefined responses."""

    _shared_state: SharedState | None = None
    bound_tools: list[BaseTool] = []

    def __init__(self, responses: list[AIMessage] | None = None, **kwargs):
        super().__init__(**kwargs)
        if responses is not None:
            self._shared_state = SharedState(responses)

    @property
    def call_count(self) -> int:
        return self._shared_state.call_count if self._shared_state else 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self._shared_state is None:
            raise RuntimeError("Mock not initialized with responses")
        if self._shared_state.call_count >= len(self._shared_state.responses):
            raise RuntimeError(
                f"Mock exhausted: {self._shared_state.call_count} calls but only {len(self._shared_state.responses)} responses"
            )
        response = self._shared_state.responses[self._shared_state.call_count]
        self._shared_state.call_count += 1
        return ChatResult(generations=[ChatGeneration(message=response)])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop, run_manager, **kwargs)

    def bind_tools(
        self,
        tools: Sequence[BaseTool],
        **kwargs,
    ) -> "MockChatModel":
        """Bind tools to the model (share state with parent)."""
        new_model = MockChatModel.__new__(MockChatModel)
        BaseChatModel.__init__(new_model)
        new_model._shared_state = self._shared_state
        new_model.bound_tools = list(tools)
        return new_model

    def bind(self, **kwargs) -> "MockChatModel":
        """Bind additional kwargs (e.g., tool_choice)."""
        new_model = MockChatModel.__new__(MockChatModel)
        BaseChatModel.__init__(new_model)
        new_model._shared_state = self._shared_state
        new_model.bound_tools = self.bound_tools
        return new_model

    @property
    def _llm_type(self) -> str:
        return "mock"


class TestCreateStateWithInput:
    """Test cases for create_state_with_input function."""

    def test_creates_combined_state_class(self):
        """Should create state class combining AgentGraphState and input schema."""

        class MyInput(BaseModel):
            query: str

        result = create_state_with_input(MyInput)
        assert issubclass(result, AgentGraphState)
        assert "query" in result.model_fields
        assert "messages" in result.model_fields

    def test_with_base_model_input(self):
        """Should work with plain BaseModel."""
        result = create_state_with_input(BaseModel)
        assert issubclass(result, AgentGraphState)

    def test_with_complex_input_schema(self):
        """Should handle complex input schemas."""

        class ComplexInput(BaseModel):
            name: str
            count: int = 0
            tags: list[str] = Field(default_factory=list)

        result = create_state_with_input(ComplexInput)
        assert "name" in result.model_fields
        assert "count" in result.model_fields
        assert "tags" in result.model_fields


class TestCreateAgent:
    """Test cases for create_agent function."""

    def test_returns_state_graph(self):
        """Should return a StateGraph instance."""
        model = MockChatModel(responses=[])
        messages = [SystemMessage(content="You are a helpful assistant.")]

        result = create_agent(model=model, tools=[], messages=messages)

        assert isinstance(result, StateGraph)

    def test_graph_has_required_nodes(self):
        """Should create graph with init, agent, and terminate nodes."""
        model = MockChatModel(responses=[])
        messages = [SystemMessage(content="Test")]

        builder = create_agent(model=model, tools=[], messages=messages)

        # Check nodes are registered
        assert "init" in builder.nodes
        assert "agent" in builder.nodes
        assert "terminate" in builder.nodes

    def test_graph_includes_tool_nodes(self):
        """Should add tool nodes for each provided tool."""

        @tool
        def search_tool(query: str) -> str:
            """Search for information."""
            return f"Results for: {query}"

        @tool
        def calc_tool(expression: str) -> str:
            """Calculate expression."""
            return f"Result: {expression}"

        model = MockChatModel(responses=[])
        messages = [SystemMessage(content="Test")]

        builder = create_agent(
            model=model, tools=[search_tool, calc_tool], messages=messages
        )

        assert "search_tool" in builder.nodes
        assert "calc_tool" in builder.nodes

    def test_uses_custom_config(self):
        """Should apply custom configuration."""
        model = MockChatModel(responses=[])
        messages = [SystemMessage(content="Test")]
        config = AgentGraphConfig(recursion_limit=100)

        builder = create_agent(model=model, tools=[], messages=messages, config=config)

        assert isinstance(builder, StateGraph)

    def test_with_input_schema(self):
        """Should accept input schema parameter."""

        class MyInput(BaseModel):
            query: str

        model = MockChatModel(responses=[])
        messages = [SystemMessage(content="Test")]

        builder = create_agent(
            model=model, tools=[], messages=messages, input_schema=MyInput
        )

        assert isinstance(builder, StateGraph)

    def test_with_output_schema(self):
        """Should accept output schema parameter."""

        class MyOutput(BaseModel):
            result: str

        model = MockChatModel(responses=[])
        messages = [SystemMessage(content="Test")]

        builder = create_agent(
            model=model, tools=[], messages=messages, output_schema=MyOutput
        )

        assert isinstance(builder, StateGraph)

    def test_with_callable_messages(self):
        """Should accept callable that generates messages from state."""

        class MyInput(BaseModel):
            topic: str

        def message_factory(state: MyInput):
            return [
                SystemMessage(content="You are an expert."),
                HumanMessage(content=f"Tell me about {state.topic}"),
            ]

        model = MockChatModel(responses=[])

        builder = create_agent(
            model=model,
            tools=[],
            messages=message_factory,
            input_schema=MyInput,
        )

        assert isinstance(builder, StateGraph)


class DefaultOutput(BaseModel):
    """Default output schema for e2e tests."""

    success: bool
    message: str = ""


class TestCreateAgentE2E:
    """End-to-end tests for agent execution."""

    @pytest.mark.asyncio
    async def test_direct_end_execution(self):
        """Should complete when LLM calls end_execution directly."""
        model = MockChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "end_execution",
                            "args": {"success": True, "message": "Task completed"},
                            "id": "call_1",
                        }
                    ],
                )
            ]
        )
        messages = [
            SystemMessage(content="You are helpful."),
            HumanMessage(content="Say hello"),
        ]

        builder = create_agent(
            model=model, tools=[], messages=messages, output_schema=DefaultOutput
        )
        graph = builder.compile()

        result = await graph.ainvoke({})

        assert result["success"] is True
        assert result["message"] == "Task completed"

    @pytest.mark.asyncio
    async def test_tool_use_then_end_execution(self):
        """Should execute tool and then complete with end_execution."""

        @tool
        def get_weather(city: str) -> str:
            """Get weather for a city."""
            return f"Weather in {city}: Sunny, 72F"

        model = MockChatModel(
            responses=[
                # First call: use the tool
                AIMessage(
                    content="Let me check the weather.",
                    tool_calls=[
                        {
                            "name": "get_weather",
                            "args": {"city": "Paris"},
                            "id": "call_weather",
                        }
                    ],
                ),
                # Second call: end execution with result
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "end_execution",
                            "args": {
                                "success": True,
                                "message": "The weather in Paris is sunny.",
                            },
                            "id": "call_end",
                        }
                    ],
                ),
            ]
        )
        messages = [
            SystemMessage(content="You are a weather assistant."),
            HumanMessage(content="What's the weather in Paris?"),
        ]

        builder = create_agent(
            model=model,
            tools=[get_weather],
            messages=messages,
            output_schema=DefaultOutput,
        )
        graph = builder.compile()

        result = await graph.ainvoke({})

        assert result["success"] is True
        assert "Paris" in result["message"] or "sunny" in result["message"].lower()
        assert model.call_count == 2

    @pytest.mark.asyncio
    async def test_multiple_tool_calls(self):
        """Should handle multiple sequential tool calls."""

        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        @tool
        def multiply(a: int, b: int) -> int:
            """Multiply two numbers."""
            return a * b

        model = MockChatModel(
            responses=[
                # First: add
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "add", "args": {"a": 2, "b": 3}, "id": "call_add"}
                    ],
                ),
                # Second: multiply
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "multiply",
                            "args": {"a": 5, "b": 4},
                            "id": "call_mult",
                        }
                    ],
                ),
                # Third: end
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "end_execution",
                            "args": {"success": True, "message": "Calculations done"},
                            "id": "call_end",
                        }
                    ],
                ),
            ]
        )
        messages = [HumanMessage(content="Do some math")]

        builder = create_agent(
            model=model,
            tools=[add, multiply],
            messages=messages,
            output_schema=DefaultOutput,
        )
        graph = builder.compile()

        result = await graph.ainvoke({})

        assert result["success"] is True
        assert model.call_count == 3

    @pytest.mark.asyncio
    async def test_raise_error_terminates_with_exception(self):
        """Should raise AgentTerminationException when raise_error is called."""
        model = MockChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "raise_error",
                            "args": {
                                "message": "Cannot complete task",
                                "details": "Missing required data",
                            },
                            "id": "call_error",
                        }
                    ],
                )
            ]
        )
        messages = [HumanMessage(content="Do something impossible")]

        builder = create_agent(model=model, tools=[], messages=messages)
        graph = builder.compile()

        with pytest.raises(AgentTerminationException) as exc_info:
            await graph.ainvoke({})

        assert "Cannot complete task" in str(exc_info.value.error_info.title)

    @pytest.mark.asyncio
    async def test_with_custom_output_schema(self):
        """Should validate output against custom schema."""

        class TaskResult(BaseModel):
            task_id: str
            status: str
            result_data: dict[str, Any] = Field(default_factory=dict)

        model = MockChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "end_execution",
                            "args": {
                                "task_id": "task-123",
                                "status": "completed",
                                "result_data": {"key": "value"},
                            },
                            "id": "call_end",
                        }
                    ],
                )
            ]
        )
        messages = [HumanMessage(content="Process task")]

        builder = create_agent(
            model=model, tools=[], messages=messages, output_schema=TaskResult
        )
        graph = builder.compile()

        result = await graph.ainvoke({})

        assert result["task_id"] == "task-123"
        assert result["status"] == "completed"
        assert result["result_data"] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_with_input_schema_and_dynamic_messages(self):
        """Should pass input state to dynamic message factory."""

        class QueryInput(BaseModel):
            question: str
            context: str = ""

        def create_messages(state: QueryInput):
            messages = [SystemMessage(content="Answer based on context.")]
            if state.context:
                messages.append(HumanMessage(content=f"Context: {state.context}"))
            messages.append(HumanMessage(content=state.question))
            return messages

        model = MockChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "end_execution",
                            "args": {"success": True, "message": "Answered"},
                            "id": "call_end",
                        }
                    ],
                )
            ]
        )

        builder = create_agent(
            model=model,
            tools=[],
            messages=create_messages,
            input_schema=QueryInput,
            output_schema=DefaultOutput,
        )
        graph = builder.compile()

        result = await graph.ainvoke(
            {"question": "What is AI?", "context": "AI is artificial intelligence."}
        )

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_parallel_tool_calls_filtered(self):
        """Should handle parallel tool calls with control flow tools filtered."""

        @tool
        def fetch_data(source: str) -> str:
            """Fetch data from source."""
            return f"Data from {source}"

        model = MockChatModel(
            responses=[
                # LLM tries to call tool AND end_execution (control flow filtered)
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "fetch_data",
                            "args": {"source": "api"},
                            "id": "call_fetch",
                        },
                        {
                            "name": "end_execution",
                            "args": {"success": True},
                            "id": "call_end",
                        },
                    ],
                ),
                # Next turn: just end
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "end_execution",
                            "args": {"success": True, "message": "Done"},
                            "id": "call_end_2",
                        }
                    ],
                ),
            ]
        )
        messages = [HumanMessage(content="Fetch and complete")]

        builder = create_agent(
            model=model,
            tools=[fetch_data],
            messages=messages,
            output_schema=DefaultOutput,
        )
        graph = builder.compile()

        result = await graph.ainvoke({})

        assert result["success"] is True
        # Router should have filtered end_execution, executed tool, then ended
        assert model.call_count == 2

    @pytest.mark.asyncio
    async def test_empty_tools_list(self):
        """Should work with no tools provided."""
        model = MockChatModel(
            responses=[
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "end_execution",
                            "args": {"success": True, "message": "No tools needed"},
                            "id": "call_end",
                        }
                    ],
                )
            ]
        )
        messages = [HumanMessage(content="Just respond")]

        builder = create_agent(
            model=model, tools=[], messages=messages, output_schema=DefaultOutput
        )
        graph = builder.compile()

        result = await graph.ainvoke({})

        assert result["success"] is True
        assert result["message"] == "No tools needed"
