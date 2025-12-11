from typing import Any, Literal, TypedDict

import pytest
from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel
from uipath.runtime.schema import UiPathRuntimeGraph, UiPathRuntimeNode

from uipath_langchain.chat import UiPathChat
from uipath_langchain.runtime.schema import get_graph_schema


@tool
def search_movies(query: str) -> str:
    """Search for movies based on a query."""
    return f"Search results for: {query}"


def test_agent_graph_schema():
    """Test that create_agent produces the expected graph schema structure."""

    # Setup
    movie_system_prompt = """You are an advanced AI assistant specializing in movie research and analysis."""
    llm = UiPathChat(
        model="claude-3-7-sonnet-latest",
        access_token="test-token",
        azure_endpoint="test-base-url",
        client_id="test-client-id",
    )
    graph: CompiledStateGraph[Any, Any, Any, Any] = create_agent(
        llm, tools=[search_movies], system_prompt=movie_system_prompt
    )

    # Get the schema
    schema = get_graph_schema(graph, xray=1)

    # Validate nodes
    node_ids = [node.id for node in schema.nodes]
    assert "__start__" in node_ids
    assert "__end__" in node_ids
    assert "model" in node_ids
    assert "tools" in node_ids
    assert len(node_ids) == 4

    # Validate node types
    node_types = {node.id: node.type for node in schema.nodes}
    assert node_types["__start__"] == "__start__"
    assert node_types["__end__"] == "__end__"
    assert node_types["model"] == "model"
    assert node_types["tools"] == "tool"

    # Validate model metadata
    model_node = next(node for node in schema.nodes if node.id == "model")
    model_metadata = model_node.metadata
    assert model_metadata is not None
    assert model_metadata.get("model_name") == "claude-3-7-sonnet-latest"
    assert "max_tokens" in model_metadata

    # Validate tools metadata
    tools_node = next(node for node in schema.nodes if node.id == "tools")
    tools_metadata = tools_node.metadata
    assert tools_metadata is not None
    assert tools_metadata.get("tool_names") == ["search_movies"]
    assert tools_metadata.get("tool_count") == 1

    # Validate edges
    edges = [(edge.source, edge.target) for edge in schema.edges]
    expected_edges = [
        ("__start__", "model"),
        ("model", "__end__"),
        ("model", "tools"),
        ("tools", "model"),
    ]

    assert len(edges) == len(expected_edges)
    for expected_edge in expected_edges:
        assert expected_edge in edges, f"Missing edge: {expected_edge}"

    # Validate no duplicate edges
    assert len(edges) == len(set(edges)), "Duplicate edges found"

    # Validate all edges have null labels
    for edge in schema.edges:
        assert edge.label is None or edge.label == ""


def test_supervisor_graph_schema():
    """Test that a supervisor graph produces the expected schema structure."""

    @tool
    def search_tool(query: str) -> str:
        """Search for information."""
        return f"Search results for: {query}"

    @tool
    def calculator_tool(expression: str) -> str:
        """Calculate a mathematical expression."""
        return f"Result of {expression}"

    # Setup
    members = ["researcher", "coder"]

    system_prompt = (
        f"You are a supervisor managing workers: {members}. "
        "Respond with the worker to act next or FINISH when done."
    )

    class Router(TypedDict):
        """Worker to route to next."""

        next: Literal["researcher", "coder", "FINISH"]

    llm = UiPathChat(
        model="claude-3-7-sonnet-latest",
        access_token="test-token",
        azure_endpoint="test-base-url",
        client_id="test-client-id",
    )

    class GraphInput(BaseModel):
        question: str

    class GraphOutput(BaseModel):
        answer: str

    class State(MessagesState):
        next: str

    def get_message_text(msg: BaseMessage) -> str:
        if isinstance(msg.content, str):
            return msg.content
        return ""

    def input_node_func(state: GraphInput) -> dict[str, Any]:
        return {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=state.question),
            ],
            "next": "",
        }

    def supervisor_node_func(state: State) -> dict[str, Any] | GraphOutput:
        # Simplified synchronous version
        goto = state.get("next", "researcher")  # Mock routing
        if goto == "FINISH":
            return GraphOutput(answer=get_message_text(state["messages"][-1]))
        else:
            return {"next": goto}

    def route_supervisor(
        state: State,
    ) -> Literal["researcher", "coder", "__end__"]:
        next_node = state.get("next", "")
        if next_node == "researcher":
            return "researcher"
        elif next_node == "coder":
            return "coder"
        else:
            return "__end__"

    research_agent: CompiledStateGraph[Any, Any, Any, Any] = create_agent(
        llm, tools=[search_tool], system_prompt="You are a researcher."
    )

    def research_node(state: State):
        return {
            "messages": [HumanMessage(content="Research complete", name="researcher")]
        }

    code_agent: CompiledStateGraph[Any, Any, Any, Any] = create_agent(
        llm, tools=[calculator_tool]
    )

    def code_node(state: State):
        return {"messages": [HumanMessage(content="Code executed", name="coder")]}

    # Build graph
    builder = StateGraph(State, input_schema=GraphInput, output_schema=GraphOutput)
    builder.add_node("input", input_node_func)
    builder.add_node("supervisor", supervisor_node_func)
    builder.add_node("researcher", research_agent)  # Pass agent directly
    builder.add_node("coder", code_agent)  # Pass agent directly

    builder.add_edge(START, "input")
    builder.add_edge("input", "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {"researcher": "researcher", "coder": "coder", END: END},
    )
    builder.add_edge("researcher", "supervisor")
    builder.add_edge("coder", "supervisor")

    graph = builder.compile()

    schema = get_graph_schema(graph)

    # Validate top-level nodes
    node_ids = [node.id for node in schema.nodes]
    assert "__start__" in node_ids
    assert "__end__" in node_ids
    assert "input" in node_ids
    assert "supervisor" in node_ids
    assert "researcher" in node_ids
    assert "coder" in node_ids
    assert len(node_ids) == 6

    # Validate top-level edges
    edges = [(edge.source, edge.target) for edge in schema.edges]
    expected_edges = [
        ("__start__", "input"),
        ("input", "supervisor"),
        ("supervisor", "researcher"),
        ("supervisor", "coder"),
        ("supervisor", "__end__"),
        ("researcher", "supervisor"),
        ("coder", "supervisor"),
    ]

    assert len(edges) == len(expected_edges), (
        f"Expected {len(expected_edges)} edges, got {len(edges)}"
    )
    for expected_edge in expected_edges:
        assert expected_edge in edges, f"Missing edge: {expected_edge}"

    # Validate no duplicate edges
    assert len(edges) == len(set(edges)), "Duplicate edges found"

    # Validate researcher node has subgraph
    researcher_node: UiPathRuntimeNode = next(
        node for node in schema.nodes if node.id == "researcher"
    )
    assert researcher_node.subgraph is not None, (
        "Researcher node should have a subgraph"
    )

    # Validate researcher subgraph structure
    researcher_subgraph: UiPathRuntimeGraph = researcher_node.subgraph
    researcher_node_ids = [node.id for node in researcher_subgraph.nodes]
    assert "__start__" in researcher_node_ids
    assert "__end__" in researcher_node_ids
    assert "model" in researcher_node_ids
    assert "tools" in researcher_node_ids

    # Validate researcher tools metadata
    researcher_tools_node: UiPathRuntimeNode = next(
        node for node in researcher_subgraph.nodes if node.id == "tools"
    )
    researcher_tools_metadata = researcher_tools_node.metadata
    assert researcher_tools_metadata is not None
    assert researcher_tools_metadata.get("tool_names") == ["search_tool"]
    assert researcher_tools_metadata.get("tool_count") == 1

    # Validate coder node has subgraph
    coder_node: UiPathRuntimeNode = next(
        node for node in schema.nodes if node.id == "coder"
    )
    assert coder_node.subgraph is not None, "Coder node should have a subgraph"

    # Validate coder subgraph structure
    coder_subgraph = coder_node.subgraph
    coder_node_ids = [node.id for node in coder_subgraph.nodes]
    assert "__start__" in coder_node_ids
    assert "__end__" in coder_node_ids
    assert "model" in coder_node_ids
    assert "tools" in coder_node_ids

    # Validate coder tools metadata
    coder_tools_node: UiPathRuntimeNode = next(
        node for node in coder_subgraph.nodes if node.id == "tools"
    )
    coder_tools_metadata = coder_tools_node.metadata
    assert coder_tools_metadata is not None
    assert coder_tools_metadata.get("tool_names") == ["calculator_tool"]
    assert coder_tools_metadata.get("tool_count") == 1

    # Validate input and supervisor nodes have NO subgraph
    input_node_obj: UiPathRuntimeNode = next(
        node for node in schema.nodes if node.id == "input"
    )
    assert input_node_obj.subgraph is None, "Input node should not have a subgraph"

    supervisor_node_obj: UiPathRuntimeNode = next(
        node for node in schema.nodes if node.id == "supervisor"
    )
    assert supervisor_node_obj.subgraph is None, (
        "Supervisor node should not have a subgraph"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
