"""LangGraph agent for Code Refactoring Assistant.

Correct MCP pattern: ReAct agent + prompt fetching.
"""

import json
import sys
from pathlib import Path
from typing import Any, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent

# ============= INITIALIZE ONCE =============

model = ChatAnthropic(
    model_name="claude-3-7-sonnet-latest",
    timeout=60,
    stop=None
)

server_path = Path(__file__).parent / "server.py"
client = MultiServerMCPClient({
    "code-refactoring": {
        "command": sys.executable,
        "args": [str(server_path)],
        "transport": "stdio",
    },
})


# ============= STATE =============

class GraphInput(TypedDict):
    code: str

class GraphState(TypedDict):
    code: str
    agent_result: Any
    prompt_name: str
    issue_type: str

class GraphOutput(TypedDict):
    result: str


async def agent_node(state: GraphState) -> GraphState:

    tools = await client.get_tools()
    agent = create_react_agent(model, tools)

    result = await agent.ainvoke({
        "messages": [
            HumanMessage(content=f"Analyze this code and suggest refactoring:\n\n{state['code']}")
        ]
    })

    prompt_name = "simplify_conditional_prompt"  # Default with _prompt suffix
    issue_type = "deep_nesting"

    for msg in result["messages"]:
        # Check for tool messages with content
        if hasattr(msg, 'content'):
            try:
                content = msg.content
                # Handle both string and list content
                if isinstance(content, list):
                    # Content is a list of content blocks
                    for block in content:
                        if isinstance(block, dict) and block.get('type') == 'text':
                            text = block.get('text', '')
                            data = json.loads(text)
                            if isinstance(data, dict) and "prompt_name" in data:
                                prompt_name = data["prompt_name"]
                                issue_type = data.get("issue_type", issue_type)
                                break
                elif isinstance(content, str):
                    # Content is a string
                    data = json.loads(content)
                    if isinstance(data, dict) and "prompt_name" in data:
                        prompt_name = data["prompt_name"]
                        issue_type = data.get("issue_type", issue_type)
                        break
            except (json.JSONDecodeError, TypeError, KeyError):
                continue

    return {
        "code": state["code"],
        "agent_result": result,
        "prompt_name": prompt_name,
        "issue_type": issue_type
    }


async def prompt_node(state: GraphState) -> GraphOutput:
    """Fetch MCP prompt and generate final refactoring."""

    prompt_name = state["prompt_name"]

    # Build arguments based on the specific prompt type
    arguments = {"code": state["code"]}

    # Add prompt-specific arguments
    if "simplify_conditional" in prompt_name:
        arguments["pattern"] = "guard_clause"
    elif "extract_method" in prompt_name:
        arguments["target_lines"] = "auto"
    elif "remove_duplication" in prompt_name:
        arguments["duplicate_blocks"] = ""
    elif "improve_naming" in prompt_name:
        arguments["symbols"] = ""
        arguments["context"] = ""

    prompt_messages = await client.get_prompt(
        "code-refactoring",
        prompt_name,
        arguments=arguments
    )

    final_response = await model.ainvoke(prompt_messages)

    return {"result": final_response.content}


async def make_graph():
    """Create the graph with 2 nodes."""

    builder = StateGraph(GraphState, input=GraphInput, output=GraphOutput)

    builder.add_node("agent", agent_node)
    builder.add_node("prompt", prompt_node)

    builder.add_edge(START, "agent")
    builder.add_edge("agent", "prompt")
    builder.add_edge("prompt", END)

    return builder.compile()


# Create the graph
graph = make_graph()
