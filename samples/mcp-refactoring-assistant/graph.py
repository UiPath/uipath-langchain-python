"""LangGraph agent for Code Refactoring Assistant.
"""

import json
import sys
from pathlib import Path

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field

from uipath_langchain.chat import UiPathChat

model = UiPathChat(model="gpt-4o-2024-08-06", temperature=0.7)

server_path = Path(__file__).parent / "server.py"

manager = None


class StructuredOutput(BaseModel):
    """Structured output from the agent containing refactoring guide information."""
    prompt_name: str = Field(description="The name of the refactoring prompt to use (e.g., 'extract_function', 'simplify_conditional')")
    arguments: dict = Field(description="Arguments to pass to the refactoring prompt including complexity and smell information")


class MCPManager:
    def __init__(self):
        self._client: MultiServerMCPClient | None = None
        self._react_agent = None

    async def get_client(self):
        if self._client is None:
            self._client = MultiServerMCPClient({
                "code-refactoring": {
                    "command": sys.executable,
                    "args": [str(server_path)],
                    "transport": "stdio",
                },
            })
        return self._client

    async def get_agent(self):
        if self._react_agent is None:
            client = await self.get_client()
            tools = await client.get_tools()

            model_with_tools = model.bind_tools(tools)

            self._react_agent = create_agent(
                model_with_tools,
                tools=tools,
                response_format=ToolStrategy(StructuredOutput)
            )
        return self._react_agent


manager = MCPManager()


class Input(BaseModel):
    """Input schema: the code to analyze."""
    code: str


class Output(BaseModel):
    """Output schema: just the refactoring result."""
    result: str


class State(BaseModel):
    """Internal state for the graph (not exposed to user)."""
    code: str
    prompt_name: str = ""
    prompt_arguments: dict = {}


async def agent_node(state: State) -> State:
    """Agent analyzes code and determines which prompt to use."""
    react_agent = await manager.get_agent()

    system_msg = SystemMessage(
        content=(
            "You are a refactoring assistant.\n\n"
            "1) Analyze this code using analyze_code_complexity\n"
            "2) Detect issues using detect_code_smells\n"
            "3) Call get_refactoring_guide with:\n"
            "   - issue_type: the main issue detected\n"
            "   - code: the code to refactor\n"
            "   - complexity_info: results from step 1\n"
            "   - smells_info: results from step 2\n\n"
            "After gathering this information, provide a structured output with:\n"
            "- prompt_name: the refactoring strategy to use\n"
            "- arguments: all relevant information gathered from the tools"
        )
    )

    user_msg = HumanMessage(content=state.code)

    messages_state = MessagesState(messages=[system_msg, user_msg])
    result = await react_agent.ainvoke(messages_state)

    structured_response = result.get("structured_response")

    if not structured_response or not structured_response.prompt_name:
        return State(code=state.code, prompt_name="", prompt_arguments={})

    prompt_args = structured_response.arguments or {}
    prompt_args.setdefault('code', state.code)

    return State(
        code=state.code,
        prompt_name=structured_response.prompt_name,
        prompt_arguments=prompt_args
    )


async def prompt_node(state: State) -> Output:
    """Fetch prompt using client.get_prompt() and generate final response."""
    if not state.prompt_name:
        return Output(
            result="Unable to determine appropriate refactoring prompt. "
                   "Please ensure the agent analyzed the code and called get_refactoring_guide."
        )

    client = await manager.get_client()
    serialized_args = {
        key: json.dumps(value) if isinstance(value, dict) else value
        for key, value in state.prompt_arguments.items()
    }

    prompt_messages = await client.get_prompt(
        "code-refactoring",
        state.prompt_name,
        arguments=serialized_args
    )

    final_response = await model.ainvoke(prompt_messages)

    return Output(result=final_response.content)


# Build and compile graph at module level
builder = StateGraph(State, input=Input, output=Output)
builder.add_node("agent", agent_node)
builder.add_node("prompt", prompt_node)

builder.add_edge(START, "agent")
builder.add_edge("agent", "prompt")
builder.add_edge("prompt", END)

graph = builder.compile()

