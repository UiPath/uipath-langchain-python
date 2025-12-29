"""LangGraph agent for Code Refactoring Assistant.
"""

import json
import sys
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel

from uipath_langchain.chat import UiPathChat

model = UiPathChat(model="gpt-4o-2024-08-06", temperature=0.7)

server_path = Path(__file__).parent / "server.py"

manager = None


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

            model_with_tools = model.bind_tools(
                tools,
                tool_choice={"type": "function", "function": {"name": "get_refactoring_guide"}}
            )
            self._react_agent = create_agent(model_with_tools, tools=tools)
        return self._react_agent


manager = MCPManager()


def _try_parse_json(value) -> dict | None:
    """Robustly parse JSON from various formats (dict, str, list)."""
    if value is None:
        return None

    if isinstance(value, dict):
        if 'text' in value:
            return _try_parse_json(value['text'])
        if 'prompt_name' in value or 'error' in value:
            return value
        return None

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None

    if isinstance(value, list):
        for item in value:
            parsed = _try_parse_json(item)
            if parsed:
                return parsed

    return None


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
            'The get_refactoring_guide tool will return {"prompt_name": "...", "arguments": {...}} ready for the next step.'
        )
    )

    user_msg = HumanMessage(content=state.code)

    messages_state = MessagesState(messages=[system_msg, user_msg])
    result = await react_agent.ainvoke(messages_state)

    prompt_name = ""
    prompt_args: dict = {}

    for msg in result.get("messages", []):
        if hasattr(msg, 'type') and msg.type == 'tool':
            tool_name = getattr(msg, 'name', None) or getattr(msg, 'tool', None)
            if tool_name == 'get_refactoring_guide':
                data = _try_parse_json(getattr(msg, 'content', None))
                if data and 'prompt_name' in data:
                    prompt_name = data['prompt_name']
                    prompt_args = data.get('arguments', {}) or {}
                    break

    if not prompt_name:
        return State(code=state.code, prompt_name="", prompt_arguments={})

    prompt_args.setdefault('code', state.code)

    return State(code=state.code, prompt_name=prompt_name, prompt_arguments=prompt_args)


async def prompt_node(state: State) -> Output:
    """Fetch prompt using client.get_prompt() and generate final response."""
    if not state.prompt_name:
        return Output(
            result="Unable to determine appropriate refactoring prompt. "
                   "Please ensure the agent analyzed the code and called get_refactoring_guide."
        )

    client = await manager.get_client()

    prompt_messages = await client.get_prompt(
        "code-refactoring",
        state.prompt_name,
        arguments=state.prompt_arguments
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


