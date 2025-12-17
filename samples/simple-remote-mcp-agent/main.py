from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from uipath.agent.models.agent import AgentMcpResourceConfig, AgentMcpTool
from uipath.agent.react import AGENT_SYSTEM_PROMPT_TEMPLATE

from uipath_langchain.agent.react import create_agent
from uipath_langchain.agent.tools import create_mcp_tools
from uipath_langchain.chat import OpenAIModels, UiPathChat

# LLM Model Configuration
llm = UiPathChat(
    model=OpenAIModels.gpt_5_mini_2025_08_07,
    temperature=0.0,
    max_tokens=16384,
)


# Input/Output Models
class AgentInput(BaseModel):
    pass


class AgentOutput(BaseModel):
    content: str | None = Field(None, description="Output content")


# Agent Messages Function
def create_messages(state: AgentInput) -> Sequence[SystemMessage | HumanMessage]:
    # Apply system prompt template
    current_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    system_prompt_content = f"""You are an agentic assistant."""
    enhanced_system_prompt = (
        AGENT_SYSTEM_PROMPT_TEMPLATE.replace("{{systemPrompt}}", system_prompt_content)
        .replace("{{currentDate}}", current_date)
        .replace("{{agentName}}", "Mr Assistant")
    )

    return [
        SystemMessage(content=enhanced_system_prompt),
        HumanMessage(content=f"""Do me an RPA echo of "asdf"."""),
    ]


mcpTools = [
    AgentMcpResourceConfig(
        name="uipath-server",
        description="a",
        folderPath="611ca479-1f38-4abc-b2c6-6a61fa002978",
        slug="uipath-server",
        availableTools=[],
        isEnabled=True,
    ),
    AgentMcpResourceConfig(
        name="uipath-server-2",
        description="a",
        folderPath="611ca479-1f38-4abc-b2c6-6a61fa002978",
        slug="uipath-server-2",
        availableTools=[],
        isEnabled=True,
    ),
    AgentMcpResourceConfig(
        name="hello-world",
        description="a",
        folderPath="611ca479-1f38-4abc-b2c6-6a61fa002978",
        slug="hello-world",
        availableTools=[],
        isEnabled=True,
    ),
    AgentMcpResourceConfig(
        name="mcp-hello-world-24-25",
        description="a",
        folderPath="611ca479-1f38-4abc-b2c6-6a61fa002978",
        slug="mcp-hello-world-24-25",
        availableTools=[
            AgentMcpTool(
                name="add",
                description="""
                Add two numbers together.
                    Args:
                        a: First number
                        b: Second number

                    Returns:
                        Sum of a and b
                """,
                inputSchema={
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "title": "A"},
                        "b": {"type": "number", "title": "B"},
                    },
                    "required": ["a", "b"],
                },
            )
        ],
        isEnabled=True,
    ),
]

all_tools: list[BaseTool] = []


@asynccontextmanager
async def make_graph():
    async with create_mcp_tools(mcpTools) as tools:
        yield create_agent(
            model=llm,
            tools=tools + all_tools,
            messages=create_messages,
            input_schema=AgentInput,
            output_schema=AgentOutput,
        )
