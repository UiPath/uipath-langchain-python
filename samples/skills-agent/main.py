"""A LangGraph agent that answers accounts-payable tasks with bound skill bundles."""

from typing import Annotated

from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import Runnable
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from uipath.platform.skills import SkillBundleRef, VersionMode

from uipath_langchain.chat import UiPathChatAnthropicBedrock
from uipath_langchain.skills import UiPathSkills

BASE_PROMPT = (
    "You are an accounts-payable assistant. Answer the user's task concisely. "
    "When one of your skills matches the task, load it and follow its steps."
)

# Declared at import time; nothing is fetched until ``SKILLS.load()`` runs.
SKILLS = UiPathSkills(
    bundles=[
        SkillBundleRef(
            name="finance",
            folder_path="Shared/Skills",
            version_mode=VersionMode.PINNED,
            version="3",
        ),
        SkillBundleRef(
            name="ops", folder_path="Shared/Skills", version_mode=VersionMode.LATEST
        ),
    ]
)


class GraphInput(BaseModel):
    task: str


class GraphOutput(BaseModel):
    answer: str
    skills_used: list[str]


class GraphState(BaseModel):
    task: str
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)


def make_llm() -> BaseChatModel:
    """The chat model, through the UiPath LLM gateway."""
    return UiPathChatAnthropicBedrock(model="anthropic.claude-haiku-4-5-20251001-v1:0")


def skills_used(messages: list[AnyMessage]) -> list[str]:
    """Qualified names of the skills ``load_skill`` returned, in first-load order."""
    requested: dict[str, str] = {}
    names: list[str] = []
    for message in messages:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                if call["name"] == "load_skill" and call["id"]:
                    requested[call["id"]] = str(call["args"].get("name", ""))
        elif isinstance(message, ToolMessage) and message.tool_call_id in requested:
            name = requested[message.tool_call_id]
            loaded = message.status != "error" and not message.text.startswith(
                "Unknown skill"
            )
            if loaded and name not in names:
                names.append(name)
    return names


async def make_graph() -> CompiledStateGraph[GraphState, None, GraphInput, GraphOutput]:
    """Resolve the skill bundles once, then build the agent around them."""
    await SKILLS.load()
    load_skill = SKILLS.load_skill_tool()
    tools = [load_skill] if load_skill is not None else []
    system_prompt = "\n\n".join(p for p in [BASE_PROMPT, SKILLS.prompt_section()] if p)
    model: Runnable[LanguageModelInput, AIMessage] | None = None

    async def agent(state: GraphState) -> dict[str, list[AnyMessage]]:
        nonlocal model
        if model is None:
            # Built on the first step, not in the factory, so `uipath init`
            # needs no LLM credentials.
            llm = make_llm()
            model = llm.bind_tools(tools) if tools else llm
        prompt = [SystemMessage(system_prompt), HumanMessage(state.task)]
        response = await model.ainvoke([*prompt, *state.messages])
        return {"messages": [response]}

    def finish(state: GraphState) -> GraphOutput:
        last = next(m for m in reversed(state.messages) if isinstance(m, AIMessage))
        return GraphOutput(answer=last.text, skills_used=skills_used(state.messages))

    builder = StateGraph(GraphState, input_schema=GraphInput, output_schema=GraphOutput)
    builder.add_node("agent", agent)
    builder.add_node("finish", finish)
    builder.add_edge(START, "agent")
    if tools:
        builder.add_node("tools", ToolNode(tools))
        builder.add_conditional_edges(
            "agent", tools_condition, {"tools": "tools", END: "finish"}
        )
        builder.add_edge("tools", "agent")
    else:
        builder.add_edge("agent", "finish")
    builder.add_edge("finish", END)
    return builder.compile()
