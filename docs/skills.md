# Skills

A coded LangGraph agent can be bound to [skill bundles](../core/skills.md): versioned groups of skills, each a folder with a `SKILL.md` and its reference files. `uipath_langchain.skills` wires them into a graph: the skill descriptions go into the system prompt, and the model reads a skill's full `SKILL.md` on demand through a `load_skill` tool.

!!! warning "Preview Feature"
    The Orchestrator Skills API is not available yet. Until it ships, `UIPATH_SKILLS_SOURCE` defaults to `local` and bundles are read from `./skills` in the project; see [Local development](../core/skills.md#local-development). `orchestrator` becomes the default once the API is live.

How bundles are resolved, where their files land, what happens when one cannot be loaded, and how bindings and per-environment overrides work is described once, on the SDK's [Skills](../core/skills.md) page. This page covers the LangChain side.

---

## UiPathSkills

Declare the bundles at module level with `UiPathSkills`. Nothing is fetched and no credentials are needed until `load()` runs, so importing the agent module stays cheap.

```python
from uipath.platform.skills import SkillBundleRef, VersionMode
from uipath_langchain.skills import UiPathSkills

SKILLS = UiPathSkills(
    bundles=[
        SkillBundleRef(name="finance", folder_path="Shared/Skills", version_mode=VersionMode.PINNED, version="3"),
        SkillBundleRef(name="ops", folder_path="Shared/Skills", version_mode=VersionMode.LATEST),
    ]
)
```

| Call | Returns |
| --- | --- |
| `await SKILLS.load()` | The resolved `SkillSet`. Runs once per process; later and concurrent calls share the result. Never raises for a bundle or platform error: those land in `SkillSet.failures`. |
| `SKILLS.prompt_section()` | The `## Skills` system-prompt section plus how to use the tool; `""` before `load()` or when no bundle has a skill. |
| `SKILLS.load_skill_tool()` | The `load_skill` tool. After `load()`: a tool whose `name` argument is an enum of the qualified `<bundle>/<skill>` names, or `None` when there are no skills. Before `load()`: a lazy tool that loads the set on its first call. |
| `SKILLS.advanced_middleware(backend)` | The middleware for an advanced (deepagents) agent; see [Advanced agents](#advanced-agents). |
| `SKILLS.cleanup()` | Removes the set's temp folder and forgets the set. |

Each `SkillBundleRef` also needs a `skillBundle` entry in `bindings.json`; see [Bindings](../core/skills.md#bindings).

---

## Graph Factory

Resolve the bundles in a graph factory: `langgraph.json` points to an async function that builds the graph, and the runtime awaits it before the run starts.

```json
{
  "graphs": {
    "agent": "./main.py:make_graph"
  }
}
```

```python
import functools
from typing import Annotated

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field

from uipath_langchain.chat import UiPathChatAnthropicBedrock

BASE_PROMPT = "You are an accounts-payable assistant. When one of your skills matches the task, load it and follow its steps."


class GraphInput(BaseModel):
    task: str


class GraphState(BaseModel):
    task: str
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)


@functools.cache
def make_llm():
    # Built on the first agent step and reused, so `uipath init` needs no LLM credentials.
    return UiPathChatAnthropicBedrock(model="anthropic.claude-haiku-4-5-20251001-v1:0")


async def make_graph():
    await SKILLS.load()
    load_skill = SKILLS.load_skill_tool()
    tools = [load_skill] if load_skill is not None else []
    system_prompt = "\n\n".join(p for p in [BASE_PROMPT, SKILLS.prompt_section()] if p)

    async def agent(state: GraphState):
        llm = make_llm()
        model = llm.bind_tools(tools) if tools else llm
        prompt = [SystemMessage(system_prompt), HumanMessage(state.task)]
        return {"messages": [await model.ainvoke([*prompt, *state.messages])]}

    builder = StateGraph(GraphState, input_schema=GraphInput)
    builder.add_node("agent", agent)
    builder.add_edge(START, "agent")
    if tools:
        builder.add_node("tools", ToolNode(tools))
        builder.add_conditional_edges("agent", tools_condition)
        builder.add_edge("tools", "agent")
    else:
        builder.add_edge("agent", END)
    return builder.compile()
```

- `uipath init` also calls the factory to build `entry-points.json`, so run `uipath auth` first: without credentials the bundles do not resolve, and the graph is built without the `tools` node. The chat model is built on the first agent step, not in the factory, so that `init` itself needs no LLM credentials.
- When no bundle has a skill, `load_skill_tool()` returns `None`: the graph runs without the tool and without a `## Skills` section.

The [skills-agent sample](https://github.com/UiPath/uipath-langchain-python/tree/main/samples/skills-agent) is this graph with an output node that reports the skills the model loaded, plus a `skills/` folder, `bindings.json` and an override example to run it locally.

---

## Module-Level Graph

When the graph is a module-level `graph` instead of a factory, call `await SKILLS.load()` in the first node. Give `ToolNode` the lazy tool from `load_skill_tool()` at import time: it loads the set (or reuses the cached one) on its first call. In the node, bind the model to the tool `load_skill_tool()` returns after `load()`, which lists the skill names as an enum.

```python
from langgraph.graph import MessagesState

tools = [SKILLS.load_skill_tool()]  # lazy: nothing is fetched at import


async def agent(state: MessagesState):
    await SKILLS.load()  # once per process; cached afterwards
    load_skill = SKILLS.load_skill_tool()
    llm = make_llm()  # the cached model from the factory example
    model = llm.bind_tools([load_skill]) if load_skill is not None else llm
    system = SystemMessage("\n\n".join(p for p in [BASE_PROMPT, SKILLS.prompt_section()] if p))
    return {"messages": [await model.ainvoke([system, *state["messages"]])]}


builder = StateGraph(MessagesState)
builder.add_node("agent", agent)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")
graph = builder.compile()
```

The set is resolved once per process. A job resumed in a new process resolves the bundles again, so a `latest` bundle may come back newer.

---

## Advanced Agents

An advanced (deepagents) agent reads skills from its workspace with `read_file` instead of a `load_skill` tool. Load the bundles into the `skills` folder of the backend's root, so they appear at `/skills/<bundle>/<skill>/`, and pass `SKILLS.advanced_middleware(backend)`:

```python
from pathlib import Path

from deepagents.backends import FilesystemBackend
from uipath_langchain.agent.advanced import create_advanced_agent
from uipath_langchain.skills import autonomous_skills_directive


async def make_graph():
    workspace = Path("./workspace").resolve()
    backend = FilesystemBackend(root_dir=workspace, virtual_mode=True)
    await SKILLS.load(destination=workspace / "skills")
    model = UiPathChatAnthropicBedrock(model="anthropic.claude-haiku-4-5-20251001-v1:0")
    return create_advanced_agent(
        model,
        system_prompt=BASE_PROMPT + "\n\n" + autonomous_skills_directive(),
        backend=backend,
        middleware=SKILLS.advanced_middleware(backend),
    )
```

- deepagents needs the model when the graph is built, so the model is created in the factory and `uipath init` needs LLM credentials too; building the model on the first step applies to the plain LangGraph patterns above only.
- Pass the middleware, not `create_advanced_agent`'s `skills` argument; do not pass both.
- The middleware applies to the main agent only. Subagents do not see the skills list; the prompt tells the main agent to put a skill's `SKILL.md` path in the task it delegates.
- `destination` is emptied on every load; see [Files on disk](../core/skills.md#files-on-disk).
- `autonomous_skills_directive()` tells an agent that runs without a human in the loop to take a skill's defaults instead of asking, and to stop rather than run a step that needs approval or would delete or overwrite something. Leave it out for a conversational agent, which can ask the user.
- The middleware records a `Skill load` span whenever the model reads a `SKILL.md` with `read_file`.

---

## Traces

`uipath run agent '{...}' --trace-file trace.jsonl` writes the spans to a JSON Lines file. A run of the sample that loads one skill contains, among the LangGraph spans:

```
Skill bundles resolved                   skill.source=local  skill.bundle.count=2  skill.count=4
  event skill.bundle.resolved            skill.bundle.name=finance  skill.version=3  skill.version_mode=pinned
  event skill.bundle.resolved            skill.bundle.name=ops      skill.version=1  skill.version_mode=latest
Skill load - ops/escalation-policy       skill.version=1  skill.invoked_by=model
```

What each span and event carries is listed under [Traces](../core/skills.md#traces).

---

## See Also

- [Declare and load bundles](../core/skills.md#declare-and-load-bundles): version modes, folders and what a failed bundle looks like.
- [Local development](../core/skills.md#local-development) and [Environment variables](../core/environment_variables.md#skill-bundles): the `local` source and the `orchestrator` placeholder.
- [Bindings](../core/skills.md#bindings) and [Overrides per environment](../core/skills.md#overrides-per-environment).
- [Limits](../core/skills.md#limits): file, bundle and time limits.
