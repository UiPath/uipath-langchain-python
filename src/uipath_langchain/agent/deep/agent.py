"""Deep agent builder."""

from typing import Any, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph


def create_deep_agent(
    model: BaseChatModel,
    system_prompt: str = "",
    tools: Sequence[BaseTool] = (),
    subagents: Sequence[dict[str, Any]] = (),
    **kwargs: Any,
) -> CompiledStateGraph:
    """Create a deep agent.

    Deep agents provide built-in capabilities for:
    - Planning (write_todos, read_todos)
    - Filesystem operations (read_file, write_file, edit_file, ls, glob, grep)
    - Sub-agent delegation (task)
    - Auto-summarization for long conversations

    Args:
        model: A BaseChatModel instance.
        system_prompt: Instructions for the agent.
        tools: Custom tools to provide to the agent.
        subagents: Optional list of subagent configurations. Each subagent is a dict
            with keys: name, description, system_prompt, tools, model.
        **kwargs: Additional keyword arguments forwarded to the underlying
            ``deepagents.create_deep_agent`` (e.g. ``middleware``,
            ``interrupt_on``, ``checkpointer``).

    Returns:
        Compiled LangGraph agent ready for execution.
    """
    from deepagents import create_deep_agent as _create_deep_agent

    return _create_deep_agent(
        model=model,
        system_prompt=system_prompt,
        tools=list(tools),
        subagents=list(subagents),
        **kwargs,
    )
