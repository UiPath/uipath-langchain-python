"""Deep agent builder."""

from collections.abc import Sequence

from deepagents import CompiledSubAgent, SubAgent
from deepagents import create_deep_agent as _create_deep_agent
from deepagents.backends import BackendProtocol
from deepagents.backends.protocol import BackendFactory
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph


def create_deep_agent(
    model: BaseChatModel,
    system_prompt: str = "",
    tools: Sequence[BaseTool] = (),
    subagents: Sequence[SubAgent | CompiledSubAgent] = (),
    backend: BackendProtocol | BackendFactory | None = None,
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
        subagents: Optional list of subagent configurations. Each entry is a
            ``SubAgent`` (name, description, system_prompt, and optional tools/model/middleware)
            or a ``CompiledSubAgent`` (name, description, and a pre-built runnable).
        backend: Storage backend for filesystem operations. Can be a
            ``BackendProtocol`` instance, a factory callable, or ``None``
            (uses the default in-state backend).

    Returns:
        Compiled LangGraph agent ready for execution.
    """
    return _create_deep_agent(
        model=model,
        system_prompt=system_prompt,
        tools=list(tools),
        subagents=list(subagents),
        backend=backend,
    )
