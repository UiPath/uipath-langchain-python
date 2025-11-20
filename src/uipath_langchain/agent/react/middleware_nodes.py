from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence, Tuple

from .middleware_types import AgentMiddleware
from .types import AgentGraphState


def create_middleware_nodes(
    middlewares: Sequence[AgentMiddleware] | None, hook_name: str
) -> Dict[str, Callable[[AgentGraphState], Any]]:
    nodes: Dict[str, Callable[[AgentGraphState], Any]] = {}

    if not middlewares:
        return nodes

    for idx, mw in enumerate(middlewares):
        hook = getattr(mw, hook_name, None)
        if not callable(hook):
            continue

        node_name = f"{mw.name}_{hook_name}_{idx}"

        async def node(state: AgentGraphState, _fn: Any = hook):
            result = _fn(state.messages, lambda msgs: msgs)
            if hasattr(result, "__await__"):
                result = await result
            if isinstance(result, list):
                return {"messages": result}
            return {"messages": state.messages}

        nodes[node_name] = node

    return nodes
