from __future__ import annotations

from typing import Any

from langchain_core.messages import AnyMessage
from langchain_core.tools import BaseTool


class AgentMiddleware:
    """Minimal AgentMiddleware compatibility layer.

    This mirrors the public surface used by our agent graph wiring and can be
    removed once we migrate to Langchain v 1.0.
    """

    # Optional: middlewares may register additional tools
    tools: list[BaseTool] = []

    @property
    def name(self) -> str:
        return self.__class__.__name__

    # Hook capability: by default, support any hook that is implemented
    def supports_hook(self, hook_name: str) -> bool:
        fn = getattr(self, hook_name, None)
        return callable(fn)

    # ---- Agent-level hooks (used today) ----
    def before_agent(self, messages: list[AnyMessage], handler: Any) -> Any:
        """Run before the agent node. May return a new messages list or None."""
        return None

    async def abefore_agent(self, messages: list[AnyMessage], handler: Any) -> Any:
        """Async variant of before_agent."""
        return self.before_agent(messages, handler)

    def after_agent(self, messages: list[AnyMessage], handler: Any) -> Any:
        """Run after the agent step. May return a new messages list or None."""
        return None

    async def aafter_agent(self, messages: list[AnyMessage], handler: Any) -> Any:
        """Async variant of after_agent."""
        return self.after_agent(messages, handler)

    # ---- Placeholders for future hooks (model/tool) ----
    def before_model(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        return None

    def after_model(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        return None

    def before_tool(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        return None

    def after_tool(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        return None
