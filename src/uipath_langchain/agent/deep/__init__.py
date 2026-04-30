"""Deep agent support, built on the optional `deepagents` package.

Install the optional extra to use this module:

    pip install 'uipath-langchain[deep]'
    uv add 'uipath-langchain[deep]'

The `deepagents` types re-exported here (``SubAgent``, ``CompiledSubAgent``,
``BackendProtocol``, ``BackendFactory``) are loaded lazily so importing this
package without the extra installed does not crash — only attribute access
will raise ``ImportError`` with the install hint.
"""

from .agent import create_deep_agent, create_deep_agent_graph
from .types import DeepAgentGraphState
from .utils import create_state_with_input

_INSTALL_HINT = (
    "deepagents is required for deep agents. Install with: "
    "pip install 'uipath-langchain[deep]' "
    "(or: uv add 'uipath-langchain[deep]')"
)


def __getattr__(name: str):
    if name in ("SubAgent", "CompiledSubAgent"):
        try:
            import deepagents

            return getattr(deepagents, name)
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc
    if name == "BackendProtocol":
        try:
            from deepagents.backends import BackendProtocol

            return BackendProtocol
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc
    if name == "BackendFactory":
        try:
            from deepagents.backends.protocol import BackendFactory

            return BackendFactory
        except ImportError as exc:
            raise ImportError(_INSTALL_HINT) from exc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BackendFactory",
    "BackendProtocol",
    "CompiledSubAgent",
    "DeepAgentGraphState",
    "SubAgent",
    "create_deep_agent",
    "create_deep_agent_graph",
    "create_state_with_input",
]
