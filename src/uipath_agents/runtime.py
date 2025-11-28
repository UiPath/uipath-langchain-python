from uipath.runtime import UiPathRuntimeContext, UiPathRuntimeFactoryRegistry

from ._cli.constants import AGENT_ENTRYPOINT
from ._cli.runtime.factory import AgentRuntimeFactory


def _create_agent_factory(context: UiPathRuntimeContext | None) -> AgentRuntimeFactory:
    """Create an AgentRuntimeFactory with the given context."""
    if context is None:
        raise ValueError("UiPathRuntimeContext is required for AgentRuntimeFactory")
    return AgentRuntimeFactory(context=context)


def register_runtime_factory() -> None:
    """Register Agents runtime factory."""
    UiPathRuntimeFactoryRegistry.register(
        "agents",
        factory_callable=_create_agent_factory,
        config_file=AGENT_ENTRYPOINT,
    )
