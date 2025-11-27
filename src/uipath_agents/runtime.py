from uipath.runtime import UiPathRuntimeFactoryRegistry

from uipath_agents._cli.runtime_factory import AgentRuntimeFactory


def register_runtime_factory():
    """Register Agents runtime factory."""
    UiPathRuntimeFactoryRegistry.register(
        "agents",
        factory_callable=lambda context: AgentRuntimeFactory(
            context=context,
        ),
        config_file="agent.json",
    )
