from uipath._cli.middlewares import Middlewares

from uipath_agents._cli.cli_pull import agents_pull_middleware


def register_middleware() -> None:
    """Register uipath-agents middleware plugins.

    Called by the entry point system when uipath-agents is installed.
    """
    Middlewares.register("pull", agents_pull_middleware)
