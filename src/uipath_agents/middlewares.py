from uipath._cli.middlewares import Middlewares

from uipath_agents._cli.cli_debug import agents_debug_middleware
from uipath_agents._cli.cli_run import agents_run_middleware


def register_middleware():
    """This function will be called by the entry point system when uipath_agents is installed"""
    Middlewares.register("run", agents_run_middleware)
    Middlewares.register("debug", agents_debug_middleware)
