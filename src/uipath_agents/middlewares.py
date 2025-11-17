from uipath._cli.middlewares import Middlewares

from uipath_agents._cli.cli_debug import lowcode_debug_middleware
from uipath_agents._cli.cli_dev import lowcode_dev_middleware
from uipath_agents._cli.cli_run import lowcode_run_middleware


def register_middleware():
    """This function will be called by the entry point system when uipath_agents is installed"""
    Middlewares.register("run", lowcode_run_middleware)
    Middlewares.register("debug", lowcode_debug_middleware)
    Middlewares.register("dev", lowcode_dev_middleware)
