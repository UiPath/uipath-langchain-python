from uipath._cli.middlewares import Middlewares

from uipath_lowcode._cli.cli_run import lowcode_run_middleware


def register_middleware():
    """This function will be called by the entry point system when uipath_lowcode is installed"""
    Middlewares.register("run", lowcode_run_middleware)
