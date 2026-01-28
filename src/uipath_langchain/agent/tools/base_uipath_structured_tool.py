from inspect import signature
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langchain_core.tools.base import _get_runnable_config_param


class BaseUiPathStructuredTool(StructuredTool):
    """Base class for UiPath structured tools.

    Extends LangChain's StructuredTool to override the _run and _arun methods.
    The only difference is that the self reference variable is renamed, to avoid conflicts with payload keys.

    DO NOT CHANGE ANYTHING IN THESE METHODS.
    There are tests that verify the implementations against the upstream LangChain implementations.

    """

    def _run(
        __obj_internal_self__,
        *args: Any,
        config: RunnableConfig,
        run_manager: CallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> Any:
        """Use the tool.

        Args:
            *args: Positional arguments to pass to the tool
            config: Configuration for the run
            run_manager: Optional callback manager to use for the run
            **kwargs: Keyword arguments to pass to the tool

        Returns:
            The result of the tool execution
        """
        if __obj_internal_self__.func:
            if run_manager and signature(__obj_internal_self__.func).parameters.get(
                "callbacks"
            ):
                kwargs["callbacks"] = run_manager.get_child()
            if config_param := _get_runnable_config_param(__obj_internal_self__.func):
                kwargs[config_param] = config
            return __obj_internal_self__.func(*args, **kwargs)
        msg = "StructuredTool does not support sync invocation."
        raise NotImplementedError(msg)

    async def _arun(
        __obj_internal_self__,
        *args: Any,
        config: RunnableConfig,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> Any:
        """Use the tool asynchronously.

        Args:
            *args: Positional arguments to pass to the tool
            config: Configuration for the run
            run_manager: Optional callback manager to use for the run
            **kwargs: Keyword arguments to pass to the tool

        Returns:
            The result of the tool execution
        """
        if __obj_internal_self__.coroutine:
            if run_manager and signature(
                __obj_internal_self__.coroutine
            ).parameters.get("callbacks"):
                kwargs["callbacks"] = run_manager.get_child()
            if config_param := _get_runnable_config_param(
                __obj_internal_self__.coroutine
            ):
                kwargs[config_param] = config
            return await __obj_internal_self__.coroutine(*args, **kwargs)

        # If self.coroutine is None, then this will delegate to the default
        # implementation which is expected to delegate to _run on a separate thread.
        return await super()._arun(
            *args, config=config, run_manager=run_manager, **kwargs
        )
