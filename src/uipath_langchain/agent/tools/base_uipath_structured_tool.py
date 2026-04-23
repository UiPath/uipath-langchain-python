from inspect import signature
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool
from langchain_core.tools.base import _get_runnable_config_param
from langchain_core.utils.pydantic import get_fields
from pydantic import BaseModel


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

    def _parse_input(
        self, tool_input: str | dict[str, Any], tool_call_id: str | None
    ) -> str | dict[str, Any]:
        """Parse and validate tool input, resolving aliased fields by Python name.

        Unlike _run/_arun, this method intentionally diverges from upstream.

        Upstream StructuredTool._parse_input builds the kwargs dict via
        getattr(validated_instance, alias). For aliases that shadow inherited
        BaseModel members (e.g. 'schema', 'copy', 'validate', 'dict', 'json'),
        this returns the inherited method instead of the aliased field value.
        Fields produced by jsonschema-pydantic-converter for reserved JSON property
        names use exactly such aliases (schema -> schema_ with alias='schema').
        """
        parsed = super()._parse_input(tool_input, tool_call_id)
        if not isinstance(parsed, dict) or not isinstance(tool_input, dict):
            return parsed

        input_args = self.args_schema
        if not (isinstance(input_args, type) and issubclass(input_args, BaseModel)):
            return parsed

        fields = get_fields(input_args)
        alias_to_name = {
            field.alias: name
            for name, field in fields.items()
            if field.alias and field.alias != name
        }
        if not alias_to_name:
            return parsed

        result = input_args.model_validate(tool_input)
        for alias, python_name in alias_to_name.items():
            if alias in parsed:
                parsed[alias] = getattr(result, python_name)
        return parsed
