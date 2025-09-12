from typing import Any, Optional, Union
from uuid import uuid4

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph
from uipath._cli._runtime._contracts import UiPathRuntimeContext, UiPathTraceContext

from .._utils._graph import LangGraphConfig


class LangGraphRuntimeContext(UiPathRuntimeContext):
    """Context information passed throughout the runtime execution."""

    langgraph_config: Optional[LangGraphConfig] = None
    state_graph: Optional[StateGraph[Any, Any]] = None
    output: Optional[Any] = None
    state: Optional[Any] = (
        None  # TypedDict issue, the actual type is: Optional[langgraph.types.StateSnapshot]
    )
    memory: Optional[AsyncSqliteSaver] = None
    langsmith_tracing_enabled: Union[str, bool, None] = False
    resume_triggers_table: str = "__uipath_resume_triggers"


class LangGraphRuntimeContextBuilder:
    """Builder class for LangGraphRuntimeContext following the builder pattern."""

    def __init__(self):
        self._kwargs = {}

    def with_defaults(
        self, config_path: Optional[str] = None, **kwargs
    ) -> "LangGraphRuntimeContextBuilder":
        """Apply default configuration similar to LangGraphRuntimeContext.from_config().

        Args:
            config_path: Path to the configuration file (defaults to UIPATH_CONFIG_PATH env var or "uipath.json")
            **kwargs: Additional keyword arguments to pass to with_defaults

        Returns:
            Self for method chaining
        """
        from os import environ as env

        resolved_config_path = config_path or env.get(
            "UIPATH_CONFIG_PATH", "uipath.json"
        )
        self._kwargs["config_path"] = resolved_config_path

        bool_map = {"true": True, "false": False}
        tracing = env.get("UIPATH_TRACING_ENABLED", True)
        if isinstance(tracing, str) and tracing.lower() in bool_map:
            tracing = bool_map[tracing.lower()]

        self._kwargs.update(
            {
                "job_id": env.get("UIPATH_JOB_KEY"),
                "execution_id": env.get("UIPATH_JOB_KEY"),
                "trace_id": env.get("UIPATH_TRACE_ID"),
                "tracing_enabled": tracing,
                "logs_min_level": env.get("LOG_LEVEL", "INFO"),
                "langsmith_tracing_enabled": env.get("LANGSMITH_TRACING", False),
                **kwargs,  # Allow overriding defaults with provided kwargs
            }
        )

        self._kwargs["trace_context"] = UiPathTraceContext(
            trace_id=env.get("UIPATH_TRACE_ID"),
            parent_span_id=env.get("UIPATH_PARENT_SPAN_ID"),
            root_span_id=env.get("UIPATH_ROOT_SPAN_ID"),
            enabled=tracing,
            job_id=env.get("UIPATH_JOB_KEY"),
            org_id=env.get("UIPATH_ORGANIZATION_ID"),
            tenant_id=env.get("UIPATH_TENANT_ID"),
            process_key=env.get("UIPATH_PROCESS_UUID"),
            folder_key=env.get("UIPATH_FOLDER_KEY"),
            reference_id=env.get("UIPATH_JOB_KEY") or str(uuid4()),
        )

        return self

    def with_entrypoint(
        self, entrypoint: Optional[str]
    ) -> "LangGraphRuntimeContextBuilder":
        """Set the entrypoint for the runtime context.

        Args:
            entrypoint: The entrypoint to execute

        Returns:
            Self for method chaining
        """
        if entrypoint is not None:
            self._kwargs["entrypoint"] = entrypoint
        return self

    def with_input(
        self, input_data: Optional[str] = None, input_file: Optional[str] = None
    ) -> "LangGraphRuntimeContextBuilder":
        """Set the input data for the runtime context.

        Args:
            input_data: The input data as a string
            input_file: Path to the input file

        Returns:
            Self for method chaining
        """
        if input_data is not None:
            self._kwargs["input"] = input_data
        if input_file is not None:
            self._kwargs["input_file"] = input_file
        return self

    def with_resume(self, enable: bool = True) -> "LangGraphRuntimeContextBuilder":
        """Enable or disable resume mode for the runtime context.

        Args:
            enable: Whether to enable resume mode (defaults to True)

        Returns:
            Self for method chaining
        """
        self._kwargs["resume"] = enable
        return self

    def with_langgraph_config(
        self, config: LangGraphConfig
    ) -> "LangGraphRuntimeContextBuilder":
        """Set the LangGraph configuration.

        Args:
            config: The LangGraph configuration

        Returns:
            Self for method chaining
        """
        self._kwargs["langgraph_config"] = config
        return self

    def mark_eval_run(self, enable: bool = True) -> "LangGraphRuntimeContextBuilder":
        """Mark this as an evaluation run.

        Args:
            enable: Whether this is an eval run (defaults to True)

        Returns:
            Self for method chaining
        """
        self._kwargs["is_eval_run"] = enable
        return self

    def build(self) -> "LangGraphRuntimeContext":
        """Build and return the LangGraphRuntimeContext instance.

        Returns:
            A configured LangGraphRuntimeContext instance
        """
        config_path = self._kwargs.pop("config_path", None)
        if config_path:
            # Create context from config first, then update with any additional kwargs
            context = LangGraphRuntimeContext.from_config(config_path)
            for key, value in self._kwargs.items():
                if hasattr(context, key):
                    setattr(context, key, value)
            return context
        else:
            return LangGraphRuntimeContext(**self._kwargs)
