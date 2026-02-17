from typing import Any, AsyncGenerator

from uipath.agent.models.agent import AgentDefinition
from uipath.runtime import (
    UiPathExecuteOptions,
    UiPathRuntimeEvent,
    UiPathRuntimeResult,
    UiPathStreamOptions,
)
from uipath.runtime.errors import UiPathBaseRuntimeError
from uipath.runtime.schema import UiPathRuntimeSchema
from uipath_langchain.runtime import UiPathLangGraphRuntime

from uipath_agents._errors import ExceptionMapper

from .utils import validate_json_against_json_schema


class AgentsLangGraphRuntime(UiPathLangGraphRuntime):
    """Agent runtime that processes agent.json instead of graph introspection."""

    def __init__(
        self,
        *args,
        agent_definition: AgentDefinition,
        **kwargs,
    ):
        """Initialize the runtime.

        Args:
            agent_definition: Pre-loaded agent configuration
            *args: Passed to parent
            **kwargs: Passed to parent
        """
        super().__init__(*args, **kwargs)
        self._agent_definition = agent_definition

    async def execute(
        self,
        input: dict[str, Any] | None = None,
        options: UiPathExecuteOptions | None = None,
    ) -> UiPathRuntimeResult:
        """Execute the agent runtime.

        Args:
            input: Input data for the runtime
            options: Execution options
        Returns:
            The runtime result
        """
        if not options or not options.resume:
            validate_json_against_json_schema(
                self._agent_definition.input_schema, input
            )
        return await super().execute(input, options)

    async def stream(
        self,
        input: dict[str, Any] | None = None,
        options: UiPathStreamOptions | None = None,
    ) -> AsyncGenerator[UiPathRuntimeEvent, None]:
        """Stream the agent runtime execution.

        Args:
            input: Input data for the runtime
            options: Streaming options
        Yields:
            Runtime events as they occur
        """
        if not options or not options.resume:
            validate_json_against_json_schema(
                self._agent_definition.input_schema, input
            )
        async for event in super().stream(input, options):
            yield event

    def get_agent_model(self) -> str | None:
        """Get the agent's configured LLM model.

        Returns:
            The model name (e.g., 'gpt-4o-2024-11-20'), or None if not found.
        """
        if self._agent_definition.settings and self._agent_definition.settings.model:
            return self._agent_definition.settings.model
        return None

    async def get_schema(self) -> UiPathRuntimeSchema:
        """Return the runtime schema from the pre-loaded agent definition.

        The schema includes agent settings (model, temperature, etc.) in the
        metadata field, allowing evaluation tools to read and override them.
        """
        if self.entrypoint is None:
            raise ValueError("Agent runtime requires an entrypoint to be set")

        # Include agent settings in metadata for eval tools to access/override
        metadata = None
        if self._agent_definition.settings:
            metadata = {
                "settings": self._agent_definition.settings.model_dump(
                    exclude_none=True
                )
            }

        return UiPathRuntimeSchema(
            filePath=self.entrypoint,
            uniqueId=self.runtime_id,
            type="agent",
            input=self._agent_definition.input_schema or {},
            output=self._agent_definition.output_schema or {},
            metadata=metadata,
        )

    def create_runtime_error(self, e: Exception) -> UiPathBaseRuntimeError:
        """Handle execution errors using ExceptionTranslator for all exceptions.

        Completely overrides parent implementation to use ExceptionTranslator
        for proper classification and user-actionable error messages.
        """
        mapped_exc = ExceptionMapper.map_runtime(e)
        return mapped_exc
