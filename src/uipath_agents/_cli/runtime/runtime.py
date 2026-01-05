from pathlib import Path

from uipath.runtime.schema import UiPathRuntimeSchema
from uipath_langchain.runtime.runtime import UiPathLangGraphRuntime

from ..utils import load_agent_configuration


class AgentsLangGraphRuntime(UiPathLangGraphRuntime):
    """Agent runtime that processes agent.json instead of graph introspection.

    Implements LLMAgentRuntimeProtocol to provide agent model information
    for features like 'same-as-agent' model resolution in evaluators.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the runtime."""
        super().__init__(*args, **kwargs)
        self._agent_model: str | None = None
        self._agent_model_loaded: bool = False

    def get_agent_model(self) -> str | None:
        """Get the agent's configured LLM model.

        Implements LLMAgentRuntimeProtocol. Loads the agent configuration
        from agent.json if not already loaded and returns the model setting.

        Returns:
            The model name (e.g., 'gpt-4o-2024-11-20'), or None if not found.
        """
        if not self._agent_model_loaded:
            self._agent_model_loaded = True
            try:
                if self.entrypoint is None:
                    self._agent_model = None
                else:
                    agent_json_path = Path.cwd() / self.entrypoint
                    agent_definition = load_agent_configuration(agent_json_path)
                    self._agent_model = agent_definition.settings.model
            except Exception:
                self._agent_model = None
        return self._agent_model

    async def get_schema(self) -> UiPathRuntimeSchema:
        """Return the runtime schema from agent.json configuration."""
        if self.entrypoint is None:
            raise ValueError("Agent runtime requires an entrypoint to be set")

        agent_json_path = Path.cwd() / self.entrypoint
        agent_definition = load_agent_configuration(agent_json_path)

        return UiPathRuntimeSchema(
            filePath=self.entrypoint,
            uniqueId=self.runtime_id,
            type="agent",
            input=agent_definition.input_schema or {},
            output=agent_definition.output_schema or {},
        )
