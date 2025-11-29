from pathlib import Path

from uipath.runtime.schema import UiPathRuntimeSchema
from uipath_langchain.runtime.runtime import UiPathLangGraphRuntime

from ..utils import load_agent_configuration


class AgentsLangGraphRuntime(UiPathLangGraphRuntime):
    """Agent runtime that processes agent.json instead of graph introspection."""

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
