"""Simple loader for langgraph.json configuration."""

import json
import os


class LangGraphConfig:
    """Simple loader for langgraph.json configuration."""

    def __init__(self, config_path: str = "langgraph.json"):
        """
        Initialize configuration loader.

        Args:
            config_path: Path to langgraph.json file
        """
        self.config_path = config_path
        self._graphs: dict[str, str] | None = None

    @property
    def exists(self) -> bool:
        """Check if langgraph.json exists."""
        return os.path.exists(self.config_path)

    @property
    def graphs(self) -> dict[str, str]:
        """
        Get graph name -> path mapping from config.

        Returns:
            Dictionary mapping graph names to file paths (e.g., {"agent": "agent.py:graph"})
        """
        if self._graphs is None:
            self._graphs = self._load_graphs()
        return self._graphs

    def _load_graphs(self) -> dict[str, str]:
        """Load graph definitions from langgraph.json."""
        if not self.exists:
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)

            if "graphs" not in config:
                raise ValueError("Missing required 'graphs' field in langgraph.json")

            graphs = config["graphs"]
            if not isinstance(graphs, dict):
                raise ValueError("'graphs' must be a dictionary")

            return graphs

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {self.config_path}: {e}") from e

    @property
    def entrypoints(self) -> list[str]:
        """Get list of available graph entrypoints."""
        return list(self.graphs.keys())
