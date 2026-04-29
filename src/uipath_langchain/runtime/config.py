"""Simple loader for langgraph.json configuration."""

import json
import os
from typing import Any


class LangGraphConfig:
    """Simple loader for langgraph.json configuration."""

    def __init__(self, config_path: str = "langgraph.json"):
        """
        Initialize configuration loader.

        Args:
            config_path: Path to langgraph.json file
        """
        self.config_path = config_path
        self._raw: dict[str, Any] | None = None

    @property
    def exists(self) -> bool:
        """Check if langgraph.json exists."""
        return os.path.exists(self.config_path)

    def _load(self) -> dict[str, Any]:
        if self._raw is not None:
            return self._raw
        if not self.exists:
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        try:
            with open(self.config_path, "r") as f:
                self._raw = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {self.config_path}: {e}") from e
        return self._raw

    @property
    def graphs(self) -> dict[str, str]:
        """
        Get graph name -> path mapping from config.

        Returns:
            Dictionary mapping graph names to file paths (e.g., {"agent": "agent.py:graph"})
        """
        config = self._load()
        if "graphs" not in config:
            raise ValueError("Missing required 'graphs' field in langgraph.json")
        graphs = config["graphs"]
        if not isinstance(graphs, dict):
            raise ValueError("'graphs' must be a dictionary")
        return graphs

    @property
    def allowed_msgpack_modules(self) -> list[tuple[str, str]] | None:
        """Read `checkpointer.serde.allowed_msgpack_modules` from langgraph.json."""
        config = self._load()
        checkpointer = config.get("checkpointer")
        if not isinstance(checkpointer, dict):
            return None
        serde = checkpointer.get("serde")
        if not isinstance(serde, dict):
            return None
        modules = serde.get("allowed_msgpack_modules")
        if modules is None:
            return None
        if not isinstance(modules, list):
            raise ValueError(
                "'checkpointer.serde.allowed_msgpack_modules' must be a list "
                "of [module, class_name] pairs"
            )
        result: list[tuple[str, str]] = []
        for entry in modules:
            if (
                not isinstance(entry, list)
                or len(entry) != 2
                or not all(isinstance(part, str) for part in entry)
            ):
                raise ValueError(
                    f"Invalid entry in checkpointer.serde.allowed_msgpack_modules: "
                    f"{entry!r} (expected [module, class_name])"
                )
            result.append((entry[0], entry[1]))
        return result

    @property
    def entrypoints(self) -> list[str]:
        """Get list of available graph entrypoints."""
        return list(self.graphs.keys())
