"""Graph loading utilities for LangGraph JSON configuration."""

import importlib.util
import inspect
import logging
import os
import sys
from pathlib import Path
from typing import Any

from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)


class LangGraphLoader:
    """Loads a graph from a Python file path (e.g., 'agent.py:graph')."""

    def __init__(self, name: str, file_path: str, variable_name: str):
        """
        Initialize the graph loader.

        Args:
            name: Human-readable name for the graph
            file_path: Path to the Python file containing the graph
            variable_name: Name of the variable/function in the file
        """
        self.name = name
        self.file_path = file_path
        self.variable_name = variable_name
        self._context_manager: Any = None

    @classmethod
    def from_path_string(cls, name: str, path: str) -> "LangGraphLoader":
        """
        Create a GraphLoader from a path string like 'agent.py:graph'.

        Args:
            name: Human-readable name for the graph
            path: Path string in format 'file_path:variable_name'

        Returns:
            GraphLoader instance
        """
        if ":" not in path:
            raise ValueError(f"Invalid path format: {path}. Expected 'file:variable'")

        file_path, variable_name = path.split(":", 1)
        return cls(name=name, file_path=file_path, variable_name=variable_name)

    async def load(
        self,
    ) -> StateGraph[Any, Any, Any] | CompiledStateGraph[Any, Any, Any, Any]:
        """
        Load and return the graph.

        Returns:
            StateGraph or CompiledStateGraph instance

        Raises:
            ValueError: If file path is outside current directory
            FileNotFoundError: If file doesn't exist
            ImportError: If module can't be loaded
            TypeError: If loaded object isn't a valid graph
        """
        # Validate and normalize paths
        cwd = os.path.abspath(os.getcwd())
        abs_file_path = os.path.abspath(os.path.normpath(self.file_path))

        if not abs_file_path.startswith(cwd):
            raise ValueError(
                f"Graph file must be within current directory. Got: {self.file_path}"
            )

        if not os.path.exists(abs_file_path):
            raise FileNotFoundError(f"Graph file not found: {abs_file_path}")

        # Ensure current directory and src/ are in sys.path
        self._setup_python_path(cwd)

        # Import the module
        module = self._import_module(abs_file_path)

        # Get the graph object/function
        graph_obj = getattr(module, self.variable_name, None)
        if graph_obj is None:
            raise AttributeError(
                f"'{self.variable_name}' not found in {self.file_path}"
            )

        # Resolve the graph (handle functions, async functions, context managers)
        graph = await self._resolve_graph(graph_obj)

        # Validate it's a valid graph type
        if not isinstance(graph, (StateGraph, CompiledStateGraph)):
            raise TypeError(
                f"Expected StateGraph or CompiledStateGraph, got {type(graph).__name__}"
            )

        return graph

    def _setup_python_path(self, cwd: str) -> None:
        """Add current directory and src/ to Python path if needed."""
        if cwd not in sys.path:
            sys.path.insert(0, cwd)

        # Support src-layout projects (mimics editable install)
        src_dir = os.path.join(cwd, "src")
        if os.path.isdir(src_dir) and src_dir not in sys.path:
            sys.path.insert(0, src_dir)

    def _import_module(self, abs_file_path: str) -> Any:
        """Import a Python module from a file path."""
        module_name = Path(abs_file_path).stem
        spec = importlib.util.spec_from_file_location(module_name, abs_file_path)

        if not spec or not spec.loader:
            raise ImportError(f"Could not load module from: {abs_file_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        return module

    async def _resolve_graph(
        self, graph_obj: Any
    ) -> StateGraph[Any, Any, Any] | CompiledStateGraph[Any, Any, Any, Any]:
        """
        Resolve a graph object that might be:
        - A direct StateGraph/CompiledStateGraph
        - A function that returns a graph
        - An async function that returns a graph
        - An async context manager that yields a graph
        """
        # Handle callable (function or async function)
        if callable(graph_obj):
            if inspect.iscoroutinefunction(graph_obj):
                graph_obj = await graph_obj()
            else:
                graph_obj = graph_obj()

        # Handle async context manager
        if hasattr(graph_obj, "__aenter__") and callable(graph_obj.__aenter__):
            self._context_manager = graph_obj
            return await graph_obj.__aenter__()

        return graph_obj

    async def cleanup(self) -> None:
        """Clean up resources (e.g., exit async context managers)."""
        if self._context_manager:
            try:
                await self._context_manager.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error during graph cleanup: {e}")
            finally:
                self._context_manager = None
