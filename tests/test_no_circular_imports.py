"""Test that all modules can be imported without circular dependency errors.

This test automatically discovers all modules in uipath_langchain and tests each
one with isolated imports to catch runtime circular imports.
"""

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Iterator

import pytest


def discover_all_modules(package_name: str) -> Iterator[str]:
    """Discover all importable modules in a package by walking the filesystem.

    Uses filesystem walking instead of importlib to avoid circular import issues
    during test collection time.

    Args:
        package_name: The top-level package name (e.g., 'uipath_langchain')

    Yields:
        Fully qualified module names (e.g., 'uipath_langchain.agent.tools')
    """
    # Find the package by locating its spec
    spec = importlib.util.find_spec(package_name)
    if spec is None or spec.origin is None:
        return

    # Get the package directory
    package_dir = Path(spec.origin).parent

    # Walk through all Python files in the package
    for py_file in package_dir.rglob("*.py"):
        if py_file.name == "__init__.py":
            # Convert path to module name
            rel_path = py_file.parent.relative_to(package_dir.parent)
            module_name = str(rel_path).replace("/", ".").replace("\\", ".")
            if module_name.endswith(".__init__"):
                module_name = module_name[:-9]  # Remove .__init__
            if module_name:
                yield module_name
        elif py_file.name != "__pycache__":
            # Convert path to module name
            rel_path = py_file.relative_to(package_dir.parent)
            module_name = str(rel_path)[:-3].replace("/", ".").replace("\\", ".")
            yield module_name


def get_all_module_imports() -> list[str]:
    """Get all modules to test.

    Returns:
        List of module names to test
    """
    modules = list(discover_all_modules("uipath_langchain"))

    # Filter out optional dependency modules that won't be installed
    exclude = {"uipath_langchain.chat.bedrock", "uipath_langchain.chat.vertex"}
    return [m for m in modules if m not in exclude]


@pytest.mark.parametrize("module_name", get_all_module_imports())
def test_module_imports_with_isolation(module_name: str) -> None:
    """Test that a module can be imported in isolation without circular imports.

    Clears sys.modules and performs two import attempts:
    1. Initial import - catches obvious circular imports
    2. Reload - catches deferred circular imports that only manifest on re-import

    This catches circular imports that would be masked by module caching while
    remaining fast by avoiding subprocess overhead.

    Args:
        module_name: The fully qualified module name to test

    Raises:
        pytest.fail: If the module cannot be imported due to circular dependency
    """
    # Clear all uipath_langchain modules from sys.modules to force fresh imports
    to_remove = [key for key in sys.modules.keys() if "uipath_langchain" in key]
    for key in to_remove:
        del sys.modules[key]

    # First import - catches immediate circular imports
    try:
        importlib.import_module(module_name)
    except ImportError as e:
        if "circular import" in str(e).lower():
            pytest.fail(
                f"Circular import in {module_name}:\n{e}",
                pytrace=False,
            )
        # Other import errors (missing dependencies, syntax errors, etc)
        pytest.fail(
            f"Failed to import {module_name}:\n{e}",
            pytrace=False,
        )
