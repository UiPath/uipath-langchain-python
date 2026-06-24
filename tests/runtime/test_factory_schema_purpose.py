"""Tests for the `purpose="schema"` branch of `UiPathLangGraphRuntimeFactory.new_runtime`.

Regression coverage for the SQLite "database is locked" failure mode where
the LangGraph factory opened `__uipath/state.db` on every `new_runtime` call
— including schema-only paths used by `uipath init` — causing concurrent
init runs to race the checkpoint setup.
"""

import os
import tempfile
from typing import Any
from unittest.mock import patch

import pytest
from uipath.runtime import UiPathRuntimeContext

from uipath_langchain.runtime.factory import UiPathLangGraphRuntimeFactory


def _build_graph_module(tmp_dir: str) -> None:
    """Write a minimal langgraph.json + graph module into ``tmp_dir``."""
    (open(os.path.join(tmp_dir, "agent.py"), "w")).write(
        "from langgraph.graph import END, START, StateGraph\n"
        "from pydantic import BaseModel\n"
        "\n"
        "class _Input(BaseModel):\n"
        "    name: str = ''\n"
        "\n"
        "class _Output(BaseModel):\n"
        "    greeting: str = ''\n"
        "\n"
        "def node(state):\n"
        "    return {'greeting': f'hello {state.name}'}\n"
        "\n"
        "builder = StateGraph(_Input, input_schema=_Input, output_schema=_Output)\n"
        "builder.add_node('node', node)\n"
        "builder.add_edge(START, 'node')\n"
        "builder.add_edge('node', END)\n"
        "graph = builder.compile()\n"
    )
    (open(os.path.join(tmp_dir, "langgraph.json"), "w")).write(
        '{"graphs": {"agent": "./agent.py:graph"}}\n'
    )


@pytest.mark.asyncio
async def test_schema_purpose_skips_sqlite_checkpointer() -> None:
    """`purpose="schema"` must not open `__uipath/state.db`."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        prev_cwd = os.getcwd()
        os.chdir(tmp_dir)
        try:
            _build_graph_module(tmp_dir)
            context = UiPathRuntimeContext.with_defaults(entrypoint="agent")
            factory = UiPathLangGraphRuntimeFactory(context=context)

            with patch.object(
                factory, "_get_memory", side_effect=AssertionError("must not open")
            ):
                runtime = await factory.new_runtime(
                    "agent", runtime_id="schema-only", purpose="schema"
                )
                schema = await runtime.get_schema()
                await runtime.dispose()
                await factory.dispose()

            assert schema.input is not None
            assert schema.output is not None
            assert not os.path.exists(os.path.join(tmp_dir, "__uipath", "state.db")), (
                "state.db must not be created on the schema-only path"
            )
        finally:
            os.chdir(prev_cwd)


@pytest.mark.asyncio
async def test_default_purpose_still_opens_sqlite_checkpointer() -> None:
    """Backwards compatibility: omitting `purpose` keeps the existing behaviour."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        prev_cwd = os.getcwd()
        os.chdir(tmp_dir)
        try:
            _build_graph_module(tmp_dir)
            context = UiPathRuntimeContext.with_defaults(entrypoint="agent")
            factory = UiPathLangGraphRuntimeFactory(context=context)

            calls: list[Any] = []
            original_get_memory = factory._get_memory

            async def tracking_get_memory():
                calls.append(True)
                return await original_get_memory()

            with patch.object(factory, "_get_memory", side_effect=tracking_get_memory):
                runtime = await factory.new_runtime("agent", runtime_id="exec-default")
                await runtime.dispose()
                await factory.dispose()

            assert calls, "execute-purpose path must still call _get_memory"
        finally:
            os.chdir(prev_cwd)


@pytest.mark.asyncio
async def test_schema_purpose_does_not_pollute_graph_cache() -> None:
    """Schema-purpose compilations (no checkpointer) must not leak into the cache."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        prev_cwd = os.getcwd()
        os.chdir(tmp_dir)
        try:
            _build_graph_module(tmp_dir)
            context = UiPathRuntimeContext.with_defaults(entrypoint="agent")
            factory = UiPathLangGraphRuntimeFactory(context=context)

            runtime = await factory.new_runtime(
                "agent", runtime_id="schema-only", purpose="schema"
            )
            await runtime.get_schema()
            await runtime.dispose()

            assert "agent" not in factory._graph_cache, (
                "schema-purpose compilation must not be cached, since it was "
                "built without a checkpointer and is not interchangeable with "
                "the execute-purpose compilation"
            )
            await factory.dispose()
        finally:
            os.chdir(prev_cwd)
