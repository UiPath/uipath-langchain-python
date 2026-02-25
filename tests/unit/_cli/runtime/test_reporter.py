"""Tests for ReporterRuntime."""

import pytest

from uipath_agents._cli.runtime.reporter import ReporterRuntime


class TestReporterRuntimeExecute:
    """ReporterRuntime.execute() re-raises the stored error."""

    @pytest.mark.asyncio
    async def test_raises_stored_error(self) -> None:
        error = RuntimeError("graph build failed")
        runtime = ReporterRuntime(error)

        with pytest.raises(RuntimeError, match="graph build failed"):
            await runtime.execute()


class TestReporterRuntimeStream:
    """ReporterRuntime.stream() re-raises the stored error."""

    @pytest.mark.asyncio
    async def test_raises_on_iteration(self) -> None:
        error = RuntimeError("startup failure")
        runtime = ReporterRuntime(error)

        with pytest.raises(RuntimeError, match="startup failure"):
            async for _ in runtime.stream():
                pass
