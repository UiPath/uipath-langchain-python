"""Unit tests for conversational licensing functions."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from uipath_agents._services.licensing_service import (
    register_conversational_licensing_async,
)


class TestRegisterConversationalLicensingAsync:
    """Tests for the register_conversational_licensing_async helper."""

    def _make_agent_definition(
        self, model: str = "gpt-4o", is_conversational: bool = True
    ) -> MagicMock:
        agent_def = MagicMock()
        agent_def.settings.model = model
        agent_def.is_conversational = is_conversational
        return agent_def

    @pytest.mark.asyncio
    async def test_skips_when_agent_definition_is_none(self) -> None:
        mock_service_cls = MagicMock()
        with patch(
            "uipath_agents._services.licensing_service.LicensingService",
            mock_service_cls,
        ):
            await register_conversational_licensing_async(
                None,
                agenthub_config="conversationalruntime",
            )
        mock_service_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_model_name_is_none(self) -> None:
        agent_def = self._make_agent_definition()
        agent_def.settings.model = None
        mock_service_cls = MagicMock()
        with patch(
            "uipath_agents._services.licensing_service.LicensingService",
            mock_service_cls,
        ):
            await register_conversational_licensing_async(
                agent_def,
                agenthub_config="conversationalruntime",
            )
        mock_service_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_registers_consumption(self) -> None:
        agent_def = self._make_agent_definition()
        mock_service = AsyncMock()
        with (
            patch("uipath.platform.UiPath") as mock_uipath_cls,
            patch(
                "uipath_agents._services.licensing_service.LicensingService",
                return_value=mock_service,
            ),
        ):
            mock_uipath_cls.return_value = MagicMock()
            await register_conversational_licensing_async(
                agent_def,
                agenthub_config="conversationalruntime",
            )
        mock_service.register_conversational_consumption_async.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_passes_all_params_to_service(self) -> None:
        agent_def = self._make_agent_definition()
        mock_service = AsyncMock()
        with (
            patch("uipath.platform.UiPath") as mock_uipath_cls,
            patch(
                "uipath_agents._services.licensing_service.LicensingService",
                return_value=mock_service,
            ),
        ):
            mock_uipath_cls.return_value = MagicMock()
            await register_conversational_licensing_async(
                agent_def,
                agenthub_config="conversationalplayground",
                is_byo_execution=True,
                job_key="test-job-key",
            )
        mock_service.register_conversational_consumption_async.assert_awaited_once_with(
            "gpt-4o",
            agenthub_config="conversationalplayground",
            is_byo_execution=True,
            job_key="test-job-key",
        )

    @pytest.mark.asyncio
    async def test_defaults_byo_to_false(self) -> None:
        agent_def = self._make_agent_definition()
        mock_service = AsyncMock()
        with (
            patch("uipath.platform.UiPath") as mock_uipath_cls,
            patch(
                "uipath_agents._services.licensing_service.LicensingService",
                return_value=mock_service,
            ),
        ):
            mock_uipath_cls.return_value = MagicMock()
            await register_conversational_licensing_async(
                agent_def,
                agenthub_config="conversationalruntime",
            )
        mock_service.register_conversational_consumption_async.assert_awaited_once_with(
            "gpt-4o",
            agenthub_config="conversationalruntime",
            is_byo_execution=False,
            job_key=None,
        )

    @pytest.mark.asyncio
    async def test_silently_catches_exceptions(self) -> None:
        agent_def = self._make_agent_definition()
        with patch(
            "uipath.platform.UiPath",
            side_effect=RuntimeError("connection failed"),
        ):
            await register_conversational_licensing_async(
                agent_def,
                agenthub_config="conversationalruntime",
            )
