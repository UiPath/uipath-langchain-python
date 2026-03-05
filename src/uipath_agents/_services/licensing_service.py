"""Service for notifying AgentHub of agent licensing."""

import logging

from uipath._utils import Endpoint, RequestSpec
from uipath.agent.models.agent import AgentDefinition
from uipath.platform.common import BaseService

logger = logging.getLogger("uipath")


class LicensingService(BaseService):
    """Service for notifying AgentHub of licensing at agent start."""

    async def register_consumption_async(
        self, model_name: str, *, job_key: str | None = None
    ) -> None:
        """Register licensing consumption for the given model."""
        headers: dict[str, str] = {}
        if job_key:
            headers["X-UiPath-JobKey"] = job_key

        spec = RequestSpec(
            method="POST",
            endpoint=Endpoint("/agenthub_/llm/api/execution-cost-tmp"),
            params={"modelName": model_name},
            headers=headers,
        )

        response = await self.request_async(
            spec.method,
            url=spec.endpoint,
            params=spec.params,
            headers=spec.headers,
        )

        if not response.is_success:
            logger.warning(
                "Licensing registration failed: %s %s",
                response.status_code,
                response.text,
            )

    async def register_conversational_consumption_async(
        self,
        model_name: str,
        *,
        agenthub_config: str,
        is_byo_execution: bool = False,
        job_key: str | None = None,
    ) -> None:
        """Register conversational licensing consumption for the given model."""
        headers: dict[str, str] = {"X-UiPath-AgentHub-Config": agenthub_config}
        if job_key:
            headers["X-UiPath-JobKey"] = job_key

        params: dict[str, str] = {"modelName": model_name, "messages": "1"}
        if is_byo_execution:
            params["isByoExecution"] = "true"

        spec = RequestSpec(
            method="POST",
            endpoint=Endpoint("/agenthub_/llm/api/conversational-consumption"),
            params=params,
            headers=headers,
        )

        response = await self.request_async(
            spec.method,
            url=spec.endpoint,
            params=spec.params,
            headers=spec.headers,
        )

        if not response.is_success:
            logger.warning(
                "Conversational licensing registration failed: %s %s",
                response.status_code,
                response.text,
            )


def _create_licensing_service() -> LicensingService:
    from uipath.platform import UiPath

    uipath = UiPath()
    return LicensingService(
        config=uipath._config,
        execution_context=uipath._execution_context,
    )


async def register_licensing_async(
    agent_definition: AgentDefinition | None,
    job_key: str | None = None,
) -> None:
    """Register licensing consumption for a fresh agent start.

    Silently catches all exceptions so it never blocks agent execution.
    """
    try:
        if not agent_definition:
            return
        model_name = agent_definition.settings.model
        if not model_name:
            return

        service = _create_licensing_service()
        await service.register_consumption_async(model_name, job_key=job_key)
    except Exception:
        logger.debug("Failed to register licensing", exc_info=True)


async def register_conversational_licensing_async(
    agent_definition: AgentDefinition | None,
    *,
    agenthub_config: str,
    is_byo_execution: bool = False,
    job_key: str | None = None,
) -> None:
    """Register conversational licensing consumption for a completed exchange.

    Silently catches all exceptions so it never blocks agent execution.
    """
    try:
        if not agent_definition:
            return
        model_name = agent_definition.settings.model
        if not model_name:
            return

        service = _create_licensing_service()
        await service.register_conversational_consumption_async(
            model_name,
            agenthub_config=agenthub_config,
            is_byo_execution=is_byo_execution,
            job_key=job_key,
        )
    except Exception:
        logger.debug("Failed to register conversational licensing", exc_info=True)
