"""Service for notifying AgentHub of agent licensing."""

import logging

from uipath._utils import Endpoint, RequestSpec
from uipath.agent.models.agent import AgentDefinition
from uipath.platform.common import BaseService, UiPathApiConfig, UiPathExecutionContext

logger = logging.getLogger("uipath")


class LicensingService(BaseService):
    """Service for notifying AgentHub of licensing at agent start."""

    def __init__(
        self, config: UiPathApiConfig, execution_context: UiPathExecutionContext
    ) -> None:
        super().__init__(config=config, execution_context=execution_context)

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

        from uipath.platform import UiPath

        uipath = UiPath()
        service = LicensingService(
            config=uipath._config,
            execution_context=uipath._execution_context,
        )
        await service.register_consumption_async(str(model_name), job_key=job_key)
    except Exception:
        logger.debug("Failed to register licensing", exc_info=True)
