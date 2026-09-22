"""Jev classification tool.

Jev (TypeSafe AI) is a classification model, not an LLM: it evaluates typed
questions (``choice``, ``score``, ``noul``) against a ``state`` and returns
calibrated answers. The questions are authored in Agent Builder; the agent's
LLM only supplies the ``state`` to classify.
"""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import StructuredTool
from pydantic import ValidationError
from uipath.agent.models.agent import AgentGenericToolResourceConfig
from uipath.core.feature_flags import FeatureFlags
from uipath.eval.mocks import mockable
from uipath.llm_client.settings import UiPathBaseSettings
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
    AgentStartupError,
    AgentStartupErrorCode,
)
from uipath_langchain.agent.react.jsonschema_pydantic_converter import (
    create_model,
    create_output_model,
)
from uipath_langchain.agent.tools.structured_tool_with_argument_properties import (
    StructuredToolWithArgumentProperties,
)
from uipath_langchain.agent.tools.utils import sanitize_tool_name

from .jev_client import JevClient, JevClientConfig
from .jev_settings import JEV_INPUT_SCHEMA, JevToolSettings

JEV_SUB_TYPE = "jev"
JEV_TOOL_FF = "EnableJevTool"


def create_jev_tool(
    resource: AgentGenericToolResourceConfig,
    llm: BaseChatModel,
    client: JevClient | None = None,
) -> StructuredTool:
    """Create the Jev classification tool from a generic tool resource.

    Args:
        resource: Generic tool resource with ``sub_type == "jev"``.
        llm: The agent's chat model. Its UiPath client settings are reused
            when Jev is routed via LLM Gateway.
        client: Jev client override. Resolved from the environment on the
            first call when omitted, so mocked evaluations need no API key.

    Raises:
        AgentStartupError: If the feature is disabled or the settings are invalid.
    """
    if not FeatureFlags.is_flag_enabled(JEV_TOOL_FF, default=False):
        raise AgentStartupError(
            code=AgentStartupErrorCode.INVALID_TOOL_CONFIG,
            title="Jev tool is not enabled",
            detail=f"Tool '{resource.name}' uses Jev, which is not enabled "
            f"(feature flag '{JEV_TOOL_FF}').",
            category=UiPathErrorCategory.USER,
        )

    try:
        settings = JevToolSettings.model_validate(resource.properties.settings)
    except ValidationError as e:
        raise AgentStartupError(
            code=AgentStartupErrorCode.INVALID_TOOL_CONFIG,
            title="Invalid Jev tool settings",
            detail=f"Tool '{resource.name}' has invalid Jev settings: {e}",
            category=UiPathErrorCategory.USER,
        ) from e

    tool_name = sanitize_tool_name(resource.name)
    input_model = create_model(JEV_INPUT_SCHEMA)
    output_model = create_output_model(settings.output_schema(), resource.name)
    api_questions = settings.api_questions()
    gateway_settings = _get_uipath_client_settings(llm)
    resolved_client = client

    @mockable(
        name=resource.name,
        description=resource.description,
        input_schema=input_model.model_json_schema(),
        output_schema=output_model.model_json_schema(),
        example_calls=[],
    )
    async def jev_tool_fn(**kwargs: Any) -> dict[str, Any]:
        nonlocal resolved_client
        if resolved_client is None:
            resolved_client = JevClient(
                JevClientConfig.from_environment(), gateway_settings=gateway_settings
            )

        response = await resolved_client.system_one(
            state=kwargs["state"], model=settings.model, questions=api_questions
        )
        try:
            return settings.to_output(response["answers"])
        except (KeyError, TypeError) as e:
            raise AgentRuntimeError(
                code=AgentRuntimeErrorCode.LLM_INVALID_RESPONSE,
                title="Invalid response from Jev",
                detail=f"Jev response is missing answer data: {e!r}",
                category=UiPathErrorCategory.SYSTEM,
            ) from e

    return StructuredToolWithArgumentProperties(
        name=tool_name,
        description=resource.description,
        args_schema=input_model,
        coroutine=jev_tool_fn,
        output_type=output_model,
        argument_properties=resource.argument_properties,
        metadata={
            "tool_type": resource.type.lower(),
            "sub_type": JEV_SUB_TYPE,
            "display_name": tool_name,
            "args_schema": input_model,
            "output_schema": output_model,
        },
    )


def _get_uipath_client_settings(llm: BaseChatModel) -> UiPathBaseSettings | None:
    # UiPath chat models carry the agent's client settings (incl. the AgentHub
    # licensing config); reuse them so gateway calls are attributed the same way.
    settings = getattr(llm, "client_settings", None)
    return settings if isinstance(settings, UiPathBaseSettings) else None
