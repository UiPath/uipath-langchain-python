"""Internal tool that fetches the case entity (business object, documents, comments) for a PIMs case."""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import StructuredTool
from uipath.agent.models.agent import AgentInternalToolResourceConfig
from uipath.eval.mocks import mockable
from uipath.platform import UiPath
from uipath.platform.errors import EnrichedException
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions import raise_for_enriched
from uipath_langchain.agent.react.jsonschema_pydantic_converter import create_model
from uipath_langchain.agent.tools.structured_tool_with_argument_properties import (
    StructuredToolWithArgumentProperties,
)
from uipath_langchain.agent.tools.utils import sanitize_tool_name

PIMS_CASE_ENTITY_PATH = "pims_/api/v1/cases/{instance_id}/case-json"

_GET_CASE_ENTITY_ERRORS: dict[
    tuple[int, str | None], tuple[str, UiPathErrorCategory]
] = {
    (404, None): (
        "Case not found for tool '{tool}': {message}",
        UiPathErrorCategory.USER,
    ),
    (401, None): (
        "Unauthorized when fetching case entity for tool '{tool}': {message}",
        UiPathErrorCategory.SYSTEM,
    ),
    (403, None): (
        "Forbidden when fetching case entity for tool '{tool}': {message}",
        UiPathErrorCategory.USER,
    ),
}


def create_get_case_entity_tool(
    resource: AgentInternalToolResourceConfig, llm: BaseChatModel
) -> StructuredTool:
    """Create the GetCaseEntity internal tool.

    Calls the PIMs ``case-json`` endpoint with the agent's UiPath credentials
    and folder context. Returns the case entity payload — the business object,
    linked documents, and comments — used by the agent to ground reasoning in
    case facts.

    The folder key is taken from the optional ``folderKey`` argument when
    provided, otherwise from the runtime's ``UIPATH_FOLDER_KEY`` environment
    variable via :class:`uipath.platform.common.ApiClient`.
    """
    tool_name = sanitize_tool_name(resource.name)
    input_model = create_model(resource.input_schema)
    output_model = create_model(resource.output_schema)

    @mockable(
        name=resource.name,
        description=resource.description,
        input_schema=input_model.model_json_schema(),
        output_schema=output_model.model_json_schema(),
    )
    async def tool_fn(**kwargs: Any) -> Any:
        instance_id = kwargs.get("instanceId")
        if not instance_id:
            raise ValueError("Argument 'instanceId' is required")

        folder_key_override = kwargs.get("folderKey")

        client = UiPath()
        url = PIMS_CASE_ENTITY_PATH.format(instance_id=instance_id)
        request_kwargs: dict[str, Any] = {}
        if folder_key_override:
            request_kwargs["headers"] = {"x-uipath-folderkey": folder_key_override}
        else:
            request_kwargs["include_folder_headers"] = True

        try:
            response = await client.api_client.request_async(
                "GET", url, **request_kwargs
            )
        except EnrichedException as e:
            raise_for_enriched(
                e, _GET_CASE_ENTITY_ERRORS, title=tool_name, tool=tool_name
            )
            raise

        return response.json()

    return StructuredToolWithArgumentProperties(
        name=tool_name,
        description=resource.description,
        args_schema=input_model,
        coroutine=tool_fn,
        output_type=output_model,
        argument_properties=resource.argument_properties,
        metadata={
            "tool_type": resource.type.lower(),
            "display_name": tool_name,
            "args_schema": input_model,
            "output_schema": output_model,
        },
    )
