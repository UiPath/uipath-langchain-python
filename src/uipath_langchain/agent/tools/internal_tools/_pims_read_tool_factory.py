"""Shared factory for PIMs read tools.

All read-only PIMs tools (case state, case plan, case entity, execution trace)
share the same shape: ``GET`` a PIMs endpoint scoped by ``instanceId``, accept
an optional ``folderKey`` override, return the parsed JSON body, and map known
HTTP errors to friendly :class:`AgentRuntimeError` instances. Only the path
template and the human-readable subject (e.g. "case plan") differ per tool.

This module provides that shared builder. Each per-tool file collapses to a
path constant plus a thin call to :func:`build_pims_read_tool`.
"""

from typing import Any

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


def _build_error_map(
    subject: str,
) -> dict[tuple[int, str | None], tuple[str, UiPathErrorCategory]]:
    """Build the standard PIMs-read error map for the given subject.

    ``subject`` is the noun used in 401/403 messages (e.g. "case plan",
    "execution trace"). The 404 message stays generic since the missing thing
    is always the case instance itself.
    """
    return {
        (404, None): (
            "Case instance not found for tool '{tool}': {message}",
            UiPathErrorCategory.USER,
        ),
        (401, None): (
            f"Unauthorized when fetching {subject} for tool '{{tool}}': {{message}}",
            UiPathErrorCategory.SYSTEM,
        ),
        (403, None): (
            f"Forbidden when fetching {subject} for tool '{{tool}}': {{message}}",
            UiPathErrorCategory.USER,
        ),
    }


def build_pims_read_tool(
    resource: AgentInternalToolResourceConfig,
    *,
    path_template: str,
    subject: str,
) -> StructuredTool:
    """Build a read-only PIMs GET tool.

    The tool takes an LLM-supplied ``instanceId`` and an optional ``folderKey``
    override, calls the PIMs endpoint at ``path_template`` (with the
    ``{instance_id}`` placeholder filled in), and returns the parsed JSON body.
    Known 4xx errors map to friendly :class:`AgentRuntimeError` instances.

    Args:
        resource: The tool resource configuration (name, description, input
            and output schemas, argument properties).
        path_template: The PIMs path with an ``{instance_id}`` placeholder
            (e.g. ``"pims_/api/v1/cases/{instance_id}/case-rules"``).
        subject: The noun used in 401/403 error messages
            (e.g. ``"case plan"``, ``"execution trace"``).

    Returns:
        A :class:`StructuredTool` ready to register in the internal-tool
        factory map.

    Notes:
        The folder key is taken from the optional ``folderKey`` argument when
        provided, otherwise from the runtime's ``UIPATH_FOLDER_KEY`` env var
        via :class:`uipath.platform.common.ApiClient`.
    """
    tool_name = sanitize_tool_name(resource.name)
    input_model = create_model(resource.input_schema)
    output_model = create_model(resource.output_schema)
    error_map = _build_error_map(subject)

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
        url = path_template.format(instance_id=instance_id)
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
                e, error_map, title=tool_name, tool=tool_name
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
