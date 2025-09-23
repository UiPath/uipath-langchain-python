import json
import logging
from typing import Any, Iterable, Optional, Type

import httpx
from jsonschema_pydantic import jsonschema_to_pydantic as create_model  # type: ignore
from langchain_core.caches import BaseCache
from langchain_core.runnables.utils import Output
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import interrupt
from pydantic import BaseModel
from uipath import UiPath
from uipath.agent.models.agent import (
    AgentDefinition,
    AgentEscalationChannel,
    AgentEscalationResourceConfig,
    AgentIntegrationToolParameter,
    AgentIntegrationToolResourceConfig,
    AgentProcessToolResourceConfig,
    AgentResourceConfig,
)
from uipath.models import CreateAction, InvokeProcess
from uipath.models.connections import ConnectionTokenType

logger = logging.getLogger(__name__)


def create_process_tool(resource: AgentProcessToolResourceConfig) -> Iterable[BaseTool]:
    async def process(**kwargs) -> BaseModel:
        return interrupt(
            InvokeProcess(
                name=resource.name,
                input_arguments=kwargs,
                process_folder_path=resource.properties.folder_path,
            )
        )

    input_schema = create_model(resource.input_schema)

    class ProcessTool(StructuredTool):
        @property
        def OutputType(self) -> type[Output]:
            return create_model(resource.output_schema)

    yield ProcessTool(
        name=resource.name,
        args_schema=input_schema,
        description=resource.description,
        coroutine=process,
    )


def create_escalation_tool_from_channel(channel: AgentEscalationChannel) -> BaseTool:
    async def escalate(**kwargs) -> BaseModel:
        recipients = channel.recipients
        if len(recipients) > 1:
            logger.warning(
                "Received more than one recipient. Defaulting to first recipient."
            )
        assignee = recipients[0].value if recipients else None
        return interrupt(
            CreateAction(
                title=channel.description,
                data=kwargs,
                assignee=assignee,
                app_name=channel.properties.app_name,
                app_folder_path=None,  # Channels specify folder name but not folder path.
                app_folder_key=channel.properties.resource_key,
                app_key=channel.properties.resource_key,
                app_version=channel.properties.app_version,
            )
        )

    input_schema = create_model(channel.input_schema)

    class EscalationTool(StructuredTool):
        @property
        def OutputType(self) -> type[Output]:
            return create_model(channel.output_schema)

    return EscalationTool(
        name=channel.name,
        args_schema=input_schema,
        description=channel.description,
        coroutine=escalate,
    )


def create_escalation_tool(
    resource: AgentEscalationResourceConfig,
) -> Iterable[BaseTool]:
    for channel in resource.channels:
        yield create_escalation_tool_from_channel(channel)


METHOD_MAP = {"GETBYID": "GET"}


def build_query_params(parameters: list[AgentIntegrationToolParameter]):
    query_params = [
        x for x in parameters if x.field_location == "query" and x.value is not None
    ]
    if query_params:
        return "?" + "&".join(f"{q.name}={q.value}" for q in query_params)
    return ""


def filter_query_params(
    kwargs: dict[str, Any], parameters: list[AgentIntegrationToolParameter]
):
    query_params = {x.name for x in parameters if x.field_location == "query"}
    non_query_params = {x.name for x in parameters if x.field_location != "query"}
    fields_to_ignore = query_params - non_query_params
    return {k: v for k, v in kwargs.items() if k not in fields_to_ignore}


def create_integration_tool(
    resource: AgentIntegrationToolResourceConfig,
) -> Iterable[BaseTool]:
    async def integration(**kwargs) -> BaseModel:
        uipath = UiPath()
        remote_connection = await uipath.connections.retrieve_async(
            resource.properties.connection.id
        )
        token = await uipath.connections.retrieve_token_async(
            resource.properties.connection.id, ConnectionTokenType.BEARER
        )
        tool_url = f"{remote_connection.api_base_uri}/v3/element/instances/{remote_connection.element_instance_id}{resource.properties.tool_path}"
        tool_url = f"{tool_url}{build_query_params(resource.properties.parameters)}"
        tool_url = tool_url.format(**kwargs)

        authorization = f"{token.token_type} {token.access_token}"
        method = METHOD_MAP.get(resource.properties.method, resource.properties.method)
        response = await httpx.AsyncClient().request(
            method,
            tool_url,
            headers={"Authorization": authorization},
            content=json.dumps(
                filter_query_params(kwargs, resource.properties.parameters)
            ),
        )
        return response.json()

    input_schema = create_model(resource.input_schema)

    class IntegrationTool(StructuredTool):
        @property
        def OutputType(self) -> type[Output]:
            return create_model({})

    yield IntegrationTool(
        name=resource.name,
        args_schema=input_schema,
        description=resource.description,
        coroutine=integration,
    )


def create_cached_wrapper_from_tool(
    wrapped: BaseTool, cache: Optional[BaseCache]
) -> BaseTool:
    if cache is None:
        return wrapped
    else:

        async def cached_invocation(**kwargs) -> BaseModel:
            namespace = f"{wrapped.name}.tool_invoke"
            key = str(kwargs)
            cached = cache.lookup(namespace, key)
            if cached:
                return cached[0]
            response = await wrapped.ainvoke(input=kwargs)
            cache.update(namespace, key, [response])
            return response

        input_schema = wrapped.args_schema

        class CachedTool(StructuredTool):
            OutputType: Type[BaseModel] = wrapped.OutputType

        return CachedTool(
            name=wrapped.name,
            args_schema=input_schema,
            description=wrapped.description,
            coroutine=cached_invocation,
        )


def create_cached_wrapper(
    tools: Iterable[BaseTool], cache: Optional[BaseCache]
) -> Iterable[BaseTool]:
    for wrapped in tools:
        yield create_cached_wrapper_from_tool(wrapped, cache)


def create_resource_tool(
    resource: AgentResourceConfig, cache: Optional[BaseCache] = None
) -> Iterable[BaseTool]:
    match resource:
        case AgentProcessToolResourceConfig():
            return create_cached_wrapper(create_process_tool(resource), cache)
        case AgentIntegrationToolResourceConfig():
            return create_cached_wrapper(create_integration_tool(resource), cache)
        case AgentEscalationResourceConfig():
            return create_cached_wrapper(create_escalation_tool(resource), cache)
        case _:
            raise NotImplementedError()


def safe_extract_tools(
    agent_definition: AgentDefinition, cache: Optional[BaseCache] = None
) -> list[BaseTool]:
    tools = []
    for resource in agent_definition.resources:
        try:
            for structured_tool in create_resource_tool(resource, cache):
                tools.append(structured_tool)
        except NotImplementedError:
            logger.warning(f"Unable to convert {resource.name} into a tool.")
    return tools
