"""Context tool creation for semantic index retrieval."""

import logging
import uuid
from typing import Any, Optional

from jsonpath_ng import parse  # type: ignore[import-untyped]
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolCall
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, Field, create_model
from uipath.agent.models.agent import (
    AgentContextResourceConfig,
    AgentContextRetrievalMode,
    AgentContextType,
    AgentMessageRole,
    AgentToolArgumentArgumentProperties,
    AgentToolArgumentProperties,
    LowCodeAgentDefinition,
)
from uipath.eval.mocks import mockable
from uipath.platform import UiPath
from uipath.platform.common import CreateBatchTransform, CreateDeepRag, UiPathConfig
from uipath.platform.context_grounding import (
    BatchTransformOutputColumn,
    CitationMode,
    DeepRagContent,
)
from uipath.platform.errors import EnrichedException
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain._utils import get_execution_folder_path
from uipath_langchain._utils.durable_interrupt import durable_interrupt
from uipath_langchain.agent.exceptions import (
    AgentStartupError,
    AgentStartupErrorCode,
    raise_for_enriched,
)
from uipath_langchain.agent.react.jsonschema_pydantic_converter import (
    create_model as create_model_from_schema,
)
from uipath_langchain.agent.react.types import AgentGraphState
from uipath_langchain.agent.tools.internal_tools.schema_utils import (
    BATCH_TRANSFORM_OUTPUT_SCHEMA,
)
from uipath_langchain.agent.tools.tool_node import ToolWrapperReturnType
from uipath_langchain.retrievers import ContextGroundingRetriever

from .structured_tool_with_argument_properties import (
    StructuredToolWithArgumentProperties,
)
from .structured_tool_with_output_type import StructuredToolWithOutputType
from .utils import sanitize_tool_name

logger = logging.getLogger(__name__)

_CONTEXT_GROUNDING_ERRORS: dict[
    tuple[int, str | None], tuple[str, UiPathErrorCategory]
] = {
    (400, None): (
        "Context grounding returned an error for index '{index}': {message}",
        UiPathErrorCategory.USER,
    ),
}


def _build_arg_props_from_settings(
    resource: AgentContextResourceConfig,
) -> dict[str, AgentToolArgumentProperties]:
    """Build argument_properties from context resource settings.

    Context resources don't receive argumentProperties from the frontend.
    Instead, we derive them from the settings when variant="argument".
    Only includes fields that belong in the tool's args_schema (i.e. query).
    """
    assert resource.settings is not None
    arg_props: dict[str, AgentToolArgumentProperties] = {}

    if resource.settings.query and resource.settings.query.variant == "argument":
        argument_path = (resource.settings.query.value or "").strip("{}")
        arg_props["query"] = AgentToolArgumentArgumentProperties(
            is_sensitive=False,
            argument_path=argument_path,
        )

    return arg_props


def _resolve_folder_path_prefix_from_state(
    resource: AgentContextResourceConfig,
    state: dict[str, Any],
) -> str | None:
    """Resolve folder_path_prefix from agent state using jsonpath from settings."""
    assert resource.settings is not None
    setting = resource.settings.folder_path_prefix
    if not setting or setting.variant != "argument" or not setting.value:
        return None
    argument_path = "$." + setting.value.strip("{}")
    matches = parse(argument_path).find(state)
    return matches[0].value if matches else None


def _resolve_file_extension(resource: AgentContextResourceConfig) -> str | None:
    """Resolve file extension from settings, returning None for 'All' or empty."""
    assert resource.settings is not None
    if resource.settings.file_extension and resource.settings.file_extension.value:
        ext = resource.settings.file_extension.value
        if ext.lower() == "all":
            return None
        return ext
    return None


def _resolve_static_folder_path_prefix(
    resource: AgentContextResourceConfig,
) -> str | None:
    """Resolve static folder_path_prefix from settings."""
    assert resource.settings is not None
    if (
        resource.settings.folder_path_prefix
        and resource.settings.folder_path_prefix.value
        and resource.settings.folder_path_prefix.variant == "static"
    ):
        return resource.settings.folder_path_prefix.value
    return None


def is_static_query(resource: AgentContextResourceConfig) -> bool:
    """Check if the resource configuration uses a static query variant."""
    assert resource.settings is not None
    if resource.settings.query is None or resource.settings.query.variant is None:
        return False
    return resource.settings.query.variant.lower() == "static"


def _extract_system_prompt(agent: LowCodeAgentDefinition | None) -> str:
    """Extract system prompt from agent definition messages."""
    if agent is None:
        return ""
    return "\n\n".join(
        msg.content
        for msg in agent.messages
        if msg.role == AgentMessageRole.SYSTEM and msg.content
    )


def create_context_tool(
    resource: AgentContextResourceConfig,
    llm: BaseChatModel | None = None,
    agent: LowCodeAgentDefinition | None = None,
) -> StructuredTool | BaseTool | None:
    assert resource.context_type is not None

    if resource.context_type == AgentContextType.DATA_FABRIC_ENTITY_SET:
        if llm is None:
            raise ValueError("Data Fabric entity set tools require an LLM instance")
        from .datafabric_tool import create_datafabric_query_tool
        from .datafabric_tool.datafabric_tool import BASE_SYSTEM_PROMPT

        return create_datafabric_query_tool(
            resource,
            llm,
            agent_config={BASE_SYSTEM_PROMPT: _extract_system_prompt(agent)},
        )

    elif resource.context_type == AgentContextType.INDEX:
        assert resource.settings is not None
        tool_name = sanitize_tool_name(resource.name)
        retrieval_mode = resource.settings.retrieval_mode.lower()

        if retrieval_mode == AgentContextRetrievalMode.DEEP_RAG.value.lower():
            return handle_deep_rag(tool_name, resource)

        if retrieval_mode == AgentContextRetrievalMode.BATCH_TRANSFORM.value.lower():
            return handle_batch_transform(tool_name, resource)

        return handle_semantic_search(tool_name, resource)

    return None

def handle_semantic_search(
    tool_name: str, resource: AgentContextResourceConfig
) -> StructuredTool:
    ensure_valid_fields(resource)
    assert resource.settings is not None

    assert resource.settings.query.variant is not None

    file_extension = _resolve_file_extension(resource)
    static_folder_path_prefix = _resolve_static_folder_path_prefix(resource)
    result_count = resource.settings.result_count
    threshold = resource.settings.threshold

    static = is_static_query(resource)
    prompt = resource.settings.query.value if static else None
    if static:
        assert prompt is not None

    arg_props = _build_arg_props_from_settings(resource)

    class ContextOutputSchemaModel(BaseModel):
        documents: list[Document] = Field(
            ..., description="List of retrieved documents."
        )

    output_model = ContextOutputSchemaModel

    schema_fields: dict[str, Any] = {}

    if "query" in arg_props:
        schema_fields["query"] = (
            str,
            Field(
                default=None,
                description="The query to search for in the knowledge base",
            ),
        )
    elif not static:
        schema_fields["query"] = (
            str,
            Field(
                ...,
                description="The query to search for in the knowledge base",
            ),
        )

    has_arg_folder = (
        resource.settings.folder_path_prefix
        and resource.settings.folder_path_prefix.variant == "argument"
        and resource.settings.folder_path_prefix.value
    )

    _resolved_arg_folder_prefix: str | None = None

    input_model = create_model("SemanticSearchInput", **schema_fields)

    @mockable(
        name=resource.name,
        description=resource.description,
        input_schema=input_model.model_json_schema(),
        output_schema=output_model.model_json_schema(),
        example_calls=[],  # Examples cannot be provided for context.
    )
    async def context_tool_fn(
        query: Optional[str] = None,
    ) -> dict[str, Any]:
        resolved_folder_path_prefix = (
            static_folder_path_prefix or _resolved_arg_folder_prefix
        )

        retriever = ContextGroundingRetriever(
            index_name=resource.index_name,
            folder_path=get_execution_folder_path(),
            number_of_results=result_count,
            threshold=threshold,
            scope_folder=resolved_folder_path_prefix,
            scope_extension=file_extension,
        )

        actual_query = prompt or query
        assert actual_query is not None
        try:
            docs = await retriever.ainvoke(actual_query)
        except EnrichedException as e:
            raise_for_enriched(
                e,
                _CONTEXT_GROUNDING_ERRORS,
                title=f"Failed to search context index '{resource.index_name}'",
                index=resource.index_name or "<unknown>",
            )
            raise
        return {
            "documents": [
                {"metadata": doc.metadata, "page_content": doc.page_content}
                for doc in docs
            ]
        }

    if arg_props or has_arg_folder:

        async def context_semantic_search_wrapper(
            tool: BaseTool,
            call: ToolCall,
            state: AgentGraphState,
        ) -> ToolWrapperReturnType:
            nonlocal _resolved_arg_folder_prefix
            _resolved_arg_folder_prefix = _resolve_folder_path_prefix_from_state(
                resource, dict(state)
            )
            return await tool.ainvoke(call)

        tool = StructuredToolWithArgumentProperties(
            name=tool_name,
            description=resource.description,
            args_schema=input_model,
            coroutine=context_tool_fn,
            output_type=output_model,
            argument_properties=arg_props,
            metadata={
                "tool_type": "context",
                "display_name": resource.name,
                "index_name": resource.index_name,
                "context_retrieval_mode": resource.settings.retrieval_mode,
            },
        )
        tool.set_tool_wrappers(awrapper=context_semantic_search_wrapper)
        return tool

    return StructuredToolWithOutputType(
        name=tool_name,
        description=resource.description,
        args_schema=input_model,
        coroutine=context_tool_fn,
        output_type=output_model,
        metadata={
            "tool_type": "context",
            "display_name": resource.name,
            "index_name": resource.index_name,
            "context_retrieval_mode": resource.settings.retrieval_mode,
        },
    )


def handle_deep_rag(
    tool_name: str, resource: AgentContextResourceConfig
) -> StructuredToolWithArgumentProperties:
    ensure_valid_fields(resource)
    assert resource.settings is not None

    assert resource.settings.query.variant is not None

    index_name = resource.index_name
    if not resource.settings.citation_mode:
        raise AgentStartupError(
            code=AgentStartupErrorCode.INVALID_TOOL_CONFIG,
            title="Missing citation mode",
            detail="Citation mode is required for Deep RAG. Please set the citation_mode field in context settings.",
            category=UiPathErrorCategory.USER,
        )
    citation_mode = CitationMode(resource.settings.citation_mode.value)

    static = is_static_query(resource)
    prompt = resource.settings.query.value if static else None
    if static:
        assert prompt is not None

    static_folder_path_prefix = _resolve_static_folder_path_prefix(resource)
    file_extension = _resolve_file_extension(resource)

    output_model = create_model(
        "DeepRagOutputModel",
        __base__=DeepRagContent,
        deep_rag_id=(str, Field(alias="deepRagId")),
    )

    arg_props = _build_arg_props_from_settings(resource)

    schema_fields: dict[str, Any] = (
        {}
        if static
        else {
            "query": (
                str,
                Field(
                    ...,
                    description="Describe the task: what to research across documents, what to synthesize, and how to cite sources",
                ),
            ),
        }
    )

    input_model = create_model("DeepRagInput", **schema_fields)

    _resolved_arg_folder_prefix: str | None = None

    @mockable(
        name=resource.name,
        description=resource.description,
        input_schema=input_model.model_json_schema(),
        output_schema=output_model.model_json_schema(),
        example_calls=[],  # Examples cannot be provided for context.
    )
    async def context_tool_fn(
        query: Optional[str] = None,
    ) -> dict[str, Any]:
        actual_prompt = prompt or query
        glob_pattern = build_glob_pattern(
            folder_path_prefix=static_folder_path_prefix or _resolved_arg_folder_prefix,
            file_extension=file_extension,
        )

        @durable_interrupt
        async def create_deep_rag():
            return CreateDeepRag(
                name=f"task-{uuid.uuid4()}",
                index_name=index_name,
                prompt=actual_prompt,
                citation_mode=citation_mode,
                index_folder_path=get_execution_folder_path(),
                glob_pattern=glob_pattern,
            )

        return await create_deep_rag()

    async def context_deep_rag_wrapper(
        tool: BaseTool,
        call: ToolCall,
        state: AgentGraphState,
    ) -> ToolWrapperReturnType:
        nonlocal _resolved_arg_folder_prefix
        _resolved_arg_folder_prefix = _resolve_folder_path_prefix_from_state(
            resource, dict(state)
        )
        return await tool.ainvoke(call)

    tool = StructuredToolWithArgumentProperties(
        name=tool_name,
        description=resource.description,
        args_schema=input_model,
        coroutine=context_tool_fn,
        output_type=output_model,
        argument_properties=arg_props,
        metadata={
            "tool_type": "context",
            "display_name": resource.name,
            "index_name": resource.index_name,
            "context_retrieval_mode": resource.settings.retrieval_mode,
        },
    )
    tool.set_tool_wrappers(awrapper=context_deep_rag_wrapper)
    return tool


def handle_batch_transform(
    tool_name: str, resource: AgentContextResourceConfig
) -> StructuredToolWithArgumentProperties:
    ensure_valid_fields(resource)
    assert resource.settings is not None

    assert resource.settings.query is not None
    assert resource.settings.query.variant is not None

    index_name = resource.index_name
    index_folder_path = get_execution_folder_path()
    if not resource.settings.web_search_grounding:
        raise AgentStartupError(
            code=AgentStartupErrorCode.INVALID_TOOL_CONFIG,
            title="Missing web search grounding",
            detail="Web search grounding field is required for Batch Transform. Please set the web_search_grounding field in context settings.",
            category=UiPathErrorCategory.USER,
        )
    enable_web_search_grounding = (
        resource.settings.web_search_grounding.value.lower() == "enabled"
    )

    batch_transform_output_columns: list[BatchTransformOutputColumn] = []
    if (output_columns := resource.settings.output_columns) is None or not len(
        output_columns
    ):
        raise AgentStartupError(
            code=AgentStartupErrorCode.INVALID_TOOL_CONFIG,
            title="Missing output columns",
            detail="Batch transform requires at least one output column to be specified in settings.output_columns. Please add output columns to the context configuration.",
            category=UiPathErrorCategory.USER,
        )

    for column in output_columns:
        batch_transform_output_columns.append(
            BatchTransformOutputColumn(
                name=column.name,
                description=column.description,
            )
        )

    static = is_static_query(resource)
    prompt = resource.settings.query.value if static else None
    if static:
        assert prompt is not None

    static_folder_path_prefix = _resolve_static_folder_path_prefix(resource)

    arg_props = _build_arg_props_from_settings(resource)

    output_model = create_model_from_schema(BATCH_TRANSFORM_OUTPUT_SCHEMA)

    schema_fields: dict[str, Any] = {}
    if not static:
        schema_fields["query"] = (
            str,
            Field(
                ...,
                description="Describe the task for each row: what to analyze, what to extract, and how to populate the output columns",
            ),
        )
    schema_fields["destination_path"] = (
        str,
        Field(
            default="output.csv",
            description="The relative file path destination for the modified csv file",
        ),
    )
    input_model = create_model("BatchTransformInput", **schema_fields)

    _resolved_arg_folder_prefix: str | None = None

    @mockable(
        name=resource.name,
        description=resource.description,
        input_schema=input_model.model_json_schema(),
        output_schema=output_model.model_json_schema(),
        example_calls=[],  # Examples cannot be provided for context.
    )
    async def context_tool_fn(
        query: Optional[str] = None,
        destination_path: str = "output.csv",
    ) -> dict[str, Any]:
        actual_prompt = prompt or query
        glob_pattern = build_glob_pattern(
            folder_path_prefix=static_folder_path_prefix or _resolved_arg_folder_prefix,
            file_extension=None,
        )

        @durable_interrupt
        async def create_batch_transform():
            return CreateBatchTransform(
                name=f"task-{uuid.uuid4()}",
                index_name=index_name,
                prompt=actual_prompt,
                destination_path=destination_path,
                index_folder_path=index_folder_path,
                enable_web_search_grounding=enable_web_search_grounding,
                output_columns=batch_transform_output_columns,
                storage_bucket_folder_path_prefix=glob_pattern,
            )

        await create_batch_transform()

        uipath = UiPath()
        result_attachment_id = await uipath.jobs.create_attachment_async(
            name=destination_path,
            source_path=destination_path,
            job_key=UiPathConfig.job_key,
        )

        return {
            "result": {
                "ID": str(result_attachment_id),
                "FullName": destination_path,
                "MimeType": "text/csv",
            }
        }

    from uipath_langchain.agent.wrappers import get_job_attachment_wrapper

    job_attachment_wrapper = get_job_attachment_wrapper(output_type=output_model)

    async def context_batch_transform_wrapper(
        tool: BaseTool,
        call: ToolCall,
        state: AgentGraphState,
    ) -> ToolWrapperReturnType:
        nonlocal _resolved_arg_folder_prefix
        _resolved_arg_folder_prefix = _resolve_folder_path_prefix_from_state(
            resource, dict(state)
        )
        return await job_attachment_wrapper(tool, call, state)

    tool = StructuredToolWithArgumentProperties(
        name=tool_name,
        description=resource.description,
        args_schema=input_model,
        coroutine=context_tool_fn,
        output_type=output_model,
        argument_properties=arg_props,
        metadata={
            "tool_type": "context",
            "display_name": resource.name,
            "index_name": resource.index_name,
            "context_retrieval_mode": resource.settings.retrieval_mode,
            "output_schema": output_model,
        },
    )
    tool.set_tool_wrappers(awrapper=job_attachment_wrapper)
    return tool


def ensure_valid_fields(resource_config: AgentContextResourceConfig):
    assert resource_config.settings is not None
    if not resource_config.settings.query.variant:
        raise AgentStartupError(
            code=AgentStartupErrorCode.INVALID_TOOL_CONFIG,
            title="Missing query variant",
            detail="Query variant is required. Please set the query variant in context settings.",
            category=UiPathErrorCategory.USER,
        )

    if is_static_query(resource_config) and not resource_config.settings.query.value:
        raise AgentStartupError(
            code=AgentStartupErrorCode.INVALID_TOOL_CONFIG,
            title="Missing static query value",
            detail="Static query requires a query value to be set. Please provide a value for the static query in context settings.",
            category=UiPathErrorCategory.USER,
        )


def _normalize_folder_prefix(folder_path_prefix: str | None) -> str:
    """Normalize a folder path prefix to a clean directory-only pattern.

    Strips leading/trailing slashes and trailing file-matching globs
    (e.g. /*, /**, /**/*) since the caller appends the file extension part.
    """
    if not folder_path_prefix:
        return "**"

    prefix = folder_path_prefix.strip("/").rstrip("/*")
    if not prefix:
        return "**"

    return prefix


def build_glob_pattern(
    folder_path_prefix: str | None, file_extension: str | None
) -> str:
    prefix = _normalize_folder_prefix(folder_path_prefix)

    # Handle extension
    extension = "*"
    if file_extension:
        ext = file_extension.lower()
        extension = f"*.{ext}"

    # Final pattern logic
    if prefix == "**":
        return "**/*" if extension == "*" else f"**/{extension}"

    return f"{prefix}/{extension}"
