"""Tests for context_tool.py module."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document
from uipath.agent.models.agent import (
    AgentContextOutputColumn,
    AgentContextQuerySetting,
    AgentContextResourceConfig,
    AgentContextRetrievalMode,
    AgentContextSettings,
    AgentContextValueSetting,
)
from uipath.platform.context_grounding import (
    CitationMode,
    DeepRagContent,
)
from uipath.platform.errors import EnrichedException
from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
    AgentStartupError,
    AgentStartupErrorCode,
)
from uipath_langchain.agent.tools.context_tool import (
    _normalize_folder_prefix,
    build_glob_pattern,
    create_context_tool,
    handle_batch_transform,
    handle_deep_rag,
    handle_semantic_search,
)
from uipath_langchain.agent.tools.structured_tool_with_argument_properties import (
    StructuredToolWithArgumentProperties,
)
from uipath_langchain.agent.tools.structured_tool_with_output_type import (
    StructuredToolWithOutputType,
)


def _make_context_resource(
    name="test_tool",
    description="Test tool",
    index_name="test-index",
    folder_path="/test/folder",
    query_value=None,
    query_variant="static",
    citation_mode_value=None,
    retrieval_mode=AgentContextRetrievalMode.SEMANTIC,
    folder_path_prefix=None,
    **kwargs,
):
    """Helper to create an AgentContextResourceConfig."""
    return AgentContextResourceConfig(
        name=name,
        description=description,
        resource_type="context",
        index_name=index_name,
        folder_path=folder_path,
        settings=AgentContextSettings(
            result_count=1,
            retrieval_mode=retrieval_mode,
            query=AgentContextQuerySetting(
                value=query_value,
                description="some description",
                variant=query_variant,
            ),
            citation_mode=citation_mode_value,
            folder_path_prefix=folder_path_prefix,
        ),
        is_enabled=True,
        **kwargs,
    )


class TestHandleDeepRag:
    """Test cases for handle_deep_rag function."""

    @pytest.fixture
    def base_resource_config(self):
        """Fixture for base resource configuration."""
        return _make_context_resource

    def test_successful_deep_rag_creation(self, base_resource_config):
        """Test successful creation of Deep RAG tool with all required fields."""
        resource = base_resource_config(
            citation_mode_value=AgentContextValueSetting(value="Inline"),
            query_value="some query",
        )

        result = handle_deep_rag("test_deep_rag", resource)

        assert isinstance(result, StructuredToolWithArgumentProperties)
        assert result.name == "test_deep_rag"
        assert result.description == "Test tool"
        assert hasattr(result.args_schema, "model_json_schema")
        assert result.args_schema.model_json_schema()["properties"] == {}
        assert issubclass(result.output_type, DeepRagContent)
        schema = result.output_type.model_json_schema()
        assert "deepRagId" in schema["properties"]
        assert schema["properties"]["deepRagId"]["type"] == "string"

    def test_deep_rag_has_tool_wrapper(self, base_resource_config):
        """Test that Deep RAG tool has a tool wrapper for static args resolution."""
        resource = base_resource_config(
            citation_mode_value=AgentContextValueSetting(value="Inline"),
            query_value="some query",
        )

        result = handle_deep_rag("test_deep_rag", resource)

        assert result.awrapper is not None

    def test_deep_rag_with_folder_path_prefix_from_settings(self, base_resource_config):
        """Test that folder_path_prefix with argument variant is resolved in wrapper, not via argument_properties."""
        resource = base_resource_config(
            citation_mode_value=AgentContextValueSetting(value="Inline"),
            query_value="some query",
            folder_path_prefix=AgentContextQuerySetting(
                value="{deepRagFolderPrefix}", variant="argument"
            ),
        )

        result = handle_deep_rag("test_deep_rag", resource)

        assert isinstance(result, StructuredToolWithArgumentProperties)
        # folder_path_prefix is resolved directly in the wrapper from state,
        # not via argument_properties or args_schema
        assert "folder_path_prefix" not in result.argument_properties
        assert isinstance(result.args_schema, type)

    def test_missing_static_query_value_raises_error(self, base_resource_config):
        """Test that missing query.value for static variant raises AgentStartupError."""
        resource = base_resource_config(query_variant="static", query_value=None)

        with pytest.raises(AgentStartupError) as exc_info:
            handle_deep_rag("test_deep_rag", resource)
        assert exc_info.value.error_info.code == AgentStartupError.full_code(
            AgentStartupErrorCode.INVALID_TOOL_CONFIG
        )

    def test_missing_query_variant_raises_error(self, base_resource_config):
        """Test that missing query.variant raises AgentStartupError."""
        resource = base_resource_config(query_value="some query")
        resource.settings.query.variant = None

        with pytest.raises(AgentStartupError) as exc_info:
            handle_deep_rag("test_deep_rag", resource)
        assert exc_info.value.error_info.code == AgentStartupError.full_code(
            AgentStartupErrorCode.INVALID_TOOL_CONFIG
        )

    def test_missing_citation_mode_raises_error(self, base_resource_config):
        """Test that missing citation_mode raises AgentStartupError."""
        resource = base_resource_config(
            query_value="some query", citation_mode_value=None
        )
        resource.settings.citation_mode = None

        with pytest.raises(AgentStartupError) as exc_info:
            handle_deep_rag("test_deep_rag", resource)
        assert exc_info.value.error_info.code == AgentStartupError.full_code(
            AgentStartupErrorCode.INVALID_TOOL_CONFIG
        )

    @pytest.mark.parametrize(
        "citation_mode_value,expected_enum",
        [
            (AgentContextValueSetting(value="Inline"), CitationMode.INLINE),
            (AgentContextValueSetting(value="Skip"), CitationMode.SKIP),
        ],
    )
    def test_citation_mode_conversion(
        self, base_resource_config, citation_mode_value, expected_enum
    ):
        """Test that citation mode is correctly converted to CitationMode enum."""
        resource = base_resource_config(
            query_value="some query", citation_mode_value=citation_mode_value
        )

        result = handle_deep_rag("test_deep_rag", resource)

        assert isinstance(result, StructuredToolWithArgumentProperties)

    def test_tool_name_preserved(self, base_resource_config):
        """Test that the sanitized tool name is correctly applied."""
        resource = base_resource_config(
            name="My Deep RAG Tool",
            citation_mode_value=AgentContextValueSetting(value="Inline"),
            query_value="some query",
        )

        result = handle_deep_rag("my_deep_rag_tool", resource)

        assert result.name == "my_deep_rag_tool"

    def test_tool_description_preserved(self, base_resource_config):
        """Test that the tool description is correctly preserved."""
        custom_description = "Custom description for Deep RAG retrieval"
        resource = base_resource_config(
            description=custom_description,
            citation_mode_value=AgentContextValueSetting(value="Inline"),
            query_value="some query",
        )

        result = handle_deep_rag("test_tool", resource)

        assert result.description == custom_description

    @pytest.mark.asyncio
    async def test_tool_with_different_citation_modes(self, base_resource_config):
        """Test tool creation and invocation with different citation modes."""
        for mode_value, expected_mode in [
            ("Inline", CitationMode.INLINE),
            ("Skip", CitationMode.SKIP),
        ]:
            resource = base_resource_config(
                query_value="test query",
                citation_mode_value=AgentContextValueSetting(value=mode_value),
            )
            tool = handle_deep_rag("test_tool", resource)

            with patch(
                "uipath_langchain._utils.durable_interrupt.decorator.interrupt"
            ) as mock_interrupt:
                mock_interrupt.return_value = {"mocked": "response"}
                assert tool.coroutine is not None
                await tool.coroutine()

                call_args = mock_interrupt.call_args[0][0]
                assert call_args.citation_mode == expected_mode

    @pytest.mark.asyncio
    async def test_unique_task_names_on_multiple_invocations(
        self, base_resource_config
    ):
        """Test that each tool invocation generates a unique task name."""
        resource = base_resource_config(
            query_value="test query",
            citation_mode_value=AgentContextValueSetting(value="Inline"),
        )
        tool = handle_deep_rag("test_tool", resource)

        task_names = []
        with patch(
            "uipath_langchain._utils.durable_interrupt.decorator.interrupt"
        ) as mock_interrupt:
            mock_interrupt.return_value = {"mocked": "response"}

            # Invoke the tool multiple times
            assert tool.coroutine is not None
            for _ in range(3):
                await tool.coroutine()
                call_args = mock_interrupt.call_args[0][0]
                task_names.append(call_args.name)

        # Verify all task names are unique
        assert len(task_names) == len(set(task_names))
        # Verify all have task- prefix
        assert all(name.startswith("task-") for name in task_names)

    def test_dynamic_query_deep_rag_creation(self, base_resource_config):
        """Test successful creation of Deep RAG tool with dynamic query."""
        resource = base_resource_config(
            query_variant="dynamic",
            query_value=None,
            citation_mode_value=AgentContextValueSetting(value="Inline"),
        )

        result = handle_deep_rag("test_deep_rag", resource)

        assert isinstance(result, StructuredToolWithArgumentProperties)
        assert result.name == "test_deep_rag"
        assert result.description == "Test tool"
        assert result.args_schema is not None  # Dynamic has input schema
        assert issubclass(result.output_type, DeepRagContent)

    def test_dynamic_query_deep_rag_has_query_parameter(self, base_resource_config):
        """Test that dynamic Deep RAG tool has query parameter in schema."""
        resource = base_resource_config(
            query_variant="dynamic",
            query_value=None,
            citation_mode_value=AgentContextValueSetting(value="Inline"),
        )

        result = handle_deep_rag("test_deep_rag", resource)

        # Check that the input schema has a query field
        assert result.args_schema is not None
        assert hasattr(result.args_schema, "model_json_schema")
        schema = result.args_schema.model_json_schema()
        assert "properties" in schema
        assert "query" in schema["properties"]
        assert schema["properties"]["query"]["type"] == "string"

    @pytest.mark.asyncio
    async def test_dynamic_query_uses_provided_query(self, base_resource_config):
        """Test that dynamic query variant uses the query parameter provided at runtime."""
        resource = base_resource_config(
            query_variant="dynamic",
            query_value=None,
            citation_mode_value=AgentContextValueSetting(value="Inline"),
        )
        tool = handle_deep_rag("test_tool", resource)

        with patch(
            "uipath_langchain._utils.durable_interrupt.decorator.interrupt"
        ) as mock_interrupt:
            mock_interrupt.return_value = {"mocked": "response"}
            assert tool.coroutine is not None
            await tool.coroutine(query="runtime provided query")

            call_args = mock_interrupt.call_args[0][0]
            assert call_args.prompt == "runtime provided query"

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"UIPATH_FOLDER_PATH": "/Shared/TestFolder"})
    async def test_deep_rag_uses_execution_folder_path(self, base_resource_config):
        """Test that CreateDeepRag receives index_folder_path from the execution environment."""
        resource = base_resource_config(
            query_variant="static",
            query_value="test query",
            citation_mode_value=AgentContextValueSetting(value="Inline"),
        )
        tool = handle_deep_rag("test_tool", resource)

        with patch(
            "uipath_langchain._utils.durable_interrupt.decorator.interrupt"
        ) as mock_interrupt:
            mock_interrupt.return_value = {"mocked": "response"}
            assert tool.coroutine is not None
            await tool.coroutine()

            deep_rag_arg = mock_interrupt.call_args[0][0]
            assert deep_rag_arg.index_folder_path == "/Shared/TestFolder"


class TestCreateContextTool:
    """Test cases for create_context_tool function."""

    @pytest.fixture
    def semantic_search_config(self):
        """Fixture for semantic search configuration."""
        return _make_context_resource(
            name="test_semantic_search",
            description="Test semantic search",
            retrieval_mode=AgentContextRetrievalMode.SEMANTIC,
            query_variant="dynamic",
        )

    @pytest.fixture
    def deep_rag_config(self):
        """Fixture for deep RAG configuration."""
        return _make_context_resource(
            name="test_deep_rag",
            description="Test Deep RAG",
            retrieval_mode=AgentContextRetrievalMode.DEEP_RAG,
            query_value="test query",
            query_variant="static",
            citation_mode_value=AgentContextValueSetting(value="Inline"),
        )

    def test_create_semantic_search_tool(self, semantic_search_config):
        """Test that semantic search retrieval mode creates semantic search tool."""
        result = create_context_tool(semantic_search_config)

        assert isinstance(result, StructuredToolWithOutputType)
        assert result.name == "test_semantic_search"
        assert result.args_schema is not None  # Semantic search has input schema

    def test_create_deep_rag_tool(self, deep_rag_config):
        """Test that deep_rag retrieval mode creates Deep RAG tool."""
        result = create_context_tool(deep_rag_config)

        assert isinstance(result, StructuredToolWithArgumentProperties)
        assert result.name == "test_deep_rag"
        assert hasattr(result.args_schema, "model_json_schema")
        assert result.args_schema.model_json_schema()["properties"] == {}
        assert issubclass(result.output_type, DeepRagContent)

    def test_case_insensitive_retrieval_mode(self, deep_rag_config):
        """Test that retrieval mode matching is case-insensitive."""
        # Test with uppercase
        deep_rag_config.settings.retrieval_mode = "DEEPRAG"
        result = create_context_tool(deep_rag_config)
        assert isinstance(result, StructuredToolWithArgumentProperties)

        # Test with mixed case
        deep_rag_config.settings.retrieval_mode = "deeprag"
        result = create_context_tool(deep_rag_config)
        assert isinstance(result, StructuredToolWithArgumentProperties)


class TestHandleSemanticSearch:
    """Test cases for handle_semantic_search function."""

    @pytest.fixture
    def semantic_config(self):
        """Fixture for semantic search configuration."""
        return _make_context_resource(
            name="semantic_tool",
            description="Semantic search tool",
            retrieval_mode=AgentContextRetrievalMode.SEMANTIC,
            query_variant="dynamic",
        )

    def test_semantic_search_tool_creation(self, semantic_config):
        """Test successful creation of semantic search tool."""
        result = handle_semantic_search("semantic_tool", semantic_config)

        assert isinstance(result, StructuredToolWithOutputType)
        assert result.name == "semantic_tool"
        assert result.description == "Semantic search tool"
        assert result.args_schema is not None

    def test_semantic_search_has_query_parameter(self, semantic_config):
        """Test that semantic search tool has query parameter in schema."""
        result = handle_semantic_search("semantic_tool", semantic_config)

        # Check that the input schema has a query field
        assert result.args_schema is not None
        assert hasattr(result.args_schema, "model_json_schema")
        schema = result.args_schema.model_json_schema()
        assert "properties" in schema
        assert "query" in schema["properties"]
        assert schema["properties"]["query"]["type"] == "string"

    @pytest.mark.asyncio
    async def test_semantic_search_returns_documents(self, semantic_config):
        """Test that semantic search tool returns documents."""
        tool = handle_semantic_search("semantic_tool", semantic_config)

        # Mock the retriever
        mock_documents = [
            Document(page_content="Test content 1", metadata={"source": "doc1"}),
            Document(page_content="Test content 2", metadata={"source": "doc2"}),
        ]

        with patch(
            "uipath_langchain.agent.tools.context_tool.ContextGroundingRetriever"
        ) as mock_retriever_class:
            mock_retriever = AsyncMock()
            mock_retriever.ainvoke.return_value = mock_documents
            mock_retriever_class.return_value = mock_retriever

            # Recreate the tool with mocked retriever
            tool = handle_semantic_search("semantic_tool", semantic_config)
            assert tool.coroutine is not None
            result = await tool.coroutine(query="test query")

            assert "documents" in result
            assert len(result["documents"]) == 2
            assert result["documents"][0]["page_content"] == "Test content 1"

    def test_static_query_semantic_search_creation(self):
        """Test successful creation of semantic search tool with static query."""
        resource = _make_context_resource(
            name="semantic_tool",
            description="Semantic search tool",
            retrieval_mode=AgentContextRetrievalMode.SEMANTIC,
            query_value="predefined static query",
            query_variant="static",
        )

        result = handle_semantic_search("semantic_tool", resource)

        assert isinstance(result, StructuredToolWithOutputType)
        assert result.name == "semantic_tool"
        assert result.description == "Semantic search tool"
        assert hasattr(result.args_schema, "model_json_schema")
        assert result.args_schema.model_json_schema()["properties"] == {}

    @pytest.mark.asyncio
    async def test_static_query_uses_predefined_query(self):
        """Test that static query variant uses the predefined query value."""
        resource = _make_context_resource(
            name="semantic_tool",
            description="Semantic search tool",
            retrieval_mode=AgentContextRetrievalMode.SEMANTIC,
            query_value="predefined static query",
            query_variant="static",
        )

        mock_documents = [
            Document(page_content="Test content", metadata={"source": "doc1"}),
        ]

        with patch(
            "uipath_langchain.agent.tools.context_tool.ContextGroundingRetriever"
        ) as mock_retriever_class:
            mock_retriever = AsyncMock()
            mock_retriever.ainvoke.return_value = mock_documents
            mock_retriever_class.return_value = mock_retriever

            tool = handle_semantic_search("semantic_tool", resource)
            assert tool.coroutine is not None
            result = await tool.coroutine()

            # Verify the retriever was called with the static query value
            mock_retriever.ainvoke.assert_called_once_with("predefined static query")
            assert "documents" in result
            assert len(result["documents"]) == 1

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"UIPATH_FOLDER_PATH": "/Shared/TestFolder"})
    async def test_semantic_search_uses_execution_folder_path(self, semantic_config):
        """Test that ContextGroundingRetriever receives folder_path from the execution environment."""
        with patch(
            "uipath_langchain.agent.tools.context_tool.ContextGroundingRetriever"
        ) as mock_retriever_class:
            mock_retriever = AsyncMock()
            mock_retriever.ainvoke.return_value = []
            mock_retriever_class.return_value = mock_retriever

            tool = handle_semantic_search("semantic_tool", semantic_config)
            assert tool.coroutine is not None
            await tool.coroutine(query="test query")

            call_kwargs = mock_retriever_class.call_args[1]
            assert call_kwargs["folder_path"] == "/Shared/TestFolder"


class TestHandleBatchTransform:
    """Test cases for handle_batch_transform function."""

    @pytest.fixture
    def batch_transform_config(self):
        """Fixture for batch transform configuration with static query."""
        return AgentContextResourceConfig(
            name="batch_transform_tool",
            description="Batch transform tool",
            resource_type="context",
            index_name="test-index",
            folder_path="/test/folder",
            settings=AgentContextSettings(
                result_count=5,
                retrieval_mode=AgentContextRetrievalMode.BATCH_TRANSFORM,
                query=AgentContextQuerySetting(
                    value="transform this data",
                    description="Static query for batch transform",
                    variant="static",
                ),
                web_search_grounding=AgentContextValueSetting(value="enabled"),
                output_columns=[
                    AgentContextOutputColumn(
                        name="output_col1", description="First output column"
                    ),
                    AgentContextOutputColumn(
                        name="output_col2", description="Second output column"
                    ),
                ],
            ),
            is_enabled=True,
        )

    def test_static_query_batch_transform_creation(self, batch_transform_config):
        """Test successful creation of batch transform tool with static query."""
        result = handle_batch_transform("batch_transform_tool", batch_transform_config)

        assert isinstance(result, StructuredToolWithArgumentProperties)
        assert result.name == "batch_transform_tool"
        assert result.description == "Batch transform tool"
        assert result.args_schema is not None  # Has destination_path parameter
        # Output model is built from the job-attachment schema so that the
        # job_attachment_wrapper can locate and register the attachment.
        output_schema = result.output_type.model_json_schema()
        assert "result" in output_schema.get("properties", {})

    def test_static_query_batch_transform_has_destination_path_only(
        self, batch_transform_config
    ):
        """Test that static batch transform only has destination_path in schema."""
        result = handle_batch_transform("batch_transform_tool", batch_transform_config)

        assert result.args_schema is not None
        assert hasattr(result.args_schema, "model_json_schema")
        schema = result.args_schema.model_json_schema()
        assert "properties" in schema
        assert "destination_path" in schema["properties"]
        assert "query" not in schema["properties"]  # No query for static

    def test_dynamic_query_batch_transform_creation(self):
        """Test successful creation of batch transform tool with dynamic query."""
        resource = AgentContextResourceConfig(
            name="batch_transform_tool",
            description="Batch transform tool",
            resource_type="context",
            index_name="test-index",
            folder_path="/test/folder",
            settings=AgentContextSettings(
                result_count=5,
                retrieval_mode=AgentContextRetrievalMode.BATCH_TRANSFORM,
                query=AgentContextQuerySetting(
                    value=None,
                    description="Dynamic query for batch transform",
                    variant="dynamic",
                ),
                web_search_grounding=AgentContextValueSetting(value="enabled"),
                output_columns=[
                    AgentContextOutputColumn(
                        name="output_col1", description="First output column"
                    ),
                ],
            ),
            is_enabled=True,
        )

        result = handle_batch_transform("batch_transform_tool", resource)

        assert isinstance(result, StructuredToolWithArgumentProperties)
        assert result.name == "batch_transform_tool"
        assert result.args_schema is not None
        output_schema = result.output_type.model_json_schema()
        assert "result" in output_schema.get("properties", {})

    def test_dynamic_query_batch_transform_has_both_parameters(self):
        """Test that dynamic batch transform has both query and destination_path."""
        resource = AgentContextResourceConfig(
            name="batch_transform_tool",
            description="Batch transform tool",
            resource_type="context",
            index_name="test-index",
            folder_path="/test/folder",
            settings=AgentContextSettings(
                result_count=5,
                retrieval_mode=AgentContextRetrievalMode.BATCH_TRANSFORM,
                query=AgentContextQuerySetting(
                    value=None,
                    description="Dynamic query for batch transform",
                    variant="dynamic",
                ),
                web_search_grounding=AgentContextValueSetting(value="enabled"),
                output_columns=[
                    AgentContextOutputColumn(
                        name="output_col1", description="First output column"
                    ),
                ],
            ),
            is_enabled=True,
        )

        result = handle_batch_transform("batch_transform_tool", resource)

        assert result.args_schema is not None
        assert hasattr(result.args_schema, "model_json_schema")
        schema = result.args_schema.model_json_schema()
        assert "properties" in schema
        assert "query" in schema["properties"]
        assert "destination_path" in schema["properties"]

    def test_batch_transform_with_folder_path_prefix_from_settings(self):
        """Test that batch transform builds argument_properties from settings."""
        resource = AgentContextResourceConfig(
            name="batch_transform_tool",
            description="Batch transform tool",
            resource_type="context",
            index_name="test-index",
            folder_path="/test/folder",
            settings=AgentContextSettings(
                result_count=5,
                retrieval_mode=AgentContextRetrievalMode.BATCH_TRANSFORM,
                query=AgentContextQuerySetting(
                    value="transform query",
                    description="Static query",
                    variant="static",
                ),
                web_search_grounding=AgentContextValueSetting(value="enabled"),
                output_columns=[
                    AgentContextOutputColumn(
                        name="output_col1", description="First output column"
                    ),
                ],
                folder_path_prefix=AgentContextQuerySetting(
                    value="{batchFolderPrefix}", variant="argument"
                ),
            ),
            is_enabled=True,
        )

        result = handle_batch_transform("batch_transform_tool", resource)

        assert isinstance(result, StructuredToolWithArgumentProperties)
        # folder_path_prefix is resolved directly in the wrapper from state,
        # not via argument_properties or args_schema
        assert "folder_path_prefix" not in result.argument_properties
        assert isinstance(result.args_schema, type)

    @pytest.mark.asyncio
    async def test_static_query_batch_transform_uses_predefined_query(
        self, batch_transform_config
    ):
        """Test that static query variant uses the predefined query value."""
        tool = handle_batch_transform("batch_transform_tool", batch_transform_config)

        mock_uipath = AsyncMock()
        mock_uipath.jobs.create_attachment_async = AsyncMock(return_value="att-id-1")
        with (
            patch(
                "uipath_langchain._utils.durable_interrupt.decorator.interrupt"
            ) as mock_interrupt,
            patch(
                "uipath_langchain.agent.tools.context_tool.UiPath",
                return_value=mock_uipath,
            ),
        ):
            mock_interrupt.return_value = {"mocked": "response"}
            assert tool.coroutine is not None
            await tool.coroutine(destination_path="/output/result.csv")

            call_args = mock_interrupt.call_args[0][0]
            assert call_args.prompt == "transform this data"
            assert call_args.destination_path == "/output/result.csv"

    @pytest.mark.asyncio
    async def test_dynamic_query_batch_transform_uses_provided_query(self):
        """Test that dynamic query variant uses the query parameter provided at runtime."""
        resource = AgentContextResourceConfig(
            name="batch_transform_tool",
            description="Batch transform tool",
            resource_type="context",
            index_name="test-index",
            folder_path="/test/folder",
            settings=AgentContextSettings(
                result_count=5,
                retrieval_mode=AgentContextRetrievalMode.BATCH_TRANSFORM,
                query=AgentContextQuerySetting(
                    value=None,
                    description="Dynamic query for batch transform",
                    variant="dynamic",
                ),
                web_search_grounding=AgentContextValueSetting(value="enabled"),
                output_columns=[
                    AgentContextOutputColumn(
                        name="output_col1", description="First output column"
                    ),
                ],
            ),
            is_enabled=True,
        )

        tool = handle_batch_transform("batch_transform_tool", resource)

        mock_uipath = AsyncMock()
        mock_uipath.jobs.create_attachment_async = AsyncMock(return_value="att-id-2")
        with (
            patch(
                "uipath_langchain._utils.durable_interrupt.decorator.interrupt"
            ) as mock_interrupt,
            patch(
                "uipath_langchain.agent.tools.context_tool.UiPath",
                return_value=mock_uipath,
            ),
        ):
            mock_interrupt.return_value = {"mocked": "response"}
            assert tool.coroutine is not None
            await tool.coroutine(
                query="runtime provided query", destination_path="/output/result.csv"
            )

            call_args = mock_interrupt.call_args[0][0]
            assert call_args.prompt == "runtime provided query"
            assert call_args.destination_path == "/output/result.csv"

    @pytest.mark.asyncio
    async def test_static_query_batch_transform_uses_default_destination_path(
        self, batch_transform_config
    ):
        """Test that static batch transform uses default destination_path when not provided."""
        tool = handle_batch_transform("batch_transform_tool", batch_transform_config)

        mock_uipath = AsyncMock()
        mock_uipath.jobs.create_attachment_async = AsyncMock(return_value="att-id-3")
        with (
            patch(
                "uipath_langchain._utils.durable_interrupt.decorator.interrupt"
            ) as mock_interrupt,
            patch(
                "uipath_langchain.agent.tools.context_tool.UiPath",
                return_value=mock_uipath,
            ),
        ):
            mock_interrupt.return_value = {"mocked": "response"}
            assert tool.coroutine is not None
            # Call without providing destination_path
            await tool.coroutine()

            call_args = mock_interrupt.call_args[0][0]
            assert call_args.prompt == "transform this data"
            assert call_args.destination_path == "output.csv"

    @pytest.mark.asyncio
    async def test_dynamic_query_batch_transform_uses_default_destination_path(self):
        """Test that dynamic batch transform uses default destination_path when not provided."""
        resource = AgentContextResourceConfig(
            name="batch_transform_tool",
            description="Batch transform tool",
            resource_type="context",
            index_name="test-index",
            folder_path="/test/folder",
            settings=AgentContextSettings(
                result_count=5,
                retrieval_mode=AgentContextRetrievalMode.BATCH_TRANSFORM,
                query=AgentContextQuerySetting(
                    value=None,
                    description="Dynamic query for batch transform",
                    variant="dynamic",
                ),
                web_search_grounding=AgentContextValueSetting(value="enabled"),
                output_columns=[
                    AgentContextOutputColumn(
                        name="output_col1", description="First output column"
                    ),
                ],
            ),
            is_enabled=True,
        )

        tool = handle_batch_transform("batch_transform_tool", resource)

        mock_uipath = AsyncMock()
        mock_uipath.jobs.create_attachment_async = AsyncMock(return_value="att-id-4")
        with (
            patch(
                "uipath_langchain._utils.durable_interrupt.decorator.interrupt"
            ) as mock_interrupt,
            patch(
                "uipath_langchain.agent.tools.context_tool.UiPath",
                return_value=mock_uipath,
            ),
        ):
            mock_interrupt.return_value = {"mocked": "response"}
            assert tool.coroutine is not None
            # Call with only query, no destination_path
            await tool.coroutine(query="runtime provided query")

            call_args = mock_interrupt.call_args[0][0]
            assert call_args.prompt == "runtime provided query"
            assert call_args.destination_path == "output.csv"

    @pytest.mark.asyncio
    @patch.dict(os.environ, {"UIPATH_FOLDER_PATH": "/Shared/TestFolder"})
    async def test_batch_transform_uses_execution_folder_path(
        self, batch_transform_config
    ):
        """Test that CreateBatchTransform receives index_folder_path from the execution environment."""
        tool = handle_batch_transform("batch_transform_tool", batch_transform_config)

        mock_uipath = MagicMock()
        mock_uipath.jobs.create_attachment_async = AsyncMock(return_value="att-id")
        with (
            patch(
                "uipath_langchain._utils.durable_interrupt.decorator.interrupt"
            ) as mock_interrupt,
            patch(
                "uipath_langchain.agent.tools.context_tool.UiPath",
                return_value=mock_uipath,
            ),
        ):
            mock_interrupt.return_value = MagicMock()
            assert tool.coroutine is not None
            await tool.coroutine(destination_path="output.csv")

            batch_transform_arg = mock_interrupt.call_args[0][0]
            assert batch_transform_arg.index_folder_path == "/Shared/TestFolder"


class TestBuildGlobPattern:
    """Test cases for build_glob_pattern function."""

    # --- No prefix ---

    def test_no_prefix_no_extension(self):
        assert build_glob_pattern(None, None) == "**/*"

    def test_empty_string_prefix_treated_as_no_prefix(self):
        assert build_glob_pattern("", None) == "**/*"

    def test_no_prefix_with_extension(self):
        assert build_glob_pattern(None, "pdf") == "**/*.pdf"

    def test_no_prefix_extension_uppercased(self):
        """Extension should be lowercased before building the pattern."""
        assert build_glob_pattern(None, "PDF") == "**/*.pdf"

    def test_no_prefix_extension_mixed_case(self):
        assert build_glob_pattern(None, "TxT") == "**/*.txt"

    # --- Explicit "**" prefix ---

    def test_double_star_prefix_no_extension(self):
        assert build_glob_pattern("**", None) == "**/*"

    def test_double_star_prefix_with_extension(self):
        assert build_glob_pattern("**", "pdf") == "**/*.pdf"

    # --- Prefix with trailing slash stripped ---

    def test_prefix_trailing_slash_stripped(self):
        assert build_glob_pattern("documents/", "pdf") == "documents/*.pdf"

    def test_prefix_multiple_trailing_slashes_stripped(self):
        assert build_glob_pattern("documents///", "pdf") == "documents/*.pdf"

    # --- Prefix with leading slash stripped ---

    def test_prefix_leading_slash_stripped(self):
        assert build_glob_pattern("/documents", "pdf") == "documents/*.pdf"

    def test_prefix_leading_and_trailing_slash_stripped(self):
        assert build_glob_pattern("/documents/", None) == "documents/*"

    # --- Prefix starting with "**" is kept as-is ---

    def test_prefix_starting_with_double_star_kept(self):
        assert build_glob_pattern("**/documents", "pdf") == "**/documents/*.pdf"

    def test_prefix_starting_with_double_star_leading_slash_not_stripped(self):
        """A prefix that already starts with ** is not modified further."""
        assert build_glob_pattern("**/docs/", "txt") == "**/docs/*.txt"

    # --- Normal prefix without slashes ---

    def test_simple_prefix_no_extension(self):
        assert build_glob_pattern("folder", None) == "folder/*"

    def test_simple_prefix_with_extension(self):
        assert build_glob_pattern("folder", "pdf") == "folder/*.pdf"

    def test_nested_prefix_with_extension(self):
        assert (
            build_glob_pattern("folder/subfolder", "docx") == "folder/subfolder/*.docx"
        )

    # --- All supported extensions ---

    @pytest.mark.parametrize("ext", ["pdf", "txt", "docx", "csv"])
    def test_supported_extensions(self, ext):
        assert build_glob_pattern(None, ext) == f"**/*.{ext}"

    def test_unsupported_extension_still_works(self):
        """Extensions outside the named set are handled identically."""
        assert build_glob_pattern("data", "xlsx") == "data/*.xlsx"

    # --- Trailing file-matching globs stripped ---

    def test_prefix_with_trailing_star(self):
        """Trailing /* is stripped since extension is appended separately."""
        assert build_glob_pattern("documents/*", "pdf") == "documents/*.pdf"

    def test_prefix_with_trailing_double_star(self):
        """Trailing /** is stripped."""
        assert build_glob_pattern("documents/**", "pdf") == "documents/*.pdf"

    def test_prefix_with_trailing_double_star_star(self):
        """Trailing /**/* is stripped."""
        assert build_glob_pattern("documents/**/*", "pdf") == "documents/*.pdf"

    def test_match_all_glob_treated_as_no_prefix(self):
        """/**/* is a match-all pattern and should be treated as no prefix."""
        assert build_glob_pattern("/**/*", "pdf") == "**/*.pdf"

    def test_star_slash_star_treated_as_no_prefix(self):
        """*/* is a match-all pattern and should be treated as no prefix."""
        assert build_glob_pattern("*/*", "pdf") == "**/*.pdf"

    def test_double_star_slash_star_treated_as_no_prefix(self):
        """**/* is a match-all pattern and should be treated as no prefix."""
        assert build_glob_pattern("**/*", "pdf") == "**/*.pdf"


class TestNormalizeFolderPrefix:
    """Test cases for _normalize_folder_prefix function."""

    # --- None / empty ---

    def test_none_returns_double_star(self):
        assert _normalize_folder_prefix(None) == "**"

    def test_empty_string_returns_double_star(self):
        assert _normalize_folder_prefix("") == "**"

    def test_only_slashes_returns_double_star(self):
        assert _normalize_folder_prefix("///") == "**"

    # --- Leading/trailing slash stripping ---

    def test_strips_leading_slash(self):
        assert _normalize_folder_prefix("/documents") == "documents"

    def test_strips_trailing_slash(self):
        assert _normalize_folder_prefix("documents/") == "documents"

    def test_strips_both_slashes(self):
        assert _normalize_folder_prefix("/documents/") == "documents"

    # --- Trailing glob stripping ---

    def test_strips_trailing_star(self):
        assert _normalize_folder_prefix("documents/*") == "documents"

    def test_strips_trailing_double_star(self):
        assert _normalize_folder_prefix("documents/**") == "documents"

    def test_strips_trailing_double_star_star(self):
        assert _normalize_folder_prefix("documents/**/*") == "documents"

    def test_nested_prefix_strips_trailing_glob(self):
        assert _normalize_folder_prefix("folder/subfolder/*") == "folder/subfolder"

    def test_nested_prefix_strips_trailing_double_star_star(self):
        assert _normalize_folder_prefix("folder/subfolder/**/*") == "folder/subfolder"

    # --- Match-all patterns become ** ---

    def test_star_slash_star_returns_double_star(self):
        assert _normalize_folder_prefix("*/*") == "**"

    def test_double_star_slash_star_returns_double_star(self):
        assert _normalize_folder_prefix("**/*") == "**"

    def test_slash_double_star_slash_star_returns_double_star(self):
        assert _normalize_folder_prefix("/**/*") == "**"

    # --- Preserves valid prefixes ---

    def test_simple_prefix(self):
        assert _normalize_folder_prefix("folder") == "folder"

    def test_nested_prefix(self):
        assert _normalize_folder_prefix("folder/subfolder") == "folder/subfolder"

    def test_double_star_prefix_preserved(self):
        assert _normalize_folder_prefix("**/documents") == "**/documents"

    def test_double_star_nested_prefix_preserved(self):
        assert _normalize_folder_prefix("**/docs/reports") == "**/docs/reports"


class TestSemanticSearchErrorHandling:
    """Test error handling for semantic search HTTP failures."""

    @pytest.fixture
    def semantic_config(self):
        return _make_context_resource(
            name="test_search",
            description="Test search",
            retrieval_mode=AgentContextRetrievalMode.SEMANTIC,
            query_variant="dynamic",
        )

    @pytest.mark.asyncio
    async def test_400_raises_agent_runtime_error_with_user_category(
        self, semantic_config, make_enriched_exception
    ):
        with patch(
            "uipath_langchain.agent.tools.context_tool.ContextGroundingRetriever"
        ) as mock_retriever_class:
            mock_retriever = AsyncMock()
            mock_retriever.ainvoke.side_effect = make_enriched_exception(
                400, "One or more validation errors occurred."
            )
            mock_retriever_class.return_value = mock_retriever

            tool = handle_semantic_search("test_search", semantic_config)
            assert tool.coroutine is not None

            with pytest.raises(AgentRuntimeError) as exc_info:
                await tool.coroutine(query="test query")
            assert exc_info.value.error_info.category == UiPathErrorCategory.USER
            assert exc_info.value.error_info.code == AgentRuntimeError.full_code(
                AgentRuntimeErrorCode.HTTP_ERROR
            )

    @pytest.mark.asyncio
    async def test_non_400_enriched_exception_propagates(
        self, semantic_config, make_enriched_exception
    ):
        with patch(
            "uipath_langchain.agent.tools.context_tool.ContextGroundingRetriever"
        ) as mock_retriever_class:
            mock_retriever = AsyncMock()
            mock_retriever.ainvoke.side_effect = make_enriched_exception(
                500, "Internal Server Error"
            )
            mock_retriever_class.return_value = mock_retriever

            tool = handle_semantic_search("test_search", semantic_config)
            assert tool.coroutine is not None

            with pytest.raises(EnrichedException):
                await tool.coroutine(query="test query")
