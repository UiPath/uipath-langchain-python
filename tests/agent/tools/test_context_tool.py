"""Tests for context_tool.py module."""

from unittest.mock import AsyncMock, patch

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

from uipath_langchain.agent.exceptions import AgentStartupError, AgentStartupErrorCode
from uipath_langchain.agent.tools.context_tool import (
    create_context_tool,
    handle_batch_transform,
    handle_deep_rag,
    handle_semantic_search,
)
from uipath_langchain.agent.tools.structured_tool_with_output_type import (
    StructuredToolWithOutputType,
)


class TestHandleDeepRag:
    """Test cases for handle_deep_rag function."""

    @pytest.fixture
    def base_resource_config(self):
        """Fixture for base resource configuration."""

        def _create_config(
            name="test_deep_rag",
            description="Test Deep RAG tool",
            index_name="test-index",
            folder_path="/test/folder",
            query_value=None,
            query_variant="static",
            citation_mode_value=None,
            retrieval_mode=AgentContextRetrievalMode.SEMANTIC,
        ):
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
                ),
                is_enabled=True,
            )

        return _create_config

    def test_successful_deep_rag_creation(self, base_resource_config):
        """Test successful creation of Deep RAG tool with all required fields."""
        resource = base_resource_config(
            citation_mode_value=AgentContextValueSetting(value="Inline"),
            query_value="some query",
        )

        result = handle_deep_rag("test_deep_rag", resource)

        assert isinstance(result, StructuredToolWithOutputType)
        assert result.name == "test_deep_rag"
        assert result.description == "Test Deep RAG tool"
        assert hasattr(result.args_schema, "model_json_schema")
        assert result.args_schema.model_json_schema()["properties"] == {}
        assert issubclass(result.output_type, DeepRagContent)
        schema = result.output_type.model_json_schema()
        assert "deepRagId" in schema["properties"]
        assert schema["properties"]["deepRagId"]["type"] == "string"

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

        assert isinstance(result, StructuredToolWithOutputType)

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
                "uipath_langchain.agent.tools.durable_interrupt.decorator.interrupt"
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
            "uipath_langchain.agent.tools.durable_interrupt.decorator.interrupt"
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

        assert isinstance(result, StructuredToolWithOutputType)
        assert result.name == "test_deep_rag"
        assert result.description == "Test Deep RAG tool"
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
            "uipath_langchain.agent.tools.durable_interrupt.decorator.interrupt"
        ) as mock_interrupt:
            mock_interrupt.return_value = {"mocked": "response"}
            assert tool.coroutine is not None
            await tool.coroutine(query="runtime provided query")

            call_args = mock_interrupt.call_args[0][0]
            assert call_args.prompt == "runtime provided query"


class TestCreateContextTool:
    """Test cases for create_context_tool function."""

    @pytest.fixture
    def semantic_search_config(self):
        """Fixture for semantic search configuration."""
        return AgentContextResourceConfig(
            name="test_semantic_search",
            description="Test semantic search",
            resource_type="context",
            index_name="test-index",
            folder_path="/test/folder",
            settings=AgentContextSettings(
                result_count=10,
                retrieval_mode=AgentContextRetrievalMode.SEMANTIC,
                query=AgentContextQuerySetting(
                    value=None,
                    description="Query for semantic search",
                    variant="dynamic",
                ),
            ),
            is_enabled=True,
        )

    @pytest.fixture
    def deep_rag_config(self):
        """Fixture for deep RAG configuration."""
        return AgentContextResourceConfig(
            name="test_deep_rag",
            description="Test Deep RAG",
            resource_type="context",
            index_name="test-index",
            folder_path="/test/folder",
            settings=AgentContextSettings(
                result_count=5,
                retrieval_mode=AgentContextRetrievalMode.DEEP_RAG,
                query=AgentContextQuerySetting(
                    value="test query",
                    description="Test query description",
                    variant="static",
                ),
                citation_mode=AgentContextValueSetting(value="Inline"),
            ),
            is_enabled=True,
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

        assert isinstance(result, StructuredToolWithOutputType)
        assert result.name == "test_deep_rag"
        assert hasattr(result.args_schema, "model_json_schema")
        assert result.args_schema.model_json_schema()["properties"] == {}
        assert issubclass(result.output_type, DeepRagContent)

    def test_case_insensitive_retrieval_mode(self, deep_rag_config):
        """Test that retrieval mode matching is case-insensitive."""
        # Test with uppercase
        deep_rag_config.settings.retrieval_mode = "DEEP_RAG"
        result = create_context_tool(deep_rag_config)
        assert isinstance(result, StructuredToolWithOutputType)

        # Test with mixed case
        deep_rag_config.settings.retrieval_mode = "Deep_Rag"
        result = create_context_tool(deep_rag_config)
        assert isinstance(result, StructuredToolWithOutputType)


class TestHandleSemanticSearch:
    """Test cases for handle_semantic_search function."""

    @pytest.fixture
    def semantic_config(self):
        """Fixture for semantic search configuration."""
        return AgentContextResourceConfig(
            name="semantic_tool",
            description="Semantic search tool",
            resource_type="context",
            index_name="test-index",
            folder_path="/test/folder",
            settings=AgentContextSettings(
                result_count=5,
                retrieval_mode=AgentContextRetrievalMode.SEMANTIC,
                query=AgentContextQuerySetting(
                    value=None,
                    description="Query for semantic search",
                    variant="dynamic",
                ),
            ),
            is_enabled=True,
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
        resource = AgentContextResourceConfig(
            name="semantic_tool",
            description="Semantic search tool",
            resource_type="context",
            index_name="test-index",
            folder_path="/test/folder",
            settings=AgentContextSettings(
                result_count=5,
                retrieval_mode=AgentContextRetrievalMode.SEMANTIC,
                query=AgentContextQuerySetting(
                    value="predefined static query",
                    description="Static query for semantic search",
                    variant="static",
                ),
            ),
            is_enabled=True,
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
        resource = AgentContextResourceConfig(
            name="semantic_tool",
            description="Semantic search tool",
            resource_type="context",
            index_name="test-index",
            folder_path="/test/folder",
            settings=AgentContextSettings(
                result_count=5,
                retrieval_mode=AgentContextRetrievalMode.SEMANTIC,
                query=AgentContextQuerySetting(
                    value="predefined static query",
                    description="Static query for semantic search",
                    variant="static",
                ),
            ),
            is_enabled=True,
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

        assert isinstance(result, StructuredToolWithOutputType)
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

        assert isinstance(result, StructuredToolWithOutputType)
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
                "uipath_langchain.agent.tools.durable_interrupt.decorator.interrupt"
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
                "uipath_langchain.agent.tools.durable_interrupt.decorator.interrupt"
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
                "uipath_langchain.agent.tools.durable_interrupt.decorator.interrupt"
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
                "uipath_langchain.agent.tools.durable_interrupt.decorator.interrupt"
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
