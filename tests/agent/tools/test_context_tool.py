"""Tests for context_tool.py module."""

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.documents import Document
from uipath.agent.models.agent import (
    AgentContextQuerySetting,
    AgentContextResourceConfig,
    AgentContextRetrievalMode,
    AgentContextSettings,
    AgentContextValueSetting,
)
from uipath.platform.context_grounding import CitationMode, DeepRagResponse

from uipath_langchain.agent.tools.context_tool import (
    create_context_tool,
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
                        variant="variant",
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
        assert result.args_schema is None
        assert result.output_type == DeepRagResponse

    def test_missing_query_object_raises_error(self, base_resource_config):
        """Test that missing query object raises ValueError."""
        resource = base_resource_config(query_value=None)
        resource.settings.query = None

        with pytest.raises(ValueError, match="Query object is required"):
            handle_deep_rag("test_deep_rag", resource)

    def test_missing_query_value_raises_error(self, base_resource_config):
        """Test that missing query.value raises ValueError."""
        resource = base_resource_config()
        resource.settings.query.value = None

        with pytest.raises(ValueError, match="Query prompt is required"):
            handle_deep_rag("test_deep_rag", resource)

    def test_missing_citation_mode_raises_error(self, base_resource_config):
        """Test that missing citation_mode raises ValueError."""
        resource = base_resource_config(
            query_value="some query", citation_mode_value=None
        )
        resource.settings.citation_mode = None

        with pytest.raises(ValueError, match="Citation mode is required for Deep RAG"):
            handle_deep_rag("test_deep_rag", resource)

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
                "uipath_langchain.agent.tools.context_tool.interrupt"
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
            "uipath_langchain.agent.tools.context_tool.interrupt"
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
        assert result.args_schema is None  # Deep RAG has no input schema
        assert result.output_type == DeepRagResponse

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
            assert result["documents"][0].page_content == "Test content 1"
