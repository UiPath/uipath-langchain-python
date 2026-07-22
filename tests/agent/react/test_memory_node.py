"""Tests for memory recall node and memory integration in create_agent."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables.graph import Edge
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field

from uipath_langchain.agent.multimodal import FileInfo
from uipath_langchain.agent.react.agent import create_agent
from uipath_langchain.agent.react.jsonschema_pydantic_converter import create_model
from uipath_langchain.agent.react.memory_node import (
    _ANALYSIS_MAX_RETRIES,
    _ANALYSIS_TIMEOUT_SECONDS,
    _build_analysis_model,
    _build_search_fields,
    _enabled_file_prompts,
    _fetch_space_settings,
    _normalize_attachment_items,
    _render_analysis_prompt,
    _top_level_attachment_field,
    _top_level_attachment_fields,
    create_memory_recall_node,
)
from uipath_langchain.agent.react.types import (
    AgentGraphNode,
    MemoryConfig,
)


def _attachment_input_model() -> type[BaseModel]:
    """Build an input model with a top-level attachment field and a text field."""
    schema = {
        "type": "object",
        "properties": {
            "vendorName": {"type": "string"},
            "invoiceDocument": {"$ref": "#/$defs/job-attachment"},
        },
        "$defs": {
            "job-attachment": {
                "type": "object",
                "x-uipath-ref-type": "job-attachment",
                "properties": {
                    "ID": {"type": "string"},
                    "FullName": {"type": "string"},
                    "MimeType": {"type": "string"},
                },
            }
        },
    }
    return create_model(schema)


_ATTACHMENT_DICT = {
    "ID": "11111111-1111-1111-1111-111111111111",
    "FullName": "invoice.pdf",
    "MimeType": "application/pdf",
}


def _file_settings(
    *,
    enabled: bool = True,
    analysis_prompt: str = "Summarize {FILE} for {AGENT_PROMPT}",
    agent_prompt_reference: str | None = "You triage invoices.",
    analysis_model: str | None = None,
    key_field: str = "invoiceDocument",
) -> dict[str, Any]:
    return {
        "memorySpaceId": "space-123",
        "agentPromptReference": agent_prompt_reference,
        "analysisModel": analysis_model,
        "fileAnalysis": [
            {
                "keyPath": ["agent-input", key_field],
                "enabled": enabled,
                "analysisPrompt": analysis_prompt,
            }
        ],
    }


class _TopicInput(BaseModel):
    """Reusable test input schema."""

    topic: str = Field(default="")


class _TopicLevelInput(BaseModel):
    """Reusable test input schema with two fields."""

    topic: str = Field(default="")
    level: str = Field(default="")


def _make_state(**user_fields: Any) -> dict[str, Any]:
    """Create a state dict with standard internal fields + user fields."""
    return {"messages": [], "inner_state": {}, **user_fields}


class TestBuildSearchFields:
    def test_basic_fields(self) -> None:
        fields = _build_search_fields({"topic": "python", "level": "advanced"})
        assert len(fields) == 2
        key_paths = [f.key_path for f in fields]
        assert ["agent-input", "topic"] in key_paths
        assert ["agent-input", "level"] in key_paths

    def test_filters_none_and_uipath_prefix(self) -> None:
        fields = _build_search_fields(
            {"topic": "py", "uipath__settings": {}, "empty": None}
        )
        assert len(fields) == 1
        assert fields[0].key_path == ["agent-input", "topic"]

    def test_empty_input(self) -> None:
        assert _build_search_fields({}) == []


class TestMemoryRecallNode:
    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.react.memory_node.UiPath")
    async def test_returns_injection_on_success(
        self, mock_uipath_cls: MagicMock
    ) -> None:
        mock_sdk = MagicMock()
        mock_uipath_cls.return_value = mock_sdk
        mock_response = MagicMock()
        mock_response.results = [MagicMock()]
        mock_response.system_prompt_injection = "\n\nBased on past runs..."
        mock_sdk.memory.search_async = AsyncMock(return_value=mock_response)

        config = MemoryConfig(memory_space_id="space-123", field_weights={"topic": 1.0})
        node = create_memory_recall_node(config, input_schema=_TopicInput)

        result = await node(_make_state(topic="python"))
        assert result["inner_state"]["memory_injection"] == "\n\nBased on past runs..."

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.react.memory_node.UiPath")
    async def test_returns_empty_on_failure(self, mock_uipath_cls: MagicMock) -> None:
        mock_sdk = MagicMock()
        mock_uipath_cls.return_value = mock_sdk
        mock_sdk.memory.search_async = AsyncMock(side_effect=Exception("fail"))

        config = MemoryConfig(memory_space_id="space-123", field_weights={"topic": 1.0})
        node = create_memory_recall_node(config, input_schema=_TopicInput)

        result = await node(_make_state(topic="python"))
        assert result == {}

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_user_inputs(self) -> None:
        config = MemoryConfig(memory_space_id="space-123", field_weights={"topic": 1.0})
        node = create_memory_recall_node(config, input_schema=_TopicInput)

        result = await node(_make_state())
        assert result == {}


class TestCreateAgentWithMemory:
    """Test that create_agent adds MEMORY_RECALL node when memory config is provided."""

    @pytest.fixture
    def mock_model(self) -> MagicMock:
        return MagicMock(spec=BaseChatModel)

    @pytest.fixture
    def messages(self) -> list[SystemMessage]:
        return [SystemMessage(content="You are a helpful assistant.")]

    @pytest.fixture
    def memory_config(self) -> MemoryConfig:
        return MemoryConfig(
            memory_space_id="test-space-id", field_weights={"topic": 1.0}
        )

    def test_graph_has_memory_recall_node(
        self,
        mock_model: MagicMock,
        messages: list[SystemMessage],
        memory_config: MemoryConfig,
    ) -> None:
        result: StateGraph[Any] = create_agent(
            mock_model, [], messages, memory=memory_config
        )
        graph = result.compile().get_graph()
        assert AgentGraphNode.MEMORY_RECALL in graph.nodes

    def test_graph_edges_with_memory(
        self,
        mock_model: MagicMock,
        messages: list[SystemMessage],
        memory_config: MemoryConfig,
    ) -> None:
        result: StateGraph[Any] = create_agent(
            mock_model, [], messages, memory=memory_config
        )
        graph = result.compile().get_graph()
        assert Edge("__start__", AgentGraphNode.MEMORY_RECALL) in graph.edges
        assert Edge(AgentGraphNode.MEMORY_RECALL, AgentGraphNode.INIT) in graph.edges
        assert Edge("__start__", AgentGraphNode.INIT) not in graph.edges

    def test_graph_without_memory_has_no_recall_node(
        self, mock_model: MagicMock, messages: list[SystemMessage]
    ) -> None:
        result: StateGraph[Any] = create_agent(mock_model, [], messages)
        graph = result.compile().get_graph()
        assert AgentGraphNode.MEMORY_RECALL not in graph.nodes
        assert Edge("__start__", AgentGraphNode.INIT) in graph.edges


class TestTopLevelAttachmentField:
    def test_single(self) -> None:
        assert _top_level_attachment_field("$.doc") == "doc"

    def test_list(self) -> None:
        assert _top_level_attachment_field("$.docs[*]") == "docs"

    def test_nested_single_out_of_scope(self) -> None:
        assert _top_level_attachment_field("$.a.b") is None

    def test_nested_array_out_of_scope(self) -> None:
        assert _top_level_attachment_field("$.a[*][*]") is None

    def test_non_dollar_prefix(self) -> None:
        assert _top_level_attachment_field("doc") is None

    def test_fields_from_input_model(self) -> None:
        assert _top_level_attachment_fields(_attachment_input_model()) == {
            "invoiceDocument"
        }


class TestEnabledFilePrompts:
    def test_enabled_valid(self) -> None:
        result = _enabled_file_prompts(_file_settings(), {"invoiceDocument": 1.0})
        assert result == {"invoiceDocument": "Summarize {FILE} for {AGENT_PROMPT}"}

    def test_disabled_excluded(self) -> None:
        assert _enabled_file_prompts(_file_settings(enabled=False), None) == {}

    def test_wrong_key_path_prefix_excluded(self) -> None:
        settings = _file_settings()
        settings["fileAnalysis"][0]["keyPath"] = ["tool-output", "invoiceDocument"]
        assert _enabled_file_prompts(settings, None) == {}

    def test_wrong_key_path_length_excluded(self) -> None:
        settings = _file_settings()
        settings["fileAnalysis"][0]["keyPath"] = ["agent-input"]
        assert _enabled_file_prompts(settings, None) == {}

    def test_empty_prompt_excluded(self) -> None:
        settings = _file_settings()
        settings["fileAnalysis"][0]["analysisPrompt"] = ""
        assert _enabled_file_prompts(settings, None) == {}

    def test_field_not_in_weights_excluded(self) -> None:
        assert _enabled_file_prompts(_file_settings(), {"vendorName": 1.0}) == {}

    def test_no_weights_includes_all(self) -> None:
        assert _enabled_file_prompts(_file_settings(), None) == {
            "invoiceDocument": "Summarize {FILE} for {AGENT_PROMPT}"
        }

    def test_missing_file_analysis_key(self) -> None:
        assert _enabled_file_prompts({"memorySpaceId": "x"}, None) == {}


class TestNormalizeAttachmentItems:
    def test_none(self) -> None:
        assert _normalize_attachment_items(None) == []

    def test_single_dict(self) -> None:
        assert _normalize_attachment_items({"ID": "1"}) == [{"ID": "1"}]

    def test_list_filters_falsy(self) -> None:
        assert _normalize_attachment_items([{"ID": "1"}, None, {}]) == [{"ID": "1"}]

    def test_empty_list(self) -> None:
        assert _normalize_attachment_items([]) == []


class TestRenderAnalysisPrompt:
    def test_both_placeholders(self) -> None:
        assert (
            _render_analysis_prompt("Read {FILE} as {AGENT_PROMPT}", "an agent")
            == "Read the attached file as an agent"
        )

    def test_agent_prompt_absent_removed(self) -> None:
        assert (
            _render_analysis_prompt("Read {FILE} as {AGENT_PROMPT}", None)
            == "Read the attached file as "
        )

    def test_no_placeholders(self) -> None:
        assert _render_analysis_prompt("Read it", "x") == "Read it"


class TestBuildAnalysisModel:
    def test_no_name_copies_agent_model(self) -> None:
        agent_model = MagicMock(spec=BaseChatModel)
        agent_model.model_copy = Mock(return_value="copied")
        assert _build_analysis_model(agent_model, None) == "copied"
        agent_model.model_copy.assert_called_once_with(
            update={"disable_streaming": True}
        )

    @patch("uipath_langchain.agent.react.memory_node.get_chat_model")
    def test_name_builds_pinned_model(self, get_chat_model: MagicMock) -> None:
        fresh = MagicMock(spec=BaseChatModel)
        fresh.model_copy = Mock(return_value="pinned")
        get_chat_model.return_value = fresh
        agent_model = MagicMock(spec=BaseChatModel)

        assert _build_analysis_model(agent_model, "gpt-4o") == "pinned"
        get_chat_model.assert_called_once_with(
            "gpt-4o",
            timeout=_ANALYSIS_TIMEOUT_SECONDS,
            max_retries=_ANALYSIS_MAX_RETRIES,
        )
        agent_model.model_copy.assert_not_called()

    @patch("uipath_langchain.agent.react.memory_node.get_chat_model")
    def test_name_build_failure_falls_back(self, get_chat_model: MagicMock) -> None:
        get_chat_model.side_effect = RuntimeError("no such model")
        agent_model = MagicMock(spec=BaseChatModel)
        agent_model.model_copy = Mock(return_value="fallback")

        assert _build_analysis_model(agent_model, "bad") == "fallback"
        agent_model.model_copy.assert_called_once_with(
            update={"disable_streaming": True}
        )


def _make_analysis_model(text: str) -> MagicMock:
    model = MagicMock(spec=BaseChatModel)
    model.ainvoke = AsyncMock(return_value=AIMessage(content=text))
    model.model_copy = Mock(return_value=model)
    return model


def _mock_search(uipath_cls: MagicMock) -> AsyncMock:
    sdk = MagicMock()
    response = MagicMock()
    response.results = [MagicMock()]
    response.system_prompt_injection = "INJECTION"
    sdk.memory.search_async = AsyncMock(return_value=response)
    uipath_cls.return_value = sdk
    return sdk.memory.search_async


def _field_values(request: Any) -> dict[tuple[str, ...], str]:
    return {tuple(f.key_path): f.value for f in request.fields}


def _search_request(search: AsyncMock) -> Any:
    assert search.await_args is not None
    return search.await_args.kwargs["request"]


class TestMemoryRecallFileAnalysis:
    """Query-side file-key analysis: derive ephemeral text and search on it."""

    @pytest.fixture
    def config(self) -> MemoryConfig:
        return MemoryConfig(
            memory_space_id="space-123",
            field_weights={"invoiceDocument": 1.0, "vendorName": 1.0},
        )

    def _node(self, config: MemoryConfig, model: MagicMock) -> Any:
        return create_memory_recall_node(
            config, input_schema=_attachment_input_model(), model=model
        )

    def _state(self) -> dict[str, Any]:
        return _make_state(vendorName="Acme", invoiceDocument=dict(_ATTACHMENT_DICT))

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.react.memory_node.UiPath")
    @patch(
        "uipath_langchain.agent.react.memory_node._fetch_space_settings",
        new_callable=AsyncMock,
    )
    async def test_settings_fetch_skipped_when_attachment_not_in_key(
        self,
        fetch: AsyncMock,
        uipath_cls: MagicMock,
    ) -> None:
        # The attachment field is excluded from the memory key (field_weights),
        # so it can never be analyzed — the settings GET must not fire at all.
        config = MemoryConfig(
            memory_space_id="space-123",
            field_weights={"vendorName": 1.0},
        )
        _mock_search(uipath_cls)
        model = _make_analysis_model("TEXT")

        await self._node(config, model)(self._state())

        fetch.assert_not_awaited()
        model.ainvoke.assert_not_awaited()

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.react.memory_node.UiPath")
    @patch(
        "uipath_langchain.agent.react.memory_node.add_files_to_message",
        new_callable=AsyncMock,
    )
    @patch(
        "uipath_langchain.agent.react.memory_node.resolve_attachments_to_file_infos",
        new_callable=AsyncMock,
    )
    @patch(
        "uipath_langchain.agent.react.memory_node._fetch_space_settings",
        new_callable=AsyncMock,
    )
    async def test_substitutes_derived_text_into_search_field(
        self,
        fetch: AsyncMock,
        resolve: AsyncMock,
        add_files: AsyncMock,
        uipath_cls: MagicMock,
        config: MemoryConfig,
    ) -> None:
        fetch.return_value = _file_settings()
        resolve.return_value = [
            FileInfo(url="u", name="invoice.pdf", mime_type="application/pdf")
        ]
        add_files.return_value = HumanMessage(content="<file>")
        search = _mock_search(uipath_cls)
        model = _make_analysis_model("INVOICE SUMMARY")

        state = self._state()
        result = await self._node(config, model)(state)

        request = _search_request(search)
        values = _field_values(request)
        assert values[("agent-input", "invoiceDocument")] == "INVOICE SUMMARY"
        assert values[("agent-input", "vendorName")] == "Acme"

        model.ainvoke.assert_awaited_once()
        system_message = model.ainvoke.await_args[0][0][0]
        assert isinstance(system_message, SystemMessage)
        assert (
            system_message.content
            == "Summarize the attached file for You triage invoices."
        )

        # state / agent input must keep the RAW attachment ref (never mutated)
        assert state["invoiceDocument"] == _ATTACHMENT_DICT
        assert result["inner_state"]["memory_injection"] == "INJECTION"

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.react.memory_node.UiPath")
    @patch(
        "uipath_langchain.agent.react.memory_node.add_files_to_message",
        new_callable=AsyncMock,
    )
    @patch(
        "uipath_langchain.agent.react.memory_node.resolve_attachments_to_file_infos",
        new_callable=AsyncMock,
    )
    @patch(
        "uipath_langchain.agent.react.memory_node._fetch_space_settings",
        new_callable=AsyncMock,
    )
    async def test_agent_prompt_absent_renders_empty(
        self,
        fetch: AsyncMock,
        resolve: AsyncMock,
        add_files: AsyncMock,
        uipath_cls: MagicMock,
        config: MemoryConfig,
    ) -> None:
        fetch.return_value = _file_settings(agent_prompt_reference=None)
        resolve.return_value = [
            FileInfo(url="u", name="invoice.pdf", mime_type="application/pdf")
        ]
        add_files.return_value = HumanMessage(content="<file>")
        _mock_search(uipath_cls)
        model = _make_analysis_model("TEXT")

        await self._node(config, model)(self._state())

        system_message = model.ainvoke.await_args[0][0][0]
        assert system_message.content == "Summarize the attached file for "

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.react.memory_node.UiPath")
    @patch(
        "uipath_langchain.agent.react.memory_node.resolve_attachments_to_file_infos",
        new_callable=AsyncMock,
    )
    @patch(
        "uipath_langchain.agent.react.memory_node._fetch_space_settings",
        new_callable=AsyncMock,
    )
    async def test_settings_fetch_failure_skips_analysis(
        self,
        fetch: AsyncMock,
        resolve: AsyncMock,
        uipath_cls: MagicMock,
        config: MemoryConfig,
    ) -> None:
        fetch.return_value = None
        search = _mock_search(uipath_cls)
        model = _make_analysis_model("SHOULD NOT RUN")

        await self._node(config, model)(self._state())

        model.ainvoke.assert_not_awaited()
        resolve.assert_not_awaited()
        values = _field_values(_search_request(search))
        # falls back to today's behavior: the raw attachment ref is searched
        assert (
            "11111111-1111-1111-1111-111111111111"
            in values[("agent-input", "invoiceDocument")]
        )

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.react.memory_node.UiPath")
    @patch(
        "uipath_langchain.agent.react.memory_node.resolve_attachments_to_file_infos",
        new_callable=AsyncMock,
    )
    @patch(
        "uipath_langchain.agent.react.memory_node._fetch_space_settings",
        new_callable=AsyncMock,
    )
    async def test_disabled_field_untouched(
        self,
        fetch: AsyncMock,
        resolve: AsyncMock,
        uipath_cls: MagicMock,
        config: MemoryConfig,
    ) -> None:
        fetch.return_value = _file_settings(enabled=False)
        search = _mock_search(uipath_cls)
        model = _make_analysis_model("SHOULD NOT RUN")

        await self._node(config, model)(self._state())

        model.ainvoke.assert_not_awaited()
        resolve.assert_not_awaited()
        values = _field_values(_search_request(search))
        assert (
            "11111111-1111-1111-1111-111111111111"
            in values[("agent-input", "invoiceDocument")]
        )

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.react.memory_node.UiPath")
    @patch(
        "uipath_langchain.agent.react.memory_node.add_files_to_message",
        new_callable=AsyncMock,
    )
    @patch(
        "uipath_langchain.agent.react.memory_node.resolve_attachments_to_file_infos",
        new_callable=AsyncMock,
    )
    @patch(
        "uipath_langchain.agent.react.memory_node._fetch_space_settings",
        new_callable=AsyncMock,
    )
    async def test_empty_output_skips_field_entirely(
        self,
        fetch: AsyncMock,
        resolve: AsyncMock,
        add_files: AsyncMock,
        uipath_cls: MagicMock,
        config: MemoryConfig,
    ) -> None:
        fetch.return_value = _file_settings()
        resolve.return_value = [
            FileInfo(url="u", name="invoice.pdf", mime_type="application/pdf")
        ]
        add_files.return_value = HumanMessage(content="<file>")
        search = _mock_search(uipath_cls)
        model = _make_analysis_model("   ")  # whitespace only

        await self._node(config, model)(self._state())

        values = _field_values(_search_request(search))
        # enabled file field with no usable text is dropped, never sent as raw ref
        assert ("agent-input", "invoiceDocument") not in values
        assert values[("agent-input", "vendorName")] == "Acme"

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.react.memory_node.UiPath")
    @patch(
        "uipath_langchain.agent.react.memory_node.resolve_attachments_to_file_infos",
        new_callable=AsyncMock,
    )
    @patch(
        "uipath_langchain.agent.react.memory_node._fetch_space_settings",
        new_callable=AsyncMock,
    )
    async def test_analysis_exception_skips_field(
        self,
        fetch: AsyncMock,
        resolve: AsyncMock,
        uipath_cls: MagicMock,
        config: MemoryConfig,
    ) -> None:
        fetch.return_value = _file_settings()
        resolve.side_effect = RuntimeError("resolve boom")
        search = _mock_search(uipath_cls)
        model = _make_analysis_model("unused")

        result = await self._node(config, model)(self._state())

        values = _field_values(_search_request(search))
        assert ("agent-input", "invoiceDocument") not in values
        assert values[("agent-input", "vendorName")] == "Acme"
        # node still returns the recall injection despite the per-field failure
        assert result["inner_state"]["memory_injection"] == "INJECTION"

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.react.memory_node.UiPath")
    @patch(
        "uipath_langchain.agent.react.memory_node.resolve_attachments_to_file_infos",
        new_callable=AsyncMock,
    )
    @patch(
        "uipath_langchain.agent.react.memory_node._fetch_space_settings",
        new_callable=AsyncMock,
    )
    async def test_unresolvable_attachment_skips_field(
        self,
        fetch: AsyncMock,
        resolve: AsyncMock,
        uipath_cls: MagicMock,
        config: MemoryConfig,
    ) -> None:
        fetch.return_value = _file_settings()
        resolve.return_value = []  # nothing resolvable → no file to analyze
        search = _mock_search(uipath_cls)
        model = _make_analysis_model("SHOULD NOT RUN")

        await self._node(config, model)(self._state())

        # no files means no LLM call and the field is dropped, not sent as raw ref
        model.ainvoke.assert_not_awaited()
        values = _field_values(_search_request(search))
        assert ("agent-input", "invoiceDocument") not in values
        assert values[("agent-input", "vendorName")] == "Acme"

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.react.memory_node.get_chat_model")
    @patch("uipath_langchain.agent.react.memory_node.UiPath")
    @patch(
        "uipath_langchain.agent.react.memory_node.add_files_to_message",
        new_callable=AsyncMock,
    )
    @patch(
        "uipath_langchain.agent.react.memory_node.resolve_attachments_to_file_infos",
        new_callable=AsyncMock,
    )
    @patch(
        "uipath_langchain.agent.react.memory_node._fetch_space_settings",
        new_callable=AsyncMock,
    )
    async def test_analysis_model_pinned_from_settings(
        self,
        fetch: AsyncMock,
        resolve: AsyncMock,
        add_files: AsyncMock,
        uipath_cls: MagicMock,
        get_chat_model: MagicMock,
        config: MemoryConfig,
    ) -> None:
        fetch.return_value = _file_settings(analysis_model="gpt-4o")
        resolve.return_value = [
            FileInfo(url="u", name="invoice.pdf", mime_type="application/pdf")
        ]
        add_files.return_value = HumanMessage(content="<file>")
        search = _mock_search(uipath_cls)

        pinned = _make_analysis_model("PINNED TEXT")
        get_chat_model.return_value = pinned
        agent_model = _make_analysis_model("AGENT TEXT")

        await self._node(config, agent_model)(self._state())

        get_chat_model.assert_called_once_with(
            "gpt-4o",
            timeout=_ANALYSIS_TIMEOUT_SECONDS,
            max_retries=_ANALYSIS_MAX_RETRIES,
        )
        pinned.ainvoke.assert_awaited_once()
        agent_model.ainvoke.assert_not_awaited()
        values = _field_values(_search_request(search))
        assert values[("agent-input", "invoiceDocument")] == "PINNED TEXT"

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.react.memory_node.get_chat_model")
    @patch("uipath_langchain.agent.react.memory_node.UiPath")
    @patch(
        "uipath_langchain.agent.react.memory_node.add_files_to_message",
        new_callable=AsyncMock,
    )
    @patch(
        "uipath_langchain.agent.react.memory_node.resolve_attachments_to_file_infos",
        new_callable=AsyncMock,
    )
    @patch(
        "uipath_langchain.agent.react.memory_node._fetch_space_settings",
        new_callable=AsyncMock,
    )
    async def test_analysis_model_falls_back_to_agent_model(
        self,
        fetch: AsyncMock,
        resolve: AsyncMock,
        add_files: AsyncMock,
        uipath_cls: MagicMock,
        get_chat_model: MagicMock,
        config: MemoryConfig,
    ) -> None:
        fetch.return_value = _file_settings(analysis_model="unavailable")
        resolve.return_value = [
            FileInfo(url="u", name="invoice.pdf", mime_type="application/pdf")
        ]
        add_files.return_value = HumanMessage(content="<file>")
        search = _mock_search(uipath_cls)
        get_chat_model.side_effect = RuntimeError("model not found")
        agent_model = _make_analysis_model("AGENT TEXT")

        await self._node(config, agent_model)(self._state())

        agent_model.ainvoke.assert_awaited_once()
        values = _field_values(_search_request(search))
        assert values[("agent-input", "invoiceDocument")] == "AGENT TEXT"

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.react.memory_node.UiPath")
    @patch(
        "uipath_langchain.agent.react.memory_node.resolve_attachments_to_file_infos",
        new_callable=AsyncMock,
    )
    @patch(
        "uipath_langchain.agent.react.memory_node._fetch_space_settings",
        new_callable=AsyncMock,
    )
    async def test_no_model_skips_analysis(
        self,
        fetch: AsyncMock,
        resolve: AsyncMock,
        uipath_cls: MagicMock,
        config: MemoryConfig,
    ) -> None:
        # model=None → file analysis pre-step is skipped, settings never fetched
        _mock_search(uipath_cls)
        node = create_memory_recall_node(
            config, input_schema=_attachment_input_model(), model=None
        )

        await node(self._state())

        fetch.assert_not_awaited()
        resolve.assert_not_awaited()

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.react.memory_node.UiPath")
    @patch(
        "uipath_langchain.agent.react.memory_node._fetch_space_settings",
        new_callable=AsyncMock,
    )
    async def test_text_only_agent_skips_settings_fetch(
        self,
        fetch: AsyncMock,
        uipath_cls: MagicMock,
    ) -> None:
        # An agent whose input schema declares no attachment fields must not pay
        # the settings GET, even with a model available — recall stays
        # byte-identical to the pre-file-analysis behavior.
        _mock_search(uipath_cls)
        config = MemoryConfig(memory_space_id="space-123", field_weights={"topic": 1.0})
        node = create_memory_recall_node(
            config, input_schema=_TopicInput, model=_make_analysis_model("x")
        )

        result = await node(_make_state(topic="python"))

        fetch.assert_not_awaited()
        assert result["inner_state"]["memory_injection"] == "INJECTION"


class TestFetchSpaceSettings:
    """Direct coverage of the settings GET soft-fail paths."""

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.react.memory_node.UiPath")
    async def test_success_returns_json(self, uipath_cls: MagicMock) -> None:
        sdk = MagicMock()
        response = MagicMock()
        response.json.return_value = {"memorySpaceId": "space-123"}
        sdk.api_client.request_async = AsyncMock(return_value=response)
        uipath_cls.return_value = sdk

        result = await _fetch_space_settings("space-123")

        assert result == {"memorySpaceId": "space-123"}
        args, kwargs = sdk.api_client.request_async.await_args
        assert args[0] == "GET"
        assert args[1] == "/llmopstenant_/api/Memory/space-123/settings"
        assert kwargs["include_folder_headers"] is True

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.react.memory_node.UiPath")
    async def test_http_error_returns_none(self, uipath_cls: MagicMock) -> None:
        sdk = MagicMock()
        response = MagicMock()
        response.raise_for_status.side_effect = RuntimeError("500 server error")
        sdk.api_client.request_async = AsyncMock(return_value=response)
        uipath_cls.return_value = sdk

        assert await _fetch_space_settings("space-123") is None

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.react.memory_node.UiPath")
    async def test_malformed_json_returns_none(self, uipath_cls: MagicMock) -> None:
        sdk = MagicMock()
        response = MagicMock()
        response.json.side_effect = ValueError("not json")
        sdk.api_client.request_async = AsyncMock(return_value=response)
        uipath_cls.return_value = sdk

        assert await _fetch_space_settings("space-123") is None

    @pytest.mark.asyncio
    @patch("uipath_langchain.agent.react.memory_node.UiPath")
    async def test_request_exception_returns_none(self, uipath_cls: MagicMock) -> None:
        sdk = MagicMock()
        sdk.api_client.request_async = AsyncMock(side_effect=RuntimeError("boom"))
        uipath_cls.return_value = sdk

        assert await _fetch_space_settings("space-123") is None
