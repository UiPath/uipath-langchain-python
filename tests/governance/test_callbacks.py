"""Tests for the LangChain governance callback handler."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, Generation, LLMResult
from uipath.core.governance.exceptions import GovernanceBlockException

from uipath_langchain.governance import GovernanceCallbackHandler
from uipath_langchain.governance.callbacks import _BEFORE_MODEL_TEXT_CAP

LOGGER_PATH = "uipath_langchain.governance.callbacks.logger"


@pytest.fixture
def evaluator() -> MagicMock:
    return MagicMock()


@pytest.fixture
def handler(evaluator: MagicMock) -> GovernanceCallbackHandler:
    return GovernanceCallbackHandler(
        evaluator=evaluator,
        agent_name="test-agent",
        session_id="test-session",
    )


class TestSubclassesBaseCallbackHandler:
    def test_is_base_callback_handler(self, handler: GovernanceCallbackHandler) -> None:
        # The handler must be a real LangChain BaseCallbackHandler so
        # LangChain's dispatch / tracer wiring treats it natively.
        assert isinstance(handler, BaseCallbackHandler)

    def test_ignore_flags_override_parent_properties(
        self, handler: GovernanceCallbackHandler
    ) -> None:
        # Chain notifications skipped — the governance host owns
        # BEFORE_AGENT / AFTER_AGENT and would otherwise double-fire.
        assert handler.ignore_chain is True
        assert handler.ignore_retriever is True
        assert handler.ignore_retry is True
        assert handler.ignore_custom_event is True
        # LLM / chat model / tool / agent events stay on.
        assert handler.ignore_llm is False
        assert handler.ignore_chat_model is False
        assert handler.ignore_agent is False


class TestCallbackHandlerLLM:
    def test_on_llm_start_invokes_evaluator_with_latest_prompt(
        self, handler: GovernanceCallbackHandler, evaluator: MagicMock
    ) -> None:
        """Only the latest prompt feeds BEFORE_MODEL — prior prompts in a
        batched call would re-fire rules on content the LLM has
        already responded to in earlier batches."""
        handler.on_llm_start({"name": "m"}, ["a", "b"])
        evaluator.evaluate_before_model.assert_called_once()
        kwargs = evaluator.evaluate_before_model.call_args.kwargs
        assert kwargs["model_input"] == "b"
        assert kwargs["agent_name"] == "test-agent"
        assert kwargs["runtime_id"] == "test-session"
        # ``trace_id`` is intentionally NOT passed — correlation is
        # owned by the layer below the evaluator. The handler is
        # env-free.
        assert "trace_id" not in kwargs

    def test_on_llm_start_increments_counter(
        self, handler: GovernanceCallbackHandler
    ) -> None:
        handler.on_llm_start({}, ["p"])
        handler.on_llm_start({}, ["p"])
        assert handler._session_state["llm_calls"] == 2

    def test_on_llm_start_empty_prompts(
        self, handler: GovernanceCallbackHandler, evaluator: MagicMock
    ) -> None:
        handler.on_llm_start({}, [])
        assert evaluator.evaluate_before_model.call_args.kwargs["model_input"] == ""

    def test_on_llm_start_propagates_block(
        self, handler: GovernanceCallbackHandler, evaluator: MagicMock
    ) -> None:
        evaluator.evaluate_before_model.side_effect = GovernanceBlockException(
            "blocked"
        )
        with pytest.raises(GovernanceBlockException):
            handler.on_llm_start({}, ["p"])

    def test_on_llm_start_swallows_other_exceptions(
        self,
        handler: GovernanceCallbackHandler,
        evaluator: MagicMock,
    ) -> None:
        evaluator.evaluate_before_model.side_effect = RuntimeError("nope")
        with patch(LOGGER_PATH) as mock_logger:
            handler.on_llm_start({}, ["p"])  # must not raise
        mock_logger.warning.assert_called_once()
        assert "on_llm_start" in mock_logger.warning.call_args.args[0]

    def test_on_chat_model_start_latest_message_only(
        self, handler: GovernanceCallbackHandler, evaluator: MagicMock
    ) -> None:
        """Only the LAST message in the prompt is scanned.

        Without this scoping, a violation in turn 3's user message
        would keep re-firing on every subsequent LLM call because
        that text stays in the prompt for context.
        """
        handler.on_chat_model_start(
            {},
            [[SimpleNamespace(content="hello"), SimpleNamespace(content="world")]],
        )
        model_input = evaluator.evaluate_before_model.call_args.kwargs["model_input"]
        assert model_input == "world"
        assert "hello" not in model_input

    def test_on_chat_model_start_dict_messages_latest_only(
        self, handler: GovernanceCallbackHandler, evaluator: MagicMock
    ) -> None:
        """Dict-shaped (LangGraph state) messages: latest is extracted."""
        handler.on_chat_model_start(
            {},
            [[{"content": "from dict"}, {"role": "user", "content": "another"}]],
        )
        model_input = evaluator.evaluate_before_model.call_args.kwargs["model_input"]
        assert model_input == "another"
        assert "from dict" not in model_input

    def test_on_chat_model_start_dict_message_missing_content(
        self, handler: GovernanceCallbackHandler, evaluator: MagicMock
    ) -> None:
        handler.on_chat_model_start({}, [[{"role": "user"}]])
        assert evaluator.evaluate_before_model.call_args.kwargs["model_input"] == ""

    def test_on_chat_model_start_list_of_blocks_content(
        self, handler: GovernanceCallbackHandler, evaluator: MagicMock
    ) -> None:
        """Multi-block content (text + function_call) is extracted cleanly.

        Regression for the prior ``str(msg.content)`` path which produced
        ``[{'type': ..., 'text': ...}]`` dict-repr noise instead of
        clean text. Field-precise rules can't navigate that shape.
        """
        latest = SimpleNamespace(
            content=[
                {"type": "text", "text": "Here's the answer:"},
                {
                    "type": "function_call",
                    "name": "end_execution",
                    "arguments": '{"content":"Cost: $1,200"}',
                    "id": "fc_abc",
                },
            ]
        )
        handler.on_chat_model_start({}, [[SimpleNamespace(content="old"), latest]])
        model_input = evaluator.evaluate_before_model.call_args.kwargs["model_input"]
        assert "Here's the answer:" in model_input
        assert "Cost: $1,200" in model_input
        # No dict-syntax noise from str(list).
        assert "{'type'" not in model_input

    def test_on_chat_model_start_empty_messages(
        self, handler: GovernanceCallbackHandler, evaluator: MagicMock
    ) -> None:
        handler.on_chat_model_start({}, [])
        assert evaluator.evaluate_before_model.call_args.kwargs["model_input"] == ""

    def test_on_chat_model_start_empty_inner_batch(
        self, handler: GovernanceCallbackHandler, evaluator: MagicMock
    ) -> None:
        handler.on_chat_model_start({}, [[]])
        assert evaluator.evaluate_before_model.call_args.kwargs["model_input"] == ""

    def test_on_chat_model_start_caps_model_input(
        self, handler: GovernanceCallbackHandler, evaluator: MagicMock
    ) -> None:
        """``model_input`` is bounded so a runaway prompt can't dominate scan time."""
        huge = SimpleNamespace(content="x" * (_BEFORE_MODEL_TEXT_CAP + 1000))
        handler.on_chat_model_start({}, [[huge]])
        model_input = evaluator.evaluate_before_model.call_args.kwargs["model_input"]
        assert len(model_input) == _BEFORE_MODEL_TEXT_CAP

    def test_on_chat_model_start_block_list_stops_at_remaining_budget(
        self, handler: GovernanceCallbackHandler, evaluator: MagicMock
    ) -> None:
        """The block walk exits early once the per-call cap is exhausted."""
        first = "a" * _BEFORE_MODEL_TEXT_CAP  # consumes the entire budget
        latest = SimpleNamespace(
            content=[
                {"type": "text", "text": first},
                {"type": "text", "text": "MUST_NOT_APPEAR"},
            ]
        )
        handler.on_chat_model_start({}, [[latest]])
        model_input = evaluator.evaluate_before_model.call_args.kwargs["model_input"]
        assert "MUST_NOT_APPEAR" not in model_input
        assert len(model_input) == _BEFORE_MODEL_TEXT_CAP

    def test_on_chat_model_start_block_list_skips_non_dict_entries(
        self, handler: GovernanceCallbackHandler, evaluator: MagicMock
    ) -> None:
        """Non-dict entries inside a content list are silently skipped."""
        latest = SimpleNamespace(
            content=[
                "ignored-string-block",
                {"type": "text", "text": "kept"},
                42,
                None,
            ]
        )
        handler.on_chat_model_start({}, [[latest]])
        model_input = evaluator.evaluate_before_model.call_args.kwargs["model_input"]
        assert model_input == "kept"

    def test_on_chat_model_start_propagates_block(
        self, handler: GovernanceCallbackHandler, evaluator: MagicMock
    ) -> None:
        evaluator.evaluate_before_model.side_effect = GovernanceBlockException("x")
        with pytest.raises(GovernanceBlockException):
            handler.on_chat_model_start({}, [[SimpleNamespace(content="x")]])

    def test_on_chat_model_start_swallows_other_exceptions(
        self,
        handler: GovernanceCallbackHandler,
        evaluator: MagicMock,
    ) -> None:
        evaluator.evaluate_before_model.side_effect = RuntimeError("oops")
        with patch(LOGGER_PATH) as mock_logger:
            handler.on_chat_model_start({}, [[SimpleNamespace(content="x")]])
        mock_logger.warning.assert_called_once()
        assert "on_chat_model_start" in mock_logger.warning.call_args.args[0]

    def test_on_llm_end_extracts_plain_text(
        self, handler: GovernanceCallbackHandler, evaluator: MagicMock
    ) -> None:
        response = LLMResult(generations=[[Generation(text="output")]])
        handler.on_llm_end(response)
        kwargs = evaluator.evaluate_after_model.call_args.kwargs
        assert kwargs["model_output"] == "output"

    def test_on_llm_end_response_without_generations(
        self, handler: GovernanceCallbackHandler, evaluator: MagicMock
    ) -> None:
        handler.on_llm_end(LLMResult(generations=[]))
        assert evaluator.evaluate_after_model.call_args.kwargs["model_output"] == ""

    def test_on_llm_end_propagates_block(
        self, handler: GovernanceCallbackHandler, evaluator: MagicMock
    ) -> None:
        evaluator.evaluate_after_model.side_effect = GovernanceBlockException("x")
        with pytest.raises(GovernanceBlockException):
            handler.on_llm_end(LLMResult(generations=[]))

    def test_on_llm_end_caps_model_output(
        self, handler: GovernanceCallbackHandler, evaluator: MagicMock
    ) -> None:
        """A runaway / batched response is capped so the AFTER_MODEL
        scan budget matches BEFORE_MODEL and the runtime side's cap.
        """
        # Many large generations across batched gen_lists.
        big = "y" * 50_000
        response = LLMResult(
            generations=[
                [Generation(text=big)],
                [Generation(text=big), Generation(text=big)],
            ]
        )
        handler.on_llm_end(response)
        model_output = evaluator.evaluate_after_model.call_args.kwargs["model_output"]
        assert len(model_output) == _BEFORE_MODEL_TEXT_CAP

    def test_on_llm_end_skips_empty_generation_text(
        self, handler: GovernanceCallbackHandler, evaluator: MagicMock
    ) -> None:
        """Generations with no extractable text don't bloat the output."""
        response = LLMResult(
            generations=[[Generation(text=""), Generation(text="kept")]]
        )
        handler.on_llm_end(response)
        assert evaluator.evaluate_after_model.call_args.kwargs["model_output"] == "kept"

    def test_on_llm_end_swallows_other_exceptions(
        self,
        handler: GovernanceCallbackHandler,
        evaluator: MagicMock,
    ) -> None:
        evaluator.evaluate_after_model.side_effect = RuntimeError("nope")
        with patch(LOGGER_PATH) as mock_logger:
            handler.on_llm_end(LLMResult(generations=[]))
        mock_logger.warning.assert_called_once()
        assert "on_llm_end" in mock_logger.warning.call_args.args[0]

    def test_on_llm_error_logs(
        self,
        handler: GovernanceCallbackHandler,
    ) -> None:
        with patch(LOGGER_PATH) as mock_logger:
            handler.on_llm_error(RuntimeError("boom"))
        mock_logger.warning.assert_called_once()
        assert "LLM error" in mock_logger.warning.call_args.args[0]


class TestExtractGenerationText:
    def test_returns_text_for_plain_generation(self) -> None:
        gen = Generation(text="hello")
        assert GovernanceCallbackHandler._extract_generation_text(gen) == "hello"

    def test_chat_generation_string_content(self) -> None:
        gen = ChatGeneration(message=AIMessage(content="rich"))
        assert GovernanceCallbackHandler._extract_generation_text(gen) == "rich"

    def test_returns_empty_when_generation_text_empty(self) -> None:
        assert (
            GovernanceCallbackHandler._extract_generation_text(Generation(text=""))
            == ""
        )

    def test_extracts_from_block_list_content(self) -> None:
        gen = ChatGeneration(
            message=AIMessage(
                content=[
                    {"type": "text", "text": "alpha"},
                    {"type": "tool_use", "input": {"q": "beta"}},
                ]
            )
        )
        out = GovernanceCallbackHandler._extract_generation_text(gen)
        assert "alpha" in out
        assert "beta" in out

    def test_block_list_skips_non_dict_entries(self) -> None:
        gen = ChatGeneration(
            message=AIMessage(
                content=["string-entry", {"type": "text", "text": "kept"}]
            )
        )
        assert GovernanceCallbackHandler._extract_generation_text(gen) == "kept"

    def test_chat_generation_with_empty_block_list_falls_back_to_text(self) -> None:
        """When all blocks yield no text, fall back to ``gen.text``."""
        gen = ChatGeneration(message=AIMessage(content=[]))
        assert GovernanceCallbackHandler._extract_generation_text(gen) == ""


class TestExtractBlockText:
    def test_plain_text_block(self) -> None:
        assert (
            GovernanceCallbackHandler._extract_block_text(
                {"type": "text", "text": "hello"}
            )
            == "hello"
        )

    def test_function_call_arguments_block(self) -> None:
        assert (
            GovernanceCallbackHandler._extract_block_text(
                {"type": "function_call", "arguments": '{"a":1}'}
            )
            == '{"a":1}'
        )

    def test_thinking_block(self) -> None:
        assert (
            GovernanceCallbackHandler._extract_block_text(
                {"type": "thinking", "thinking": "step by step"}
            )
            == "step by step"
        )

    def test_tool_use_input_extracts_string_values(self) -> None:
        result = GovernanceCallbackHandler._extract_block_text(
            {"type": "tool_use", "input": {"query": "search", "id": "ignored"}}
        )
        assert "search" in result
        assert "ignored" in result  # both are strings; metadata filtering is by key

    def test_input_ignores_non_string_values(self) -> None:
        result = GovernanceCallbackHandler._extract_block_text(
            {"input": {"a": 123, "b": ["nested"], "c": "kept"}}
        )
        assert result == "kept"

    def test_metadata_only_block_returns_empty(self) -> None:
        assert (
            GovernanceCallbackHandler._extract_block_text(
                {"type": "tool_use", "id": "abc", "name": "search", "status": "ok"}
            )
            == ""
        )

    def test_combined_fields_all_collected(self) -> None:
        result = GovernanceCallbackHandler._extract_block_text(
            {
                "type": "tool_use",
                "text": "T",
                "arguments": "A",
                "thinking": "Th",
                "input": {"k": "I"},
            }
        )
        for token in ("T", "A", "Th", "I"):
            assert token in result

    def test_empty_block(self) -> None:
        assert GovernanceCallbackHandler._extract_block_text({}) == ""


class TestCallbackHandlerTools:
    def test_on_tool_start_with_inputs(
        self, handler: GovernanceCallbackHandler, evaluator: MagicMock
    ) -> None:
        handler.on_tool_start({"name": "search"}, "fallback", inputs={"q": "v"})
        kwargs = evaluator.evaluate_tool_call.call_args.kwargs
        assert kwargs["tool_name"] == "search"
        assert kwargs["tool_args"] == {"q": "v"}
        assert kwargs["session_state"] is handler._session_state

    def test_on_tool_start_without_inputs_uses_input_str(
        self, handler: GovernanceCallbackHandler, evaluator: MagicMock
    ) -> None:
        handler.on_tool_start({"name": "calc"}, "1+2")
        kwargs = evaluator.evaluate_tool_call.call_args.kwargs
        assert kwargs["tool_args"] == {"input": "1+2"}

    def test_on_tool_start_unknown_name_when_missing(
        self, handler: GovernanceCallbackHandler, evaluator: MagicMock
    ) -> None:
        handler.on_tool_start({}, "x")
        assert evaluator.evaluate_tool_call.call_args.kwargs["tool_name"] == "unknown"

    def test_on_tool_start_increments_counter(
        self, handler: GovernanceCallbackHandler
    ) -> None:
        handler.on_tool_start({}, "x")
        handler.on_tool_start({}, "y")
        assert handler._session_state["tool_calls"] == 2

    def test_on_tool_start_propagates_block(
        self, handler: GovernanceCallbackHandler, evaluator: MagicMock
    ) -> None:
        evaluator.evaluate_tool_call.side_effect = GovernanceBlockException("no")
        with pytest.raises(GovernanceBlockException):
            handler.on_tool_start({}, "x")

    def test_on_tool_start_swallows_other_exceptions(
        self,
        handler: GovernanceCallbackHandler,
        evaluator: MagicMock,
    ) -> None:
        evaluator.evaluate_tool_call.side_effect = RuntimeError("nope")
        with patch(LOGGER_PATH) as mock_logger:
            handler.on_tool_start({}, "x")
        mock_logger.warning.assert_called_once()
        assert "on_tool_start" in mock_logger.warning.call_args.args[0]

    def test_on_tool_end_with_output(
        self, handler: GovernanceCallbackHandler, evaluator: MagicMock
    ) -> None:
        handler.on_tool_end({"answer": 42})
        kwargs = evaluator.evaluate_after_tool.call_args.kwargs
        assert "42" in kwargs["tool_result"]
        assert kwargs["tool_name"] == "unknown"

    def test_on_tool_end_uses_tool_name_from_run_id(
        self, handler: GovernanceCallbackHandler, evaluator: MagicMock
    ) -> None:
        handler.on_tool_start({"name": "search"}, "q", run_id="run-1")
        handler.on_tool_end("result", run_id="run-1")
        assert evaluator.evaluate_after_tool.call_args.kwargs["tool_name"] == "search"
        # The run_id mapping is cleaned up so a stale entry isn't reused.
        assert "run-1" not in handler._tool_runs

    def test_on_tool_end_unknown_when_run_id_not_recorded(
        self, handler: GovernanceCallbackHandler, evaluator: MagicMock
    ) -> None:
        handler.on_tool_end("r", run_id="never-started")
        assert evaluator.evaluate_after_tool.call_args.kwargs["tool_name"] == "unknown"

    def test_on_tool_start_handles_none_serialized(
        self, handler: GovernanceCallbackHandler, evaluator: MagicMock
    ) -> None:
        handler.on_tool_start(None, "x")  # type: ignore[arg-type]
        assert evaluator.evaluate_tool_call.call_args.kwargs["tool_name"] == "unknown"

    def test_on_tool_end_with_none_output(
        self, handler: GovernanceCallbackHandler, evaluator: MagicMock
    ) -> None:
        handler.on_tool_end(None)
        assert evaluator.evaluate_after_tool.call_args.kwargs["tool_result"] == ""

    def test_on_tool_end_propagates_block(
        self, handler: GovernanceCallbackHandler, evaluator: MagicMock
    ) -> None:
        evaluator.evaluate_after_tool.side_effect = GovernanceBlockException("x")
        with pytest.raises(GovernanceBlockException):
            handler.on_tool_end("out")

    def test_on_tool_end_swallows_other_exceptions(
        self,
        handler: GovernanceCallbackHandler,
        evaluator: MagicMock,
    ) -> None:
        evaluator.evaluate_after_tool.side_effect = RuntimeError("err")
        with patch(LOGGER_PATH) as mock_logger:
            handler.on_tool_end("out")
        mock_logger.warning.assert_called_once()
        assert "on_tool_end" in mock_logger.warning.call_args.args[0]

    def test_on_tool_error_logs(
        self,
        handler: GovernanceCallbackHandler,
    ) -> None:
        with patch(LOGGER_PATH) as mock_logger:
            handler.on_tool_error(RuntimeError("broke"))
        mock_logger.warning.assert_called_once()
        assert "Tool error" in mock_logger.warning.call_args.args[0]

    def test_on_tool_error_pops_run_id_mapping(
        self, handler: GovernanceCallbackHandler
    ) -> None:
        """``on_tool_error`` cleans up ``_tool_runs`` so failed tool calls
        don't accumulate over the lifetime of a governed session.
        """
        handler.on_tool_start({"name": "search"}, "q", run_id="run-err")
        assert handler._tool_runs.get("run-err") == "search"
        handler.on_tool_error(RuntimeError("boom"), run_id="run-err")
        assert "run-err" not in handler._tool_runs

    def test_on_tool_error_without_run_id_does_not_crash(
        self, handler: GovernanceCallbackHandler
    ) -> None:
        # No run_id kwargs — should still log and not raise.
        handler.on_tool_error(RuntimeError("boom"))
        assert handler._tool_runs == {}

    def test_on_tool_start_block_pops_run_id_mapping(
        self, handler: GovernanceCallbackHandler, evaluator: MagicMock
    ) -> None:
        """If BEFORE_TOOL evaluation BLOCKS, the recorded mapping is
        dropped — the tool never runs and ``on_tool_end`` will not fire.
        Leaving the entry would leak across blocked turns.
        """
        evaluator.evaluate_tool_call.side_effect = GovernanceBlockException("nope")
        with pytest.raises(GovernanceBlockException):
            handler.on_tool_start({"name": "search"}, "q", run_id="run-blocked")
        assert "run-blocked" not in handler._tool_runs

    def test_on_tool_start_swallowed_error_preserves_mapping(
        self, handler: GovernanceCallbackHandler, evaluator: MagicMock
    ) -> None:
        """When the evaluator raises a non-block exception, we swallow
        and the tool still runs — the mapping must survive so
        ``on_tool_end`` can resolve the tool name.
        """
        evaluator.evaluate_tool_call.side_effect = RuntimeError("flaky")
        with patch(LOGGER_PATH):
            handler.on_tool_start({"name": "search"}, "q", run_id="run-flaky")
        assert handler._tool_runs.get("run-flaky") == "search"


class TestCallbackHandlerInit:
    def test_session_state_initialized(self, evaluator: MagicMock) -> None:
        h = GovernanceCallbackHandler(
            evaluator=evaluator, agent_name="a", session_id="s"
        )
        assert h._session_state == {"tool_calls": 0, "llm_calls": 0}
        assert h._agent_name == "a"
        assert h._session_id == "s"
