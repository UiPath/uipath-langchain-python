"""LangChain governance callback handler.

A :class:`langchain_core.callbacks.BaseCallbackHandler` that calls a
framework-agnostic :class:`~uipath.core.adapters.EvaluatorProtocol`
on the model and tool lifecycle.

Wiring lives in :class:`UiPathLangGraphRuntimeFactory`: passing an
``evaluator`` to ``new_runtime`` causes the factory to build this
handler and hand it to :class:`UiPathLangGraphRuntime` through the
existing ``callbacks`` constructor arg. No adapter registry, no global
state, no import-time mutation.

Intercepts:

- ``on_llm_start`` / ``on_chat_model_start`` / ``on_llm_end`` → BEFORE_MODEL / AFTER_MODEL
- ``on_tool_start`` / ``on_tool_end``                         → TOOL_CALL / AFTER_TOOL

Chain-level boundaries (BEFORE_AGENT / AFTER_AGENT) are intentionally
*not* fired from here — they are owned by the governance host that
drives the agent. ``ignore_chain = True`` makes LangChain skip chain
notifications entirely, avoiding duplicate boundary evaluations.

Audit emission and enforcement (raising
:class:`GovernanceBlockException` on DENY) are owned by the evaluator
itself. This module just hooks the framework callbacks, extracts the
data, and calls ``evaluator.evaluate_*``; block exceptions propagate,
everything else is logged and swallowed so a governance bug never
breaks an agent run.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    Generation,
    GenerationChunk,
    LLMResult,
)
from uipath.core.adapters import EvaluatorProtocol
from uipath.core.governance.exceptions import GovernanceBlockException

GenerationLike = Generation | ChatGeneration | GenerationChunk | ChatGenerationChunk

logger = logging.getLogger(__name__)

# Cap on the text scanned per model hook, so a long history / runaway
# response can't blow scan-time budgets.
_BEFORE_MODEL_TEXT_CAP = 64000


class GovernanceCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that fires governance evaluation.

    The evaluator owns audit emission and DENY-raising. Each ``on_*``
    callback only extracts the relevant payload and calls the matching
    ``evaluate_*`` method; :class:`GovernanceBlockException` is allowed
    to propagate, anything else is logged and swallowed.
    """

    run_inline: bool = True
    raise_error: bool = True
    ignore_llm: bool = False
    # Chain-level events (BEFORE_AGENT / AFTER_AGENT) are owned by the
    # governance host, so this handler skips them to avoid duplicate
    # boundary evaluations.
    ignore_chain: bool = True
    ignore_agent: bool = False
    ignore_retriever: bool = True
    ignore_retry: bool = True
    ignore_chat_model: bool = False
    ignore_custom_event: bool = True

    def __init__(
        self,
        evaluator: EvaluatorProtocol,
        agent_name: str,
        session_id: str,
    ) -> None:
        self._evaluator = evaluator
        self._agent_name = agent_name
        self._session_id = session_id
        # ``trace_id`` is intentionally NOT held here. Trace correlation
        # is owned by the layer below: OTel-backed sinks read the live
        # span via ``trace.get_current_span()`` on the caller's thread;
        # HTTP-bound consumers resolve the canonical trace id at call
        # time. The callback handler is env-free and just forwards
        # extracted payload to the evaluator.
        self._session_state: Dict[str, Any] = {"tool_calls": 0, "llm_calls": 0}
        # Tool name lookup keyed by LangChain ``run_id`` so ``on_tool_end``
        # can report the actual tool name to AFTER_TOOL evaluation.
        self._tool_runs: Dict[str, str] = {}

    # ----- LLM callbacks ---------------------------------------------------

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: list[str],
        **kwargs: Any,
    ) -> None:
        """Evaluate BEFORE_MODEL rules at LLM start (non-chat completion)."""
        try:
            self._session_state["llm_calls"] = (
                self._session_state.get("llm_calls", 0) + 1
            )
            # Take only the latest prompt. Re-scanning every prompt in a
            # batched call would re-fire rules on prior turns' content
            # that's still in the prompt for context.
            model_input = (prompts[-1] if prompts else "")[:_BEFORE_MODEL_TEXT_CAP]
            self._evaluator.evaluate_before_model(
                model_input=model_input,
                agent_name=self._agent_name,
                runtime_id=self._session_id,
            )
        except GovernanceBlockException:
            raise
        except Exception as e:
            logger.warning("on_llm_start governance check failed (continuing): %s", e)

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: list[list[Any]],
        **kwargs: Any,
    ) -> None:
        """Evaluate BEFORE_MODEL rules for chat models.

        Scans only the **latest message** in the prompt — not the full
        chat history. The LLM still receives the entire history (this
        callback doesn't mutate ``messages``), but the governance
        evaluator focuses on the new content the agent is about to
        respond to. Without this scoping, a violation in turn 3's user
        message would keep re-firing on turns 4, 5, 6 ... because that
        text stays in the prompt for context.

        List-of-blocks content (multimodal, function-call, tool_use,
        extended thinking) is walked via :meth:`_extract_block_text` so
        dict-syntax noise from ``str(list)`` doesn't leak into the
        regex-scanned blob.
        """
        try:
            self._session_state["llm_calls"] = (
                self._session_state.get("llm_calls", 0) + 1
            )
            model_input = self._latest_message_input(messages)
            self._evaluator.evaluate_before_model(
                model_input=model_input,
                agent_name=self._agent_name,
                runtime_id=self._session_id,
            )
        except GovernanceBlockException:
            raise
        except Exception as e:
            logger.warning(
                "on_chat_model_start governance check failed (continuing): %s", e
            )

    @staticmethod
    def _latest_message_input(messages: list[list[Any]]) -> str:
        """Extract content from the most-recent message in the prompt.

        ``messages`` is LangChain's nested shape ``list[list[BaseMessage]]``
        — the outer list is for batched calls (rare); the inner list is
        the full message stack for one call. We take the last entry of
        the last inner list. For string content, that's used directly;
        for list-of-blocks content, :meth:`_extract_block_text` pulls
        the text / arguments / input / thinking fields cleanly.

        Returns ``""`` (empty) when the message stack is empty or the
        last message carries no extractable content.
        """
        if not messages:
            return ""
        last_batch = messages[-1]
        if not last_batch:
            return ""
        last_msg = last_batch[-1]
        # BaseMessage exposes ``.content``; dict-shaped messages
        # (LangGraph state, raw OpenAI format) carry it under the same
        # key.
        content = getattr(last_msg, "content", None)
        if content is None and isinstance(last_msg, dict):
            content = last_msg.get("content")
        if isinstance(content, str):
            return content[:_BEFORE_MODEL_TEXT_CAP]
        if isinstance(content, list):
            return GovernanceCallbackHandler._blocks_to_text(content)
        return ""

    @staticmethod
    def _blocks_to_text(content: list[Any]) -> str:
        """Concatenate governance-relevant text from a list of content blocks.

        Walks list-of-blocks message content (multimodal, function-call,
        tool_use, extended thinking) via :meth:`_extract_block_text`,
        capping the joined result at ``_BEFORE_MODEL_TEXT_CAP``.
        """
        pieces = (
            GovernanceCallbackHandler._extract_block_text(block)
            for block in content
            if isinstance(block, dict)
        )
        return GovernanceCallbackHandler._join_within_cap(pieces, "\n")

    @staticmethod
    def _join_within_cap(pieces: Iterable[str], sep: str) -> str:
        """Join non-empty ``pieces`` with ``sep``, stopping at the text cap.

        Shared accumulator for the model-input/output scan blobs: appends
        pieces until ``_BEFORE_MODEL_TEXT_CAP`` characters are reached
        (counting the separator), then caps the joined result.
        """
        out: list[str] = []
        remaining = _BEFORE_MODEL_TEXT_CAP
        for piece in pieces:
            if remaining <= 0:
                break
            if piece:
                out.append(piece)
                remaining -= len(piece) + len(sep)
        return sep.join(out)[:_BEFORE_MODEL_TEXT_CAP]

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Evaluate AFTER_MODEL rules at LLM end.

        Concatenates text from every generation. The result is capped at
        ``_BEFORE_MODEL_TEXT_CAP`` to match the BEFORE_MODEL budget, so
        batched calls or a runaway single response can't blow scan budgets.
        """
        try:
            model_output = self._collect_generations_text(response)
            self._evaluator.evaluate_after_model(
                model_output=model_output,
                agent_name=self._agent_name,
                runtime_id=self._session_id,
            )
        except GovernanceBlockException:
            raise
        except Exception as e:
            logger.warning("on_llm_end governance check failed (continuing): %s", e)

    def _collect_generations_text(self, response: LLMResult) -> str:
        """Concatenate text across all generations, capped at the text budget."""
        pieces = (
            self._extract_generation_text(gen)
            for gen_list in response.generations
            for gen in gen_list
        )
        return self._join_within_cap(pieces, "")

    @staticmethod
    def _extract_generation_text(gen: GenerationLike) -> str:
        """Return the text payload of a LangChain generation.

        ``Generation.text`` is set from ``message.content`` only when content
        is a plain ``str``. For chat models whose content is a list of
        content blocks (multimodal, tool calls, "submit final answer"
        function calls, extended thinking) ``.text`` is ``""``. In that case
        walk ``gen.message.content`` so the governance evaluator sees the
        actual assistant text.
        """
        if isinstance(gen, (ChatGeneration, ChatGenerationChunk)):
            content = gen.message.content
            if isinstance(content, list):
                parts = [
                    GovernanceCallbackHandler._extract_block_text(block)
                    for block in content
                    if isinstance(block, dict)
                ]
                joined = "\n".join(p for p in parts if p)
                if joined:
                    return joined
        return gen.text or ""

    @staticmethod
    def _extract_block_text(block: Dict[str, Any]) -> str:
        """Return any governance-relevant text from a content block.

        Covers the common block shapes across providers:

        - ``{"type": "text", "text": "..."}`` — plain text block.
        - ``{"type": "function_call", "arguments": "<json>"}`` — OpenAI
          function call; ``arguments`` is JSON-encoded and routinely
          carries the user-visible reply (e.g. ``end_execution(content=...)``
          tools used as a "submit final answer" pattern).
        - ``{"type": "tool_use", "input": {...}}`` — Anthropic tool use;
          string values in ``input`` are the assistant's outgoing payload.
        - ``{"type": "thinking", "thinking": "..."}`` — Claude extended
          thinking (governance-relevant: hidden reasoning can also leak
          commitments and PII).

        Metadata-only keys (``id``, ``call_id``, ``name``, ``status``,
        ``type``, ...) are excluded so the scanned text isn't padded with
        opaque identifiers that could false-positive a rule.
        """
        parts: list[str] = []
        text_value = block.get("text")
        if isinstance(text_value, str):
            parts.append(text_value)
        arguments_value = block.get("arguments")
        if isinstance(arguments_value, str):
            parts.append(arguments_value)
        thinking_value = block.get("thinking")
        if isinstance(thinking_value, str):
            parts.append(thinking_value)
        input_value = block.get("input")
        if isinstance(input_value, dict):
            parts.extend(v for v in input_value.values() if isinstance(v, str))
        return "\n".join(p for p in parts if p)

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        logger.warning("LLM error in governed session %s: %s", self._session_id, error)

    # ----- Tool callbacks --------------------------------------------------

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        inputs: Dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Evaluate TOOL_CALL rules at tool start.

        ``run_id → tool_name`` is recorded so ``on_tool_end`` /
        ``on_tool_error`` can report the actual tool. If the evaluator
        BLOCKS, the tool is aborted, ``on_tool_end`` will not fire, and
        the mapping is dropped to keep ``_tool_runs`` from growing
        unbounded across blocked turns.
        """
        run_id = kwargs.get("run_id")
        run_id_str = str(run_id) if run_id is not None else None
        try:
            self._session_state["tool_calls"] = (
                self._session_state.get("tool_calls", 0) + 1
            )
            tool_name = (serialized or {}).get("name", "unknown")
            if run_id_str is not None:
                self._tool_runs[run_id_str] = tool_name
            tool_args = inputs or {"input": input_str}
            self._evaluator.evaluate_tool_call(
                tool_name=tool_name,
                tool_args=tool_args,
                agent_name=self._agent_name,
                runtime_id=self._session_id,
                session_state=self._session_state,
            )
        except GovernanceBlockException:
            # Tool will not run → no on_tool_end is coming. Drop the
            # mapping so it does not accumulate across blocked turns.
            if run_id_str is not None:
                self._tool_runs.pop(run_id_str, None)
            raise
        except Exception as e:
            logger.warning("on_tool_start governance check failed (continuing): %s", e)

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        """Evaluate AFTER_TOOL rules at tool end."""
        try:
            run_id = kwargs.get("run_id")
            tool_name = "unknown"
            if run_id is not None:
                tool_name = self._tool_runs.pop(str(run_id), "unknown")
            tool_result = str(output) if output is not None else ""
            self._evaluator.evaluate_after_tool(
                tool_name=tool_name,
                tool_result=tool_result,
                agent_name=self._agent_name,
                runtime_id=self._session_id,
            )
        except GovernanceBlockException:
            raise
        except Exception as e:
            logger.warning("on_tool_end governance check failed (continuing): %s", e)

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        # Tool errored out — on_tool_end will not fire. Pop the mapping
        # so a session with many failing tool calls does not leak.
        run_id = kwargs.get("run_id")
        if run_id is not None:
            self._tool_runs.pop(str(run_id), None)
        logger.warning("Tool error in governed session %s: %s", self._session_id, error)
