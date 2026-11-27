"""The files a middleware guardrail may forward, rebuilt from the agent state.

``create_agent`` keeps no attachment registry, so every hook rebuilds one from what
the run holds. Matching a payload against it, the reference format and the API limits
live in ``uipath.platform.guardrails.attachments``. Only references are forwarded; the
Agents backend resolves each id through Orchestrator.
"""

import json
import logging
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from pydantic import ValidationError
from uipath.platform.attachments import Attachment
from uipath.platform.guardrails import (
    GuardrailAttachment,
    discover_attachments,
    iter_attachment_mentions,
    mention_id,
    to_references,
)

from uipath_langchain._utils._attachments import (
    ATTACHMENT_ID_EXTRA,
    ATTACHMENT_MIME_TYPE_EXTRA,
    ATTACHMENT_NAME_EXTRA,
    ATTACHMENTS_BLOCK_PREFIX,
    ATTACHMENTS_BLOCK_SUFFIX,
)
from uipath_langchain.agent.attachments.job_attachments import (
    guardrail_references,
    held_attachments,
    parse_attachments_from_conversation_messages,
    parse_tool_content,
)
from uipath_langchain.agent.attachments.mime_types import with_guardrail_mime_type

logger = logging.getLogger(__name__)

JOB_ATTACHMENTS_STATE_KEY = "uipath__job_attachments"
NO_TEXT_PLACEHOLDER = "[no text content]"

_FILE_BLOCK_TYPES = frozenset({"file", "image"})


def build_registry(state: Mapping[str, Any] | None) -> dict[str, Attachment]:
    """Collect the attachments the run holds.

    Sources, in order, where a later source never replaces an attachment an earlier
    one found: the attachments of conversational HumanMessages, every
    ``<uip:attachments>`` block in Human and Tool message text, file and image blocks
    tagged with an attachment id, the files tool results describe with a name and a
    type, the ``uipath__job_attachments`` state an advanced agent keeps, and last the
    files tool results name without a type, held with an empty one.

    Args:
        state: The agent state, or any mapping with ``messages``; may be None.

    Returns:
        The attachments keyed by canonical lowercase UUID string. A part of the state
        that cannot be read is skipped, so this never raises.
    """
    registry: dict[str, Attachment] = {}
    try:
        if not isinstance(state, Mapping):
            return registry
        messages = [
            message
            for message in state.get("messages") or []
            if isinstance(message, BaseMessage)
        ]
        for source in _MESSAGE_SOURCES:
            _add_from_messages(registry, source, messages)
        stored = state.get(JOB_ATTACHMENTS_STATE_KEY)
        if isinstance(stored, Mapping):
            _add(registry, _validated(stored.values()))
        _add_from_messages(registry, _untyped_tool_result_attachments, messages)
    except Exception:
        logger.warning(
            "Could not read the run's attachments; the guardrail sees only those "
            "found so far.",
            exc_info=True,
        )
    return registry


def registry_references(
    registry: Mapping[str, Attachment], *extra: Iterable[Any]
) -> list[GuardrailAttachment]:
    """Return references for every attachment the run holds, for an input check.

    Args:
        registry: The run's attachments, as :func:`build_registry` returns them.
        *extra: Further candidates, such as the attachments a state key carries,
            ordered after the held files (see :func:`guardrail_references`).

    Returns:
        Up to five references.
    """
    return guardrail_references(registry.values(), *extra)


def output_references(
    message: AIMessage | None,
    registry: Mapping[str, Attachment],
    *extra: Iterable[Any],
    structured_response: Any = None,
) -> list[GuardrailAttachment] | None:
    """Return references for the held attachments an output names.

    An output check judges what the model or the agent produced, so only the files
    its tool calls, its content or the structured response mention are forwarded.

    Args:
        message: The model's or the agent's last message; None reads only the
            structured response.
        registry: The run's attachments, as :func:`build_registry` returns them.
        *extra: Further candidates, such as the attachments the output returns,
            ordered after the held files (see :func:`guardrail_references`).
        structured_response: The agent's structured response, if any.

    Returns:
        Up to five references; ``None`` when the output cannot be read, so the
        check says nothing about files and the text is evaluated as before.
    """
    if message is None:
        held = held_attachments({"structured_response": structured_response}, registry)
        return None if held is None else guardrail_references(held, *extra)
    try:
        output = {
            "tool_calls": [call.get("args") for call in message.tool_calls],
            "content": message.content,
            "structured_response": structured_response,
        }
    except Exception:
        logger.warning(
            "Could not read the output for attachments; the guardrail evaluates "
            "without files.",
            exc_info=True,
        )
        return None
    held = held_attachments(output, registry)
    return None if held is None else guardrail_references(held, *extra)


def tool_result_references(
    result_data: Any, registry: Mapping[str, Attachment]
) -> list[GuardrailAttachment] | None:
    """Return references for the files a tool result names or returns.

    Held files keep the registry's name and type, whatever the result says about
    them; a file the result describes with its own name is new output of the tool and
    is forwarded as described, or with a fallback type when it names none (see
    :func:`guardrail_references`).

    Args:
        result_data: The tool's parsed result.
        registry: The run's attachments, as :func:`build_registry` returns them.

    Returns:
        Up to five references, held files first; ``None`` when the result cannot
        be scanned.
    """
    held = held_attachments(result_data, registry)
    if held is None or discover_attachments(result_data) is None:
        return None
    new = (
        mention
        for mention in iter_attachment_mentions(result_data)
        if mention_id(mention) not in registry
    )
    return guardrail_references(held, new)


def _add_from_messages(
    registry: dict[str, Attachment],
    source: Callable[[BaseMessage], Iterable[Attachment]],
    messages: list[BaseMessage],
) -> None:
    for message in messages:
        try:
            _add(registry, source(message))
        except Exception:
            logger.warning(
                "Could not read the attachments of a %s; skipping it.",
                type(message).__name__,
                exc_info=True,
            )


def _add(registry: dict[str, Attachment], attachments: Iterable[Attachment]) -> None:
    for attachment in attachments:
        key = mention_id(attachment)
        if key is not None:
            registry.setdefault(key, attachment)


def _validated(candidates: Iterable[Any]) -> Iterator[Attachment]:
    for candidate in candidates:
        try:
            attachment = Attachment.model_validate(candidate)
        except ValidationError:
            logger.debug("Skipping an invalid attachment.", exc_info=True)
            continue
        yield attachment


def _conversation_attachments(message: BaseMessage) -> Iterable[Attachment]:
    return parse_attachments_from_conversation_messages([message]).values()


def _block_attachments(message: BaseMessage) -> Iterator[Attachment]:
    if not isinstance(message, (HumanMessage, ToolMessage)):
        return
    for text in _texts(message.content):
        for payload in _block_payloads(text):
            yield from _validated(payload if isinstance(payload, list) else [payload])


def _texts(content: Any) -> Iterator[str]:
    if isinstance(content, str):
        yield content
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, Mapping) and part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str):
                    yield text


def _block_payloads(text: str) -> Iterator[Any]:
    position = text.find(ATTACHMENTS_BLOCK_PREFIX)
    while position != -1:
        start = position + len(ATTACHMENTS_BLOCK_PREFIX)
        end = text.find(ATTACHMENTS_BLOCK_SUFFIX, start)
        if end == -1:
            return
        try:
            payload = json.loads(text[start:end])
        except json.JSONDecodeError:
            logger.debug("Skipping an attachments block that is not JSON.")
        else:
            yield payload
        position = text.find(
            ATTACHMENTS_BLOCK_PREFIX, end + len(ATTACHMENTS_BLOCK_SUFFIX)
        )


def _tagged_block_attachments(message: BaseMessage) -> Iterator[Attachment]:
    if not isinstance(message, (HumanMessage, ToolMessage)) or not isinstance(
        message.content, list
    ):
        return
    for part in message.content:
        if not isinstance(part, Mapping) or part.get("type") not in _FILE_BLOCK_TYPES:
            continue
        extras = part.get("extras")
        if not isinstance(extras, Mapping) or not extras.get(ATTACHMENT_ID_EXTRA):
            continue
        mime_type = extras.get(ATTACHMENT_MIME_TYPE_EXTRA)
        if not isinstance(mime_type, str):
            mime_type = part.get("mime_type")
            if not mime_type:
                continue
        name = (
            extras.get(ATTACHMENT_NAME_EXTRA)
            or part.get("filename")
            or extras.get("filename")
        )
        if not name:
            logger.debug("Skipping a tagged file block without a file name.")
            continue
        yield from _validated(
            [
                {
                    "ID": extras[ATTACHMENT_ID_EXTRA],
                    "FullName": name,
                    "MimeType": mime_type,
                }
            ]
        )


def _tool_result_attachments(message: BaseMessage) -> Iterator[Attachment]:
    if not isinstance(message, ToolMessage):
        return
    for mention in iter_attachment_mentions(parse_tool_content(message.content)):
        yield from _validated(map(_as_mention, to_references([mention])))


def _untyped_tool_result_attachments(message: BaseMessage) -> Iterator[Attachment]:
    if not isinstance(message, ToolMessage):
        return
    for mention in iter_attachment_mentions(parse_tool_content(message.content)):
        typed = with_guardrail_mime_type(mention)
        if typed is mention:
            continue
        for reference in to_references([typed]):
            yield from _validated([{**_as_mention(reference), "MimeType": ""}])


def _as_mention(reference: GuardrailAttachment) -> dict[str, str]:
    return {
        "ID": reference.id,
        "FullName": reference.file_name,
        "MimeType": reference.mime_type,
    }


_MESSAGE_SOURCES: tuple[Callable[[BaseMessage], Iterable[Attachment]], ...] = (
    _conversation_attachments,
    _block_attachments,
    _tagged_block_attachments,
    _tool_result_attachments,
)
