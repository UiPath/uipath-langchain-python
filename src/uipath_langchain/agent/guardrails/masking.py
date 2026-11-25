"""Placeholders for the values a mask guardrail replaces, and their restoration."""

import json
import re
import secrets
from collections import Counter
from typing import Any, Callable, Iterable, Mapping

# ``[Email-Tool-7f3a9c-1]``: category, scope, the run's tag, number. The LLM
# Gateway's own masking writes ``[Email-1]`` and, when it restores a response,
# replaces that token bracketed or not, in any case and anywhere in the text. The
# scope keeps ``Email-1`` from ever appearing inside one of ours, and the run tag
# keeps a run from restoring placeholders another run wrote, such as a parent
# agent's placeholders in the input of an agent it calls as a tool.
_PLACEHOLDER = re.compile(r"\[([A-Za-z0-9]+)-(Agent|Tool)-([0-9a-f]{6})-([0-9]+)\]")


def placeholder_label(category: str) -> str:
    """The category as it appears in a placeholder: its ASCII letters and digits."""
    return re.sub(r"[^A-Za-z0-9]", "", category) or "PII"


def new_run_tag(texts: Iterable[str]) -> str:
    """A tag for the run's placeholders that no placeholder in ``texts`` carries."""
    taken = {match.group(3) for text in texts for match in _PLACEHOLDER.finditer(text)}
    while (tag := secrets.token_hex(3)) in taken:
        pass
    return tag


def map_strings(data: Any, transform: Callable[[str], str]) -> Any:
    """``data`` with ``transform`` applied to every string value; keys are kept."""
    if isinstance(data, str):
        return transform(data)
    if isinstance(data, dict):
        return {key: map_strings(value, transform) for key, value in data.items()}
    if isinstance(data, list):
        return [map_strings(item, transform) for item in data]
    if isinstance(data, tuple):
        return tuple(map_strings(item, transform) for item in data)
    return data


def map_content(
    content: str | list[Any], transform: Callable[[str], str]
) -> str | list[Any]:
    """A message content with ``transform`` applied to its text."""
    if isinstance(content, str):
        return transform(content)
    blocks: list[Any] = []
    for block in content:
        if isinstance(block, str):
            blocks.append(transform(block))
        elif (
            isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ):
            blocks.append({**block, "text": transform(block["text"])})
        else:
            blocks.append(block)
    return blocks


def map_tool_content(
    content: str | list[Any], transform: Callable[[str], str]
) -> str | list[Any]:
    """A tool result with ``transform`` applied to its text.

    A JSON result is parsed and serialized again, so a value is never spliced into
    the serialized text where its quotes or escapes would break it.
    """
    if not isinstance(content, str):
        return map_content(content, transform)
    try:
        parsed = json.loads(content)
    except ValueError:
        return transform(content)
    mapped = map_strings(parsed, transform)
    if mapped == parsed:
        return content
    return json.dumps(mapped, ensure_ascii=content.isascii())


def decode_json_escapes(text: str) -> str:
    """``text`` as it reads once the JSON escapes of a serialized payload are decoded."""
    if "\\" not in text:
        return text
    try:
        decoded = json.loads(f'"{text}"')
    except ValueError:
        return text
    return decoded if isinstance(decoded, str) and decoded else text


class PlaceholderAssigner:
    """Gives each value a placeholder such as ``[Email-Tool-7f3a9c-1]``.

    Placeholders are numbered per category and scope across the run, and a value
    the run has masked before in the same scope keeps its placeholder.
    """

    def __init__(self, masked_values: Mapping[str, str], scope: str, tag: str) -> None:
        self._scope = scope
        self._tag = tag
        self._placeholders: dict[str, str] = {}
        self._counts: dict[str, int] = {}
        for key, value in masked_values.items():
            match = _PLACEHOLDER.fullmatch(key)
            if match is None or match.group(2) != scope:
                continue
            self._placeholders[value] = key
            label, number = match.group(1), int(match.group(4))
            self._counts[label] = max(self._counts.get(label, 0), number)
        self.added: dict[str, str] = {}

    def placeholder(self, value: str, category: str) -> str:
        """The placeholder of ``value``, assigning the next one if it has none."""
        placeholder = self._placeholders.get(value)
        if placeholder is None:
            label = placeholder_label(category)
            self._counts[label] = self._counts.get(label, 0) + 1
            placeholder = f"[{label}-{self._scope}-{self._tag}-{self._counts[label]}]"
            self._placeholders[value] = placeholder
            self.added[placeholder] = value
        return placeholder


class Masker:
    """Replaces values with their placeholders inside strings, longest value first.

    Every string is scanned once, so a placeholder already written is never
    matched again.
    """

    def __init__(
        self, categories: Mapping[str, str], assigner: PlaceholderAssigner
    ) -> None:
        self._categories = dict(categories)
        self._assigner = assigner
        ordered = sorted(self._categories, key=lambda value: (-len(value), value))
        self._pattern = (
            re.compile("|".join(re.escape(value) for value in ordered))
            if ordered
            else None
        )
        self._masked: set[str] = set()
        self._written: set[str] = set()

    def mask_text(self, text: str) -> str:
        """``text`` with every value replaced by its placeholder."""
        if self._pattern is None:
            return text
        return self._pattern.sub(self._replace, text)

    def mask(self, data: Any) -> Any:
        """``data`` with every value replaced inside its strings."""
        return map_strings(data, self.mask_text)

    def placeholders(self) -> list[str]:
        """The placeholders written, sorted."""
        return sorted(self._written)

    def counts(self) -> dict[str, int]:
        """How many distinct values of each category were replaced."""
        return dict(Counter(self._categories[value] for value in self._masked))

    def missed_categories(self) -> list[str]:
        """The categories of the values found nowhere, not even inside a replaced value."""
        missed = {
            self._categories[value]
            for value in self._categories
            if value not in self._masked
            and not any(value in masked for masked in self._masked)
        }
        return sorted(missed)

    def _replace(self, match: re.Match[str]) -> str:
        value = match.group(0)
        self._masked.add(value)
        placeholder = self._assigner.placeholder(value, self._categories[value])
        self._written.add(placeholder)
        return placeholder


class Restorer:
    """Puts back the values behind the run's placeholders."""

    def __init__(self, masked_values: Mapping[str, str]) -> None:
        self._masked_values = masked_values
        self._restored: Counter[str] = Counter()

    def restore_text(self, text: str) -> str:
        """``text`` with every known placeholder replaced by its value."""
        if not self._masked_values or "[" not in text:
            return text
        return _PLACEHOLDER.sub(self._replace, text)

    def counts(self) -> dict[str, int]:
        """How many placeholders of each category were restored."""
        return dict(self._restored)

    def _replace(self, match: re.Match[str]) -> str:
        value = self._masked_values.get(match.group(0))
        if value is None:
            return match.group(0)
        self._restored[match.group(1)] += 1
        return value
