"""Placeholders for the values a mask guardrail replaces, and their restoration."""

import json
from itertools import chain, repeat

import pytest

from uipath_langchain.agent.guardrails import masking
from uipath_langchain.agent.guardrails.masking import (
    Masker,
    PlaceholderAssigner,
    Restorer,
    decode_json_escapes,
    map_content,
    map_strings,
    map_tool_content,
    new_run_tag,
    placeholder_label,
)

TAG = "7f3a9c"


def agent(label: str, number: int) -> str:
    return f"[{label}-Agent-{TAG}-{number}]"


def tool(label: str, number: int) -> str:
    return f"[{label}-Tool-{TAG}-{number}]"


class TestPlaceholders:
    def test_placeholders_are_numbered_per_category(self):
        assigner = PlaceholderAssigner({}, "Agent", TAG)

        assert assigner.placeholder("jane@example.com", "Email") == agent("Email", 1)
        assert assigner.placeholder("Jane Doe", "Person") == agent("Person", 1)
        assert assigner.placeholder("john@example.com", "Email") == agent("Email", 2)
        assert assigner.added == {
            agent("Email", 1): "jane@example.com",
            agent("Person", 1): "Jane Doe",
            agent("Email", 2): "john@example.com",
        }

    def test_the_same_value_keeps_its_placeholder(self):
        assigner = PlaceholderAssigner({}, "Agent", TAG)

        first = assigner.placeholder("Jane Doe", "Person")
        again = assigner.placeholder("Jane Doe", "Organization")

        assert first == again == agent("Person", 1)
        assert assigner.added == {agent("Person", 1): "Jane Doe"}

    def test_numbering_continues_across_the_run_within_a_scope(self):
        assigner = PlaceholderAssigner(
            {
                tool("Email", 1): "jane@example.com",
                tool("Email", 3): "ann@example.com",
                agent("Email", 7): "eve@example.com",
            },
            "Tool",
            TAG,
        )

        assert assigner.placeholder("ann@example.com", "Email") == tool("Email", 3)
        assert assigner.placeholder("bob@example.com", "Email") == tool("Email", 4)
        assert assigner.placeholder("Bob", "Person") == tool("Person", 1)
        assert assigner.added == {
            tool("Email", 4): "bob@example.com",
            tool("Person", 1): "Bob",
        }

    def test_a_value_masked_in_another_scope_gets_its_own_placeholder(self):
        assigner = PlaceholderAssigner(
            {agent("Email", 1): "jane@example.com"}, "Tool", TAG
        )

        assert assigner.placeholder("jane@example.com", "Email") == tool("Email", 1)

    @pytest.mark.parametrize(
        ("category", "label"),
        [
            ("Email", "Email"),
            ("USSocialSecurityNumber", "USSocialSecurityNumber"),
            ("EU-Passport Number", "EUPassportNumber"),
            ("Email-1", "Email1"),
            ("---", "PII"),
            ("", "PII"),
        ],
    )
    def test_a_placeholder_never_holds_the_gateways_token(self, category, label):
        placeholder = PlaceholderAssigner({}, "Agent", TAG).placeholder(
            "value", category
        )

        assert placeholder_label(category) == label
        assert placeholder == agent(label, 1)
        # The LLM Gateway restores "[Email-1]" and "Email-1" anywhere, in any case.
        assert f"{label}-1".lower() not in placeholder.lower()


class TestRunTag:
    def test_a_tag_is_six_hex_digits(self):
        tag = new_run_tag([])

        assert len(tag) == 6
        assert int(tag, 16) >= 0

    def test_a_tag_another_run_used_is_drawn_again(self, monkeypatch):
        draws = chain(["1a2b3c", "1a2b3c"], repeat(TAG))
        monkeypatch.setattr(masking.secrets, "token_hex", lambda _n: next(draws))

        tag = new_run_tag(["From the parent: [Email-Tool-1a2b3c-1]"])

        assert tag == TAG


class TestMasker:
    def test_longer_values_are_replaced_first(self):
        masker = Masker(
            {"John": "Person", "John Smith": "Person"},
            PlaceholderAssigner({}, "Agent", TAG),
        )

        masked = masker.mask_text("John Smith met John.")

        assert masked == f"{agent('Person', 1)} met {agent('Person', 2)}."
        assert masker.counts() == {"Person": 2}
        assert masker.placeholders() == [agent("Person", 1), agent("Person", 2)]

    def test_a_placeholder_already_written_is_not_matched_again(self):
        masker = Masker(
            {"Jane": "Person", "Person": "Other"}, PlaceholderAssigner({}, "Agent", TAG)
        )

        assert masker.mask_text("Jane") == agent("Person", 1)

    def test_structured_data_is_masked_inside_its_string_values_only(self):
        masker = Masker(
            {"jane@example.com": "Email"}, PlaceholderAssigner({}, "Tool", TAG)
        )

        masked = masker.mask(
            {
                "jane@example.com": 1,
                "to": ["jane@example.com", 7, None],
                "body": {"text": "Write to jane@example.com today", "urgent": True},
            }
        )

        assert masked == {
            "jane@example.com": 1,
            "to": [tool("Email", 1), 7, None],
            "body": {"text": f"Write to {tool('Email', 1)} today", "urgent": True},
        }

    def test_values_found_nowhere_are_reported_by_category(self):
        masker = Masker(
            {"Jane Doe": "Person", "Doe": "Person", "555-0100": "PhoneNumber"},
            PlaceholderAssigner({}, "Agent", TAG),
        )

        masker.mask_text("Call Jane Doe")

        # "Doe" was replaced as part of "Jane Doe"; the phone number never appeared.
        assert masker.missed_categories() == ["PhoneNumber"]


class TestRestorer:
    def test_known_placeholders_are_restored_and_counted(self):
        restorer = Restorer(
            {agent("Email", 1): "jane@example.com", agent("Person", 1): "Jane"}
        )

        restored = restorer.restore_text(
            f"{agent('Person', 1)} <{agent('Email', 1)}>, cc {agent('Email', 1)}, "
            f"{agent('Email', 9)}, [Email-1], [Email-Agent-1a2b3c-1]"
        )

        assert restored == (
            "Jane <jane@example.com>, cc jane@example.com, "
            f"{agent('Email', 9)}, [Email-1], [Email-Agent-1a2b3c-1]"
        )
        assert restorer.counts() == {"Person": 1, "Email": 2}

    def test_a_restored_value_is_not_restored_again(self):
        restorer = Restorer(
            {agent("Person", 1): agent("Person", 2), agent("Person", 2): "Jane"}
        )

        assert restorer.restore_text(agent("Person", 1)) == agent("Person", 2)

    def test_a_json_result_stays_valid_json(self):
        restorer = Restorer(
            {tool("Person", 1): 'Ann "Annie" O\\Neil', tool("City", 1): "Zürich"}
        )

        content = map_tool_content(
            json.dumps({"name": tool("Person", 1), "city": tool("City", 1), "n": 2}),
            restorer.restore_text,
        )

        assert isinstance(content, str)
        assert json.loads(content) == {
            "name": 'Ann "Annie" O\\Neil',
            "city": "Zürich",
            "n": 2,
        }

    def test_a_plain_text_result_is_restored_in_place(self):
        restorer = Restorer({tool("Email", 1): "jane@example.com"})

        assert (
            map_tool_content(f"Sent to {tool('Email', 1)}.", restorer.restore_text)
            == "Sent to jane@example.com."
        )

    def test_an_unchanged_json_result_keeps_its_text(self):
        restorer = Restorer({tool("Email", 1): "jane@example.com"})
        content = '{"status":  "sent"}'

        assert map_tool_content(content, restorer.restore_text) is content


class TestHelpers:
    def test_map_strings_keeps_keys_and_other_values(self):
        assert map_strings(
            {"a": "x", "b": ("x", 1), "c": [{"x": "x"}], "d": 1.5}, str.upper
        ) == {"a": "X", "b": ("X", 1), "c": [{"x": "X"}], "d": 1.5}

    def test_map_content_maps_text_blocks_only(self):
        content = [
            "plain",
            {"type": "text", "text": "block"},
            {"type": "image_url", "image_url": {"url": "data:x"}},
        ]

        assert map_content(content, str.upper) == [
            "PLAIN",
            {"type": "text", "text": "BLOCK"},
            {"type": "image_url", "image_url": {"url": "data:x"}},
        ]

    @pytest.mark.parametrize(
        ("text", "decoded"),
        [
            ("Jos\\u00e9 Garc\\u00eda", "José García"),
            ('Ann \\"Annie\\" Lee', 'Ann "Annie" Lee'),
            ("1 Main St\\nSpringfield", "1 Main St\nSpringfield"),
            ("jane@example.com", "jane@example.com"),
            ('broken " quote \\', 'broken " quote \\'),
        ],
    )
    def test_decode_json_escapes(self, text, decoded):
        assert decode_json_escapes(text) == decoded
