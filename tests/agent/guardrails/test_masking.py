"""Placeholders for the values a mask guardrail replaces, and their restoration."""

import json

import pytest

from uipath_langchain.agent.guardrails.masking import (
    Masker,
    PlaceholderAssigner,
    Restorer,
    decode_json_escapes,
    map_content,
    map_strings,
    map_tool_content,
    placeholder_label,
)


class TestPlaceholders:
    def test_placeholders_are_numbered_per_category(self):
        assigner = PlaceholderAssigner({})

        assert assigner.placeholder("jane@example.com", "Email") == "[Email_1]"
        assert assigner.placeholder("Jane Doe", "Person") == "[Person_1]"
        assert assigner.placeholder("john@example.com", "Email") == "[Email_2]"
        assert assigner.added == {
            "[Email_1]": "jane@example.com",
            "[Person_1]": "Jane Doe",
            "[Email_2]": "john@example.com",
        }

    def test_the_same_value_keeps_its_placeholder(self):
        assigner = PlaceholderAssigner({})

        first = assigner.placeholder("Jane Doe", "Person")
        again = assigner.placeholder("Jane Doe", "Organization")

        assert first == again == "[Person_1]"
        assert assigner.added == {"[Person_1]": "Jane Doe"}

    def test_numbering_continues_across_the_run(self):
        assigner = PlaceholderAssigner(
            {"[Email_1]": "jane@example.com", "[Email_3]": "ann@example.com"}
        )

        assert assigner.placeholder("ann@example.com", "Email") == "[Email_3]"
        assert assigner.placeholder("bob@example.com", "Email") == "[Email_4]"
        assert assigner.placeholder("Bob", "Person") == "[Person_1]"
        assert assigner.added == {"[Email_4]": "bob@example.com", "[Person_1]": "Bob"}

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
    def test_a_placeholder_never_holds_a_hyphen(self, category, label):
        placeholder = PlaceholderAssigner({}).placeholder("value", category)

        assert placeholder_label(category) == label
        assert placeholder == f"[{label}_1]"
        assert "-" not in placeholder


class TestMasker:
    def test_longer_values_are_replaced_first(self):
        masker = Masker(
            {"John": "Person", "John Smith": "Person"}, PlaceholderAssigner({})
        )

        masked = masker.mask_text("John Smith met John.")

        assert masked == "[Person_1] met [Person_2]."
        assert masker.counts() == {"Person": 2}

    def test_a_placeholder_already_written_is_not_matched_again(self):
        masker = Masker({"Jane": "Person", "Person": "Other"}, PlaceholderAssigner({}))

        assert masker.mask_text("Jane") == "[Person_1]"

    def test_structured_data_is_masked_inside_its_string_values_only(self):
        masker = Masker({"jane@example.com": "Email"}, PlaceholderAssigner({}))

        masked = masker.mask(
            {
                "jane@example.com": 1,
                "to": ["jane@example.com", 7, None],
                "body": {"text": "Write to jane@example.com today", "urgent": True},
            }
        )

        assert masked == {
            "jane@example.com": 1,
            "to": ["[Email_1]", 7, None],
            "body": {"text": "Write to [Email_1] today", "urgent": True},
        }

    def test_values_found_nowhere_are_reported_by_category(self):
        masker = Masker(
            {"Jane Doe": "Person", "Doe": "Person", "555-0100": "PhoneNumber"},
            PlaceholderAssigner({}),
        )

        masker.mask_text("Call Jane Doe")

        # "Doe" was replaced as part of "Jane Doe"; the phone number never appeared.
        assert masker.missed_categories() == ["PhoneNumber"]


class TestRestorer:
    def test_known_placeholders_are_restored_and_counted(self):
        restorer = Restorer({"[Email_1]": "jane@example.com", "[Person_1]": "Jane"})

        restored = restorer.restore_text(
            "[Person_1] <[Email_1]>, cc [Email_1], [Email_9], [Email-1]"
        )

        assert (
            restored
            == "Jane <jane@example.com>, cc jane@example.com, [Email_9], [Email-1]"
        )
        assert restorer.counts() == {"Person": 1, "Email": 2}

    def test_a_restored_value_is_not_restored_again(self):
        restorer = Restorer({"[Person_1]": "[Person_2]", "[Person_2]": "Jane"})

        assert restorer.restore_text("[Person_1]") == "[Person_2]"

    def test_a_json_result_stays_valid_json(self):
        restorer = Restorer({"[Person_1]": 'Ann "Annie" O\\Neil', "[City_1]": "Zürich"})

        content = map_tool_content(
            json.dumps({"name": "[Person_1]", "city": "[City_1]", "n": 2}),
            restorer.restore_text,
        )

        assert isinstance(content, str)
        assert json.loads(content) == {
            "name": 'Ann "Annie" O\\Neil',
            "city": "Zürich",
            "n": 2,
        }

    def test_a_plain_text_result_is_restored_in_place(self):
        restorer = Restorer({"[Email_1]": "jane@example.com"})

        assert (
            map_tool_content("Sent to [Email_1].", restorer.restore_text)
            == "Sent to jane@example.com."
        )

    def test_an_unchanged_json_result_keeps_its_text(self):
        restorer = Restorer({"[Email_1]": "jane@example.com"})
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
