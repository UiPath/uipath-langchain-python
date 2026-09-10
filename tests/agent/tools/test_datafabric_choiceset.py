"""Tests for choice-set awareness in the Data Fabric NL-to-SQL pipeline.

Covers:
- _resolve_choiceset_labels: happy path, None service, exception
- build_entity_context: choice-set label appending, caching, empty labels
- build_sql_context / build: entities_service threading
- Choice-set field tagging in rendered prompt output
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

from uipath_langchain.agent.tools.datafabric_tool.datafabric_prompt_builder import (
    _resolve_choiceset_labels,
    build,
    build_entity_context,
    build_sql_context,
)

# ---------------------------------------------------------------------------
# Helpers — mirrors test_datafabric_prompt_builder.py conventions
# ---------------------------------------------------------------------------


def _cs_value(display_name: str) -> SimpleNamespace:
    """Fake ChoiceSetValue with a display_name."""
    return SimpleNamespace(display_name=display_name)


def _fake_field(**overrides: Any) -> SimpleNamespace:
    defaults = dict(
        name="status",
        display_name="Status",
        sql_type=SimpleNamespace(name="varchar"),
        description="The status field",
        allowed_values=None,
        examples=None,
        good_for_aggregation=False,
        good_for_grouping=True,
        good_for_filtering=True,
        is_foreign_key=False,
        is_required=False,
        is_unique=False,
        is_hidden_field=False,
        is_system_field=False,
        field_display_type=None,
        reference_entity=None,
        reference_field=None,
        choiceset_id=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _fake_entity(*fields: Any, name: str = "Ticket", **overrides: Any) -> Any:
    defaults = dict(
        id="entity-1",
        display_name="Ticket",
        description="Support tickets",
        record_count=10,
    )
    defaults.update(overrides)
    return SimpleNamespace(name=name, fields=list(fields), **defaults)


# ---------------------------------------------------------------------------
# _resolve_choiceset_labels
# ---------------------------------------------------------------------------


class TestResolveChoicesetLabels:
    def test_returns_labels_on_success(self) -> None:
        svc = MagicMock()
        svc.get_choiceset_values.return_value = [
            _cs_value("Low"),
            _cs_value("Medium"),
            _cs_value("High"),
        ]
        labels = _resolve_choiceset_labels(svc, "cs-123")
        svc.get_choiceset_values.assert_called_once_with("cs-123")
        assert labels == ["Low", "Medium", "High"]

    def test_returns_empty_when_service_is_none(self) -> None:
        assert _resolve_choiceset_labels(None, "cs-123") == []

    def test_returns_empty_on_exception(self) -> None:
        svc = MagicMock()
        svc.get_choiceset_values.side_effect = RuntimeError("network error")
        labels = _resolve_choiceset_labels(svc, "cs-123")
        assert labels == []

    def test_returns_empty_when_service_returns_empty(self) -> None:
        svc = MagicMock()
        svc.get_choiceset_values.return_value = []
        assert _resolve_choiceset_labels(svc, "cs-123") == []


# ---------------------------------------------------------------------------
# build_entity_context — choice-set integration
# ---------------------------------------------------------------------------


class TestBuildEntityContextChoiceSets:
    def test_choiceset_labels_appended_to_description(self) -> None:
        svc = MagicMock()
        svc.get_choiceset_values.return_value = [
            _cs_value("Critical"),
            _cs_value("Normal"),
        ]
        field = _fake_field(
            name="priority",
            display_name="Priority",
            description="Ticket priority",
            choiceset_id="cs-priority",
        )
        entity = _fake_entity(field, name="Ticket")

        ctx = build_entity_context(entity, entities_service=svc)

        priority_field = next(
            f for f in ctx.entity_schema.fields if f.name == "priority"
        )
        assert priority_field.description is not None
        assert "Critical" in priority_field.description
        assert "Normal" in priority_field.description
        assert "allowed values:" in priority_field.description

    def test_choiceset_labels_cached_across_fields(self) -> None:
        """Two fields sharing the same choiceset_id should only fetch once."""
        svc = MagicMock()
        svc.get_choiceset_values.return_value = [_cs_value("A"), _cs_value("B")]

        f1 = _fake_field(name="f1", choiceset_id="cs-shared")
        f2 = _fake_field(name="f2", choiceset_id="cs-shared")
        entity = _fake_entity(f1, f2)

        build_entity_context(entity, entities_service=svc)

        svc.get_choiceset_values.assert_called_once_with("cs-shared")

    def test_choiceset_no_service_no_labels(self) -> None:
        """When entities_service is None, choice-set fields get no suffix."""
        field = _fake_field(
            name="priority",
            description="Ticket priority",
            choiceset_id="cs-priority",
        )
        entity = _fake_entity(field)

        ctx = build_entity_context(entity, entities_service=None)

        priority_field = next(
            f for f in ctx.entity_schema.fields if f.name == "priority"
        )
        assert priority_field.description == "Ticket priority"

    def test_choiceset_empty_labels_no_suffix(self) -> None:
        """Empty label list should not append anything to the description."""
        svc = MagicMock()
        svc.get_choiceset_values.return_value = []

        field = _fake_field(
            name="category",
            description="Category",
            choiceset_id="cs-empty",
        )
        entity = _fake_entity(field)

        ctx = build_entity_context(entity, entities_service=svc)

        cat_field = next(f for f in ctx.entity_schema.fields if f.name == "category")
        assert cat_field.description == "Category"

    def test_choiceset_fetch_failure_no_suffix(self) -> None:
        """Service exception should not crash and should not append suffix."""
        svc = MagicMock()
        svc.get_choiceset_values.side_effect = RuntimeError("timeout")

        field = _fake_field(
            name="priority",
            description="Priority",
            choiceset_id="cs-priority",
        )
        entity = _fake_entity(field)

        ctx = build_entity_context(entity, entities_service=svc)

        priority_field = next(
            f for f in ctx.entity_schema.fields if f.name == "priority"
        )
        assert priority_field.description == "Priority"

    def test_choiceset_no_base_description(self) -> None:
        """When description is empty, no leading separator is produced."""
        svc = MagicMock()
        svc.get_choiceset_values.return_value = [_cs_value("X"), _cs_value("Y")]

        field = _fake_field(
            name="tag",
            description="",
            choiceset_id="cs-tag",
        )
        entity = _fake_entity(field)

        ctx = build_entity_context(entity, entities_service=svc)

        tag_field = next(f for f in ctx.entity_schema.fields if f.name == "tag")
        assert tag_field.description is not None
        assert not tag_field.description.startswith(" —")
        assert "X" in tag_field.description
        assert "Y" in tag_field.description

    def test_field_without_choiceset_unaffected(self) -> None:
        """A normal field without choiceset_id is not altered."""
        svc = MagicMock()
        field = _fake_field(name="title", description="Ticket title")
        entity = _fake_entity(field)

        ctx = build_entity_context(entity, entities_service=svc)

        title_field = next(f for f in ctx.entity_schema.fields if f.name == "title")
        assert title_field.description == "Ticket title"
        svc.get_choiceset_values.assert_not_called()

    def test_multiple_choicesets_different_ids(self) -> None:
        """Different choiceset_ids are fetched independently."""
        svc = MagicMock()
        svc.get_choiceset_values.side_effect = lambda cs_id: {
            "cs-1": [_cs_value("Open"), _cs_value("Closed")],
            "cs-2": [_cs_value("Bug"), _cs_value("Feature")],
        }[cs_id]

        f1 = _fake_field(name="status", description="Status", choiceset_id="cs-1")
        f2 = _fake_field(name="type", description="Type", choiceset_id="cs-2")
        entity = _fake_entity(f1, f2)

        ctx = build_entity_context(entity, entities_service=svc)

        status_field = next(f for f in ctx.entity_schema.fields if f.name == "status")
        type_field = next(f for f in ctx.entity_schema.fields if f.name == "type")
        assert status_field.description is not None
        assert type_field.description is not None
        assert "Open" in status_field.description
        assert "Bug" in type_field.description
        assert svc.get_choiceset_values.call_count == 2


# ---------------------------------------------------------------------------
# build_sql_context — entities_service threading
# ---------------------------------------------------------------------------


class TestBuildSqlContextChoiceSets:
    def test_entities_service_passed_to_build_entity_context(self) -> None:
        svc = MagicMock()
        svc.get_choiceset_values.return_value = [_cs_value("Yes"), _cs_value("No")]

        field = _fake_field(
            name="approved", description="Approved?", choiceset_id="cs-approved"
        )
        entity = _fake_entity(field)

        ctx = build_sql_context([entity], entities_service=svc)

        approved_field = next(
            f
            for f in ctx.entity_contexts[0].entity_schema.fields
            if f.name == "approved"
        )
        assert approved_field.description is not None
        assert "Yes" in approved_field.description

    def test_entities_service_none_does_not_crash(self) -> None:
        field = _fake_field(name="priority", choiceset_id="cs-priority")
        entity = _fake_entity(field)

        ctx = build_sql_context([entity], entities_service=None)
        assert len(ctx.entity_contexts) == 1


# ---------------------------------------------------------------------------
# build() — full prompt rendering with choice-sets
# ---------------------------------------------------------------------------


class TestBuildFullPromptChoiceSets:
    def test_choice_set_labels_in_rendered_prompt(self) -> None:
        svc = MagicMock()
        svc.get_choiceset_values.return_value = [
            _cs_value("Low"),
            _cs_value("Medium"),
            _cs_value("High"),
        ]

        field = _fake_field(
            name="priority",
            display_name="Priority",
            description="Ticket priority",
            choiceset_id="cs-priority",
        )
        entity = _fake_entity(field)

        prompt = build([entity], entities_service=svc)

        assert "Low" in prompt
        assert "Medium" in prompt
        assert "High" in prompt
        assert "allowed values:" in prompt

    def test_entities_service_threaded_to_build(self) -> None:
        """The build() function must pass entities_service all the way down."""
        svc = MagicMock()
        svc.get_choiceset_values.return_value = [_cs_value("Active")]

        field = _fake_field(name="state", choiceset_id="cs-state")
        entity = _fake_entity(field)

        prompt = build([entity], entities_service=svc)

        assert "Active" in prompt
        svc.get_choiceset_values.assert_called_once_with("cs-state")

    def test_choice_set_guidance_in_prompt_template(self) -> None:
        """The v1 prompt template should include choice-set field guidance."""
        prompt = build([_fake_entity(_fake_field())])

        assert "CHOICE-SET FIELDS" in prompt

    def test_choice_set_constraint_in_sql_constraints(self) -> None:
        """SQL_CONSTRAINTS must document choice-set field behavior."""
        prompt = build([_fake_entity(_fake_field())])

        assert "choice_set" in prompt.lower() or "choice-set" in prompt.lower()
