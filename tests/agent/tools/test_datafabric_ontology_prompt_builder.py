"""Tests for the ontology-tool inner prompt builder.

Verifies the ontology prompt embeds OWL + R2RML (ontology-tool-only) and still
renders the shared entity-schema tables — separate from the entity tool's prompt.
"""

from types import SimpleNamespace

from uipath_langchain.agent.tools.datafabric_tool.datafabric_ontology_prompt_builder import (  # noqa: E501
    build,
)


def _fake_field(**overrides):
    return SimpleNamespace(
        name="status",
        display_name="Status",
        sql_type=SimpleNamespace(name="varchar"),
        description="The canonical workflow status",
        allowed_values=["Open", "Closed"],
        examples=["Open"],
        good_for_aggregation=False,
        good_for_grouping=True,
        good_for_filtering=True,
        is_foreign_key=False,
        is_required=False,
        is_unique=False,
        is_hidden_field=False,
        is_system_field=False,
        **overrides,
    )


def _fake_entity(*fields, **overrides):
    return SimpleNamespace(
        id="entity-1",
        name="Ticket",
        display_name="Ticket",
        description="Support tickets",
        record_count=10,
        fields=list(fields),
        **overrides,
    )


def test_embeds_ontology_and_r2rml_sections_in_order():
    prompt = build(
        [_fake_entity(_fake_field())],
        ontology_text="OWL_BODY_XYZ",
        r2rml_text="R2RML_BODY_XYZ",
    )

    assert "## Available Ontology" in prompt
    assert "OWL_BODY_XYZ" in prompt
    assert "## Ontology→Table Mapping (R2RML)" in prompt
    assert "R2RML_BODY_XYZ" in prompt
    # OWL section precedes the R2RML section.
    assert prompt.index("Available Ontology") < prompt.index("Ontology→Table Mapping")


def test_still_renders_entity_schema_tables():
    prompt = build(
        [_fake_entity(_fake_field())],
        ontology_text="OWL",
        r2rml_text="MAP",
    )
    assert "## All available Data Fabric Entities" in prompt
    assert "| status |" in prompt
    assert "Ticket" in prompt


def test_omits_sections_when_empty():
    prompt = build([_fake_entity(_fake_field())])
    assert "## Available Ontology" not in prompt
    assert "## Ontology→Table Mapping (R2RML)" not in prompt


def test_no_entities_returns_empty():
    assert build([], ontology_text="OWL", r2rml_text="MAP") == ""


def test_format_context_renders_agent_instructions_and_ontology_description():
    # `build()` folds resource_description into the strategy prompt (ctx.resource_description
    # stays None), so exercise the Agent-Instructions + Ontology-description sections directly.
    from uipath_langchain.agent.tools.datafabric_tool.datafabric_ontology_prompt_builder import (
        format_ontology_context,
    )
    from uipath_langchain.agent.tools.datafabric_tool.models import SQLContext

    ctx = SQLContext(
        base_system_prompt="You are an agent.",
        resource_description="Domain notes for the ontology.",
        sql_expert_system_prompt="strategy",
        constraints="constraints",
        entity_contexts=[],
    )
    prompt = format_ontology_context(ctx)
    assert "## Agent Instructions" in prompt
    assert "You are an agent." in prompt
    assert "## Ontology description" in prompt
    assert "Domain notes for the ontology." in prompt
