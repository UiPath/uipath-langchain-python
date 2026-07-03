from types import SimpleNamespace

from uipath_langchain.agent.tools.datafabric_tool.datafabric_prompt_builder import build


def _fake_field(**overrides):
    defaults = dict(
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
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


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


def test_build_renders_ecp_aware_prompt_strategy():
    """When ECP metadata is present the v1 prompt strategy upgrades to
    ECP-aware value resolution and aggregation hints."""
    prompt = build([_fake_entity(_fake_field())], resource_description="")

    # ECP-aware value resolution strategy (overrides the generic default)
    assert "allowed_values" in prompt
    assert "good_for_aggregation" in prompt

    # Entity schema table renders the field
    assert "| status |" in prompt
    assert "Ticket" in prompt


def test_build_includes_domain_guidance_in_rendered_prompt():
    prompt = build(
        [_fake_entity(_fake_field())],
        resource_description="Use business-friendly ticket language.",
    )

    assert "## Domain Guidance" in prompt
    assert "Use business-friendly ticket language." in prompt


def test_entity_prompt_never_contains_ontology_sections():
    # The entity tool's prompt is ontology-free; ontology content lives only in
    # datafabric_ontology_prompt_builder.
    prompt = build([_fake_entity(_fake_field())])
    assert "## Available Ontology" not in prompt
    assert "## Ontology→Table Mapping (R2RML)" not in prompt


def _system_field(name, type_name="datetimeoffset", **overrides):
    """A fake auto-added system/audit field (Id, CreateTime, ...)."""
    return _fake_field(
        name=name,
        display_name=name,
        sql_type=SimpleNamespace(name=type_name),
        description="System built-in field",
        is_system_field=True,
        allowed_values=None,
        examples=None,
        good_for_aggregation=False,
        good_for_grouping=False,
        good_for_filtering=False,
        **overrides,
    )


def test_surfaces_tagged_system_fields_with_descriptions():
    """System fields are surfaced, tagged ``system``, in a Description-column table."""
    entity = _fake_entity(
        _fake_field(name="status"),
        _system_field("CreateTime"),
        _system_field("CreatedBy", type_name="uniqueidentifier"),
    )
    prompt = build([entity])

    assert "| Field | Type | Description |" in prompt
    assert "| CreateTime | datetimeoffset, system |" in prompt
    assert "System built-in field" in prompt
    # system/audit field-selection guidance is part of the prompt
    assert "SYSTEM / AUDIT FIELDS" in prompt


def test_query_patterns_exclude_system_fields():
    """Surfaced system fields must never drive the derived query patterns."""
    entity = _fake_entity(
        _fake_field(name="status"),
        _system_field("CreateTime"),
        _system_field("Id", type_name="uniqueidentifier"),
    )
    prompt = build([entity])

    patterns_block = prompt.split("Query Patterns for Ticket", 1)[1]
    assert "CreateTime" not in patterns_block
    assert "Id" not in patterns_block
