from types import SimpleNamespace

from uipath_langchain.agent.tools.datafabric_tool.datafabric_prompt_builder import (
    build,
    build_entity_context,
)


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
        is_primary_key=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _pk_field(**overrides):
    """A system-managed primary key field (e.g. the platform ``Id``)."""
    return _fake_field(
        name="Id",
        display_name="Id",
        sql_type=SimpleNamespace(name="uniqueidentifier"),
        description="Record id",
        is_system_field=True,
        is_primary_key=True,
        is_unique=True,
        **overrides,
    )


def _fake_entity(*fields, **overrides):
    overrides.setdefault("entity_type", "Entity")
    overrides.setdefault("external_fields", None)
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


def _field_names(ctx):
    return [f.name for f in ctx.entity_schema.fields]


def test_writable_entity_retains_primary_key():
    """Writable entity surfaces the system primary key so the agent can fetch
    record ids (record_id for writes) — clearing the rowid/record_id gap."""
    ctx = build_entity_context(_fake_entity(_pk_field(), _fake_field()))
    names = _field_names(ctx)
    assert "Id" in names, "primary key must be surfaced for writable entities"
    # Id is first in every projection so record-level reads return it.
    show_all = next(p for p in ctx.query_patterns if p.intent == "Show all")
    assert "SELECT Id" in show_all.sql
    # A real column to ORDER BY — never the non-existent 'rowid' pseudo-column.
    assert "ORDER BY Id" in show_all.sql
    assert "rowid" not in show_all.sql
    # An explicit record-lookup pattern is offered for writes.
    assert any("update/delete" in p.intent for p in ctx.query_patterns)


def test_readonly_entity_excludes_system_primary_key():
    """Federated/read-only entities keep all system fields hidden — no write
    means no need for identity-for-mutation."""
    ctx = build_entity_context(
        _fake_entity(_pk_field(), _fake_field(), external_fields=[object()])
    )
    names = _field_names(ctx)
    assert "Id" not in names
    assert "status" in names
    # No record-lookup pattern for a non-writable entity.
    assert not any("update/delete" in p.intent for p in ctx.query_patterns)


def test_other_system_fields_stay_hidden_on_writable_entity():
    """Only the primary key is surfaced; other system fields remain noise."""
    ctx = build_entity_context(
        _fake_entity(
            _pk_field(),
            _fake_field(name="CreateTime", is_system_field=True, is_primary_key=False),
            _fake_field(),
        )
    )
    names = _field_names(ctx)
    assert "Id" in names
    assert "CreateTime" not in names
    assert "status" in names


def test_system_pk_name_collision_with_user_field_not_duplicated():
    """P3 collision guard: a user/CSV field sharing the system PK name must not
    produce a duplicate column row."""
    ctx = build_entity_context(
        _fake_entity(
            _fake_field(name="Id", is_system_field=False, is_primary_key=False),
            _pk_field(),  # system Id with the same name
            _fake_field(),
        )
    )
    names = _field_names(ctx)
    assert names.count("Id") == 1
