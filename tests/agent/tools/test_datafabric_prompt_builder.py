from types import SimpleNamespace

from uipath_langchain.agent.tools.datafabric_tool.datafabric_prompt_builder import build


def _fake_field(name="status", **overrides):
    defaults = dict(
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
        field_display_type=None,
        reference_entity=None,
        reference_field=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(name=name, **defaults)


def _fake_fk_field(name="account", ref_table="Account", ref_field="Id", **overrides):
    return _fake_field(
        name=name,
        display_name=name.title(),
        description=f"Reference to {ref_table}",
        is_foreign_key=True,
        field_display_type="Relationship",
        reference_entity=SimpleNamespace(name=ref_table),
        reference_field=SimpleNamespace(definition=SimpleNamespace(name=ref_field)),
        **overrides,
    )


def _fake_entity(*fields, name="Ticket", **overrides):
    defaults = dict(
        id="entity-1",
        display_name="Ticket",
        description="Support tickets",
        record_count=10,
    )
    defaults.update(overrides)
    return SimpleNamespace(name=name, fields=list(fields), **defaults)


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


def test_relationship_field_renders_join_when_target_entity_present():
    order = _fake_entity(
        _fake_field(),
        _fake_fk_field(ref_field="Name"),
        name="Order",
        display_name="Order",
    )
    account = _fake_entity(
        _fake_field(name="Name"), name="Account", display_name="Account"
    )

    prompt = build([order, account])

    # The FK column is tagged; the join is spelled out against the target Id as a
    # LEFT JOIN (keeps parent rows), and the representative field is surfaced.
    assert "| account | varchar, fk |" in prompt
    assert "**Relationships for Order:**" in prompt
    assert "LEFT JOIN Account ON Account.Id = Order.account" in prompt
    assert "representative field `Account.Name`" in prompt


def test_relationship_detected_via_display_type_without_is_foreign_key():
    # A Relationship-typed field with is_foreign_key unset must still be tagged
    # fk and rendered in the Relationships section.
    relationship_field = _fake_field(
        name="account",
        display_name="Account",
        field_display_type="Relationship",
        reference_entity=SimpleNamespace(name="Account"),
        reference_field=SimpleNamespace(definition=SimpleNamespace(name="Name")),
    )
    order = _fake_entity(relationship_field, name="Order", display_name="Order")
    account = _fake_entity(
        _fake_field(name="Name"), name="Account", display_name="Account"
    )

    prompt = build([order, account])

    assert "| account | varchar, fk |" in prompt
    assert "LEFT JOIN Account ON Account.Id = Order.account" in prompt


def test_v1_prompt_documents_left_vs_inner_join_intent():
    prompt = build([_fake_entity(_fake_field())])

    # The relationship guidance explains when to use LEFT vs INNER.
    assert "LEFT JOIN" in prompt
    assert "INNER JOIN" in prompt


def test_relationship_subsection_absent_when_no_foreign_keys():
    prompt = build([_fake_entity(_fake_field())])

    # The rendered per-entity header (distinct from the static prompt guidance
    # that mentions "Relationships for <table>") must not appear.
    assert "**Relationships for Ticket:**" not in prompt


def test_relationship_omitted_when_target_entity_not_in_set():
    # Order references Account, but Account is not part of the entity set, so a
    # join would be unusable — the relationship line must be suppressed.
    order = _fake_entity(
        _fake_field(), _fake_fk_field(), name="Order", display_name="Order"
    )

    prompt = build([order])

    assert "**Relationships for Order:**" not in prompt
    assert "INNER JOIN Account" not in prompt


def test_v1_prompt_documents_relationship_fields():
    prompt = build([_fake_entity(_fake_field())])

    assert "RELATIONSHIP FIELDS" in prompt
