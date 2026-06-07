from types import SimpleNamespace

from uipath_langchain.agent.tools.datafabric_tool.datafabric_prompt_builder import build


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
