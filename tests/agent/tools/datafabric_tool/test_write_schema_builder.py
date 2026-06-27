"""Tests for write_schema_builder — NL schema generation for write tool descriptions."""

from __future__ import annotations

from uipath_langchain.agent.tools.datafabric_tool.models import (
    EntityWriteSchema,
    WritableFieldInfo,
)
from uipath_langchain.agent.tools.datafabric_tool.write_schema_builder import (
    build_write_tool_description,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _field(
    name: str,
    type_name: str = "varchar",
    is_required: bool = False,
    is_choiceset: bool = False,
    choiceset_id: str | None = None,
    description: str | None = None,
) -> WritableFieldInfo:
    return WritableFieldInfo(
        name=name,
        display_name=name,
        type_name=type_name,
        is_required=is_required,
        is_choiceset=is_choiceset,
        choiceset_id=choiceset_id,
        description=description,
    )


def _schema(
    entity_key: str,
    display_name: str,
    fields: list[WritableFieldInfo],
) -> EntityWriteSchema:
    return EntityWriteSchema(
        entity_key=entity_key,
        display_name=display_name,
        writable_fields=fields,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildWriteToolDescription:
    """Tests for build_write_tool_description."""

    def test_empty_schemas_returns_no_entities_message(self) -> None:
        result = build_write_tool_description({})
        assert "No writable entities" in result

    def test_single_entity_with_fields(self) -> None:
        schemas = {
            "Orders": _schema(
                "Orders",
                "Orders",
                [
                    _field("OrderName", is_required=True),
                    _field("Amount", type_name="decimal"),
                ],
            )
        }
        result = build_write_tool_description(schemas)
        assert "### Orders" in result
        assert "OrderName" in result
        assert "required" in result
        assert "DECIMAL" in result

    def test_multiple_entities_sorted_by_key(self) -> None:
        schemas = {
            "Zebra": _schema("Zebra", "Zebra", [_field("Name")]),
            "Alpha": _schema("Alpha", "Alpha", [_field("Value")]),
        }
        result = build_write_tool_description(schemas)
        alpha_pos = result.index("### Alpha")
        zebra_pos = result.index("### Zebra")
        assert alpha_pos < zebra_pos

    def test_choiceset_field_shows_choice_set_indicator(self) -> None:
        schemas = {
            "Orders": _schema(
                "Orders",
                "Orders",
                [
                    _field(
                        "Status",
                        type_name="varchar",
                        is_choiceset=True,
                        choiceset_id="OrderStatusCS",
                    ),
                ],
            )
        }
        result = build_write_tool_description(schemas)
        assert "CHOICE_SET" in result

    def test_entity_access_restricts_operations(self) -> None:
        schemas = {
            "RefundRequest": _schema(
                "RefundRequest",
                "Refund Request",
                [_field("Amount", type_name="decimal", is_required=True)],
            )
        }
        access = {"RefundRequest": {"insert"}}
        result = build_write_tool_description(schemas, entity_access=access)
        assert "### Refund Request (insert)" in result

    def test_entity_access_multiple_ops(self) -> None:
        schemas = {
            "Orders": _schema(
                "Orders",
                "Orders",
                [_field("Status")],
            )
        }
        access = {"Orders": {"update", "delete"}}
        result = build_write_tool_description(schemas, entity_access=access)
        assert "(delete, update)" in result

    def test_no_entity_access_shows_all_ops(self) -> None:
        schemas = {
            "Orders": _schema(
                "Orders",
                "Orders",
                [_field("Name")],
            )
        }
        result = build_write_tool_description(schemas, entity_access=None)
        assert "(delete, insert, update)" in result

    def test_operations_section_present(self) -> None:
        schemas = {
            "Orders": _schema("Orders", "Orders", [_field("Name")]),
        }
        result = build_write_tool_description(schemas)
        assert "Operations:" in result
        assert "- insert:" in result
        assert "- update:" in result
        assert "- delete:" in result

    def test_query_first_advice_present(self) -> None:
        schemas = {
            "Orders": _schema("Orders", "Orders", [_field("Name")]),
        }
        result = build_write_tool_description(schemas)
        assert "Query the entity first" in result

    def test_required_field_formatting(self) -> None:
        schemas = {
            "Items": _schema(
                "Items",
                "Items",
                [
                    _field("ItemName", is_required=True),
                    _field("Notes", is_required=False),
                ],
            )
        }
        result = build_write_tool_description(schemas)
        # Required field should have "required" in its line
        lines = result.split("\n")
        item_name_line = [ln for ln in lines if "ItemName" in ln][0]
        notes_line = [ln for ln in lines if "Notes" in ln][0]
        assert "required" in item_name_line
        assert "required" not in notes_line

    def test_hero_case_refund_schema(self) -> None:
        """Full hero case: multiple entities with mixed ops, choicesets, required fields."""
        schemas = {
            "PurchaseOrder": _schema(
                "PurchaseOrder",
                "PurchaseOrder",
                [
                    _field("OrderNumber", is_required=True),
                    _field("TotalAmount", type_name="decimal"),
                    _field(
                        "OrderStatus",
                        is_choiceset=True,
                        choiceset_id="OrderStatusCS",
                    ),
                ],
            ),
            "RefundRequest": _schema(
                "RefundRequest",
                "RefundRequest",
                [
                    _field("ApprovedAmount", type_name="decimal", is_required=True),
                    _field("Reason", is_required=True),
                    _field("OrderRef"),
                    _field("CustomerRef"),
                    _field("RefundStatus", is_choiceset=True, choiceset_id="RefundCS"),
                ],
            ),
            "CustomerRisk": _schema(
                "CustomerRisk",
                "CustomerRisk",
                [
                    _field("RiskScore", type_name="int"),
                    _field("LifetimeValue", type_name="decimal"),
                    _field("FraudFlag", type_name="bit"),
                ],
            ),
        }
        access = {
            "PurchaseOrder": {"update"},
            "RefundRequest": {"insert"},
            "CustomerRisk": {"update"},
        }
        result = build_write_tool_description(schemas, entity_access=access)

        # All entities present
        assert "PurchaseOrder" in result
        assert "RefundRequest" in result
        assert "CustomerRisk" in result

        # Ops are correct
        assert "(update)" in result  # PurchaseOrder and CustomerRisk
        assert "(insert)" in result  # RefundRequest

        # ChoiceSet indicators
        assert "CHOICE_SET" in result

        # Required fields
        assert "required" in result


# ---------------------------------------------------------------------------
# WritableFieldInfo choiceset extension
# ---------------------------------------------------------------------------


class TestWritableFieldInfoChoiceset:
    """Tests for the choiceset extension on WritableFieldInfo."""

    def test_choiceset_fields_default_to_false(self) -> None:
        field = WritableFieldInfo(
            name="Status",
            display_name="Status",
            type_name="varchar",
            is_required=False,
        )
        assert field.is_choiceset is False
        assert field.choiceset_id is None
        assert field.allowed_values is None

    def test_choiceset_field_with_id(self) -> None:
        field = WritableFieldInfo(
            name="Status",
            display_name="Status",
            type_name="varchar",
            is_required=False,
            choiceset_id="OrderStatusCS",
            is_choiceset=True,
        )
        assert field.is_choiceset is True
        assert field.choiceset_id == "OrderStatusCS"

    def test_allowed_values_can_be_set(self) -> None:
        field = WritableFieldInfo(
            name="Status",
            display_name="Status",
            type_name="varchar",
            is_required=False,
            choiceset_id="OrderStatusCS",
            is_choiceset=True,
            allowed_values=["Pending", "Approved", "Denied"],
        )
        assert field.allowed_values == ["Pending", "Approved", "Denied"]
