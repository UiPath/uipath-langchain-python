"""Tests for Data Fabric write validation logic."""

from __future__ import annotations

from unittest.mock import MagicMock

from uipath_langchain.agent.tools.datafabric_tool.compiled_ontology import (
    CompiledOntology,
)
from uipath_langchain.agent.tools.datafabric_tool.models import (
    DataFabricWriteInput,
    EntityWriteOperation,
    EntityWriteSchema,
    WritableFieldInfo,
)
from uipath_langchain.agent.tools.datafabric_tool.write_validation import (
    derive_writable_fields,
    is_entity_writable,
    validate_mutation_intent,
)


def _make_field(
    name: str,
    display_name: str | None = None,
    is_system_field: bool = False,
    is_hidden_field: bool = False,
    is_primary_key: bool = False,
    is_attachment: bool = False,
    is_required: bool = False,
    sql_type_name: str = "varchar",
    description: str | None = None,
    choiceset_id: str | None = None,
) -> MagicMock:
    """Create a mock FieldMetadata object."""
    field = MagicMock()
    field.name = name
    field.display_name = display_name or name
    field.is_system_field = is_system_field
    field.is_hidden_field = is_hidden_field
    field.is_primary_key = is_primary_key
    field.is_attachment = is_attachment
    field.is_required = is_required
    field.description = description
    field.choiceset_id = choiceset_id
    sql_type = MagicMock()
    sql_type.name = sql_type_name
    field.sql_type = sql_type
    return field


def _make_entity(
    name: str,
    fields: list[MagicMock],
    entity_type: str = "Entity",
    external_fields: list | None = None,
) -> MagicMock:
    """Create a mock Entity object."""
    entity = MagicMock()
    entity.name = name
    entity.display_name = name
    entity.fields = fields
    entity.entity_type = entity_type
    entity.external_fields = external_fields
    return entity


class TestIsEntityWritable:
    """Tests for is_entity_writable."""

    def test_native_entity_is_writable(self) -> None:
        entity = _make_entity("Orders", [], entity_type="Entity", external_fields=None)
        assert is_entity_writable(entity) is True

    def test_native_entity_empty_external_fields_is_writable(self) -> None:
        entity = _make_entity("Orders", [], entity_type="Entity", external_fields=[])
        assert is_entity_writable(entity) is True

    def test_federated_entity_is_not_writable(self) -> None:
        entity = _make_entity(
            "SalesForceAccounts",
            [],
            entity_type="Entity",
            external_fields=[{"source": "salesforce"}],
        )
        assert is_entity_writable(entity) is False

    def test_choiceset_is_not_writable(self) -> None:
        entity = _make_entity("StatusOptions", [], entity_type="ChoiceSet")
        assert is_entity_writable(entity) is False

    def test_system_entity_is_not_writable(self) -> None:
        entity = _make_entity("AuditLog", [], entity_type="SystemEntity")
        assert is_entity_writable(entity) is False

    def test_internal_entity_is_not_writable(self) -> None:
        entity = _make_entity("Internal", [], entity_type="InternalEntity")
        assert is_entity_writable(entity) is False


class TestDeriveWritableFields:
    """Tests for derive_writable_fields."""

    def test_filters_system_fields(self) -> None:
        entity = _make_entity(
            "Orders",
            [
                _make_field("Id", is_primary_key=True),
                _make_field("CreatedOn", is_system_field=True),
                _make_field("ModifiedOn", is_system_field=True),
                _make_field("OrderName"),
                _make_field("Amount", sql_type_name="decimal"),
            ],
        )
        result = derive_writable_fields(entity)
        names = [f.name for f in result]
        assert "OrderName" in names
        assert "Amount" in names
        assert "Id" not in names
        assert "CreatedOn" not in names
        assert "ModifiedOn" not in names

    def test_filters_hidden_fields(self) -> None:
        entity = _make_entity(
            "Orders",
            [
                _make_field("InternalRef", is_hidden_field=True),
                _make_field("OrderName"),
            ],
        )
        result = derive_writable_fields(entity)
        names = [f.name for f in result]
        assert "OrderName" in names
        assert "InternalRef" not in names

    def test_filters_attachment_fields(self) -> None:
        entity = _make_entity(
            "Orders",
            [
                _make_field("Document", is_attachment=True),
                _make_field("OrderName"),
            ],
        )
        result = derive_writable_fields(entity)
        names = [f.name for f in result]
        assert "OrderName" in names
        assert "Document" not in names

    def test_preserves_required_flag(self) -> None:
        entity = _make_entity(
            "Orders",
            [
                _make_field("OrderName", is_required=True),
                _make_field("Notes"),
            ],
        )
        result = derive_writable_fields(entity)
        by_name = {f.name: f for f in result}
        assert by_name["OrderName"].is_required is True
        assert by_name["Notes"].is_required is False

    def test_preserves_type_and_description(self) -> None:
        entity = _make_entity(
            "Orders",
            [
                _make_field(
                    "Amount",
                    sql_type_name="decimal",
                    description="Order total",
                ),
            ],
        )
        result = derive_writable_fields(entity)
        assert len(result) == 1
        assert result[0].type_name == "decimal"
        assert result[0].description == "Order total"

    def test_empty_fields(self) -> None:
        entity = _make_entity("Empty", [])
        assert derive_writable_fields(entity) == []

    def test_none_fields(self) -> None:
        entity = MagicMock()
        entity.fields = None
        assert derive_writable_fields(entity) == []

    def test_all_fields_filtered_out(self) -> None:
        entity = _make_entity(
            "SystemOnly",
            [
                _make_field("Id", is_primary_key=True),
                _make_field("CreatedOn", is_system_field=True),
            ],
        )
        assert derive_writable_fields(entity) == []

    def test_returns_empty_for_non_writable_entity(self) -> None:
        """Federated entities return no writable fields even if they have user fields."""
        entity = _make_entity(
            "Federated",
            [_make_field("Name"), _make_field("Value")],
            entity_type="Entity",
            external_fields=[{"source": "ext"}],
        )
        assert derive_writable_fields(entity) == []

    def test_returns_empty_for_choiceset(self) -> None:
        entity = _make_entity(
            "StatusOptions",
            [_make_field("Label")],
            entity_type="ChoiceSet",
        )
        assert derive_writable_fields(entity) == []

    def test_choiceset_field_sets_is_choiceset(self) -> None:
        """Field with choiceset_id gets is_choiceset=True and stores the id."""
        entity = _make_entity(
            "Orders",
            [
                _make_field(
                    "Status",
                    choiceset_id="OrderStatusCS",
                ),
            ],
        )
        result = derive_writable_fields(entity)
        assert len(result) == 1
        assert result[0].is_choiceset is True
        assert result[0].choiceset_id == "OrderStatusCS"

    def test_non_choiceset_field_has_is_choiceset_false(self) -> None:
        """Field without choiceset_id gets is_choiceset=False."""
        entity = _make_entity(
            "Orders",
            [_make_field("OrderName")],
        )
        result = derive_writable_fields(entity)
        assert len(result) == 1
        assert result[0].is_choiceset is False
        assert result[0].choiceset_id is None

    def test_mixed_choiceset_and_regular_fields(self) -> None:
        """Mix of choiceset and regular fields are handled correctly."""
        entity = _make_entity(
            "Orders",
            [
                _make_field("OrderName"),
                _make_field("Status", choiceset_id="OrderStatusCS"),
                _make_field("Amount", sql_type_name="decimal"),
            ],
        )
        result = derive_writable_fields(entity)
        by_name = {f.name: f for f in result}
        assert by_name["OrderName"].is_choiceset is False
        assert by_name["Status"].is_choiceset is True
        assert by_name["Status"].choiceset_id == "OrderStatusCS"
        assert by_name["Amount"].is_choiceset is False


class TestValidateMutationIntent:
    """Tests for validate_mutation_intent."""

    def _schema(self) -> dict[str, EntityWriteSchema]:
        """Build a sample write_schemas dict."""
        return {
            "Orders": EntityWriteSchema(
                entity_key="Orders",
                display_name="Orders",
                writable_fields=[
                    WritableFieldInfo(
                        name="OrderName",
                        display_name="Order Name",
                        type_name="varchar",
                        is_required=True,
                    ),
                    WritableFieldInfo(
                        name="Amount",
                        display_name="Amount",
                        type_name="decimal",
                        is_required=False,
                    ),
                    WritableFieldInfo(
                        name="Notes",
                        display_name="Notes",
                        type_name="varchar",
                        is_required=False,
                    ),
                ],
            )
        }

    # -- Structural validation --

    def test_delete_requires_record_id(self) -> None:
        intent = DataFabricWriteInput(
            entity_key="Orders",
            operation=EntityWriteOperation.delete,
        )
        errors = validate_mutation_intent(intent, self._schema())
        assert any("record_id" in e for e in errors)

    def test_update_requires_record_id(self) -> None:
        intent = DataFabricWriteInput(
            entity_key="Orders",
            operation=EntityWriteOperation.update,
            fields={"Amount": 100},
        )
        errors = validate_mutation_intent(intent, self._schema())
        assert len(errors) == 1
        assert "record_id" in errors[0]

    def test_insert_requires_fields(self) -> None:
        intent = DataFabricWriteInput(
            entity_key="Orders",
            operation=EntityWriteOperation.insert,
        )
        errors = validate_mutation_intent(intent, self._schema())
        assert any("fields" in e for e in errors)

    def test_update_requires_fields(self) -> None:
        intent = DataFabricWriteInput(
            entity_key="Orders",
            operation=EntityWriteOperation.update,
            record_id="rec-1",
        )
        errors = validate_mutation_intent(intent, self._schema())
        assert len(errors) == 1
        assert "fields" in errors[0]

    # -- Entity not configured for writes --

    def test_entity_not_in_schemas_returns_error(self) -> None:
        """Entity not in write_schemas returns a validation error."""
        intent = DataFabricWriteInput(
            entity_key="Unknown",
            operation=EntityWriteOperation.insert,
            fields={"Anything": "goes"},
        )
        errors = validate_mutation_intent(intent, write_schemas=self._schema())
        assert len(errors) == 1
        assert "not configured for writes" in errors[0]
        assert "Orders" in errors[0]  # listed as writable

    def test_no_write_schemas_returns_error(self) -> None:
        """No write_schemas at all means nothing is writable."""
        intent = DataFabricWriteInput(
            entity_key="SomeEntity",
            operation=EntityWriteOperation.insert,
            fields={"AnyField": "value"},
        )
        errors = validate_mutation_intent(intent, write_schemas=None)
        assert len(errors) == 1
        assert "not configured for writes" in errors[0]

    def test_empty_write_schemas_returns_error(self) -> None:
        """Empty write_schemas means nothing is writable."""
        intent = DataFabricWriteInput(
            entity_key="SomeEntity",
            operation=EntityWriteOperation.delete,
            record_id="rec-99",
        )
        errors = validate_mutation_intent(intent, write_schemas={})
        assert len(errors) == 1
        assert "not configured for writes" in errors[0]

    # -- Strict mode (context-derived) --

    def test_strict_insert_valid(self) -> None:
        intent = DataFabricWriteInput(
            entity_key="Orders",
            operation=EntityWriteOperation.insert,
            fields={"OrderName": "Test", "Amount": 50},
        )
        errors = validate_mutation_intent(intent, self._schema())
        assert errors == []

    def test_strict_insert_missing_required(self) -> None:
        intent = DataFabricWriteInput(
            entity_key="Orders",
            operation=EntityWriteOperation.insert,
            fields={"Amount": 50},
        )
        errors = validate_mutation_intent(intent, self._schema())
        assert len(errors) == 1
        assert "OrderName" in errors[0]

    def test_strict_insert_unknown_field(self) -> None:
        intent = DataFabricWriteInput(
            entity_key="Orders",
            operation=EntityWriteOperation.insert,
            fields={"OrderName": "Test", "Bogus": "value"},
        )
        errors = validate_mutation_intent(intent, self._schema())
        assert len(errors) == 1
        assert "Bogus" in errors[0]

    def test_strict_update_no_required_enforcement(self) -> None:
        """UPDATE does not enforce required fields - agent decides what to change."""
        intent = DataFabricWriteInput(
            entity_key="Orders",
            operation=EntityWriteOperation.update,
            record_id="rec-1",
            fields={"Notes": "Updated notes"},
        )
        errors = validate_mutation_intent(intent, self._schema())
        assert errors == []

    def test_strict_update_unknown_field(self) -> None:
        intent = DataFabricWriteInput(
            entity_key="Orders",
            operation=EntityWriteOperation.update,
            record_id="rec-1",
            fields={"Bogus": "value"},
        )
        errors = validate_mutation_intent(intent, self._schema())
        assert len(errors) == 1
        assert "Bogus" in errors[0]

    def test_strict_delete_valid(self) -> None:
        intent = DataFabricWriteInput(
            entity_key="Orders",
            operation=EntityWriteOperation.delete,
            record_id="rec-1",
        )
        errors = validate_mutation_intent(intent, self._schema())
        assert errors == []

    def test_multiple_errors(self) -> None:
        """INSERT with unknown fields AND missing required fields."""
        intent = DataFabricWriteInput(
            entity_key="Orders",
            operation=EntityWriteOperation.insert,
            fields={"Bogus": "value"},
        )
        errors = validate_mutation_intent(intent, self._schema())
        assert len(errors) == 2  # unknown field + missing required


class TestValidateMutationIntentWithOntology:
    """Ontology-constrained operation validation (optional layer)."""

    def _schema(self) -> dict[str, EntityWriteSchema]:
        return {
            "Orders": EntityWriteSchema(
                entity_key="Orders",
                display_name="Orders",
                writable_fields=[
                    WritableFieldInfo(
                        name="OrderName",
                        display_name="Order Name",
                        type_name="varchar",
                        is_required=True,
                    ),
                    WritableFieldInfo(
                        name="Amount",
                        display_name="Amount",
                        type_name="decimal",
                        is_required=False,
                    ),
                ],
            )
        }

    def test_ontology_disallows_operation_returns_error(self) -> None:
        """Ontology allows only update on Orders -> insert is rejected."""
        ontology = CompiledOntology(entity_access={"Orders": {"update"}})
        intent = DataFabricWriteInput(
            entity_key="Orders",
            operation=EntityWriteOperation.insert,
            fields={"OrderName": "Test"},
        )
        errors = validate_mutation_intent(intent, self._schema(), ontology)
        assert len(errors) == 1
        assert "not allowed" in errors[0]
        assert "insert" in errors[0]
        assert "update" in errors[0]  # lists allowed ops

    def test_ontology_allows_operation_passes(self) -> None:
        """Ontology allows insert -> a valid insert passes."""
        ontology = CompiledOntology(entity_access={"Orders": {"insert", "update"}})
        intent = DataFabricWriteInput(
            entity_key="Orders",
            operation=EntityWriteOperation.insert,
            fields={"OrderName": "Test", "Amount": 50},
        )
        errors = validate_mutation_intent(intent, self._schema(), ontology)
        assert errors == []

    def test_ontology_allows_operation_update_passes(self) -> None:
        ontology = CompiledOntology(entity_access={"Orders": {"update"}})
        intent = DataFabricWriteInput(
            entity_key="Orders",
            operation=EntityWriteOperation.update,
            record_id="rec-1",
            fields={"Amount": 99},
        )
        errors = validate_mutation_intent(intent, self._schema(), ontology)
        assert errors == []

    def test_ontology_none_preserves_existing_behavior(self) -> None:
        """ontology=None -> metadata-only validation, insert still allowed."""
        intent = DataFabricWriteInput(
            entity_key="Orders",
            operation=EntityWriteOperation.insert,
            fields={"OrderName": "Test", "Amount": 50},
        )
        errors = validate_mutation_intent(intent, self._schema(), None)
        assert errors == []

    def test_ontology_without_entry_for_entity_does_not_constrain(self) -> None:
        """Entity absent from ontology.entity_access -> no op constraint applied."""
        ontology = CompiledOntology(entity_access={"OtherEntity": {"update"}})
        intent = DataFabricWriteInput(
            entity_key="Orders",
            operation=EntityWriteOperation.insert,
            fields={"OrderName": "Test"},
        )
        errors = validate_mutation_intent(intent, self._schema(), ontology)
        assert errors == []

    def test_ontology_read_only_entity_rejected(self) -> None:
        """Entity known to the ontology but with no write ops -> read-only reject."""
        # Orders is declared (known) but not in entity_access -> read-only.
        ontology = CompiledOntology(
            known_entities={"Orders", "RefundRequest"},
            entity_access={"RefundRequest": {"insert"}},
        )
        intent = DataFabricWriteInput(
            entity_key="Orders",
            operation=EntityWriteOperation.update,
            record_id="rec-1",
            fields={"Amount": 10},
        )
        errors = validate_mutation_intent(intent, self._schema(), ontology)
        assert len(errors) == 1
        assert "read-only" in errors[0]

    def test_ontology_unknown_entity_falls_back_to_metadata(self) -> None:
        """Entity unknown to the ontology (not in known_entities) is NOT rejected
        on a read-only basis — it falls back to metadata validation."""
        # Orders is neither in known_entities nor entity_access -> unknown.
        ontology = CompiledOntology(
            known_entities={"RefundRequest"},
            entity_access={"RefundRequest": {"insert"}},
        )
        intent = DataFabricWriteInput(
            entity_key="Orders",
            operation=EntityWriteOperation.insert,
            fields={"OrderName": "Test", "Amount": 50},
        )
        errors = validate_mutation_intent(intent, self._schema(), ontology)
        # Passes metadata validation (no read-only rejection applied).
        assert errors == []
