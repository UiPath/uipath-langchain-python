"""Integration tests for the full Data Fabric write tool creation pipeline.

Verifies end-to-end tool creation, metadata, schema, validation, and
execution flow WITHOUT a live UiPath connection. All platform calls are
mocked.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.tools import BaseTool
from uipath.agent.models.agent import AgentContextResourceConfig, AgentContextType
from uipath.platform.entities import DataFabricEntityItem

from uipath_langchain.agent.tools.datafabric_tool import create_datafabric_tools
from uipath_langchain.agent.tools.datafabric_tool.models import (
    DataFabricQueryInput,
    DataFabricWriteInput,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    """Create a mock Entity object with .name, .display_name, .fields."""
    entity = MagicMock()
    entity.name = name
    # In these tests the entity id equals the name so the handler's name->id
    # translation is an identity (a distinct-id case is covered separately).
    entity.id = name
    entity.display_name = name
    entity.fields = fields
    entity.entity_type = entity_type
    entity.external_fields = external_fields
    return entity


def _make_resolution(entities: list[MagicMock]) -> MagicMock:
    """Create a mock resolution result from resolve_entity_set_async."""
    resolution = MagicMock()
    resolution.entities = entities
    resolution.entities_service = MagicMock()
    return resolution


def _make_context_resource(
    name: str = "my_data_fabric",
    description: str = "Test Data Fabric tool",
    entity_items: list[dict] | None = None,
) -> AgentContextResourceConfig:
    """Build an AgentContextResourceConfig with DATA_FABRIC_ENTITY_SET type."""
    if entity_items is None:
        entity_items = [
            {
                "id": "e1",
                "name": "Orders",
                "folderId": "f1",
                "description": "Customer orders",
            },
            {
                "id": "e2",
                "name": "Products",
                "folderId": "f1",
                "description": "Product catalog",
            },
        ]
    entity_set = [DataFabricEntityItem.model_validate(item) for item in entity_items]
    return AgentContextResourceConfig(
        name=name,
        description=description,
        resource_type="context",
        context_type=AgentContextType.DATA_FABRIC_ENTITY_SET,
        entity_set=entity_set,
        is_enabled=True,
    )


def _mock_llm() -> MagicMock:
    """Create a mock LLM for the read tool's subgraph."""
    return MagicMock()


# ---------------------------------------------------------------------------
# 1. create_datafabric_tools returns [read_tool, write_tool]
# ---------------------------------------------------------------------------


class TestCreateDatafabricToolsReturnsToolPair:
    """Verify create_datafabric_tools returns a list with read + write tools."""

    def test_returns_list_of_two_tools(self) -> None:
        resource = _make_context_resource()
        tools = create_datafabric_tools(resource, _mock_llm())
        assert isinstance(tools, list)
        assert len(tools) == 2
        assert all(isinstance(t, BaseTool) for t in tools)

    def test_read_tool_is_first(self) -> None:
        resource = _make_context_resource()
        tools = create_datafabric_tools(resource, _mock_llm())
        read_tool = tools[0]
        assert read_tool.metadata is not None
        assert read_tool.metadata.get("tool_type") == "datafabric_sql"

    def test_write_tool_is_second(self) -> None:
        resource = _make_context_resource()
        tools = create_datafabric_tools(resource, _mock_llm())
        write_tool = tools[1]
        assert write_tool.metadata is not None
        assert write_tool.metadata.get("tool_type") == "datafabric_write"

    def test_tool_names_match_convention(self) -> None:
        resource = _make_context_resource(name="customer_data")
        tools = create_datafabric_tools(
            resource, _mock_llm(), tool_name="customer_data"
        )
        assert tools[0].name == "customer_data"
        assert tools[1].name == "customer_data_write"


# ---------------------------------------------------------------------------
# 2. Write tool metadata
# ---------------------------------------------------------------------------


class TestWriteToolMetadata:
    """Verify the write tool's metadata flags."""

    def test_tool_type_is_datafabric_write(self) -> None:
        tools = create_datafabric_tools(_make_context_resource(), _mock_llm())
        write_tool = tools[1]
        assert write_tool.metadata is not None
        assert write_tool.metadata["tool_type"] == "datafabric_write"

    def test_require_conversational_confirmation_is_true(self) -> None:
        tools = create_datafabric_tools(_make_context_resource(), _mock_llm())
        write_tool = tools[1]
        assert write_tool.metadata is not None
        assert write_tool.metadata["require_conversational_confirmation"] is True


# ---------------------------------------------------------------------------
# 3. Write tool args_schema is DataFabricWriteInput
# ---------------------------------------------------------------------------


class TestWriteToolArgsSchema:
    """Verify the write tool's args_schema matches DataFabricWriteInput."""

    def test_args_schema_is_datafabric_write_input(self) -> None:
        tools = create_datafabric_tools(_make_context_resource(), _mock_llm())
        write_tool = tools[1]
        assert write_tool.args_schema is DataFabricWriteInput

    def test_schema_has_expected_fields(self) -> None:
        tools = create_datafabric_tools(_make_context_resource(), _mock_llm())
        schema = tools[1].args_schema.model_json_schema()
        props = schema["properties"]
        assert "entity_key" in props
        assert "operation" in props
        assert "record_id" in props
        assert "fields" in props


# ---------------------------------------------------------------------------
# 4. DataFabricWriteHandler validates before executing
# ---------------------------------------------------------------------------


class TestWriteHandlerValidationAndExecution:
    """End-to-end handler tests with mocked SDK. Verifies validation
    intercepts bad inputs and valid operations reach the API.
    """

    @pytest.fixture
    def mock_entities(self) -> list[MagicMock]:
        """Two mock entities: Orders (with required OrderName) and Products."""
        orders = _make_entity(
            "Orders",
            [
                _make_field("Id", is_primary_key=True),
                _make_field("CreatedOn", is_system_field=True),
                _make_field("OrderName", is_required=True),
                _make_field("Amount", sql_type_name="decimal"),
                _make_field("Notes"),
            ],
        )
        products = _make_entity(
            "Products",
            [
                _make_field("Id", is_primary_key=True),
                _make_field("ProductName", is_required=True),
                _make_field("Price", sql_type_name="decimal"),
            ],
        )
        return [orders, products]

    @pytest.fixture
    def mock_record(self) -> MagicMock:
        """A mock record returned by insert/update."""
        record = MagicMock()
        record.id = "rec-123"
        record.model_dump.return_value = {"Id": "rec-123", "OrderName": "Test"}
        return record

    @pytest.mark.asyncio
    async def test_insert_with_valid_fields_calls_api(
        self, mock_entities: list[MagicMock], mock_record: MagicMock
    ) -> None:
        """INSERT with valid fields on a context-derived entity calls insert_record_async."""
        resolution = _make_resolution(mock_entities)
        resolution.entities_service.insert_record_async = AsyncMock(
            return_value=mock_record
        )

        with patch("uipath.platform.UiPath") as mock_uipath_cls:
            mock_sdk = MagicMock()
            mock_sdk.entities.resolve_entity_set_async = AsyncMock(
                return_value=resolution
            )
            mock_uipath_cls.return_value = mock_sdk

            resource = _make_context_resource()
            tools = create_datafabric_tools(resource, _mock_llm())
            write_tool = tools[1]

            result_str = await write_tool.ainvoke(
                {
                    "entity_key": "Orders",
                    "operation": "insert",
                    "fields": {"OrderName": "New Order", "Amount": 99.99},
                }
            )
            result = json.loads(result_str)

            assert result["success"] is True
            assert result["operation"] == "insert"
            resolution.entities_service.insert_record_async.assert_awaited_once_with(
                "Orders", {"OrderName": "New Order", "Amount": 99.99}
            )

    @pytest.mark.asyncio
    async def test_update_with_record_id_and_fields_calls_api(
        self, mock_entities: list[MagicMock], mock_record: MagicMock
    ) -> None:
        """UPDATE with record_id + fields calls update_record_async."""
        resolution = _make_resolution(mock_entities)
        resolution.entities_service.update_record_async = AsyncMock(
            return_value=mock_record
        )

        with patch("uipath.platform.UiPath") as mock_uipath_cls:
            mock_sdk = MagicMock()
            mock_sdk.entities.resolve_entity_set_async = AsyncMock(
                return_value=resolution
            )
            mock_uipath_cls.return_value = mock_sdk

            tools = create_datafabric_tools(_make_context_resource(), _mock_llm())
            write_tool = tools[1]

            result_str = await write_tool.ainvoke(
                {
                    "entity_key": "Orders",
                    "operation": "update",
                    "record_id": "rec-1",
                    "fields": {"Amount": 150},
                }
            )
            result = json.loads(result_str)

            assert result["success"] is True
            assert result["operation"] == "update"
            resolution.entities_service.update_record_async.assert_awaited_once_with(
                "Orders", "rec-1", {"Amount": 150}
            )

    @pytest.mark.asyncio
    async def test_entity_name_translated_to_id_for_crud(
        self, mock_record: MagicMock
    ) -> None:
        """The LLM addresses the entity by name, but the CRUD call must use the
        entity's GUID id. The handler translates name -> id before executing."""
        orders = _make_entity("Orders", [_make_field("OrderName")])
        orders.id = "orders-guid-123"  # id distinct from the name
        resolution = _make_resolution([orders])
        resolution.entities_service.insert_record_async = AsyncMock(
            return_value=mock_record
        )

        with patch("uipath.platform.UiPath") as mock_uipath_cls:
            mock_sdk = MagicMock()
            mock_sdk.entities.resolve_entity_set_async = AsyncMock(
                return_value=resolution
            )
            mock_uipath_cls.return_value = mock_sdk

            tools = create_datafabric_tools(_make_context_resource(), _mock_llm())
            result = json.loads(
                await tools[1].ainvoke(
                    {
                        "entity_key": "Orders",  # name, as the LLM sees it
                        "operation": "insert",
                        "fields": {"OrderName": "X"},
                    }
                )
            )

            assert result["success"] is True
            # Executor called with the GUID id, not the entity name.
            resolution.entities_service.insert_record_async.assert_awaited_once_with(
                "orders-guid-123", {"OrderName": "X"}
            )
            # The result still reports the friendly name back to the model.
            assert result["entity_key"] == "Orders"

    @pytest.mark.asyncio
    async def test_delete_with_record_id_calls_api(
        self, mock_entities: list[MagicMock]
    ) -> None:
        """DELETE with record_id calls delete_record_async."""
        resolution = _make_resolution(mock_entities)
        resolution.entities_service.delete_record_async = AsyncMock(return_value=None)

        with patch("uipath.platform.UiPath") as mock_uipath_cls:
            mock_sdk = MagicMock()
            mock_sdk.entities.resolve_entity_set_async = AsyncMock(
                return_value=resolution
            )
            mock_uipath_cls.return_value = mock_sdk

            tools = create_datafabric_tools(_make_context_resource(), _mock_llm())
            write_tool = tools[1]

            result_str = await write_tool.ainvoke(
                {
                    "entity_key": "Orders",
                    "operation": "delete",
                    "record_id": "rec-42",
                }
            )
            result = json.loads(result_str)

            assert result["success"] is True
            assert result["operation"] == "delete"
            assert result["record_id"] == "rec-42"
            resolution.entities_service.delete_record_async.assert_awaited_once_with(
                "Orders", "rec-42"
            )

    @pytest.mark.asyncio
    async def test_insert_missing_required_field_returns_validation_error(
        self, mock_entities: list[MagicMock]
    ) -> None:
        """INSERT missing a required field returns a validation error, does NOT call API."""
        resolution = _make_resolution(mock_entities)
        resolution.entities_service.insert_record_async = AsyncMock()

        with patch("uipath.platform.UiPath") as mock_uipath_cls:
            mock_sdk = MagicMock()
            mock_sdk.entities.resolve_entity_set_async = AsyncMock(
                return_value=resolution
            )
            mock_uipath_cls.return_value = mock_sdk

            tools = create_datafabric_tools(_make_context_resource(), _mock_llm())
            write_tool = tools[1]

            # OrderName is required but not provided
            result_str = await write_tool.ainvoke(
                {
                    "entity_key": "Orders",
                    "operation": "insert",
                    "fields": {"Amount": 50},
                }
            )
            result = json.loads(result_str)

            assert result["success"] is False
            assert len(result["errors"]) >= 1
            assert "OrderName" in result["errors"][0]
            # API should NOT have been called
            resolution.entities_service.insert_record_async.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_insert_with_system_field_name_returns_validation_error(
        self, mock_entities: list[MagicMock]
    ) -> None:
        """INSERT with a system field name returns validation error."""
        resolution = _make_resolution(mock_entities)
        resolution.entities_service.insert_record_async = AsyncMock()

        with patch("uipath.platform.UiPath") as mock_uipath_cls:
            mock_sdk = MagicMock()
            mock_sdk.entities.resolve_entity_set_async = AsyncMock(
                return_value=resolution
            )
            mock_uipath_cls.return_value = mock_sdk

            tools = create_datafabric_tools(_make_context_resource(), _mock_llm())
            write_tool = tools[1]

            # CreatedOn is a system field and not writable
            result_str = await write_tool.ainvoke(
                {
                    "entity_key": "Orders",
                    "operation": "insert",
                    "fields": {"OrderName": "Test", "CreatedOn": "2026-01-01"},
                }
            )
            result = json.loads(result_str)

            assert result["success"] is False
            assert any("CreatedOn" in e for e in result["errors"])
            resolution.entities_service.insert_record_async.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_write_to_unknown_entity_returns_validation_error(
        self, mock_entities: list[MagicMock]
    ) -> None:
        """Writing to an entity not in write_schemas returns a validation error."""
        resolution = _make_resolution(mock_entities)
        resolution.entities_service.insert_record_async = AsyncMock()

        with patch("uipath.platform.UiPath") as mock_uipath_cls:
            mock_sdk = MagicMock()
            mock_sdk.entities.resolve_entity_set_async = AsyncMock(
                return_value=resolution
            )
            mock_uipath_cls.return_value = mock_sdk

            tools = create_datafabric_tools(_make_context_resource(), _mock_llm())
            write_tool = tools[1]

            # "UnknownEntity" is not in the resolved schemas
            result_str = await write_tool.ainvoke(
                {
                    "entity_key": "UnknownEntity",
                    "operation": "insert",
                    "fields": {"AnyField": "value"},
                }
            )
            result = json.loads(result_str)

            assert result["success"] is False
            assert any("not configured for writes" in e for e in result["errors"])
            resolution.entities_service.insert_record_async.assert_not_awaited()


# ---------------------------------------------------------------------------
# 5. Federated / non-writable entities
# ---------------------------------------------------------------------------


class TestFederatedEntitiesNotWritable:
    """Verify that federated and non-native entities are excluded from writes."""

    @pytest.mark.asyncio
    async def test_all_federated_entities_reject_writes(self) -> None:
        """When all resolved entities are federated, writes are rejected."""
        federated_orders = _make_entity(
            "Orders",
            [
                _make_field("Id", is_primary_key=True),
                _make_field("OrderName", is_required=True),
            ],
            entity_type="Entity",
            external_fields=[{"source": "salesforce"}],
        )
        federated_products = _make_entity(
            "Products",
            [
                _make_field("Id", is_primary_key=True),
                _make_field("ProductName", is_required=True),
            ],
            entity_type="Entity",
            external_fields=[{"source": "sap"}],
        )
        resolution = _make_resolution([federated_orders, federated_products])
        resolution.entities_service.insert_record_async = AsyncMock()

        with patch("uipath.platform.UiPath") as mock_uipath_cls:
            mock_sdk = MagicMock()
            mock_sdk.entities.resolve_entity_set_async = AsyncMock(
                return_value=resolution
            )
            mock_uipath_cls.return_value = mock_sdk

            tools = create_datafabric_tools(_make_context_resource(), _mock_llm())
            write_tool = tools[1]

            result_str = await write_tool.ainvoke(
                {
                    "entity_key": "Orders",
                    "operation": "insert",
                    "fields": {"OrderName": "Test"},
                }
            )
            result = json.loads(result_str)

            assert result["success"] is False
            assert any("not configured for writes" in e for e in result["errors"])
            resolution.entities_service.insert_record_async.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_choiceset_entity_not_writable(self) -> None:
        """ChoiceSet entities are excluded from writes."""
        choiceset = _make_entity(
            "StatusOptions",
            [_make_field("Label")],
            entity_type="ChoiceSet",
        )
        native = _make_entity(
            "Orders",
            [
                _make_field("Id", is_primary_key=True),
                _make_field("OrderName", is_required=True),
            ],
        )
        resolution = _make_resolution([choiceset, native])
        resolution.entities_service.insert_record_async = AsyncMock()

        with patch("uipath.platform.UiPath") as mock_uipath_cls:
            mock_sdk = MagicMock()
            mock_sdk.entities.resolve_entity_set_async = AsyncMock(
                return_value=resolution
            )
            mock_uipath_cls.return_value = mock_sdk

            tools = create_datafabric_tools(_make_context_resource(), _mock_llm())
            write_tool = tools[1]

            # ChoiceSet should be rejected
            result_str = await write_tool.ainvoke(
                {
                    "entity_key": "StatusOptions",
                    "operation": "insert",
                    "fields": {"Label": "Active"},
                }
            )
            result = json.loads(result_str)
            assert result["success"] is False
            assert any("not configured for writes" in e for e in result["errors"])


# ---------------------------------------------------------------------------
# 5b. Ontology prunes read-only entities from the write tool description
# ---------------------------------------------------------------------------


# Refund-set ontology: Customer is read-only (df:ReadableEntity); the rest are
# writable (df:WritableEntity) with action-derived operations.
_REFUND_PRUNE_OWL = """
@prefix df:   <https://ontology.uipath.com/datafabric#> .
@prefix ex:   <https://ontology.example.com/refund#> .
@prefix owl:  <http://www.w3.org/2002/07/owl#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

ex:Customer a owl:Class ; rdfs:subClassOf df:ReadableEntity ; df:entityKey "Customer" .
ex:RefundRequest a owl:Class ; rdfs:subClassOf df:WritableEntity ; df:entityKey "RefundRequest" .
ex:Order a owl:Class ; rdfs:subClassOf df:WritableEntity ; df:entityKey "Order" .
ex:Risk a owl:Class ; rdfs:subClassOf df:WritableEntity ; df:entityKey "Risk" .
ex:Contact a owl:Class ; rdfs:subClassOf df:WritableEntity ; df:entityKey "Contact" .

ex:CreateRefund a df:InsertAction ; df:writeOperation "insert" ; df:targetEntity ex:RefundRequest .
ex:UpdateOrder a df:UpdateAction ; df:writeOperation "update" ; df:targetEntity ex:Order .
ex:UpdateRisk a df:UpdateAction ; df:writeOperation "update" ; df:targetEntity ex:Risk .
ex:UpdateContact a df:UpdateAction ; df:writeOperation "update" ; df:targetEntity ex:Contact .
"""


class TestOntologyPrunesReadOnlyFromWriteDescription:
    """The write tool description must not advertise read-only entities."""

    @pytest.fixture
    def refund_entities(self) -> list[MagicMock]:
        """All five entities resolve as native/writable from metadata alone.

        Customer is metadata-writable but the ontology marks it read-only, so
        it must be pruned from the write schemas/description.
        """
        return [
            _make_entity("Customer", [_make_field("Name", is_required=True)]),
            _make_entity("RefundRequest", [_make_field("Amount", is_required=True)]),
            _make_entity("Order", [_make_field("Status")]),
            _make_entity("Risk", [_make_field("Score", sql_type_name="int")]),
            _make_entity("Contact", [_make_field("Resolution")]),
        ]

    @pytest.mark.asyncio
    async def test_read_only_customer_pruned_from_write_schemas(
        self, refund_entities: list[MagicMock]
    ) -> None:
        resolution = _make_resolution(refund_entities)
        # Inject the ontology via the fetch hook the handler looks for.
        resolution.entities_service.get_ontology_file_async = AsyncMock(
            return_value=_REFUND_PRUNE_OWL
        )

        with patch("uipath.platform.UiPath") as mock_uipath_cls:
            mock_sdk = MagicMock()
            mock_sdk.entities.resolve_entity_set_async = AsyncMock(
                return_value=resolution
            )
            mock_uipath_cls.return_value = mock_sdk

            entity_items = [
                {"id": n, "name": n, "folderId": "f1"}
                for n in ["Customer", "RefundRequest", "Order", "Risk", "Contact"]
            ]
            resource = _make_context_resource(entity_items=entity_items)
            tools = create_datafabric_tools(resource, _mock_llm())
            write_handler = tools[1].coroutine

            await write_handler._ensure_initialized()

            schemas = write_handler._write_schemas
            assert "Customer" not in schemas  # read-only -> pruned
            assert set(schemas.keys()) == {"RefundRequest", "Order", "Risk", "Contact"}

            description = write_handler._write_tool_description
            assert "Customer" not in description
            assert "RefundRequest" in description
            assert "Order" in description
            assert "Risk" in description
            assert "Contact" in description


# ---------------------------------------------------------------------------
# 6. Read tool is unchanged (renumbered from 5)
# ---------------------------------------------------------------------------


class TestReadToolUnchanged:
    """Verify the read tool retains its original properties."""

    def test_read_tool_type_is_datafabric_sql(self) -> None:
        tools = create_datafabric_tools(_make_context_resource(), _mock_llm())
        read_tool = tools[0]
        assert read_tool.metadata is not None
        assert read_tool.metadata["tool_type"] == "datafabric_sql"

    def test_read_tool_args_schema_is_datafabric_query_input(self) -> None:
        tools = create_datafabric_tools(_make_context_resource(), _mock_llm())
        read_tool = tools[0]
        assert read_tool.args_schema is DataFabricQueryInput

    def test_read_tool_description_mentions_query(self) -> None:
        tools = create_datafabric_tools(_make_context_resource(), _mock_llm())
        read_tool = tools[0]
        assert "Query" in read_tool.description or "query" in read_tool.description


# ---------------------------------------------------------------------------
# 7. context_tool.py integration
# ---------------------------------------------------------------------------


class TestContextToolIntegration:
    """Verify create_context_tool returns a list for DATA_FABRIC_ENTITY_SET."""

    def test_create_context_tool_returns_list_for_entity_set(self) -> None:
        from uipath_langchain.agent.tools.context_tool import create_context_tool

        resource = _make_context_resource()
        mock_llm = _mock_llm()

        with patch(
            "uipath_langchain.agent.tools.datafabric_tool.create_datafabric_tools"
        ) as mock_create:
            mock_create.return_value = [MagicMock(), MagicMock()]
            result = create_context_tool(resource, llm=mock_llm)

            assert isinstance(result, list)
            assert len(result) == 2
            mock_create.assert_called_once()

    def test_create_context_tool_raises_without_llm(self) -> None:
        from uipath_langchain.agent.tools.context_tool import create_context_tool

        resource = _make_context_resource()

        with pytest.raises(ValueError, match="LLM"):
            create_context_tool(resource, llm=None)
