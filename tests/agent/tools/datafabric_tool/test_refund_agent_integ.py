"""Integration test: simulates the refund agent's tool-calling pattern.

Proves the AGENT plane works for the Contact Center Refund Agent hero case.
The LLM reads entities via ``query_datafabric`` and writes via
``query_datafabric_write``.  This test mocks the platform layer and
invokes the real write handler directly with structured args, verifying
that the correct EntitiesService methods are called with the correct
arguments.

Hero case entities (from RFC p1-write-rfc-v2-ontology.md):
  - Customer  (federated / read-only)
  - Contact   (native / read + write)
  - Order     (native / read + write)
  - CustomerRisk (native / read + write)
  - RefundRequest (native / write - insert)
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from uipath_langchain.agent.tools.datafabric_tool import create_datafabric_tools

# ---------------------------------------------------------------------------
# Helpers (same patterns as test_write_integration.py / test_write_validation.py)
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
    display_name: str | None = None,
) -> MagicMock:
    """Create a mock Entity object."""
    entity = MagicMock()
    entity.name = name
    # id == name here so the handler's name->id translation is an identity.
    entity.id = name
    entity.display_name = display_name or name
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


def _make_record(record_id: str = "rec-new", data: dict | None = None) -> MagicMock:
    """Create a mock record returned by insert/update."""
    record = MagicMock()
    record.id = record_id
    record.model_dump.return_value = {"Id": record_id, **(data or {})}
    return record


def _mock_llm() -> MagicMock:
    return MagicMock()


# ---------------------------------------------------------------------------
# Hero-case entity factory
# ---------------------------------------------------------------------------


def _make_refund_entities() -> list[MagicMock]:
    """Build the 5 entities from the refund agent hero case.

    - Customer: federated (external_fields non-empty), read-only
    - Contact: native, writable (ContactReason, RequestedRefundAmount,
      OrderId, ResolutionStatus)
    - Order: native, writable (OrderNumber, TotalAmount, Status)
    - CustomerRisk: native, writable (RiskScore, LifetimeValue, FraudFlag)
    - RefundRequest: native, writable (ApprovedAmount[required],
      Reason[required], OrderId, CustomerId, Status)
    """
    customer = _make_entity(
        "Customer",
        [
            _make_field("Id", is_primary_key=True),
            _make_field("CreatedOn", is_system_field=True),
            _make_field("Name", description="Customer full name"),
            _make_field("AccountTier", description="Gold/Silver/Bronze"),
        ],
        entity_type="Entity",
        external_fields=[{"source": "salesforce"}],  # federated
        display_name="Customer",
    )

    contact = _make_entity(
        "Contact",
        [
            _make_field("Id", is_primary_key=True),
            _make_field("CreatedOn", is_system_field=True),
            _make_field(
                "ContactReason",
                choiceset_id="ContactReasonCS",
                description="Reason for contact",
            ),
            _make_field(
                "RequestedRefundAmount",
                sql_type_name="decimal",
                description="Requested refund amount",
            ),
            _make_field("OrderId", description="References Order"),
            _make_field(
                "ResolutionStatus",
                choiceset_id="ResolutionStatusCS",
                description="Open, Approved, Denied, Escalated",
            ),
        ],
        display_name="Contact",
    )

    order = _make_entity(
        "Order",
        [
            _make_field("Id", is_primary_key=True),
            _make_field("CreatedOn", is_system_field=True),
            _make_field("OrderNumber", is_required=True),
            _make_field("TotalAmount", sql_type_name="decimal"),
            _make_field(
                "Status",
                choiceset_id="OrderStatusCS",
                description="Placed, Shipped, Delivered, Returned, Cancelled",
            ),
        ],
        display_name="Order",
    )

    customer_risk = _make_entity(
        "CustomerRisk",
        [
            _make_field("Id", is_primary_key=True),
            _make_field("CreatedOn", is_system_field=True),
            _make_field(
                "RiskScore",
                sql_type_name="int",
                description="Numeric risk score, additive",
            ),
            _make_field(
                "LifetimeValue",
                sql_type_name="decimal",
                description="Customer lifetime value",
            ),
            _make_field("FraudFlag", sql_type_name="bit", description="Fraud flag"),
        ],
        display_name="Customer Risk",
    )

    refund_request = _make_entity(
        "RefundRequest",
        [
            _make_field("Id", is_primary_key=True),
            _make_field("CreatedOn", is_system_field=True),
            _make_field(
                "ApprovedAmount",
                sql_type_name="decimal",
                is_required=True,
                description="Approved refund amount",
            ),
            _make_field(
                "Reason",
                is_required=True,
                description="Reason for refund",
            ),
            _make_field("OrderId", description="References Order"),
            _make_field("CustomerId", description="References Customer"),
            _make_field(
                "Status",
                choiceset_id="RefundStatusCS",
                description="Pending, Processed, Failed",
            ),
        ],
        display_name="Refund Request",
    )

    return [customer, contact, order, customer_risk, refund_request]


def _make_refund_context_resource():
    """Build an AgentContextResourceConfig for the refund hero case."""
    from uipath.agent.models.agent import (
        AgentContextResourceConfig,
        AgentContextType,
    )
    from uipath.platform.entities import DataFabricEntityItem

    items = [
        {
            "id": "e-cust",
            "name": "Customer",
            "folderId": "f1",
            "description": "Customer from CRM",
        },
        {
            "id": "e-contact",
            "name": "Contact",
            "folderId": "f1",
            "description": "Inbound contact",
        },
        {
            "id": "e-order",
            "name": "Order",
            "folderId": "f1",
            "description": "Order records",
        },
        {
            "id": "e-risk",
            "name": "CustomerRisk",
            "folderId": "f1",
            "description": "Risk scoring",
        },
        {
            "id": "e-refund",
            "name": "RefundRequest",
            "folderId": "f1",
            "description": "Refund records",
        },
    ]
    entity_set = [DataFabricEntityItem.model_validate(item) for item in items]
    return AgentContextResourceConfig(
        name="refund_data",
        description="Refund agent data fabric",
        resource_type="context",
        context_type=AgentContextType.DATA_FABRIC_ENTITY_SET,
        entity_set=entity_set,
        is_enabled=True,
    )


# ---------------------------------------------------------------------------
# Shared fixture: patched SDK returning refund entities
# ---------------------------------------------------------------------------


@pytest.fixture
def refund_resolution():
    """Resolution with all 5 refund hero case entities."""
    entities = _make_refund_entities()
    return _make_resolution(entities)


@pytest.fixture
def refund_tools(refund_resolution):
    """Create read + write tools with mocked SDK resolution."""
    with patch("uipath.platform.UiPath") as mock_uipath_cls:
        mock_sdk = MagicMock()
        mock_sdk.entities.resolve_entity_set_async = AsyncMock(
            return_value=refund_resolution
        )
        mock_uipath_cls.return_value = mock_sdk
        resource = _make_refund_context_resource()
        tools = create_datafabric_tools(resource, _mock_llm())
        # Yield both the tools and the resolution so tests can set up
        # EntitiesService mocks and verify calls.
        yield tools, refund_resolution


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRefundAgentFlow:
    """Integration test: simulates the refund agent's tool-calling pattern."""

    # -- Tool creation --

    @pytest.mark.asyncio
    async def test_write_tool_created_for_writable_entities(self, refund_tools) -> None:
        """create_datafabric_tools returns read + write tools.
        Write tool excludes federated Customer from write schemas.
        """
        tools, resolution = refund_tools
        assert len(tools) == 2

        write_tool = tools[1]
        assert write_tool.metadata is not None
        assert write_tool.metadata["tool_type"] == "datafabric_write"

        # Invoke once to trigger lazy init (with a simple validation-error call)
        resolution.entities_service.insert_record_async = AsyncMock()
        result_str = await write_tool.ainvoke(
            {
                "entity_key": "Customer",
                "operation": "insert",
                "fields": {"Name": "Test"},
            }
        )
        result = json.loads(result_str)
        # Customer is federated -> not configured for writes
        assert result["success"] is False
        assert any("not configured for writes" in e for e in result["errors"])

    @pytest.mark.asyncio
    async def test_write_tool_description_contains_writable_entities(
        self, refund_tools
    ) -> None:
        """After lazy init, handler has write schemas for Contact, Order,
        CustomerRisk, RefundRequest but NOT Customer.
        """
        tools, resolution = refund_tools
        write_tool = tools[1]

        # Trigger lazy init
        resolution.entities_service.insert_record_async = AsyncMock()
        await write_tool.ainvoke(
            {
                "entity_key": "__trigger_init__",
                "operation": "insert",
                "fields": {"x": 1},
            }
        )

        # Access the handler's resolved write_schemas
        handler = write_tool.coroutine
        assert handler._write_schemas is not None
        writable_keys = set(handler._write_schemas.keys())
        assert "Contact" in writable_keys
        assert "Order" in writable_keys
        assert "CustomerRisk" in writable_keys
        assert "RefundRequest" in writable_keys
        assert "Customer" not in writable_keys

    # -- Individual write operations --

    @pytest.mark.asyncio
    async def test_insert_refund_request(self, refund_tools) -> None:
        """Agent creates a RefundRequest record with correct fields."""
        tools, resolution = refund_tools
        write_tool = tools[1]

        insert_record = _make_record(
            "refund-001",
            {
                "ApprovedAmount": 200.00,
                "Reason": "Auto-approved: risk score below threshold",
                "OrderId": "order-uuid",
                "CustomerId": "customer-uuid",
                "Status": "Pending",
            },
        )
        resolution.entities_service.insert_record_async = AsyncMock(
            return_value=insert_record
        )

        result_str = await write_tool.ainvoke(
            {
                "entity_key": "RefundRequest",
                "operation": "insert",
                "fields": {
                    "ApprovedAmount": 200.00,
                    "Reason": "Auto-approved: risk score below threshold",
                    "OrderId": "order-uuid",
                    "CustomerId": "customer-uuid",
                    "Status": "Pending",
                },
            }
        )
        result = json.loads(result_str)

        assert result["success"] is True
        assert result["operation"] == "insert"
        assert result["entity_key"] == "RefundRequest"
        assert result["record_id"] == "refund-001"

        resolution.entities_service.insert_record_async.assert_awaited_once_with(
            "RefundRequest",
            {
                "ApprovedAmount": 200.00,
                "Reason": "Auto-approved: risk score below threshold",
                "OrderId": "order-uuid",
                "CustomerId": "customer-uuid",
                "Status": "Pending",
            },
        )

    @pytest.mark.asyncio
    async def test_update_order_status(self, refund_tools) -> None:
        """Agent updates Order status to Returned."""
        tools, resolution = refund_tools
        write_tool = tools[1]

        update_record = _make_record("order-uuid", {"Status": "Returned"})
        resolution.entities_service.update_record_async = AsyncMock(
            return_value=update_record
        )

        result_str = await write_tool.ainvoke(
            {
                "entity_key": "Order",
                "operation": "update",
                "record_id": "order-uuid",
                "fields": {"Status": 3},  # ChoiceSet NumberId for "Returned"
            }
        )
        result = json.loads(result_str)

        assert result["success"] is True
        assert result["operation"] == "update"
        assert result["entity_key"] == "Order"

        resolution.entities_service.update_record_async.assert_awaited_once_with(
            "Order", "order-uuid", {"Status": 3}
        )

    @pytest.mark.asyncio
    async def test_update_customer_risk(self, refund_tools) -> None:
        """Agent updates CustomerRisk: increment score, decrement LTV."""
        tools, resolution = refund_tools
        write_tool = tools[1]

        update_record = _make_record(
            "risk-uuid",
            {
                "RiskScore": 3,
                "LifetimeValue": 4800.00,
            },
        )
        resolution.entities_service.update_record_async = AsyncMock(
            return_value=update_record
        )

        # Agent read current values (RiskScore=2, LTV=5000) and computed new ones
        result_str = await write_tool.ainvoke(
            {
                "entity_key": "CustomerRisk",
                "operation": "update",
                "record_id": "risk-uuid",
                "fields": {"RiskScore": 3, "LifetimeValue": 4800.00},
            }
        )
        result = json.loads(result_str)

        assert result["success"] is True
        assert result["operation"] == "update"

        resolution.entities_service.update_record_async.assert_awaited_once_with(
            "CustomerRisk",
            "risk-uuid",
            {"RiskScore": 3, "LifetimeValue": 4800.00},
        )

    @pytest.mark.asyncio
    async def test_update_contact_resolution(self, refund_tools) -> None:
        """Agent updates Contact resolution to Approved."""
        tools, resolution = refund_tools
        write_tool = tools[1]

        update_record = _make_record(
            "contact-uuid",
            {
                "ResolutionStatus": "Approved",
            },
        )
        resolution.entities_service.update_record_async = AsyncMock(
            return_value=update_record
        )

        result_str = await write_tool.ainvoke(
            {
                "entity_key": "Contact",
                "operation": "update",
                "record_id": "contact-uuid",
                "fields": {"ResolutionStatus": "Approved"},
            }
        )
        result = json.loads(result_str)

        assert result["success"] is True
        assert result["operation"] == "update"

        resolution.entities_service.update_record_async.assert_awaited_once_with(
            "Contact",
            "contact-uuid",
            {"ResolutionStatus": "Approved"},
        )

    # -- Federated entity rejection --

    @pytest.mark.asyncio
    async def test_write_to_federated_entity_rejected(self, refund_tools) -> None:
        """Writing to Customer (federated) returns validation error."""
        tools, resolution = refund_tools
        write_tool = tools[1]

        resolution.entities_service.insert_record_async = AsyncMock()

        result_str = await write_tool.ainvoke(
            {
                "entity_key": "Customer",
                "operation": "insert",
                "fields": {"Name": "New Customer", "AccountTier": "Gold"},
            }
        )
        result = json.loads(result_str)

        assert result["success"] is False
        assert any("not configured for writes" in e for e in result["errors"])
        # Writable entities should be listed in the error
        assert any("Contact" in e for e in result["errors"])
        assert any("Order" in e for e in result["errors"])
        assert any("CustomerRisk" in e for e in result["errors"])
        assert any("RefundRequest" in e for e in result["errors"])
        # API must NOT be called
        resolution.entities_service.insert_record_async.assert_not_awaited()

    # -- Full refund flow --

    @pytest.mark.asyncio
    async def test_full_refund_flow(self, refund_tools) -> None:
        """End-to-end: all 4 writes in sequence, all succeed, all verified.

        Simulates the complete agent SOP:
        1. Insert RefundRequest
        2. Update Order status to Returned
        3. Update CustomerRisk (score + LTV)
        4. Update Contact resolution to Approved
        """
        tools, resolution = refund_tools
        write_tool = tools[1]

        # Set up mocks for all 4 operations
        refund_record = _make_record(
            "refund-001",
            {
                "ApprovedAmount": 200.00,
                "Reason": "Auto-approved: risk score below threshold",
            },
        )
        order_record = _make_record("order-uuid", {"Status": "Returned"})
        risk_record = _make_record(
            "risk-uuid",
            {
                "RiskScore": 3,
                "LifetimeValue": 4800.00,
            },
        )
        contact_record = _make_record(
            "contact-uuid",
            {
                "ResolutionStatus": "Approved",
            },
        )

        resolution.entities_service.insert_record_async = AsyncMock(
            return_value=refund_record
        )
        resolution.entities_service.update_record_async = AsyncMock(
            side_effect=[order_record, risk_record, contact_record]
        )

        # Step 1: Insert RefundRequest
        r1 = json.loads(
            await write_tool.ainvoke(
                {
                    "entity_key": "RefundRequest",
                    "operation": "insert",
                    "fields": {
                        "ApprovedAmount": 200.00,
                        "Reason": "Auto-approved: risk score below threshold",
                        "OrderId": "order-uuid",
                        "CustomerId": "customer-uuid",
                        "Status": "Pending",
                    },
                }
            )
        )
        assert r1["success"] is True
        assert r1["operation"] == "insert"
        assert r1["entity_key"] == "RefundRequest"

        # Step 2: Update Order -> Returned
        r2 = json.loads(
            await write_tool.ainvoke(
                {
                    "entity_key": "Order",
                    "operation": "update",
                    "record_id": "order-uuid",
                    "fields": {"Status": 3},
                }
            )
        )
        assert r2["success"] is True
        assert r2["operation"] == "update"
        assert r2["entity_key"] == "Order"

        # Step 3: Update CustomerRisk
        r3 = json.loads(
            await write_tool.ainvoke(
                {
                    "entity_key": "CustomerRisk",
                    "operation": "update",
                    "record_id": "risk-uuid",
                    "fields": {"RiskScore": 3, "LifetimeValue": 4800.00},
                }
            )
        )
        assert r3["success"] is True
        assert r3["operation"] == "update"
        assert r3["entity_key"] == "CustomerRisk"

        # Step 4: Update Contact -> Approved
        r4 = json.loads(
            await write_tool.ainvoke(
                {
                    "entity_key": "Contact",
                    "operation": "update",
                    "record_id": "contact-uuid",
                    "fields": {"ResolutionStatus": "Approved"},
                }
            )
        )
        assert r4["success"] is True
        assert r4["operation"] == "update"
        assert r4["entity_key"] == "Contact"

        # Verify all 4 calls were made with correct args
        resolution.entities_service.insert_record_async.assert_awaited_once_with(
            "RefundRequest",
            {
                "ApprovedAmount": 200.00,
                "Reason": "Auto-approved: risk score below threshold",
                "OrderId": "order-uuid",
                "CustomerId": "customer-uuid",
                "Status": "Pending",
            },
        )
        assert resolution.entities_service.update_record_async.await_count == 3
        update_calls = resolution.entities_service.update_record_async.await_args_list
        # Call 0: Order
        assert update_calls[0].args == ("Order", "order-uuid", {"Status": 3})
        # Call 1: CustomerRisk
        assert update_calls[1].args == (
            "CustomerRisk",
            "risk-uuid",
            {"RiskScore": 3, "LifetimeValue": 4800.00},
        )
        # Call 2: Contact
        assert update_calls[2].args == (
            "Contact",
            "contact-uuid",
            {"ResolutionStatus": "Approved"},
        )

    # -- Validation edge cases --

    @pytest.mark.asyncio
    async def test_insert_missing_required_field_rejected(self, refund_tools) -> None:
        """Insert RefundRequest without ApprovedAmount returns validation error."""
        tools, resolution = refund_tools
        write_tool = tools[1]

        resolution.entities_service.insert_record_async = AsyncMock()

        # Missing ApprovedAmount (required) and Reason (required)
        result_str = await write_tool.ainvoke(
            {
                "entity_key": "RefundRequest",
                "operation": "insert",
                "fields": {
                    "OrderId": "order-uuid",
                    "CustomerId": "customer-uuid",
                    "Status": "Pending",
                },
            }
        )
        result = json.loads(result_str)

        assert result["success"] is False
        assert any("ApprovedAmount" in e for e in result["errors"])
        assert any("Reason" in e for e in result["errors"])
        resolution.entities_service.insert_record_async.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_delete_requires_record_id(self, refund_tools) -> None:
        """Delete without record_id returns validation error."""
        tools, resolution = refund_tools
        write_tool = tools[1]

        resolution.entities_service.delete_record_async = AsyncMock()

        result_str = await write_tool.ainvoke(
            {
                "entity_key": "Order",
                "operation": "delete",
            }
        )
        result = json.loads(result_str)

        assert result["success"] is False
        assert any("record_id" in e for e in result["errors"])
        resolution.entities_service.delete_record_async.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_update_unknown_field_rejected(self, refund_tools) -> None:
        """Update with a field not in the entity schema returns validation error."""
        tools, resolution = refund_tools
        write_tool = tools[1]

        resolution.entities_service.update_record_async = AsyncMock()

        result_str = await write_tool.ainvoke(
            {
                "entity_key": "Order",
                "operation": "update",
                "record_id": "order-uuid",
                "fields": {"BogusField": "value"},
            }
        )
        result = json.loads(result_str)

        assert result["success"] is False
        assert any("BogusField" in e for e in result["errors"])
        resolution.entities_service.update_record_async.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_insert_with_system_field_rejected(self, refund_tools) -> None:
        """Insert with system field (CreatedOn) returns validation error."""
        tools, resolution = refund_tools
        write_tool = tools[1]

        resolution.entities_service.insert_record_async = AsyncMock()

        result_str = await write_tool.ainvoke(
            {
                "entity_key": "RefundRequest",
                "operation": "insert",
                "fields": {
                    "ApprovedAmount": 100.00,
                    "Reason": "Test",
                    "CreatedOn": "2026-01-01",
                },
            }
        )
        result = json.loads(result_str)

        assert result["success"] is False
        assert any("CreatedOn" in e for e in result["errors"])
        resolution.entities_service.insert_record_async.assert_not_awaited()
