"""Tests for EoG node functions."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

from uipath_langchain.agent.eog.nodes import (
    _bind_primary,
    _bind_secondary,
    _make_discover,
    _make_gather,
    _make_label,
    _make_seed,
    _type_from_param_name,
    pop_node,
    propagate_node,
    resolve_entity_type,
    should_continue,
    synthesize_node,
    update_node,
)
from uipath_langchain.agent.eog.types import (
    Belief,
    EoGState,
    FunctionSpec,
    InvestigationConfig,
    LedgerEntry,
)


# ── Test fixtures ────────────────────────────────────────────────

_TEST_FUNCTIONS = [
    {
        "name": "exceptionContext",
        "label": "Exception context",
        "description": "Full context for a tolerance exception",
        "params": [{"name": "exceptionId", "type": "xsd:string", "required": True}],
        "outputs": [
            {"name": "invoiceId", "type": "xsd:string"},
            {"name": "poAmount", "type": "xsd:decimal"},
            {"name": "supplierName", "type": "xsd:string"},
        ],
        "touches": ["ToleranceException", "Invoice", "PurchaseOrder", "Supplier"],
    },
    {
        "name": "openExceptions",
        "label": "Open tolerance exceptions",
        "description": "List all open exceptions",
        "params": [],
        "outputs": [{"name": "exceptionId", "type": "xsd:string"}],
        "touches": ["ToleranceException", "Invoice"],
    },
    {
        "name": "matchingExceptionRules",
        "label": "Matching exception rules",
        "description": "Find auto-resolution rules for a commodity",
        "params": [{"name": "commodityId", "type": "xsd:string", "required": True}],
        "outputs": [{"name": "ruleId", "type": "xsd:string"}],
        "touches": ["ExceptionRule", "Supplier"],
    },
]

_EXCEPTION_RESULT = {
    "rows": [{
        "exceptionType": "price_variance",
        "actualPct": 4.0,
        "invoiceId": "INV-2004",
        "poId": "PO-1004",
        "supplierId": "SUP-004",
        "commodityId": "COM-113",
    }],
}


def _make_config(**overrides: Any) -> InvestigationConfig:
    defaults = {
        "label_vocabulary": ["Source", "DerivedEffect", "PolicyViolation", "Defer"],
        "seed_records": ["EXC-002"],
        "max_steps": 50,
    }
    defaults.update(overrides)
    return InvestigationConfig(**defaults)


# ── resolve_entity_type ──────────────────────────────────────────

class TestResolveEntityType:
    def test_known_prefix(self) -> None:
        assert resolve_entity_type("EXC-002") == "ToleranceException"
        assert resolve_entity_type("INV-2004") == "Invoice"
        assert resolve_entity_type("SUP-001") == "Supplier"
        assert resolve_entity_type("PO-1004") == "PurchaseOrder"

    def test_unknown_prefix(self) -> None:
        assert resolve_entity_type("UNKNOWN-1") is None


# ── _type_from_param_name ────────────────────────────────────────

class TestTypeFromParamName:
    def test_known_params(self) -> None:
        assert _type_from_param_name("invoiceId") == "Invoice"
        assert _type_from_param_name("supplierId") == "Supplier"
        assert _type_from_param_name("exceptionId") == "ToleranceException"
        assert _type_from_param_name("commodityId") == "Commodity"

    def test_unknown_param(self) -> None:
        assert _type_from_param_name("randomField") is None


# ── _bind_primary ────────────────────────────────────────────────

class TestBindPrimary:
    def test_binds_matching_key(self) -> None:
        fn = FunctionSpec(
            name="exceptionContext",
            params=[{"name": "exceptionId", "type": "xsd:string", "required": True}],
        )
        result = _bind_primary(fn, "EXC-002", "ToleranceException")
        assert result == {"exceptionId": "EXC-002"}

    def test_returns_none_for_unmatched_required(self) -> None:
        fn = FunctionSpec(
            name="matchingExceptionRules",
            params=[{"name": "commodityId", "type": "xsd:string", "required": True}],
        )
        result = _bind_primary(fn, "EXC-002", "ToleranceException")
        assert result is None

    def test_no_params(self) -> None:
        fn = FunctionSpec(name="openExceptions", params=[])
        result = _bind_primary(fn, "EXC-002", "ToleranceException")
        assert result is None


# ── _bind_secondary ──────────────────────────────────────────────

class TestBindSecondary:
    def test_binds_from_discovered_values(self) -> None:
        fn = FunctionSpec(
            name="matchingExceptionRules",
            params=[{"name": "commodityId", "type": "xsd:string", "required": True}],
        )
        discovered = {"commodityId": "COM-113"}
        result = _bind_secondary(fn, discovered)
        assert result == {"commodityId": "COM-113"}

    def test_returns_none_when_missing_required(self) -> None:
        fn = FunctionSpec(
            name="matchingExceptionRules",
            params=[{"name": "commodityId", "type": "xsd:string", "required": True}],
        )
        result = _bind_secondary(fn, {})
        assert result is None


# ── Seed ──────────────────────────────────────────────────────────

class TestSeed:
    @pytest.mark.asyncio
    async def test_seed_creates_beliefs(self) -> None:
        cfg = _make_config()
        seed = _make_seed(cfg)
        result = await seed(EoGState())

        assert "EXC-002" in result["beliefs"]
        assert result["beliefs"]["EXC-002"].label == "Defer"
        assert result["active_set"] == ["EXC-002"]
        assert result["discovered_records"]["EXC-002"] == "ToleranceException"

    @pytest.mark.asyncio
    async def test_seed_multiple_entities(self) -> None:
        cfg = _make_config(seed_records=["EXC-001", "EXC-002"])
        seed = _make_seed(cfg)
        result = await seed(EoGState())

        assert len(result["beliefs"]) == 2
        assert len(result["active_set"]) == 2


# ── Pop ───────────────────────────────────────────────────────────

class TestPop:
    @pytest.mark.asyncio
    async def test_pop_dequeues_fifo(self) -> None:
        state = EoGState(active_set=["a", "b", "c"])
        result = await pop_node(state)
        assert result["current_record"] == "a"
        assert result["active_set"] == ["b", "c"]

    @pytest.mark.asyncio
    async def test_pop_empty(self) -> None:
        state = EoGState(active_set=[])
        result = await pop_node(state)
        assert result["current_record"] == ""


# ── should_continue ──────────────────────────────────────────────

class TestShouldContinue:
    def test_continue_when_entity_and_budget(self) -> None:
        state = EoGState(
            current_record="EXC-002",
            steps_taken=0,
            investigation_config=_make_config(),
        )
        assert should_continue(state) == "discover"

    def test_synthesize_when_empty(self) -> None:
        state = EoGState(
            current_record="",
            investigation_config=_make_config(),
        )
        assert should_continue(state) == "synthesize"

    def test_synthesize_when_budget_exhausted(self) -> None:
        state = EoGState(
            current_record="EXC-002",
            steps_taken=50,
            investigation_config=_make_config(max_steps=50),
        )
        assert should_continue(state) == "synthesize"


# ── Discover ──────────────────────────────────────────────────────

class TestDiscover:
    @pytest.mark.asyncio
    async def test_discover_fetches_functions(self) -> None:
        mock_client = AsyncMock()
        mock_client.list_functions.return_value = _TEST_FUNCTIONS

        discover = _make_discover(mock_client, "s2p")
        state = EoGState(
            current_record="EXC-002",
            discovered_records={"EXC-002": "ToleranceException"},
        )
        result = await discover(state)

        mock_client.list_functions.assert_called_once_with(
            "s2p", touches="ToleranceException",
        )
        assert "ToleranceException" in result["function_cache"]
        assert result["context_packet"]["entity_type"] == "ToleranceException"

    @pytest.mark.asyncio
    async def test_discover_uses_cache(self) -> None:
        mock_client = AsyncMock()

        discover = _make_discover(mock_client, "s2p")
        state = EoGState(
            current_record="EXC-002",
            discovered_records={"EXC-002": "ToleranceException"},
            function_cache={"ToleranceException": _TEST_FUNCTIONS},
        )
        result = await discover(state)

        mock_client.list_functions.assert_not_called()
        assert len(result["context_packet"]["functions"]) == 3


# ── Gather ────────────────────────────────────────────────────────

class TestGather:
    @pytest.mark.asyncio
    async def test_gather_invokes_and_discovers(self) -> None:
        mock_client = AsyncMock()
        mock_client.invoke_function.side_effect = [
            _EXCEPTION_RESULT,  # exceptionContext
            {"rows": [{"exceptionId": "EXC-001"}, {"exceptionId": "EXC-002"}]},  # openExceptions
            {"rows": []},  # matchingExceptionRules (secondary)
        ]

        gather = _make_gather(mock_client, "s2p")
        state = EoGState(
            current_record="EXC-002",
            discovered_records={"EXC-002": "ToleranceException"},
            investigation_config=_make_config(),
            context_packet={
                "entity_id": "EXC-002",
                "entity_type": "ToleranceException",
                "functions": _TEST_FUNCTIONS,
                "evidence": [],
            },
        )
        result = await gather(state)

        # Should have evidence from invoked functions
        evidence = result["context_packet"]["evidence"]
        assert len(evidence) >= 2

        # Should discover new entities from results
        discovered = result["discovered_records"]
        assert "INV-2004" in discovered
        assert discovered["INV-2004"] == "Invoice"

        # Should create Defer beliefs for discovered entities
        assert "INV-2004" in result["beliefs"]
        assert result["beliefs"]["INV-2004"].label == "Defer"


# ── Update ────────────────────────────────────────────────────────

class TestUpdate:
    @pytest.mark.asyncio
    async def test_update_writes_belief(self) -> None:
        state = EoGState(
            current_record="EXC-002",
            beliefs={"EXC-002": Belief(label="Defer")},
            policy_result={"label": "Source", "evidence": "price variance"},
            investigation_config=_make_config(),
        )
        result = await update_node(state)

        assert result["beliefs"]["EXC-002"].label == "Source"
        assert len(result["ledger"]) == 1
        assert result["ledger"][0].old_label == "Defer"
        assert result["ledger"][0].new_label == "Source"
        assert result["steps_taken"] == 1

    @pytest.mark.asyncio
    async def test_update_damping(self) -> None:
        state = EoGState(
            current_record="EXC-002",
            beliefs={"EXC-002": Belief(label="A", flip_count=4)},
            policy_result={"label": "B", "evidence": "oscillating"},
            investigation_config=_make_config(max_flips=3),
        )
        result = await update_node(state)
        assert result["beliefs"]["EXC-002"].label == "Defer"


# ── Propagate ────────────────────────────────────────────────────

class TestPropagate:
    @pytest.mark.asyncio
    async def test_propagate_on_change(self) -> None:
        state = EoGState(
            current_record="EXC-002",
            beliefs={
                "EXC-002": Belief(label="Source", evidence="test"),
                "INV-2004": Belief(label="Defer"),
            },
            discovered_records={
                "EXC-002": "ToleranceException",
                "INV-2004": "Invoice",
            },
            ledger=[
                LedgerEntry(
                    timestamp=1.0, entity_id="EXC-002",
                    old_label="Defer", new_label="Source", evidence="test",
                ),
            ],
            investigation_config=_make_config(),
            context_packet={
                "discovered_values": {"invoiceId": "INV-2004"},
            },
            active_set=[],
        )
        result = await propagate_node(state)

        assert "INV-2004" in result.get("inbox", {})
        assert "INV-2004" in result.get("active_set", [])
        assert len(result.get("explanatory_edges", [])) == 1

    @pytest.mark.asyncio
    async def test_no_propagate_when_no_change(self) -> None:
        state = EoGState(
            current_record="EXC-002",
            beliefs={"EXC-002": Belief(label="Defer")},
            ledger=[
                LedgerEntry(
                    timestamp=1.0, entity_id="EXC-002",
                    old_label="Defer", new_label="Defer", evidence="same",
                ),
            ],
            investigation_config=_make_config(),
            context_packet={"discovered_values": {}},
        )
        result = await propagate_node(state)
        assert result == {}


# ── Synthesize ────────────────────────────────────────────────────

class TestSynthesize:
    @pytest.mark.asyncio
    async def test_synthesize_finds_frontier(self) -> None:
        state = EoGState(
            beliefs={
                "EXC-002": Belief(label="Source", evidence="root cause"),
                "INV-2004": Belief(label="DerivedEffect", evidence="flagged"),
                "PO-1004": Belief(label="Defer"),
            },
            discovered_records={
                "EXC-002": "ToleranceException",
                "INV-2004": "Invoice",
                "PO-1004": "PurchaseOrder",
            },
            investigation_config=_make_config(),
            explanatory_edges=[],
        )
        result = await synthesize_node(state)

        frontier = result["frontier"]
        entity_ids = {f["entity"] for f in frontier}
        assert "EXC-002" in entity_ids
        assert "INV-2004" in entity_ids
        assert "PO-1004" not in entity_ids  # still Defer

    @pytest.mark.asyncio
    async def test_synthesize_irreducibility(self) -> None:
        from uipath_langchain.agent.eog.types import ExplanatoryEdge

        state = EoGState(
            beliefs={
                "EXC-002": Belief(label="Source"),
                "INV-2004": Belief(label="DerivedEffect"),
            },
            discovered_records={
                "EXC-002": "ToleranceException",
                "INV-2004": "Invoice",
            },
            investigation_config=_make_config(),
            explanatory_edges=[
                ExplanatoryEdge(
                    source="EXC-002", target="INV-2004",
                    relationship="invoiceId",
                ),
            ],
        )
        result = await synthesize_node(state)

        frontier = result["frontier"]
        entity_ids = {f["entity"] for f in frontier}
        assert "EXC-002" in entity_ids
        assert "INV-2004" not in entity_ids  # explained by EXC-002
