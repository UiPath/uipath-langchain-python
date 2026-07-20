"""Tests for EoG state types and reducers."""

from __future__ import annotations

import operator

from uipath_langchain.agent.eog.types import (
    Belief,
    EoGState,
    ExplanatoryEdge,
    FunctionSpec,
    InvestigationConfig,
    LedgerEntry,
)
from uipath_langchain.agent.react.reducers import merge_dicts


class TestBelief:
    def test_creation_defaults(self) -> None:
        b = Belief(label="Source")
        assert b.label == "Source"
        assert b.evidence == ""
        assert b.flip_count == 0

    def test_creation_with_values(self) -> None:
        b = Belief(label="PolicyViolation", evidence="found in audit", flip_count=2)
        assert b.label == "PolicyViolation"
        assert b.evidence == "found in audit"
        assert b.flip_count == 2

    def test_serialization_roundtrip(self) -> None:
        b = Belief(label="Source", evidence="test")
        data = b.model_dump()
        assert data == {"label": "Source", "evidence": "test", "flip_count": 0}
        restored = Belief.model_validate(data)
        assert restored == b


class TestFunctionSpec:
    def test_required_params(self) -> None:
        fn = FunctionSpec(
            name="exceptionContext",
            params=[
                {"name": "exceptionId", "type": "xsd:string", "required": True},
            ],
            outputs=[
                {"name": "invoiceId", "type": "xsd:string"},
                {"name": "poAmount", "type": "xsd:decimal"},
            ],
            touches=["ToleranceException", "Invoice", "PurchaseOrder"],
        )
        assert fn.required_params == ["exceptionId"]
        assert fn.param_names == {"exceptionId"}
        assert fn.output_names == {"invoiceId", "poAmount"}

    def test_no_params(self) -> None:
        fn = FunctionSpec(name="openExceptions")
        assert fn.required_params == []
        assert fn.param_names == set()
        assert fn.output_names == set()

    def test_touches_list(self) -> None:
        fn = FunctionSpec(
            name="invoiceDetail",
            touches=["Invoice", "Supplier", "PurchaseOrder"],
        )
        assert "Invoice" in fn.touches
        assert len(fn.touches) == 3


class TestLedgerEntry:
    def test_creation(self) -> None:
        entry = LedgerEntry(
            timestamp=1000.0,
            entity_id="e1",
            old_label=None,
            new_label="Source",
            evidence="initial",
        )
        assert entry.entity_id == "e1"
        assert entry.old_label is None
        assert entry.function_used is None


class TestExplanatoryEdge:
    def test_creation(self) -> None:
        edge = ExplanatoryEdge(source="a", target="b", relationship="causes")
        assert edge.evidence == ""


class TestInvestigationConfig:
    def test_defaults(self) -> None:
        cfg = InvestigationConfig(label_vocabulary=["Source", "Defer"])
        assert cfg.seed_entities == []
        assert cfg.max_steps == 50
        assert cfg.max_flips == 3
        assert cfg.default_label == "Defer"
        assert cfg.max_results_per_function == 50

    def test_custom_values(self) -> None:
        cfg = InvestigationConfig(
            label_vocabulary=["A", "B"],
            seed_entities=["e1", "e2"],
            max_steps=10,
            max_flips=1,
            default_label="Unknown",
            max_results_per_function=5,
        )
        assert cfg.max_steps == 10
        assert len(cfg.seed_entities) == 2


class TestEoGState:
    def test_default_state(self) -> None:
        state = EoGState()
        assert state.active_set == []
        assert state.current_entity == ""
        assert state.beliefs == {}
        assert state.ledger == []
        assert state.steps_taken == 0
        assert state.investigation_config is None
        assert state.function_cache == {}
        assert state.discovered_entities == {}

    def test_beliefs_merge_dicts_reducer(self) -> None:
        left = {"e1": Belief(label="A")}
        right = {"e2": Belief(label="B")}
        merged = merge_dicts(left, right)
        assert "e1" in merged
        assert "e2" in merged

    def test_beliefs_merge_dicts_override(self) -> None:
        left = {"e1": Belief(label="A")}
        right = {"e1": Belief(label="B")}
        merged = merge_dicts(left, right)
        assert merged["e1"].label == "B"

    def test_ledger_operator_add(self) -> None:
        entry1 = LedgerEntry(
            timestamp=1.0, entity_id="e1", old_label=None,
            new_label="A", evidence="x",
        )
        entry2 = LedgerEntry(
            timestamp=2.0, entity_id="e2", old_label=None,
            new_label="B", evidence="y",
        )
        result = operator.add([entry1], [entry2])
        assert len(result) == 2

    def test_state_with_function_cache(self) -> None:
        state = EoGState(
            function_cache={
                "Invoice": [{"name": "invoiceDetail", "touches": ["Invoice"]}],
            },
            discovered_entities={"INV-2004": "Invoice"},
        )
        assert "Invoice" in state.function_cache
        assert state.discovered_entities["INV-2004"] == "Invoice"

    def test_state_with_config(self) -> None:
        cfg = InvestigationConfig(label_vocabulary=["X"])
        state = EoGState(investigation_config=cfg)
        assert state.investigation_config is not None
        assert state.investigation_config.label_vocabulary == ["X"]
