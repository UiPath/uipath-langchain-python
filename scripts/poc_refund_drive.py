"""Ontology Write POC — drive the real write tool against staging.

Loads the POC artifacts produced by ``poc_refund_setup.sh``, injects the
ontology via the same bridge the CLI uses (monkeypatching the not-yet-shipped
``EntitiesService.get_ontology_file_async``), builds the real Data Fabric
tools, then runs the refund flow by invoking the write tool handler directly —
the exact callable an agent's tool node calls. Proves: ontology compile +
inject -> write validation -> EntitiesService CRUD -> records persisted ->
verified by read-back.

This is the deterministic counterpart to ``run_agent_with_ontology.py`` (which
puts an LLM in the loop). Use this to confirm the writes actually land.

Prereq: UIPATH_* env vars set (see scripts/POC_README.md), and
        ``poc_refund_setup.sh`` already run.
Usage:  uv run python scripts/poc_refund_drive.py [OUT_DIR]   (default: ./poc_out)
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import uipath.platform.entities._entities_service as es_mod

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "./poc_out")
ONTOLOGY_TTL = (OUT / "refund_ontology.ttl").read_text()


async def _get_ontology_file_async(self, file_type, *args, **kwargs):  # noqa: ANN001
    """Bridge: return the POC ontology for the 'owl' file type.

    Stands in for the platform method that has not shipped yet, so the
    handler's own _maybe_compile_ontology picks it up naturally.
    """
    return ONTOLOGY_TTL if file_type == "owl" else None


es_mod.EntitiesService.get_ontology_file_async = _get_ontology_file_async

from uipath.agent.models.agent import (  # noqa: E402
    AgentContextResourceConfig,
    AgentContextType,
)
from uipath.platform.entities import DataFabricEntityItem  # noqa: E402

from uipath_langchain.agent.tools.datafabric_tool import (  # noqa: E402
    create_datafabric_tools,
)


def _load_ids() -> dict[str, str]:
    ids: dict[str, str] = {}
    for line in (OUT / "refund_ids.env").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            ids[k] = v
    return ids


class _NoLLM:
    """Stub LLM — only needed to build the read tool; never invoked here."""

    def bind_tools(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
        return self


async def main() -> int:
    ids = _load_ids()
    items = [
        DataFabricEntityItem.model_validate(d)
        for d in json.loads((OUT / "refund_entity_set.json").read_text())
    ]
    by_suffix = {it.name.rsplit("_", 1)[1]: it.name for it in items}

    resource = AgentContextResourceConfig(
        name="refund_context",
        description="Refund processing context",
        context_type=AgentContextType.DATA_FABRIC_ENTITY_SET,
        entity_set=items,
    )
    tools = create_datafabric_tools(resource, _NoLLM())  # type: ignore[arg-type]
    handler = next(t for t in tools if t.name.endswith("_write")).coroutine

    await handler._ensure_initialized()
    onto = handler._compiled_ontology
    print("=== ONTOLOGY STATUS ===")
    if onto and not onto.is_empty():
        print(
            "  ACTIVE — entity_access:",
            {k: sorted(v) for k, v in onto.entity_access.items()},
        )
    else:
        print("  INACTIVE (metadata-only)")

    # SOP decision is fixed for the seeded scenario (Delivered, score 2, $200).
    amt, score, ltv = 200.0, 2, 5000.0
    print(
        f"\nDECIDE: order Delivered, risk {score} < 3, amount {amt} <= 500 -> APPROVE\n"
    )

    async def write(**kw):  # noqa: ANN003
        out = json.loads(await handler(**kw))
        print(
            f"  {kw['operation']:6} {kw['entity_key'].rsplit('_', 1)[1]:14} -> success={out['success']}"
        )
        return out

    print("=== WRITES (real handler, ontology validating) ===")
    await write(
        entity_key=by_suffix["RefundRequest"],
        operation="insert",
        fields={
            "ApprovedAmount": amt,
            "Reason": "Auto-approved: low risk",
            "OrderRef": ids["ORDER_ID"],
            "CustomerRef": ids["CUSTOMER_ID"],
            "RefundStatus": "Pending",
        },
    )
    await write(
        entity_key=by_suffix["PurchaseOrder"],
        operation="update",
        record_id=ids["ORDER_ID"],
        fields={"OrderStatus": "Returned"},
    )
    await write(
        entity_key=by_suffix["CustomerRisk"],
        operation="update",
        record_id=ids["RISK_ID"],
        fields={"RiskScore": score + 1, "LifetimeValue": ltv - amt},
    )
    await write(
        entity_key=by_suffix["Contact"],
        operation="update",
        record_id=ids["CONTACT_ID"],
        fields={"Resolution": "Approved"},
    )

    # Verify by read-back through the resolved service.
    from uipath.platform import UiPath

    svc = (await UiPath().entities.resolve_entity_set_async(items)).entities_service

    def g(rec, k):  # noqa: ANN001
        return rec.get(k) if isinstance(rec, dict) else getattr(rec, k, None)

    order = await svc.get_record_async(by_suffix["PurchaseOrder"], ids["ORDER_ID"])
    risk = await svc.get_record_async(by_suffix["CustomerRisk"], ids["RISK_ID"])
    contact = await svc.get_record_async(by_suffix["Contact"], ids["CONTACT_ID"])

    print("\n=== VERIFY (read-back) ===")
    checks = [
        ("Order.OrderStatus", g(order, "OrderStatus"), "Returned"),
        ("Risk.RiskScore", int(g(risk, "RiskScore")), 3),
        ("Risk.LifetimeValue", float(g(risk, "LifetimeValue")), 4800.0),
        ("Contact.Resolution", g(contact, "Resolution"), "Approved"),
    ]
    ok = 0
    for label, actual, expected in checks:
        good = actual == expected
        ok += good
        print(f"  {'OK ' if good else 'XX '} {label} = {actual} (expected {expected})")
    print(f"\n=== {ok}/{len(checks)} verified ===")
    return 0 if ok == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
