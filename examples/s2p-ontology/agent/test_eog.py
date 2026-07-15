"""Test script for the S2P EoG investigation agent.

Prerequisites:
  1. FQS mock running: cd ../fqs-data && ./fqs_start.sh
  2. Ontology-runtime running:
     java -jar ontology-app.jar --spring.profiles.active=local
  3. S2P ontology deployed: cd ../scripts && ./run-e2e.sh

Run:
  python test_eog.py
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any


async def main() -> None:
    """Run an end-to-end EoG investigation on the S2P ontology."""
    # Import the compiled graph and its config from the sibling module.
    from graph import graph, investigation_config
    from uipath_langchain.agent.eog import InvestigationConfig

    print("=" * 60)
    print("S2P EoG Investigation Agent -- Test Run")
    print("=" * 60)

    # Seed the investigation with entity types that have open exceptions.
    seed_entities = [
        "ToleranceException",
        "Invoice",
        "PurchaseOrder",
        "Supplier",
    ]

    config = InvestigationConfig(
        label_vocabulary=investigation_config.label_vocabulary,
        seed_entities=seed_entities,
        max_steps=investigation_config.max_steps,
        max_flips=investigation_config.max_flips,
        default_label=investigation_config.default_label,
        max_results_per_function=investigation_config.max_results_per_function,
    )

    print(f"\nSeeding investigation with: {seed_entities}")
    print(f"Label vocabulary: {config.label_vocabulary}")
    print(f"Max steps: {config.max_steps}")
    print("-" * 60)

    try:
        result: dict[str, Any] = await graph.ainvoke(
            {"investigation_config": config},
        )

        print("\n" + "=" * 60)
        print("INVESTIGATION COMPLETE")
        print("=" * 60)

        # Print ledger (step-by-step trace)
        if "ledger" in result:
            print(f"\nLedger ({len(result['ledger'])} entries):")
            for entry in result["ledger"]:
                old = entry.get("old_label", "--")
                new = entry.get("new_label", "?")
                entity = entry.get("entity_id", "?")
                evidence = entry.get("evidence", "")[:80]
                print(f"  [{entity}] {old} -> {new}: {evidence}")

        # Print beliefs
        if "beliefs" in result:
            print("\nFinal Beliefs:")
            for entity, belief in result["beliefs"].items():
                if isinstance(belief, dict):
                    label = belief.get("label", "?")
                    evidence = belief.get("evidence", "")[:80]
                else:
                    label = belief.label
                    evidence = belief.evidence[:80]
                print(f"  [{entity}] {label}: {evidence}")

        # Print frontier
        if "frontier" in result:
            print(
                f"\nExplanatory Frontier"
                f" ({len(result['frontier'])} findings):"
            )
            for item in result["frontier"]:
                print(f"  - {item}")

        print(f"\nSteps taken: {result.get('steps_taken', '?')}")

    except Exception as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
