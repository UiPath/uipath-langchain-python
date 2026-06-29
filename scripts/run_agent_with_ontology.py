#!/usr/bin/env python
"""Run a coded Data Fabric agent with an OWL ontology injected into the write path.

This is a standalone CLI helper, not part of the shipped package.

What it does
------------
1. Loads an OWL 2 QL Turtle ontology and compiles it up front with
   ``compile_ontology`` so the user can see exactly what the ontology
   contributes (entity access, HITL operations, state/reference/measure
   fields, relationships) before any agent work happens. This also
   validates the .ttl.
2. Bridges a gap in the platform package: the runtime's
   ``DataFabricWriteHandler._maybe_compile_ontology`` calls
   ``entities_service.get_ontology_file_async("owl")`` to fetch the ontology,
   but the platform does NOT yet expose that method (verified: the attribute
   is absent on ``uipath.platform.entities._entities_service.EntitiesService``).
   This CLI monkeypatches that method onto the class so it returns the
   user-supplied .ttl text. That activates the real ontology-compilation path
   inside the handler exactly as it will behave once the platform ships the
   method -- the ontology is compiled and used in write validation and the
   write tool's description.
3. Builds the Data Fabric read + write tools, force-initializes the write
   handler so the compiled ontology and the generated write tool description
   can be printed, then (unless ``--dry-run``) runs the coded ReAct agent.

The monkeypatch target is::

    uipath.platform.entities._entities_service.EntitiesService.get_ontology_file_async

Usage
-----
Offline dry-run (compiles + prints ontology facts without the LLM; degrades
gracefully if entity resolution needs network and fails)::

    uv run python scripts/run_agent_with_ontology.py \
        --ontology /path/to/p1-owl-write-extension.ttl \
        --entity-set scripts/sample_refund_entity_set.json \
        --prompt "test" --dry-run

Real run against staging (requires ``uip login`` + real entity ids in the
entity-set JSON)::

    uv run python scripts/run_agent_with_ontology.py \
        --ontology /path/to/ontology.ttl \
        --entity-set scripts/sample_refund_entity_set.json \
        --prompt "Process the refund for contact ..." \
        --model anthropic.claude-sonnet-4-5-20250929-v1:0
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

# The handler resolves entities lazily through ``UiPath().entities`` and the
# EntitiesService instance is not reachable from here, so we patch the method
# at the class level. The handler's own ``_maybe_compile_ontology`` then finds
# it via ``getattr`` and fetches the ontology naturally.
ENTITIES_SERVICE_MODULE = "uipath.platform.entities._entities_service"

# Default to a UiPath-gateway-routable model. Override with --model.
DEFAULT_MODEL = "anthropic.claude-sonnet-4-5-20250929-v1:0"

DEFAULT_SYSTEM_PROMPT = (
    "You are a Data Fabric operations agent. Use the read tool to discover "
    "records and field values before writing. Use the write tool only with "
    "valid entity keys, operations, and fields. Respect any human-in-the-loop "
    "(HITL) requirements: for operations marked HITL, ask for explicit "
    "confirmation before executing. Never invent record IDs -- always look "
    "them up first."
)

# Module-level holder so the monkeypatched method (defined once, bound to the
# class) can read whichever ontology text the current CLI invocation supplied.
_ONTOLOGY_TEXT: str | None = None


def _install_ontology_monkeypatch(ttl_text: str) -> str:
    """Patch EntitiesService.get_ontology_file_async to return the .ttl text.

    Returns a human-readable description of the exact patch target. Raises on
    failure so the caller can fall back to the direct-set strategy.
    """
    global _ONTOLOGY_TEXT
    _ONTOLOGY_TEXT = ttl_text

    import importlib

    module = importlib.import_module(ENTITIES_SERVICE_MODULE)
    service_cls = module.EntitiesService

    async def get_ontology_file_async(self: Any, file_type: str = "owl") -> str | None:
        """Injected by run_agent_with_ontology.py (CLI bridge).

        Returns the user-supplied ontology text for the ``owl`` file type.
        """
        if file_type and file_type.lower() != "owl":
            return None
        return _ONTOLOGY_TEXT

    service_cls.get_ontology_file_async = get_ontology_file_async  # type: ignore[attr-defined]
    return f"{ENTITIES_SERVICE_MODULE}.EntitiesService.get_ontology_file_async"


def _print_ontology_facts(compiled: Any, *, header: str) -> None:
    """Pretty-print the facts extracted from a CompiledOntology."""
    print(f"\n=== {header} ===")
    if compiled is None:
        print("(no compiled ontology)")
        return
    if hasattr(compiled, "is_empty") and compiled.is_empty():
        print("(ontology compiled but EMPTY -- no facts extracted)")
        return

    def _fmt_set_map(m: dict[str, Any]) -> str:
        if not m:
            return "    (none)"
        lines = []
        for key in sorted(m):
            val = m[key]
            if isinstance(val, set):
                val = sorted(val)
            lines.append(f"    {key}: {val}")
        return "\n".join(lines)

    print("  entity_access (entity -> allowed ops):")
    print(_fmt_set_map(compiled.entity_access))
    print("  hitl_operations (entity -> ops requiring HITL):")
    print(_fmt_set_map(compiled.hitl_operations))
    print("  state_fields (entity.field -> state machine / choiceset):")
    print(_fmt_set_map(compiled.state_fields))
    print("  reference_fields (entity.field -> referenced entity):")
    print(_fmt_set_map(compiled.reference_fields))
    print("  measure_fields (entity.field -> semantics):")
    print(_fmt_set_map(compiled.measure_fields))
    print("  entity_relationships (entity -> related entities):")
    print(_fmt_set_map(compiled.entity_relationships))


def _print_ontology_debug(
    owl_turtle: str, compiled: Any, *, debug_ontology: bool
) -> None:
    """Print the human-readable IR and, when enabled, the raw OWL Turtle.

    Uses ``ontology_compiler.format_ontology_debug`` so the CLI output mirrors
    exactly what the runtime emits to debug logs.
    """
    if compiled is None:
        return
    from uipath_langchain.agent.tools.datafabric_tool.ontology_compiler import (
        format_ontology_debug,
    )

    if debug_ontology:
        # Full block: raw OWL + human-readable IR.
        print()
        print(format_ontology_debug(owl_turtle, compiled))
    else:
        # Human-readable IR only (no raw-OWL dump).
        print("\n=== COMPILED ONTOLOGY (human-readable IR) ===")
        print(compiled.to_human_readable())


def _load_entity_set(path: Path) -> list[Any]:
    """Load a JSON list of DataFabricEntityItem dicts into model instances."""
    import json

    from uipath.platform.entities import DataFabricEntityItem

    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError(
            f"Entity-set JSON must be a list of objects; got {type(raw).__name__}."
        )
    return [DataFabricEntityItem.model_validate(item) for item in raw]


def _find_write_handler(tools: list[Any]) -> Any:
    """Locate the DataFabricWriteHandler from the created tools."""
    for tool in tools:
        meta = getattr(tool, "metadata", None) or {}
        if meta.get("tool_type") == "datafabric_write":
            # The handler is the tool's coroutine callable.
            handler = getattr(tool, "coroutine", None) or getattr(tool, "func", None)
            return tool, handler
    return None, None


class _AuthOrNetworkError(RuntimeError):
    """Raised to signal an expected auth/network failure (no traceback wanted)."""


async def _async_main(args: argparse.Namespace) -> int:
    ontology_path = Path(args.ontology).expanduser()
    entity_set_path = Path(args.entity_set).expanduser()

    if not ontology_path.is_file():
        print(f"ERROR: ontology file not found: {ontology_path}", file=sys.stderr)
        return 2
    if not entity_set_path.is_file():
        print(f"ERROR: entity-set file not found: {entity_set_path}", file=sys.stderr)
        return 2

    ttl_text = ontology_path.read_text()

    # --- Step 1: compile + print ontology facts up front (no network). -------
    # This validates the .ttl before any agent work and shows the user exactly
    # what the ontology contributes.
    from uipath_langchain.agent.tools.datafabric_tool.ontology_compiler import (
        OntologyCompileError,
        compile_ontology,
    )

    try:
        standalone_compiled = compile_ontology(ttl_text)
    except OntologyCompileError as exc:
        print(f"ERROR: failed to compile ontology: {exc}", file=sys.stderr)
        return 2

    print(f"Loaded ontology: {ontology_path}")
    _print_ontology_facts(
        standalone_compiled, header="ONTOLOGY FACTS (standalone compile)"
    )
    # Richer human-readable IR (and, when --debug-ontology, the raw OWL too).
    _print_ontology_debug(
        ttl_text, standalone_compiled, debug_ontology=args.debug_ontology
    )

    # --- Step 2: load entity set. --------------------------------------------
    try:
        entity_items = _load_entity_set(entity_set_path)
    except Exception as exc:
        print(f"ERROR: failed to load entity-set JSON: {exc}", file=sys.stderr)
        return 2
    print(f"\nLoaded {len(entity_items)} entity item(s) from {entity_set_path}")

    # --- Step 3: build the resource config. ----------------------------------
    from uipath.agent.models.agent import (
        AgentContextResourceConfig,
        AgentContextType,
    )

    resource = AgentContextResourceConfig(
        name=args.resource_name,
        description="Data Fabric entity set for the ontology-injected agent run.",
        context_type=AgentContextType.DATA_FABRIC_ENTITY_SET,
        entity_set=entity_items,
    )

    # --- Step 4: install the monkeypatch BEFORE the handler runs. ------------
    try:
        patch_target = _install_ontology_monkeypatch(ttl_text)
        print(f"\nInstalled ontology bridge at: {patch_target}")
    except Exception as exc:
        print(
            f"WARNING: class-level monkeypatch failed ({exc}); "
            "will fall back to direct-set on the handler.",
            file=sys.stderr,
        )
        patch_target = None

    # --- Step 5 + 6: build the LLM and the Data Fabric tools. ----------------
    # Everything from here may need network/auth. Wrap so the common auth case
    # degrades gracefully (and, in --dry-run, still shows the ontology facts).
    try:
        from uipath_langchain.agent.tools.datafabric_tool import (
            create_datafabric_tools,
        )
        from uipath_langchain.chat import get_chat_model

        try:
            # agenthub_config carries the AgentHub OpCode that routes LLM
            # licensing on the gateway. Without it the call defaults to an
            # unlicensed product path and the gateway returns 403 "License
            # not available for LLM usage". "agentsplayground" uses the
            # developer's debug/playground quota — appropriate for a local run.
            llm = get_chat_model(args.model, agenthub_config=args.agenthub_config)
        except Exception as exc:
            raise _AuthOrNetworkError(
                f"could not construct chat model {args.model!r}: {exc}"
            ) from exc

        tools = create_datafabric_tools(resource, llm)
        write_tool, write_handler = _find_write_handler(tools)
        if write_handler is None:
            print("ERROR: could not locate the write tool/handler.", file=sys.stderr)
            return 2

        # --- Step 7: force-initialize the write handler (needs network). -----
        try:
            await write_handler._ensure_initialized()
        except Exception as exc:
            raise _AuthOrNetworkError(f"entity resolution failed: {exc}") from exc

    except _AuthOrNetworkError as exc:
        print(
            "\n--- Tool build / entity resolution did not complete ---\n"
            f"What failed: {exc}\n"
            "This step needs UiPath auth + network (UiPath(), the LLM gateway, "
            "and staging entity resolution). Run `uip login` and ensure the "
            "entity-set JSON has real entity ids/folderIds for your tenant.",
            file=sys.stderr,
        )
        if args.dry_run:
            print(
                "\n--dry-run: the standalone ontology facts above were compiled "
                "WITHOUT network and are valid. Skipping the live write tool "
                "description (it requires resolved entities)."
            )
            return 0
        return 1

    # --- Verify the ontology is actually ACTIVE in the handler. --------------
    compiled = getattr(write_handler, "_compiled_ontology", None)

    if compiled is None and patch_target is None:
        # Fallback strategy: directly set the compiled ontology and rebuild the
        # description. Used only if class-level patching was unavailable.
        from uipath_langchain.agent.tools.datafabric_tool.write_schema_builder import (
            build_write_tool_description,
        )

        compiled = standalone_compiled
        write_handler._compiled_ontology = compiled
        write_handler._write_tool_description = build_write_tool_description(
            write_handler._write_schemas,
            entity_access=compiled.entity_access,
        )
        print("Applied direct-set fallback for the compiled ontology.")

    if compiled is not None and not (
        hasattr(compiled, "is_empty") and compiled.is_empty()
    ):
        print("\nontology ACTIVE -- handler compiled and is using the ontology.")
        _print_ontology_facts(compiled, header="ONTOLOGY FACTS (active in handler)")
        _print_ontology_debug(ttl_text, compiled, debug_ontology=args.debug_ontology)
    else:
        print(
            "\nontology INACTIVE (fell back to metadata-only). The handler did "
            "not pick up a compiled ontology."
        )

    description = getattr(write_handler, "_write_tool_description", None) or getattr(
        write_tool, "description", ""
    )
    print("\n=== GENERATED WRITE TOOL DESCRIPTION ===")
    print(description)

    # --- Step 8: dry-run stops here. -----------------------------------------
    if args.dry_run:
        print("\n--dry-run: not invoking the LLM. Exiting.")
        return 0

    # --- Step 9: build messages, the agent, and run it. ----------------------
    from langchain_core.messages import HumanMessage, SystemMessage

    from uipath_langchain.agent.react import create_agent

    if args.system_prompt:
        system_text = Path(args.system_prompt).expanduser().read_text()
    else:
        system_text = DEFAULT_SYSTEM_PROMPT

    messages = [SystemMessage(content=system_text), HumanMessage(content=args.prompt)]

    try:
        graph = create_agent(llm, tools, messages).compile()
        # The refund flow needs many steps (several reads, a decision, then 4
        # writes). The default recursion_limit (25) can be exhausted before the
        # terminal write batch executes, leaving the writes only *planned*.
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=args.prompt)]},
            config={"recursion_limit": 80},
        )
    except Exception as exc:
        print(
            "\n--- Agent run failed ---\n"
            f"What failed: {exc}\n"
            "The agent run needs the LLM gateway + staging entities. "
            "Use --dry-run to validate the ontology compilation offline.",
            file=sys.stderr,
        )
        return 1

    print("\n=== AGENT RUN RESULT ===")
    result_messages = result.get("messages", []) if isinstance(result, dict) else []
    for msg in result_messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for call in tool_calls:
                name = call.get("name") if isinstance(call, dict) else None
                cargs = call.get("args") if isinstance(call, dict) else None
                print(f"[tool call] {name}({cargs})")
    final = result_messages[-1] if result_messages else None
    print("\n=== FINAL MESSAGE ===")
    print(getattr(final, "content", final) if final is not None else "(no messages)")
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load an OWL ontology, inject it into the Data Fabric write tool's "
            "fetch path, and run a coded agent."
        )
    )
    parser.add_argument(
        "--ontology", required=True, help="Path to the OWL 2 QL Turtle (.ttl) file."
    )
    parser.add_argument(
        "--entity-set",
        required=True,
        help="Path to a JSON list of DataFabricEntityItem dicts "
        "(id, name, folderId, referenceKey, description).",
    )
    parser.add_argument(
        "--prompt", required=True, help="The user prompt for the agent."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"UiPath-gateway model name (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--agenthub-config",
        default="agentsplayground",
        help="AgentHub OpCode for LLM-gateway licensing routing "
        "(default: agentsplayground — uses the developer's playground quota). "
        "Without a valid value the gateway returns 403 'License not available'.",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional path to a system-prompt/SOP .txt file. A generic default "
        "is used when omitted.",
    )
    parser.add_argument(
        "--resource-name",
        default="datafabric",
        help="Name for the Data Fabric context resource (default: datafabric).",
    )
    parser.add_argument(
        "--debug-ontology",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Dump the raw OWL Turtle alongside the human-readable compiled IR "
        "(default: on). Use --no-debug-ontology to print only the IR.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do NOT call the LLM. Build tools, compile + inject the ontology, "
        "print the ontology facts and the generated write tool description, "
        "then exit. Degrades gracefully offline.",
    )
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    return asyncio.run(_async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
