import inspect
import logging
import sys
from types import ModuleType
from typing import Any, Type, cast

from jsonschema_pydantic_converter import transform_with_modules
from pydantic import BaseModel, PydanticUndefinedAnnotation

from uipath_langchain.agent.exceptions import AgentStartupError, AgentStartupErrorCode

logger = logging.getLogger(__name__)

# Marker left on any OUTPUT-schema node whose $ref target could not be resolved.
# The converter discards $defs names and non-standard (x-*) keys but preserves the
# standard `title`/`description` annotations on a property, so the marker lives as
# annotations rather than a named type. Downstream can detect an unresolved field
# via ``title == _UNRESOLVED_TYPE_TITLE``. See create_output_model.
_UNRESOLVED_TYPE_TITLE = "UiPathUnresolvedType"

# Shared pseudo-module for all dynamically created types
# This allows get_type_hints() to resolve forward references
_DYNAMIC_MODULE_NAME = "jsonschema_pydantic_converter._dynamic"


def _get_or_create_dynamic_module() -> ModuleType:
    """Get or create the shared pseudo-module for dynamic types."""
    if _DYNAMIC_MODULE_NAME not in sys.modules:
        pseudo_module = ModuleType(_DYNAMIC_MODULE_NAME)
        pseudo_module.__doc__ = (
            "Shared module for dynamically generated Pydantic models from JSON schemas"
        )
        sys.modules[_DYNAMIC_MODULE_NAME] = pseudo_module
    return sys.modules[_DYNAMIC_MODULE_NAME]


def create_model(
    schema: dict[str, Any],
) -> Type[BaseModel]:
    """Convert a JSON schema dict to a Pydantic model.

    Raises:
        AgentStartupError: If the schema contains a type that cannot be resolved.
    """
    try:
        model, namespace = transform_with_modules(schema)
    except PydanticUndefinedAnnotation as e:
        # Strip the __ prefix the converter adds to forward references
        # so the user sees the original type name from their JSON schema.
        type_name = e.name.lstrip("_") if e.name else None
        raise AgentStartupError(
            code=AgentStartupErrorCode.INVALID_TOOL_CONFIG,
            title="Invalid schema",
            detail=(
                f"Type '{type_name}' could not be resolved. "
                f"Check that all $ref targets have matching entries in $defs."
            ),
        ) from e

    pseudo_module = _get_or_create_dynamic_module()

    for type_name, type_def in namespace.items():
        setattr(pseudo_module, type_name, type_def)
        if inspect.isclass(type_def) and issubclass(type_def, BaseModel):
            type_def.__module__ = _DYNAMIC_MODULE_NAME
            # per-class marker; survives the shared module being overwritten.
            cast(Any, type_def).__uipath_marker_name__ = type_name

    setattr(pseudo_module, model.__name__, model)
    model.__module__ = _DYNAMIC_MODULE_NAME

    return model


def _ref_resolves(ref: str, root: dict[str, Any]) -> bool:
    """Whether a local JSON-pointer ``$ref`` (``#/...``) resolves within `root`.

    External/URL refs and the bare ``#`` (whole-document) ref return False: the
    converter cannot resolve them either, so they are treated as dangling.
    """
    if not ref.startswith("#/"):
        return False
    node: Any = root
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")  # JSON-pointer unescape
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return False
    return True


def _neutralize_dangling_refs(
    schema: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Return a copy of `schema` with every unresolvable ``$ref`` replaced.

    A ``$ref`` is dangling when its target is not present under ``$defs``/
    ``definitions`` (e.g. a .NET ``Nullable<decimal>`` serialized without its
    definition). Each dangling ref node is replaced *in place* by a permissive,
    self-documenting placeholder (accepts any value; the original ref is kept in
    its ``description``), so valid sibling fields and valid ``$ref``s -- including
    those nested in arrays, objects, or ``$defs`` -- are preserved. This keeps the
    output schema usable by best-effort features instead of discarding it whole.

    Returns:
        A tuple of (sanitized schema copy, list of the dangling ref strings found).
    """
    dropped: list[str] = []

    def visit(node: Any) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and not _ref_resolves(ref, schema):
                dropped.append(ref)
                return {
                    "title": _UNRESOLVED_TYPE_TITLE,
                    "description": (
                        f"Unresolved $ref '{ref}'; original type could not be "
                        "resolved at startup, so this field accepts any value."
                    ),
                }
            return {key: visit(value) for key, value in node.items()}
        if isinstance(node, list):
            return [visit(item) for item in node]
        return node

    return visit(schema), dropped


def create_output_model(
    schema: dict[str, Any],
    tool_name: str,
) -> Type[BaseModel]:
    """Convert a tool's OUTPUT JSON schema to a Pydantic model.

    Unresolvable ``$ref``s -- the malformed output schema seen in practice (see
    _neutralize_dangling_refs) -- are neutralized in place so all valid fields are
    kept; since an output schema drives only best-effort features (job-attachment
    discovery, output guardrails, eval simulations), losing a single unresolvable
    field is preferable to failing startup.

    Any *other* conversion failure is deliberately left fatal: we would rather fail
    loudly at startup than swallow an unexpected malformation into a degraded model
    that fails obscurely at runtime.

    Returns:
        The converted model, with dangling refs neutralized.

    Raises:
        AgentStartupError: If the schema is unparseable for a reason other than a
            dangling ``$ref``.
    """
    sanitized, dropped = _neutralize_dangling_refs(schema)
    if dropped:
        logger.warning(
            "Tool %r output schema had %d unresolvable $ref(s) (%s); each replaced "
            "with a permissive %r placeholder. Output schema does not affect the "
            "core tool call, so agent startup is not blocked.",
            tool_name,
            len(dropped),
            ", ".join(sorted(set(dropped))),
            _UNRESOLVED_TYPE_TITLE,
        )
    return create_model(sanitized)
