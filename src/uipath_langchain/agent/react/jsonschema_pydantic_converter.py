"""Build Pydantic models from JSON Schema at runtime.

An agent's inputs, outputs, and every tool's parameters are stored as JSON
Schema, while LangChain needs a Pydantic class to validate the arguments a model
produces. This module is the bridge, and the only entry point callers use.

Two backends implement the conversion, chosen by the
``EnableDatamodelCodeGeneratorConverter`` feature flag:

* flag off (the default) -- :mod:`._legacy_converter`, backed by
  ``jsonschema-pydantic-converter``;
* flag on -- :mod:`._datamodel_code_generator_converter`, backed by
  ``datamodel-code-generator``, which names generated types after the schema
  instead of ``DynamicType_N``, homes every class in the schema's own module,
  and repairs property names that are not valid Python identifiers.

Both resolve ``$ref``s completely, an inline object holding one included. The flag
exists to roll the newer backend out gradually, not to escape a broken one.

Both produce a model with the same observable contract: the original JSON
property names on the wire, ``__uipath_marker_name__`` on types reached through a
``$ref``, generated classes reachable through a module in ``sys.modules``, and an
``AgentStartupError`` naming the type when a ``$ref`` cannot be resolved.

Classes are built once per backend and schema, in a module named after a hash
of both. A suspended run's checkpoint stores values by ``module.Class``, so that
name must resolve to the same class in whichever process resumes it.
"""

import hashlib
import json
import logging
import threading
from typing import Any, Type

from pydantic import BaseModel
from uipath.core.feature_flags import FeatureFlags

from . import _datamodel_code_generator_converter, _legacy_converter
from ._schema_refs import UNRESOLVED_TYPE_TITLE, neutralize_dangling_refs

logger = logging.getLogger(__name__)

# Selects the datamodel-code-generator backend. Off by default: the legacy
# converter stays in charge until the new path has been exercised in the wild.
DATAMODEL_CODE_GENERATOR_CONVERTER_FF = "EnableDatamodelCodeGeneratorConverter"

_DYNAMIC_MODULE_PREFIX = "jsonschema_pydantic_converter._dynamic"

_models_by_module: dict[str, Type[BaseModel]] = {}
_models_lock = threading.Lock()

__all__ = [
    "create_model",
    "create_output_model",
]


def _datamodel_code_generator_enabled() -> bool:
    """Whether to build models with the code-generator backend."""
    return FeatureFlags.is_flag_enabled(
        DATAMODEL_CODE_GENERATOR_CONVERTER_FF, default=False
    )


def create_model(
    schema: dict[str, Any],
) -> Type[BaseModel]:
    """Convert a JSON schema dict to a Pydantic model.

    Raises:
        AgentStartupError: If the schema contains a type that cannot be resolved.
    """
    backend = (
        _datamodel_code_generator_converter
        if _datamodel_code_generator_enabled()
        else _legacy_converter
    )
    canonical = json.dumps(schema, sort_keys=True)
    digest = hashlib.sha256(f"{backend.__name__}:{canonical}".encode()).hexdigest()
    module_name = f"{_DYNAMIC_MODULE_PREFIX}_{digest}"

    with _models_lock:
        model = _models_by_module.get(module_name)
        if model is None:
            model = backend.create_model(schema, module_name)
            _models_by_module[module_name] = model
    return model


def create_output_model(
    schema: dict[str, Any],
    tool_name: str,
) -> Type[BaseModel]:
    """Convert a tool's OUTPUT JSON schema to a Pydantic model.

    Unresolvable ``$ref``s -- the malformed output schema seen in practice (see
    neutralize_dangling_refs) -- are neutralized in place so all valid fields are
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
    sanitized, dropped = neutralize_dangling_refs(schema)
    if dropped:
        logger.warning(
            "Tool %r output schema had %d unresolvable $ref(s) (%s); each replaced "
            "with a permissive %r placeholder. Output schema does not affect the "
            "core tool call, so agent startup is not blocked.",
            tool_name,
            len(dropped),
            ", ".join(sorted(set(dropped))),
            UNRESOLVED_TYPE_TITLE,
        )
    return create_model(sanitized)
