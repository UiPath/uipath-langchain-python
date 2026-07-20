"""EoG (Explanations over Graphs) agent pattern.

Lazy ontology traversal — no upfront graph fetch. Function definitions
(with ``touches``, ``outputs``, ``params``) are the navigation contract.
"""

from __future__ import annotations

from .agent import create_eog_agent
from .ontology_client import OntologyClient
from .types import Belief, EoGState, FunctionSpec, InvestigationConfig, LedgerEntry

__all__ = [
    "create_eog_agent",
    "OntologyClient",
    "EoGState",
    "Belief",
    "FunctionSpec",
    "LedgerEntry",
    "InvestigationConfig",
]
