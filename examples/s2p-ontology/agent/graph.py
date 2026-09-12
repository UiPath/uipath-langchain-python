"""S2P EoG Investigation Agent.

Demonstrates the EoG (Explanations over Graphs) pattern on the S2P procurement
ontology. Connects to a local ontology-runtime and uses graph-guided investigation
to diagnose procurement issues (tolerance exceptions, maverick spend, etc.).
"""

import os

from uipath_langchain.agent.eog import (
    InvestigationConfig,
    OntologyClient,
    create_eog_agent,
)

# LLM -- support both UiPath Chat and direct OpenAI
try:
    from uipath_langchain.chat import UiPathChat

    llm = UiPathChat(
        model=os.getenv("LLM_MODEL", "gpt-4o-2024-08-06"),
        temperature=0.0,
    )
except Exception:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o-2024-08-06"),
        temperature=0.0,
    )

# Ontology client
client = OntologyClient(
    base_url=os.getenv("ONTOLOGY_BASE_URL", "http://localhost:5002"),
    account=os.getenv("ONTOLOGY_ACCOUNT", "datafabric"),
    tenant=os.getenv("ONTOLOGY_TENANT", "DefaultTenant"),
)

# S2P investigation configuration
S2P_LABELS = [
    "Source",  # Root cause / origin of the issue
    "DerivedEffect",  # Consequence of an upstream issue
    "PolicyViolation",  # SHACL/business rule violation
    "CandidateMatch",  # Potential resolution or match
    "SupportingEvidence",  # Corroborating data point
    "Contradiction",  # Data that conflicts with the emerging explanation
    "Defer",  # Insufficient evidence to classify
]

investigation_config = InvestigationConfig(
    label_vocabulary=S2P_LABELS,
    seed_entities=[],  # Populated by seed function or at invocation
    max_steps=30,
    max_flips=3,
    default_label="Defer",
    max_results_per_function=50,
)

# Create EoG agent (returns uncompiled StateGraph)
eog_agent = create_eog_agent(
    model=llm,
    ontology_client=client,
    ontology_name=os.getenv("ONTOLOGY_NAME", "s2p"),
    investigation_config=investigation_config,
)

# Compile and export
graph = eog_agent.compile()
