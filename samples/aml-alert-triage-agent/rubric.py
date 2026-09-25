"""AML triage rubric — the policy artifact, versioned separately from agent code.

Derived from public FATF / Wolfsberg money-laundering typologies. This file IS the
Jev question set: change policy here, not in the agent.

Design constraints imposed by Jev's documented weaknesses (model-jaggedness/jev-1.13):
  - unreliable at math and counting      -> all arithmetic is done upstream by the LLM
  - cannot reliably order/compare dates  -> spans are pre-computed into plain integers
  - poor at multi-step indirection       -> every question is atomic and self-contained
  - accuracy drops with irrelevant ctx   -> state passed to Jev is a tight, derived digest
"""

from typesafe_sdk import Choice, Noul, Score

RUBRIC_VERSION = "1.0.0"

# --- Red flags: one atomic yes/no each (Noul -> float 0..1) ---------------------
RED_FLAGS = {
    "structuring": (
        "Do the individual transaction amounts appear deliberately kept below a "
        "regulatory reporting threshold, rather than reflecting natural business amounts?"
    ),
    "rapid_pass_through": (
        "Are credited funds moved out again so quickly that the account behaves as a "
        "conduit rather than as a place where value is held?"
    ),
    "shell_company_indicators": (
        "Does the customer profile show characteristics of a shell company, such as "
        "recent incorporation, an opaque jurisdiction, or no evidence of real operations?"
    ),
    "high_risk_jurisdiction": (
        "Does the transaction routing involve a jurisdiction recognised as carrying "
        "elevated money-laundering risk?"
    ),
    "purpose_mismatch": (
        "Is the observed activity inconsistent with the customer's stated business purpose?"
    ),
}

# --- Risk level: ordered rubric (Score -> index into this sequence) -------------
RISK_LEVELS = [
    "Low: activity is consistent with the stated business purpose and no red flags are present.",
    "Medium: a single red flag is present and a plausible legitimate explanation exists.",
    "High: multiple red flags corroborate each other and supporting documentation is absent.",
    "Critical: the pattern shows strong placement or layering indicators with no legitimate rationale.",
]

# --- Disposition: the triage decision (Choice -> one key) -----------------------
DISPOSITIONS = {
    "escalate": "Refer to an investigator to consider a suspicious activity report.",
    "close": "No further action; the activity is adequately explained by the customer profile.",
    "need_info": "A decision is not possible without additional documentation or customer contact.",
}


def build_questions() -> dict:
    """The full Jev question set — one API call evaluates all of these in parallel."""
    q = {name: Noul(instructions=text) for name, text in RED_FLAGS.items()}
    q["risk_level"] = Score(
        instructions="Assign the overall money-laundering risk level for this alert.",
        criteria=RISK_LEVELS,
    )
    q["disposition"] = Choice(
        instructions="Choose the correct triage disposition for this alert.",
        criteria=DISPOSITIONS,
    )
    return q
