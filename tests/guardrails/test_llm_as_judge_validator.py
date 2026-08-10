"""Re-export test for LLMAsJudgeValidator via uipath_langchain.guardrails.

The validator itself lives in uipath-platform; this only verifies that the
uipath_langchain.guardrails re-export exposes it and that it builds the expected
guardrail. Guarded so langchain CI skips cleanly until the uipath-platform release
that ships LLMAsJudgeValidator is available (remove the guard after the bump).
"""

import pytest

llm_as_judge_guardrails = pytest.importorskip("uipath.platform.guardrails.decorators")
if not hasattr(llm_as_judge_guardrails, "LLMAsJudgeValidator"):
    pytest.skip(
        "LLMAsJudgeValidator not in installed uipath-platform yet",
        allow_module_level=True,
    )

from uipath_langchain.guardrails import LLMAsJudgeValidator  # noqa: E402

_RULE = "The joke must be genuinely funny, clean, and on-topic."


def test_reexported_from_langchain_guardrails() -> None:
    guardrail = LLMAsJudgeValidator(
        guardrail_text=_RULE, model="gpt-4o-2024-08-06"
    ).get_built_in_guardrail(name="Judge", description=None, enabled_for_evals=True)
    assert guardrail.validator_type == "llm_as_judge"
    params = {p.id: p for p in guardrail.validator_parameters}
    assert params["guardrailText"].value == _RULE
    assert params["model"].value == "gpt-4o-2024-08-06"
    assert params["threshold"].value == 2.0
    # Decorator-path convention: scope comes from the wrapped target.
    assert guardrail.selector is None
