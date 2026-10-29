"""The ``samples/skills-agent`` graph, run end to end with a scripted chat model."""

import importlib.util
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from pydantic import Field
from uipath.platform.common import (
    ResourceOverwrite,
    ResourceOverwriteParser,
    ResourceOverwritesContext,
)

SAMPLE = Path(__file__).parents[2] / "samples" / "skills-agent"
TASK = "Supplier Acme's invoice INV-1042 is 45 days overdue. What should I do?"
ANSWER = "Notify the budget owner, then escalate to the finance manager."


class _ScriptedModel(GenericFakeChatModel):
    """Replies from ``messages`` in order and records every prompt it was sent."""

    prompts: list[list[BaseMessage]] = Field(default_factory=list)
    bound: list[str] = Field(default_factory=list)
    built: int = 0
    """How many times the sample's ``make_llm`` returned this model."""

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "_ScriptedModel":
        self.bound = [tool.name for tool in tools]
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.prompts.append(list(messages))
        return super()._generate(messages, stop, run_manager, **kwargs)


@pytest.fixture(scope="module")
def sample() -> ModuleType:
    """The sample's ``main.py``, imported by path."""
    spec = importlib.util.spec_from_file_location(
        "skills_agent_sample", SAMPLE / "main.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def environment(monkeypatch: pytest.MonkeyPatch, sample: ModuleType) -> Iterator[None]:
    """Read the sample's own bundles locally; forget the cached set after each test."""
    monkeypatch.setenv("UIPATH_SKILLS_SOURCE", "local")
    monkeypatch.setenv("UIPATH_SKILLS_LOCAL_ROOT", str(SAMPLE / "skills"))
    monkeypatch.setenv("UIPATH_URL", "https://test.uipath.com/org/tenant")
    monkeypatch.setenv("UIPATH_ACCESS_TOKEN", "secret")
    sample.SKILLS.cleanup()
    yield
    sample.SKILLS.cleanup()


def _script(
    monkeypatch: pytest.MonkeyPatch, sample: ModuleType, replies: list[AIMessage]
) -> _ScriptedModel:
    model = _ScriptedModel(messages=iter(replies))

    def make_llm() -> _ScriptedModel:
        model.built += 1
        return model

    monkeypatch.setattr(sample, "make_llm", make_llm)
    return model


def _load(*names: str) -> AIMessage:
    """A model turn that calls ``load_skill`` once per name."""
    return AIMessage(
        content="",
        tool_calls=[
            {"name": "load_skill", "args": {"name": name}, "id": f"call-{i}"}
            for i, name in enumerate(names)
        ],
    )


def _skill_loads(exporter: InMemorySpanExporter) -> list[str]:
    return [
        s.name
        for s in exporter.get_finished_spans()
        if s.name.startswith("Skill load - ")
    ]


async def test_sample_loads_a_skill_and_reports_it(
    monkeypatch: pytest.MonkeyPatch,
    sample: ModuleType,
    exporter: InMemorySpanExporter,
) -> None:
    replies = [_load("ops/escalation-policy"), AIMessage(content=ANSWER)]
    model = _script(monkeypatch, sample, replies)

    graph = await sample.make_graph()
    assert set(graph.nodes) >= {"agent", "tools", "finish"}
    assert model.built == 0
    output = await graph.ainvoke({"task": TASK})

    assert output == {"answer": ANSWER, "skills_used": ["ops/escalation-policy"]}
    assert model.built == 1
    assert model.bound == ["load_skill"]
    system = model.prompts[0][0]
    assert isinstance(system, SystemMessage)
    assert "## Skills" in system.text
    assert "ops/escalation-policy" in system.text
    tool_result = model.prompts[1][-1]
    assert isinstance(tool_result, ToolMessage)
    assert tool_result.text.startswith("# ops/escalation-policy (bundle ops v1)")
    assert _skill_loads(exporter) == ["Skill load - ops/escalation-policy"]


async def test_sample_does_not_count_a_failed_load(
    monkeypatch: pytest.MonkeyPatch,
    sample: ModuleType,
    exporter: InMemorySpanExporter,
) -> None:
    replies = [
        _load("ops/no-such-skill"),
        _load("ops/escalation-policy"),
        AIMessage(content=ANSWER),
    ]
    model = _script(monkeypatch, sample, replies)

    output = await (await sample.make_graph()).ainvoke({"task": TASK})

    assert output["skills_used"] == ["ops/escalation-policy"]
    rejected = model.prompts[1][-1]
    assert isinstance(rejected, ToolMessage) and rejected.status == "error"
    assert _skill_loads(exporter) == ["Skill load - ops/escalation-policy"]
    assert model.built == 1


def test_skills_used_skips_an_unknown_skill_answer(sample: ModuleType) -> None:
    # The lazy tool takes any string and answers an unknown name with text.
    messages = [
        _load("ops/nope", "finance/vendor-lookup"),
        ToolMessage("Unknown skill 'ops/nope'. Available: ...", tool_call_id="call-0"),
        ToolMessage(
            "# finance/vendor-lookup (bundle finance v3)", tool_call_id="call-1"
        ),
    ]
    assert sample.skills_used(messages) == ["finance/vendor-lookup"]


async def test_sample_reads_its_own_bundles(
    monkeypatch: pytest.MonkeyPatch,
    sample: ModuleType,
    exporter: InMemorySpanExporter,
) -> None:
    replies = [_load("ops/escalation-policy"), AIMessage(content=ANSWER)]
    _script(monkeypatch, sample, replies)

    output = await (await sample.make_graph()).ainvoke({"task": TASK})

    assert output == {"answer": ANSWER, "skills_used": ["ops/escalation-policy"]}
    skill_set = sample.SKILLS.skill_set
    assert skill_set.source == "local"
    assert {b.ref.name: b.version for b in skill_set.bundles} == {
        "finance": "3",
        "ops": "1",
    }
    assert _skill_loads(exporter) == ["Skill load - ops/escalation-policy"]


async def test_sample_applies_a_binding_override(
    monkeypatch: pytest.MonkeyPatch, sample: ModuleType
) -> None:
    _script(monkeypatch, sample, [])

    async def overwrites() -> dict[str, ResourceOverwrite]:
        key = "skillBundle.ops"
        value = {"name": "finance"}
        return {key: ResourceOverwriteParser.parse(key, value)}

    async with ResourceOverwritesContext(overwrites):
        await sample.make_graph()

    skill_set = sample.SKILLS.skill_set
    assert [(b.ref.name, b.ref.lookup_name) for b in skill_set.bundles] == [
        ("finance", "finance"),
        ("ops", "finance"),
    ]
    assert skill_set.failures == []
    assert "ops/invoice-triage" in sample.SKILLS.prompt_section()
    assert "ops/escalation-policy" not in sample.SKILLS.prompt_section()


async def test_sample_runs_without_tools_when_no_bundle_resolves(
    monkeypatch: pytest.MonkeyPatch, sample: ModuleType, tmp_path: Path
) -> None:
    monkeypatch.setenv("UIPATH_SKILLS_LOCAL_ROOT", str(tmp_path))
    model = _script(monkeypatch, sample, [AIMessage(content=ANSWER)])

    graph = await sample.make_graph()
    assert "tools" not in graph.nodes
    output = await graph.ainvoke({"task": TASK})

    assert output == {"answer": ANSWER, "skills_used": []}
    assert model.bound == []
    assert "## Skills" not in model.prompts[0][0].text
    assert sample.SKILLS.skill_set.failures
