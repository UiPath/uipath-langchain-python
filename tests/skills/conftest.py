"""Fixtures shared by the skills tests."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from uipath.platform import UiPath
from uipath.platform.skills import SkillBundleRef, SkillSet, VersionMode

FIXTURES = Path(__file__).parent / "fixtures" / "skills"
"""Bundles ``finance`` (v3: ``invoice-triage``, ``vendor-lookup``) and ``ops`` (v1: ``vendor-lookup``)."""


@pytest.fixture
def skill_set(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[SkillSet]:
    """A real ``SkillSet`` materialized from the fixture bundles into ``tmp_path``."""
    monkeypatch.setenv("UIPATH_SKILLS_SOURCE", "local")
    monkeypatch.setenv("UIPATH_SKILLS_LOCAL_ROOT", str(FIXTURES))
    monkeypatch.setenv("UIPATH_FOLDER_PATH", "Shared/Skills")
    monkeypatch.setenv("UIPATH_URL", "https://test.uipath.com/org/tenant")
    monkeypatch.setenv("UIPATH_ACCESS_TOKEN", "secret")
    skill_set = UiPath().skills.load_all(
        [
            SkillBundleRef(name="finance", version_mode=VersionMode.LATEST),
            SkillBundleRef(name="ops", version_mode=VersionMode.PINNED, version="1"),
        ],
        destination=tmp_path / "skills",
    )
    assert not skill_set.failures, skill_set.failures
    yield skill_set
    # ``destination`` lives under ``tmp_path``, so no cleanup is needed.


@pytest.fixture
def empty_skill_set() -> SkillSet:
    """A set with no bundles at all."""
    return SkillSet(source="local", bundles=[], failures=[])


@pytest.fixture
def exporter(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
    """Capture the spans the SDK's skills package emits."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # The global tracer provider can be set only once per process, so swap the
    # SDK module-level tracer instead of installing a provider.
    monkeypatch.setattr(
        "uipath.platform.skills._telemetry._tracer", provider.get_tracer("test")
    )
    return exporter
