"""A fake Orchestrator for the output-file check: attachments, job links, calls."""

from dataclasses import dataclass, field
from typing import Any

import httpx
from uipath.platform.attachments import BlobFileAccessInfo
from uipath.platform.errors import EnrichedException

JOB_KEY = "33333333-3333-3333-3333-333333333333"


@dataclass
class FakeOrchestrator:
    existing: dict[str, str]
    """Attachment id to file name, for every attachment that exists."""

    linked: list[str]
    """Attachment ids already linked to the current job."""

    lookups: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)


def _not_found() -> EnrichedException:
    request = httpx.Request("GET", "https://example.test/odata/Attachments(x)")
    response = httpx.Response(404, request=request, text="Attachment not found.")
    return EnrichedException(
        httpx.HTTPStatusError("not found", request=request, response=response)
    )


def patch_orchestrator(
    monkeypatch: Any,
    *,
    existing: dict[str, str],
    linked: list[str] | None = None,
) -> FakeOrchestrator:
    """Point the output-file check at a fake current job and attachment store."""
    monkeypatch.setenv("UIPATH_JOB_KEY", JOB_KEY)
    monkeypatch.delenv("UIPATH_FOLDER_KEY", raising=False)
    fake = FakeOrchestrator(existing=existing, linked=list(linked or []))

    class FakeAttachments:
        async def get_blob_file_access_uri_async(
            self, *, key: Any, **_: Any
        ) -> BlobFileAccessInfo:
            fake.lookups.append(str(key))
            name = fake.existing.get(str(key))
            if name is None:
                raise _not_found()
            return BlobFileAccessInfo(id=key, uri="https://blob.test", name=name)

    class FakeJobs:
        async def list_attachments_async(self, **_: Any) -> list[str]:
            return fake.linked

        async def link_attachment_async(self, *, attachment_key: Any, **_: Any) -> None:
            fake.links.append(str(attachment_key))

    class FakeUiPath:
        attachments = FakeAttachments()
        jobs = FakeJobs()

    monkeypatch.setattr(
        "uipath_langchain.agent.attachments.output_files.UiPath",
        lambda *args, **kwargs: FakeUiPath(),
    )
    return fake
