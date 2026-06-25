"""Typed errors for unsupported file attachments.

Format support for non-image attachments is delegated to the LLM/provider (#842):
we do not gate on a MIME allow-list. When a provider cannot read a file's type it
rejects it during request conversion with a bare ``ValueError`` (e.g. AWS Bedrock
Converse raises ``"Unsupported MIME type: ..."``). The file type is the user's
choice, so that provider verdict is translated into a ``USER`` error at the
model-invocation boundary rather than surfacing as an opaque ``Unknown``.
"""

from uipath.runtime.errors import UiPathErrorCategory

from uipath_langchain.agent.exceptions.exceptions import (
    AgentRuntimeError,
    AgentRuntimeErrorCode,
)

# Marker emitted by provider request-conversion code (langchain_aws Bedrock
# Converse) when an attachment's MIME type is not accepted. Matched across the
# exception chain — we react to the provider's verdict, we do not maintain our
# own list of supported types.
_UNSUPPORTED_MIME_MARKER = "Unsupported MIME type"


def raise_for_unsupported_attachment(exc: BaseException) -> None:
    """Translate a provider 'unsupported MIME type' rejection into a USER error.

    Walks the exception's ``__cause__``/``__context__`` chain for the provider
    marker. If found, raises a USER-categorized :class:`AgentRuntimeError` chained
    from the original. No-op otherwise, so callers fall through to their normal
    error handling.

    Args:
        exc: The exception raised by the model invocation.

    Raises:
        AgentRuntimeError: USER-categorized ``FILE_ERROR`` when the chain carries
            an unsupported-MIME-type provider rejection.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ValueError) and _UNSUPPORTED_MIME_MARKER in str(current):
            raise AgentRuntimeError(
                code=AgentRuntimeErrorCode.FILE_ERROR,
                title="Unsupported file attachment format.",
                detail=(
                    "An attachment has a file type the model does not support. "
                    "Remove the attachment or convert it to a supported format. "
                    f"Provider detail: {current}"
                ),
                category=UiPathErrorCategory.USER,
            ) from exc
        current = current.__cause__ or current.__context__
