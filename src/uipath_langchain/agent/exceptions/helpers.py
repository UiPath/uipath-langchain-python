"""Helpers for raising structured errors from HTTP exceptions."""

from collections import defaultdict

from uipath.platform.errors import EnrichedException
from uipath.runtime.errors import UiPathErrorCategory

from .exceptions import AgentRuntimeError, AgentRuntimeErrorCode


def raise_for_enriched(
    e: EnrichedException,
    known_errors: dict[tuple[int, str | None], tuple[str, UiPathErrorCategory]],
    **context: str,
) -> None:
    """Raise AgentRuntimeError if the exception matches a known error pattern.

    Matches on ``(status_code, error_code)`` pairs. Use ``None`` as error_code
    to match any error with that status code. More specific matches (with
    error_code) are tried first.

    Each value is a ``(template, category)`` pair. Message templates can use
    ``{keyword}`` placeholders filled from *context*, plus ``{message}`` for
    the server's own error message.

    Does nothing if no match is found — caller should re-raise the original.

    Example::

        try:
            await client.processes.invoke_async(name=name, folder_path=folder)
        except EnrichedException as e:
            raise_for_enriched(
                e,
                {
                    (404, "1002"): ("Process '{process}' not found.", UiPathErrorCategory.USER),
                    (409, None): ("Conflict: {message}", UiPathErrorCategory.DEPLOYMENT),
                },
                process=name,
            )
            raise
    """
    info = e.error_info
    error_code = info.error_code if info else None
    server_message = (info.message if info else None) or ""
    context["message"] = server_message

    # Try specific match first, then wildcard
    entry = known_errors.get((e.status_code, error_code))
    if entry is None:
        entry = known_errors.get((e.status_code, None))
    if entry is None:
        return

    template, category = entry
    detail = template.format_map(defaultdict(lambda: "<unknown>", context))
    raise AgentRuntimeError(
        code=AgentRuntimeErrorCode.HTTP_ERROR,
        title=detail,
        detail=detail,
        category=category,
        status=e.status_code,
        should_wrap=False,
    ) from e
