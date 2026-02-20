def __getattr__(name):
    if name == "AnalyzeAttachmentsTool":
        from .attachments import AnalyzeAttachmentsTool

        return AnalyzeAttachmentsTool
    if name == "requires_approval":
        from .hitl import requires_approval

        return requires_approval
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AnalyzeAttachmentsTool",
    "requires_approval",
]
