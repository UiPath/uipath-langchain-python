"""Services for the LTL Claims Agent System."""

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name == "UiPathService":
        from .uipath_service import UiPathService
        return UiPathService
    elif name == "UiPathServiceError":
        from .uipath_service import UiPathServiceError
        return UiPathServiceError
    elif name == "uipath_service":
        from .uipath_service import uipath_service
        return uipath_service
    elif name == "ProcessingHistoryService":
        from .processing_history_service import ProcessingHistoryService
        return ProcessingHistoryService
    elif name == "ProcessingHistoryServiceError":
        from .processing_history_service import ProcessingHistoryServiceError
        return ProcessingHistoryServiceError
    elif name == "InputManager":
        from .input_manager import InputManager
        return InputManager
    elif name == "QueueInputSource":
        from .input_manager import QueueInputSource
        return QueueInputSource
    elif name == "FileInputSource":
        from .input_manager import FileInputSource
        return FileInputSource
    elif name == "ClaimInput":
        from .input_manager import ClaimInput
        return ClaimInput
    elif name == "DocumentReference":
        from .input_manager import DocumentReference
        return DocumentReference
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    "UiPathService",
    "UiPathServiceError", 
    "uipath_service",
    "ProcessingHistoryService",
    "ProcessingHistoryServiceError",
    "InputManager",
    "QueueInputSource",
    "FileInputSource",
    "ClaimInput",
    "DocumentReference"
]