"""deepagents file backend rooted in the host-managed workspace."""

import logging
import tempfile
import threading
from pathlib import Path

from deepagents.backends import FilesystemBackend
from uipath.runtime import get_workspace_path
from uipath.runtime.errors import UiPathRuntimeError

logger = logging.getLogger(__name__)


class UiPathWorkspaceBackend(FilesystemBackend):
    """File backend that reads and writes the workspace of the current run.

    The root folder is looked up on every file operation, so one backend
    created when the graph is built follows the workspace of whichever run is
    executing. In a UiPath job the host persists that folder across suspend
    and resume. Outside a managed run the backend uses one temporary folder,
    which is not persisted.

    Args:
        virtual_mode: Resolve agent paths under the workspace root and block
            traversal outside it.
        max_file_size_mb: Largest file the grep fallback reads.
    """

    def __init__(self, *, virtual_mode: bool = True, max_file_size_mb: int = 10):
        self._fallback_root: Path | None = None
        self._fallback_lock = threading.Lock()
        super().__init__(virtual_mode=virtual_mode, max_file_size_mb=max_file_size_mb)

    @property
    def cwd(self) -> Path:
        """The current run's workspace, or the fallback folder outside one."""
        try:
            return get_workspace_path().resolve()
        except UiPathRuntimeError:
            return self._fallback()

    @cwd.setter
    def cwd(self, value: Path) -> None:
        """Ignore assignments; the root always comes from the current run."""

    def _fallback(self) -> Path:
        with self._fallback_lock:
            if self._fallback_root is None:
                self._fallback_root = Path(
                    tempfile.mkdtemp(prefix="uipath-workspace-")
                ).resolve()
                logger.warning(
                    "No managed workspace in this run. Agent files go to %s "
                    "and are not kept across suspend and resume.",
                    self._fallback_root,
                )
            return self._fallback_root
