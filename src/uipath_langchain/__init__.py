from ._utils import (
    _compat as _,  # noqa: F401 — installs sys.unraisablehook shim on Python < 3.12
)
from .middlewares import register_middleware  # noqa: E402

__all__ = ["register_middleware"]
