"""linalgpy - Python bindings for the linalg C library (via ctypes)."""
from __future__ import annotations

from ._loader import lib
from . import _cdefs  # noqa: F401  (registers ctypes signatures as a side effect)


def version() -> str:
    """Return the C library's version string."""
    return lib.linalg_version().decode("ascii")


__all__ = ["version", "Matrix"]


def __getattr__(name: str):
    # Lazy import so the package still loads (for version()) even while
    # matrix.py is a work in progress.
    if name == "Matrix":
        from .matrix import Matrix

        return Matrix
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
