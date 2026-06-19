"""Locate and load the compiled linalg shared library via ctypes."""
from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
_BUILD_BIN = _PKG_ROOT / "external" / "linalg" / "build" / "bin"


def _candidates() -> list[Path]:
    """Possible locations of the shared library, most specific first."""
    # 1. explicit override
    env = os.environ.get("LINALG_DLL")
    names = ["liblinalg.dll", "linalg.dll", "liblinalg.so", "liblinalg.dylib"]
    found: list[Path] = []
    if env:
        found.append(Path(env))
    # 2. the build output directory
    for name in names:
        found.append(_BUILD_BIN / name)
    return found


def load_library() -> ctypes.CDLL:
    """Return the loaded linalg CDLL, or raise a helpful error."""
    for path in _candidates():
        if path.is_file():
            # On Windows, make sure the directory is searched for any
            # dependent DLLs (libgcc, etc. from the MinGW runtime).
            if sys.platform.startswith("win"):
                os.add_dll_directory(str(path.parent))
            return ctypes.CDLL(str(path))

    searched = "\n  ".join(str(p) for p in _candidates())
    raise OSError(
        "Could not find the compiled linalg library. Build it first with:\n"
        "    python build_lib.py\n"
        "Searched:\n  " + searched
    )


# A single shared handle for the whole package.
lib = load_library()
