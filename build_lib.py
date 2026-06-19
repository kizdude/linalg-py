"""Build the bundled linalg C library (the git submodule) into a shared library.

Run this once after cloning (and after pulling new library changes):

    python build_lib.py

It invokes CMake on external/linalg and leaves the compiled liblinalg.dll
(/.so/.dylib) in external/linalg/build/bin/, where the ctypes loader looks for
it. Requires CMake and a C compiler (GCC/MinGW on Windows) on PATH.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LIB_SRC = ROOT / "external" / "linalg"
BUILD_DIR = LIB_SRC / "build"


def _have(tool: str) -> bool:
    return shutil.which(tool) is not None


def main() -> int:
    if not (LIB_SRC / "CMakeLists.txt").exists():
        print(
            "error: external/linalg is empty. Initialise the submodule first:\n"
            "    git submodule update --init --recursive",
            file=sys.stderr,
        )
        return 1
    if not _have("cmake"):
        print("error: cmake not found on PATH", file=sys.stderr)
        return 1

    # Prefer MinGW Makefiles on Windows (no MSVC in this toolchain); elsewhere
    # let CMake pick its default generator.
    configure = ["cmake", "-S", str(LIB_SRC), "-B", str(BUILD_DIR)]
    if sys.platform.startswith("win") and _have("mingw32-make"):
        configure += ["-G", "MinGW Makefiles"]

    print("==> configuring:", " ".join(configure))
    subprocess.run(configure, check=True)

    build = ["cmake", "--build", str(BUILD_DIR)]
    print("==> building:", " ".join(build))
    subprocess.run(build, check=True)

    out = BUILD_DIR / "bin"
    libs = list(out.glob("*linalg*.dll")) + list(out.glob("*linalg*.so")) + list(
        out.glob("*linalg*.dylib")
    )
    if not libs:
        print(f"error: build finished but no shared library found in {out}", file=sys.stderr)
        return 1
    print("==> built:", libs[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
