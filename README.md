# linalgpy

Python bindings for the [linalg](../linalg) C library, via **ctypes** FFI. The C
library does the math; `linalgpy` wraps it in a Pythonic `Matrix` class with
construction from lists/numpy, operator overloading, and automatic cleanup.

The C library is bundled as a git submodule under `external/linalg` and compiled
to a shared library that ctypes loads at runtime.

## Setup

```sh
# 1. get the submodule (if you cloned without --recursive)
git submodule update --init --recursive

# 2. create a virtual environment and install
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"   # Windows
# source .venv/bin/activate on macOS/Linux

# 3. build the bundled C library -> external/linalg/build/bin/liblinalg.dll
python build_lib.py
```

## Use

```python
from linalgpy import Matrix

A = Matrix([[4, 7], [2, 6]])
print(A.inverse().to_numpy())
print((A @ A).determinant())
```

## Test

```sh
python -m pytest
```

Tests cross-check every operation against numpy.

## Layout

```
linalgpy/
  _loader.py   locates + loads the shared library
  _cdefs.py    ctypes argtype/restype declarations for the C API
  matrix.py    the Pythonic Matrix wrapper
build_lib.py   compiles the submodule via CMake
external/linalg  the C library (git submodule)
```
