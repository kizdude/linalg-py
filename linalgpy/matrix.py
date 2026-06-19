"""A Pythonic Matrix class wrapping the linalg C ``Mat *`` handle.

This is the layer you implement. The C library does the math; this class makes
it feel like Python: construction from lists/numpy, indexing, operators, and
automatic cleanup.

THE OWNERSHIP RULE (read this first):
  Every C ``Mat *`` is owned by exactly ONE Matrix instance, which frees it in
  __del__. C functions like mat_mul return a brand-new Mat* — wrap each one in
  its own Matrix via Matrix._wrap so it gets freed exactly once. Never wrap the
  same handle twice (double-free) and never use a Matrix after it's gone.
"""
from __future__ import annotations

from ctypes import c_double, POINTER, cast

import numpy as np

from ._loader import lib
from . import _cdefs  # noqa: F401  (ensures signatures are registered)


class LinalgError(Exception):
    """Raised when a C call fails (NULL return: bad shapes, singular, OOM)."""


class Matrix:
    # ------------------------------------------------------------------ construct
    def __init__(self, data):
        """Build a Matrix from a 2D sequence (list of lists) or numpy array."""
        a = np.asarray(data, dtype=float)
        rows, cols = a.shape
        flat = a.flatten().tolist()
        arr = (c_double * (rows * cols))(*flat)
        self._handle = lib.mat_from_array(rows, cols, arr)
        if not self._handle:
            raise LinalgError("Failed to create matrix")

    @classmethod
    def _wrap(cls, handle):
        """Wrap an existing C Mat* handle (from a C call) in a Matrix."""
        if not handle:
            raise LinalgError("C function returned NULL")
        obj = cls.__new__(cls)
        obj._handle = handle
        return obj

    def __del__(self):
        if hasattr(self, "_handle") and self._handle:
            lib.mat_free(self._handle)

    # ------------------------------------------------------------------ introspect
    @property
    def rows(self) -> int:
        return lib.mat_rows(self._handle)

    @property
    def cols(self) -> int:
        return lib.mat_cols(self._handle)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)

    def __getitem__(self, ij):
        i, j = ij
        return lib.mat_get(self._handle, i, j)

    def __setitem__(self, ij, value):
        i, j = ij
        lib.mat_set(self._handle, i, j, value)

    # ------------------------------------------------------------------ operators
    def __add__(self, other):
        return Matrix._wrap(lib.mat_add(self._handle, other._handle))

    def __sub__(self, other):
        return Matrix._wrap(lib.mat_sub(self._handle, other._handle))

    def __matmul__(self, other):
        return Matrix._wrap(lib.mat_mul(self._handle, other._handle))

    def __mul__(self, scalar):
        return Matrix._wrap(lib.mat_scale(self._handle, scalar))

    @property
    def T(self):
        return Matrix._wrap(lib.mat_transpose(self._handle))

    def __eq__(self, other):
       if not isinstance(other, Matrix):
            return NotImplemented
       return bool(lib.mat_equal(self._handle, other._handle, 1e-9))

    # ------------------------------------------------------------------ linear algebra
    def determinant(self) -> float:
        return lib.mat_determinant(self._handle)

    def inverse(self) -> "Matrix":
        return Matrix._wrap(lib.mat_inverse(self._handle))

    def solve(self, b: "Matrix") -> "Matrix":
        return Matrix._wrap(lib.mat_solve(self._handle, b._handle))

    # ------------------------------------------------------------------ construct helpers
    @classmethod
    def zeros(cls, rows: int, cols: int) -> "Matrix":
        return cls._wrap(lib.mat_zeros(rows, cols))

    @classmethod
    def identity(cls, n: int) -> "Matrix":
        return cls._wrap(lib.mat_identity(n))

    # ------------------------------------------------------------------ numpy interop
    def to_numpy(self):
        r, c = self.rows, self.cols
        buf = (c_double * (r * c))()
        lib.mat_to_array(self._handle, buf)
        return np.array(buf, dtype=float).reshape((r, c))

    @classmethod
    def from_numpy(cls, arr) -> "Matrix":
        return cls(arr)

    # ------------------------------------------------------------------ display
    def __repr__(self) -> str:
        return f"Matrix({self.to_numpy().tolist()})"
