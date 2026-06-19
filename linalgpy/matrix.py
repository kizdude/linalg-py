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

from ._loader import lib
from . import _cdefs  # noqa: F401  (ensures signatures are registered)


class LinalgError(Exception):
    """Raised when a C call fails (NULL return: bad shapes, singular, OOM)."""


class Matrix:
    # ------------------------------------------------------------------ construct
    def __init__(self, data):
        """Build a Matrix from a 2D sequence (list of lists) or numpy array.

        TODO:
          - figure out rows/cols from `data`
          - flatten the values row-major into a (c_double * (rows*cols)) array
          - call lib.mat_from_array(rows, cols, arr); store the handle
          - raise LinalgError if the C call returns NULL (falsy c_void_p)
        """
        raise NotImplementedError

    @classmethod
    def _wrap(cls, handle):
        """Wrap an existing C Mat* handle (from a C call) in a Matrix.

        TODO:
          - if handle is NULL/falsy -> raise LinalgError
          - create an instance WITHOUT calling __init__ (cls.__new__(cls)),
            set its ._handle, and return it
        This is how every operation result becomes a managed Matrix.
        """
        raise NotImplementedError

    def __del__(self):
        # TODO: free the C matrix exactly once (guard: it may not exist if
        # __init__ raised before assigning ._handle).
        ...

    # ------------------------------------------------------------------ introspect
    @property
    def rows(self) -> int:
        raise NotImplementedError

    @property
    def cols(self) -> int:
        raise NotImplementedError

    @property
    def shape(self) -> tuple[int, int]:
        raise NotImplementedError

    def __getitem__(self, ij):
        # ij is a tuple (i, j). TODO: return lib.mat_get(...)
        raise NotImplementedError

    def __setitem__(self, ij, value):
        # TODO: lib.mat_set(...)
        raise NotImplementedError

    # ------------------------------------------------------------------ operators
    def __add__(self, other):
        raise NotImplementedError

    def __sub__(self, other):
        raise NotImplementedError

    def __matmul__(self, other):
        # the @ operator -> mat_mul
        raise NotImplementedError

    def __mul__(self, scalar):
        # scalar multiply -> mat_scale
        raise NotImplementedError

    @property
    def T(self):
        # transpose
        raise NotImplementedError

    def __eq__(self, other):
        # use mat_equal with a small tolerance
        raise NotImplementedError

    # ------------------------------------------------------------------ linear algebra
    def determinant(self) -> float:
        raise NotImplementedError

    def inverse(self) -> "Matrix":
        raise NotImplementedError

    def solve(self, b: "Matrix") -> "Matrix":
        raise NotImplementedError

    # ------------------------------------------------------------------ construct helpers
    @classmethod
    def zeros(cls, rows: int, cols: int) -> "Matrix":
        raise NotImplementedError

    @classmethod
    def identity(cls, n: int) -> "Matrix":
        raise NotImplementedError

    # ------------------------------------------------------------------ numpy interop
    def to_numpy(self):
        # TODO: read rows*cols doubles out with mat_to_array, reshape
        raise NotImplementedError

    @classmethod
    def from_numpy(cls, arr) -> "Matrix":
        raise NotImplementedError

    # ------------------------------------------------------------------ display
    def __repr__(self) -> str:
        raise NotImplementedError
