"""ctypes signature declarations for the linalg C API.

ctypes needs to be told each function's argument and return types; otherwise it
assumes everything is a C int, which silently corrupts pointers and doubles.
We declare them once here against the shared `lib` handle, then the Pythonic
wrappers (matrix.py) just call e.g. ``lib.mat_mul(a, b)``.

The opaque ``Mat *`` handle is represented as ``c_void_p`` — Python never needs
to see inside the struct.
"""
from __future__ import annotations

import ctypes
from ctypes import c_void_p, c_int, c_double, c_char_p, POINTER

from ._loader import lib

# Mat* is an opaque pointer.
MatPtr = c_void_p
_dbl_p = POINTER(c_double)


def _declare(name: str, restype, argtypes) -> None:
    fn = getattr(lib, name)
    fn.restype = restype
    fn.argtypes = argtypes


# --- meta ---
_declare("linalg_version", c_char_p, [])

# --- lifecycle / construction ---
_declare("mat_create", MatPtr, [c_int, c_int])
_declare("mat_zeros", MatPtr, [c_int, c_int])
_declare("mat_identity", MatPtr, [c_int])
_declare("mat_from_array", MatPtr, [c_int, c_int, _dbl_p])
_declare("mat_copy", MatPtr, [MatPtr])
_declare("mat_free", None, [MatPtr])

# --- introspection ---
_declare("mat_rows", c_int, [MatPtr])
_declare("mat_cols", c_int, [MatPtr])
_declare("mat_get", c_double, [MatPtr, c_int, c_int])
_declare("mat_get_col", MatPtr, [MatPtr, c_int])
_declare("mat_set", None, [MatPtr, c_int, c_int, c_double])
_declare("mat_to_array", None, [MatPtr, _dbl_p])

# --- arithmetic (return new Mat*) ---
_declare("mat_add", MatPtr, [MatPtr, MatPtr])
_declare("mat_sub", MatPtr, [MatPtr, MatPtr])
_declare("mat_scale", MatPtr, [MatPtr, c_double])
_declare("mat_mul", MatPtr, [MatPtr, MatPtr])
_declare("mat_transpose", MatPtr, [MatPtr])
_declare("mat_equal", c_int, [MatPtr, MatPtr, c_double])

# --- decompositions / solvers ---
_declare("mat_determinant", c_double, [MatPtr])
_declare("mat_inverse", MatPtr, [MatPtr])
_declare("mat_solve", MatPtr, [MatPtr, MatPtr])
_declare("mat_norm", c_double, [MatPtr])
_declare("mat_dot", c_double, [MatPtr, MatPtr, c_int])
# QR uses output pointers: int mat_qr(const Mat*, Mat** Q, Mat** R)
_declare("mat_qr", c_int, [MatPtr, POINTER(MatPtr), POINTER(MatPtr)])
