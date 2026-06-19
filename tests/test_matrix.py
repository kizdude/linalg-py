"""Spec for the Matrix wrapper. Results are cross-checked against numpy.

Run with:  python -m pytest   (after: python build_lib.py)
"""
import numpy as np
import pytest

from linalgpy import Matrix
from linalgpy.matrix import LinalgError


def test_construct_and_shape():
    m = Matrix([[1, 2, 3], [4, 5, 6]])
    assert m.rows == 2
    assert m.cols == 3
    assert m.shape == (2, 3)


def test_indexing():
    m = Matrix([[1, 2], [3, 4]])
    assert m[0, 0] == 1
    assert m[1, 0] == 3
    m[1, 0] = 9
    assert m[1, 0] == 9


def test_add_sub_scale():
    a = Matrix([[1, 2], [3, 4]])
    b = Matrix([[10, 20], [30, 40]])
    assert (a + b).to_numpy().tolist() == [[11, 22], [33, 44]]
    assert (b - a).to_numpy().tolist() == [[9, 18], [27, 36]]
    assert (a * 2).to_numpy().tolist() == [[2, 4], [6, 8]]


def test_matmul_against_numpy():
    A = np.array([[1.0, 2, 3], [4, 5, 6]])
    B = np.array([[7.0, 8], [9, 10], [11, 12]])
    got = (Matrix(A) @ Matrix(B)).to_numpy()
    assert np.allclose(got, A @ B)


def test_transpose():
    m = Matrix([[1, 2, 3], [4, 5, 6]])
    assert m.T.to_numpy().tolist() == [[1, 4], [2, 5], [3, 6]]


def test_equality():
    assert Matrix([[1, 2], [3, 4]]) == Matrix([[1, 2], [3, 4]])
    assert Matrix([[1, 2], [3, 4]]) != Matrix([[1, 2], [3, 5]])


def test_identity_zeros():
    assert Matrix.identity(3).to_numpy().tolist() == np.eye(3).tolist()
    assert Matrix.zeros(2, 2).to_numpy().tolist() == [[0, 0], [0, 0]]


def test_determinant():
    A = np.array([[4.0, 3], [6, 3]])
    assert Matrix(A).determinant() == pytest.approx(np.linalg.det(A))


def test_inverse_against_numpy():
    A = np.array([[4.0, 7], [2, 6]])
    inv = Matrix(A).inverse().to_numpy()
    assert np.allclose(inv, np.linalg.inv(A))


def test_solve():
    A = np.array([[3.0, 2], [1, 2]])
    b = np.array([[5.0], [5]])
    x = Matrix(A).solve(Matrix(b)).to_numpy()
    assert np.allclose(A @ x, b)


def test_numpy_roundtrip():
    arr = np.array([[1.5, 2.5], [3.5, 4.5]])
    assert np.allclose(Matrix.from_numpy(arr).to_numpy(), arr)


def test_shape_mismatch_raises():
    a = Matrix([[1, 2, 3]])
    b = Matrix([[1, 2]])
    with pytest.raises(LinalgError):
        _ = a + b


def test_singular_inverse_raises():
    with pytest.raises(LinalgError):
        Matrix([[1, 2], [2, 4]]).inverse()
