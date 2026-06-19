"""Demo app for the linalgpy binding.

Two parts:
  1. numeric_demo()  - solve a system, invert, cross-check against numpy
  2. visualize()     - show how a 2x2 matrix transforms the plane (unit circle,
                       grid, and basis vectors), computing the transform through
                       the C library via your Matrix class.

Run:  .venv/Scripts/python examples/demo.py
"""
from __future__ import annotations

import numpy as np

from linalgpy import Matrix


# ---------------------------------------------------------------- part 1: numbers
def numeric_demo() -> None:
    print("=== numeric demo ===")
    A = Matrix([[3.0, 2.0], [1.0, 2.0]])
    b = Matrix([[5.0], [5.0]])

    print(f"detA = {A.determinant()}")
    Ainv = A.inverse()
    print(f"Ainv =\n{Ainv.to_numpy()}")
    print(f"(A @ Ainv) =\n{(A @ Ainv).to_numpy()}")
    x = A.solve(b)
    print(f"x =\n{x.to_numpy()}")
    print(f"(A @ x) =\n{(A @ x).to_numpy()}")
    det = A.determinant()
    Ainv_np = np.linalg.inv(A.to_numpy())
    x_np = np.linalg.solve(A.to_numpy(), b.to_numpy())
    print(f"det = {det:.3f}  (numpy: {np.linalg.det(A.to_numpy()):.3f})")
    print(f"Ainv =\n{Ainv.to_numpy()}\n(numpy:\n{Ainv_np})")
    print(f"x =\n{x.to_numpy()}\n(numpy:\n{x_np})")


# ------------------------------------------------------------ part 2: transform
def transform_points(M: Matrix, pts: np.ndarray) -> np.ndarray:
    """Apply 2x2 transform M to a 2xN array of column points; return 2xN."""
    result = M @ Matrix.from_numpy(pts)
    return result.to_numpy()


def _unit_shapes():
    """The things we'll draw: a unit circle and the two basis vectors."""
    t = np.linspace(0, 2 * np.pi, 200)
    circle = np.vstack([np.cos(t), np.sin(t)])          # 2 x 200
    basis = np.array([[1.0, 0.0], [0.0, 1.0]])          # columns e1, e2
    return circle, basis


def visualize() -> None:
    import matplotlib.pyplot as plt

    # A transform with rotation + shear + scale, defined as a Matrix.
    M = Matrix([[1.0, 0.8], [0.3, 1.2]])
    circle, basis = _unit_shapes()

    tcircle = transform_points(M, circle)
    tbasis = transform_points(M, basis)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 5))
    for ax, (c, bvec), title in (
        (ax0, (circle, basis), "before"),
        (ax1, (tcircle, tbasis), "after  (M applied)"),
    ):
        ax.plot(c[0], c[1], color="tab:blue")
        ax.quiver([0, 0], [0, 0], bvec[0], bvec[1], angles="xy",
                  scale_units="xy", scale=1, color=["tab:red", "tab:green"])
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)

    fig.suptitle(f"Linear transform  M = {M.to_numpy().tolist()}  (det = {M.determinant():.3f})")
    fig.tight_layout()
    out = "transform.png"
    fig.savefig(out, dpi=110)
    print(f"saved visualization -> {out}")


if __name__ == "__main__":
    numeric_demo()
    visualize()
