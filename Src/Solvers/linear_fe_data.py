"""Linear (P1) triangle finite-element helper data and mappings."""

import numpy as np


def shape_functions(u, v):
    """Reference-triangle shape functions at (u, v)."""
    return np.array([1.0 - u - v, u, v], dtype=float)


def shape_function_derivatives():
    """Reference derivatives dN/du, dN/dv for P1 triangle (constant)."""
    return np.array(
        [
            [-1.0, -1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )


def jacobian(xcoord, ycoord):
    """2x2 Jacobian from local triangle coordinates."""
    dN = shape_function_derivatives()
    J = np.zeros((2, 2), dtype=float)
    for a in range(3):
        J[0, 0] += dN[a, 0] * xcoord[a]
        J[0, 1] += dN[a, 0] * ycoord[a]
        J[1, 0] += dN[a, 1] * xcoord[a]
        J[1, 1] += dN[a, 1] * ycoord[a]
    return J


def jacobian_det(J):
    """Determinant of Jacobian."""
    return float(np.linalg.det(J))


def jacobian_inv(J):
    """Inverse Jacobian with simple singularity check."""
    det = jacobian_det(J)
    if abs(det) < 1e-14:
        raise ValueError("Jacobian is singular or near-singular.")
    return np.linalg.inv(J)
