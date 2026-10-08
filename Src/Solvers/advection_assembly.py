"""Advection matrix assembly helpers for P1 triangular elements."""

from __future__ import annotations

import numpy as np

try:
    from scipy.sparse import coo_matrix
except Exception:  # optional dependency
    coo_matrix = None

from linear_fe_data import jacobian, jacobian_det, shape_function_derivatives


_NODAL_QUAD_POINTS = (
    (1.0 / 6.0, 1.0 / 6.0),
    (2.0 / 3.0, 1.0 / 6.0),
    (1.0 / 6.0, 2.0 / 3.0),
)
_NODAL_QUAD_WEIGHTS = (1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0)


def validate_velocity_nodes(velocity_nodes, num_nodes: int) -> np.ndarray:
    """Validate nodal P1 velocity data and return a float64 array."""
    if not isinstance(velocity_nodes, np.ndarray):
        raise ValueError("velocity_nodes must be a NumPy array.")

    velocity_nodes = np.asarray(velocity_nodes, dtype=np.float64)
    expected_shape = (int(num_nodes), 2)
    if velocity_nodes.shape != expected_shape:
        raise ValueError(
            "velocity_nodes must have shape "
            f"{expected_shape}, got {velocity_nodes.shape}."
        )
    if not np.all(np.isfinite(velocity_nodes)):
        raise ValueError("velocity_nodes must contain only finite values.")
    return velocity_nodes


def assemble_advection_nodal_quadrature(coords, elements, velocity_nodes):
    """Assemble C_ij = int phi_i (v_h dot grad phi_j) dx for nodal P1 velocity."""
    coords = np.asarray(coords, dtype=np.float64)
    elements = np.asarray(elements, dtype=int)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(
            f"coords must have shape (num_nodes, 2) or wider, got {coords.shape}."
        )
    coords = coords[:, :2]
    if elements.ndim != 2 or elements.shape[1] != 3:
        raise ValueError(
            f"elements must have shape (num_elements, 3), got {elements.shape}."
        )

    num_nodes = coords.shape[0]
    velocity_nodes = validate_velocity_nodes(velocity_nodes, num_nodes)
    dN_ref = shape_function_derivatives()

    rows = []
    cols = []
    data = []

    for elem_nodes in elements:
        xcoord = coords[elem_nodes, 0]
        ycoord = coords[elem_nodes, 1]
        J = jacobian(xcoord, ycoord)
        detJ = jacobian_det(J)
        if abs(detJ) < 1.0e-14:
            raise ValueError("Jacobian is singular or near-singular.")
        grad_phi = np.linalg.solve(J, dN_ref.T).T

        local_velocity = velocity_nodes[elem_nodes]
        clocal = np.zeros((3, 3), dtype=np.float64)
        for weight, (xi, eta) in zip(_NODAL_QUAD_WEIGHTS, _NODAL_QUAD_POINTS):
            N = np.array([1.0 - xi - eta, xi, eta], dtype=np.float64)
            v_q = N @ local_velocity
            adv_grad = grad_phi @ v_q
            clocal += weight * abs(detJ) * np.outer(N, adv_grad)

        for i_local, i_global in enumerate(elem_nodes):
            for j_local, j_global in enumerate(elem_nodes):
                rows.append(int(i_global))
                cols.append(int(j_global))
                data.append(float(clocal[i_local, j_local]))

    if coo_matrix is None:
        matrix = np.zeros((num_nodes, num_nodes), dtype=np.float64)
        np.add.at(matrix, (rows, cols), data)
        return matrix
    return coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()


def advection_matrix_to_dense(matrix) -> np.ndarray:
    """Return a dense ndarray from either SciPy sparse or ndarray assembly output."""
    if hasattr(matrix, "toarray"):
        return matrix.toarray()
    return np.asarray(matrix, dtype=np.float64)


__all__ = [
    "advection_matrix_to_dense",
    "assemble_advection_nodal_quadrature",
    "validate_velocity_nodes",
]
