"""Application1 deterministic Hessian-vector products for Gaussian source controls."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import run_optimization as app1
from multi_source_utils import (
    source_derivatives_from_control,
    source_directional_derivative_from_control,
    source_hessian_directional_from_control,
)


@dataclass(frozen=True)
class HessianVectorDetails:
    """Intermediate fields used to assemble a Hessian-vector product."""

    base_state: np.ndarray
    base_source_adjoint: np.ndarray
    tangent_state: np.ndarray
    incremental_source_adjoint: np.ndarray


def _source_adjoint_for_residual(
    state_out: np.ndarray,
    target_state: np.ndarray,
    mesh_data,
    pde: dict,
    eta1: float,
) -> np.ndarray:
    coord = np.asarray(mesh_data["coord"], dtype=np.float64)
    connections = np.asarray(mesh_data["connections"], dtype=int)
    return app1.gt.rk4_discrete_source_adjoint(
        npoin=coord.shape[0],
        nelem=connections.shape[0],
        ngauss=pde["ngauss"],
        lx=pde["lx"],
        ly=pde["ly"],
        k=pde["k"],
        vx=pde["vx"],
        vy=pde["vy"],
        dt=pde["dt"],
        tfinal=pde["tfinal"],
        jprint=pde["jprint"],
        coord=coord,
        connections=connections,
        state_out=state_out,
        ud=target_state,
        eta1=eta1,
        st_ngauss=3,
        mesh_data=mesh_data,
        dirichlet_tags=["left"],
        measurement_boundary=mesh_data.get("measurement_boundary", "right"),
        velocity_nodes=pde.get("velocity_nodes"),
        objective_time_rule=pde.get("objective_time_rule", "exclude_first"),
    )


def _effective_load_vector(mesh_data, pde: dict, scalar_func) -> np.ndarray:
    coord = np.asarray(mesh_data["coord"], dtype=np.float64)
    connections = np.asarray(mesh_data["connections"], dtype=int)
    load_vector = app1.gt.assemble_load_vector(
        npoin=coord.shape[0],
        nelem=connections.shape[0],
        coord=coord,
        connections=connections,
        ngauss=pde["ngauss"],
        scalar_func=scalar_func,
    )
    mass_lumped = app1.gt.assemble_lumped_mass(
        npoin=coord.shape[0],
        nelem=connections.shape[0],
        coord=coord,
        connections=connections,
        ngauss=pde["ngauss"],
    )
    dirichlet_nodes = np.asarray(
        mesh_data["boundary_nodes_by_name"].get("left", np.array([], dtype=int)),
        dtype=int,
    )
    mass_lumped[dirichlet_nodes] = 1.0
    if np.any(mass_lumped <= 0.0):
        raise ValueError("Non-positive lumped mass encountered in Hessian assembly.")
    return load_vector / mass_lumped


def _assemble_source_gradient_from_adjoint(
    source_adjoint: np.ndarray,
    derivative_specs: list[dict],
    mesh_data,
    pde: dict,
) -> np.ndarray:
    values = []
    for spec in derivative_specs:
        effective_load = _effective_load_vector(mesh_data, pde, spec["func"])
        values.append(float(np.dot(source_adjoint, effective_load)))
    return np.asarray(values, dtype=np.float64)


def tangent_forward_solve(
    z: np.ndarray,
    direction: np.ndarray,
    mesh_data,
    pde: dict,
) -> np.ndarray:
    """Solve the RK4-consistent tangent forward problem for S_z(z)[direction]."""
    tangent_source = source_directional_derivative_from_control(
        z, direction, r2=pde["sigma2"]
    )
    return app1.gt.run_forward(mesh_data, pde, z, source_func=tangent_source)


def hessian_vector_product(
    z: np.ndarray,
    direction: np.ndarray,
    mesh_data,
    pde: dict,
    target_state: np.ndarray,
    *,
    eta1: float = app1.ETA1,
    lam: float = app1.LAMBDA_REG,
    mode: str = "gauss_newton",
    base_state: np.ndarray | None = None,
    base_source_adjoint: np.ndarray | None = None,
    return_details: bool = False,
):
    """Return H(z) direction for the deterministic Application1 reduced objective.

    ``mode="gauss_newton"`` differentiates only the residual-through-state
    term.  ``mode="exact"`` additionally includes the base-adjoint contraction
    with D^2 S(z)[direction, e_j], matching finite differences of the current
    RK4-discrete-adjoint gradient.
    """
    z = np.asarray(z, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    # Any M-source control [x1, y1, A1, ..., xM, yM, AM] (length 3M).
    if z.size == 0 or z.size % 3 or direction.size != z.size:
        raise ValueError(
            "Expected z of length 3M and a direction of the same length, got "
            f"{z.size} and {direction.size}."
        )
    if mode not in {"gauss_newton", "exact"}:
        raise ValueError("mode must be 'gauss_newton' or 'exact'")

    if base_state is None:
        base_state = app1.gt.run_forward(mesh_data, pde, z)
    if base_source_adjoint is None and mode == "exact":
        base_source_adjoint = _source_adjoint_for_residual(
            base_state, target_state, mesh_data, pde, eta1
        )

    tangent_state = tangent_forward_solve(z, direction, mesh_data, pde)
    zero_target = np.zeros_like(tangent_state)
    incremental_source_adjoint = _source_adjoint_for_residual(
        tangent_state, zero_target, mesh_data, pde, eta1
    )

    first_derivatives = source_derivatives_from_control(z, r2=pde["sigma2"])
    hv = _assemble_source_gradient_from_adjoint(
        incremental_source_adjoint, first_derivatives, mesh_data, pde
    )
    if mode == "exact":
        second_derivatives = source_hessian_directional_from_control(
            z, direction, r2=pde["sigma2"]
        )
        hv += _assemble_source_gradient_from_adjoint(
            base_source_adjoint, second_derivatives, mesh_data, pde
        )
    if lam:
        hv += float(lam) * direction

    if return_details:
        details = HessianVectorDetails(
            base_state=np.asarray(base_state, dtype=np.float64),
            base_source_adjoint=(
                np.zeros_like(incremental_source_adjoint)
                if base_source_adjoint is None
                else np.asarray(base_source_adjoint, dtype=np.float64)
            ),
            tangent_state=tangent_state,
            incremental_source_adjoint=incremental_source_adjoint,
        )
        return hv, details
    return hv


def hessian_matrix(
    z: np.ndarray,
    mesh_data,
    pde: dict,
    target_state: np.ndarray,
    *,
    eta1: float = app1.ETA1,
    lam: float = app1.LAMBDA_REG,
    mode: str = "gauss_newton",
) -> np.ndarray:
    """Fill the dense deterministic Hessian by coordinate HVPs."""
    z = np.asarray(z, dtype=np.float64)
    dim = z.size
    base_state = app1.gt.run_forward(mesh_data, pde, z)
    base_source_adjoint = None
    if mode == "exact":
        base_source_adjoint = _source_adjoint_for_residual(
            base_state, target_state, mesh_data, pde, eta1
        )

    columns = []
    for column_index in range(dim):
        direction = np.zeros(dim, dtype=np.float64)
        direction[column_index] = 1.0
        columns.append(
            hessian_vector_product(
                z,
                direction,
                mesh_data,
                pde,
                target_state,
                eta1=eta1,
                lam=lam,
                mode=mode,
                base_state=base_state,
                base_source_adjoint=base_source_adjoint,
            )
        )
    return np.column_stack(columns)


__all__ = [
    "HessianVectorDetails",
    "hessian_matrix",
    "hessian_vector_product",
    "tangent_forward_solve",
]
