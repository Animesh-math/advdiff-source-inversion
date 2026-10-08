"""Adjoint solver for 2D advection-diffusion with right-boundary tracking (eta2=0).

Assumptions:
- The forward state snapshots are provided at uniform times t_n = n * dt * jprint.
- Tracking boundary is selected by name from mesh_data (default: mesh_data["measurement_boundary"]).
- Alternatively, mesh_data["observation"] may carry a discrete observation
  operator (see observation_operator.py); it then supplies the adjoint forcing
  eta1 * C^T W (C U^n - d^n) through the exact transpose of its own operator,
  and the boundary path below is bypassed entirely.
- Desired boundary data u_d is either:
  (i) callable: u_d(x, y, t), or
  (ii) nodal array with same shape as state snapshots (nframes, npoin).
- Because eta2 = 0, the terminal adjoint condition is homogeneous: p(T) = 0.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

try:
    from scipy.linalg import lu_factor, lu_solve
except Exception:  # optional dependency
    lu_factor = None
    lu_solve = None

_THIS_FILE = Path(__file__).resolve()
_SRC_SOLVERS = _THIS_FILE.parent
if str(_SRC_SOLVERS) not in sys.path:
    sys.path.append(str(_SRC_SOLVERS))

from linear_fe_data import jacobian, jacobian_det, shape_function_derivatives, shape_functions
from pde_params import PDEParams
from quadrature_data import interval_quadrature_0_to_1, triangle_quadrature
from advection_assembly import (
    advection_matrix_to_dense,
    assemble_advection_nodal_quadrature,
)


def _dirichlet_nodes_from_mesh_data(mesh_data, dirichlet_tags):
    """Get adjoint Dirichlet nodes by named boundary tags from mesh_data."""
    nodes_by_name = mesh_data.get("boundary_nodes_by_name", {}) if mesh_data else {}
    if not dirichlet_tags:
        dirichlet_tags = ["left"]

    nodes = []
    for name in dirichlet_tags:
        tag_nodes = nodes_by_name.get(name)
        if tag_nodes is not None:
            nodes.extend(np.asarray(tag_nodes, dtype=int).tolist())
    if not nodes:
        raise ValueError(
            "No adjoint Dirichlet nodes found in mesh_data for requested tags: "
            f"{dirichlet_tags}."
        )
    return np.unique(np.asarray(nodes, dtype=int))


def _tracking_edges_from_mesh_data(mesh_data, measurement_boundary):
    """Get tracking boundary edges by boundary name from mesh_data."""
    edges_by_name = mesh_data.get("boundary_edges_by_name", {}) if mesh_data else {}
    if measurement_boundary is None:
        measurement_boundary = mesh_data.get("measurement_boundary", "right") if mesh_data else "right"

    edges = edges_by_name.get(measurement_boundary)
    if edges is None or len(edges) == 0:
        raise ValueError(
            f"No tracking edges found for measurement boundary '{measurement_boundary}'."
        )
    return np.asarray(edges, dtype=int)


def _assemble_primal_operator(npoin, nelem, coord, connections, params: PDEParams):
    npoin_per_elem = 3
    weight, xgauss, ygauss = triangle_quadrature(params.ngauss)
    dN = shape_function_derivatives()

    kmatrix = np.zeros((npoin, npoin), dtype=np.float64)
    nodal_advection = None
    if params.velocity_nodes is not None:
        nodal_advection = advection_matrix_to_dense(
            assemble_advection_nodal_quadrature(coord, connections, params.velocity_nodes)
        )

    for ielem in range(nelem):
        klocal = np.zeros((npoin_per_elem, npoin_per_elem), dtype=np.float64)

        elem_nodes = connections[ielem]
        xcoord = coord[elem_nodes, 0]
        ycoord = coord[elem_nodes, 1]

        J = jacobian(xcoord, ycoord)
        detJ = jacobian_det(J)

        for jpoint in range(npoin_per_elem):
            for ipoint in range(npoin_per_elem):
                for igauss in range(params.ngauss):
                    u = xgauss[igauss]
                    v = ygauss[igauss]

                    bj = J[1, 1] * dN[jpoint, 0] - J[0, 1] * dN[jpoint, 1]
                    bi = J[1, 1] * dN[ipoint, 0] - J[0, 1] * dN[ipoint, 1]
                    cj = J[0, 0] * dN[jpoint, 1] - J[1, 0] * dN[jpoint, 0]
                    ci = J[0, 0] * dN[ipoint, 1] - J[1, 0] * dN[ipoint, 0]

                    klocal[jpoint, ipoint] += (
                        params.k * (bi * bj + ci * cj) * weight[igauss] / detJ
                    )

                    nsf = shape_functions(u, v)
                    if nodal_advection is None:
                        klocal[jpoint, ipoint] += (
                            (params.vx * bi + params.vy * ci) * nsf[jpoint] * weight[igauss]
                        )

        for jpoint in range(npoin_per_elem):
            jglob = elem_nodes[jpoint]
            for ipoint in range(npoin_per_elem):
                iglob = elem_nodes[ipoint]
                kmatrix[jglob, iglob] += klocal[jpoint, ipoint]

    if nodal_advection is not None:
        kmatrix += nodal_advection

    return kmatrix


def _assemble_lumped_mass(npoin, nelem, coord, connections, ngauss):
    """Assemble lumped mass consistent with forward semi-discretization."""
    npoin_per_elem = 3
    weight, xgauss, ygauss = triangle_quadrature(ngauss)

    mass_lumped = np.zeros(npoin, dtype=np.float64)

    for ielem in range(nelem):
        mlocal = np.zeros((npoin_per_elem, npoin_per_elem), dtype=np.float64)

        elem_nodes = connections[ielem]
        xcoord = coord[elem_nodes, 0]
        ycoord = coord[elem_nodes, 1]

        J = jacobian(xcoord, ycoord)
        detJ = jacobian_det(J)

        for jpoint in range(npoin_per_elem):
            for ipoint in range(npoin_per_elem):
                for igauss in range(ngauss):
                    nsf = shape_functions(xgauss[igauss], ygauss[igauss])
                    mlocal[jpoint, ipoint] += (
                        nsf[jpoint] * nsf[ipoint] * weight[igauss] * detJ
                    )

        for jpoint in range(npoin_per_elem):
            jglob = elem_nodes[jpoint]
            for ipoint in range(npoin_per_elem):
                mass_lumped[jglob] += mlocal[jpoint, ipoint]

    return mass_lumped


def _apply_homogeneous_dirichlet_rows(matrix: np.ndarray, boundary_points: np.ndarray):
    for ipoin in boundary_points:
        matrix[ipoin, :] = 0.0
        matrix[ipoin, ipoin] = 1.0


def _evaluate_ud_on_edge(ud, a, b, na, xa, ya, xb, yb, s, time):
    if callable(ud):
        xq = (1.0 - s) * xa + s * xb
        yq = (1.0 - s) * ya + s * yb
        return float(ud(xq, yq, time))

    return float((1.0 - s) * ud[na, a] + s * ud[na, b])


def _boundary_tracking_rhs_all_times(
    state_out: np.ndarray,
    ud,
    times: np.ndarray,
    tracking_edges: np.ndarray,
    coord: np.ndarray,
    npoin: int,
    eta1: float,
    st_ngauss: int,
):
    st_weight, st_xgauss = interval_quadrature_0_to_1(st_ngauss)

    rhs_all = np.zeros((state_out.shape[0], npoin), dtype=np.float64)

    for na, time in enumerate(times):
        rhs = rhs_all[na]

        for edge in tracking_edges:
            a, b = int(edge[0]), int(edge[1])
            xa, ya = coord[a, 0], coord[a, 1]
            xb, yb = coord[b, 0], coord[b, 1]
            edge_len = float(np.hypot(xb - xa, yb - ya))

            for igauss, s in enumerate(st_xgauss):
                n1 = 1.0 - s
                n2 = s

                uq = float(n1 * state_out[na, a] + n2 * state_out[na, b])
                udq = _evaluate_ud_on_edge(ud, a, b, na, xa, ya, xb, yb, s, float(time))
                mismatch = uq - udq

                rhs[a] += eta1 * mismatch * n1 * st_weight[igauss] * edge_len
                rhs[b] += eta1 * mismatch * n2 * st_weight[igauss] * edge_len

    return rhs_all


def _observation_from_mesh_data(mesh_data):
    """Non-boundary observation operator wired into ``mesh_data``, if any.

    ``None`` selects the historical boundary tracking path, which is left
    untouched.  A node-sensor operator (``Src/Solvers/observation_operator.py``)
    supplies ``eta1 * C^T W (C U^n - d^n)`` directly, i.e. the exact transpose of
    its own observation operator.
    """
    if not isinstance(mesh_data, dict):
        return None
    return mesh_data.get("observation")


def _time_weights(nframes: int, dt_out: float, rule: str) -> np.ndarray:
    if rule == "exclude_first":
        weights = np.full(nframes, dt_out, dtype=np.float64)
        weights[0] = 0.0
        return weights
    if rule == "trapezoid":
        weights = np.full(nframes, dt_out, dtype=np.float64)
        weights[0] *= 0.5
        weights[-1] *= 0.5
        return weights
    raise ValueError(f"Unknown objective time rule: {rule}")


def _apply_rk4_transpose_polynomial(vector, apply_a_transpose, dt: float, source: bool):
    a1 = apply_a_transpose(vector)
    a2 = apply_a_transpose(a1)
    a3 = apply_a_transpose(a2)
    if source:
        return dt * (vector + 0.5 * dt * a1 + (dt**2 / 6.0) * a2 + (dt**3 / 24.0) * a3)
    a4 = apply_a_transpose(a3)
    return vector + dt * a1 + 0.5 * dt**2 * a2 + (dt**3 / 6.0) * a3 + (dt**4 / 24.0) * a4


def rk4_discrete_source_adjoint(
    npoin,
    nelem,
    ngauss,
    lx,
    ly,
    k,
    vx,
    vy,
    dt,
    tfinal,
    jprint,
    coord,
    connections,
    state_out,
    ud,
    eta1=1.0,
    st_ngauss=3,
    mesh_data=None,
    dirichlet_tags=None,
    measurement_boundary=None,
    velocity_nodes=None,
    objective_time_rule="exclude_first",
):
    """Return dJ/dc for the exact discrete RK4 map u_{n+1}=R u_n + B c.

    The forward solver advances the semi-discrete system
    ``u' = -M_lumped^{-1} K u + M_lumped^{-1} source`` with classical RK4.
    This helper applies the exact transpose of that RK4 update map.  The
    returned vector pairs with ``dc/dz = M_lumped^{-1} dsource/dz``.
    """
    _ = (lx, ly, tfinal)
    if jprint != 1:
        raise NotImplementedError(
            "Exact RK4 discrete adjoint currently requires jprint == 1 because "
            "only saved states are available to the gradient code."
        )

    coord = np.asarray(coord, dtype=np.float64)
    connections = np.asarray(connections, dtype=int)
    state_out = np.asarray(state_out, dtype=np.float64)

    params = PDEParams(
        k=k,
        velocity=(vx, vy),
        dt=dt,
        tfinal=tfinal,
        ngauss=ngauss,
        st_ngauss=st_ngauss,
        dirichlet_tags=dirichlet_tags,
        velocity_nodes=velocity_nodes,
    )

    kmatrix = _assemble_primal_operator(
        npoin=npoin,
        nelem=nelem,
        coord=coord,
        connections=connections,
        params=params,
    )

    if mesh_data is None:
        raise ValueError("mesh_data is required for adjoint boundary-tag handling.")

    dirichlet_nodes = _dirichlet_nodes_from_mesh_data(mesh_data, params.dirichlet_tags)
    _apply_homogeneous_dirichlet_rows(kmatrix, dirichlet_nodes)

    mass_lumped = _assemble_lumped_mass(
        npoin=npoin,
        nelem=nelem,
        coord=coord,
        connections=connections,
        ngauss=params.ngauss,
    )
    mass_lumped[dirichlet_nodes] = 1.0
    if np.any(mass_lumped <= 0.0):
        raise ValueError("Non-positive lumped mass encountered in RK4 adjoint.")
    inv_mass = 1.0 / mass_lumped

    observation = _observation_from_mesh_data(mesh_data)
    nframes = state_out.shape[0]
    dt_out = params.dt * jprint
    times = np.arange(nframes, dtype=np.float64) * dt_out

    if not callable(ud):
        ud = np.asarray(ud, dtype=np.float64)
        if ud.shape != state_out.shape:
            raise ValueError("If ud is an array, it must have shape (nframes, npoin).")

    if observation is not None:
        rhs_all = observation.adjoint_rhs_all_times(state_out, ud, eta1, npoin)
    else:
        tracking_edges = _tracking_edges_from_mesh_data(
            mesh_data, measurement_boundary
        )
        rhs_all = _boundary_tracking_rhs_all_times(
            state_out=state_out,
            ud=ud,
            times=times,
            tracking_edges=tracking_edges,
            coord=coord,
            npoin=npoin,
            eta1=eta1,
            st_ngauss=params.st_ngauss,
        )
    objective_weights = _time_weights(nframes, dt_out, objective_time_rule)

    def apply_a_transpose(vector):
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            result = -(kmatrix.T @ (inv_mass * vector))
        if not np.all(np.isfinite(result)):
            raise FloatingPointError(
                "Non-finite vector encountered in RK4 discrete adjoint matvec."
            )
        return result

    source_adjoint = np.zeros(npoin, dtype=np.float64)
    lam = objective_weights[-1] * rhs_all[-1]
    lam[dirichlet_nodes] = 0.0

    for n in range(nframes - 2, -1, -1):
        lam[dirichlet_nodes] = 0.0
        source_adjoint += _apply_rk4_transpose_polynomial(
            lam, apply_a_transpose, params.dt, source=True
        )
        if n > 0:
            lam = _apply_rk4_transpose_polynomial(
                lam, apply_a_transpose, params.dt, source=False
            )
            lam += objective_weights[n] * rhs_all[n]
            lam[dirichlet_nodes] = 0.0

    source_adjoint[dirichlet_nodes] = 0.0
    return source_adjoint


def adv_diffusion_adjoint(
    npoin,
    nelem,
    ngauss,
    lx,
    ly,
    k,
    vx,
    vy,
    dt,
    tfinal,
    jprint,
    coord,
    connections,
    state_out,
    ud,
    eta1=1.0,
    st_ngauss=3,
    mesh_data=None,
    dirichlet_tags=None,
    measurement_boundary=None,
    velocity_nodes=None,
):
    """Solve adjoint equation backward in time for right-boundary tracking (eta2=0)."""
    _ = (lx, ly, tfinal)

    coord = np.asarray(coord, dtype=np.float64)
    connections = np.asarray(connections, dtype=int)
    state_out = np.asarray(state_out, dtype=np.float64)

    params = PDEParams(
        k=k,
        velocity=(vx, vy),
        dt=dt,
        tfinal=tfinal,
        ngauss=ngauss,
        st_ngauss=st_ngauss,
        dirichlet_tags=dirichlet_tags,
        velocity_nodes=velocity_nodes,
    )

    kmatrix = _assemble_primal_operator(
        npoin=npoin,
        nelem=nelem,
        coord=coord,
        connections=connections,
        params=params,
    )

    if mesh_data is None:
        raise ValueError("mesh_data is required for adjoint boundary-tag handling.")

    dirichlet_nodes = _dirichlet_nodes_from_mesh_data(mesh_data, params.dirichlet_tags)
    _apply_homogeneous_dirichlet_rows(kmatrix, dirichlet_nodes)

    # Forward dynamics uses u_t = M^{-1}(b - K u). The corresponding adjoint
    # time operator therefore needs K^T M^{-1}, not just K^T.
    mass_lumped = _assemble_lumped_mass(
        npoin=npoin,
        nelem=nelem,
        coord=coord,
        connections=connections,
        ngauss=params.ngauss,
    )
    mass_lumped[dirichlet_nodes] = 1.0
    if np.any(mass_lumped <= 0.0):
        raise ValueError("Non-positive lumped mass encountered in adjoint solver.")
    inv_mass = 1.0 / mass_lumped

    observation = _observation_from_mesh_data(mesh_data)

    nframes = state_out.shape[0]
    dt_out = params.dt * jprint
    times = np.arange(nframes, dtype=np.float64) * dt_out

    if not callable(ud):
        ud = np.asarray(ud, dtype=np.float64)
        if ud.shape != state_out.shape:
            raise ValueError("If ud is an array, it must have shape (nframes, npoin).")

    if observation is not None:
        rhs_all = observation.adjoint_rhs_all_times(state_out, ud, eta1, npoin)
    else:
        tracking_edges = _tracking_edges_from_mesh_data(
            mesh_data, measurement_boundary
        )
        rhs_all = _boundary_tracking_rhs_all_times(
            state_out=state_out,
            ud=ud,
            times=times,
            tracking_edges=tracking_edges,
            coord=coord,
            npoin=npoin,
            eta1=eta1,
            st_ngauss=params.st_ngauss,
        )

    adjoint_out = np.zeros_like(state_out, dtype=np.float64)
    p = np.zeros(npoin, dtype=np.float64)  # terminal condition p(T)=0 because eta2=0

    operator_t = kmatrix.T * inv_mass[np.newaxis, :]
    system_matrix = np.eye(npoin, dtype=np.float64) + dt_out * operator_t
    lu = piv = None
    if lu_factor is not None and lu_solve is not None:
        lu, piv = lu_factor(system_matrix)

    for n in range(nframes - 1, 0, -1):
        # Backward implicit Euler step:
        # (I + dt_out * K^T) p_prev = p + dt_out * rhs(t_n)
        rhs_step = p + dt_out * rhs_all[n]
        if lu is not None:
            p_prev = lu_solve((lu, piv), rhs_step)
        else:
            p_prev = np.linalg.solve(system_matrix, rhs_step)

        # Enforce homogeneous adjoint values on the Dirichlet boundary subset.
        p_prev[dirichlet_nodes] = 0.0

        adjoint_out[n - 1] = p_prev
        p = p_prev

    return adjoint_out


__all__ = ["adv_diffusion_adjoint", "rk4_discrete_source_adjoint"]
