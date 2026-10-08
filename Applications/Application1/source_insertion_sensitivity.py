r"""Adjoint-based source-insertion sensitivity field for Application1.

Given a baseline control ``z_bar`` in which one source is INACTIVE (its amplitude
set exactly to 0, so its center is irrelevant) and one source is ACTIVE, this
module computes, for every candidate center ``q = (qx, qy)``, the directional
derivative of the existing reduced objective with respect to the AMPLITUDE of a
hypothetical unit Gaussian inserted at ``q``:

    S(q) = d/deps  J( z_bar with the inactive source replaced by [qx, qy, eps] )
                                                                    evaluated at eps = 0.

Discrete derivation (matches the exact RK4 discrete-adjoint gradient the
optimizer actually uses -- see ``Test Application2/gradient_test.py``
``compute_gradient_parts``).  For control component ``j`` the discrete gradient is

    dJ/dz_j = lambda * z_j + < source_adjoint , inv_mass * db_j >           (1)

where
  * ``source_adjoint`` = ``rk4_discrete_source_adjoint(state(z_bar), u_d)`` is the
    accumulated discrete adjoint source term.  It is the exact transpose-RK4
    time integral of the adjoint state p driven by the boundary misfit
    (u(z_bar) - u_d) on the measurement boundary; the helper accumulates it
    BACKWARD in time internally and returns a single chronological-independent
    per-node vector, so callers never touch the backward time ordering.
  * ``inv_mass`` = 1 / lumped mass (with the left-Dirichlet rows set to 1, exactly
    as the gradient code does), and
  * ``db_j`` = ``assemble_load_vector(dS/dz_j)`` is the FE load vector of the
    source's derivative w.r.t. ``z_j``.

For the amplitude of a Gaussian centered at ``q`` the source derivative is simply
the UNIT-amplitude Gaussian, ``dS/dA = G_q(x) = exp(-|x - q|^2 / sigma2)`` (see
``multi_source_utils.source_derivatives_from_control`` -> ``dsdA``).  Substituting
into (1) with ``z_j`` = that amplitude gives

    S(q) = < source_adjoint , inv_mass * assemble_load_vector(G_q) > .        (2)

This is the discrete equivalent of  S(q) = \int_0^T \int_Omega p G_q dx dt ,
carrying the sign convention of the existing discrete adjoint/Lagrangian.

Regularization: Application1 uses ``LAMBDA_REG = 0`` (no Tikhonov term), so the
regularization contributes nothing.  Even if it were nonzero, the first-order
contribution of ``0.5 * lambda * z.z`` to a FRESHLY inserted amplitude ``eps`` is
``lambda * eps`` evaluated at ``eps = 0``, which is 0.  We therefore include no
regularization term in S (matching the task requirement).

Sign convention: ``S(q) = dJ/dA_new``.  Inserting a POSITIVE amplitude ``eps``
changes J by ``~ S(q) * eps``; hence ``S(q) < 0`` means a positive-amplitude
insertion at ``q`` DESCENDS the objective.  The best reinsertion location is

    q_new = argmin_q S(q),   valid only if S(q_new) < 0.

Efficiency: expanding the load-vector inner product in (2) node-by-node and
using ``db_j[i] = sum_e sum_g G_q(x_g) phi_i(x_g) w_g detJ_e`` gives the exactly
equivalent, quadrature-accumulated form

    S(q) = sum_e sum_g  G_q(x_g) * W_h(x_g) * (w_g * detJ_e) ,                (3)

with ``W = source_adjoint * inv_mass`` and ``W_h`` its P1 interpolant evaluated at
the SAME triangle quadrature points ``x_g`` used to assemble the load vectors.
Form (3) is the exact discrete gradient (not a continuous approximation) and
needs only ONE forward and ONE adjoint solve for ``z_bar``; every candidate is an
inexpensive Gaussian-weighted quadrature sum, with no dense
candidate-by-space-by-time array and no PDE solve per candidate.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

import gradient_test as gt  # Test Application2 reduced objective/adjoint (reused as-is)


# ---------------------------------------------------------------------------
# Step 1: adjoint weight field W = source_adjoint * inv_mass  (1 fwd + 1 adjoint)
# ---------------------------------------------------------------------------
def compute_adjoint_weight_field(
    mesh_data,
    pde: dict,
    z_bar: np.ndarray,
    ud_out: np.ndarray,
    eta1: float,
    objective_time_rule: str = "exclude_first",
    dirichlet_tags=("left",),
    state_out: Optional[np.ndarray] = None,
):
    """Return ``(state_out, source_adjoint, inv_mass, W)`` for the baseline ``z_bar``.

    ``state_out`` is the forward solution of ``z_bar`` (reused for the objective and
    the adjoint).  ``source_adjoint`` is the exact RK4 discrete source adjoint for
    the fixed baseline state and residual (u(z_bar) - u_d).  ``inv_mass`` is the
    reciprocal lumped mass with left-Dirichlet rows pinned to 1, exactly as in
    ``gradient_test.compute_gradient_parts``.  ``W = source_adjoint * inv_mass`` is
    the per-node weight field whose Gaussian-weighted quadrature integral is the
    amplitude sensitivity S(q).
    """
    coord = np.asarray(mesh_data["coord"], dtype=np.float64)
    connections = np.asarray(mesh_data["connections"], dtype=int)
    npoin = coord.shape[0]
    nelem = connections.shape[0]

    if state_out is None:
        # ONE forward solve for the baseline z_bar (source 1 amplitude = 0).
        state_out = gt.run_forward(mesh_data, pde, z_bar)

    # ONE adjoint solve for that fixed state and residual.  This accumulates the
    # backward-in-time discrete adjoint into a single per-node vector.
    source_adjoint = gt.rk4_discrete_source_adjoint(
        npoin=npoin,
        nelem=nelem,
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
        ud=ud_out,
        eta1=eta1,
        st_ngauss=3,
        mesh_data=mesh_data,
        dirichlet_tags=list(dirichlet_tags),
        measurement_boundary=mesh_data.get("measurement_boundary", "right"),
        velocity_nodes=pde.get("velocity_nodes"),
        objective_time_rule=objective_time_rule,
    )

    # Lumped mass with left-Dirichlet rows -> 1, identical to the gradient code.
    mass_lumped = gt.assemble_lumped_mass(
        npoin=npoin,
        nelem=nelem,
        coord=coord,
        connections=connections,
        ngauss=pde["ngauss"],
    )
    dirichlet_nodes = np.asarray(
        mesh_data["boundary_nodes_by_name"].get("left", np.array([], dtype=int)),
        dtype=int,
    )
    mass_lumped[dirichlet_nodes] = 1.0
    inv_mass = 1.0 / mass_lumped

    W = source_adjoint * inv_mass
    return state_out, source_adjoint, inv_mass, W


# ---------------------------------------------------------------------------
# Step 2: precompute the quadrature amplitude-weight A_g = W_h(x_g) * w_g * detJ
# ---------------------------------------------------------------------------
def precompute_quadrature_weights(mesh_data, pde: dict, W: np.ndarray):
    """Return ``(qx, qy, ag)`` for every element quadrature point.

    ``ag[g] = W_h(x_g) * weight_g * detJ_e`` is the per-quad-point amplitude
    weight, using the SAME ``triangle_quadrature(ngauss)`` rule and the same P1
    shape functions that ``assemble_load_vector`` uses.  Summing
    ``ag[g] * G_q(x_g)`` over quad points reproduces the discrete gradient (3).
    """
    coord = np.asarray(mesh_data["coord"], dtype=np.float64)
    connections = np.asarray(mesh_data["connections"], dtype=int)
    ngauss = pde["ngauss"]

    weight, xgauss, ygauss = gt.triangle_quadrature(ngauss)
    # P1 shape-function values at each reference quadrature point (3 per gp).
    nsf = np.array(
        [gt.shape_functions(xgauss[g], ygauss[g]) for g in range(ngauss)],
        dtype=np.float64,
    )  # (ngauss, 3)

    nelem = connections.shape[0]
    qx = np.empty(nelem * ngauss, dtype=np.float64)
    qy = np.empty(nelem * ngauss, dtype=np.float64)
    ag = np.empty(nelem * ngauss, dtype=np.float64)

    idx = 0
    for ielem in range(nelem):
        elem = connections[ielem]
        xcoord = coord[elem, 0]
        ycoord = coord[elem, 1]
        J = gt.jacobian(xcoord, ycoord)
        detJ = gt.jacobian_det(J)
        w_elem = W[elem]  # nodal weight at the 3 element nodes
        for g in range(ngauss):
            phi = nsf[g]
            qx[idx] = float(phi @ xcoord)
            qy[idx] = float(phi @ ycoord)
            # W_h(x_g) = sum_local W[node] * phi_local(x_g)
            wh = float(phi @ w_elem)
            ag[idx] = wh * weight[g] * detJ
            idx += 1

    return qx, qy, ag


# ---------------------------------------------------------------------------
# Step 3: S(q_k) for every candidate  (cheap Gaussian-weighted quadrature sum)
# ---------------------------------------------------------------------------
def sensitivity_at_candidates(
    candidate_xy: np.ndarray,
    qx: np.ndarray,
    qy: np.ndarray,
    ag: np.ndarray,
    sigma2: float,
    chunk: int = 200,
) -> np.ndarray:
    """Return ``S(q_k)`` for each candidate via ``S = sum_g ag[g] G_{q_k}(x_g)``.

    Chunked over candidates to bound memory (each chunk builds a
    ``chunk x n_quad`` Gaussian block).  This is form (3): the exact discrete
    amplitude gradient, evaluated with NO extra PDE solve.
    """
    candidate_xy = np.asarray(candidate_xy, dtype=np.float64)
    n = candidate_xy.shape[0]
    out = np.empty(n, dtype=np.float64)
    inv_sigma2 = 1.0 / float(sigma2)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        cx = candidate_xy[start:stop, 0][:, None]  # (m,1)
        cy = candidate_xy[start:stop, 1][:, None]
        dist2 = (qx[None, :] - cx) ** 2 + (qy[None, :] - cy) ** 2  # (m, n_quad)
        # Distances are >= 0, so the exponent is <= 0 and the Gaussian is in
        # (0, 1]; far pairs underflow to subnormals/0, which is harmless. The
        # errstate guard only silences benign BLAS subnormal warnings.
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            gaussian = np.exp(-dist2 * inv_sigma2)
            out[start:stop] = gaussian @ ag
    return out


# ---------------------------------------------------------------------------
# Exact cross-check: S(q) via the literal load-vector inner product (2)
# ---------------------------------------------------------------------------
def sensitivity_via_load_vector(
    mesh_data,
    pde: dict,
    source_adjoint: np.ndarray,
    inv_mass: np.ndarray,
    q: np.ndarray,
) -> float:
    """Reference S(q) using the literal discrete gradient path (2).

    Assembles the unit Gaussian load vector ``db = load_vector(G_q)`` with the
    same quadrature the optimizer uses, then returns
    ``< source_adjoint, inv_mass * db >``.  Used only to confirm that the fast
    quadrature accumulation (3) reproduces the exact discrete gradient.
    """
    coord = np.asarray(mesh_data["coord"], dtype=np.float64)
    connections = np.asarray(mesh_data["connections"], dtype=int)
    qx, qy = float(q[0]), float(q[1])
    r2 = float(pde["sigma2"])

    def gaussian(x: float, y: float) -> float:
        return float(np.exp(-(((x - qx) ** 2 + (y - qy) ** 2) / r2)))

    db = gt.assemble_load_vector(
        npoin=coord.shape[0],
        nelem=connections.shape[0],
        coord=coord,
        connections=connections,
        ngauss=pde["ngauss"],
        scalar_func=gaussian,
    )
    return float(np.dot(source_adjoint, inv_mass * db))
