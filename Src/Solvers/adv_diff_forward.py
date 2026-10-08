from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

# Minimal path bridge so this script can import shared solver data modules
# from Advection Diffusion PDECO/Src/Solvers when run directly.
_THIS_FILE = Path(__file__).resolve()
_SRC_SOLVERS = _THIS_FILE.parent
if str(_SRC_SOLVERS) not in sys.path:
    sys.path.append(str(_SRC_SOLVERS))

from linear_fe_data import jacobian, jacobian_det, shape_function_derivatives, shape_functions
from pde_params import PDEParams
from quadrature_data import triangle_quadrature
from advection_assembly import (
    advection_matrix_to_dense,
    assemble_advection_nodal_quadrature,
)


def _dirichlet_nodes_from_mesh_data(mesh_data, dirichlet_tags):
    """Get Dirichlet nodes by named boundary tags from mesh_data."""
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
            "No Dirichlet nodes found in mesh_data for requested tags: "
            f"{dirichlet_tags}. Ensure mesh_data contains boundary_nodes_by_name."
        )
    return np.unique(np.asarray(nodes, dtype=int))


def _assemble_system(npoin, nelem, coord, connections, source_func, params: PDEParams):
    """Assemble stiffness, source, and lumped mass vector with P1 triangles."""
    npoin_per_elem = 3
    weight, xgauss, ygauss = triangle_quadrature(params.ngauss)
    dN = shape_function_derivatives()

    kmatrix = np.zeros((npoin, npoin), dtype=np.float64)
    source = np.zeros(npoin, dtype=np.float64)
    mass_lumped = np.zeros(npoin, dtype=np.float64)
    nodal_advection = None
    if params.velocity_nodes is not None:
        nodal_advection = advection_matrix_to_dense(
            assemble_advection_nodal_quadrature(coord, connections, params.velocity_nodes)
        )

    for ielem in range(nelem):
        klocal = np.zeros((npoin_per_elem, npoin_per_elem), dtype=np.float64)
        mlocal = np.zeros((npoin_per_elem, npoin_per_elem), dtype=np.float64)
        sourcelocal = np.zeros(npoin_per_elem, dtype=np.float64)

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

                    # Diffusion term (matches existing implementation).
                    klocal[jpoint, ipoint] += (
                        params.k * (bi * bj + ci * cj) * weight[igauss] / detJ
                    )

                    nsf = shape_functions(u, v)
                    if nodal_advection is None:
                        # Advection term (kept consistent with existing indexing choice).
                        klocal[jpoint, ipoint] += (
                            (params.vx * bi + params.vy * ci) * nsf[jpoint] * weight[igauss]
                        )
                    # Consistent mass contribution.
                    mlocal[jpoint, ipoint] += nsf[jpoint] * nsf[ipoint] * weight[igauss] * detJ

            for igauss in range(params.ngauss):
                u = xgauss[igauss]
                v = ygauss[igauss]
                nsf = shape_functions(u, v)

                x = np.dot(nsf, xcoord)
                y = np.dot(nsf, ycoord)

                sourcelocal[jpoint] += source_func(x, y) * nsf[jpoint] * weight[igauss] * detJ

        for jpoint in range(npoin_per_elem):
            jglob = elem_nodes[jpoint]
            for ipoint in range(npoin_per_elem):
                iglob = elem_nodes[ipoint]
                kmatrix[jglob, iglob] += klocal[jpoint, ipoint]
                mass_lumped[jglob] += mlocal[jpoint, ipoint]
            source[jglob] += sourcelocal[jpoint]

    if nodal_advection is not None:
        kmatrix += nodal_advection

    return kmatrix, source, mass_lumped


def _apply_homogeneous_dirichlet(kmatrix, source, boundary_points):
    """Apply homogeneous Dirichlet BC on the provided node ids."""
    npoin = kmatrix.shape[0]
    for ipoin in boundary_points:
        kmatrix[ipoin, :] = 0.0
        kmatrix[ipoin, ipoin] = 1.0
        source[ipoin] = 0.0


def _rk4_step(unknown, kmatrix, source, inv_mass, dt):
    """One RK4 step for M du/dt = source - K u using lumped M^{-1}."""

    def rhs(u):
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            val = source - kmatrix @ u
        if not np.all(np.isfinite(val)):
            raise FloatingPointError("Non-finite RHS encountered in forward RK4 step.")
        return inv_mass * val

    k1 = rhs(unknown)
    k2 = rhs(unknown + 0.5 * dt * k1)
    k3 = rhs(unknown + 0.5 * dt * k2)
    k4 = rhs(unknown + dt * k3)
    return unknown + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def adv_diff_dyn(
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
    source_func,
    mesh_data=None,
    dirichlet_tags=None,
    velocity_nodes=None,
):
    """Compatibility API matching source_identification/solvers/adv_diff_dyn.py."""
    _ = (lx, ly)  # kept for API compatibility with existing caller signatures

    coord = np.asarray(coord, dtype=np.float64)
    connections = np.asarray(connections, dtype=int)

    params = PDEParams(
        k=k,
        velocity=(vx, vy),
        dt=dt,
        tfinal=tfinal,
        ngauss=ngauss,
        dirichlet_tags=dirichlet_tags,
        velocity_nodes=velocity_nodes,
    )

    kmatrix, source, mass_lumped = _assemble_system(
        npoin=npoin,
        nelem=nelem,
        coord=coord,
        connections=connections,
        source_func=source_func,
        params=params,
    )

    if mesh_data is None:
        mesh_data = {"coord": coord}
    boundary_points = _dirichlet_nodes_from_mesh_data(mesh_data, params.dirichlet_tags)
    _apply_homogeneous_dirichlet(kmatrix, source, boundary_points)
    # Avoid divide-by-zero and keep homogeneous Dirichlet nodes pinned to 0.
    mass_lumped[boundary_points] = 1.0
    if np.any(mass_lumped <= 0.0):
        raise ValueError("Non-positive lumped mass encountered in forward solver.")
    inv_mass = 1.0 / mass_lumped

    time = 0.0
    itime = 0
    iprint = 0
    # Number of time-integration steps performed by:
    # while time + dt/2 < tfinal: time += dt
    nsteps = int(np.floor((params.tfinal + 0.5 * params.dt) / params.dt))
    # Saved frames include the initial condition at t=0.
    nprint = nsteps // jprint + 1

    unknown = np.zeros(npoin, dtype=np.float64)
    unknown_out = np.zeros((nprint, npoin), dtype=np.float64)
    unknown_out[iprint] = unknown

    while time + params.dt / 2.0 < params.tfinal:
        time += params.dt
        itime += 1
        unknown = _rk4_step(unknown, kmatrix, source, inv_mass, params.dt)
        unknown[boundary_points] = 0.0

        if itime % jprint == 0:
            iprint += 1
            if iprint >= nprint:
                raise IndexError(
                    "Forward output indexing exceeded allocation: "
                    f"iprint={iprint}, nprint={nprint}. Check dt/tfinal/jprint setup."
                )
            unknown_out[iprint] = unknown
    return unknown_out


__all__ = ["adv_diff_dyn"]
