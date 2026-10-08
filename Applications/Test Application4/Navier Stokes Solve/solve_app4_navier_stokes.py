"""Steady Navier-Stokes velocity field generator for Test Application4.

This script is intentionally standalone.  It reads the existing App4 Gmsh
mesh, solves a steady incompressible Navier-Stokes problem with FEniCSx, writes
ParaView files, and exports the computed velocity in the App4 transport mesh
node order for later use by the inverse source-identification driver.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from mpi4py import MPI
from petsc4py import PETSc

import numpy as np
import ufl
from basix.ufl import element
from dolfinx import default_real_type, fem, geometry, io
from dolfinx.fem.petsc import LinearProblem
from dolfinx.io import gmsh as dolfinx_gmsh


SCRIPT_DIR = Path(__file__).resolve().parent
APP_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = APP_DIR.parents[1]
OUTPUT_DIR = APP_DIR / "output" / "navier_stokes"

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from mesh_data_config import (  # noqa: E402
    BOTTOM_PHYSICAL_ID,
    BUILDING_1_PHYSICAL_ID,
    BUILDING_2_PHYSICAL_ID,
    BUILDING_3_PHYSICAL_ID,
    BUILDING_4_PHYSICAL_ID,
    BUILDING_5_PHYSICAL_ID,
    INNER_CORNER_PHYSICAL_ID,
    LEFT_PHYSICAL_ID,
    MESH_FILE,
    RIGHT_PHYSICAL_ID,
    TOP_PHYSICAL_ID,
    build_mesh_data,
)


WALL_TAGS = {
    "bottom": BOTTOM_PHYSICAL_ID,
    "top": TOP_PHYSICAL_ID,
    "inner_corner": INNER_CORNER_PHYSICAL_ID,
    "building_1": BUILDING_1_PHYSICAL_ID,
    "building_2": BUILDING_2_PHYSICAL_ID,
    "building_3": BUILDING_3_PHYSICAL_ID,
    "building_4": BUILDING_4_PHYSICAL_ID,
    "building_5": BUILDING_5_PHYSICAL_ID,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Solve steady incompressible Navier-Stokes on the App4 L-shaped "
            "city domain and export velocity values at App4 transport nodes."
        )
    )
    parser.add_argument(
        "--nu",
        type=float,
        default=0.10,
        help=(
            "Kinematic viscosity. Default 0.10 is deliberately moderate for "
            "a stable first steady solve on the complex App4 geometry."
        ),
    )
    parser.add_argument(
        "--umax",
        type=float,
        default=1.0,
        help="Maximum inflow speed used in u_in(y)=4*Umax*y*(1-y).",
    )
    parser.add_argument(
        "--max-picard-iter",
        type=int,
        default=20,
        help="Maximum number of Picard fixed-point iterations.",
    )
    parser.add_argument(
        "--picard-tol",
        type=float,
        default=1.0e-8,
        help="Relative L2 velocity-update tolerance for Picard convergence.",
    )
    return parser.parse_args()


def rank0_print(comm: MPI.Comm, text: str = "") -> None:
    if comm.rank == 0:
        print(text, flush=True)


def parabolic_inflow_values(points: np.ndarray, umax: float) -> np.ndarray:
    y = points[1]
    profile = 4.0 * umax * y * (1.0 - y)
    return np.vstack((profile, np.zeros_like(profile)))


def load_app4_mesh(comm: MPI.Comm):
    if not MESH_FILE.exists():
        raise FileNotFoundError(f"App4 mesh file does not exist: {MESH_FILE}")
    mesh_data = dolfinx_gmsh.read_from_msh(MESH_FILE, comm, rank=0, gdim=2)
    if mesh_data.facet_tags is None:
        raise RuntimeError("The imported Gmsh mesh has no facet tags.")
    return mesh_data.mesh, mesh_data.cell_tags, mesh_data.facet_tags, mesh_data.physical_groups


def make_function_spaces(msh):
    gdim = msh.geometry.dim
    velocity_element = element(
        "Lagrange", msh.basix_cell(), degree=2, shape=(gdim,), dtype=default_real_type
    )
    pressure_element = element(
        "Lagrange", msh.basix_cell(), degree=1, dtype=default_real_type
    )
    V = fem.functionspace(msh, velocity_element)
    Q = fem.functionspace(msh, pressure_element)
    return V, Q


def make_boundary_conditions(msh, facet_tags, V, Q, umax: float):
    fdim = msh.topology.dim - 1
    inflow_facets = facet_tags.find(LEFT_PHYSICAL_ID)
    outflow_facets = facet_tags.find(RIGHT_PHYSICAL_ID)
    wall_facets = np.concatenate(
        [facet_tags.find(tag) for tag in WALL_TAGS.values()]
    ).astype(np.int32)

    inflow = fem.Function(V)
    inflow.name = "inflow_velocity"
    inflow.interpolate(lambda x: parabolic_inflow_values(x, umax))
    zero_velocity = np.zeros(msh.geometry.dim, dtype=PETSc.ScalarType)

    bc_inflow = fem.dirichletbc(
        inflow, fem.locate_dofs_topological(V, fdim, inflow_facets)
    )
    bc_walls = fem.dirichletbc(
        zero_velocity, fem.locate_dofs_topological(V, fdim, wall_facets), V
    )
    bc_pressure_out = fem.dirichletbc(
        PETSc.ScalarType(0.0),
        fem.locate_dofs_topological(Q, fdim, outflow_facets),
        Q,
    )
    return [bc_inflow, bc_walls, bc_pressure_out], {
        "inflow_facets": inflow_facets,
        "outflow_facets": outflow_facets,
        "wall_facets": wall_facets,
    }


def l2_norm(msh, expr) -> float:
    local = fem.assemble_scalar(fem.form(ufl.inner(expr, expr) * ufl.dx))
    return float(np.sqrt(msh.comm.allreduce(local, op=MPI.SUM)))


def solve_picard(msh, facet_tags, V, Q, nu: float, umax: float, max_iter: int, tol: float):
    """Solve (w . grad) u - nu Laplace(u) + grad(p) = 0 repeatedly.

    Picard iteration freezes the advecting velocity w from the previous
    nonlinear iterate.  Iteration 1 therefore solves a Stokes-like problem
    because w starts at zero in the interior.
    """
    bcs, boundary_sets = make_boundary_conditions(msh, facet_tags, V, Q, umax)

    u = ufl.TrialFunction(V)
    p = ufl.TrialFunction(Q)
    v = ufl.TestFunction(V)
    q = ufl.TestFunction(Q)

    w = fem.Function(V)
    w.name = "picard_frozen_velocity"
    uh = fem.Function(V)
    uh.name = "velocity"
    ph = fem.Function(Q)
    ph.name = "pressure"

    zero_force = fem.Constant(msh, np.zeros(msh.geometry.dim, dtype=PETSc.ScalarType))
    a = [
        [
            (
                nu * ufl.inner(ufl.grad(u), ufl.grad(v))
                + ufl.inner(ufl.dot(ufl.grad(u), w), v)
            )
            * ufl.dx,
            -ufl.inner(p, ufl.div(v)) * ufl.dx,
        ],
        # The pressure-pressure block is mathematically zero.  It is still
        # assembled explicitly so DOLFINx can apply p=0 on the outflow dofs.
        [
            ufl.inner(ufl.div(u), q) * ufl.dx,
            fem.Constant(msh, PETSc.ScalarType(0.0)) * p * q * ufl.dx,
        ],
    ]
    L = [ufl.inner(zero_force, v) * ufl.dx, ufl.ZeroBaseForm((q,))]

    problem = LinearProblem(
        a,
        L,
        bcs=bcs,
        u=[uh, ph],
        kind="mpi",
        petsc_options_prefix="app4_navier_stokes_picard_",
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "ksp_error_if_not_converged": True,
        },
    )

    diagnostics = []
    converged = False
    for iteration in range(1, max_iter + 1):
        problem.solve()
        uh.x.scatter_forward()
        ph.x.scatter_forward()

        update_norm = l2_norm(msh, uh - w)
        velocity_norm = l2_norm(msh, uh)
        rel_update = update_norm / max(velocity_norm, 1.0e-14)
        diagnostics.append((iteration, update_norm, rel_update, velocity_norm))
        rank0_print(
            msh.comm,
            (
                f"Picard iteration {iteration:02d}: "
                f"||u_k-u_(k-1)||_L2={update_norm:.6e}, "
                f"relative={rel_update:.6e}, ||u_k||_L2={velocity_norm:.6e}"
            ),
        )
        w.x.array[:] = uh.x.array
        w.x.scatter_forward()
        if rel_update < tol:
            converged = True
            break

    return uh, ph, diagnostics, converged, boundary_sets


def write_outputs(msh, uh, ph) -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    velocity_path = OUTPUT_DIR / "app4_navier_stokes_velocity.bp"
    pressure_path = OUTPUT_DIR / "app4_navier_stokes_pressure.bp"

    p1_vector = element(
        "Lagrange", msh.basix_cell(), degree=1, shape=(msh.geometry.dim,), dtype=default_real_type
    )
    V_vis = fem.functionspace(msh, p1_vector)
    u_vis = fem.Function(V_vis)
    u_vis.name = "velocity"
    u_vis.interpolate(uh)
    ph.name = "pressure"

    with io.VTXWriter(msh.comm, velocity_path, u_vis, "bp4") as writer:
        writer.write(0.0)
    with io.VTXWriter(msh.comm, pressure_path, ph, "bp4") as writer:
        writer.write(0.0)
    return [velocity_path, pressure_path]


def points_for_dolfinx(coords: np.ndarray) -> np.ndarray:
    points = np.zeros((coords.shape[0], 3), dtype=np.float64)
    points[:, :2] = coords[:, :2]
    return points


def find_cells_for_points(msh, points: np.ndarray, padding: float):
    tree = geometry.bb_tree(msh, msh.topology.dim, padding=padding)
    candidates = geometry.compute_collisions_points(tree, points)
    collisions = geometry.compute_colliding_cells(msh, candidates, points)
    cells = np.full(points.shape[0], -1, dtype=np.int32)
    for i in range(points.shape[0]):
        links = collisions.links(i)
        if len(links) > 0:
            cells[i] = links[0]
    return cells


def boundary_fallback_values(
    coords: np.ndarray, failed: np.ndarray, boundary_nodes_by_name: dict[str, np.ndarray], umax: float
):
    values = np.full((len(failed), 2), np.nan, dtype=np.float64)
    node_to_names: dict[int, list[str]] = {}
    for name, nodes in boundary_nodes_by_name.items():
        for node in np.asarray(nodes, dtype=int):
            node_to_names.setdefault(int(node), []).append(name)

    filled = 0
    unresolved = []
    wall_names = set(WALL_TAGS)
    for local_index, node in enumerate(failed):
        names = set(node_to_names.get(int(node), []))
        y = coords[node, 1]
        if "left" in names:
            values[local_index] = [4.0 * umax * y * (1.0 - y), 0.0]
            filled += 1
        elif names & wall_names:
            values[local_index] = [0.0, 0.0]
            filled += 1
        else:
            unresolved.append(int(node))
    return values, filled, unresolved


def evaluate_velocity_at_app4_nodes(uh, transport_mesh_data, umax: float):
    coords = np.asarray(transport_mesh_data["coord"], dtype=np.float64)
    points = points_for_dolfinx(coords)
    cells = find_cells_for_points(uh.function_space.mesh, points, padding=1.0e-10)

    velocity_nodes = np.full((coords.shape[0], 2), np.nan, dtype=np.float64)
    found = np.where(cells >= 0)[0]
    if len(found) > 0:
        values = uh.eval(points[found], cells[found])
        velocity_nodes[found, :] = np.asarray(values, dtype=np.float64)[:, :2]

    failed = np.where(cells < 0)[0]
    fallback_count = 0
    unresolved: list[int] = []
    if len(failed) > 0:
        fallback_values, fallback_count, unresolved = boundary_fallback_values(
            coords,
            failed,
            transport_mesh_data["boundary_nodes_by_name"],
            umax,
        )
        velocity_nodes[failed, :] = fallback_values

    return coords, velocity_nodes, {
        "evaluated_count": int(len(found)),
        "failed_initial_count": int(len(failed)),
        "fallback_count": int(fallback_count),
        "unresolved_nodes": unresolved,
    }


def validate_velocity_nodes(coords, velocity_nodes, transport_mesh_data, umax: float) -> dict[str, float]:
    if velocity_nodes.shape != (coords.shape[0], 2):
        raise ValueError(
            f"velocity_nodes shape {velocity_nodes.shape} does not match "
            f"({coords.shape[0]}, 2)"
        )
    if not np.all(np.isfinite(velocity_nodes)):
        bad = np.where(~np.isfinite(velocity_nodes).all(axis=1))[0]
        raise ValueError(f"velocity_nodes contains non-finite values at nodes {bad[:20]}")

    boundary_nodes = transport_mesh_data["boundary_nodes_by_name"]
    wall_nodes = np.unique(
        np.concatenate([np.asarray(boundary_nodes[name], dtype=int) for name in WALL_TAGS])
    )
    wall_max = float(np.max(np.linalg.norm(velocity_nodes[wall_nodes], axis=1)))

    left_nodes = np.asarray(boundary_nodes["left"], dtype=int)
    expected_left = 4.0 * umax * coords[left_nodes, 1] * (1.0 - coords[left_nodes, 1])
    left_error = float(np.max(np.abs(velocity_nodes[left_nodes, 0] - expected_left)))
    left_y_error = float(np.max(np.abs(velocity_nodes[left_nodes, 1])))

    all_boundary = np.unique(
        np.concatenate([np.asarray(nodes, dtype=int) for nodes in boundary_nodes.values()])
    )
    interior_mask = np.ones(coords.shape[0], dtype=bool)
    interior_mask[all_boundary] = False
    magnitudes = np.linalg.norm(velocity_nodes, axis=1)
    interior_max = float(np.max(magnitudes[interior_mask]))
    right_nodes = np.asarray(boundary_nodes["right"], dtype=int)
    right_max = float(np.max(magnitudes[right_nodes]))

    if interior_max <= 1.0e-10:
        raise ValueError("Velocity is essentially zero at all interior App4 nodes.")
    if right_max <= 1.0e-10:
        raise ValueError("Velocity is essentially zero near the right outflow.")

    return {
        "wall_max_velocity": wall_max,
        "left_profile_x_max_error": left_error,
        "left_profile_y_max_abs": left_y_error,
        "interior_max_velocity": interior_max,
        "right_outflow_max_velocity": right_max,
    }


def save_npz(coords, velocity_nodes) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = OUTPUT_DIR / "app4_navier_stokes_velocity_nodes.npz"
    np.savez(npz_path, coords=coords, velocity_nodes=velocity_nodes)
    return npz_path


def print_header(args: argparse.Namespace) -> None:
    print("App4 steady incompressible Navier-Stokes solve")
    print(f"script directory: {SCRIPT_DIR}")
    print(f"project root: {PROJECT_ROOT}")
    print(f"App4 mesh file: {MESH_FILE}")
    print()
    print("PDE:")
    print("  (v . grad) v - nu * Laplace(v) + grad(p) = 0")
    print("  div(v) = 0")
    print("  f = 0")
    print()
    print("Parameters:")
    print(f"  Umax = {args.umax}")
    print(f"  nu = {args.nu}")
    print(
        "  note: nu=0.10 is the default because it is a stable moderate "
        "first Reynolds-number choice for this complex nonconvex geometry."
    )
    print("  nonlinear iteration = Picard fixed point")
    print(
        "  Picard linearization: solve (w . grad) v - nu*Laplace(v) + grad(p)=0 "
        "with w frozen from the previous velocity iterate."
    )
    print()


def main() -> None:
    args = parse_args()
    comm = MPI.COMM_WORLD

    if comm.size != 1:
        raise RuntimeError(
            "This standalone export script expects serial execution so the NPZ "
            "node ordering is unambiguous. Run it without mpirun."
        )

    if comm.rank == 0:
        print_header(args)

    transport_mesh_data = build_mesh_data()
    transport_coords = np.asarray(transport_mesh_data["coord"], dtype=np.float64)
    transport_connections = np.asarray(transport_mesh_data["connections"], dtype=int)

    msh, cell_tags, facet_tags, physical_groups = load_app4_mesh(comm)
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    V, Q = make_function_spaces(msh)

    rank0_print(comm, "Domain/mesh summary:")
    rank0_print(comm, f"  DOLFINx geometry dimension: {msh.geometry.dim}")
    rank0_print(comm, f"  DOLFINx topology dimension: {msh.topology.dim}")
    rank0_print(comm, f"  DOLFINx local geometry nodes: {msh.geometry.x.shape[0]}")
    rank0_print(
        comm,
        f"  DOLFINx local cells: {msh.topology.index_map(msh.topology.dim).size_local}",
    )
    rank0_print(comm, f"  App4 transport nodes: {transport_coords.shape[0]}")
    rank0_print(comm, f"  App4 transport triangles: {transport_connections.shape[0]}")
    rank0_print(comm, f"  physical groups from .msh: {physical_groups}")
    if cell_tags is not None:
        rank0_print(comm, f"  cell tag values: {np.unique(cell_tags.values)}")
    rank0_print(comm)

    rank0_print(comm, "Finite element spaces:")
    rank0_print(comm, "  velocity: continuous Lagrange P2 vector Taylor-Hood space")
    rank0_print(comm, "  pressure: continuous Lagrange P1 Taylor-Hood space")
    rank0_print(comm, f"  local velocity dofs: {V.dofmap.index_map.size_local * V.dofmap.index_map_bs}")
    rank0_print(comm, f"  local pressure dofs: {Q.dofmap.index_map.size_local * Q.dofmap.index_map_bs}")
    rank0_print(comm)

    rank0_print(comm, "Boundary condition summary:")
    rank0_print(
        comm,
        f"  left/inflow tag {LEFT_PHYSICAL_ID}: v=(4*Umax*y*(1-y), 0)",
    )
    rank0_print(comm, f"  right/outflow tag {RIGHT_PHYSICAL_ID}: p=0")
    rank0_print(
        comm,
        "  walls/buildings/rest: v=(0,0) on "
        + ", ".join(f"{name} tag {tag}" for name, tag in WALL_TAGS.items()),
    )
    rank0_print(comm)

    uh, ph, diagnostics, converged, boundary_sets = solve_picard(
        msh,
        facet_tags,
        V,
        Q,
        nu=args.nu,
        umax=args.umax,
        max_iter=args.max_picard_iter,
        tol=args.picard_tol,
    )
    rank0_print(comm, f"Picard converged: {converged}")
    rank0_print(
        comm,
        "boundary facet counts: "
        + ", ".join(f"{key}={len(value)}" for key, value in boundary_sets.items()),
    )
    rank0_print(comm)

    output_paths = write_outputs(msh, uh, ph)
    coords, velocity_nodes, eval_info = evaluate_velocity_at_app4_nodes(
        uh, transport_mesh_data, args.umax
    )
    validation = validate_velocity_nodes(coords, velocity_nodes, transport_mesh_data, args.umax)
    npz_path = save_npz(coords, velocity_nodes)

    magnitudes = np.linalg.norm(velocity_nodes, axis=1)
    rank0_print(comm, "App4 node export summary:")
    rank0_print(comm, f"  coords shape: {coords.shape}")
    rank0_print(comm, f"  velocity_nodes shape: {velocity_nodes.shape}")
    rank0_print(comm, f"  evaluated by collision tree: {eval_info['evaluated_count']}")
    rank0_print(comm, f"  initially failed evaluations: {eval_info['failed_initial_count']}")
    rank0_print(comm, f"  boundary fallback values used: {eval_info['fallback_count']}")
    rank0_print(comm, f"  unresolved nodes after fallback: {len(eval_info['unresolved_nodes'])}")
    rank0_print(comm, f"  velocity magnitude min: {float(np.min(magnitudes)):.6e}")
    rank0_print(comm, f"  velocity magnitude max: {float(np.max(magnitudes)):.6e}")
    rank0_print(comm, f"  validation metrics: {validation}")
    rank0_print(comm, "  velocity_nodes export succeeded: yes")
    rank0_print(comm)

    rank0_print(comm, "Output paths written:")
    for path in output_paths + [npz_path]:
        rank0_print(comm, f"  {path}")


if __name__ == "__main__":
    main()
