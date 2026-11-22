"""Steady Navier-Stokes ambient-wind velocity solve for Application1.

This is the Application1 analogue of
App4's ``Navier Stokes Solve/solve_app4_navier_stokes.py``.  It reads
the DC Navy Yard Gmsh mesh, solves a steady incompressible Navier-Stokes problem
with FEniCSx (Taylor-Hood P2/P1, Picard fixed-point iteration, MUMPS LU), writes
ParaView files, and exports the computed velocity in the Application1 transport
mesh node order for the future advection-diffusion driver.

Differences from App4, all driven by the single wind vector in
``navier_stokes_config.WIND_VELOCITY``:

  * the four outer rectangle boundaries are classified as inflow / outflow /
    tangential automatically from  flux = WIND . outward_normal;
  * every inflow boundary gets a uniform Dirichlet velocity  u = WIND_VELOCITY;
  * every outflow boundary keeps the App4 pressure-outlet (p = 0) treatment;
  * all 21 building walls (tags 101..121) are no-slip  u = (0, 0).

Run with fenicsx-env (NOT the inverse environment -- this imports DOLFINx):

    conda run -n fenicsx-env python \
        "Applications/Application1/Navier Stokes Solve/solve_application1_navier_stokes.py"
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
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
    BUILDING_PHYSICAL_IDS,
    BUILDING_TAG_TO_NAME,
    MESH_FILE,
    OUTER_PHYSICAL_IDS,
    OUTER_TAG_TO_NAME,
    build_mesh_data,
)
import navier_stokes_config as cfg  # noqa: E402


# Stable primary output filenames (overwritten every run).
NPZ_NAME = "application1_navier_stokes_velocity_nodes.npz"
VELOCITY_BP_NAME = "application1_navier_stokes_velocity.bp"
PRESSURE_BP_NAME = "application1_navier_stokes_pressure.bp"
METADATA_NAME = "application1_navier_stokes_run_metadata.json"
VELOCITY_PNG_NAME = "application1_velocity_magnitude.png"
BC_PREVIEW_PNG_NAME = "application1_boundary_conditions.png"
CONVERGENCE_CSV_NAME = "application1_convergence_history.csv"
CONVERGENCE_PNG_NAME = "application1_convergence_history.png"
STREAMLINES_PNG_NAME = "application1_streamlines.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Solve steady incompressible Navier-Stokes on the DC Navy Yard "
            "domain for a configured ambient wind and export velocity at the "
            "Application1 transport nodes. Edit navier_stokes_config.WIND_VELOCITY "
            "to change the wind."
        )
    )
    parser.add_argument("--nu", type=float, default=cfg.NU,
                        help=f"Effective kinematic viscosity [m^2/s]. Default {cfg.NU} (from config).")
    parser.add_argument("--max-picard-iter", type=int, default=cfg.MAX_PICARD_ITER,
                        help="Maximum Picard fixed-point iterations.")
    parser.add_argument("--picard-tol", type=float, default=cfg.PICARD_TOL,
                        help="Relative L2 velocity-update tolerance for Picard convergence.")
    return parser.parse_args()


def rank0_print(comm: MPI.Comm, text: str = "") -> None:
    if comm.rank == 0:
        print(text, flush=True)


# --------------------------------------------------------------------------- #
# Wind-driven boundary classification
# --------------------------------------------------------------------------- #
def _cardinal_from_normal(normal) -> str:
    nx, ny = normal
    if abs(nx) >= abs(ny):
        return "east" if nx > 0 else "west"
    return "north" if ny > 0 else "south"


def outer_boundary_normals(msh, facet_tags):
    """Geometric outward unit normal for each outer rectangle tag.

    The outer boundaries are axis-aligned bounding-box edges, so each tagged
    facet set has one (nearly) constant coordinate; the outward normal points
    from the domain center toward that side.
    """
    X = msh.geometry.x
    minx, maxx = float(X[:, 0].min()), float(X[:, 0].max())
    miny, maxy = float(X[:, 1].min()), float(X[:, 1].max())
    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, 0)
    f2v = msh.topology.connectivity(tdim - 1, 0)

    normals = {}
    for tag in OUTER_PHYSICAL_IDS:
        facets = facet_tags.find(tag)
        if len(facets) == 0:
            raise RuntimeError(f"Outer boundary tag {tag} has no facets in the mesh.")
        verts = np.unique(np.concatenate([f2v.links(f) for f in facets]))
        vx = X[verts, 0]
        vy = X[verts, 1]
        if np.ptp(vx) <= np.ptp(vy):  # vertical edge, x ~ const
            nx = 1.0 if abs(vx.mean() - maxx) < abs(vx.mean() - minx) else -1.0
            normals[tag] = (nx, 0.0)
        else:                          # horizontal edge, y ~ const
            ny = 1.0 if abs(vy.mean() - maxy) < abs(vy.mean() - miny) else -1.0
            normals[tag] = (0.0, ny)
    return normals, (minx, maxx, miny, maxy)


def classify_outer_boundaries(normals, wind, tol):
    """Return per-tag classification records and grouped tag lists.

    Raises if the wind is zero, or if there is no inflow or no outflow.
    """
    wx, wy = float(wind[0]), float(wind[1])
    speed = (wx * wx + wy * wy) ** 0.5
    if speed <= tol:
        raise ValueError(
            f"WIND_VELOCITY = ({wx}, {wy}) has magnitude {speed:.3e} <= tolerance "
            f"{tol:.1e}; a zero wind gives a meaningless solve. Set a nonzero wind."
        )

    records = []
    inflow_tags, outflow_tags, tangential_tags = [], [], []
    for tag in OUTER_PHYSICAL_IDS:
        n = normals[tag]
        flux = wx * n[0] + wy * n[1]
        if flux < -tol:
            cls = "inflow"
            inflow_tags.append(tag)
        elif flux > tol:
            cls = "outflow"
            outflow_tags.append(tag)
        else:
            cls = "tangential"
            tangential_tags.append(tag)
        records.append({
            "tag": int(tag),
            "name": OUTER_TAG_TO_NAME[tag],
            "cardinal": _cardinal_from_normal(n),
            "outward_normal": [float(n[0]), float(n[1])],
            "flux": float(flux),
            "class": cls,
        })

    if not inflow_tags:
        raise ValueError(
            "No inflow boundary was found for the configured wind; every outer "
            "normal has flux >= -tol. Refusing to run a meaningless solve."
        )
    if not outflow_tags:
        raise ValueError(
            "No outflow boundary was found for the configured wind; every outer "
            "normal has flux <= tol. Refusing to run a meaningless solve."
        )
    return records, inflow_tags, outflow_tags, tangential_tags


# --------------------------------------------------------------------------- #
# Mesh + finite element spaces
# --------------------------------------------------------------------------- #
def load_mesh(comm: MPI.Comm):
    if not MESH_FILE.exists():
        raise FileNotFoundError(f"DC mesh file does not exist: {MESH_FILE}")
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
    return fem.functionspace(msh, velocity_element), fem.functionspace(msh, pressure_element)


def _facets_for_tags(facet_tags, tags):
    found = [facet_tags.find(int(t)) for t in tags]
    found = [f for f in found if len(f) > 0]
    if not found:
        return np.zeros(0, dtype=np.int32)
    return np.concatenate(found).astype(np.int32)


def make_boundary_conditions(msh, facet_tags, V, Q, inflow_tags, outflow_tags, wind):
    """Wind-driven boundary conditions (Option 2, velocity-normal sign).

    inflow tags       -> Dirichlet  u = WIND_VELOCITY  (uniform far-field)
    building tags      -> Dirichlet  u = (0, 0)         (no-slip)
    outflow tags       -> Dirichlet  p = 0              (pressure outlet / do-nothing)
    tangential tags    -> natural (do-nothing); none for the default wind
    """
    fdim = msh.topology.dim - 1
    inflow_facets = _facets_for_tags(facet_tags, inflow_tags)
    outflow_facets = _facets_for_tags(facet_tags, outflow_tags)
    wall_facets = _facets_for_tags(facet_tags, BUILDING_PHYSICAL_IDS)

    wind_value = np.array([wind[0], wind[1]], dtype=PETSc.ScalarType)
    zero_velocity = np.zeros(msh.geometry.dim, dtype=PETSc.ScalarType)

    bc_inflow = fem.dirichletbc(
        wind_value, fem.locate_dofs_topological(V, fdim, inflow_facets), V
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


def solve_picard(msh, facet_tags, V, Q, nu, inflow_tags, outflow_tags, wind, max_iter, tol):
    """Steady NS by Picard fixed point: freeze advecting velocity w, solve Oseen.

    The Picard initial guess is the uniform wind field, so iteration 1 solves an
    Oseen problem advected by the ambient wind rather than a pure Stokes problem.
    """
    bcs, boundary_sets = make_boundary_conditions(
        msh, facet_tags, V, Q, inflow_tags, outflow_tags, wind
    )

    u = ufl.TrialFunction(V)
    p = ufl.TrialFunction(Q)
    v = ufl.TestFunction(V)
    q = ufl.TestFunction(Q)

    w = fem.Function(V)
    w.name = "picard_frozen_velocity"
    wind_value = np.array([wind[0], wind[1]], dtype=PETSc.ScalarType)
    w.interpolate(lambda x: np.vstack((
        np.full(x.shape[1], wind_value[0]),
        np.full(x.shape[1], wind_value[1]),
    )))
    w.x.scatter_forward()

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
            ) * ufl.dx,
            -ufl.inner(p, ufl.div(v)) * ufl.dx,
        ],
        # Pressure-pressure block is mathematically zero; assembled explicitly
        # so DOLFINx can apply the p=0 outflow Dirichlet dofs.
        [
            ufl.inner(ufl.div(u), q) * ufl.dx,
            fem.Constant(msh, PETSc.ScalarType(0.0)) * p * q * ufl.dx,
        ],
    ]
    L = [ufl.inner(zero_force, v) * ufl.dx, ufl.ZeroBaseForm((q,))]

    problem = LinearProblem(
        a, L, bcs=bcs, u=[uh, ph], kind="mpi",
        petsc_options_prefix="application1_navier_stokes_picard_",
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
            f"Picard iteration {iteration:02d}: "
            f"||u_k-u_(k-1)||_L2={update_norm:.6e}, "
            f"relative={rel_update:.6e}, ||u_k||_L2={velocity_norm:.6e}",
        )
        w.x.array[:] = uh.x.array
        w.x.scatter_forward()
        if rel_update < tol:
            converged = True
            break

    return uh, ph, diagnostics, converged, boundary_sets


# --------------------------------------------------------------------------- #
# ParaView output
# --------------------------------------------------------------------------- #
def write_paraview(msh, uh, ph) -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    velocity_path = OUTPUT_DIR / VELOCITY_BP_NAME
    pressure_path = OUTPUT_DIR / PRESSURE_BP_NAME

    p1_vector = element(
        "Lagrange", msh.basix_cell(), degree=1, shape=(msh.geometry.dim,),
        dtype=default_real_type,
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


# --------------------------------------------------------------------------- #
# Evaluate velocity at Application1 transport nodes
# --------------------------------------------------------------------------- #
def points_for_dolfinx(coords: np.ndarray) -> np.ndarray:
    points = np.zeros((coords.shape[0], 3), dtype=np.float64)
    points[:, :2] = coords[:, :2]
    return points


def find_cells_for_points(msh, points, padding):
    tree = geometry.bb_tree(msh, msh.topology.dim, padding=padding)
    candidates = geometry.compute_collisions_points(tree, points)
    collisions = geometry.compute_colliding_cells(msh, candidates, points)
    cells = np.full(points.shape[0], -1, dtype=np.int32)
    for i in range(points.shape[0]):
        links = collisions.links(i)
        if len(links) > 0:
            cells[i] = links[0]
    return cells


def evaluate_velocity_at_nodes(uh, mesh_data, wind, inflow_names, padding):
    """Evaluate the P2 velocity at every transport node, with an exact fallback.

    Failed collision-tree evaluations (points sitting exactly on facets/vertices)
    are resolved from the boundary they belong to:
      * building nodes -> (0, 0)                (exact no-slip value)
      * inflow-boundary nodes -> WIND_VELOCITY  (exact Dirichlet value)
      * anything else -> nearest successfully evaluated node.
    """
    coords = np.asarray(mesh_data["coord"], dtype=np.float64)
    points = points_for_dolfinx(coords)
    cells = find_cells_for_points(uh.function_space.mesh, points, padding)

    velocity_nodes = np.full((coords.shape[0], 2), np.nan, dtype=np.float64)
    found = np.where(cells >= 0)[0]
    if len(found) > 0:
        values = uh.eval(points[found], cells[found])
        velocity_nodes[found, :] = np.asarray(values, dtype=np.float64)[:, :2]

    failed = np.where(cells < 0)[0]
    info = {
        "evaluated_count": int(len(found)),
        "failed_initial_count": int(len(failed)),
        "fallback_building": 0,
        "fallback_inflow": 0,
        "fallback_nearest": 0,
    }
    if len(failed) == 0:
        return coords, velocity_nodes, info

    nodes = mesh_data["boundary_nodes_by_name"]
    building_nodes = set()
    for name in mesh_data["building_boundary_names"]:
        if name in nodes:
            building_nodes.update(int(n) for n in nodes[name])
    inflow_nodes = set()
    for name in inflow_names:
        if name in nodes:
            inflow_nodes.update(int(n) for n in nodes[name])

    found_coords = coords[found, :2] if len(found) > 0 else np.zeros((0, 2))
    for idx in failed:
        node = int(idx)
        if node in building_nodes:
            velocity_nodes[idx, :] = (0.0, 0.0)
            info["fallback_building"] += 1
        elif node in inflow_nodes:
            velocity_nodes[idx, :] = (float(wind[0]), float(wind[1]))
            info["fallback_inflow"] += 1
        elif len(found) > 0:
            j = int(np.argmin(np.sum((found_coords - coords[idx, :2]) ** 2, axis=1)))
            velocity_nodes[idx, :] = velocity_nodes[found[j], :]
            info["fallback_nearest"] += 1
        else:
            raise RuntimeError("No cells could be evaluated anywhere; solve failed.")
    return coords, velocity_nodes, info


def validate_velocity_nodes(coords, velocity_nodes, mesh_data):
    if velocity_nodes.shape != (coords.shape[0], 2):
        raise ValueError(
            f"velocity_nodes shape {velocity_nodes.shape} != ({coords.shape[0]}, 2)"
        )
    if not np.all(np.isfinite(velocity_nodes)):
        bad = np.where(~np.isfinite(velocity_nodes).all(axis=1))[0]
        raise ValueError(f"velocity_nodes contains non-finite values at nodes {bad[:20]}")

    nodes = mesh_data["boundary_nodes_by_name"]
    building_nodes = np.unique(np.concatenate(
        [np.asarray(nodes[name], dtype=int)
         for name in mesh_data["building_boundary_names"] if name in nodes]
    )) if any(name in nodes for name in mesh_data["building_boundary_names"]) else np.zeros(0, int)
    wall_max = float(np.max(np.linalg.norm(velocity_nodes[building_nodes], axis=1))) \
        if len(building_nodes) else 0.0

    all_boundary = np.unique(np.concatenate(
        [np.asarray(v, dtype=int) for v in nodes.values()]
    ))
    interior_mask = np.ones(coords.shape[0], dtype=bool)
    interior_mask[all_boundary] = False
    magnitudes = np.linalg.norm(velocity_nodes, axis=1)
    interior_max = float(np.max(magnitudes[interior_mask])) if np.any(interior_mask) else 0.0

    if interior_max <= 1.0e-12:
        raise ValueError("Velocity is essentially zero at all interior nodes.")
    return {
        "building_wall_max_velocity": wall_max,
        "interior_max_velocity": interior_max,
        "global_max_velocity": float(np.max(magnitudes)),
        "global_min_velocity": float(np.min(magnitudes)),
    }


def save_npz(coords, velocity_nodes) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = OUTPUT_DIR / NPZ_NAME
    np.savez(npz_path, coords=coords, velocity_nodes=velocity_nodes)
    return npz_path


# --------------------------------------------------------------------------- #
# Plots + convergence history + metadata
# --------------------------------------------------------------------------- #
def write_convergence_history(diagnostics) -> Path:
    csv_path = OUTPUT_DIR / CONVERGENCE_CSV_NAME
    lines = ["iteration,update_norm_l2,relative_update,velocity_norm_l2"]
    for it, upd, rel, vnorm in diagnostics:
        lines.append(f"{it},{upd:.12e},{rel:.12e},{vnorm:.12e}")
    csv_path.write_text("\n".join(lines) + "\n")
    return csv_path


def make_plots(coords, connections, velocity_nodes, diagnostics, classification, mesh_data):
    """Best-effort matplotlib diagnostics. Never fatal to the solve/NPZ."""
    written = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
    except Exception as exc:  # noqa: BLE001
        print(f"[plots] matplotlib unavailable ({exc}); skipping plots", flush=True)
        return written

    x = coords[:, 0]
    y = coords[:, 1]
    tri = mtri.Triangulation(x, y, connections)
    speed = np.linalg.norm(velocity_nodes, axis=1)

    # 1. velocity magnitude + quiver
    try:
        fig, ax = plt.subplots(figsize=(6, 12))
        tpc = ax.tripcolor(tri, speed, shading="gouraud", cmap="viridis")
        step = max(1, coords.shape[0] // 700)
        ax.quiver(x[::step], y[::step], velocity_nodes[::step, 0], velocity_nodes[::step, 1],
                  color="white", scale_units="xy", angles="xy", width=0.002, alpha=0.7)
        fig.colorbar(tpc, ax=ax, shrink=0.5, label="|u| [m/s]")
        ax.set_aspect("equal"); ax.set_xlabel("x (east) [m]"); ax.set_ylabel("y (north) [m]")
        ax.set_title("Application1 Navier-Stokes velocity magnitude")
        fig.tight_layout(); path = OUTPUT_DIR / VELOCITY_PNG_NAME
        fig.savefig(path, dpi=150); plt.close(fig); written.append(path)
    except Exception as exc:  # noqa: BLE001
        print(f"[plots] velocity magnitude plot failed ({exc})", flush=True)

    # 2. streamlines on a regular grid
    try:
        from scipy.interpolate import griddata
        gx = np.linspace(x.min(), x.max(), 220)
        gy = np.linspace(y.min(), y.max(), 320)
        GX, GY = np.meshgrid(gx, gy)
        U = griddata((x, y), velocity_nodes[:, 0], (GX, GY), method="linear")
        Vv = griddata((x, y), velocity_nodes[:, 1], (GX, GY), method="linear")
        spd = np.sqrt(U ** 2 + Vv ** 2)
        fig, ax = plt.subplots(figsize=(6, 12))
        ax.streamplot(GX, GY, U, Vv, color=spd, cmap="viridis", density=1.4, linewidth=0.7)
        ax.set_aspect("equal"); ax.set_xlabel("x (east) [m]"); ax.set_ylabel("y (north) [m]")
        ax.set_title("Application1 Navier-Stokes streamlines")
        fig.tight_layout(); path = OUTPUT_DIR / STREAMLINES_PNG_NAME
        fig.savefig(path, dpi=150); plt.close(fig); written.append(path)
    except Exception as exc:  # noqa: BLE001
        print(f"[plots] streamlines plot skipped ({exc})", flush=True)

    # 3. boundary-condition preview
    try:
        cls_color = {"inflow": "#2ca02c", "outflow": "#d62728", "tangential": "#7f7f7f"}
        edges = mesh_data["boundary_edges_by_name"]
        fig, ax = plt.subplots(figsize=(6, 12))
        ax.triplot(tri, color="0.85", lw=0.2)
        seen = set()
        for rec in classification:
            name = rec["name"]; cls = rec["class"]
            if name not in edges:
                continue
            lbl = cls if cls not in seen else None
            seen.add(cls)
            for e in edges[name]:
                ax.plot(coords[e, 0], coords[e, 1], color=cls_color[cls], lw=2.5, label=lbl)
                lbl = None
            mid = coords[np.unique(edges[name].reshape(-1))].mean(axis=0)
            ax.annotate(f"{name}\n{rec['cardinal']} {cls}", (mid[0], mid[1]),
                        fontsize=7, ha="center", va="center",
                        bbox=dict(boxstyle="round", fc="white", ec=cls_color[cls], alpha=0.8))
        seen_b = False
        for name in mesh_data["building_boundary_names"]:
            if name not in edges:
                continue
            lbl = "no-slip buildings" if not seen_b else None
            seen_b = True
            for e in edges[name]:
                ax.plot(coords[e, 0], coords[e, 1], color="black", lw=1.2, label=lbl)
                lbl = None
        ax.set_aspect("equal"); ax.set_xlabel("x (east) [m]"); ax.set_ylabel("y (north) [m]")
        ax.set_title("Application1 boundary-condition classification")
        ax.legend(loc="upper right", fontsize=7)
        fig.tight_layout(); path = OUTPUT_DIR / BC_PREVIEW_PNG_NAME
        fig.savefig(path, dpi=150); plt.close(fig); written.append(path)
    except Exception as exc:  # noqa: BLE001
        print(f"[plots] boundary-condition preview failed ({exc})", flush=True)

    # 4. convergence history
    try:
        arr = np.asarray(diagnostics, dtype=float)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.semilogy(arr[:, 0], arr[:, 2], "o-")
        ax.set_xlabel("Picard iteration"); ax.set_ylabel("relative L2 velocity update")
        ax.set_title("Application1 Picard convergence"); ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout(); path = OUTPUT_DIR / CONVERGENCE_PNG_NAME
        fig.savefig(path, dpi=150); plt.close(fig); written.append(path)
    except Exception as exc:  # noqa: BLE001
        print(f"[plots] convergence plot failed ({exc})", flush=True)

    return written


def write_metadata(args, classification, inflow_tags, outflow_tags, tangential_tags,
                   diagnostics, converged, validation, npz_path, output_files,
                   dolfinx_counts, transport_counts) -> Path:
    def recs(tags):
        return [r for r in classification if r["tag"] in set(int(t) for t in tags)]

    final_residual = float(diagnostics[-1][2]) if diagnostics else None
    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "wind_velocity": [float(cfg.WIND_VELOCITY[0]), float(cfg.WIND_VELOCITY[1])],
        "wind_velocity_units": "m/s, +x=east, +y=north",
        "wind_speed": cfg.wind_speed(),
        "flux_tolerance": cfg.FLUX_TOLERANCE,
        "outer_tag_cardinal_mapping": {
            str(r["tag"]): {"name": r["name"], "cardinal": r["cardinal"],
                            "outward_normal": r["outward_normal"]}
            for r in classification
        },
        "inflow_boundaries": recs(inflow_tags),
        "outflow_boundaries": recs(outflow_tags),
        "tangential_boundaries": recs(tangential_tags),
        "building_tags": [int(t) for t in BUILDING_PHYSICAL_IDS],
        "building_boundary_condition": "no-slip u=(0,0)",
        "viscosity_nu_m2_per_s": float(args.nu),
        "viscosity_note": "effective/eddy kinematic viscosity, not molecular air viscosity",
        "density_rho_kg_per_m3": float(cfg.RHO),
        "density_note": "informational; solver uses incompressible kinematic form (no rho term)",
        "numerical_method": "Taylor-Hood P2/P1, Picard fixed-point, PETSc MUMPS LU",
        "picard_tol": float(args.picard_tol),
        "picard_max_iter": int(args.max_picard_iter),
        "picard_iterations": int(diagnostics[-1][0]) if diagnostics else 0,
        "picard_converged": bool(converged),
        "final_relative_residual": final_residual,
        "velocity_validation": validation,
        "dolfinx_mesh": dolfinx_counts,
        "transport_mesh": transport_counts,
        "mesh_file": str(MESH_FILE),
        "npz_path": str(npz_path),
        "npz_schema": {"coords": "(num_nodes, 3)", "velocity_nodes": "(num_nodes, 2)"},
        "output_files": [str(p) for p in output_files],
    }
    path = OUTPUT_DIR / METADATA_NAME
    path.write_text(json.dumps(meta, indent=2) + "\n")
    return path


# --------------------------------------------------------------------------- #
# Header / main
# --------------------------------------------------------------------------- #
def print_header(args, classification, inflow_tags, outflow_tags, tangential_tags):
    print("=" * 70)
    print("Application1 steady incompressible Navier-Stokes (DC Navy Yard)")
    print("=" * 70)
    print(f"mesh file        : {MESH_FILE}")
    print(f"output directory : {OUTPUT_DIR}")
    print()
    print("PDE (incompressible kinematic form):")
    print("  (v . grad) v - nu * Laplace(v) + grad(p) = 0")
    print("  div(v) = 0")
    print()
    print("Parameters:")
    print(f"  WIND_VELOCITY = ({cfg.WIND_VELOCITY[0]}, {cfg.WIND_VELOCITY[1]}) m/s "
          f"(+x=east, +y=north), |wind| = {cfg.wind_speed():.4f} m/s")
    print(f"  nu (effective/eddy kinematic viscosity) = {args.nu} m^2/s")
    print(f"  rho (informational) = {cfg.RHO} kg/m^3")
    print(f"  nonlinear method = Picard fixed point (max {args.max_picard_iter}, "
          f"rel tol {args.picard_tol})")
    print()
    print("Outer boundary tag -> cardinal -> classification "
          "(flux = WIND . outward_normal):")
    for r in classification:
        print(f"  tag {r['tag']} {r['name']:<7} {r['cardinal']:<6} "
              f"n=({r['outward_normal'][0]:+.0f},{r['outward_normal'][1]:+.0f}) "
              f"flux={r['flux']:+.4f} -> {r['class'].upper()}")
    print(f"  inflow tags     : {inflow_tags}  -> Dirichlet u = WIND")
    print(f"  outflow tags    : {outflow_tags}  -> pressure outlet p = 0")
    print(f"  tangential tags : {tangential_tags}  -> natural/do-nothing")
    print(f"  building tags   : {list(BUILDING_PHYSICAL_IDS)}  -> no-slip u = 0")
    print("=" * 70)
    print()


def main() -> None:
    args = parse_args()
    comm = MPI.COMM_WORLD
    if comm.size != 1:
        raise RuntimeError(
            "Run this export script serially so the NPZ node ordering is "
            "unambiguous (do not use mpirun)."
        )

    transport_mesh_data = build_mesh_data()
    transport_coords = np.asarray(transport_mesh_data["coord"], dtype=np.float64)
    transport_conn = np.asarray(transport_mesh_data["connections"], dtype=int)

    msh, cell_tags, facet_tags, physical_groups = load_mesh(comm)
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)

    # Verify every required tag is present before doing any work.
    facet_tag_values = set(int(t) for t in np.unique(facet_tags.values))
    missing_outer = [t for t in OUTER_PHYSICAL_IDS if t not in facet_tag_values]
    missing_bldg = [t for t in BUILDING_PHYSICAL_IDS if t not in facet_tag_values]
    if missing_outer:
        raise RuntimeError(f"Missing outer boundary tags in mesh: {missing_outer}")
    if missing_bldg:
        raise RuntimeError(f"Missing building tags in mesh: {missing_bldg}")

    normals, bbox = outer_boundary_normals(msh, facet_tags)
    classification, inflow_tags, outflow_tags, tangential_tags = classify_outer_boundaries(
        normals, cfg.WIND_VELOCITY, cfg.FLUX_TOLERANCE
    )
    print_header(args, classification, inflow_tags, outflow_tags, tangential_tags)

    V, Q = make_function_spaces(msh)
    dolfinx_nodes = int(msh.geometry.x.shape[0])
    dolfinx_cells = int(msh.topology.index_map(msh.topology.dim).size_local)
    rank0_print(comm, "Mesh summary:")
    rank0_print(comm, f"  domain bbox: x[{bbox[0]:.1f}, {bbox[1]:.1f}] "
                      f"y[{bbox[2]:.1f}, {bbox[3]:.1f}] (meters)")
    rank0_print(comm, f"  DOLFINx nodes: {dolfinx_nodes}, cells: {dolfinx_cells}")
    rank0_print(comm, f"  transport nodes: {transport_coords.shape[0]}, "
                      f"triangles: {transport_conn.shape[0]}")
    rank0_print(comm, f"  velocity dofs: {V.dofmap.index_map.size_local * V.dofmap.index_map_bs}, "
                      f"pressure dofs: {Q.dofmap.index_map.size_local * Q.dofmap.index_map_bs}")
    rank0_print(comm)

    uh, ph, diagnostics, converged, boundary_sets = solve_picard(
        msh, facet_tags, V, Q, args.nu, inflow_tags, outflow_tags,
        cfg.WIND_VELOCITY, args.max_picard_iter, args.picard_tol,
    )
    rank0_print(comm, f"Picard converged: {converged} "
                      f"(final relative update {diagnostics[-1][2]:.3e})")
    rank0_print(comm, "boundary facet counts: " + ", ".join(
        f"{k}={len(v)}" for k, v in boundary_sets.items()))
    rank0_print(comm)

    if not np.all(np.isfinite(uh.x.array)) or not np.all(np.isfinite(ph.x.array)):
        raise RuntimeError("Non-finite values in the computed velocity/pressure solution.")

    paraview_paths = write_paraview(msh, uh, ph)

    inflow_names = [OUTER_TAG_TO_NAME[t] for t in inflow_tags]
    coord_scale = max(bbox[1] - bbox[0], bbox[3] - bbox[2])
    coords, velocity_nodes, eval_info = evaluate_velocity_at_nodes(
        uh, transport_mesh_data, cfg.WIND_VELOCITY, inflow_names,
        padding=1.0e-6 * coord_scale,
    )
    validation = validate_velocity_nodes(coords, velocity_nodes, transport_mesh_data)
    npz_path = save_npz(coords, velocity_nodes)
    csv_path = write_convergence_history(diagnostics)
    plot_paths = make_plots(coords, transport_conn, velocity_nodes, diagnostics,
                            classification, transport_mesh_data)

    output_files = paraview_paths + [npz_path, csv_path] + plot_paths
    meta_path = write_metadata(
        args, classification, inflow_tags, outflow_tags, tangential_tags,
        diagnostics, converged, validation, npz_path, output_files,
        {"nodes": dolfinx_nodes, "cells": dolfinx_cells},
        {"nodes": int(transport_coords.shape[0]), "triangles": int(transport_conn.shape[0])},
    )
    output_files.append(meta_path)

    magnitudes = np.linalg.norm(velocity_nodes, axis=1)
    rank0_print(comm, "Node export summary:")
    rank0_print(comm, f"  coords shape: {coords.shape}, velocity_nodes shape: {velocity_nodes.shape}")
    rank0_print(comm, f"  evaluated by collision tree: {eval_info['evaluated_count']}")
    rank0_print(comm, f"  fallback (building/inflow/nearest): "
                      f"{eval_info['fallback_building']}/{eval_info['fallback_inflow']}/"
                      f"{eval_info['fallback_nearest']}")
    rank0_print(comm, f"  velocity magnitude min/max: "
                      f"{float(np.min(magnitudes)):.6e} / {float(np.max(magnitudes)):.6e}")
    rank0_print(comm, f"  building wall max |u|: {validation['building_wall_max_velocity']:.6e}")
    rank0_print(comm)

    # ------------------------------------------------------------------- #
    # Final concise summary
    # ------------------------------------------------------------------- #
    print("=" * 70)
    print("APPLICATION1 NAVIER-STOKES RUN SUMMARY")
    print("=" * 70)
    print(f"folder            : {APP_DIR}")
    print(f"mesh (.msh)       : {MESH_FILE}")
    print(f"mesh (.xdmf)      : {APP_DIR / 'DC_mesh' / 'dc_navy_yard_mesh.xdmf'}")
    print(f"facets (.xdmf)    : {APP_DIR / 'DC_mesh' / 'dc_navy_yard_facets.xdmf'}")
    print(f"wind vector       : ({cfg.WIND_VELOCITY[0]}, {cfg.WIND_VELOCITY[1]}) m/s "
          f"(+x=east,+y=north), |wind|={cfg.wind_speed():.4f}")
    print("outer tag mapping : 11=bottom/SOUTH, 12=top/NORTH, 13=left/WEST, 14=right/EAST")
    for r in classification:
        print(f"  {r['name']:<7}({r['cardinal']:<6}) flux={r['flux']:+.4f} -> {r['class'].upper()}")
    print(f"viscosity nu      : {args.nu} m^2/s (effective/eddy)")
    print("numerical method  : Taylor-Hood P2/P1, Picard, MUMPS LU")
    print(f"converged         : {converged} in {diagnostics[-1][0]} iterations")
    print(f"final residual    : {diagnostics[-1][2]:.6e} (rel. L2 velocity update)")
    print(f"velocity finite   : {bool(np.all(np.isfinite(velocity_nodes)))}")
    print(f"building wall |u| : {validation['building_wall_max_velocity']:.3e} (approx zero)")
    print("output files:")
    for p in output_files:
        print(f"  {p}")
    print()
    print("Rerun command (edit WIND_VELOCITY in navier_stokes_config.py, then):")
    print('  conda run -n fenicsx-env python \\')
    print('      "Applications/Application1/Navier Stokes Solve/solve_application1_navier_stokes.py"')
    print("=" * 70)


if __name__ == "__main__":
    main()
