"""Steady Navier-Stokes ventilation flow for Application2 (one mesh level).

Solves, on the coarse OR fine treatment-room mesh, independently,

    -nu Laplace(v) + (v . grad) v + grad p = 0,    div v = 0      in Omega,
    v = g_in   (downward parabolic profile)          on the top-wall supplies 21-23,
    v = 0      (no slip)                             on walls 41-47, partitions 51-52, beds 61-63,
                                                     bathroom wall 71-72,
    (nu grad v - p I) n = 0   (natural outflow)      on outlet_door 31 and outlet_exhaust 32,

with Taylor-Hood P2/P1 elements and Picard iteration (as Application1), each
linear Oseen system solved by MUMPS LU.  p is the kinematic pressure.

Weak form and natural-outflow derivation
----------------------------------------
For test functions (w, q) with w = 0 on the Dirichlet boundary (inlets + walls):

    int (-nu Laplace v + grad p) . w
        = int nu grad v : grad w - int p div w - int_{dOmega} ((nu grad v - p I) n) . w ds .

Since w = 0 on inlets and walls, the boundary integral only lives on the
outlets.  The discrete problem is

    find (u, p):  int nu grad u : grad w + ((grad u) a) . w - p div w = 0,
                  int q div u = 0,

with no outlet boundary term, i.e. the outlet traction (nu grad u - p I) n is
weakly ZERO: this is the natural (do-nothing) outflow condition for the
vector-Laplacian (NOT the symmetric-gradient) stress.  No Dirichlet condition is
placed on any pressure dof, and no pressure pinning / mean-value constraint is
added: because the outlets have positive measure, the traction condition
determines the pressure constant, and the saddle-point matrix is nonsingular.
This is verified at runtime: (i) zero pressure Dirichlet dofs, (ii) MUMPS
null-pivot detection on the final matrix, (iii) the assembled nonlinear momentum
residual vanishes on the (free) outlet velocity dofs, while the inlet/wall rows
carry the reaction tractions.  ``a`` is the Picard-frozen velocity (a = 0 on
iteration 1, i.e. a Stokes solve; thereafter the previous iterate).

The three beds are solid rectangular holes in the mesh (idealized 2D impermeable
obstacles); their edges belong to the wall group and carry no slip.  No outlet
pressure or velocity is prescribed: the split between the corridor door and the
bathroom exhaust, and any backflow, are outputs of the solve (reported).  Passive
natural outlets do not represent mechanical suction.

Outputs go to ``<out-root>/<level>/navier_stokes/``.  ParaView: ``*.bp`` (P2
velocity / P1 pressure; ADIOS2 VTX reader), ``*_P1_<level>.xdmf`` (Xdmf3 reader)
and ``*_P1_<level>.vtu`` (built-in XML unstructured-grid reader).  Run serially with
fenicsx-env:

    FX=/path/to/fenicsx-env/bin/python    # or: conda activate fenicsx-env; FX=python
    $FX "Applications/Application2/Navier Stokes Solve/solve_application2_navier_stokes.py" --level coarse
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

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
REPO_ROOT = APP_DIR.parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import room_geometry_config as geo  # noqa: E402
import navier_stokes_config as cfg  # noqa: E402
import mesh_data_config as mdc  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description="Application2 steady NS ventilation solve (one mesh level).")
    ap.add_argument("--level", choices=list(geo.MESH_LEVELS), required=True)
    ap.add_argument("--out-root", type=Path, default=mdc.OUTPUT_DIR)
    return ap.parse_args()


def say(text=""):
    print(text, flush=True)


# --------------------------------------------------------------------------- #
# Probe locations shared by both levels (common physical points)
# --------------------------------------------------------------------------- #
GRID_SPACING = 0.05   # cell-centred grid; no probe lies on a boundary line
LINE_SPACING = 0.005
PROBE_MARGIN = 1.0e-9


def inside_solid(x, y, margin=0.0):
    """Partitions AND beds (all solids removed from the fluid domain)."""
    return geo.inside_solid(x, y, margin)


def probe_sets():
    h = GRID_SPACING
    gx = np.arange(geo.ROOM_X_MIN + h / 2, geo.DOMAIN_X_MAX, h)
    gy = np.arange(geo.ROOM_Y_MIN + h / 2, geo.ROOM_Y_MAX, h)
    GX, GY = np.meshgrid(gx, gy)
    keep = geo.inside_fluid_domain(GX.ravel(), GY.ravel())
    grid = np.column_stack([GX.ravel()[keep], GY.ravel()[keep]])
    e = PROBE_MARGIN
    xs = np.linspace(geo.ROOM_X_MIN + e, geo.DOMAIN_X_MAX - e, int(round(geo.DOMAIN_X_MAX / LINE_SPACING)) + 1)
    ys = np.linspace(geo.ROOM_Y_MIN + e, geo.ROOM_Y_MAX - e, int(round(6.0 / LINE_SPACING)) + 1)
    lines = {}
    # keep only points strictly inside the fluid (drops solids, their faces and the exterior)
    fluid = lambda P: P[geo.inside_fluid_domain(P[:, 0], P[:, 1], margin=1e-6)]  # noqa: E731
    for yl in (1.0, 1.5, geo.PARTITION_BOTTOM_Y, 4.5):       # y = 1 passes through the bathroom doorway
        lines[f"h_y{yl:g}"] = fluid(np.column_stack([xs, np.full_like(xs, yl)]))
    for xl in (*geo.VENT_CENTERS_X, 10.0):                    # x = 10: bathroom
        lines[f"v_x{xl:g}"] = fluid(np.column_stack([np.full_like(ys, xl), ys]))
    yd = np.linspace(geo.BATH_DOORWAY_Y[0] + e, geo.BATH_DOORWAY_Y[1] - e, 801)
    lines["doorway_x9"] = np.column_stack([np.full_like(yd, geo.ROOM_X_MAX), yd])
    return {"grid": grid, "grid_shape_x": gx, "grid_shape_y": gy, "grid_keep_mask": keep}, lines


def eval_at(fun, points2):
    msh = fun.function_space.mesh
    pts = np.zeros((points2.shape[0], 3))
    pts[:, :2] = points2
    tree = geometry.bb_tree(msh, msh.topology.dim, padding=1e-10)
    cand = geometry.compute_collisions_points(tree, pts)
    coll = geometry.compute_colliding_cells(msh, cand, pts)
    cells = np.full(len(pts), -1, dtype=np.int32)
    for i in range(len(pts)):
        lk = coll.links(i)
        if len(lk):
            cells[i] = lk[0]
    bs = fun.function_space.dofmap.index_map_bs
    out = np.full((len(pts), bs), np.nan)
    ok = cells >= 0
    if ok.any():
        out[ok] = np.asarray(fun.eval(pts[ok], cells[ok])).reshape(-1, bs)
    return out, int((~ok).sum())


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    args = parse_args()
    comm = MPI.COMM_WORLD
    if comm.size != 1:
        raise RuntimeError("run serially (transport node ordering must be unambiguous)")
    level = args.level
    out_dir = Path(args.out_root) / level / "navier_stokes"
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    msh_path = mdc.mesh_file(level, args.out_root)
    md = mdc.build_mesh_data(level, args.out_root)  # transport (meshio) view of the same file
    say("=" * 72)
    say(f"Application2 steady Navier-Stokes  level={level}")
    say("=" * 72)
    say(f"mesh       : {msh_path}  sha256={md['mesh_sha256'][:16]}...")
    say(f"output     : {out_dir}")
    say(f"parameters : {json.dumps(cfg.as_dict())}")

    mdata = dolfinx_gmsh.read_from_msh(str(msh_path), comm, rank=0, gdim=2)
    msh, facet_tags = mdata.mesh, mdata.facet_tags
    tdim, fdim = msh.topology.dim, msh.topology.dim - 1
    msh.topology.create_connectivity(fdim, tdim)
    present = set(int(t) for t in np.unique(facet_tags.values))
    missing = [t for t in geo.BOUNDARY_TAGS.values() if t not in present]
    if missing:
        raise RuntimeError(f"mesh is missing facet tags {missing}")

    V = fem.functionspace(msh, element("Lagrange", msh.basix_cell(), 2, shape=(2,), dtype=default_real_type))
    Q = fem.functionspace(msh, element("Lagrange", msh.basix_cell(), 1, dtype=default_real_type))
    n_vdofs = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    n_pdofs = Q.dofmap.index_map.size_local
    n_vertices = msh.geometry.x.shape[0]
    n_cells = msh.topology.index_map(tdim).size_local
    say(f"mesh       : vertices={n_vertices} triangles={n_cells}  velocity dofs={n_vdofs} "
        f"pressure dofs={n_pdofs} total={n_vdofs + n_pdofs}")

    # ---------------- boundary conditions
    def facets(tags):
        return np.concatenate([facet_tags.find(int(t)) for t in tags]).astype(np.int32)

    inlet_facets = facets(geo.INLET_TAG_LIST)
    wall_facets = facets(geo.WALL_TAG_LIST)
    outlet_facets = facets(geo.OUTLET_TAG_LIST)

    def inlet_profile(x):
        return cfg.inlet_velocity(x)      # downward parabola on each top-wall supply

    g_in = fem.Function(V)
    g_in.interpolate(inlet_profile)
    inlet_bdofs = fem.locate_dofs_topological(V, fdim, inlet_facets)
    wall_bdofs = fem.locate_dofs_topological(V, fdim, wall_facets)
    outlet_bdofs = fem.locate_dofs_topological(V, fdim, outlet_facets)
    bc_in = fem.dirichletbc(g_in, inlet_bdofs)
    bc_wall = fem.dirichletbc(np.zeros(2, dtype=PETSc.ScalarType), wall_bdofs, V)
    bcs = [bc_in, bc_wall]          # NOTE: no pressure BC of any kind
    n_dir_p = 0
    say(f"BCs        : inlet Dirichlet blocks={len(inlet_bdofs)}, wall no-slip blocks={len(wall_bdofs)}, "
        f"pressure Dirichlet dofs={n_dir_p}; outlets natural (nu grad v - p I) n = 0")

    # ---------------- Picard
    u, p = ufl.TrialFunction(V), ufl.TrialFunction(Q)
    w, q = ufl.TestFunction(V), ufl.TestFunction(Q)
    a_fr = fem.Function(V, name="picard_frozen_velocity")   # iteration 1: zero -> Stokes
    uh = fem.Function(V, name="velocity")
    ph = fem.Function(Q, name="pressure")
    nu = fem.Constant(msh, PETSc.ScalarType(cfg.NU))
    a_form = [
        [(nu * ufl.inner(ufl.grad(u), ufl.grad(w)) + ufl.inner(ufl.dot(ufl.grad(u), a_fr), w)) * ufl.dx,
         -ufl.inner(p, ufl.div(w)) * ufl.dx],
        # pressure-pressure block is identically zero (assembled as an explicit zero block
        # for a square blocked structure; no pressure BC/stabilisation is applied).
        [ufl.inner(ufl.div(u), q) * ufl.dx,
         fem.Constant(msh, PETSc.ScalarType(0.0)) * p * q * ufl.dx],
    ]
    L_form = [ufl.inner(fem.Constant(msh, np.zeros(2, dtype=PETSc.ScalarType)), w) * ufl.dx,
              ufl.ZeroBaseForm((q,))]
    petsc_opts = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",
                  "ksp_error_if_not_converged": True}
    problem = LinearProblem(a_form, L_form, bcs=bcs, u=[uh, ph], kind="mpi",
                            petsc_options_prefix=f"application2_ns_{level}_", petsc_options=petsc_opts)

    def l2(expr):
        return float(np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(expr, expr) * ufl.dx)),
                                            op=MPI.SUM)))

    history, converged = [], False
    t0 = time.perf_counter()
    for it in range(1, cfg.MAX_PICARD_ITER + 1):
        problem.solve()
        uh.x.scatter_forward()
        ph.x.scatter_forward()
        upd = l2(uh - a_fr)
        vn = l2(uh)
        rel = upd / max(vn, 1e-300)
        history.append((it, upd, rel, vn))
        say(f"Picard {it:03d}: ||u_k-u_k-1||_L2={upd:.6e}  relative={rel:.6e}  ||u_k||_L2={vn:.6e}")
        a_fr.x.array[:] = uh.x.array
        a_fr.x.scatter_forward()
        if not (np.isfinite(rel)):
            break
        if rel < cfg.PICARD_TOL:
            converged = True
            break
    picard_seconds = time.perf_counter() - t0
    finite = bool(np.all(np.isfinite(uh.x.array)) and np.all(np.isfinite(ph.x.array)))
    say(f"Picard converged={converged} iterations={history[-1][0]} final relative update={history[-1][2]:.3e} "
        f"({picard_seconds:.1f}s)")

    # ---------------- linear-system checks on the final Picard matrix
    A, b, x = problem.A, problem.b, problem.x
    r = b.duplicate()
    A.mult(x, r)
    r.axpy(-1.0, b)
    lin_res = float(r.norm() / max(b.norm(), 1e-300))
    ksp = PETSc.KSP().create(comm)
    ksp.setOperators(A)
    ksp.setType("preonly")
    pc = ksp.getPC()
    pc.setType("lu")
    pc.setFactorSolverType("mumps")
    pc.setFactorSetUpSolverType()
    F = pc.getFactorMatrix()
    F.setMumpsIcntl(24, 1)      # null-pivot detection
    F.setMumpsCntl(3, 1e-12)
    ksp.setUp()
    null_pivots = int(F.getMumpsInfog(28))
    infog1 = int(F.getMumpsInfog(1))
    ksp.destroy()
    n_dir_v = sum(int(bc.dof_indices()[1]) for bc in bcs)
    say(f"system     : size={A.getSize()} last linear rel. residual={lin_res:.2e} MUMPS INFOG(1)={infog1} "
        f"null pivots INFOG(28)={null_pivots} velocity Dirichlet dofs={n_dir_v} pressure Dirichlet dofs={n_dir_p}")

    # ---------------- nonlinear residual / natural-outflow check
    R_mom = fem.assemble_vector(fem.form(
        (nu * ufl.inner(ufl.grad(uh), ufl.grad(w)) + ufl.inner(ufl.dot(ufl.grad(uh), uh), w)
         - ph * ufl.div(w)) * ufl.dx)).array.reshape(-1, 2)
    R_con = fem.assemble_vector(fem.form(ufl.div(uh) * q * ufl.dx)).array
    dir_blocks = np.union1d(inlet_bdofs, wall_bdofs)
    out_only = np.setdiff1d(outlet_bdofs, dir_blocks)   # outlet dofs excluding the shared end points
    free = np.setdiff1d(np.arange(R_mom.shape[0]), np.union1d(dir_blocks, outlet_bdofs))
    rnorm = lambda arr: float(np.max(np.abs(arr))) if arr.size else 0.0  # noqa: E731
    residual = {
        "momentum_max_abs_on_outlet_dofs": rnorm(R_mom[out_only]),
        "momentum_max_abs_on_interior_free_dofs": rnorm(R_mom[free]),
        "momentum_max_abs_on_dirichlet_dofs_reaction": rnorm(R_mom[dir_blocks]),
        "continuity_max_abs": rnorm(R_con),
        "n_outlet_velocity_blocks_free": int(len(out_only)),
        "note": ("R(w) = int nu grad u:grad w + ((grad u) u).w - p div w for every P2 basis w at the final iterate. "
                 "Natural outflow <=> R = 0 on the free outlet dofs (same level as interior dofs: Picard lag + "
                 "round-off); inlet/wall rows hold the boundary reaction (traction) and are O(reaction)."),
    }
    residual["outlet_over_reaction"] = (residual["momentum_max_abs_on_outlet_dofs"]
                                        / max(residual["momentum_max_abs_on_dirichlet_dofs_reaction"], 1e-300))
    say(f"residual   : outlet {residual['momentum_max_abs_on_outlet_dofs']:.2e}  interior "
        f"{residual['momentum_max_abs_on_interior_free_dofs']:.2e}  Dirichlet reaction "
        f"{residual['momentum_max_abs_on_dirichlet_dofs_reaction']:.2e}  continuity {residual['continuity_max_abs']:.2e}")

    # ---------------- fluxes, divergence, outlet traction (Taylor-Hood field)
    ds = ufl.Measure("ds", domain=msh, subdomain_data=facet_tags)
    nvec = ufl.FacetNormal(msh)
    one = fem.Constant(msh, PETSc.ScalarType(1.0))
    scal = lambda form: float(comm.allreduce(fem.assemble_scalar(fem.form(form)), op=MPI.SUM))  # noqa: E731
    flux_th = {nm: scal(ufl.dot(uh, nvec) * ds(tag)) for nm, tag in geo.BOUNDARY_TAGS.items()}
    inflow_th = sum(flux_th[n] for n in geo.INLET_NAMES)
    outflow_th = sum(flux_th[n] for n in geo.OUTLET_NAMES)
    net_th = sum(flux_th.values())
    prescribed_per_inlet = -cfg.INLET_MEAN_SPEED * geo.VENT_WIDTH
    traction = ufl.dot(nu * ufl.grad(uh), nvec) - ph * nvec
    outlets = {}
    un = ufl.dot(uh, nvec)
    for nm in geo.OUTLET_NAMES:
        tag = geo.OUTLET_TAGS[nm]
        Lo = scal(one * ds(tag))
        # backflow = inward normal velocity (u.n < 0) on a natural outlet
        back_flux = scal(ufl.conditional(ufl.lt(un, 0), un, 0) * ds(tag))
        back_len = scal(ufl.conditional(ufl.lt(un, 0), 1.0, 0.0) * ds(tag))
        odofs = fem.locate_dofs_topological(V, fdim, facet_tags.find(int(tag)))
        n_out = np.asarray(geo.OUTLET_NORMALS[nm])
        un_dofs = uh.x.array.reshape(-1, 2)[odofs] @ n_out
        outlets[nm] = {
            "length_m": Lo, "flux": flux_th[nm],
            "share_of_total_inflow": flux_th[nm] / -inflow_th,
            "share_of_total_outflow": flux_th[nm] / outflow_th,
            "mean_normal_velocity": flux_th[nm] / Lo,
            "mean_pressure": scal(ph * ds(tag)) / Lo,
            "mean_normal_traction": scal(ufl.dot(traction, nvec) * ds(tag)) / Lo,
            "mean_abs_normal_traction": scal(abs(ufl.dot(traction, nvec)) * ds(tag)) / Lo,
            "backflow_flux_inward": back_flux,
            "backflow_length_m": back_len,
            "backflow_length_fraction": back_len / Lo,
            "min_normal_velocity_at_P2_dofs": float(un_dofs.min()),
            "max_normal_velocity_at_P2_dofs": float(un_dofs.max()),
            "n_P2_dofs_with_inflow": int((un_dofs < -1e-12).sum()),
        }
    div_l2 = l2(ufl.div(uh))
    grad_l2 = float(np.sqrt(scal(ufl.inner(ufl.grad(uh), ufl.grad(uh)) * ufl.dx)))
    th = {
        "flux_by_tag": flux_th,
        "inflow_total": inflow_th, "outflow_total": outflow_th, "net_boundary_flux": net_th,
        "normalized_imbalance_abs_net_over_abs_inflow": abs(net_th) / abs(inflow_th),
        "prescribed_inflow_per_inlet": prescribed_per_inlet,
        "inlet_flux_error_vs_prescribed": {n: flux_th[n] - prescribed_per_inlet for n in geo.INLET_NAMES},
        "wall_flux_total": sum(flux_th[n] for n in geo.WALL_NAMES),
        "div_L2": div_l2, "grad_L2": grad_l2, "div_L2_over_grad_L2": div_l2 / grad_l2,
        "integral_div": scal(ufl.div(uh) * ufl.dx),
        "velocity_L2": l2(uh), "fluid_area_m2": scal(one * ufl.dx),
        "pressure_min": float(ph.x.array.min()), "pressure_max": float(ph.x.array.max()),
        "outlets": outlets,
        "sign_convention": "flux = int u.n ds with outward n: inflow negative, outflow positive",
    }
    say(f"fluxes TH  : inlets {[round(flux_th[n], 10) for n in geo.INLET_NAMES]} outlets "
        f"{[round(flux_th[n], 6) for n in geo.OUTLET_NAMES]} net {net_th:.2e} "
        f"(normalized {th['normalized_imbalance_abs_net_over_abs_inflow']:.2e}); "
        f"||div u||/||grad u||={th['div_L2_over_grad_L2']:.2e}")

    # ---------------- transport P1 velocity in meshio node order
    V1 = fem.functionspace(msh, element("Lagrange", msh.basix_cell(), 1, shape=(2,), dtype=default_real_type))
    u1 = fem.Function(V1, name="velocity_P1")
    u1.interpolate(uh)   # P1 nodal interpolation = exact P2 vertex values
    from scipy.spatial import cKDTree
    coords = np.asarray(md["coord"], dtype=np.float64)
    tree = cKDTree(coords[:, :2])

    def to_transport_order(space, values, bs):
        dc = space.tabulate_dof_coordinates()[:, :2]
        d, idx = tree.query(dc)
        if d.max() > 1e-12 or len(np.unique(idx)) != coords.shape[0] or len(idx) != coords.shape[0]:
            raise RuntimeError(f"DOLFINx P1 dofs do not map 1:1 onto the transport nodes (max dist {d.max()})")
        out = np.empty((coords.shape[0], bs))
        out[idx] = values.reshape(-1, bs)
        return out, float(d.max())

    vel_nodes, map_dist_v = to_transport_order(V1, u1.x.array, 2)
    p_nodes, map_dist_p = to_transport_order(Q, ph.x.array, 1)
    p_nodes = p_nodes[:, 0]

    conn = np.asarray(md["connections"], dtype=int)
    X, Y = coords[:, 0], coords[:, 1]
    t = conn
    det = (X[t[:, 1]] - X[t[:, 0]]) * (Y[t[:, 2]] - Y[t[:, 0]]) - (X[t[:, 2]] - X[t[:, 0]]) * (Y[t[:, 1]] - Y[t[:, 0]])
    area = 0.5 * np.abs(det)
    bgrad = np.stack([Y[t[:, 1]] - Y[t[:, 2]], Y[t[:, 2]] - Y[t[:, 0]], Y[t[:, 0]] - Y[t[:, 1]]], 1) / det[:, None]
    cgrad = np.stack([X[t[:, 2]] - X[t[:, 1]], X[t[:, 0]] - X[t[:, 2]], X[t[:, 1]] - X[t[:, 0]]], 1) / det[:, None]
    uu, vv = vel_nodes[t, 0], vel_nodes[t, 1]
    dudx, dudy, dvdx, dvdy = (bgrad * uu).sum(1), (cgrad * uu).sum(1), (bgrad * vv).sum(1), (cgrad * vv).sum(1)
    div_p1 = dudx + dvdy
    p1_div_l2 = float(np.sqrt(np.sum(area * div_p1 ** 2)))
    p1_grad_l2 = float(np.sqrt(np.sum(area * (dudx ** 2 + dudy ** 2 + dvdx ** 2 + dvdy ** 2))))
    # boundary fluxes of the P1 field (trapezoid = exact for a linear field); outward
    # normal of each boundary edge oriented away from the third vertex of its triangle
    flux_p1 = {}
    edge_to_tri = {}
    for k, tri_ in enumerate(conn):
        for i, j, o in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            edge_to_tri[(min(tri_[i], tri_[j]), max(tri_[i], tri_[j]))] = tri_[o]
    for nm, edges in md["boundary_edges_by_name"].items():
        tot = 0.0
        for e0, e1 in edges:
            o = edge_to_tri[(min(e0, e1), max(e0, e1))]
            pa, pb, po = coords[e0, :2], coords[e1, :2], coords[o, :2]
            tv = pb - pa
            nn = np.array([tv[1], -tv[0]])
            if np.dot(nn, po - pa) > 0:
                nn = -nn
            tot += 0.5 * np.dot(vel_nodes[e0] + vel_nodes[e1], nn)   # |edge| * mean(u).n_hat
        flux_p1[nm] = float(tot)
    p1_in = sum(flux_p1[n] for n in geo.INLET_NAMES)
    p1_net = sum(flux_p1.values())
    p1 = {
        "flux_by_tag": flux_p1, "inflow_total": p1_in,
        "outflow_total": sum(flux_p1[n] for n in geo.OUTLET_NAMES),
        "net_boundary_flux": p1_net, "integral_div": float(np.sum(area * div_p1)),
        "normalized_imbalance_abs_net_over_abs_inflow": abs(p1_net) / abs(p1_in),
        "div_L2": p1_div_l2, "grad_L2": p1_grad_l2, "div_L2_over_grad_L2": p1_div_l2 / p1_grad_l2,
        "div_abs_max": float(np.abs(div_p1).max()),
        "wall_node_max_speed": float(np.max(np.linalg.norm(vel_nodes[np.unique(np.concatenate(
            [md["boundary_nodes_by_name"][n] for n in geo.WALL_NAMES]))], axis=1))),
        "dof_to_node_map_max_distance": max(map_dist_v, map_dist_p),
        "note": "continuous P1 interpolant of the P2 field at mesh vertices; not exactly divergence-free",
    }
    say(f"transport  : P1 net flux {p1_net:.2e} (normalized {p1['normalized_imbalance_abs_net_over_abs_inflow']:.2e}) "
        f"||div||/||grad||={p1['div_L2_over_grad_L2']:.2e}  wall |u|max={p1['wall_node_max_speed']:.1e}")

    # ---------------- probes at common physical locations
    grid_info, lines = probe_sets()
    gu, gmiss_u = eval_at(uh, grid_info["grid"])
    gp, gmiss_p = eval_at(ph, grid_info["grid"])
    line_vals, line_miss = {}, 0
    bay_flux = {}
    for name, pts in lines.items():
        lv, nmiss = eval_at(uh, pts)
        lp, _ = eval_at(ph, pts)
        line_miss += nmiss
        line_vals[name] = (pts, lv, lp[:, 0])
    pts, lv, _ = line_vals[f"h_y{geo.PARTITION_BOTTOM_Y:g}"]
    for bay, rng in geo.BAYS.items():
        m = (pts[:, 0] >= rng["x"][0]) & (pts[:, 0] <= rng["x"][1])
        xb, vy = pts[m, 0], lv[m, 1]
        bay_flux[str(bay)] = {
            "x_range": rng["x"],
            "net_upward_flux_across_y3": float(np.trapezoid(vy, xb)),
            "upward_part": float(np.trapezoid(np.maximum(vy, 0), xb)),
            "downward_part": float(np.trapezoid(np.minimum(vy, 0), xb)),
            "note": "per metre depth [m^2/s]; trapezoid on 5 mm samples of the P2 field; negative = net "
                    "flow from the bay down into the aisle",
        }
    pts, lv, _ = line_vals["doorway_x9"]
    doorway = {
        "y_range": list(geo.BATH_DOORWAY_Y),
        "net_flux_into_bathroom_+x": float(np.trapezoid(lv[:, 0], pts[:, 1])),
        "into_bathroom_part": float(np.trapezoid(np.maximum(lv[:, 0], 0), pts[:, 1])),
        "back_into_room_part": float(np.trapezoid(np.minimum(lv[:, 0], 0), pts[:, 1])),
        "note": "flux of the P2 field across x = 9, 0.6 < y < 1.4 (interior doorway), trapezoid on 1 mm samples",
    }
    say(f"probes     : grid {len(grid_info['grid'])} pts (missed {gmiss_u}), line samples missed {line_miss}; "
        f"bay net flux across y=3: " + ", ".join(f"bay{k}={v['net_upward_flux_across_y3']:.4f}"
                                                 for k, v in bay_flux.items())
        + f"; doorway net flux into bathroom {doorway['net_flux_into_bathroom_+x']:.4f}")
    say("outlets    : " + "; ".join(
        f"{nm}: flux {o['flux']:.5f} ({100 * o['share_of_total_inflow']:.1f}% of inflow), backflow "
        f"{o['backflow_flux_inward']:.2e} over {o['backflow_length_m']:.3f} m, min u.n {o['min_normal_velocity_at_P2_dofs']:.2e}"
        for nm, o in outlets.items()))

    # ---------------- save fields
    assoc = {
        "application": "Application2", "mesh_level": level, "mesh_file": str(msh_path),
        "mesh_sha256": md["mesh_sha256"], "geometry_sha256": md["geometry_sha256"],
        "transport_coords_sha256": mdc.coords_fingerprint(coords),
        "n_transport_nodes": int(coords.shape[0]), "n_triangles": int(conn.shape[0]),
        "rule": "use ONLY with mesh_data_config.build_mesh_data(level); verified by load_velocity_nodes()",
    }
    npz_nodes = mdc.velocity_npz_file(level, args.out_root)
    np.savez(npz_nodes, coords=coords, velocity_nodes=vel_nodes, pressure_nodes=p_nodes,
             connections=conn, association_json=np.array(json.dumps(assoc)))
    full_npz = out_dir / f"application2_navier_stokes_full_fields_{level}.npz"
    np.savez(full_npz,
             velocity_P2_dofs=uh.x.array.copy(), velocity_P2_block_size=2,
             velocity_P2_dof_coords=V.tabulate_dof_coordinates()[:, :2].copy(),
             velocity_P2_cell_dofs=np.asarray(V.dofmap.list).copy(),
             pressure_P1_dofs=ph.x.array.copy(),
             pressure_P1_dof_coords=Q.tabulate_dof_coordinates()[:, :2].copy(),
             pressure_P1_cell_dofs=np.asarray(Q.dofmap.list).copy(),
             mesh_geometry_x=msh.geometry.x[:, :2].copy(),
             mesh_cells=np.asarray(msh.geometry.dofmaps[0]).copy(),
             mesh_input_global_indices=np.asarray(msh.geometry.input_global_indices).copy(),
             picard_history=np.asarray(history, dtype=float),
             association_json=np.array(json.dumps(assoc)))
    probes_npz = out_dir / f"application2_probes_{level}.npz"
    pr = {"grid_points": grid_info["grid"], "grid_velocity": gu, "grid_pressure": gp[:, 0],
          "grid_x": grid_info["grid_shape_x"], "grid_y": grid_info["grid_shape_y"],
          "grid_keep_mask": grid_info["grid_keep_mask"]}
    for name, (pts, lv, lp) in line_vals.items():
        pr[f"line_{name}_points"] = pts
        pr[f"line_{name}_velocity"] = lv
        pr[f"line_{name}_pressure"] = lp
    np.savez(probes_npz, **pr)
    vis = []
    for fn, nm in ((uh, "velocity_P2"), (ph, "pressure_P1")):
        try:
            path = out_dir / f"application2_{nm}_{level}.bp"
            with io.VTXWriter(comm, path, [fn], "bp4") as wr:
                wr.write(0.0)
            vis.append(str(path))
        except Exception as exc:  # noqa: BLE001  (dof arrays in the full-field NPZ are the record)
            say(f"[vtx] {nm} skipped: {exc}")
    try:
        import meshio
        vtu = out_dir / f"application2_velocity_pressure_P1_{level}.vtu"
        vel3 = np.column_stack([vel_nodes, np.zeros(len(vel_nodes))])
        meshio.write(vtu, meshio.Mesh(coords, [("triangle", conn)],
                                      point_data={"velocity": vel3, "pressure": p_nodes,
                                                  "speed": np.linalg.norm(vel_nodes, axis=1)}), binary=True)
        vis.append(str(vtu))
    except Exception as exc:  # noqa: BLE001
        say(f"[vtu] skipped: {exc}")
    try:
        xdmf = out_dir / f"application2_velocity_pressure_P1_{level}.xdmf"
        with io.XDMFFile(comm, str(xdmf), "w") as f:
            f.write_mesh(msh)
            f.write_function(u1, 0.0)
            f.write_function(ph, 0.0)
        vis.append(str(xdmf))
    except Exception as exc:  # noqa: BLE001
        say(f"[xdmf] skipped: {exc}")
    hist_csv = out_dir / f"application2_picard_history_{level}.csv"
    hist_csv.write_text("iteration,update_norm_l2,relative_update,velocity_norm_l2\n"
                        + "".join(f"{i},{a:.12e},{b_:.12e},{c:.12e}\n" for i, a, b_, c in history))

    # ---------------- metadata
    def git(*a):
        try:
            return subprocess.run(["git", "-C", str(REPO_ROOT), *a], capture_output=True, text=True,
                                  check=True).stdout.strip()
        except Exception:  # noqa: BLE001
            return None
    import dolfinx, basix, scipy  # noqa: E401
    mesh_info = json.loads(mdc.mesh_info_file(level, args.out_root).read_text())
    status = "converged" if (converged and finite and null_pivots == 0) else "NOT_CONVERGED_OR_INVALID"
    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": status, "level": level, "purpose": geo.MESH_LEVEL_PURPOSE[level],
        "command": " ".join([sys.executable] + sys.argv),
        "pde": "-nu Laplace(v) + (v.grad)v + grad p = 0, div v = 0 (kinematic, vector-Laplacian viscous term)",
        "boundary_conditions": {
            "inlets": f"Dirichlet: parabolic upward profile, mean {cfg.INLET_MEAN_SPEED} m/s, peak {cfg.INLET_PEAK_SPEED} m/s",
            "inlets_location": "top wall y = 6, flow downward into the room",
            "walls_partitions_beds_bathroom_wall": "no slip v = 0 (tags 41-47, 51-52, 61-63, 71-72)",
            "beds": "idealized 2D impermeable obstacles: rectangular holes in the mesh",
            "bathroom_doorway": "interior fluid passage (no boundary condition)",
            "outlets": "outlet_door (y = 0) and outlet_exhaust (x = 11): natural (nu grad v - p I) n = 0; NO outlet "
                       "pressure value, NO outlet velocity, NO pinning, NO mean constraint; flow split is computed",
        },
        "parameters": cfg.as_dict(),
        "discretization": "Taylor-Hood P2 (vector) / P1, Picard (iteration 1 = Stokes), PETSc MUMPS LU",
        "petsc_options": petsc_opts,
        "mesh": {"vertices": int(n_vertices), "triangles": int(n_cells), "velocity_dofs": int(n_vdofs),
                 "pressure_dofs": int(n_pdofs), "total_dofs": int(n_vdofs + n_pdofs),
                 "mesh_info_file": str(mdc.mesh_info_file(level, args.out_root)),
                 "sizes_m": mesh_info["sizes_m"], "min_angle_deg": mesh_info["min_angle_deg"]},
        "association": assoc,
        "picard": {"converged": converged, "iterations": int(history[-1][0]),
                   "final_relative_update": float(history[-1][2]), "tolerance": cfg.PICARD_TOL,
                   "max_iter": cfg.MAX_PICARD_ITER, "seconds": picard_seconds, "finite": finite},
        "linear_system": {"size": list(A.getSize()), "last_relative_residual": lin_res,
                          "mumps_INFOG1": infog1, "mumps_null_pivots_INFOG28": null_pivots,
                          "velocity_dirichlet_dofs": n_dir_v, "pressure_dirichlet_dofs": n_dir_p},
        "nonlinear_residual_natural_outflow_check": residual,
        "taylor_hood_diagnostics": th,
        "transport_P1_diagnostics": p1,
        "bay_flux_across_partition_tip_line_y3": bay_flux,
        "bathroom_doorway_flux": doorway,
        "probes": {"file": str(probes_npz), "grid_spacing_m": GRID_SPACING, "line_spacing_m": LINE_SPACING,
                   "grid_points": int(len(grid_info["grid"])), "grid_missed": gmiss_u, "line_missed": line_miss},
        "files": {"velocity_nodes_npz": str(npz_nodes), "full_fields_npz": str(full_npz),
                  "picard_history_csv": str(hist_csv), "visualization": vis},
        "npz_schema": {
            "velocity_nodes_npz": {"coords": "(N,3) transport node coords (meshio order)",
                                   "velocity_nodes": "(N,2) P2 velocity at the vertices",
                                   "pressure_nodes": "(N,) P1 pressure", "connections": "(T,3)",
                                   "association_json": "mesh/geometry hashes; checked by load_velocity_nodes"},
            "full_fields_npz": "P2 velocity / P1 pressure dof arrays + dof coordinates + cell dof maps + DOLFINx "
                               "geometry (reload: read the .msh serially with dolfinx.io.gmsh, same version)",
        },
        "provenance": {
            "git_commit": git("rev-parse", "HEAD"), "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty_paths": git("status", "--porcelain"),
            "python": sys.version, "platform": platform.platform(),
            "dolfinx": dolfinx.__version__, "basix": basix.__version__, "petsc4py": PETSc.Sys.getVersion(),
            "numpy": np.__version__, "scipy": scipy.__version__,
        },
        "wall_seconds_total": time.perf_counter() - t_start,
    }
    meta_path = out_dir / f"application2_navier_stokes_run_metadata_{level}.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=str) + "\n")
    say(f"status     : {status}")
    say(f"metadata   : {meta_path}")
    return 0 if status == "converged" else 2


if __name__ == "__main__":
    sys.exit(main())
