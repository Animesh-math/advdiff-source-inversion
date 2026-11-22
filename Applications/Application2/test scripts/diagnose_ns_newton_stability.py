"""Diagnostic: is a Picard failure numerical or physical?  (Application2)

For given (U_mean, nu) on one mesh level:
  1. Picard iterations (as in the production solver) until the relative update
     is below --picard-switch or --picard-iters is reached;
  2. Newton (PETSc SNES newtonls + backtracking line search, MUMPS LU) from that
     iterate on the SAME discrete equations (exact Jacobian incl. (du.grad)u);
  3. If Newton converges: linear stability of the steady discrete solution,
         J x = lambda M x   (M = velocity mass; Dirichlet rows and pressure -> infinite),
     perturbations ~ exp(-lambda t): a mode is unstable if Re(lambda) < 0.
     SLEPc shift-invert around several targets; the eigenvalues of smallest
     real part found are reported.

Same weak form and BCs as the production solver (natural outflow, no pressure
constraint).  Writes JSON to --out.  Run serially with fenicsx-env.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from mpi4py import MPI
from petsc4py import PETSc
import numpy as np
import ufl
from basix.ufl import element
from dolfinx import default_real_type, fem
from dolfinx.fem.petsc import LinearProblem, NonlinearProblem, assemble_matrix
from dolfinx.io import gmsh as dolfinx_gmsh

APP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
import room_geometry_config as geo  # noqa: E402
import mesh_data_config as mdc  # noqa: E402
import navier_stokes_config as cfg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="coarse")
    ap.add_argument("--mesh-root", type=Path, required=True)
    ap.add_argument("--U", type=float, required=True, help="mean inlet speed [m/s]")
    ap.add_argument("--nu", type=float, required=True)
    ap.add_argument("--picard-iters", type=int, default=40)
    ap.add_argument("--picard-switch", type=float, default=1e-3)
    ap.add_argument("--nev", type=int, default=12)
    ap.add_argument("--targets", type=str, default="0,0.02,0.05")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    comm = MPI.COMM_WORLD
    t0 = time.perf_counter()
    msh_path = mdc.mesh_file(a.level, a.mesh_root)
    md = dolfinx_gmsh.read_from_msh(str(msh_path), comm, rank=0, gdim=2)
    msh, ft = md.mesh, md.facet_tags
    fdim = 1
    msh.topology.create_connectivity(1, 2)
    V = fem.functionspace(msh, element("Lagrange", msh.basix_cell(), 2, shape=(2,), dtype=default_real_type))
    Q = fem.functionspace(msh, element("Lagrange", msh.basix_cell(), 1, dtype=default_real_type))
    fac = lambda tags: np.concatenate([ft.find(int(t)) for t in tags]).astype(np.int32)  # noqa: E731
    peak = 1.5 * a.U

    def prof(x):
        return cfg.inlet_velocity(x, peak)

    g = fem.Function(V)
    g.interpolate(prof)
    bcs = [fem.dirichletbc(g, fem.locate_dofs_topological(V, fdim, fac(geo.INLET_TAG_LIST))),
           fem.dirichletbc(np.zeros(2, dtype=PETSc.ScalarType),
                           fem.locate_dofs_topological(V, fdim, fac(geo.WALL_TAG_LIST)), V)]
    nu = fem.Constant(msh, PETSc.ScalarType(a.nu))
    zero = fem.Constant(msh, PETSc.ScalarType(0.0))
    u, p = ufl.TrialFunction(V), ufl.TrialFunction(Q)
    w, q = ufl.TestFunction(V), ufl.TestFunction(Q)
    uh, ph, af = fem.Function(V), fem.Function(Q), fem.Function(V)
    opts = {"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"}

    # ---- 1. Picard
    A_p = [[(nu * ufl.inner(ufl.grad(u), ufl.grad(w)) + ufl.inner(ufl.dot(ufl.grad(u), af), w)) * ufl.dx,
            -ufl.inner(p, ufl.div(w)) * ufl.dx],
           [ufl.inner(ufl.div(u), q) * ufl.dx, zero * p * q * ufl.dx]]
    L_p = [ufl.inner(fem.Constant(msh, np.zeros(2)), w) * ufl.dx, ufl.ZeroBaseForm((q,))]
    lp = LinearProblem(A_p, L_p, bcs=bcs, u=[uh, ph], kind="mpi", petsc_options_prefix="pic_",
                       petsc_options=opts)
    l2 = lambda e: float(np.sqrt(fem.assemble_scalar(fem.form(ufl.inner(e, e) * ufl.dx))))  # noqa: E731
    pic = []
    for k in range(1, a.picard_iters + 1):
        lp.solve()
        rel = l2(uh - af) / l2(uh)
        pic.append(rel)
        af.x.array[:] = uh.x.array
        if rel < a.picard_switch:
            break
    print(f"Picard: {len(pic)} it, last rel update {pic[-1]:.2e}", flush=True)

    # ---- 2. Newton on F(u,p) = 0
    F = [(nu * ufl.inner(ufl.grad(uh), ufl.grad(w)) + ufl.inner(ufl.dot(ufl.grad(uh), uh), w)
          - ph * ufl.div(w)) * ufl.dx,
         ufl.inner(ufl.div(uh), q) * ufl.dx]
    du, dp = ufl.TrialFunction(V), ufl.TrialFunction(Q)
    J = [[ufl.derivative(F[0], uh, du), ufl.derivative(F[0], ph, dp)],
         [ufl.derivative(F[1], uh, du), zero * dp * q * ufl.dx]]
    newton_hist = []
    nopts = {"snes_type": "newtonls", "snes_linesearch_type": "bt", "snes_rtol": 1e-12, "snes_atol": 1e-13,
             "snes_stol": 0.0, "snes_max_it": 30, **opts}
    prob = NonlinearProblem(F, [uh, ph], bcs=bcs, J=J, kind="mpi", petsc_options_prefix="nwt_",
                            petsc_options=nopts)
    snes = prob.solver
    snes.setMonitor(lambda s, it, r: newton_hist.append(float(r)))
    try:
        prob.solve()
    except Exception as exc:  # noqa: BLE001
        print("newton raised", exc, flush=True)
    reason = int(snes.getConvergedReason())
    nits = int(snes.getIterationNumber())
    converged = reason > 0
    print(f"Newton: reason={reason} its={nits} residuals={['%.2e' % r for r in newton_hist]}", flush=True)
    out = {"level": a.level, "U_mean": a.U, "nu": a.nu, "Re_slot": a.U * geo.VENT_WIDTH / a.nu,
           "Re_room": a.U * 6.0 / a.nu, "picard_iterations": len(pic), "picard_last_rel_update": pic[-1],
           "picard_history": pic, "newton_converged": converged, "newton_reason": reason,
           "newton_iterations": nits, "newton_residual_history": newton_hist}

    if converged:
        # bay-entrance fluxes to compare with Picard solutions
        out["velocity_L2"] = l2(uh)
        ds = ufl.Measure("ds", domain=msh, subdomain_data=ft)
        n = ufl.FacetNormal(msh)
        out["outlet_flux"] = {nm: float(fem.assemble_scalar(fem.form(ufl.dot(uh, n) * ds(t))))
                              for nm, t in geo.OUTLET_TAGS.items()}
        # ---- 3. linear stability
        from slepc4py import SLEPc
        Jm = assemble_matrix(fem.form(J), bcs=bcs, kind="mpi")
        Jm.assemble()
        Mf = [[ufl.inner(du, w) * ufl.dx, zero * ufl.inner(dp, ufl.div(w)) * ufl.dx],
              [zero * ufl.inner(ufl.div(du), q) * ufl.dx, zero * dp * q * ufl.dx]]
        Mm = assemble_matrix(fem.form(Mf), bcs=bcs, diag=0.0, kind="mpi")
        Mm.assemble()
        eigs = []
        for tgt in [float(t) for t in a.targets.split(",")]:
            E = SLEPc.EPS().create(comm)
            E.setOperators(Jm, Mm)
            E.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
            E.setDimensions(a.nev)
            E.setTarget(tgt)
            E.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
            st = E.getST()
            st.setType("sinvert")
            ksp = st.getKSP()
            ksp.setType("preonly")
            ksp.getPC().setType("lu")
            ksp.getPC().setFactorSolverType("mumps")
            E.setTolerances(1e-10, 500)
            E.solve()
            for i in range(E.getConverged()):
                lam = E.getEigenvalue(i)
                eigs.append((float(lam.real), float(lam.imag), tgt))
            E.destroy()
        uniq = []
        for re_, im_, tg in sorted(eigs):
            if not any(abs(re_ - r) < 1e-8 * max(1, abs(re_)) and abs(im_ - i) < 1e-8 * max(1, abs(im_))
                       for r, i, _ in uniq):
                uniq.append((re_, im_, tg))
        out["eigenvalues_smallest_real_part"] = [{"re": r, "im": i, "target": t} for r, i, t in uniq[:12]]
        lead = uniq[0]
        out["leading_eigenvalue"] = {"re": lead[0], "im": lead[1]}
        out["linearly_stable"] = bool(lead[0] > 0)
        out["eigen_note"] = ("J x = lambda M x; perturbation ~ exp(-lambda t). Re(lambda) < 0 => unstable. "
                             "Shift-invert finds eigenvalues nearest each target only, so this is a local search "
                             "for the least-stable modes, not a full spectrum.")
        print("leading eigenvalues (re, im):", [(round(r, 6), round(i, 6)) for r, i, _ in uniq[:6]], flush=True)
    out["seconds"] = time.perf_counter() - t0
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
