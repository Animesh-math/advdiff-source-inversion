"""Controlled pair comparisons + boundary-wise observability for App1 (diagnostic only).

Part 1 (task 6): for four specified center pairs, jointly fit bounded amplitudes
in the exact discrete observation norm, then compare
  * the EXACT forward objective (one real multi-source forward solve +
    ``gradient_test.compute_objective``), and
  * the quadratic pair model assembled from cached unit-source boundary traces.
Because the transport PDE is linear in the source amplitudes these must agree to
roundoff for fixed centers and fixed sigma2; any mismatch is a real bug.

Part 2 (task 8): boundary-resolved observability of each true source --
per-boundary unit-response energy, data energy, kappa contribution, and the
coherence between the two sources' traces.

Run (from this directory):
    python3 \
        diagnose_sensitivity_pairs_and_observability.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import sys as _sys_dc
from pathlib import Path as _Path_dc
_sys_dc.path.insert(0, str(_Path_dc(__file__).resolve().parents[1]))  # _diag_common
import _diag_common as D
import run_optimization as ro
import gradient_test as gt
import sensitivity_initialization as sinit
import source_insertion_sensitivity as sis


OUT = D.APP_DIR / "output" / "sensitivity_pair_diagnostics"
TRUE1 = np.array([250.0, 130.0])
TRUE2 = np.array([15.0, 580.0])
LO, HI = D.AMPLITUDE_BOUNDS


# --------------------------------------------------------------------------
# Part 1: controlled pair comparisons
# --------------------------------------------------------------------------
def screened_nearest(mesh_data, pde, admiss, obs_state, top_k=24):
    """Re-run the production level-1 screen and return its nearest candidates."""
    coord = np.asarray(mesh_data["coord"], dtype=np.float64)[:, :2]
    outer = sinit._outer_boundary_node_coords(mesh_data, coord)
    bld = sinit._building_boundary_node_coords(mesh_data, coord)
    mask = sinit._base_admissibility_mask(coord, admiss, outer, bld, 0.0, 0.0)
    _, _sa, _im, W = sis.compute_adjoint_weight_field(
        mesh_data, pde, sinit._DUMMY_Z, obs_state, ro.ETA1,
        ro.OBJECTIVE_TIME_RULE, dirichlet_tags=("left",),
        state_out=np.zeros_like(obs_state),
    )
    qx, qy, ag = sis.precompute_quadrature_weights(mesh_data, pde, W)
    g = sis.sensitivity_at_candidates(coord, qx, qy, ag, pde["sigma2"])
    radius = 2.0 * np.sqrt(pde["sigma2"])
    scr = sinit._topk_diverse(g, np.flatnonzero(mask), coord, top_k, radius)
    scr = [i for i in scr if g[i] < 0.0]
    xy = coord[scr]
    n1 = scr[int(np.argmin(np.hypot(xy[:, 0] - TRUE1[0], xy[:, 1] - TRUE1[1])))]
    n2 = scr[int(np.argmin(np.hypot(xy[:, 0] - TRUE2[0], xy[:, 1] - TRUE2[1])))]
    return coord[n1].copy(), coord[n2].copy(), len(scr)


def pair_report(tag, boundaries, sigma2):
    print(f"\n{'=' * 78}\n=== PAIR COMPARISON  {tag}  boundaries={list(boundaries)} "
          f"sigma2={sigma2:g}\n{'=' * 78}")
    mesh_data, pde, admiss = D.build_setup(boundaries, sigma2)
    ctx = D.obs_ctx(mesh_data, pde)
    obs_state = D.true_state(mesh_data, pde)
    ud = ctx.extract(obs_state)
    j_zero = ctx.tracking_objective(np.zeros_like(ud), ud)

    cA1, cA2, n_scr = screened_nearest(mesh_data, pde, admiss, obs_state)
    print(f"screened set size {n_scr}; nearest to TRUE1 ({cA1[0]:.2f},{cA1[1]:.2f}) "
          f"d={np.hypot(*(cA1-TRUE1)):.2f} m; nearest to TRUE2 "
          f"({cA2[0]:.2f},{cA2[1]:.2f}) d={np.hypot(*(cA2-TRUE2)):.2f} m")

    pairs = {
        "A_nearest_screened": (cA1, cA2),
        "B_initializer_selected": (np.array([211.57190537278075, 173.41167643928202]),
                                   np.array([285.1289895203663, 126.40948346827645])),
        "C_true_centers": (TRUE1.copy(), TRUE2.copy()),
        "D_final_collapsed": (np.array([250.16547859860574, 129.91362435515708]),
                              np.array([196.90916378875914, 169.33074286874367])),
    }

    trace_cache = {}

    def trace(c):
        key = (round(float(c[0]), 9), round(float(c[1]), 9))
        if key not in trace_cache:
            trace_cache[key] = ctx.extract(D.unit_state(mesh_data, pde, c[0], c[1]))
        return trace_cache[key]

    rows = []
    for name, (c1, c2) in pairs.items():
        t1, t2 = trace(c1), trace(c2)
        G = np.array([[ctx.inner(t1, t1), ctx.inner(t1, t2)],
                      [ctx.inner(t2, t1), ctx.inner(t2, t2)]])
        b = np.array([-ctx.inner(t1, ud), -ctx.inner(t2, ud)])
        amps = sinit._solve_box_qp(G, b, LO, HI)
        a_unb = np.linalg.solve(G, -b)
        # quadratic model objective
        j_model = float(j_zero + 0.5 * amps @ G @ amps + b @ amps)
        # exact forward objective: one real multi-source forward solve
        z = np.array([c1[0], c1[1], amps[0], c2[0], c2[1], amps[1]])
        st = gt.run_forward(mesh_data, pde, z)
        j_exact = float(gt.compute_objective(
            st, obs_state, mesh_data, eta1=ro.ETA1, lam=ro.LAMBDA_REG, z=z,
            dt_out=float(pde["dt"] * pde["jprint"]),
            time_rule=ro.OBJECTIVE_TIME_RULE,
        ))
        # trace-space objective of the fitted linear combination
        j_trace = ctx.tracking_objective(amps[0] * t1 + amps[1] * t2, ud)
        resid = amps[0] * t1 + amps[1] * t2 - ud
        rows.append(dict(
            pair=name,
            x1=float(c1[0]), y1=float(c1[1]), x2=float(c2[0]), y2=float(c2[1]),
            separation=float(np.hypot(*(c1 - c2))),
            a1=float(amps[0]), a2=float(amps[1]),
            a1_unbounded=float(a_unb[0]), a2_unbounded=float(a_unb[1]),
            gram_11=float(G[0, 0]), gram_12=float(G[0, 1]), gram_22=float(G[1, 1]),
            coherence=float(G[0, 1] / np.sqrt(G[0, 0] * G[1, 1])),
            J_quadratic_model=j_model,
            J_trace_space=float(j_trace),
            J_exact_forward=j_exact,
            rel_model_vs_exact=float(abs(j_model - j_exact) / max(abs(j_exact), 1e-30)),
            residual_norm=float(np.sqrt(ctx.norm2(resid))),
            data_norm=float(np.sqrt(ctx.norm2(ud))),
            frac_explained=float(1.0 - j_exact / j_zero),
        ))
        r = rows[-1]
        print(f"\n{name}:")
        print(f"  centers ({c1[0]:.3f},{c1[1]:.3f}) + ({c2[0]:.3f},{c2[1]:.3f})  "
              f"sep={r['separation']:.1f} m  coherence={r['coherence']:.6f}")
        print(f"  amps bounded ({amps[0]:.6f}, {amps[1]:.6f})   "
              f"unbounded ({a_unb[0]:.6f}, {a_unb[1]:.6f})")
        print(f"  J quadratic model = {j_model:.10e}")
        print(f"  J exact forward   = {j_exact:.10e}   rel diff "
              f"{r['rel_model_vs_exact']:.3e}")
        print(f"  residual/data norm = {r['residual_norm']:.4e} / {r['data_norm']:.4e}"
              f"   explained {100*r['frac_explained']:.4f}%")

    with open(OUT / f"pair_comparison_{tag}_sigma{int(sigma2)}.json", "w") as fh:
        json.dump(dict(tag=tag, boundaries=list(boundaries), sigma2=sigma2,
                       j_zero=float(j_zero), pairs=rows), fh, indent=1)
    return rows, float(j_zero)


# --------------------------------------------------------------------------
# Part 2: boundary-wise observability
# --------------------------------------------------------------------------
def observability_report(sigma2, tfinal=D.TFINAL):
    print(f"\n{'=' * 78}\n=== BOUNDARY-WISE OBSERVABILITY  sigma2={sigma2:g} "
          f"tfinal={tfinal:g}\n{'=' * 78}")
    names = ("left", "right", "bottom", "top")
    mesh_data, pde, _ = D.build_setup(names, sigma2, tfinal)
    dt_out = float(pde["dt"] * pde["jprint"])

    # per-boundary trace contexts (single-name measurement boundary)
    ctxs = {}
    for n in names:
        md = dict(mesh_data)
        md["measurement_boundary"] = n
        ctxs[n] = sinit._ObsTraceContext(md, ro.ETA1, dt_out, ro.OBJECTIVE_TIME_RULE)
    ctx_union = sinit._ObsTraceContext(
        dict(mesh_data, measurement_boundary=mesh_data["measurement_boundary"]),
        ro.ETA1, dt_out, ro.OBJECTIVE_TIME_RULE)

    probes = {
        "unit_source_1@(250,130)": (TRUE1, 1.0),
        "unit_source_2@(15,580)": (TRUE2, 1.0),
        "collapsed@(199,169)": (np.array([199.0, 169.0]), 1.0),
    }
    states = {k: D.unit_state(mesh_data, pde, c[0], c[1]) for k, (c, _) in probes.items()}

    rows = []
    print(f"\n{'probe':26s} {'boundary':9s} {'kappa_b = ||du||^2_b':>22s} "
          f"{'share':>9s} {'peak |u|':>12s}")
    for k, st in states.items():
        tot = sum(ctxs[n].norm2(ctxs[n].extract(st)) for n in names)
        for n in names:
            t = ctxs[n].extract(st)
            kb = ctxs[n].norm2(t)
            rows.append(dict(probe=k, boundary=n, kappa_boundary=float(kb),
                             share=float(kb / tot) if tot > 0 else 0.0,
                             peak_abs=float(np.max(np.abs(t))),
                             n_edges=int(ctxs[n]._elen.size)))
            print(f"{k:26s} {n:9s} {kb:22.6e} {100*kb/tot:8.4f}% "
                  f"{np.max(np.abs(t)):12.4e}")
        print(f"{'':26s} {'TOTAL':9s} {tot:22.6e}")

    # data energy actually produced by each TRUE source at its true amplitude
    print("\n--- data energy per boundary from each TRUE source (true amplitudes) ---")
    amps = {"unit_source_1@(250,130)": 1.60, "unit_source_2@(15,580)": 0.98}
    energy = {}
    for k, a in amps.items():
        st = states[k]
        for n in names:
            e = a * a * ctxs[n].norm2(ctxs[n].extract(st))
            energy[(k, n)] = e
    print(f"{'boundary':9s} {'src1 energy':>16s} {'src2 energy':>16s} {'src2/src1':>12s}")
    for n in names:
        e1 = energy[("unit_source_1@(250,130)", n)]
        e2 = energy[("unit_source_2@(15,580)", n)]
        print(f"{n:9s} {e1:16.6e} {e2:16.6e} {e2/e1 if e1>0 else float('nan'):12.4e}")

    # coherence between the two true sources' traces on each boundary set
    print("\n--- coherence between the two true sources' unit traces ---")
    coh = {}
    s1, s2 = states["unit_source_1@(250,130)"], states["unit_source_2@(15,580)"]
    for n in list(names) + ["union(l+r+b)", "union(l+r+b+t)"]:
        if n in names:
            c = ctxs[n]
            t1, t2 = c.extract(s1), c.extract(s2)
        else:
            sel = ("left", "right", "bottom") if "t)" not in n else names
            md2 = ro.configure_measurement_boundaries(ro.build_mesh_data(), list(sel))
            c = sinit._ObsTraceContext(md2, ro.ETA1, dt_out, ro.OBJECTIVE_TIME_RULE)
            t1, t2 = c.extract(s1), c.extract(s2)
        n11, n22 = c.norm2(t1), c.norm2(t2)
        v = float(c.inner(t1, t2) / np.sqrt(n11 * n22)) if n11 > 0 and n22 > 0 else float("nan")
        coh[n] = v
        print(f"{n:16s} coherence = {v: .6f}   kappa1={n11:.4e} kappa2={n22:.4e} "
              f"ratio={n22/n11 if n11>0 else float('nan'):.4e}")

    # what does adding 'top' change for the union objective / data?
    print("\n--- union comparison: does 'top' carry information? ---")
    obs_state = D.true_state(mesh_data, pde)
    for sel in (("left", "right", "bottom"), names):
        md2 = ro.configure_measurement_boundaries(ro.build_mesh_data(), list(sel))
        c = sinit._ObsTraceContext(md2, ro.ETA1, dt_out, ro.OBJECTIVE_TIME_RULE)
        ud = c.extract(obs_state)
        print(f"  {'+'.join(sel):28s} J(zero)=0.5||u_d||^2 = "
              f"{c.tracking_objective(np.zeros_like(ud), ud):.10e}  "
              f"edges={c._elen.size}")

    with open(OUT / f"observability_sigma{int(sigma2)}.json", "w") as fh:
        json.dump(dict(sigma2=sigma2, tfinal=tfinal, rows=rows,
                       data_energy={f"{k[0]}|{k[1]}": float(v)
                                    for k, v in energy.items()},
                       coherence=coh), fh, indent=1)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pair_report("lrb", D.BOUNDARY_SETS["lrb"], 1600.0)
    pair_report("lrbt", D.BOUNDARY_SETS["lrbt"], 1600.0)
    pair_report("lrb", D.BOUNDARY_SETS["lrb"], 400.0)
    observability_report(1600.0)
    observability_report(400.0)
    print(f"\ntotal wall time {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
