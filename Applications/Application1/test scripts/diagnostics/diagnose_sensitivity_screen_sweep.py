"""Screen-resolution / separation sweep for the App1 sensitivity init (diagnostic only).

Answers the remedy questions raised by ``diagnose_sensitivity_pair_selection.py``:

  * how far from each true source is the nearest dictionary atom as a function of
    (top_k, diversity radius)?
  * with an exhaustive pair search over that dictionary, is the winning pair
    basin-correct (one atom near each true source)?
  * what does a minimum source separation constraint do to the production beam?
  * how much of the objective gap is pure center-discretization error on the
    dominant source (i.e. would be removed for free by L-BFGS-B)?

No production module is modified; all screening/scoring helpers are the ones
``sensitivity_initialization`` itself uses.
"""

from __future__ import annotations

import itertools
import json

import numpy as np

import sys as _sys_dc
from pathlib import Path as _Path_dc
_sys_dc.path.insert(0, str(_Path_dc(__file__).resolve().parents[1]))  # _diag_common
import _diag_common as D
import run_optimization as ro
import sensitivity_initialization as sinit
import source_insertion_sensitivity as sis


OUT = D.APP_DIR / "output" / "sensitivity_pair_diagnostics"
TRUE1 = np.array([250.0, 130.0])
TRUE2 = np.array([15.0, 580.0])
LO, HI = D.AMPLITUDE_BOUNDS
BASIN_TOL = 60.0  # "in the right basin" = within 60 m of a true center


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sigma2 = 1600.0
    mesh_data, pde, admiss = D.build_setup(D.BOUNDARY_SETS["lrb"], sigma2)
    coord = np.asarray(mesh_data["coord"], dtype=np.float64)[:, :2]
    ctx = D.obs_ctx(mesh_data, pde)
    obs_state = D.true_state(mesh_data, pde)
    ud = ctx.extract(obs_state)
    j_zero = ctx.tracking_objective(np.zeros_like(ud), ud)

    outer = sinit._outer_boundary_node_coords(mesh_data, coord)
    bld = sinit._building_boundary_node_coords(mesh_data, coord)
    static_mask = sinit._base_admissibility_mask(coord, admiss, outer, bld, 0.0, 0.0)

    _, _sa, _im, W = sis.compute_adjoint_weight_field(
        mesh_data, pde, sinit._DUMMY_Z, obs_state, ro.ETA1,
        ro.OBJECTIVE_TIME_RULE, dirichlet_tags=("left",),
        state_out=np.zeros_like(obs_state),
    )
    qx, qy, ag = sis.precompute_quadrature_weights(mesh_data, pde, W)
    g1 = sis.sensitivity_at_candidates(coord, qx, qy, ag, pde["sigma2"])

    cache = {}

    def trace(node):
        if node not in cache:
            cache[node] = ctx.extract(D.unit_state(mesh_data, pde, *coord[node]))
        return cache[node]

    def fit_pair(ni, nj):
        t1, t2 = trace(ni), trace(nj)
        G = np.array([[ctx.inner(t1, t1), ctx.inner(t1, t2)],
                      [ctx.inner(t2, t1), ctx.inner(t2, t2)]])
        b = np.array([-ctx.inner(t1, ud), -ctx.inner(t2, ud)])
        a = sinit._solve_box_qp(G, b, LO, HI)
        return a, float(j_zero + 0.5 * a @ G @ a + b @ a)

    def fit_one(ni):
        t = trace(ni)
        g = ctx.inner(t, t)
        b = -ctx.inner(t, ud)
        a = float(np.clip(-b / g, LO, HI))
        return a, float(j_zero + 0.5 * a * a * g + b * a)

    print(f"J0 = {j_zero:.6e}   admissible nodes {int(static_mask.sum())}")

    # ---------------- reference points -----------------------------------
    n_true1 = int(np.argmin(np.hypot(coord[:, 0] - TRUE1[0], coord[:, 1] - TRUE1[1])))
    n_true2 = int(np.argmin(np.hypot(coord[:, 0] - TRUE2[0], coord[:, 1] - TRUE2[1])))
    d1 = float(np.hypot(*(coord[n_true1] - TRUE1)))
    d2 = float(np.hypot(*(coord[n_true2] - TRUE2)))
    a1s, j1s = fit_one(n_true1)
    a2s, j2s = fit_one(n_true2)
    amps_bn, j_bestnodes = fit_pair(n_true1, n_true2)
    print("\n--- reference: nearest MESH NODES to the true centers ---")
    print(f"node@TRUE1 ({coord[n_true1][0]:.2f},{coord[n_true1][1]:.2f}) d={d1:.2f} m  "
          f"alone: a={a1s:.4f} J={j1s:.6e} ({100*(1-j1s/j_zero):.4f}% explained)")
    print(f"node@TRUE2 ({coord[n_true2][0]:.2f},{coord[n_true2][1]:.2f}) d={d2:.2f} m  "
          f"alone: a={a2s:.4f} J={j2s:.6e} ({100*(1-j2s/j_zero):.4f}% explained)")
    print(f"PAIR of best mesh nodes: amps ({amps_bn[0]:.4f},{amps_bn[1]:.4f}) "
          f"J={j_bestnodes:.6e} ({100*(1-j_bestnodes/j_zero):.4f}% explained)")
    print(f"  -> compare initializer selection J=1.743488e+07, "
          f"near-true screened pair J=5.111758e+07")

    # rank of the true-source-1 node in the raw G field
    order = np.argsort(g1[static_mask])
    idx_sorted = np.flatnonzero(static_mask)[order]
    rank_t1 = int(np.flatnonzero(idx_sorted == n_true1)[0]) + 1
    rank_t2 = int(np.flatnonzero(idx_sorted == n_true2)[0]) + 1
    print(f"\nraw-G rank of node@TRUE1: {rank_t1}/{static_mask.sum()}  "
          f"G={g1[n_true1]:.4e}")
    print(f"raw-G rank of node@TRUE2: {rank_t2}/{static_mask.sum()}  "
          f"G={g1[n_true2]:.4e}")

    # ---------------- screen resolution sweep ----------------------------
    print("\n" + "=" * 78)
    print("=== dictionary sweep: exhaustive pair search over the screened set ===")
    print("=" * 78)
    results = []
    for top_k, radius in [(24, 80.0), (24, 40.0), (24, 20.0),
                          (48, 80.0), (48, 40.0), (48, 20.0)]:
        scr = sinit._topk_diverse(g1, np.flatnonzero(static_mask), coord, top_k, radius)
        scr = [i for i in scr if g1[i] < 0.0]
        xy = coord[scr]
        nd1 = float(np.min(np.hypot(xy[:, 0] - TRUE1[0], xy[:, 1] - TRUE1[1])))
        nd2 = float(np.min(np.hypot(xy[:, 0] - TRUE2[0], xy[:, 1] - TRUE2[1])))
        best = None
        for i, j in itertools.combinations(range(len(scr)), 2):
            a, J = fit_pair(scr[i], scr[j])
            if best is None or J < best[1]:
                best = ((scr[i], scr[j]), J, a)
        (bi, bj), bJ, ba = best
        bd1 = min(np.hypot(*(coord[bi] - TRUE1)), np.hypot(*(coord[bj] - TRUE1)))
        bd2 = min(np.hypot(*(coord[bi] - TRUE2)), np.hypot(*(coord[bj] - TRUE2)))
        basin_ok = bool(bd1 <= BASIN_TOL and bd2 <= BASIN_TOL)
        # best pair that IS basin correct
        best_ok = None
        for i, j in itertools.combinations(range(len(scr)), 2):
            ci, cj = coord[scr[i]], coord[scr[j]]
            ok = (min(np.hypot(*(ci - TRUE1)), np.hypot(*(cj - TRUE1))) <= BASIN_TOL
                  and min(np.hypot(*(ci - TRUE2)), np.hypot(*(cj - TRUE2))) <= BASIN_TOL)
            if not ok:
                continue
            a, J = fit_pair(scr[i], scr[j])
            if best_ok is None or J < best_ok[1]:
                best_ok = ((scr[i], scr[j]), J, a)
        row = dict(top_k=top_k, radius=radius, n_screened=len(scr),
                   nearest_true1=nd1, nearest_true2=nd2,
                   best_pair=[coord[bi].tolist(), coord[bj].tolist()],
                   best_pair_J=float(bJ), best_pair_amps=[float(x) for x in ba],
                   best_pair_basin_correct=basin_ok,
                   best_basin_correct_J=(float(best_ok[1]) if best_ok else None),
                   n_solves_so_far=len(cache))
        results.append(row)
        print(f"\nK={top_k:3d} radius={radius:5.1f} m -> {len(scr):2d} atoms; "
              f"nearest to TRUE1 {nd1:6.2f} m, to TRUE2 {nd2:6.2f} m")
        print(f"   argmin-J pair ({coord[bi][0]:6.1f},{coord[bi][1]:6.1f})+"
              f"({coord[bj][0]:6.1f},{coord[bj][1]:6.1f}) "
              f"J={bJ:.4e} amps=({ba[0]:.3f},{ba[1]:.3f})  "
              f"basin-correct={basin_ok}")
        if best_ok:
            (oi, oj), oJ, oa = best_ok
            print(f"   best BASIN-CORRECT pair ({coord[oi][0]:6.1f},{coord[oi][1]:6.1f})+"
                  f"({coord[oj][0]:6.1f},{coord[oj][1]:6.1f}) J={oJ:.4e} "
                  f"(ratio to argmin-J pair {oJ/bJ:.2f}x)")
        else:
            print("   no basin-correct pair exists in this dictionary")
        print(f"   cumulative unit forward solves: {len(cache)}")

    # ---------------- separation-constraint effect on the production beam --
    print("\n" + "=" * 78)
    print("=== effect of a minimum source separation on the production beam ===")
    print("=" * 78)
    scr24 = sinit._topk_diverse(g1, np.flatnonzero(static_mask), coord, 24, 80.0)
    scr24 = [i for i in scr24 if g1[i] < 0.0]
    recs = []
    for i in scr24:
        t = trace(i)
        kap = ctx.norm2(t)
        gi = float(g1[i])
        a1d = float(np.clip(-gi / kap, LO, HI))
        recs.append((i, a1d * gi + 0.5 * a1d * a1d * kap))
    recs.sort(key=lambda r: r[1])
    beam_first = [r[0] for r in recs[:3]]
    print(f"level-1 beam (top 3 by 1-D predicted reduction): "
          + ", ".join(f"({coord[n][0]:.1f},{coord[n][1]:.1f})" for n in beam_first))

    sep_rows = []
    for sep in (0.0, 40.0, 100.0, 150.0, 200.0, 250.0):
        best = None
        for nf in beam_first:
            a_f, _ = fit_one(nf)
            for ns in scr24:
                if ns == nf:
                    continue
                if np.hypot(*(coord[ns] - coord[nf])) < sep:
                    continue
                a, J = fit_pair(nf, ns)
                if best is None or J < best[1]:
                    best = ((nf, ns), J, a)
        if best is None:
            print(f"sep>={sep:6.1f} m : no admissible pair")
            continue
        (bi, bj), bJ, ba = best
        bd1 = min(np.hypot(*(coord[bi] - TRUE1)), np.hypot(*(coord[bj] - TRUE1)))
        bd2 = min(np.hypot(*(coord[bi] - TRUE2)), np.hypot(*(coord[bj] - TRUE2)))
        ok = bd1 <= BASIN_TOL and bd2 <= BASIN_TOL
        sep_rows.append(dict(min_separation=sep,
                             pair=[coord[bi].tolist(), coord[bj].tolist()],
                             J=float(bJ), amps=[float(x) for x in ba],
                             dist_true1=float(bd1), dist_true2=float(bd2),
                             basin_correct=bool(ok)))
        print(f"sep>={sep:6.1f} m : ({coord[bi][0]:6.1f},{coord[bi][1]:6.1f})+"
              f"({coord[bj][0]:6.1f},{coord[bj][1]:6.1f})  J={bJ:.4e}  "
              f"amps=({ba[0]:.3f},{ba[1]:.3f})  d1={bd1:5.1f} d2={bd2:5.1f}  "
              f"basin-correct={ok}")

    print(f"\ntotal unit forward solves used: {len(cache)}")
    with open(OUT / "screen_sweep_lrb.json", "w") as fh:
        json.dump(dict(j_zero=float(j_zero),
                       node_true1=dict(xy=coord[n_true1].tolist(), dist=d1,
                                       amp=a1s, J=j1s, raw_g_rank=rank_t1),
                       node_true2=dict(xy=coord[n_true2].tolist(), dist=d2,
                                       amp=a2s, J=j2s, raw_g_rank=rank_t2),
                       best_mesh_node_pair=dict(J=float(j_bestnodes),
                                                amps=[float(x) for x in amps_bn]),
                       sweep=results, separation=sep_rows,
                       total_unit_solves=len(cache)), fh, indent=1)


if __name__ == "__main__":
    main()
