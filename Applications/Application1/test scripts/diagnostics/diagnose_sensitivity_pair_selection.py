"""Diagnostic-only reconstruction of the App1 sensitivity-init candidate/pair search.

READ-ONLY with respect to production behavior: it calls the *same* functions
``sensitivity_initialization._predicted_reduction_beam`` calls, in the same order,
with the same arguments, and only records more.  It never writes to any
production module and never changes an algorithmic decision.

What it produces, per measurement-boundary configuration:

  1. ``level1_candidates_<tag>.csv``  every screened first-stage candidate with
     G, kappa, a_unconstrained, a_bounded, predicted reduction, exact 1-source
     objective, rank, and whether it entered the beam.
  2. ``level2_candidates_<tag>.csv``  for each retained beam branch, every
     second-stage candidate the production code actually screens, with the
     branch G value, kappa, 1-D amplitude/reduction, rank, whether the
     production beam expanded it, and (extra, diagnostic-only) the jointly
     refit amplitudes, the 2x2 Gram matrix, and the exact pair objective.
  3. ``all_pairs_<tag>.csv``  the EXHAUSTIVE K x K pair evaluation over the
     level-1 screened dictionary using the already-cached unit traces
     (0 extra PDE solves), jointly refit and ranked by exact objective.
  4. a console report answering "was the near-true pair evaluated?".

Run (from this directory):
    python3 \
        diagnose_sensitivity_pair_selection.py
"""

from __future__ import annotations

import csv
import itertools
import json
import time
from pathlib import Path

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

TOP_K = 24
BEAM_WIDTH = 3
INIT_SIGMA2 = 1600.0
MIN_SEPARATION = None          # production default (--min-source-separation unset)
MIN_VALUE_TOL = 0.0


def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"  wrote {path}")


def run_config(tag, boundaries):
    print(f"\n{'=' * 78}\n=== configuration {tag}: measurement boundaries {list(boundaries)}\n{'=' * 78}")
    t_start = time.time()
    mesh_data, pde, admiss = D.build_setup(boundaries, INIT_SIGMA2)
    coord = np.asarray(mesh_data["coord"], dtype=np.float64)[:, :2]
    candidate_xy = coord

    ctx = D.obs_ctx(mesh_data, pde)
    print(f"observation edges in union: {ctx._elen.size}, boundary nodes: {ctx.bnodes.size}")

    # Observation data at the INIT width (what run_optimization regenerates).
    t0 = time.time()
    obs_state = D.true_state(mesh_data, pde)
    ud = ctx.extract(obs_state)
    j_zero = ctx.tracking_objective(np.zeros_like(ud), ud)
    print(f"J(zero source) at sigma2={INIT_SIGMA2:g}: {j_zero:.6e}   [{time.time()-t0:.1f}s]")

    # ---- identical masks/screen to production -------------------------------
    outer_xy = sinit._outer_boundary_node_coords(mesh_data, coord)
    building_xy = sinit._building_boundary_node_coords(mesh_data, coord)
    static_mask = sinit._base_admissibility_mask(
        candidate_xy, admiss, outer_xy, building_xy, 0.0, 0.0
    )
    screen_radius = max(float(MIN_SEPARATION or 0.0), 2.0 * np.sqrt(pde["sigma2"]))
    print(f"admissible candidate nodes: {int(static_mask.sum())} / {static_mask.size}")
    print(f"screen radius: {screen_radius:.1f} m, top_k={TOP_K}, beam_width={BEAM_WIDTH}")

    # ---- shared unit-trace cache (no eviction: this is a diagnostic) --------
    trace_cache, kappa_cache = {}, {}
    n_forward = [0]

    def unit_trace(node):
        if node not in trace_cache:
            st = D.unit_state(mesh_data, pde, candidate_xy[node][0], candidate_xy[node][1])
            n_forward[0] += 1
            trace_cache[node] = ctx.extract(st)
        return trace_cache[node]

    def unit_kappa(node):
        if node not in kappa_cache:
            kappa_cache[node] = ctx.norm2(unit_trace(node)) + float(ro.LAMBDA_REG)
        return kappa_cache[node]

    def joint_refit(nodes):
        traces = [unit_trace(n) for n in nodes]
        n = len(nodes)
        Gram = np.empty((n, n))
        rhs = np.empty(n)
        for i in range(n):
            for j in range(i, n):
                Gram[i, j] = Gram[j, i] = ctx.inner(traces[i], traces[j])
            rhs[i] = -ctx.inner(traces[i], ud)
        Gram = Gram + float(ro.LAMBDA_REG) * np.eye(n)
        amps = sinit._solve_box_qp(Gram, rhs, *D.AMPLITUDE_BOUNDS)
        return amps, Gram, rhs

    def exact_objective(nodes, amps):
        model = np.zeros_like(ud)
        for nd, a in zip(nodes, amps):
            model = model + float(a) * unit_trace(nd)
        return ctx.tracking_objective(model, ud)

    def g_field_for(nodes, amps):
        """Branch insertion-sensitivity field: 1 forward (background) + 1 adjoint."""
        if nodes:
            src = sinit._gaussian_sum_source(
                [(candidate_xy[n][0], candidate_xy[n][1], float(a))
                 for n, a in zip(nodes, amps)], pde["sigma2"]
            )
            bg = ro.gt.run_forward(mesh_data, pde, sinit._DUMMY_Z, source_func=src)
        else:
            bg = np.zeros_like(obs_state)
        _, _sa, _im, W = sis.compute_adjoint_weight_field(
            mesh_data, pde, sinit._DUMMY_Z, obs_state, ro.ETA1,
            ro.OBJECTIVE_TIME_RULE, dirichlet_tags=("left",), state_out=bg,
        )
        qx, qy, ag = sis.precompute_quadrature_weights(mesh_data, pde, W)
        return sis.sensitivity_at_candidates(candidate_xy, qx, qy, ag, pde["sigma2"])

    # ================= LEVEL 1 ==============================================
    print("\n--- level 1 (root branch) ---")
    t0 = time.time()
    g1 = g_field_for([], [])
    print(f"root adjoint + G field: {time.time()-t0:.1f}s")

    mask1 = static_mask & sinit._separation_mask(candidate_xy, [], MIN_SEPARATION)
    screened1 = sinit._topk_diverse(
        g1, np.flatnonzero(mask1), candidate_xy, TOP_K, screen_radius
    )
    screened1 = [i for i in screened1 if g1[i] < -MIN_VALUE_TOL]
    raw_argmin1 = int(np.flatnonzero(mask1)[np.argmin(g1[np.flatnonzero(mask1)])])

    lo, hi = D.AMPLITUDE_BOUNDS
    recs1 = []
    for i in screened1:
        kap = unit_kappa(i)
        g_i = float(g1[i])
        a_unc = -g_i / kap
        a_1d = float(np.clip(a_unc, lo, hi))
        recs1.append(dict(
            node=int(i), x=float(candidate_xy[i][0]), y=float(candidate_xy[i][1]),
            G=g_i, kappa=float(kap), a_unconstrained=float(a_unc), a_bounded=a_1d,
            dj_pred=float(a_1d * g_i + 0.5 * a_1d * a_1d * kap),
            dist_true1=float(np.hypot(*(candidate_xy[i] - TRUE1))),
            dist_true2=float(np.hypot(*(candidate_xy[i] - TRUE2))),
        ))
    recs1.sort(key=lambda r: r["dj_pred"])
    for rk, r in enumerate(recs1, 1):
        r["rank_dj_pred"] = rk
    # Production ranks level-1 expansions by exact 1-source objective after refit.
    for r in recs1:
        amps, Gram, _ = joint_refit([r["node"]])
        r["a_joint"] = float(amps[0])
        r["J_exact_1src"] = float(exact_objective([r["node"]], amps))
    by_exact = sorted(recs1, key=lambda r: r["J_exact_1src"])
    for rk, r in enumerate(by_exact, 1):
        r["rank_exact_1src"] = rk
    # production: expansions = top beam_width by dj_pred, then re-sorted by exact J
    expanded_nodes = [r["node"] for r in recs1[:BEAM_WIDTH]]
    beam1 = sorted(
        [r for r in recs1 if r["node"] in expanded_nodes],
        key=lambda r: r["J_exact_1src"],
    )[:BEAM_WIDTH]
    beam1_nodes = [r["node"] for r in beam1]
    for r in recs1:
        r["expanded_by_beam"] = r["node"] in expanded_nodes
        r["in_beam_after_level1"] = r["node"] in beam1_nodes
        r["is_raw_argmin_G"] = r["node"] == raw_argmin1

    print(f"raw argmin G node: ({candidate_xy[raw_argmin1][0]:.1f}, "
          f"{candidate_xy[raw_argmin1][1]:.1f})  G={g1[raw_argmin1]:.4e}")
    print(f"screened {len(recs1)} candidates; {n_forward[0]} unit forward solves so far")

    fields1 = ["rank_dj_pred", "x", "y", "node", "G", "kappa", "a_unconstrained",
               "a_bounded", "dj_pred", "a_joint", "J_exact_1src", "rank_exact_1src",
               "dist_true1", "dist_true2", "expanded_by_beam",
               "in_beam_after_level1", "is_raw_argmin_G"]
    _write_csv(OUT / f"level1_candidates_{tag}.csv", recs1, fields1)

    near1 = min(recs1, key=lambda r: r["dist_true1"])
    near2 = min(recs1, key=lambda r: r["dist_true2"])
    print(f"\nnearest screened candidate to TRUE1 (250,130): "
          f"({near1['x']:.2f},{near1['y']:.2f}) d={near1['dist_true1']:.2f} m  "
          f"rank_dj_pred={near1['rank_dj_pred']}  rank_exact={near1['rank_exact_1src']}")
    print(f"nearest screened candidate to TRUE2 ( 15,580): "
          f"({near2['x']:.2f},{near2['y']:.2f}) d={near2['dist_true2']:.2f} m  "
          f"rank_dj_pred={near2['rank_dj_pred']}  rank_exact={near2['rank_exact_1src']}")

    # ================= LEVEL 2 (production beam) ============================
    print("\n--- level 2 (production beam expansion) ---")
    recs2 = []
    branch_expansions = []
    for b_i, br in enumerate(beam1, 1):
        b_nodes = [br["node"]]
        b_amps = np.array([br["a_joint"]])
        t0 = time.time()
        g2 = g_field_for(b_nodes, b_amps)
        mask2 = static_mask & sinit._separation_mask(
            candidate_xy, [candidate_xy[n] for n in b_nodes], MIN_SEPARATION
        )
        scr2 = sinit._topk_diverse(
            g2, np.flatnonzero(mask2), candidate_xy, TOP_K, screen_radius
        )
        scr2 = [i for i in scr2 if g2[i] < -MIN_VALUE_TOL]
        rr = []
        for i in scr2:
            kap = unit_kappa(i)
            g_i = float(g2[i])
            a_unc = -g_i / kap
            a_1d = float(np.clip(a_unc, lo, hi))
            rr.append(dict(
                branch=b_i, first_x=br["x"], first_y=br["y"], first_amp=br["a_joint"],
                node=int(i), x=float(candidate_xy[i][0]), y=float(candidate_xy[i][1]),
                G_branch=g_i, kappa=float(kap), a_unconstrained=float(a_unc),
                a_bounded_1d=a_1d,
                dj_pred=float(a_1d * g_i + 0.5 * a_1d * a_1d * kap),
                separation=float(np.hypot(*(candidate_xy[i] - candidate_xy[br["node"]]))),
                dist_true1=float(np.hypot(*(candidate_xy[i] - TRUE1))),
                dist_true2=float(np.hypot(*(candidate_xy[i] - TRUE2))),
            ))
        rr.sort(key=lambda r: r["dj_pred"])
        for rk, r in enumerate(rr, 1):
            r["rank_dj_pred"] = rk
            r["expanded_by_beam"] = rk <= BEAM_WIDTH
        # diagnostic-only: joint refit + exact objective for EVERY screened cand
        for r in rr:
            nodes = [br["node"], r["node"]]
            amps, Gram, _ = joint_refit(nodes)
            r["gram_11"] = float(Gram[0, 0])
            r["gram_12"] = float(Gram[0, 1])
            r["gram_22"] = float(Gram[1, 1])
            r["coherence"] = float(Gram[0, 1] / np.sqrt(Gram[0, 0] * Gram[1, 1]))
            r["a1_joint"] = float(amps[0])
            r["a2_joint"] = float(amps[1])
            r["J_exact_pair"] = float(exact_objective(nodes, amps))
        for rk, r in enumerate(sorted(rr, key=lambda z: z["J_exact_pair"]), 1):
            r["rank_exact_pair"] = rk
        recs2.extend(rr)
        for r in rr:
            if r["expanded_by_beam"]:
                branch_expansions.append(
                    dict(nodes=[br["node"], r["node"]],
                         J=r["J_exact_pair"],
                         amps=[r["a1_joint"], r["a2_joint"]])
                )
        print(f"  branch {b_i} first=({br['x']:.1f},{br['y']:.1f}) "
              f"a={br['a_joint']:.4f} J1={br['J_exact_1src']:.4e}: "
              f"screened {len(rr)}  [{time.time()-t0:.1f}s + solves]")

    fields2 = ["branch", "first_x", "first_y", "first_amp", "rank_dj_pred", "x", "y",
               "node", "G_branch", "kappa", "a_unconstrained", "a_bounded_1d",
               "dj_pred", "separation", "gram_11", "gram_12", "gram_22", "coherence",
               "a1_joint", "a2_joint", "J_exact_pair", "rank_exact_pair",
               "expanded_by_beam", "dist_true1", "dist_true2"]
    _write_csv(OUT / f"level2_candidates_{tag}.csv", recs2, fields2)

    branch_expansions.sort(key=lambda b: b["J"])
    winner = branch_expansions[0]
    wn = winner["nodes"]
    print("\nPRODUCTION-EQUIVALENT SELECTION:")
    print(f"  centers ({candidate_xy[wn[0]][0]:.6f},{candidate_xy[wn[0]][1]:.6f}) "
          f"and ({candidate_xy[wn[1]][0]:.6f},{candidate_xy[wn[1]][1]:.6f})")
    print(f"  amps {winner['amps'][0]:.10f}, {winner['amps'][1]:.10f}")
    print(f"  exact init objective {winner['J']:.6e}  (J0 = {j_zero:.6e})")

    # ================= EXHAUSTIVE K x K over the level-1 dictionary =========
    print("\n--- exhaustive pair evaluation over the level-1 screened dictionary ---")
    dict_nodes = [r["node"] for r in recs1]
    pair_rows = []
    for i, j in itertools.combinations(range(len(dict_nodes)), 2):
        ni, nj = dict_nodes[i], dict_nodes[j]
        amps, Gram, _ = joint_refit([ni, nj])
        Jp = exact_objective([ni, nj], amps)
        pair_rows.append(dict(
            x1=float(candidate_xy[ni][0]), y1=float(candidate_xy[ni][1]),
            x2=float(candidate_xy[nj][0]), y2=float(candidate_xy[nj][1]),
            a1=float(amps[0]), a2=float(amps[1]),
            separation=float(np.hypot(*(candidate_xy[ni] - candidate_xy[nj]))),
            gram_12=float(Gram[0, 1]),
            coherence=float(Gram[0, 1] / np.sqrt(Gram[0, 0] * Gram[1, 1])),
            J_exact_pair=float(Jp),
            frac_explained=float(1.0 - Jp / j_zero),
            best_dist_true1=float(min(np.hypot(*(candidate_xy[ni] - TRUE1)),
                                      np.hypot(*(candidate_xy[nj] - TRUE1)))),
            best_dist_true2=float(min(np.hypot(*(candidate_xy[ni] - TRUE2)),
                                      np.hypot(*(candidate_xy[nj] - TRUE2)))),
        ))
    pair_rows.sort(key=lambda r: r["J_exact_pair"])
    for rk, r in enumerate(pair_rows, 1):
        r["rank"] = rk
    fieldsP = ["rank", "x1", "y1", "a1", "x2", "y2", "a2", "separation", "gram_12",
               "coherence", "J_exact_pair", "frac_explained",
               "best_dist_true1", "best_dist_true2"]
    _write_csv(OUT / f"all_pairs_{tag}.csv", pair_rows, fieldsP)

    print(f"  evaluated {len(pair_rows)} pairs with 0 extra PDE solves "
          f"(total unit forwards this config: {n_forward[0]})")
    print("  best 5 exhaustive pairs:")
    for r in pair_rows[:5]:
        print(f"    #{r['rank']} ({r['x1']:6.1f},{r['y1']:6.1f}) a={r['a1']:.3f} + "
              f"({r['x2']:6.1f},{r['y2']:6.1f}) a={r['a2']:.3f}  "
              f"J={r['J_exact_pair']:.4e} ({100*r['frac_explained']:.2f}% explained)")

    # the near-true pair, from the dictionary
    nt = [r for r in pair_rows
          if {(round(r["x1"], 6), round(r["y1"], 6)), (round(r["x2"], 6), round(r["y2"], 6))}
          == {(round(near1["x"], 6), round(near1["y"], 6)),
              (round(near2["x"], 6), round(near2["y"], 6))}]
    if nt:
        r = nt[0]
        print(f"\n  NEAR-TRUE PAIR in exhaustive ranking: rank {r['rank']}/"
              f"{len(pair_rows)}  J={r['J_exact_pair']:.6e} "
              f"({100*r['frac_explained']:.3f}% explained)  a=({r['a1']:.4f},{r['a2']:.4f})")

    summary = dict(
        tag=tag, boundaries=list(boundaries), init_sigma2=INIT_SIGMA2,
        j_zero=float(j_zero),
        n_admissible=int(static_mask.sum()), screen_radius=float(screen_radius),
        n_screened_level1=len(recs1),
        raw_argmin_G_xy=[float(candidate_xy[raw_argmin1][0]),
                         float(candidate_xy[raw_argmin1][1])],
        nearest_to_true1=dict(xy=[near1["x"], near1["y"]], dist=near1["dist_true1"],
                              rank_dj_pred=near1["rank_dj_pred"],
                              rank_exact_1src=near1["rank_exact_1src"]),
        nearest_to_true2=dict(xy=[near2["x"], near2["y"]], dist=near2["dist_true2"],
                              rank_dj_pred=near2["rank_dj_pred"],
                              rank_exact_1src=near2["rank_exact_1src"]),
        beam_level1_nodes_xy=[[r["x"], r["y"]] for r in beam1],
        production_selection=dict(
            centers=[[float(candidate_xy[wn[0]][0]), float(candidate_xy[wn[0]][1])],
                     [float(candidate_xy[wn[1]][0]), float(candidate_xy[wn[1]][1])]],
            amps=[float(a) for a in winner["amps"]],
            J_exact=float(winner["J"]),
        ),
        exhaustive_best=pair_rows[0],
        near_true_pair_rank=(nt[0]["rank"] if nt else None),
        near_true_pair_J=(nt[0]["J_exact_pair"] if nt else None),
        n_unit_forward_solves=n_forward[0],
        wall_seconds=time.time() - t_start,
    )
    with open(OUT / f"summary_{tag}.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"\nconfig {tag} done in {time.time()-t_start:.1f}s "
          f"({n_forward[0]} unit forward solves)")
    return summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out = {}
    for tag, bs in D.BOUNDARY_SETS.items():
        out[tag] = run_config(tag, bs)
    with open(OUT / "summary_all.json", "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
