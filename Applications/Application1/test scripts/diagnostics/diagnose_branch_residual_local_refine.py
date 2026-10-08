"""Branch-specific residual sensitivity with vs without local position refinement.

READ-ONLY diagnostic. No production module is modified. Reproduces the exact
Application1 sensitivity-init configuration via ``_diag_common`` (lrb, sigma2=1600)
and answers the question the existing DIAGNOSTIC_REPORT could not close:

  The production initializer ALREADY computes a branch-specific residual
  sensitivity field G_b per first-source branch (sensitivity_initialization.py
  _predicted_reduction_beam, l.861-895). It froze the first-source center at a
  coarse dictionary node. Does LOCAL POSITION REFINEMENT of the first source
  (variable-projected amplitude) BEFORE recomputing G_b move the residual field
  off the dominant plume and onto the weak true source, and does the resulting
  pair beat the production-selected pair?

Amplitude handling is stated explicitly at every step:
  * frozen 1-src amplitude a1: exact bounded 1-D fit (box QP, scalar).
  * refinement objective J_hat(c) = min_a 0.5||a t(c) - u_d||^2 : amplitude is
    VARIABLE-PROJECTED (closed form a*(c) = clip(<t(c),u_d>/<t(c),t(c)>)).
    Only the 2 position variables are nonlinearly optimized.
  * pair objective: exact bounded JOINT 2-amplitude fit (box QP), cross terms.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import sys as _sys_dc
from pathlib import Path as _Path_dc
_sys_dc.path.insert(0, str(_Path_dc(__file__).resolve().parents[1]))  # _diag_common
import _diag_common as dc
import sensitivity_initialization as sinit
import source_insertion_sensitivity as sis

RO = dc.ro
ETA1 = RO.ETA1
TIME_RULE = RO.OBJECTIVE_TIME_RULE
SIGMA2 = 1600.0
RADIUS = 2.0 * np.sqrt(SIGMA2)            # 80 m production diversity radius
TOP_K = 24
BEAM = 3
FLOOR, UPPER = dc.AMPLITUDE_BOUNDS
T1 = np.array([250.0, 130.0])
T2 = np.array([15.0, 580.0])

OUT_DIR = dc.APP_DIR / "output" / "branch_residual_refine_diag"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    mesh_data, pde, admiss = dc.build_setup(("left", "right", "bottom"), SIGMA2)
    coord = np.asarray(mesh_data["coord"], dtype=np.float64)[:, :2]
    ctx = dc.obs_ctx(mesh_data, pde)
    ud_state = dc.true_state(mesh_data, pde)
    ud_trace = ctx.extract(ud_state)
    j0 = ctx.tracking_objective(np.zeros_like(ud_trace), ud_trace)

    outer = sinit._outer_boundary_node_coords(mesh_data, mesh_data["coord"])
    build = sinit._building_boundary_node_coords(mesh_data, mesh_data["coord"])
    static_mask = sinit._base_admissibility_mask(coord, admiss, outer, build, 0.0, 0.0)
    mask_idx = np.flatnonzero(static_mask)

    solves = {"forward": 0}

    def unit_trace_at(cx, cy):
        src = sinit._gaussian_sum_source([(cx, cy, 1.0)], SIGMA2)
        st = dc.gt.run_forward(mesh_data, pde, sinit._DUMMY_Z, source_func=src)
        solves["forward"] += 1
        return ctx.extract(st)

    node_trace_cache = {}

    def node_trace(node):
        if node not in node_trace_cache:
            node_trace_cache[node] = unit_trace_at(coord[node][0], coord[node][1])
        return node_trace_cache[node]

    def scalar_fit(trace):
        """Exact bounded 1-D amplitude and its objective for a single source."""
        gram = ctx.norm2(trace)
        rhs = ctx.inner(trace, ud_trace)
        a_unc = rhs / gram if gram > 0 else 0.0
        a = float(np.clip(a_unc, FLOOR, UPPER))
        j = 0.5 * ctx.norm2(a * trace - ud_trace)
        return a, j, a_unc, gram

    def pair_fit(traces):
        """Exact bounded joint 2-amplitude fit (box QP) and exact objective."""
        n = len(traces)
        G = np.empty((n, n))
        b = np.empty(n)
        for i in range(n):
            for j in range(i, n):
                G[i, j] = G[j, i] = ctx.inner(traces[i], traces[j])
            b[i] = -ctx.inner(traces[i], ud_trace)
        amps = sinit._solve_box_qp(G, b, FLOOR, UPPER)
        model = np.zeros_like(ud_trace)
        for a, t in zip(amps, traces):
            model = model + a * t
        j = 0.5 * ctx.norm2(model - ud_trace)
        coh = G[0, 1] / np.sqrt(G[0, 0] * G[1, 1]) if G[0, 0] > 0 and G[1, 1] > 0 else 0.0
        return amps, j, float(coh)

    def residual_field(bg_state):
        """Branch residual insertion sensitivity G_b over all mesh nodes."""
        _, sadj, _inv, W = sis.compute_adjoint_weight_field(
            mesh_data, pde, sinit._DUMMY_Z, ud_state, ETA1, TIME_RULE,
            dirichlet_tags=("left",), state_out=bg_state,
        )
        qx, qy, ag = sis.precompute_quadrature_weights(mesh_data, pde, W)
        return sis.sensitivity_at_candidates(coord, qx, qy, ag, SIGMA2)

    def bg_state_from(cx, cy, a):
        src = sinit._gaussian_sum_source([(cx, cy, float(a))], SIGMA2)
        st = dc.gt.run_forward(mesh_data, pde, sinit._DUMMY_Z, source_func=src)
        solves["forward"] += 1
        return st

    def screen(field):
        return sinit._topk_diverse(field, mask_idx, coord, TOP_K, RADIUS)

    def d(a, b):
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    # ---- Level 1: zero-source field G0 -> first-source branches -------------
    print("=== Level 1: zero-source field G0, top-K screen ===", flush=True)
    G0 = residual_field(np.zeros_like(ud_state))
    scr0 = screen(G0)
    l1 = []
    for node in scr0:
        t = node_trace(node)
        a, j, a_unc, gram = scalar_fit(t)
        l1.append({"node": int(node), "xy": coord[node].tolist(), "a": a,
                   "J1": j, "dT1": d(coord[node], T1), "dT2": d(coord[node], T2)})
    l1.sort(key=lambda r: r["J1"])
    branches = l1[:BEAM]
    for bi, br in enumerate(branches, 1):
        print(f"  branch {bi}: c=({br['xy'][0]:.1f},{br['xy'][1]:.1f}) a={br['a']:.4f}"
              f" J1={br['J1']:.4e} dT1={br['dT1']:.1f} dT2={br['dT2']:.1f}", flush=True)

    # Best second candidate from a residual field, by exact pair objective.
    def best_second(field, c1xy, t1, exclude_first_node):
        scr = [n for n in screen(field) if n != exclude_first_node]
        scr = [n for n in scr if field[n] < 0.0]
        # argmin of the field itself (where the residual most wants a source)
        fmin_node = int(mask_idx[np.argmin(field[mask_idx])])
        best = None
        for node in scr[:8]:                     # top-8 diverse residual minima
            t2 = node_trace(node)
            amps, j, coh = pair_fit([t1, t2])
            rec = {"node": int(node), "xy": coord[node].tolist(),
                   "amps": [float(amps[0]), float(amps[1])], "Jpair": j,
                   "coh": coh, "dT2": d(coord[node], T2), "dT1": d(coord[node], T1)}
            if best is None or j < best["Jpair"]:
                best = rec
        # rank of the node nearest true source 2 within the screen
        dists_T2 = [d(coord[n], T2) for n in scr]
        near_rank = int(np.argmin(dists_T2)) + 1 if dists_T2 else -1
        near_node = scr[int(np.argmin(dists_T2))] if dists_T2 else -1
        return best, {
            "argmin_field_xy": coord[fmin_node].tolist(),
            "argmin_field_dT1": d(coord[fmin_node], T1),
            "argmin_field_dT2": d(coord[fmin_node], T2),
            "nearest_T2_rank_in_screen": near_rank,
            "nearest_T2_dist": d(coord[near_node], T2) if near_node >= 0 else None,
        }

    results = {"j0": j0, "branches": []}

    for bi, br in enumerate(branches, 1):
        node1 = br["node"]
        c1 = np.array(br["xy"])
        t1 = node_trace(node1)
        a1 = br["a"]
        print(f"\n=== Branch {bi} frozen c1=({c1[0]:.1f},{c1[1]:.1f}) a1={a1:.4f} "
              f"J1={br['J1']:.4e} (dT1={br['dT1']:.1f}) ===", flush=True)

        # (A) FROZEN residual field + best second
        bgA = bg_state_from(c1[0], c1[1], a1)
        GA = residual_field(bgA)
        bestA, diagA = best_second(GA, c1, t1, node1)
        print(f"  [frozen]  residual argmin at "
              f"({diagA['argmin_field_xy'][0]:.1f},{diagA['argmin_field_xy'][1]:.1f}) "
              f"dT1={diagA['argmin_field_dT1']:.1f} dT2={diagA['argmin_field_dT2']:.1f}; "
              f"nearest-T2 screen rank={diagA['nearest_T2_rank_in_screen']} "
              f"({diagA['nearest_T2_dist']:.1f} m)", flush=True)
        print(f"  [frozen]  best pair 2nd=({bestA['xy'][0]:.1f},{bestA['xy'][1]:.1f}) "
              f"dT2={bestA['dT2']:.1f} amps={bestA['amps']} coh={bestA['coh']:.3f} "
              f"Jpair={bestA['Jpair']:.4e}", flush=True)

        # (B) LOCAL REFINEMENT of c1 (variable-projected amplitude), then residual
        def Jhat(c):
            t = unit_trace_at(c[0], c[1])
            _a, j, _u, _g = scalar_fit(t)
            return j
        res = minimize(Jhat, c1, method="Nelder-Mead",
                       options={"xatol": 0.5, "fatol": 1.0, "maxfev": 60})
        c1r = res.x
        t1r = unit_trace_at(c1r[0], c1r[1])
        a1r, j1r, _u, _g = scalar_fit(t1r)
        print(f"  [refine]  c1*=({c1r[0]:.1f},{c1r[1]:.1f}) a1*={a1r:.4f} "
              f"J1*={j1r:.4e}  dT1={d(c1r,T1):.1f}  (was {br['dT1']:.1f}); "
              f"{res.nfev} fwd evals", flush=True)

        bgB = bg_state_from(c1r[0], c1r[1], a1r)
        GB = residual_field(bgB)
        bestB, diagB = best_second(GB, c1r, t1r, -1)
        print(f"  [refine]  residual argmin at "
              f"({diagB['argmin_field_xy'][0]:.1f},{diagB['argmin_field_xy'][1]:.1f}) "
              f"dT1={diagB['argmin_field_dT1']:.1f} dT2={diagB['argmin_field_dT2']:.1f}; "
              f"nearest-T2 screen rank={diagB['nearest_T2_rank_in_screen']} "
              f"({diagB['nearest_T2_dist']:.1f} m)", flush=True)
        print(f"  [refine]  best pair 2nd=({bestB['xy'][0]:.1f},{bestB['xy'][1]:.1f}) "
              f"dT2={bestB['dT2']:.1f} amps={bestB['amps']} coh={bestB['coh']:.3f} "
              f"Jpair={bestB['Jpair']:.4e}", flush=True)

        results["branches"].append({
            "branch": bi, "frozen_c1": c1.tolist(), "frozen_a1": a1,
            "frozen_J1": br["J1"], "frozen_dT1": br["dT1"],
            "frozen_residual": diagA, "frozen_best_pair": bestA,
            "refined_c1": c1r.tolist(), "refined_a1": a1r, "refined_J1": j1r,
            "refined_dT1": d(c1r, T1), "refine_nfev": int(res.nfev),
            "refined_residual": diagB, "refined_best_pair": bestB,
        })

    # ---- Reference: production selected pair and true pair -------------------
    prod = np.array([211.5719053728, 173.4116764393, 285.1289895204, 126.4094834683])
    tprodA = unit_trace_at(prod[0], prod[1])
    tprodB = unit_trace_at(prod[2], prod[3])
    _, jprod, cohprod = pair_fit([tprodA, tprodB])
    tt1 = unit_trace_at(T1[0], T1[1])
    tt2 = unit_trace_at(T2[0], T2[1])
    ampsT, jtrue, cohtrue = pair_fit([tt1, tt2])
    results["production_selected_pair_J"] = jprod
    results["production_selected_pair_coh"] = cohprod
    results["true_pair_J"] = jtrue
    results["true_pair_amps"] = [float(ampsT[0]), float(ampsT[1])]
    results["forward_solves"] = solves["forward"]

    print("\n=== Reference ===", flush=True)
    print(f"  production selected pair J = {jprod:.4e} (coh={cohprod:.3f})", flush=True)
    print(f"  TRUE pair J = {jtrue:.4e} amps={results['true_pair_amps']}", flush=True)
    print(f"  total forward solves = {solves['forward']}", flush=True)

    (OUT_DIR / "branch_residual_refine.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_DIR / 'branch_residual_refine.json'}", flush=True)


if __name__ == "__main__":
    main()
