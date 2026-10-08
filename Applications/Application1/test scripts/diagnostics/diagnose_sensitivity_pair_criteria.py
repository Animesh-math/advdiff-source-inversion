"""Compare prior-free pair-selection criteria on two candidate dictionaries (diagnostic only).

For the failed two-source case, builds the level-1 screened dictionary for
(K=24, radius=80) [production] and (K=48, radius=40) [finer], evaluates ALL pairs
exactly (jointly refit bounded amplitudes, exact objective from cached traces),
and reports the winner under three selection rules:

  1. argmin exact pair objective                       (no constraint)
  2. argmin exact pair objective with |coherence| <= c  (dimensionless, prior-free)
  3. argmin exact pair objective with separation >= s   (needs a length prior)

"Basin-correct" means one atom within 60 m of each true center.
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
DICTS = [(24, 80.0), (48, 40.0)]
COH_CAPS = (0.5, 0.3, 0.2, 0.1)
SEPS = (100.0, 150.0, 250.0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mesh_data, pde, admiss = D.build_setup(D.BOUNDARY_SETS["lrb"], 1600.0)
    coord = np.asarray(mesh_data["coord"], dtype=np.float64)[:, :2]
    ctx = D.obs_ctx(mesh_data, pde)
    obs = D.true_state(mesh_data, pde)
    ud = ctx.extract(obs)
    j0 = ctx.tracking_objective(np.zeros_like(ud), ud)
    outer = sinit._outer_boundary_node_coords(mesh_data, coord)
    bld = sinit._building_boundary_node_coords(mesh_data, coord)
    smask = sinit._base_admissibility_mask(coord, admiss, outer, bld, 0.0, 0.0)
    _, _sa, _im, W = sis.compute_adjoint_weight_field(
        mesh_data, pde, sinit._DUMMY_Z, obs, ro.ETA1, ro.OBJECTIVE_TIME_RULE,
        dirichlet_tags=("left",), state_out=np.zeros_like(obs))
    qx, qy, ag = sis.precompute_quadrature_weights(mesh_data, pde, W)
    g1 = sis.sensitivity_at_candidates(coord, qx, qy, ag, pde["sigma2"])

    cache = {}

    def trace(n):
        if n not in cache:
            cache[n] = ctx.extract(D.unit_state(mesh_data, pde, *coord[n]))
        return cache[n]

    out = {}
    for top_k, radius in DICTS:
        key = f"K{top_k}_r{int(radius)}"
        scr = [i for i in sinit._topk_diverse(
            g1, np.flatnonzero(smask), coord, top_k, radius) if g1[i] < 0.0]
        pairs = []
        for i, j in itertools.combinations(range(len(scr)), 2):
            ni, nj = scr[i], scr[j]
            t1, t2 = trace(ni), trace(nj)
            G = np.array([[ctx.inner(t1, t1), ctx.inner(t1, t2)],
                          [ctx.inner(t2, t1), ctx.inner(t2, t2)]])
            b = np.array([-ctx.inner(t1, ud), -ctx.inner(t2, ud)])
            a = sinit._solve_box_qp(G, b, LO, HI)
            pairs.append(dict(
                ni=ni, nj=nj, J=float(j0 + 0.5 * a @ G @ a + b @ a),
                coh=float(G[0, 1] / np.sqrt(G[0, 0] * G[1, 1])),
                sep=float(np.hypot(*(coord[ni] - coord[nj]))),
                a=[float(a[0]), float(a[1])],
                d1=float(min(np.hypot(*(coord[ni] - TRUE1)),
                             np.hypot(*(coord[nj] - TRUE1)))),
                d2=float(min(np.hypot(*(coord[ni] - TRUE2)),
                             np.hypot(*(coord[nj] - TRUE2)))),
            ))
        print(f"\n=== dictionary {key}: {len(scr)} atoms, {len(pairs)} pairs "
              f"(cumulative unit forwards {len(cache)}) ===")
        rules = [("argmin J (unconstrained)", lambda p: True)]
        rules += [(f"|coherence| <= {c}", (lambda c: (lambda p: abs(p['coh']) <= c))(c))
                  for c in COH_CAPS]
        rules += [(f"separation >= {s:.0f} m", (lambda s: (lambda p: p['sep'] >= s))(s))
                  for s in SEPS]
        rows = []
        for name, pred in rules:
            f = [p for p in pairs if pred(p)]
            if not f:
                print(f"  {name:26s} -> no admissible pair")
                continue
            b = min(f, key=lambda p: p["J"])
            ok = b["d1"] <= 60.0 and b["d2"] <= 60.0
            print(f"  {name:26s} -> ({coord[b['ni']][0]:6.1f},{coord[b['ni']][1]:6.1f})"
                  f"a={b['a'][0]:.3f} + ({coord[b['nj']][0]:6.1f},{coord[b['nj']][1]:6.1f})"
                  f"a={b['a'][1]:.3f}  J={b['J']:.4e}  coh={b['coh']:+.3f} "
                  f"sep={b['sep']:6.1f}  d1={b['d1']:5.1f} d2={b['d2']:5.1f}  "
                  f"basin-correct={'YES' if ok else 'no'}")
            rows.append(dict(rule=name, control=[
                float(coord[b['ni']][0]), float(coord[b['ni']][1]), b['a'][0],
                float(coord[b['nj']][0]), float(coord[b['nj']][1]), b['a'][1]],
                J=b["J"], coherence=b["coh"], separation=b["sep"],
                dist_true1=b["d1"], dist_true2=b["d2"], basin_correct=bool(ok)))
        out[key] = dict(n_atoms=len(scr), n_pairs=len(pairs), rules=rows)

    out["j_zero"] = float(j0)
    out["total_unit_solves"] = len(cache)
    print(f"\ntotal unit forward solves: {len(cache)}")
    with open(OUT / "pair_criteria_lrb.json", "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
