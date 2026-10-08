"""Candidate remedy combinations for the App1 sensitivity init (diagnostic only).

Evaluates, for the failed two-source case, what initial control the existing
predicted-reduction + beam machinery WOULD return under different
(top_k, diversity radius, minimum source separation) settings -- and, for
comparison, under an exact-objective pair ranking over the whole screened
dictionary.  Nothing in production is changed; this only measures.
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

COMBOS = [
    # (top_k, radius, min_separation, label)
    (24, 80.0, None, "production (as run)"),
    (24, 80.0, 150.0, "production + separation 150"),
    (48, 40.0, None, "K=48 r=40"),
    (48, 40.0, 150.0, "K=48 r=40 + separation 150"),
    (48, 40.0, 250.0, "K=48 r=40 + separation 250"),
    (32, 60.0, 150.0, "K=32 r=60 + separation 150"),
    (64, 40.0, 150.0, "K=64 r=40 + separation 150"),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sigma2 = 1600.0
    mesh_data, pde, admiss = D.build_setup(D.BOUNDARY_SETS["lrb"], sigma2)
    coord = np.asarray(mesh_data["coord"], dtype=np.float64)[:, :2]
    ctx = D.obs_ctx(mesh_data, pde)
    obs_state = D.true_state(mesh_data, pde)
    ud = ctx.extract(obs_state)
    j0 = ctx.tracking_objective(np.zeros_like(ud), ud)

    outer = sinit._outer_boundary_node_coords(mesh_data, coord)
    bld = sinit._building_boundary_node_coords(mesh_data, coord)
    smask = sinit._base_admissibility_mask(coord, admiss, outer, bld, 0.0, 0.0)
    _, _sa, _im, W = sis.compute_adjoint_weight_field(
        mesh_data, pde, sinit._DUMMY_Z, obs_state, ro.ETA1, ro.OBJECTIVE_TIME_RULE,
        dirichlet_tags=("left",), state_out=np.zeros_like(obs_state))
    qx, qy, ag = sis.precompute_quadrature_weights(mesh_data, pde, W)
    g1 = sis.sensitivity_at_candidates(coord, qx, qy, ag, pde["sigma2"])

    cache = {}

    def trace(n):
        if n not in cache:
            cache[n] = ctx.extract(D.unit_state(mesh_data, pde, *coord[n]))
        return cache[n]

    def fit_pair(ni, nj):
        t1, t2 = trace(ni), trace(nj)
        G = np.array([[ctx.inner(t1, t1), ctx.inner(t1, t2)],
                      [ctx.inner(t2, t1), ctx.inner(t2, t2)]])
        b = np.array([-ctx.inner(t1, ud), -ctx.inner(t2, ud)])
        a = sinit._solve_box_qp(G, b, LO, HI)
        return a, float(j0 + 0.5 * a @ G @ a + b @ a)

    rows = []
    print(f"J0 = {j0:.6e}\n")
    hdr = (f"{'configuration':32s} {'atoms':>5} {'center 1':>17s} {'center 2':>17s} "
           f"{'a1':>6} {'a2':>6} {'J_init':>11} {'d1':>6} {'d2':>6} {'ok':>3}")
    print(hdr)
    print("-" * len(hdr))
    for top_k, radius, sep, label in COMBOS:
        scr = sinit._topk_diverse(g1, np.flatnonzero(smask), coord, top_k, radius)
        scr = [i for i in scr if g1[i] < 0.0]
        best = None
        for i, j in itertools.combinations(range(len(scr)), 2):
            ni, nj = scr[i], scr[j]
            if sep and np.hypot(*(coord[ni] - coord[nj])) < sep:
                continue
            a, J = fit_pair(ni, nj)
            if best is None or J < best[1]:
                best = ((ni, nj), J, a)
        if best is None:
            print(f"{label:32s}  no admissible pair")
            continue
        (bi, bj), bJ, ba = best
        # order so that the atom nearest TRUE1 is printed first
        if np.hypot(*(coord[bj] - TRUE1)) < np.hypot(*(coord[bi] - TRUE1)):
            bi, bj = bj, bi
            ba = ba[::-1]
        d1 = float(np.hypot(*(coord[bi] - TRUE1)))
        d2 = float(np.hypot(*(coord[bj] - TRUE2)))
        ok = d1 <= 60.0 and d2 <= 60.0
        print(f"{label:32s} {len(scr):5d} "
              f"({coord[bi][0]:7.2f},{coord[bi][1]:7.2f}) "
              f"({coord[bj][0]:7.2f},{coord[bj][1]:7.2f}) "
              f"{ba[0]:6.3f} {ba[1]:6.3f} {bJ:11.4e} {d1:6.1f} {d2:6.1f} "
              f"{'YES' if ok else 'no':>3}")
        rows.append(dict(label=label, top_k=top_k, radius=radius, min_separation=sep,
                         n_atoms=len(scr),
                         control=[float(coord[bi][0]), float(coord[bi][1]), float(ba[0]),
                                  float(coord[bj][0]), float(coord[bj][1]), float(ba[1])],
                         J_init=float(bJ), frac_explained=float(1 - bJ / j0),
                         dist_true1=d1, dist_true2=d2, basin_correct=bool(ok)))

    print(f"\ntotal unit forward solves (shared cache across all combos): {len(cache)}")
    print("cost note: each configuration reuses the same atom traces; a standalone "
          "run costs 1 adjoint + (number of screened atoms) forward solves.")
    with open(OUT / "config_combos_lrb.json", "w") as fh:
        json.dump(dict(j_zero=float(j0), rows=rows, total_unit_solves=len(cache)),
                  fh, indent=1)


if __name__ == "__main__":
    main()
