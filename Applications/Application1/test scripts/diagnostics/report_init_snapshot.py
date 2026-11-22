"""Read-only live snapshot of a sensitivity initialization.

Prints the initialization fields for one or more sweep cases as soon as
``output/<case>/full/sensitivity_init_summary.json`` exists, without waiting for
the continuation solve to finish.  Truths are read from
``output/final_sweep/cases.json`` unless ``--true`` is given.

Usage:
  python3 report_init_snapshot.py <case_name> [<case_name> ...]
  python3 report_init_snapshot.py <case_name> --true X1 Y1 A1 X2 Y2 A2
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import permutations
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parents[2]
# Src/Objective (gradient_test) on the path; APP inserted last so it is FIRST
# and Application1's mesh_data_config / multi_source_utils win.
sys.path.insert(0, str(APP.parents[1] / "Src" / "Objective"))
sys.path.insert(0, str(APP))

CASES_FILE = APP / "output" / "final_sweep" / "cases.json"


def load_truth(case_name: str):
    if not CASES_FILE.exists():
        return None, None
    for rec in json.loads(CASES_FILE.read_text()):
        if rec["name"] == case_name:
            return np.asarray(rec["true"], dtype=np.float64), rec
    return None, None


def best_permutation(true_z: np.ndarray, got_z: np.ndarray):
    """Match sources by minimizing total position error; return (perm, pos, amp)."""
    t = np.asarray(true_z, dtype=np.float64).reshape(-1, 3)
    g = np.asarray(got_z, dtype=np.float64).reshape(-1, 3)
    best = None
    for perm in permutations(range(len(g))):
        pos = np.array([np.hypot(*(g[p, :2] - t[i, :2]))
                        for i, p in enumerate(perm)])
        if best is None or pos.sum() < best[1].sum():
            amp = np.array([abs(g[p, 2] - t[i, 2]) for i, p in enumerate(perm)])
            best = (perm, pos, amp)
    return best


def fmt_control(z) -> str:
    z = np.asarray(z, dtype=np.float64)
    return ("[" + ", ".join(f"{v:.4f}" for v in z) + "]")


def report(case_name: str, truth, index: int | None = None) -> int:
    path = APP / "output" / case_name / "full" / "sensitivity_init_summary.json"
    if not path.exists():
        print(f"[{case_name}] no sensitivity_init_summary.json yet at {path}")
        return 1
    d = json.loads(path.read_text())
    sel = np.asarray(d["selected_control"], dtype=np.float64)
    truth_rec = None
    if truth is None:
        truth, truth_rec = load_truth(case_name)

    head = f"CASE {index}" if index is not None else "CASE"
    print("=" * 78)
    print(f"{head}: {case_name}")
    if truth_rec is not None:
        print(f"category            : {truth_rec.get('category', '')}")
    print("=" * 78)
    if truth is not None:
        print(f"true control        : {fmt_control(truth)}")
    print(f"selected init control: {fmt_control(sel)}")

    if truth is not None:
        perm, pos, amp = best_permutation(truth, sel)
        t = np.asarray(truth).reshape(-1, 3)
        g = sel.reshape(-1, 3)
        for i, p in enumerate(perm):
            print(f"  true source {i + 1} ({t[i,0]:7.2f},{t[i,1]:7.2f}) A={t[i,2]:.3f}"
                  f"  <-  init ({g[p,0]:7.2f},{g[p,1]:7.2f}) A={g[p,2]:.3f}"
                  f"   dpos={pos[i]:8.2f} m   dA={amp[i]:.4f}")
        print(f"max initial position error : {pos.max():.2f} m")
        print(f"max initial amplitude error: {amp.max():.4f}")

    print(f"initialized exact objective : {d['initial_exact_objective']:.6e}"
          f"   (width sigma2={d.get('objective_width_sigma2')})")
    print(f"zero-source objective       : {d['zero_source_objective']:.6e}")
    j0 = float(d["zero_source_objective"])
    ji = float(d["initial_exact_objective"])
    if j0 > 0:
        print(f"explained fraction of J0    : {100.0 * (1.0 - ji / j0):.4f}%")

    branches = d.get("branch_controls") or []
    print(f"retained beam branches      : {len(branches)}")
    winning = None
    for rec in branches:
        ctrl = np.asarray(rec["control"], dtype=np.float64)
        is_win = bool(np.allclose(ctrl, sel, rtol=0, atol=1e-12))
        if is_win and winning is None:
            winning = rec["branch_index"]
        centers = ", ".join(f"({c[0]:.1f},{c[1]:.1f})" for c in rec["centers"])
        walls = ", ".join(f"{w:.2f}" for w in rec["wall_distances_m"])
        mark = "  <-- WINNER" if is_win else ""
        print(f"  branch {rec['branch_index']}: J_exact={rec['objective']:.6e}"
              f"  centers [{centers}]  wall [{walls}] m{mark}")
    print(f"winning branch index        : {winning}")
    print(f"init solve budget           : {d.get('n_forward_solves')} forward, "
          f"{d.get('n_adjoint_solves')} adjoint")
    print(f"branch dedup removed        : {d.get('n_dedup_removed')}")
    div = d.get("diversity") or {}
    if div:
        print(f"diversity screen            : obstacle_aware={div.get('obstacle_aware')}"
              f"  screens={div.get('n_screens')}"
              f"  suppressed={div.get('n_suppressed_by_distance')}"
              f"  obstacle-preserved={div.get('n_preserved_obstacle_separated')}")
    resel = d.get("second_source_reselection") or []
    acc = [r for r in resel if r.get("accepted")]
    print(f"second-source reselection   : {len(resel)} pass record(s), "
          f"{len(acc)} accepted")
    for r in acc:
        print(f"    branch {r['branch_index']} pass {r['pass']}: "
              f"({r['old_second_center'][0]:.1f},{r['old_second_center'][1]:.1f}) -> "
              f"({r['new_second_candidate'][0]:.1f},{r['new_second_candidate'][1]:.1f})"
              f"  J {r['objective_before']:.4e} -> {r['objective_after_refinement']:.4e}"
              f"  coh={r['candidate_coherence']:.3f}")
    pair = d.get("pair_refinement") or []
    pacc = [r for r in pair if r.get("accepted")]
    print(f"joint pair refinement       : {len(pair)} branch record(s), "
          f"{len(pacc)} accepted")
    for r in pacc:
        moved = ", ".join(f"{m:.2f}" for m in r["moved_m"])
        print(f"    branch {r['branch_index']}: J {r['objective_before']:.4e} -> "
              f"{r['objective_after']:.4e}  moved [{moved}] m  "
              f"evals={r['n_forward_evaluations']}")
    plots = d.get("plot_paths") or []
    print(f"sensitivity plots           : {len(plots)} file(s)")
    for p in plots:
        print(f"    {p}")
    print()
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="+")
    ap.add_argument("--true", nargs=6, type=float, default=None)
    args = ap.parse_args()
    truth = np.asarray(args.true, dtype=np.float64) if args.true else None
    rc = 0
    for i, case in enumerate(args.cases, start=1):
        rc |= report(case, truth, index=i if len(args.cases) > 1 else None)
    sys.exit(rc)


if __name__ == "__main__":
    main()
