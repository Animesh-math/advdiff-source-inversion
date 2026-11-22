"""Compare ARCHIVED failed Application1 runs against their sensitivity+raw reruns.

Usage:
  python3 report_rerun_vs_archive.py \
      --pair <archived/rel/path> <new_case_name> [--pair ... ] \
      [--master-log PATH] [--log-dir DIR] [--out-dir DIR]

``archived/rel/path`` is relative to ``Applications/Application1/output`` and
must contain ``optimization_summary.json`` (e.g.
``app1_left_right_bottom_sigma_cont_seed42/full/start_001``).  ``new_case_name``
is a case whose run lives at ``output/<case>/full``.

Reports, per case: archived provenance, true control, the original run's initial
and final control with its objective and failure classification, the new
adjoint-sensitivity initial control with source-wise initial position/amplitude
errors after optimal label matching, the new raw-mode terminal point exactly as
returned by L-BFGS-B, termination, iterations, wall-clock, source-wise final
errors, building incursions, terminal feasibility, amplitude-bound activity, and
the new recovery classification.  Nothing is projected here.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import permutations
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APP.parents[1] / "Src" / "Objective"))
sys.path.insert(0, str(APP))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mesh_data_config as mdc  # noqa: E402
from mesh_admissibility import MeshAdmissibility  # noqa: E402
import report_raw_robustness_sweep as R  # noqa: E402

OUT = APP / "output"


def match_perm(cand, true):
    c = np.asarray(cand, float).reshape(-1, 3)
    t = np.asarray(true, float).reshape(-1, 3)
    best = None
    for perm in permutations(range(len(t))):
        cost = sum(float(np.hypot(*(c[perm[i], :2] - t[i, :2])))
                   for i in range(len(t)))
        if best is None or cost < best[0]:
            best = (cost, perm)
    return best[1]


def errors(cand, true, perm):
    c = np.asarray(cand, float).reshape(-1, 3)
    t = np.asarray(true, float).reshape(-1, 3)
    pos = [float(np.hypot(*(c[perm[i], :2] - t[i, :2]))) for i in range(len(t))]
    amp = [float(abs(c[perm[i], 2] - t[i, 2])) for i in range(len(t))]
    return pos, amp


def classify_archived(d, true):
    """Classification of the ORIGINAL archived run, same 1-5 scheme."""
    final = d["final_control"]
    perm = match_perm(final, true)
    pos, amp = errors(final, true, perm)
    msg = d.get("message") or ""
    lnsrch = "ABNORMAL_TERMINATION_IN_LNSRCH" in msg
    capped = "ITERATIONS REACHED LIMIT" in msg
    if max(pos) <= R.EXACT_POS_M and max(amp) <= R.EXACT_AMP:
        return 1, perm, pos, amp
    if max(pos) <= R.ACCEPTABLE_POS_M and max(amp) <= R.ACCEPTABLE_AMP:
        return 2, perm, pos, amp
    if lnsrch or (capped and int(d.get("iterations") or 0) <= 2):
        return 5, perm, pos, amp
    return 3, perm, pos, amp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, action="append", required=True,
                    metavar=("ARCHIVED", "NEWCASE"))
    ap.add_argument("--master-log", default=None)
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (OUT / "rerun_vs_archive")
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh_data = mdc.build_mesh_data()
    admiss = MeshAdmissibility(mesh_data)
    bidx = R.BuildingIndex(mesh_data)

    table = []
    for archived_rel, new_case in args.pair:
        arch_dir = OUT / archived_rel
        old = json.loads((arch_dir / "optimization_summary.json").read_text())
        true = np.asarray(old["true_control"], float)

        print("\n" + "#" * 96)
        print(f"ARCHIVED FAILURE   {archived_rel}")
        print(f"RERUN              {new_case}")
        print("#" * 96)

        ocls, operm, opos, oamp = classify_archived(old, true)
        print("\n-- ORIGINAL (archived) --")
        print(f"  output path            : Applications/Application1/output/{archived_rel}")
        print(f"  true control           : {np.array2string(true, precision=6)}")
        print(f"  initial control        : "
              f"{np.array2string(np.asarray(old['initial_control'], float), precision=6)}")
        print(f"  initialization mode    : {old.get('initialization_mode')}")
        print(f"  final control          : "
              f"{np.array2string(np.asarray(old['final_control'], float), precision=6)}")
        print(f"  initial / final J      : {old['initial_objective']:.6e} -> "
              f"{old['final_objective']:.6e}")
        print(f"  iterations / success   : {old['iterations']} / {old.get('success')}")
        print(f"  termination            : {old.get('message')}")
        print(f"  source-wise final error (optimal permutation):")
        for i in range(len(opos)):
            print(f"    true s{i+1}: dpos = {opos[i]:9.2f} m   dA = {oamp[i]:.4f}")
        print(f"  FAILURE CLASSIFICATION : {ocls} - {R.CLASSES[ocls]}")

        rec = R.analyse(new_case, admiss, bidx, args.master_log, args.log_dir)

        init = np.asarray(rec["initial_control"], float)
        iperm = match_perm(init, true)
        ipos, iamp = errors(init, true, iperm)
        print("\n-- NEW sensitivity initialization quality --")
        print(f"  label permutation for init: {list(iperm)}")
        for i in range(len(ipos)):
            print(f"    true s{i+1}: initial dpos = {ipos[i]:9.2f} m   "
                  f"initial dA = {iamp[i]:.4f}")

        R.plot_case(rec, admiss, out_dir / f"trajectory_{new_case}.png")

        table.append({
            "archived": archived_rel,
            "new_case": new_case,
            "true": true.tolist(),
            "old_initial": old["initial_control"],
            "old_final": old["final_control"],
            "old_J0": old["initial_objective"],
            "old_Jf": old["final_objective"],
            "old_iters": old["iterations"],
            "old_success": old.get("success"),
            "old_message": old.get("message"),
            "old_pos_err": opos, "old_amp_err": oamp, "old_class": ocls,
            "new_initial": rec["initial_control"],
            "new_init_pos_err": ipos, "new_init_amp_err": iamp,
            "new_J0": rec["initial_objective"],
            "new_final": rec["final_control"],
            "new_Jf": rec["final_objective"],
            "new_iters": rec["iterations_total"],
            "new_success": rec["success"],
            "new_message": rec["message"],
            "new_pos_err": [s["position_error_m"] for s in rec["sources"]],
            "new_amp_err": [s["amplitude_error"] for s in rec["sources"]],
            "new_entered_building": rec["any_iterate_entered_building"],
            "new_feasible": rec["final_feasible"],
            "new_amp_at_bound": any(
                s["amplitude_at_lower_bound"] or s["amplitude_at_upper_bound"]
                for s in rec["sources"]),
            "new_class": rec["classification"],
            "wall_seconds": rec["wall_seconds"],
        })

    # ------------------------------------------------------ comparison table
    print("\n" + "=" * 122)
    print("ORIGINAL FAILURE  vs  SENSITIVITY-INIT + RAW RERUN")
    print("=" * 122)
    hdr = (f"{'truth':<30}{'orig Jf':>11}{'orig dx':>9}{'orig cls':>9}   "
           f"{'init dx':>9}{'new J0':>11}{'new Jf':>11}{'new dx':>9}"
           f"{'new cls':>8}{'iters':>7}{'min':>6}")
    print(hdr)
    print("-" * len(hdr))
    for r in table:
        truth = "[" + " ".join(f"{v:g}" for v in r["true"]) + "]"
        print(f"{truth:<30}{r['old_Jf']:>11.3e}{max(r['old_pos_err']):>9.1f}"
              f"{r['old_class']:>9d}   {max(r['new_init_pos_err']):>9.2f}"
              f"{r['new_J0']:>11.3e}{r['new_Jf']:>11.3e}"
              f"{max(r['new_pos_err']):>9.2e}{r['new_class']:>8d}"
              f"{r['new_iters']:>7d}"
              + (f"{r['wall_seconds']/60.0:>6.0f}" if r["wall_seconds"]
                 else f"{'n/a':>6}"))
    print("-" * len(hdr))
    recovered = sum(1 for r in table if r["new_class"] == 1)
    print(f"\nrecovered (class 1): {recovered}/{len(table)}    "
          f"acceptable (class 2): {sum(1 for r in table if r['new_class'] == 2)}"
          f"    still failing (class 3-5): "
          f"{sum(1 for r in table if r['new_class'] >= 3)}")

    (out_dir / "rerun_vs_archive.json").write_text(
        json.dumps(table, indent=2, default=str) + "\n")
    print(f"wrote {out_dir / 'rerun_vs_archive.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
