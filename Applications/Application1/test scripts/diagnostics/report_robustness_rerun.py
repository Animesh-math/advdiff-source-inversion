"""Summarize a robustness-sweep rerun folder (READ-ONLY, truth-aware).

Reports, for one ``output/<case>/full`` folder produced by
``--init-mode sensitivity`` (optionally with ``--sensitivity-branch-multistart``):
true control, initial control, retained geometric branches, per-start final
control, position/amplitude errors after optimal source matching, objectives,
optimizer message, and projection/penalty event counts.

Usage:
  cd /path/to/Advection-Diffusion-PDECO
  python3 \
    "Applications/Application1/test scripts/diagnostics/report_robustness_rerun.py" \
    <case-name> [<case-name> ...]
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = APP_DIR / "output"

FULL_RECOVERY_POSITION_TOL_M = 1.0
FULL_RECOVERY_AMPLITUDE_TOL = 0.05


def match_sources(true_control, final_control):
    """Optimal source permutation, minimizing total position error."""
    true_sources = np.asarray(true_control, dtype=np.float64).reshape(-1, 3)
    final_sources = np.asarray(final_control, dtype=np.float64).reshape(-1, 3)
    best = None
    for permutation in itertools.permutations(range(final_sources.shape[0])):
        candidate = final_sources[list(permutation)]
        errors = np.hypot(
            candidate[:, 0] - true_sources[:, 0],
            candidate[:, 1] - true_sources[:, 1],
        )
        total = float(np.sum(errors))
        if best is None or total < best[0]:
            best = (total, list(permutation), errors, candidate)
    _total, permutation, position_errors, matched = best
    amplitude_errors = np.abs(matched[:, 2] - true_sources[:, 2])
    return permutation, position_errors, amplitude_errors


def describe_start(summary: dict, label: str) -> dict:
    true_control = np.asarray(summary["true_control"], dtype=np.float64)
    final_control = np.asarray(summary["final_control"], dtype=np.float64)
    permutation, position_errors, amplitude_errors = match_sources(
        true_control, final_control
    )
    recovered = bool(
        np.all(position_errors <= FULL_RECOVERY_POSITION_TOL_M)
        and np.all(amplitude_errors <= FULL_RECOVERY_AMPLITUDE_TOL)
    )
    feasibility = summary.get("feasibility") or {}
    print(f"  {label}")
    print(f"    initial control : {np.array2string(np.asarray(summary['initial_control']), precision=4)}")
    print(f"    final control   : {np.array2string(final_control, precision=4)}")
    print(f"    source matching : final source order {permutation} vs truth")
    for index, (dp, da) in enumerate(zip(position_errors, amplitude_errors), start=1):
        print(f"      source {index}: position error {dp:10.4f} m   "
              f"amplitude error {da:.6f}")
    print(f"    objective       : {summary['initial_objective']:.6e} -> "
          f"{summary['final_objective']:.6e}")
    print(f"    gradient norm   : {summary['final_gradient_norm']:.6e}")
    print(f"    iterations      : {summary['iterations']}   "
          f"success={summary['success']}")
    print(f"    message         : {summary['message']}")
    print(f"    feasibility     : mode={feasibility.get('feasibility_mode')}  "
          f"invalid={feasibility.get('invalid_evaluations')}  "
          f"projected={feasibility.get('projected_evaluations')}  "
          f"penalized={feasibility.get('penalized_evaluations')}  "
          f"max move={feasibility.get('max_projection_distance')}")
    print(f"    full recovery   : {'YES' if recovered else 'no'}")
    return {
        "label": label,
        "final_objective": float(summary["final_objective"]),
        "position_errors": position_errors.tolist(),
        "amplitude_errors": amplitude_errors.tolist(),
        "recovered": recovered,
        "message": summary["message"],
    }


def report_stage_table(base: Path) -> None:
    path = base / "sigma_continuation_summary.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text())
    print("    continuation stages:")
    for stage in payload["stages"]:
        projected = stage.get("projected_evaluations")
        penalized = stage.get("penalized_evaluations")
        print(f"      sigma2={stage['sigma2']:>7.1f}  "
              f"J {stage['stage_initial_objective']:.4e} -> "
              f"{stage['stage_final_objective']:.4e}  "
              f"iters={stage['iterations']:>3}  "
              f"projected={projected}  penalized={penalized}  "
              f"{stage['message']}")


def report_case(case_name: str) -> None:
    base = OUTPUT_ROOT / case_name / "full"
    print(f"\n=== {case_name} ===")
    if not base.exists():
        print(f"  MISSING: {base}")
        return

    multistart_path = base / "multistart_summary.json"
    if multistart_path.exists():
        payload = json.loads(multistart_path.read_text())
        branches = payload.get("sensitivity_branches") or []
        print(f"  retained geometric branches: {len(branches)}")
        for branch in branches:
            centers = ", ".join(f"({c[0]:.2f},{c[1]:.2f})" for c in branch["centers"])
            walls = ", ".join(f"{d:.3f}" for d in branch["wall_distances_m"])
            amps = ", ".join(f"{a:.4f}" for a in branch["amplitudes"])
            print(f"    branch {branch['branch_index']}: [{centers}]  "
                  f"amps [{amps}]  J_exact={branch['objective']:.6e}  "
                  f"wall distances [{walls}] m")
        print(f"  starts run: {payload.get('num_starts')}  "
              f"best start: {payload.get('best_start_index')}")
        results = []
        for start_dir in sorted(base.glob("start_*")):
            summary_path = start_dir / "optimization_summary.json"
            if not summary_path.exists():
                continue
            summary = json.loads(summary_path.read_text())
            results.append(describe_start(summary, start_dir.name))
            report_stage_table(start_dir)
        if results:
            best = min(results, key=lambda item: item["final_objective"])
            print(f"  SELECTED (smallest exact objective): {best['label']}  "
                  f"J={best['final_objective']:.6e}  "
                  f"full recovery={'YES' if best['recovered'] else 'no'}")
        return

    summary_path = base / "optimization_summary.json"
    if not summary_path.exists():
        print(f"  MISSING: {summary_path}")
        return
    summary = json.loads(summary_path.read_text())
    print(f"  true control: {np.array2string(np.asarray(summary['true_control']), precision=4)}")
    describe_start(summary, "single start")
    report_stage_table(base)


def main() -> None:
    cases = sys.argv[1:]
    if not cases:
        print(__doc__)
        raise SystemExit(2)
    for case_name in cases:
        report_case(case_name)


if __name__ == "__main__":
    main()
