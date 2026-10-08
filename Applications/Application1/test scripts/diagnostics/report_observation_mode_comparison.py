"""Compare an archived boundary-mode run with a node-mode rerun of the same case.

Reads two completed Application1 run folders (each containing
``optimization_summary.json`` and, for continuation runs,
``sigma_continuation_summary.json``) and prints the side-by-side quantities used
to accept the boundary-equivalent node-sensor regression: selected
initialization, initial objective, per-stage objective and iterations, final
control, final objective, permutation-matched position/amplitude errors, final
gradient norm, optimizer termination, terminal feasibility and wall clock.

Usage:
    PY="python3"
    cd "Applications/Application1/test scripts/diagnostics"
    "$PY" report_observation_mode_comparison.py \
        --archived final_sweep_c06_nearwall_b13/full \
        --new node_equiv_c06_nearwall_b13/full \
        --archived-seconds 1808 --new-seconds 1900
Paths are relative to Applications/Application1/output.
"""

from __future__ import annotations

import argparse
import json
import os as _os
import sys as _sys
from pathlib import Path

import numpy as np

_APP1_DIR = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _APP1_DIR not in _sys.path:
    _sys.path.insert(0, _APP1_DIR)

import run_optimization as app1
from mesh_admissibility import MeshAdmissibility


OUTPUT_ROOT = Path(_APP1_DIR) / "output"


def load(run_dir: Path) -> dict:
    summary = json.loads((run_dir / "optimization_summary.json").read_text())
    stages_path = run_dir / "sigma_continuation_summary.json"
    summary["_stages"] = (
        json.loads(stages_path.read_text()) if stages_path.exists() else None
    )
    init_path = run_dir / "sensitivity_init_summary.json"
    summary["_init"] = (
        json.loads(init_path.read_text()) if init_path.exists() else None
    )
    return summary


def errors(true_control, control) -> dict:
    return app1.match_sources_by_permutation(
        np.asarray(true_control, dtype=np.float64),
        np.asarray(control, dtype=np.float64),
    )


def feasibility_report(control, admissibility) -> dict:
    centers = np.asarray(control, dtype=np.float64).reshape(-1, 3)[:, :2]
    return {
        "all_admissible": all(
            bool(admissibility.is_point_in_domain(float(x), float(y)))
            for x, y in centers
        ),
        "clearances_m": [
            float(admissibility.distance_to_boundary(float(x), float(y)))
            for x, y in centers
        ],
    }


def fmt(values, precision: int = 10) -> str:
    return np.array2string(
        np.asarray(values, dtype=np.float64), precision=precision, max_line_width=200
    )


def describe(name: str, run: dict, admissibility) -> None:
    observation = run.get("observation", {"observation_mode": "boundary"})
    print(f"\n===== {name} =====")
    print(f"case: {run['case_name']}")
    print(f"observation mode: {observation.get('observation_mode')}")
    if observation.get("observation_mode") == "nodes":
        print(f"  sensors: {observation.get('num_sensors')}")
        print(f"  weights: {observation.get('weight_kind')} / {observation.get('weight_convention')}")
    print(f"measurement boundaries: {run['measurement_boundaries']}")
    print(f"true control:  {fmt(run['true_control'])}")
    print(f"selected initialization: {fmt(run['input_initial_control'])}")
    init_err = errors(run["true_control"], run["input_initial_control"])
    print(
        "  init max position error: "
        f"{init_err['max_position_error_m']:.6f} m   max amplitude error: "
        f"{init_err['max_amplitude_error']:.6e}"
    )
    print(f"initial objective: {run['initial_objective']:.16e}")
    if run["_stages"]:
        for stage in run["_stages"]["stages"]:
            print(
                f"  stage {stage['stage_index']} sigma2={stage['sigma2']:g}: "
                f"J0={stage['stage_initial_objective']:.10e} "
                f"Jf={stage['stage_final_objective']:.10e} "
                f"iters={stage['iterations']} success={stage['success']}"
            )
            print(f"    stage final control: {fmt(stage['stage_final_control'])}")
    print(f"final control: {fmt(run['final_control'], 12)}")
    print(f"final objective: {run['final_objective']:.16e}")
    print(f"final gradient norm: {run['final_gradient_norm']:.16e}")
    err = errors(run["true_control"], run["final_control"])
    print(f"permutation (true -> final): {err['permutation']}")
    print(f"label permutation required: {err['labels_permuted']}")
    print(f"per-source position errors (m): {err['position_errors_m']}")
    print(f"per-source amplitude errors: {err['amplitude_errors']}")
    print(f"max position error: {err['max_position_error_m']:.6e} m")
    print(f"max amplitude error: {err['max_amplitude_error']:.6e}")
    print(f"optimizer success: {run['success']}  message: {run['message']}")
    feas = feasibility_report(run["final_control"], admissibility)
    print(f"terminal feasibility: all admissible = {feas['all_admissible']}")
    print(
        "  wall clearances: "
        + ", ".join(f"{c:.4f} m" for c in feas["clearances_m"])
    )
    fs = run.get("feasibility") or {}
    print(
        "raw-mode incursions: "
        f"{fs.get('raw_infeasible_iterates')} accepted iterate(s), "
        f"{fs.get('raw_infeasible_evaluations')} evaluation(s), "
        f"max {fs.get('raw_max_outside_distance')} m outside"
    )
    if run["_init"]:
        init = run["_init"]
        print(
            "initializer cost: "
            f"{init.get('n_forward_solves')} forward / "
            f"{init.get('n_adjoint_solves')} adjoint solves, "
            f"{init.get('n_candidate_traces')} candidate traces, "
            f"{init.get('wall_time_seconds')} s"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archived", required=True)
    parser.add_argument("--new", required=True)
    parser.add_argument("--archived-seconds", type=float, default=None)
    parser.add_argument("--new-seconds", type=float, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    archived = load(OUTPUT_ROOT / args.archived)
    new = load(OUTPUT_ROOT / args.new)
    admissibility = MeshAdmissibility(app1.build_mesh_data())

    describe("ARCHIVED BOUNDARY RUN", archived, admissibility)
    if args.archived_seconds is not None:
        print(f"wall clock: {args.archived_seconds:.0f} s")
    describe("NEW BOUNDARY-EQUIVALENT NODE RUN", new, admissibility)
    if args.new_seconds is not None:
        print(f"wall clock: {args.new_seconds:.0f} s")

    print("\n===== DIFFERENCES =====")
    for key in (
        "input_initial_control",
        "final_control",
    ):
        a = np.asarray(archived[key], dtype=np.float64)
        b = np.asarray(new[key], dtype=np.float64)
        print(f"{key}: max |difference| = {np.max(np.abs(a - b)):.6e}")
    for key in ("initial_objective", "final_objective", "final_gradient_norm"):
        a, b = float(archived[key]), float(new[key])
        rel = abs(b - a) / max(abs(a), 1.0e-300)
        print(f"{key}: archived={a:.16e} new={b:.16e} rel diff={rel:.6e}")
    if archived["_stages"] and new["_stages"]:
        for sa, sb in zip(archived["_stages"]["stages"], new["_stages"]["stages"]):
            print(
                f"stage {sa['stage_index']}: iterations {sa['iterations']} -> "
                f"{sb['iterations']}; Jf {sa['stage_final_objective']:.6e} -> "
                f"{sb['stage_final_objective']:.6e}"
            )
    print(
        "identical optimizer termination: "
        f"{archived['message'] == new['message']}"
    )

    if args.out:
        payload = {
            "archived": {k: v for k, v in archived.items() if not k.startswith("_")},
            "new": {k: v for k, v in new.items() if not k.startswith("_")},
            "archived_stages": archived["_stages"],
            "new_stages": new["_stages"],
            "archived_wall_seconds": args.archived_seconds,
            "new_wall_seconds": args.new_seconds,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
