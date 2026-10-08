"""Concise end-of-run report for one Application1 M-source recovery (read-only).

Reads the artefacts a single ``run_optimization.py`` run writes --
``sensitivity_init_summary.json``, ``optimization_summary.json``,
``sigma_continuation_summary.json`` -- plus the archived run log, and prints and
saves one summary containing the truth, the selected initialization, objectives,
the per-stage optimizer termination status, permutation-matched per-source
errors, terminal feasibility/clearances, building crossings during raw
optimization, and the full PDE-solve / wall-time accounting.

The outcome is classified with the seven labels established by the M=2
robustness sweep (the development log).  Thresholds are the sweep's, fixed
before any M=3 result was seen:

  PERFECT / machine precision  <= 1 m and <= 1e-3 amplitude with J <= 1e-10
  NEAR-EXACT                   <= 5 m and <= 1e-2
  ACCEPTABLE                   <= 20 m and <= 0.15, sources still separated
  WRONG FEASIBLE LOCAL MINIMUM / INFEASIBLE TERMINAL SOLUTION /
  INITIALIZATION FAILURE / OPTIMIZER FAILURE

Nothing here is imported by production code; nothing but the report is written.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

_APP1_DIR = Path(__file__).resolve().parents[2]
if str(_APP1_DIR) not in sys.path:
    sys.path.insert(0, str(_APP1_DIR))

import run_optimization as ro  # noqa: E402
from mesh_admissibility import MeshAdmissibility  # noqa: E402
from mesh_data_config import build_mesh_data  # noqa: E402


PERFECT_POS_M, PERFECT_AMP = 1.0, 1.0e-3
PERFECT_J = 1.0e-10
NEAR_POS_M, NEAR_AMP = 5.0, 1.0e-2
ACCEPT_POS_M, ACCEPT_AMP = 20.0, 0.15
MIN_SEPARATION_M = 20.0


def _load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def classify(summary, continuation, match, clearances, admissible_all) -> tuple[str, str]:
    if summary is None:
        return (
            "INITIALIZATION FAILURE",
            "no optimization summary exists; the run aborted before or inside "
            "the initializer",
        )
    j_final = float(summary["final_objective"])
    max_pos = float(match["max_position_error_m"])
    max_amp = float(match["max_amplitude_error"])
    if not admissible_all:
        return (
            "INFEASIBLE TERMINAL SOLUTION",
            f"a terminal source center lies outside the fluid mesh "
            f"(clearances {['%.3f' % c for c in clearances]})",
        )
    if max_pos <= PERFECT_POS_M and max_amp <= PERFECT_AMP and j_final <= PERFECT_J:
        return (
            "EXACT / MACHINE-PRECISION RECOVERY",
            f"max position error {max_pos:.4g} m <= {PERFECT_POS_M} m, "
            f"max amplitude error {max_amp:.4g} <= {PERFECT_AMP}, "
            f"J_final {j_final:.4g} <= {PERFECT_J}",
        )
    if max_pos <= NEAR_POS_M and max_amp <= NEAR_AMP:
        return (
            "NEAR-EXACT RECOVERY",
            f"max position error {max_pos:.4g} m <= {NEAR_POS_M} m, "
            f"max amplitude error {max_amp:.4g} <= {NEAR_AMP}",
        )
    if max_pos <= ACCEPT_POS_M and max_amp <= ACCEPT_AMP:
        return (
            "ACCEPTABLE RECOVERY",
            f"max position error {max_pos:.4g} m <= {ACCEPT_POS_M} m, "
            f"max amplitude error {max_amp:.4g} <= {ACCEPT_AMP}",
        )
    stages = (continuation or {}).get("stages") or []
    if stages and not bool(stages[-1].get("success", False)):
        message = str(stages[-1].get("message", ""))
        if "ABNORMAL_TERMINATION_IN_LNSRCH" in message:
            return (
                "OPTIMIZER FAILURE",
                f"the final continuation stage terminated abnormally: {message}",
            )
    return (
        "WRONG FEASIBLE LOCAL MINIMUM",
        f"terminal point is feasible but max position error {max_pos:.4g} m / "
        f"max amplitude error {max_amp:.4g} exceed the acceptable band",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="output/<case>/<mode-folder> of the run")
    parser.add_argument("--log", default=None, help="archived stdout log of the run")
    parser.add_argument("--out", default=None, help="write the report JSON here")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    summary = _load(run_dir / "optimization_summary.json")
    continuation = _load(run_dir / "sigma_continuation_summary.json")
    init = _load(run_dir / "sensitivity_init_summary.json")
    log_text = (
        Path(args.log).read_text(encoding="utf-8", errors="replace")
        if args.log and Path(args.log).exists()
        else ""
    )

    out: list[str] = []

    def emit(line=""):
        print(line)
        out.append(line)

    emit("=" * 78)
    emit(f"Application1 M-source recovery report: {run_dir}")
    emit("=" * 78)

    if summary is None:
        emit("\nNo optimization_summary.json: the run produced no optimized control.")
        if init is not None:
            emit(f"An initializer summary exists at {run_dir}/sensitivity_init_summary.json")
        label, reason = classify(None, None, None, [], False)
        emit(f"\nOUTCOME: {label}\n  {reason}")
        return 1

    true_control = np.asarray(summary["true_control"], dtype=np.float64)
    initial_control = np.asarray(summary["initial_control"], dtype=np.float64)
    final_control = np.asarray(summary["final_control"], dtype=np.float64)
    num_sources = int(summary.get("num_sources", true_control.size // 3))

    emit("\n--- Problem ---")
    emit(f"number of sources M      : {num_sources}")
    emit(f"control dimension        : {true_control.size}")
    emit(f"optimization mode        : {summary['optimization_mode']}")
    emit(f"initialization mode      : {summary['initialization_mode']}")
    emit(f"measurement boundaries   : {summary['measurement_boundaries']}")
    emit(f"feasibility mode         : {(summary.get('feasibility') or {}).get('feasibility_mode')}")
    emit(f"sigma2 continuation      : {summary.get('sigma2_continuation')}")
    emit(f"target sigma2            : {summary.get('target_sigma2')}")
    emit(f"amplitude floor          : {summary.get('amplitude_floor')}")
    emit(f"tfinal                   : {(summary.get('pde') or {}).get('tfinal')}")

    emit("\n--- Controls ---")
    emit(f"true control             : {np.array2string(true_control, precision=6)}")
    emit(f"initial control (selected): {np.array2string(initial_control, precision=6)}")
    emit(f"final control            : {np.array2string(final_control, precision=6)}")

    emit("\n--- Objectives ---")
    emit(f"initial objective        : {float(summary['initial_objective']):.10e}")
    emit(f"final objective          : {float(summary['final_objective']):.10e}")
    emit(f"final gradient norm      : {float(summary['final_gradient_norm']):.10e}")
    if init is not None:
        emit(f"zero-source objective    : {init.get('zero_source_objective')}")
        emit(f"initializer exact J      : {init.get('initial_exact_objective')}")

    emit("\n--- Initialization accounting ---")
    if init is None:
        emit("(no sensitivity_init_summary.json)")
    else:
        emit(f"score / top-K / beam     : {init.get('score')} / {init.get('top_k')} / {init.get('beam_width')}")
        emit(f"initializer forward solves : {init.get('n_forward_solves')}")
        emit(f"initializer adjoint solves : {init.get('n_adjoint_solves')}")
        emit(f"candidate traces evaluated : {init.get('n_candidate_traces')}")
        wall = init.get("wall_time_seconds")
        emit(
            "initializer wall time      : "
            + (f"{float(wall):.1f} s ({float(wall) / 60.0:.1f} min)"
               if wall is not None else "n/a")
        )
        for record in init.get("source_levels") or []:
            emit(
                f"  source level {record['level']}/{record['num_sources']}: "
                f"{record['branches_in']} branch(es) in -> "
                f"{record['expansions_considered']} expansion(s) -> "
                f"{record['branches_surviving']} surviving; "
                f"{record['n_forward_solves']} fwd / {record['n_adjoint_solves']} adj / "
                f"{record['n_candidate_traces']} candidate traces; "
                f"best J {record['best_objective']:.6e}"
            )
        branches = init.get("branch_controls") or []
        emit(f"  retained branches at the end: {len(branches)}")
        for record in branches:
            centers = ", ".join(f"({c[0]:.2f},{c[1]:.2f})" for c in record["centers"])
            emit(f"    branch {record['branch_index']}: [{centers}] "
                 f"J={record['objective']:.6e}")

    emit("\n--- Continuation stages ---")
    stages = (continuation or {}).get("stages") or []
    if not stages:
        emit(f"(single stage) iterations={summary['iterations']} "
             f"success={summary['success']} message={summary['message']}")
    total_iterations = 0
    for stage in stages:
        total_iterations += int(stage["iterations"])
        emit(
            f"  stage {stage['stage_index']} sigma2={stage['sigma2']:g}: "
            f"J {stage['stage_initial_objective']:.6e} -> "
            f"{stage['stage_final_objective']:.6e}  "
            f"iters={stage['iterations']}  success={stage['success']}  "
            f"status={stage['message']}"
        )
    if stages:
        emit(f"  total optimizer iterations across stages: {total_iterations}")

    emit("\n--- Recovery (permutation-matched) ---")
    match = ro.match_sources_by_permutation(true_control, final_control)
    init_match = ro.match_sources_by_permutation(true_control, initial_control)
    emit(f"optimal label permutation : {match['permutation']} "
         f"(labels permuted = {match['labels_permuted']})")
    for i in range(num_sources):
        j = match["permutation"][i]
        emit(
            f"  true s{i + 1} <- final s{j + 1}: position error "
            f"{match['position_errors_m'][i]:12.6f} m   amplitude error "
            f"{match['amplitude_errors'][i]:.6e}"
        )
    emit(f"max position error        : {match['max_position_error_m']:.6f} m")
    emit(f"max amplitude error       : {match['max_amplitude_error']:.6e}")
    emit(f"initialization max position error : "
         f"{init_match['max_position_error_m']:.6f} m")
    emit(f"initialization max amplitude error: "
         f"{init_match['max_amplitude_error']:.6e}")

    emit("\n--- Terminal feasibility and wall clearance ---")
    admissibility = MeshAdmissibility(build_mesh_data())
    clearances, admissible = [], []
    for index, (x, y, amplitude) in enumerate(final_control.reshape(-1, 3), start=1):
        inside = bool(admissibility.is_point_in_domain(float(x), float(y)))
        clearance = float(admissibility.distance_to_boundary(float(x), float(y)))
        clearances.append(clearance)
        admissible.append(inside)
        emit(
            f"  source {index}: ({x:9.4f},{y:9.4f}) A={amplitude:.6f}  "
            f"fluid-admissible={inside}  wall clearance={clearance:9.4f} m"
        )

    emit("\n--- Raw-mode building crossings ---")
    feasibility = summary.get("feasibility") or {}
    emit(f"invalid evaluations        : {feasibility.get('invalid_evaluations')}")
    emit(f"projected evaluations      : {feasibility.get('projected_evaluations')}")
    emit(f"penalized evaluations      : {feasibility.get('penalized_evaluations')}")
    emit(f"raw infeasible evaluations : {feasibility.get('raw_infeasible_evaluations')}")
    emit(f"raw infeasible iterates    : {feasibility.get('raw_infeasible_iterates')}")
    emit(f"raw max distance outside   : "
         f"{feasibility.get('raw_max_outside_distance')} m")
    crossing_log = feasibility.get("raw_infeasible_iterate_log") or []
    if crossing_log:
        by_source: dict[int, list[int]] = {}
        for entry in crossing_log:
            for source in entry["sources"]:
                by_source.setdefault(int(source["source_id"]), []).append(
                    int(entry["iteration"])
                )
        for source_id, iterations in sorted(by_source.items()):
            emit(
                f"  source {source_id}: {len(iterations)} in-building iterate(s), "
                f"iterations {min(iterations)}..{max(iterations)}"
            )
    else:
        emit("  no accepted iterate left the fluid mesh")

    emit("\n--- Runtime ---")
    runtime = None
    if log_text:
        matches = re.findall(r"^TOTAL WALL TIME:\s*([0-9.]+)\s*s", log_text, re.M)
        if matches:
            runtime = float(matches[-1])
    if runtime is not None:
        emit(f"total wall time            : {runtime:.1f} s ({runtime / 3600.0:.2f} h)")
    else:
        emit("total wall time            : (not recorded in the supplied log)")

    label, reason = classify(
        summary, continuation, match, clearances, all(admissible)
    )
    emit("\n" + "=" * 78)
    emit(f"OUTCOME: {label}")
    emit(f"  {reason}")
    emit("=" * 78)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(out) + "\n", encoding="utf-8")
        json_path = out_path.with_suffix(".json")
        json_path.write_text(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "num_sources": num_sources,
                    "true_control": true_control.tolist(),
                    "initial_control": initial_control.tolist(),
                    "final_control": final_control.tolist(),
                    "initial_objective": float(summary["initial_objective"]),
                    "final_objective": float(summary["final_objective"]),
                    "final_gradient_norm": float(summary["final_gradient_norm"]),
                    "recovery": match,
                    "initialization_recovery": init_match,
                    "stages": stages,
                    "total_optimizer_iterations": total_iterations,
                    "terminal_wall_clearances_m": clearances,
                    "terminal_admissible": admissible,
                    "feasibility": feasibility,
                    "initializer": {
                        key: (init or {}).get(key)
                        for key in (
                            "n_forward_solves",
                            "n_adjoint_solves",
                            "n_candidate_traces",
                            "wall_time_seconds",
                            "source_levels",
                            "zero_source_objective",
                            "initial_exact_objective",
                        )
                    },
                    "total_wall_time_seconds": runtime,
                    "outcome": label,
                    "outcome_reason": reason,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nreport: {out_path}\nreport JSON: {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
