"""Report / regression table for the source-count inference sweep.

Reads the saved output folders of the node-sensor inference cases and prints a
compact table plus, optionally, the full per-level detail.  It runs NOTHING --
it only reads ``source_count_inference_summary.json`` and
``optimization_summary.json``, so it is cheap to re-run after any sweep.

A case PASSES when
  * inference produced the expected source count,
  * the count came from a DECISION, not from hitting ``--max-sources``,
  * the first non-accepted level was actually tested and rejected (or the run
    stopped at the discrepancy floor, when that stop is enabled), and
  * ``sensors.vtu`` exists in the run folder.

Usage:
    PY="python3"
    cd "Applications/Application1/test scripts/diagnostics"
    "$PY" report_source_count_inference_sweep.py
    "$PY" report_source_count_inference_sweep.py --detail
    "$PY" report_source_count_inference_sweep.py --case my_case:3
"""

from __future__ import annotations

import argparse
import json
import os as _os
import sys as _sys
from pathlib import Path

_APP1_DIR = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _APP1_DIR not in _sys.path:
    _sys.path.insert(0, _APP1_DIR)

import source_count_inference as sci

APP_DIR = Path(_APP1_DIR)

#: (case folder, expected inferred M)
DEFAULT_CASES = [
    ("node_sensors_rbt_infer_m1_c01", 1),
    ("node_sensors_rbt_infer_m2_c02", 2),
    ("node_sensors_rbt_infer_m3_c01", 3),
    ("node_sensors_right_bottom_top_c06_infer_m", 2),
]

REJECTION_REASONS = {
    sci.STOP_AMPLITUDE_COLLAPSE,
    sci.STOP_INSUFFICIENT_ACTUAL,
    sci.STOP_NEGLIGIBLE_PREDICTED,
    sci.STOP_NO_CANDIDATE,
}


def load_case(case_name: str):
    base = APP_DIR / "output" / case_name / "full"
    inference_path = base / "source_count_inference_summary.json"
    optimization_path = base / "optimization_summary.json"
    if not inference_path.exists():
        return None
    inference = json.loads(inference_path.read_text(encoding="utf-8"))
    optimization = (
        json.loads(optimization_path.read_text(encoding="utf-8"))
        if optimization_path.exists()
        else None
    )
    return {
        "case": case_name,
        "base": base,
        "inference": inference,
        "optimization": optimization,
        "sensors_vtu": (base / "sensors.vtu").exists(),
    }


def classify_missing_final_summary(inference: dict) -> str | None:
    """Best-effort reason ``optimization_summary.json`` is missing.

    ``None`` means nothing to classify (the case wasn't asked about, or a
    final summary exists). This never raises -- an inference summary from
    before this classification existed simply falls back to
    ``"unknown_traceback"`` rather than crashing the report.
    """
    reason = inference.get("skipped_final_optimization_reason")
    if reason:
        text = str(reason)
        if "final-feasibility-polish" in text:
            return "infeasible_final_control_after_polish"
        if "validation" in text or "outside the fluid mesh" in text:
            return "validation_failure"
        return "skipped_final_optimization"
    if inference.get("inferred_control_feasible") is False:
        return "infeasible_final_control"
    status = inference.get("source_count_decision_status")
    if (
        status == "unresolved_max_sources"
        or inference.get("stop_reason") == sci.STOP_MAX_SOURCES
    ):
        return "max_sources_reached_unresolved"
    if inference.get("recovery_permutation_matched") is None and (
        inference.get("recovery_permutation_match_error")
        # Older inference summaries (pre-dating recovery_permutation_match_error)
        # only recorded inferred_matches_true_count -- the same underlying
        # condition (a permutation match is undefined for unequal M), so treat
        # it as the same signal rather than falling through to "unexplained".
        or inference.get("inferred_matches_true_count") is False
    ):
        return "permutation_count_mismatch"
    return "unknown_traceback"


def evaluate(record, expected_m: int):
    inference = record["inference"]
    inferred = int(inference["inferred_num_sources"])
    stop_reason = inference["stop_reason"]
    levels = inference["levels"]
    accepted = [L for L in levels if L.get("accepted")]
    rejected = [L for L in levels if not L.get("accepted")]

    # Prefer the explicit decision-status fields (added alongside the final
    # feasibility handoff); fall back to the raw stop reason for older
    # inference summaries that predate them, so this report never crashes on
    # an archived JSON file.
    decision_status = inference.get("source_count_decision_status")
    decision_converged = inference.get("source_count_decision_converged")
    if decision_status is None:
        decision_status, decision_converged = sci.decision_status_for_stop_reason(
            stop_reason
        )

    failures = []
    if inferred != expected_m:
        failures.append(f"inferred m={inferred}, expected {expected_m}")
    if not decision_converged:
        failures.append(
            f"source-count decision unresolved ({decision_status}), not a "
            "settled decision"
        )
    if stop_reason == sci.STOP_DISCREPANCY:
        # Legitimate stop, but only when the discrepancy stop was enabled.
        if float(inference["tolerances"]["discrepancy_tol"]) <= 0.0:
            failures.append("discrepancy stop reported while the tolerance is 0")
    elif stop_reason not in REJECTION_REASONS and stop_reason != sci.STOP_MAX_SOURCES:
        failures.append(f"unexpected stop reason {stop_reason!r}")
    if len(accepted) != expected_m:
        failures.append(f"{len(accepted)} accepted level(s), expected {expected_m}")
    if not record["sensors_vtu"]:
        failures.append("sensors.vtu missing")

    match = inference.get("recovery_permutation_matched")
    match_error = inference.get("recovery_permutation_match_error")
    optimization_exists = record["optimization"] is not None
    missing_final_summary_reason = (
        None if optimization_exists else classify_missing_final_summary(inference)
    )
    # true_num_sources is a newer field; older summaries only carry
    # true_control, so derive it from that as a fallback (never crash on a
    # missing key -- this is reporting, not the source-count decision).
    true_m = inference.get("true_num_sources")
    if true_m is None and inference.get("true_control"):
        true_m = len(inference["true_control"]) // 3
    return {
        "inferred": inferred,
        "true_m": true_m,
        "stop_reason": stop_reason,
        "decision_status": decision_status,
        "decision_converged": bool(decision_converged),
        "accepted_levels": len(accepted),
        "rejected_level": (rejected[0]["level"] if rejected else None),
        "rejected_reason": (rejected[0]["reason"] if rejected else None),
        "final_objective": float(inference["final_objective"]),
        "max_position_error": (
            None if not match else float(match["max_position_error_m"])
        ),
        "max_amplitude_error": (
            None if not match else float(match["max_amplitude_error"])
        ),
        "recovery_undefined_reason": match_error,
        "wall_time_seconds": float(inference.get("wall_time_seconds", 0.0)),
        "sensors_vtu": record["sensors_vtu"],
        "optimization_summary_exists": optimization_exists,
        "missing_final_summary_reason": missing_final_summary_reason,
        "failures": failures,
        "passed": not failures,
    }


def print_detail(record, verdict):
    inference = record["inference"]
    print(f"\n{'=' * 100}")
    print(f"CASE {record['case']}   (inferred m = {verdict['inferred']}, "
          f"true m = {verdict['true_m']}, stop: {verdict['stop_reason']})")
    print(f"{'=' * 100}")
    print(f"decision status  : {verdict['decision_status']}  "
          f"(converged={verdict['decision_converged']})")
    print(f"true control     : {inference.get('true_control')}  "
          "(SCORING ONLY -- not used to decide the reported source count)")
    print(f"final control    : "
          f"{[round(v, 9) for v in inference['final_control']]}")
    print(f"zero-source J    : {inference['zero_source_objective']:.6e}")
    print(f"final J          : {inference['final_objective']:.6e}")
    print(f"solves           : {inference['solves']}")
    print(f"inference wall   : {inference.get('wall_time_seconds', 0.0):.1f} s")
    print(f"sensors.vtu      : {record['sensors_vtu']}")
    if inference.get("recovery_permutation_matched"):
        match = inference["recovery_permutation_matched"]
        print(f"recovery         : max position "
              f"{match['max_position_error_m']:.6e} m, max amplitude "
              f"{match['max_amplitude_error']:.6e}")
    else:
        print(f"recovery         : {inference.get('recovery_note')}")
    if verdict["optimization_summary_exists"]:
        print("final summary    : optimization_summary.json present")
    else:
        print(
            "final summary    : MISSING -- "
            f"{verdict['missing_final_summary_reason']}"
        )
        skip_reason = inference.get("skipped_final_optimization_reason")
        if skip_reason:
            print(f"                   driver reason: {skip_reason}")

    header = (
        f"{'lvl':>3} {'J_before':>13} {'candidate center':>22} "
        f"{'predicted_dec':>14} {'J_after':>13} {'actual_dec':>13} "
        f"{'min_A':>12}  result"
    )
    print("\n" + header)
    print("-" * len(header))
    for level in inference["levels"]:
        candidate = level.get("candidate") or {}
        center = candidate.get("center")
        center_text = (
            "-" if center is None else f"({center[0]:.2f}, {center[1]:.2f})"
        )
        def fmt(value, width=13):
            return f"{'-':>{width}}" if value is None else f"{value:>{width}.5e}"
        result = "ACCEPT" if level.get("accepted") else f"REJECT {level['reason']}"
        print(
            f"{level['level']:>3} {level['J_before']:>13.5e} {center_text:>22} "
            f"{fmt(level.get('predicted_decrease'), 14)} "
            f"{fmt(level.get('J_after_candidate'))} "
            f"{fmt(level.get('actual_decrease'))} "
            f"{fmt(level.get('min_amplitude_after'), 12)}  {result}"
        )
        amps = level.get("amplitudes_after")
        if amps:
            print(f"{'':>3} amplitudes_after: "
                  f"{[float(f'{a:.9g}') for a in amps]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        action="append",
        default=None,
        metavar="NAME:EXPECTED_M",
        help="Override the case list, e.g. --case my_case:3 (repeatable).",
    )
    parser.add_argument("--detail", action="store_true", help="Per-level tables.")
    args = parser.parse_args()

    if args.case:
        cases = []
        for item in args.case:
            name, _, expected = item.rpartition(":")
            if not name or not expected.isdigit():
                raise SystemExit(f"--case expects NAME:EXPECTED_M, got {item!r}")
            cases.append((name, int(expected)))
    else:
        cases = DEFAULT_CASES

    rows = []
    missing = []
    for case_name, expected_m in cases:
        record = load_case(case_name)
        if record is None:
            missing.append(case_name)
            continue
        verdict = evaluate(record, expected_m)
        rows.append((case_name, expected_m, record, verdict))
        if args.detail:
            print_detail(record, verdict)

    header = (
        f"{'case':<42}{'exp':>4}{'inf':>4}{'true':>5}  {'decision_status':<24}"
        f"{'stop_reason':<38}"
        f"{'acc':>4}{'rej_lvl':>8}  {'rejected_reason':<38}"
        f"{'final_J':>13}{'max_pos_err':>13}{'max_amp_err':>13}"
        f"{'opt_sum':>9}  verdict"
    )
    print(f"\n{'=' * len(header)}")
    print("SOURCE-COUNT INFERENCE SWEEP")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for case_name, expected_m, _record, verdict in rows:
        def fmt(value):
            return "-" if value is None else f"{value:.4e}"
        print(
            f"{case_name:<42}{expected_m:>4}{verdict['inferred']:>4}"
            f"{str(verdict['true_m']):>5}  "
            f"{str(verdict['decision_status']):<24}"
            f"{str(verdict['stop_reason']):<38}"
            f"{verdict['accepted_levels']:>4}"
            f"{str(verdict['rejected_level']):>8}  "
            f"{str(verdict['rejected_reason']):<38}"
            f"{verdict['final_objective']:>13.4e}"
            f"{fmt(verdict['max_position_error']):>13}"
            f"{fmt(verdict['max_amplitude_error']):>13}"
            f"{('yes' if verdict['optimization_summary_exists'] else 'NO'):>9}  "
            f"{'PASS' if verdict['passed'] else 'FAIL'}"
        )
    print("-" * len(header))

    for case_name in missing:
        print(f"MISSING: {case_name} (no source_count_inference_summary.json)")

    no_final_summary = [
        r for r in rows if not r[3]["optimization_summary_exists"]
    ]
    for case_name, _expected, _record, verdict in no_final_summary:
        print(
            f"NO FINAL SUMMARY: {case_name} -- inferred M="
            f"{verdict['inferred']}, decision={verdict['decision_status']}, "
            f"final inference J={verdict['final_objective']:.4e}: "
            f"{verdict['missing_final_summary_reason']}"
            " (inference result reported above; this is NOT necessarily an "
            "optimizer failure)"
        )

    failed = [r for r in rows if not r[3]["passed"]]
    for case_name, _expected, _record, verdict in failed:
        print(f"\nFAIL {case_name}:")
        for message in verdict["failures"]:
            print(f"  - {message}")

    total_wall = sum(v["wall_time_seconds"] for _c, _e, _r, v in rows)
    print(f"\ncases: {len(rows)}   passed: {len(rows) - len(failed)}   "
          f"failed: {len(failed)}   missing: {len(missing)}")
    print(f"total inference wall time: {total_wall:.1f} s")
    if failed or missing:
        raise SystemExit(1)
    print("all cases passed")


if __name__ == "__main__":
    main()
