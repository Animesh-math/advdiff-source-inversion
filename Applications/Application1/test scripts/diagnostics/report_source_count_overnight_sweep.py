"""Summarize completed cases from the overnight source-count inference sweep.

Reads output/<case-name>/full/source_count_inference_summary.json (and, when
present, optimization_summary.json) for every case named in a manifest.json
written by build_source_count_overnight_sweep.py, and prints one row per case:

    case, true_m, inferred_m, stop_reason, accepted_levels,
    rejected_level_reason, final_J, max_position_error, max_amp_error,
    polish_max_iter, num_sensors, PASS/FAIL/UNKNOWN

PASS/FAIL certifies the SOURCE COUNT ONLY (inferred_m == true_m), not
recovery accuracy, matching report_source_count_inference_sweep.py's existing
convention.

Usage:
    "$PY" report_source_count_overnight_sweep.py --manifest <logdir>/manifest.json
    "$PY" report_source_count_overnight_sweep.py --manifest <logdir>/manifest.json --detail
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_APP1_DIR = Path(__file__).resolve().parents[2]
if str(_APP1_DIR) not in sys.path:
    sys.path.insert(0, str(_APP1_DIR))
import source_count_inference as sci

OUTPUT_DIR = _APP1_DIR / "output"


def decision_status(inf: dict) -> tuple[str, bool]:
    """``(source_count_decision_status, converged)``, with a fallback for
    inference summaries written before those fields existed."""
    status = inf.get("source_count_decision_status")
    converged = inf.get("source_count_decision_converged")
    if status is not None:
        return status, bool(converged)
    return sci.decision_status_for_stop_reason(inf.get("stop_reason"))


def classify_missing_final_summary(inf: dict) -> str:
    """Best-effort reason ``optimization_summary.json`` is missing.

    NOT a generic "the optimizer crashed" label: a missing final summary is
    the EXPECTED outcome when the source-count decision never converged
    (``--max-sources`` reached) or the inferred control could not be
    certified feasible, and the driver now skips final optimization
    gracefully in those cases (see run_optimization.py's
    ``skipped_final_optimization_reason`` handling) rather than crashing.
    """
    reason = inf.get("skipped_final_optimization_reason")
    if reason:
        text = str(reason)
        if "final-feasibility-polish" in text:
            return "infeasible_final_control_after_polish"
        if "validation" in text or "outside the fluid mesh" in text:
            return "validation_failure"
        return "skipped_final_optimization"
    if inf.get("inferred_control_feasible") is False:
        return "infeasible_final_control"
    _status, converged = decision_status(inf)
    if not converged:
        return "max_sources_reached_unresolved"
    if inf.get("recovery_permutation_matched") is None and (
        inf.get("recovery_permutation_match_error")
        # Older inference summaries (pre-dating recovery_permutation_match_error)
        # only recorded inferred_matches_true_count -- the same underlying
        # condition (a permutation match is undefined for unequal M), so treat
        # it as the same signal rather than falling through to "unexplained".
        or inf.get("inferred_matches_true_count") is False
    ):
        return "permutation_count_mismatch"
    if "source_count_decision_status" not in inf:
        # Predates the source-count bookkeeping fix entirely (no new fields
        # at all): most plausibly the historical write_run_outputs crash this
        # script used to describe by name. Kept as its own label so an old
        # archived run is still distinguishable from a genuinely unexplained
        # new one.
        return "unknown_traceback_pre_source_count_fix"
    return "unknown_traceback"


def load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def summarize_case(case: dict) -> dict:
    name = case["case_name"]
    true_m = case["num_true_sources"]
    run_dir = OUTPUT_DIR / name / "full"
    inf_path = run_dir / "source_count_inference_summary.json"
    opt_path = run_dir / "optimization_summary.json"

    inf = load_json(inf_path)
    opt = load_json(opt_path)

    row = {
        "case": name,
        "true_m": true_m,
        "inferred_m": None,
        "stop_reason": None,
        "decision_status": None,
        "decision_converged": None,
        "accepted_levels": None,
        "rejected_level_reason": None,
        "final_J": None,
        "max_position_error": None,
        "max_amp_error": None,
        "polish_max_iter": case["polish_max_iter"],
        "num_sensors": case["num_sensors"],
        "optimization_summary_exists": False,
        "status": "UNKNOWN",
        "note": "",
    }

    if inf is None:
        row["note"] = "no source_count_inference_summary.json found (case not run / still running / crashed early)"
        return row

    inferred_m = inf.get("inferred_num_sources")
    row["inferred_m"] = inferred_m
    row["stop_reason"] = inf.get("stop_reason")
    row["decision_status"], row["decision_converged"] = decision_status(inf)

    levels = inf.get("levels", [])
    accepted = [lv for lv in levels if lv.get("accepted")]
    rejected = [lv for lv in levels if not lv.get("accepted")]
    row["accepted_levels"] = len(accepted)
    if rejected:
        last_rej = rejected[-1]
        row["rejected_level_reason"] = last_rej.get("reason")

    matches_true = bool(inf.get("inferred_matches_true_count"))
    row["optimization_summary_exists"] = opt is not None

    if opt is not None:
        row["final_J"] = opt.get("final_objective")
        rec = opt.get("recovery_permutation_matched") or {}
        row["max_position_error"] = rec.get("max_position_error_m")
        row["max_amp_error"] = rec.get("max_amplitude_error")
    else:
        # Still report the INFERENCE result -- a missing final summary is not
        # by itself an optimizer failure; classify_missing_final_summary
        # distinguishes an unresolved max-sources decision, an infeasible
        # final control, a permutation count mismatch, and (only as a last
        # resort) an unexplained case.
        row["final_J"] = inf.get("final_objective")
        perm = inf.get("recovery_permutation_matched")
        if isinstance(perm, dict):
            row["max_position_error"] = perm.get("max_position_error_m")
            row["max_amp_error"] = perm.get("max_amplitude_error")
        reason = classify_missing_final_summary(inf)
        row["note"] = f"no optimization_summary.json: {reason}"
        if reason == "max_sources_reached_unresolved":
            row["note"] += (
                " (inferred M is a LOWER BOUND; the acceptance rule was "
                "never tested at the next level)"
            )
        elif inferred_m is not None and matches_true:
            row["note"] += " (source-count decision itself is still valid)"

    if inferred_m is None:
        row["status"] = "UNKNOWN"
    elif inferred_m == true_m:
        row["status"] = "PASS"
    else:
        row["status"] = "FAIL"

    return row


def fmt(v, width, prec=None):
    if v is None:
        s = "-"
    elif isinstance(v, float) and prec is not None:
        s = f"{v:.{prec}e}"
    else:
        s = str(v)
    return s.ljust(width)[:width]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--detail", action="store_true", help="also print rejected-level detail and sensor family")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cases = manifest["cases"]

    rows = [summarize_case(c) for c in cases]

    header = [
        ("case", 42), ("true_m", 6), ("inferred_m", 10),
        ("decision_status", 24), ("stop_reason", 34),
        ("accepted", 8), ("rejected_reason", 32), ("final_J", 12),
        ("max_pos_err", 12), ("max_amp_err", 12), ("polish_iter", 11),
        ("n_sensors", 9), ("opt_sum", 8), ("status", 8),
    ]
    print("  ".join(name.ljust(w)[:w] for name, w in header))
    print("  ".join("-" * w for _, w in header))
    for row in rows:
        cells = [
            fmt(row["case"], 42), fmt(row["true_m"], 6), fmt(row["inferred_m"], 10),
            fmt(row["decision_status"], 24),
            fmt(row["stop_reason"], 34), fmt(row["accepted_levels"], 8),
            fmt(row["rejected_level_reason"], 32), fmt(row["final_J"], 12, 4),
            fmt(row["max_position_error"], 12, 4), fmt(row["max_amp_error"], 12, 4),
            fmt(row["polish_max_iter"], 11), fmt(row["num_sensors"], 9),
            fmt("yes" if row["optimization_summary_exists"] else "NO", 8),
            fmt(row["status"], 8),
        ]
        print("  ".join(cells))

    n_pass = sum(1 for r in rows if r["status"] == "PASS")
    n_fail = sum(1 for r in rows if r["status"] == "FAIL")
    n_unknown = sum(1 for r in rows if r["status"] == "UNKNOWN")
    print(f"\n{n_pass}/{len(rows)} PASS, {n_fail} FAIL, {n_unknown} UNKNOWN "
          "(PASS/FAIL certifies the source COUNT only, not recovery accuracy)")

    if args.detail:
        print("\n--- detail ---")
        for c, row in zip(cases, rows):
            print(f"\n{row['case']}  ({c.get('sensor_family', '?')})")
            print(f"  purpose: {c.get('purpose', '')}")
            if row["note"]:
                print(f"  NOTE: {row['note']}")


if __name__ == "__main__":
    main()
