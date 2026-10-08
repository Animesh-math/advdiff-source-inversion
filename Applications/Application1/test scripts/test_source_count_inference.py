"""Focused tests for adaptive source-count inference.

Three groups, cheapest first:

A. **Stop rule** (pure python, no PDE).  ``acceptance_threshold`` is the whole
   decision rule, so it is tested directly: scale awareness, the absolute floor,
   and the boundary cases.  Then ``infer_source_count`` is driven with a STUB
   ``polish_fn`` and a stubbed screen so every stop reason is reached
   deterministically and asserted by name.

B. **Nesting / zero-amplitude behaviour** (pure python + one tiny PDE case).
   The core claim of the module is that inference must polish with amplitude
   lower bound 0, never the production floor.  Proved two ways:
     * the driver's polish callback is asserted to receive lower bound 0 and to
       build bounds whose amplitude lower entries are 0, while the production
       floor is only recorded as provenance;
     * a candidate that should collapse is shown to be REJECTED with reason
       ``rejected_amplitude_collapse`` when the bound is 0, and wrongly ACCEPTED
       when the bound is the production floor -- i.e. the floor manufactures
       evidence, which is exactly why it is excluded.

C. **c06 regression** (reads the completed inference run; does not re-run it).
   Asserts the recent recovered case infers m = 2 and that no third source was
   accepted, with the recorded reason.

Usage:
    PY="python3"
    cd "Applications/Application1/test scripts"
    "$PY" test_source_count_inference.py              # A + B + C (C skips if absent)
    "$PY" test_source_count_inference.py --unit-only  # A + B only
"""

from __future__ import annotations

import argparse
import json
import os as _os
import sys as _sys
from pathlib import Path

import numpy as np

_APP1_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _APP1_DIR not in _sys.path:
    _sys.path.insert(0, _APP1_DIR)

import source_count_inference as sci

APP_DIR = Path(_APP1_DIR)
C06_CASE = "node_sensors_right_bottom_top_c06_infer_m"

FAILURES: list[str] = []


def check(condition, message):
    print(f"  [{'PASS' if condition else 'FAIL'}] {message}")
    if not condition:
        FAILURES.append(message)
    return bool(condition)


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ---------------------------------------------------------------------------
# A. stop rule
# ---------------------------------------------------------------------------
def test_acceptance_threshold():
    section("A1. acceptance_threshold (the stop rule itself)")
    check(
        sci.acceptance_threshold(1000.0, 1.0e-3, 0.0) == 1.0,
        "relative term scales with J (1000 * 1e-3 = 1.0)",
    )
    check(
        sci.acceptance_threshold(1.0e-30, 1.0e-3, 1.0e-12) == 1.0e-12,
        "absolute floor dominates when J is at the numerical floor",
    )
    check(
        sci.acceptance_threshold(1000.0, 1.0e-3, 1.0e-12) == 1.0,
        "relative term dominates when J is large",
    )
    check(
        sci.acceptance_threshold(0.0, 1.0e-3, 0.0) == 0.0,
        "both terms zero gives a zero threshold",
    )
    # Scale invariance: the rule must not depend on the units of J.
    a = sci.acceptance_threshold(5.0, 0.01, 0.0)
    b = sci.acceptance_threshold(5.0e6, 0.01, 0.0)
    check(
        abs(b / a - 1.0e6) < 1.0e-9,
        "threshold is scale-equivariant in J (no hidden absolute scale)",
    )


def test_helpers():
    section("A2. control helpers")
    control = np.array([1.0, 2.0, 0.3, 4.0, 5.0, 0.6], dtype=np.float64)
    check(
        np.array_equal(sci.amplitudes_of(control), np.array([0.3, 0.6])),
        "amplitudes_of extracts every third entry",
    )
    check(
        sci.amplitudes_of(np.zeros(0)).size == 0,
        "amplitudes_of handles the empty (m=0) control",
    )
    grown = sci.append_source(control, 7.0, 8.0, 0.9)
    check(
        np.array_equal(grown, np.array([1.0, 2.0, 0.3, 4.0, 5.0, 0.6, 7.0, 8.0, 0.9])),
        "append_source appends [x, y, A] and preserves the prefix",
    )
    check(
        np.array_equal(sci.append_source(np.zeros(0), 1.0, 2.0, 3.0),
                       np.array([1.0, 2.0, 3.0])),
        "append_source grows the empty control to one source",
    )
    check(
        sci.NESTING_AMPLITUDE_LOWER == 0.0,
        "the nesting amplitude lower bound is 0",
    )


class _StubScreen:
    """Deterministic screen: replays a scripted list of candidate records."""

    def __init__(self, candidates, zero_source_objective=1000.0):
        self._candidates = list(candidates)
        self.zero_source_objective = zero_source_objective
        self.solves = {"forward": 0, "adjoint": 0, "candidate_traces": 0}
        self.screen_radius = 40.0

    def model_state(self, control):
        return None

    def best_candidate(self, control, model_state=None):
        if not self._candidates:
            return None
        item = self._candidates.pop(0)
        if item is None:
            return None
        return dict(item)


def _candidate(predicted_decrease, center=(10.0, 20.0), a_1d=0.5):
    return {
        "node": 0,
        "center": [float(center[0]), float(center[1])],
        "g": -1.0,
        "kappa": 1.0,
        "a_unconstrained": float(a_1d),
        "a_1d": float(a_1d),
        "delta_j": -float(predicted_decrease),
        "predicted_decrease": float(predicted_decrease),
    }


def _run_with_stub(monkey_screen, polish_fn, **kwargs):
    """Run infer_source_count with _BirthScreen replaced by a stub."""
    original = sci._BirthScreen
    sci._BirthScreen = lambda *a, **k: monkey_screen
    try:
        return sci.infer_source_count(
            None, {"sigma2": 400.0, "dt": 0.1, "jprint": 1}, None, None,
            amplitude_upper=2.0, eta1=1.0, lam=0.0,
            objective_time_rule="exclude_first",
            polish_fn=polish_fn,
            print_fn=lambda *_a, **_k: None,
            **kwargs,
        )
    finally:
        sci._BirthScreen = original


def test_stop_reasons():
    section("A3. every stop reason is reachable and correctly named")

    # -- max_sources ------------------------------------------------------
    def good_polish(control, amp_lower):
        m = sci.amplitudes_of(control).size
        return control, 1000.0 / (10.0 ** m), {"stub": True}

    summary = _run_with_stub(
        _StubScreen([_candidate(500.0), _candidate(50.0), _candidate(5.0)]),
        good_polish, max_sources=2, rel_reduction_tol=1.0e-3,
    )
    check(
        summary["stop_reason"] == sci.STOP_MAX_SOURCES
        and summary["inferred_num_sources"] == 2,
        f"max_sources_reached at the cap (got {summary['stop_reason']}, "
        f"m={summary['inferred_num_sources']})",
    )

    # -- discrepancy floor -------------------------------------------------
    summary = _run_with_stub(
        _StubScreen([_candidate(500.0)], zero_source_objective=1.0e-15),
        good_polish, max_sources=4, discrepancy_tol=1.0e-12,
    )
    check(
        summary["stop_reason"] == sci.STOP_DISCREPANCY
        and summary["inferred_num_sources"] == 0,
        f"discrepancy_floor fires before any birth (got {summary['stop_reason']})",
    )

    # -- no admissible candidate ------------------------------------------
    summary = _run_with_stub(
        _StubScreen([None]), good_polish, max_sources=4
    )
    check(
        summary["stop_reason"] == sci.STOP_NO_CANDIDATE,
        f"no_admissible_candidate when the screen returns None "
        f"(got {summary['stop_reason']})",
    )

    # -- negligible predicted reduction (no polish attempted) --------------
    calls = []

    def counting_polish(control, amp_lower):
        calls.append(control)
        return control, 1.0, {}

    summary = _run_with_stub(
        _StubScreen([_candidate(1.0e-9)], zero_source_objective=1000.0),
        counting_polish, max_sources=4, rel_reduction_tol=1.0e-3,
    )
    check(
        summary["stop_reason"] == sci.STOP_NEGLIGIBLE_PREDICTED,
        f"negligible_predicted_reduction below threshold "
        f"(got {summary['stop_reason']})",
    )
    check(
        not calls,
        "no polish is attempted when the predicted birth is already negligible",
    )

    # -- insufficient ACTUAL reduction (the nested-model trap) -------------
    def barely_better_polish(control, amp_lower):
        # A real decrease, but far below the threshold: exactly the nested-model
        # situation where J_(m+1) <= J_m is a theorem, not evidence.
        return control, 1000.0 - 1.0e-6, {}

    summary = _run_with_stub(
        _StubScreen([_candidate(500.0)], zero_source_objective=1000.0),
        barely_better_polish, max_sources=4, rel_reduction_tol=1.0e-3,
    )
    check(
        summary["stop_reason"] == sci.STOP_INSUFFICIENT_ACTUAL
        and summary["inferred_num_sources"] == 0,
        "a real-but-tiny decrease is REJECTED (decrease alone is not evidence) "
        f"(got {summary['stop_reason']})",
    )
    level = summary["levels"][-1]
    check(
        level["actual_decrease"] > 0.0 and level["accepted"] is False,
        f"the rejected level records a positive decrease "
        f"({level['actual_decrease']:.3e}) and accepted=False",
    )

    # -- amplitude collapse -------------------------------------------------
    def collapsing_polish(control, amp_lower):
        out = np.asarray(control, dtype=np.float64).copy()
        out[-1] = 0.0  # the new source collapses to zero amplitude
        return out, 1.0, {}

    summary = _run_with_stub(
        _StubScreen([_candidate(500.0)], zero_source_objective=1000.0),
        collapsing_polish, max_sources=4, rel_reduction_tol=1.0e-3, amp_tol=1.0e-3,
    )
    check(
        summary["stop_reason"] == sci.STOP_AMPLITUDE_COLLAPSE
        and summary["inferred_num_sources"] == 0,
        f"rejected_amplitude_collapse when a source goes to zero "
        f"(got {summary['stop_reason']})",
    )
    level = summary["levels"][-1]
    check(
        level["min_amplitude_after"] == 0.0,
        "the collapse level records the zero minimum amplitude",
    )
    check(
        level["actual_decrease"] > level["acceptance_threshold"],
        "the collapsed model still cleared the OBJECTIVE threshold -- only the "
        "amplitude test caught it (this is the whole point of the zero bound)",
    )


def test_summary_shape():
    section("A4. persisted summary carries every required field")

    def good_polish(control, amp_lower):
        m = sci.amplitudes_of(control).size
        return control, 1000.0 / (10.0 ** m), {"amplitude_lower_bound": amp_lower}

    summary = _run_with_stub(
        _StubScreen([_candidate(500.0), _candidate(50.0)]),
        good_polish, max_sources=1, production_amplitude_floor=0.1,
    )
    for field in (
        "inferred_num_sources", "stop_reason", "final_control", "final_objective",
        "zero_source_objective", "tolerances", "nesting", "levels", "solves",
        "wall_time_seconds", "screen", "max_sources",
    ):
        check(field in summary, f"summary has '{field}'")
    level = summary["levels"][0]
    for field in (
        "level", "J_before", "candidate", "predicted_decrease",
        "J_after_candidate", "actual_decrease", "accepted", "reason",
        "acceptance_threshold", "amplitudes_after", "min_amplitude_after",
    ):
        check(field in level, f"level record has '{field}'")
    check(
        summary["nesting"]["amplitude_lower_bound_during_inference"] == 0.0,
        "summary records the zero inference amplitude bound",
    )
    check(
        summary["nesting"]["production_amplitude_floor"] == 0.1,
        "summary records the production floor as provenance only",
    )
    json.dumps(summary)  # must be serializable
    check(True, "summary is JSON-serializable")


# ---------------------------------------------------------------------------
# B. nesting / zero-amplitude behaviour
# ---------------------------------------------------------------------------
def test_polish_receives_zero_lower_bound():
    section("B1. inference always polishes with amplitude lower bound 0")
    seen = []

    def recording_polish(control, amp_lower):
        seen.append(float(amp_lower))
        m = sci.amplitudes_of(control).size
        return control, 1000.0 / (10.0 ** m), {}

    _run_with_stub(
        _StubScreen([_candidate(500.0), _candidate(50.0)]),
        recording_polish, max_sources=2, production_amplitude_floor=0.1,
    )
    check(len(seen) == 2, f"polish was called for each level (got {len(seen)})")
    check(
        all(v == 0.0 for v in seen),
        f"every polish received amplitude lower bound 0 (got {seen})",
    )


def test_floor_would_manufacture_evidence():
    section("B2. the production floor would MANUFACTURE a spurious source")

    # One spurious candidate. Under a zero lower bound the polish drives its
    # amplitude to 0; under a production floor of 0.1 it pins at the floor.
    def polish_with_bound(bound):
        def polish(control, _amp_lower_ignored):
            out = np.asarray(control, dtype=np.float64).copy()
            out[-1] = max(0.0, bound)   # collapses to the bound, whatever it is
            return out, 1.0, {}
        return polish

    zero_bound = _run_with_stub(
        _StubScreen([_candidate(500.0)], zero_source_objective=1000.0),
        polish_with_bound(0.0), max_sources=4, amp_tol=1.0e-3,
    )
    check(
        zero_bound["stop_reason"] == sci.STOP_AMPLITUDE_COLLAPSE
        and zero_bound["inferred_num_sources"] == 0,
        "with lower bound 0 the spurious source collapses and is REJECTED",
    )

    floored = _run_with_stub(
        _StubScreen([_candidate(500.0)], zero_source_objective=1000.0),
        polish_with_bound(0.1), max_sources=1, amp_tol=1.0e-3,
    )
    check(
        floored["inferred_num_sources"] == 1,
        "with a 0.1 floor the SAME spurious source is wrongly ACCEPTED "
        "(the floor, not the data, supplied the amplitude)",
    )
    check(
        zero_bound["inferred_num_sources"] != floored["inferred_num_sources"],
        "the two bounds give different answers -- which is why inference must "
        "not use the production amplitude floor",
    )


def test_driver_bounds_are_zero_floored():
    section("B3. the driver's polish callback builds zero amplitude bounds")
    import run_optimization as app1
    from multi_source_utils import mesh_domain_active_bounds_for_mode

    bounds = mesh_domain_active_bounds_for_mode(
        "location_and_intensity", (0.0, 300.0, 0.0, 700.0), num_sources=3
    )
    zeroed = app1.apply_amplitude_floor(
        bounds, "location_and_intensity", 0.0, num_sources=3
    )
    amp_lowers = [zeroed[3 * s + 2][0] for s in range(3)]
    check(
        all(v == 0.0 for v in amp_lowers),
        f"amplitude lower bounds are all 0 (got {amp_lowers})",
    )
    xy_unchanged = all(
        zeroed[3 * s + c] == bounds[3 * s + c] for s in range(3) for c in (0, 1)
    )
    check(xy_unchanged, "position bounds are untouched by the amplitude floor")

    production = app1.apply_amplitude_floor(
        bounds, "location_and_intensity", 0.1, num_sources=3
    )
    check(
        all(production[3 * s + 2][0] == 0.1 for s in range(3)),
        "the production floor path still applies 0.1 (unchanged behaviour)",
    )


# ---------------------------------------------------------------------------
# C. c06 regression
# ---------------------------------------------------------------------------
def test_c06_regression():
    section("C. c06 regression: infers m = 2 and does not accept a third source")
    path = (
        APP_DIR / "output" / C06_CASE / "full" / "source_count_inference_summary.json"
    )
    if not path.exists():
        print(f"  [SKIP] no inference summary at {path}")
        print("         run the c06 inference case first (see the README)")
        return
    summary = json.loads(path.read_text(encoding="utf-8"))

    inferred = int(summary["inferred_num_sources"])
    check(inferred == 2, f"inferred m = 2 (got {inferred})")
    check(
        summary["inferred_matches_true_count"] is True,
        "inferred count matches the supplied true control's count",
    )

    accepted = [L for L in summary["levels"] if L.get("accepted")]
    rejected = [L for L in summary["levels"] if not L.get("accepted")]
    check(len(accepted) == 2, f"exactly two levels accepted (got {len(accepted)})")
    check(
        all(L["level"] <= 2 for L in accepted),
        "the accepted levels are levels 1 and 2",
    )

    third = [L for L in summary["levels"] if L["level"] == 3]
    if third:
        record = third[0]
        check(
            not record.get("accepted"),
            f"a third source was TESTED and REJECTED "
            f"(reason: {record.get('reason')})",
        )
        print(
            f"         level 3: predicted_decrease="
            f"{record.get('predicted_decrease')!r}  "
            f"actual_decrease={record.get('actual_decrease')!r}  "
            f"min_amplitude_after={record.get('min_amplitude_after')!r}"
        )
    else:
        check(
            summary["stop_reason"] == sci.STOP_DISCREPANCY,
            "no third level was tested, and the stop reason is the discrepancy "
            f"floor (got {summary['stop_reason']})",
        )

    check(
        summary["stop_reason"] != sci.STOP_MAX_SOURCES,
        f"inference stopped on a decision, not on --max-sources "
        f"(got {summary['stop_reason']})",
    )
    check(
        int(summary["max_sources"]) >= 4,
        f"the run had headroom for more sources (max_sources="
        f"{summary['max_sources']})",
    )
    check(
        summary["nesting"]["amplitude_lower_bound_during_inference"] == 0.0,
        "the run used the zero amplitude lower bound during inference",
    )

    match = summary.get("recovery_permutation_matched")
    if match:
        print(
            f"         recovery: max position error "
            f"{match['max_position_error_m']:.6e} m, max amplitude error "
            f"{match['max_amplitude_error']:.6e}"
        )
        check(
            match["max_position_error_m"] < 5.0,
            f"inferred centers are within 5 m of the truth "
            f"({match['max_position_error_m']:.4e} m)",
        )
    check(
        summary.get("observation", {}).get("observation_mode") == "nodes",
        "sensor metadata is recorded in the inference summary",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit-only", action="store_true")
    args = parser.parse_args()

    test_acceptance_threshold()
    test_helpers()
    test_stop_reasons()
    test_summary_shape()
    test_polish_receives_zero_lower_bound()
    test_floor_would_manufacture_evidence()
    test_driver_bounds_are_zero_floored()
    if not args.unit_only:
        test_c06_regression()

    section("RESULT")
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED:")
        for message in FAILURES:
            print(f"  - {message}")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
