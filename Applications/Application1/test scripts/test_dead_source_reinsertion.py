"""Deterministic unit tests for the dead-source sensitivity reinsertion logic.

These exercise the PURE decision helpers added to ``run_optimization.py`` for the
adaptive dead-source reinsertion feature, with NO PDE solves (fast, deterministic):

  * dead-source / partial-recovery detection (amplitude-at-floor is decisive);
  * permutation-symmetric basin de-duplication;
  * parent selection skips consumed basins (no repeated basin, no infinite loop);
  * candidate selection excludes outer-boundary nodes, respects minimum
    separation from survivors, excludes near previously used insertion
    locations, and refuses non-negative sensitivity;
  * disabled-feature backward compatibility (defaults keep the feature OFF).

Run from the repository root with the project environment:

  python3 \
      "Applications/Application1/test_dead_source_reinsertion.py"
"""

import os as _os
import sys as _sys

import numpy as np

# This test/verification script lives in Application1/"test scripts"/; put the
# parent Application1 directory on sys.path so its local modules import cleanly.
_APP1_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _APP1_DIR not in _sys.path:
    _sys.path.insert(0, _APP1_DIR)

# Importing run_optimization appends Src/Objective to sys.path (shared
# gradient_test) and Application1 (mesh modules); it does not run argparse.
import run_optimization as ro


AMP_BOUNDS = (0.1, 2.0)  # floor 0.1 as in the right+bottom experiments


def test_detect_partial_recovery():
    floor = AMP_BOUNDS[0]
    # One healthy survivor + one dead source at the floor -> partial recovery.
    partial = [150.0, 600.0, 1.2, 302.0, 0.0, 0.1]
    det = ro.detect_partial_recovery(partial, AMP_BOUNDS)
    assert det is not None, "should detect a partial recovery"
    assert det["dead_indices"] == [1], det["dead_indices"]
    assert det["survivor_indices"] == [0], det["survivor_indices"]
    assert abs(det["dead_amplitudes"][0] - 0.1) < 1e-12

    # Dead just above the floor but within the margin still counts as dead.
    near = [150.0, 600.0, 1.2, 50.0, 250.0, floor + ro.DEAD_SOURCE_AMPLITUDE_MARGIN - 1e-6]
    assert ro.detect_partial_recovery(near, AMP_BOUNDS) is not None

    # Both healthy -> not a partial recovery.
    both_ok = [150.0, 600.0, 1.2, 50.0, 250.0, 0.7]
    assert ro.detect_partial_recovery(both_ok, AMP_BOUNDS) is None

    # Both dead -> no survivor to keep -> not a reinsertion opportunity.
    both_dead = [150.0, 600.0, 0.1, 50.0, 250.0, 0.1]
    assert ro.detect_partial_recovery(both_dead, AMP_BOUNDS) is None

    # Amplitude comfortably above the margin -> survivor, not dead.
    healthy_second = [150.0, 600.0, 1.2, 50.0, 250.0, floor + ro.DEAD_SOURCE_AMPLITUDE_MARGIN + 0.2]
    assert ro.detect_partial_recovery(healthy_second, AMP_BOUNDS) is None
    print("detect_partial_recovery: PASS")


def test_bound_pinned_secondary_signal():
    class _BBox:
        bounding_box = (0.0, 301.98544, 0.0, 700.46467)

    # Dead source parked at the (xmax, ymin) corner should be flagged pinned,
    # but the pin is only a secondary diagnostic; amplitude is decisive.
    partial = [150.0, 600.0, 1.2, 301.98544, 0.0, 0.1]
    det = ro.detect_partial_recovery(partial, AMP_BOUNDS, admissibility=_BBox())
    assert det is not None
    assert det["dead_center_bound_pinned"] == [True], det["dead_center_bound_pinned"]

    # An interior dead source is still detected (amplitude), just not pinned.
    interior = [150.0, 600.0, 1.2, 150.0, 300.0, 0.1]
    det2 = ro.detect_partial_recovery(interior, AMP_BOUNDS, admissibility=_BBox())
    assert det2 is not None and det2["dead_center_bound_pinned"] == [False]
    print("bound_pinned secondary signal: PASS")


def test_basin_key_permutation_symmetry():
    # Same survivor centers, different source ordering -> same basin key.
    a = [150.0, 600.0, 1.2, 302.0, 0.0, 0.1]     # survivor is source 0
    b = [302.0, 0.0, 0.1, 150.0, 600.0, 1.2]     # survivor is source 1 (swapped)
    ka = ro.detect_partial_recovery(a, AMP_BOUNDS)["basin_key"]
    kb = ro.detect_partial_recovery(b, AMP_BOUNDS)["basin_key"]
    assert ka == kb, (ka, kb)

    # A different survivor center -> different basin key.
    c = [151.6, 601.0, 1.2, 161.7, 548.8, 0.1]
    kc = ro.detect_partial_recovery(c, AMP_BOUNDS)["basin_key"]
    assert kc != ka
    print("basin_key permutation symmetry: PASS")


def test_parent_selection_and_basin_dedup():
    class _BBox:
        bounding_box = (0.0, 302.0, 0.0, 700.0)

    records = [
        {"start_index": 1, "final_objective": 6.9e5,
         "final_control": [150.0, 600.0, 1.2, 302.0, 0.0, 0.1]},
        {"start_index": 2, "final_objective": 6.8e5,       # better survivor basin
         "final_control": [151.0, 601.0, 1.2, 25.0, 664.0, 0.1]},
        {"start_index": 3, "final_objective": 1.0e-3,      # excellent, both alive
         "final_control": [150.0, 600.0, 1.2, 50.0, 250.0, 0.7]},
    ]
    consumed: list = []

    # Lowest-objective unconsumed partial recovery is start 2.
    sel = ro._select_reinsertion_parent(records, AMP_BOUNDS, _BBox(), consumed)
    assert sel is not None and sel[1]["start_index"] == 2, sel
    consumed.append(sel[2]["basin_key"])

    # After consuming start 2's basin, start 1 is next.
    sel = ro._select_reinsertion_parent(records, AMP_BOUNDS, _BBox(), consumed)
    assert sel is not None and sel[1]["start_index"] == 1, sel
    consumed.append(sel[2]["basin_key"])

    # All partial-recovery basins consumed -> None (no infinite loop).
    sel = ro._select_reinsertion_parent(records, AMP_BOUNDS, _BBox(), consumed)
    assert sel is None, sel

    # Simulate the loop's reinsertion counter terminating within #distinct basins.
    consumed = []
    iterations = 0
    while ro._select_reinsertion_parent(records, AMP_BOUNDS, _BBox(), consumed) is not None:
        _o, _r, det = ro._select_reinsertion_parent(records, AMP_BOUNDS, _BBox(), consumed)
        consumed.append(det["basin_key"])
        iterations += 1
        assert iterations <= 5, "reinsertion selection must terminate"
    assert iterations == 2, iterations  # exactly the two distinct partial basins
    print("parent selection + basin dedup + termination: PASS")


def test_select_reinsertion_candidate():
    # Tiny synthetic vertex set (not a real mesh): 5 candidate points.
    coord = np.array([
        [287.5, 339.2],   # 0: most negative, admissible
        [150.0, 300.0],   # 1: moderately negative, admissible
        [151.0, 601.0],   # 2: near a survivor (should be excluded by separation)
        [10.0, 10.0],     # 3: an outer-boundary node (excluded)
        [100.0, 620.0],   # 4: positive sensitivity (excluded: not descent)
    ], dtype=float)
    sensitivity = np.array([-6.5e6, -1.5e5, -9.9e6, -5.0e6, +20.0])
    survivor_centers = [np.array([150.0, 600.0])]  # source at (150,600)
    boundary_nodes = {3}
    min_sep = 40.0

    # No used locations yet -> pick vertex 0 (most negative admissible).
    sel = ro.select_reinsertion_candidate(
        coord, sensitivity, survivor_centers, boundary_nodes, [], min_sep
    )
    assert sel is not None and sel[0] == 0, sel

    # Exclude near a previously used location at vertex 0 -> fall back to vertex 1.
    used = [np.array([287.5, 339.2])]
    sel = ro.select_reinsertion_candidate(
        coord, sensitivity, survivor_centers, boundary_nodes, used, min_sep
    )
    assert sel is not None and sel[0] == 1, sel

    # Vertex 2 (near survivor) and 3 (boundary) and 4 (positive) are never chosen.
    assert sel[0] not in (2, 3, 4)

    # If every admissible candidate is non-negative, return None.
    all_positive = np.array([1.0, 2.0, -9.9e6, -5.0e6, 3.0])  # negatives are excluded ones
    sel = ro.select_reinsertion_candidate(
        coord, all_positive, survivor_centers, boundary_nodes, [], min_sep
    )
    assert sel is None, sel
    print("select_reinsertion_candidate: PASS")


def test_disabled_feature_defaults():
    # Parse an adaptive command WITHOUT the reinsertion flag: feature stays OFF,
    # budget default is 2, and no other adaptive default changes.
    import sys
    argv = [
        "run_optimization.py", "--mode", "location_and_intensity",
        "--init-mode", "adaptive", "--measurement-boundaries", "right", "bottom",
    ]
    saved = sys.argv
    try:
        sys.argv = argv
        args = ro.parse_args()
    finally:
        sys.argv = saved
    assert args.adaptive_reinsert_dead_sources is False, "feature must default OFF"
    assert args.adaptive_max_reinsertions == ro.ADAPTIVE_DEFAULT_MAX_REINSERTIONS == 2
    assert ro.REINSERTION_AMPLITUDE == 0.8
    print("disabled-feature defaults: PASS")


def run():
    test_detect_partial_recovery()
    test_bound_pinned_secondary_signal()
    test_basin_key_permutation_symmetry()
    test_parent_selection_and_basin_dedup()
    test_select_reinsertion_candidate()
    test_disabled_feature_defaults()
    print("\ndead-source reinsertion logic: ALL PASS")


if __name__ == "__main__":
    run()
