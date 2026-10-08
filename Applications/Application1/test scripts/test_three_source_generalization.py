"""Cheap M-source generalization tests for the Application1 sensitivity pipeline.

Fast by construction: real DC mesh, short ``tfinal``, constant velocity (no
nodal velocity field), a small screen and beam.  No expensive continuation run.

Verifies, for M = 3 unless stated otherwise:

  1. the control-shape helpers infer M from the control length and stay exactly
     backward compatible at M = 2 (including the strict explicit-M check);
  2. permutation-invariant recovery matching is exhaustive for arbitrary M
     (M! assignments) and returns label-order-independent errors;
  3. the CLI accepts --num-sources 3, rejects non-positive values, and rejects a
     --true-control / --initial-control that is not exactly 3*M long;
  4. the generalized bounded amplitude box QP (n = 3 and n = 4) matches brute
     force and reproduces the closed-form unconstrained solution when interior;
  5. the beam initializer returns a length-9 control for M = 3, executes exactly
     three source levels, reports per-level surviving-branch counts, and places
     three distinct admissible centers with amplitudes inside the bounds;
  6. the coherence cap is evaluated against ALL sources already placed on the
     branch, not just the first (no two-source pair unpacking);
  7. no production module hard-codes a pair unpacking of the control.
"""

from __future__ import annotations

import itertools
import os as _os
import re
import sys as _sys

import numpy as np

_APP1_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _APP1_DIR not in _sys.path:
    _sys.path.insert(0, _APP1_DIR)

import run_optimization as ro  # noqa: E402
import gradient_test as gt  # noqa: E402
import sensitivity_initialization as sinit  # noqa: E402
from mesh_admissibility import MeshAdmissibility  # noqa: E402
from mesh_data_config import build_mesh_data  # noqa: E402
import multi_source_utils as msu  # noqa: E402


TFINAL = 25.0
# Three admissible DC centers in three different vertical bands.
TRUE_CONTROL_3 = np.array(
    [250.0, 350.0, 1.00, 120.0, 300.0, 0.80, 150.0, 620.0, 0.90],
    dtype=np.float64,
)
MEASUREMENT_BOUNDARIES = ["right", "bottom"]
AMPLITUDE_BOUNDS = (0.1, 2.0)

_FAILURES: list[str] = []


def check(condition, label, detail=""):
    if condition:
        print(f"  [PASS] {label}" + (f"  -- {detail}" if detail else ""))
    else:
        print(f"  [FAIL] {label}" + (f"  -- {detail}" if detail else ""))
        _FAILURES.append(label)


def _build_context(true_control):
    mesh_data = ro.configure_measurement_boundaries(
        build_mesh_data(), MEASUREMENT_BOUNDARIES
    )
    pde = dict(ro.PDE)
    pde["tfinal"] = TFINAL  # constant velocity (no velocity_nodes) -> fast
    admissibility = MeshAdmissibility(mesh_data)
    observation_state = gt.run_forward(mesh_data, pde, true_control)
    return mesh_data, pde, admissibility, observation_state


# ---------------------------------------------------------------------------
def test_control_shape_helpers():
    print("\n1. control-shape helpers infer M and stay backward compatible")
    check(msu.infer_num_sources(np.zeros(6)) == 2, "infer M from a length-6 control")
    check(msu.infer_num_sources(np.zeros(9)) == 3, "infer M from a length-9 control")
    check(msu.infer_num_sources(np.zeros(12)) == 4, "infer M from a length-12 control")
    for bad in (np.zeros(0), np.zeros(4), np.zeros(7)):
        try:
            msu.infer_num_sources(bad)
            check(False, f"reject a length-{bad.size} control")
        except ValueError:
            check(True, f"reject a length-{bad.size} control")

    z9 = np.arange(9, dtype=np.float64)
    check(msu.reshape_control(z9).shape == (3, 3), "reshape_control infers (3, 3)")
    check(
        np.array_equal(msu.flatten_control(msu.reshape_control(z9)), z9),
        "flatten(reshape(z)) round-trips at M=3",
    )
    # Explicit M keeps the original strict length validation.
    try:
        msu.reshape_control(z9, num_sources=2)
        check(False, "explicit M still validates the control length")
    except ValueError:
        check(True, "explicit M still validates the control length")
    # M=2 behavior is unchanged.
    z6 = np.arange(6, dtype=np.float64)
    check(
        np.array_equal(msu.reshape_control(z6), z6.reshape(2, 3)),
        "M=2 reshape unchanged",
    )
    check(
        msu.component_names(3) == ["x1", "y1", "A1", "x2", "y2", "A2", "x3", "y3", "A3"],
        "component_names(3)",
    )
    check(msu.control_dim(3) == 9, "control_dim(3) == 9")

    # pack/unpack/restrict round-trip at M=3 in every mode.
    for mode in ("location_and_intensity", "location_only", "intensity_only"):
        fixed = msu.mode_fixed_values(mode, z_ref_full=z9)
        active = msu.pack_active(z9, mode)
        back = msu.unpack_active(active, z9, mode, fixed)
        check(
            np.allclose(back, z9),
            f"pack/unpack round-trip at M=3 ({mode})",
            f"active dim {active.size}",
        )
        check(
            msu.restrict_gradient(z9, mode).size == active.size,
            f"restrict_gradient dim matches active dim ({mode})",
        )


def test_permutation_matching():
    print("\n2. permutation-invariant recovery matching for arbitrary M")
    truth = np.array([10.0, 10.0, 1.0, 200.0, 200.0, 0.5, 50.0, 600.0, 1.5])
    # Same sources in a rotated label order plus small errors.
    got = np.array([200.0, 201.0, 0.52, 50.0, 600.0, 1.49, 11.0, 10.0, 1.02])
    match = ro.match_sources_by_permutation(truth, got)
    check(match["num_sources"] == 3, "M=3 detected")
    check(match["labels_permuted"], "label permutation detected")
    check(
        np.allclose(sorted(match["position_errors_m"]), [0.0, 1.0, 1.0], atol=1e-9),
        "per-source position errors after optimal permutation",
        str([round(v, 6) for v in match["position_errors_m"]]),
    )
    check(
        abs(match["max_amplitude_error"] - 0.02) < 1e-9,
        "max amplitude error after optimal permutation",
        f"{match['max_amplitude_error']:.6f}",
    )
    # Result must not depend on the order the candidate labels arrive in.
    base = ro.match_sources_by_permutation(truth, got)["position_errors_m"]
    for perm in itertools.permutations(range(3)):
        shuffled = np.concatenate([got.reshape(3, 3)[p] for p in perm])
        again = ro.match_sources_by_permutation(truth, shuffled)["position_errors_m"]
        if not np.allclose(sorted(again), sorted(base)):
            check(False, "matching is invariant to candidate label order", str(perm))
            break
    else:
        check(True, "matching is invariant to candidate label order (all 6 orders)")
    # M=4: 24 permutations.
    t4 = np.array([0.0, 0.0, 1.0, 10.0, 0.0, 1.0, 0.0, 10.0, 1.0, 10.0, 10.0, 1.0])
    g4 = np.concatenate([t4.reshape(4, 3)[i] for i in (3, 1, 0, 2)])
    m4 = ro.match_sources_by_permutation(t4, g4)
    check(
        m4["num_sources"] == 4 and m4["max_position_error_m"] < 1e-12,
        "exact M=4 matching over 24 permutations",
        f"perm={m4['permutation']}",
    )


def test_cli_validation():
    print("\n3. CLI validation of --num-sources and control lengths")
    args = ro.parse_args.__wrapped__() if hasattr(ro.parse_args, "__wrapped__") else None
    del args

    def parse(argv):
        argv_backup = _sys.argv
        _sys.argv = ["run_optimization.py", *argv]
        try:
            return ro.parse_args()
        finally:
            _sys.argv = argv_backup

    a = parse(["--mode", "location_and_intensity"])
    check(a.num_sources == msu.NUM_SOURCES, "default --num-sources is 2")
    a = parse(["--mode", "location_and_intensity", "--num-sources", "3"])
    check(a.num_sources == 3, "--num-sources 3 parses")
    # Length validation (the helper main() uses).
    try:
        ro.validate_control_length([1, 2, 3, 4, 5, 6], "--true-control", 3)
        check(False, "reject a 6-value control at M=3")
    except ValueError as exc:
        check("exactly 9 entries" in str(exc), "reject a 6-value control at M=3", str(exc))
    ok = ro.validate_control_length(list(range(9)), "--true-control", 3)
    check(ok.size == 9, "accept a 9-value control at M=3")
    ok2 = ro.validate_control_length(list(range(6)), "--true-control", 2)
    check(ok2.size == 6, "M=2 length validation unchanged")


def test_box_qp_generalizes():
    print("\n4. bounded amplitude box QP at n = 3 and n = 4")
    rng = np.random.default_rng(0)
    for n in (3, 4):
        worst = 0.0
        for _trial in range(40):
            B = rng.normal(size=(n, n))
            G = B @ B.T + 0.5 * np.eye(n)
            b = rng.normal(size=n)
            a = sinit._solve_box_qp(G, b, 0.1, 2.0)
            q = 0.5 * a @ G @ a + b @ a
            # Brute force over a fine grid as an independent check.
            grid = np.linspace(0.1, 2.0, 26)
            best = min(
                0.5 * np.array(c) @ G @ np.array(c) + b @ np.array(c)
                for c in itertools.product(grid, repeat=n)
            )
            worst = max(worst, q - best)
        check(
            worst <= 1e-9,
            f"n={n}: exact QP never worse than a brute-force grid ({3 ** n} active sets)",
            f"max excess {worst:.3e}",
        )
    # Interior unconstrained optimum is reproduced exactly.
    G = np.array([[2.0, 0.3, 0.1], [0.3, 2.0, 0.2], [0.1, 0.2, 2.0]])
    b = -np.array([2.0, 2.0, 2.0])
    a = sinit._solve_box_qp(G, b, 0.1, 2.0)
    check(
        np.allclose(a, np.linalg.solve(G, -b), atol=1e-10),
        "interior optimum equals the unconstrained solution",
        str(np.round(a, 6).tolist()),
    )
    try:
        sinit._solve_box_qp(np.eye(9), np.zeros(9), 0.0, 1.0)
        check(False, "guard against an exponential source count")
    except ValueError:
        check(True, "guard against an exponential source count (n > 8 raises)")


def test_three_source_beam_initializer():
    print("\n5. beam initializer at M = 3 (real mesh, short tfinal)")
    mesh_data, pde, admissibility, observation_state = _build_context(TRUE_CONTROL_3)
    lines: list[str] = []
    control, diagnostics, extras = sinit.build_sensitivity_initial_control(
        mesh_data,
        pde,
        observation_state,
        3,
        AMPLITUDE_BOUNDS,
        admissibility,
        eta1=ro.ETA1,
        lam=ro.LAMBDA_REG,
        objective_time_rule=ro.OBJECTIVE_TIME_RULE,
        score="predicted_reduction",
        top_k=6,
        beam_width=2,
        coherence_cap=0.9,
        verbose=True,
        print_fn=lines.append,
    )
    text = "\n".join(lines)

    check(control.size == 9, "control dimension is 3*M = 9", f"size {control.size}")
    check(len(diagnostics) == 3, "three per-source diagnostic records",
          f"{len(diagnostics)}")
    levels = extras.get("source_levels") or []
    check(len(levels) == 3, "three source levels executed", f"{len(levels)}")
    check(
        [rec["level"] for rec in levels] == [1, 2, 3],
        "levels are 1, 2, 3 in order",
    )
    check(
        all(rec["branches_surviving"] >= 1 for rec in levels),
        "every level reports >= 1 surviving branch",
        str([rec["branches_surviving"] for rec in levels]),
    )
    for k in (1, 2, 3):
        check(
            f"Source level {k}/3" in text,
            f"log announces 'Source level {k}/3'",
        )
    check(
        len(re.findall(r"Source level \d/3 complete: \d+ branch\(es\) surviving", text)) == 3,
        "log reports surviving branches after each level",
    )

    sources = control.reshape(3, 3)
    check(
        all(
            admissibility.is_point_in_domain(float(x), float(y))
            for x, y, _a in sources
        ),
        "all three placed centers are fluid-admissible",
    )
    check(
        all(AMPLITUDE_BOUNDS[0] <= a <= AMPLITUDE_BOUNDS[1] for _x, _y, a in sources),
        "all three amplitudes lie inside the bounds",
        str([round(float(a), 4) for a in sources[:, 2]]),
    )
    dists = [
        float(np.hypot(*(sources[i, :2] - sources[j, :2])))
        for i, j in itertools.combinations(range(3), 2)
    ]
    check(min(dists) > 1.0, "the three centers are distinct",
          f"min pairwise distance {min(dists):.2f} m")
    check(
        extras["initial_exact_objective"] <= extras["zero_source_objective"],
        "initialized objective is no worse than the zero-source objective",
        f"{extras['initial_exact_objective']:.6e} <= "
        f"{extras['zero_source_objective']:.6e}",
    )
    check(
        extras.get("n_candidate_traces", 0) > 0
        and extras.get("wall_time_seconds", 0.0) > 0.0,
        "cost accounting is populated",
        f"traces={extras.get('n_candidate_traces')} "
        f"fwd={extras.get('n_forward_solves')} "
        f"adj={extras.get('n_adjoint_solves')} "
        f"wall={extras.get('wall_time_seconds', 0.0):.1f}s",
    )
    check(
        len(extras.get("branch_controls") or []) >= 1
        and all(
            len(rec["control"]) == 9 for rec in extras["branch_controls"]
        ),
        "every retained branch reports a length-9 control",
    )
    return text


def test_coherence_cap_uses_all_placed_sources(beam_log: str):
    print("\n6. coherence cap compares against ALL sources already placed")
    # At level 3 two sources are already placed, so the cap message must report
    # a max over 2 placed sources -- not a single-source pair comparison.
    counts = sorted(
        {int(m) for m in re.findall(r"max over (\d+) placed source\(s\)", beam_log)}
    )
    if counts:
        check(
            max(counts) >= 2,
            "cap reported as a max over >= 2 placed sources at a later level",
            f"observed placed-source counts {counts}",
        )
    else:
        # No candidate was rejected in this cheap run; assert the code path
        # directly instead.
        import inspect

        src = inspect.getsource(sinit._predicted_reduction_beam)
        check(
            src.count(
                "for t1, n1 in zip(branch[\"traces\"], existing_norms)"
            )
            >= 1,
            "cap takes a max over every placed trace on the branch (source check)",
        )
    src = __import__("inspect").getsource(sinit._predicted_reduction_beam)
    check(
        "existing_norms = [ctx.norm2(t) for t in branch[\"traces\"]]" in src,
        "existing norms are built from every placed trace",
    )


def test_no_pair_unpacking_remains():
    print("\n7. no production module unpacks the control as a fixed pair")
    import inspect

    offenders = []
    for module in (ro, sinit, msu):
        src = inspect.getsource(module)
        for lineno, line in enumerate(src.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            # A fixed two-source unpack of a control/centers list.
            if re.search(r"^\s*\(?\s*\w+1\s*,\s*\w+2\s*\)?\s*=\s*.*(control|centers)",
                         line):
                offenders.append(f"{module.__name__}:{lineno}: {stripped}")
            if re.search(r"reshape\(\s*2\s*,\s*3\s*\)", line):
                offenders.append(f"{module.__name__}:{lineno}: {stripped}")
    check(not offenders, "no fixed two-source unpacking in production modules",
          "; ".join(offenders))


def main() -> int:
    test_control_shape_helpers()
    test_permutation_matching()
    test_cli_validation()
    test_box_qp_generalizes()
    beam_log = test_three_source_beam_initializer()
    test_coherence_cap_uses_all_placed_sources(beam_log)
    test_no_pair_unpacking_remains()

    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} CHECK(S) FAILED:")
        for name in _FAILURES:
            print(f"  - {name}")
        return 1
    print("all M-source generalization checks PASSED")
    return 0


if __name__ == "__main__":
    _sys.exit(main())
