"""Focused regression for second-source re-selection against a stale residual.

The last source is screened against the residual of the placement as it stands
BEFORE joint pair refinement.  When refinement then moves the leading source, that
residual -- and the candidate it selected -- is stale.  This test checks the
re-selection stage repairs that, using only the observation data and the exact
variable-projected objective: the true source COORDINATES are never passed to the
initializer (no ``true_centers_for_plots``, no seeding, no thresholds keyed to the
truth), which is what makes the improvement a property of the algorithm rather
than of the test.

Checks:
  1. re-selection never produces a worse exact objective than pair refinement
     alone, and the returned control is the better of the two;
  2. when it fires, every logged field required for auditing a pass is present
     and self-consistent;
  3. the staleness gate is real: an unreachable threshold disables the stage and
     reproduces the pair-refinement-only control exactly;
  4. ``reselect_max_passes=0`` performs no passes;
  5. passes terminate (no cycling) and stay inside the admissible domain.

Runs at a short tfinal; the staleness/objective logic is horizon-independent.

Run:
  cd /path/to/Advection-Diffusion-PDECO
  python3 \
    "Applications/Application1/test scripts/test_second_source_reselection.py"
"""

from __future__ import annotations

import numpy as np

import _diag_common as dc
import sensitivity_initialization as sinit

ro = dc.ro
gt = dc.gt

TEST_TFINAL = 10.0
SIGMA2 = 1600.0
# Observation data only; these coordinates are never handed to the initializer.
DATA_CONTROL = np.array([260.0, 300.0, 1.55, 60.0, 180.0, 1.10])

_FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        _FAILURES.append(message)
        print(f"    FAIL  {message}")


def build(mesh_data, pde, admiss, observation, **kwargs):
    return sinit.build_sensitivity_initial_control(
        mesh_data, pde, observation, 2, dc.AMPLITUDE_BOUNDS, admiss,
        eta1=ro.ETA1, lam=ro.LAMBDA_REG,
        objective_time_rule=ro.OBJECTIVE_TIME_RULE,
        score="predicted_reduction", top_k=6, beam_width=2,
        refine_centers=True, refine_max_fev=12,
        refine_pair=True, pair_refine_max_fev=30,
        geometry_aware_branches=True, coherence_cap=0.3,
        verbose=False, **kwargs,
    )


def main() -> None:
    mesh_data, pde, admiss = dc.build_setup(
        ("left", "right", "bottom"), SIGMA2, tfinal=TEST_TFINAL
    )
    observation = gt.run_forward(mesh_data, pde, DATA_CONTROL)
    print(f"tfinal={TEST_TFINAL} sigma2={SIGMA2}; initializer receives observation "
          "data only\n")

    print("[1] baseline: pair refinement only")
    base_control, _d, base_extras = build(mesh_data, pde, admiss, observation)
    base_objective = float(base_extras["initial_exact_objective"])
    print(f"    control  = {np.array2string(base_control, precision=3)}")
    print(f"    J_exact  = {base_objective:.6e}")
    check(not base_extras["second_source_reselection"],
          "the baseline must perform no re-selection passes")

    print("[2] re-selection enabled")
    new_control, _d, new_extras = build(
        mesh_data, pde, admiss, observation,
        reselect_second=True, reselect_threshold=0.1,
        reselect_max_passes=2, reselect_refine_max_fev=30,
    )
    new_objective = float(new_extras["initial_exact_objective"])
    records = new_extras["second_source_reselection"]
    print(f"    control  = {np.array2string(new_control, precision=3)}")
    print(f"    J_exact  = {new_objective:.6e}  ({len(records)} pass(es) logged)")
    check(new_objective <= base_objective * (1.0 + 1.0e-12),
          "re-selection must never worsen the exact objective")
    check(admiss.validate_sources_in_domain(new_control),
          "the re-selected control must be admissible")

    accepted = [r for r in records if r["accepted"]]
    for record in records:
        print(f"    pass {record['pass']} on branch {record['branch_index']}: "
              f"leading source moved {record['stale_center_shift_m']:.2f} m "
              f"({record['stale_center_shift_sigma']:.2f} sigma), amplitude "
              f"{record['stale_amplitude_shift'] * 100:.1f}%")
        print(f"      second source {record['old_second_center']} -> "
              f"candidate {record['new_second_candidate']}  "
              f"moved {record['second_source_moved_m']:.2f} m")
        print(f"      J {record['objective_before']:.6e} -> "
              f"{record['objective_after_refit']:.6e} (refit) -> "
              f"{record['objective_after_refinement']:.6e} (4-D)  "
              f"accepted={record['accepted']}  "
              f"solves fwd={record['n_forward_solves']} "
              f"adj={record['n_adjoint_solves']}")
        for field in (
            "residual_first_source_centers", "residual_first_source_amplitudes",
            "old_second_amplitude", "candidate_coherence",
            "candidate_predicted_reduction", "n_refine_evaluations",
            "wall_distance_after_m",
        ):
            check(field in record, f"pass record must log {field}")
        check(record["stale_center_shift_sigma"] >= 0.1
              or record["stale_amplitude_shift"] >= 0.1,
              "a pass may only run when the staleness gate fires")
        check(record["n_adjoint_solves"] >= 1,
              "a pass must rebuild the residual adjoint")
        check(record["n_forward_solves"] >= 1,
              "a pass must rebuild the leading component's forward solution")
        check(record["objective_after_refinement"]
              <= record["objective_after_refit"] * (1.0 + 1.0e-9),
              "joint refinement must not worsen the refit objective")
    if accepted:
        best_before = min(r["objective_before"] for r in accepted)
        best_after = min(r["objective_after_refinement"] for r in accepted)
        print(f"    accepted passes: {len(accepted)}  "
              f"best J {best_before:.6e} -> {best_after:.6e}")
        check(best_after < best_before,
              "an accepted pass must strictly improve the exact objective")
        check(new_objective < base_objective,
              "an accepted pass must improve the produced control")
    else:
        print("    no pass improved on the pair-refined pair; control unchanged")
        check(np.allclose(new_control, base_control, atol=1e-9),
              "with no accepted pass the control must equal the baseline")

    print("[3] staleness gate: unreachable threshold disables the stage")
    gated_control, _d, gated_extras = build(
        mesh_data, pde, admiss, observation,
        reselect_second=True, reselect_threshold=1.0e6, reselect_max_passes=2,
    )
    print(f"    max|gated - baseline| = "
          f"{float(np.max(np.abs(gated_control - base_control))):.3e}")
    check(not gated_extras["second_source_reselection"],
          "an unreachable threshold must perform no passes")
    check(np.allclose(gated_control, base_control, atol=1e-9),
          "an unreachable threshold must reproduce the baseline control exactly")

    print("[4] max_passes=0 performs no passes")
    zero_control, _d, zero_extras = build(
        mesh_data, pde, admiss, observation,
        reselect_second=True, reselect_threshold=0.0, reselect_max_passes=0,
    )
    check(not zero_extras["second_source_reselection"],
          "max_passes=0 must perform no passes")
    check(np.allclose(zero_control, base_control, atol=1e-9),
          "max_passes=0 must reproduce the baseline control")

    print("[5] termination: passes are bounded and non-cycling")
    bounded_control, _d, bounded_extras = build(
        mesh_data, pde, admiss, observation,
        reselect_second=True, reselect_threshold=0.0, reselect_max_passes=2,
        reselect_refine_max_fev=30,
    )
    bounded_records = bounded_extras["second_source_reselection"]
    per_branch: dict[int, int] = {}
    for record in bounded_records:
        per_branch[record["branch_index"]] = per_branch.get(
            record["branch_index"], 0
        ) + 1
    print(f"    passes per branch: {per_branch or '{}'}  "
          f"J_exact={float(bounded_extras['initial_exact_objective']):.6e}")
    check(all(count <= 2 for count in per_branch.values()),
          "no branch may exceed reselect_max_passes")
    check(admiss.validate_sources_in_domain(bounded_control),
          "the produced control must stay admissible with a zero threshold")

    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} CHECK(S) FAILED:")
        for failure in _FAILURES:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("all second-source re-selection checks PASSED")


if __name__ == "__main__":
    main()
