"""Focused regressions for continuous feasibility handling, geometry-aware
branch deduplication, and joint pair refinement.

Covers the four properties the robustness-sweep follow-up depends on:

  1. an invalid trial center no longer produces the destructive ~1e12 objective
     jump under a line-search-sized perturbation (``--feasibility-mode project``),
     while the historical ``penalty`` mode still shows that cliff;
  2. the projected objective's gradient is the correct chain-rule derivative
     (the component normal to the wall is removed);
  3. genuinely valid near-wall sources -- including the Case 3b true control,
     whose strong source sits 0.37 m from a building -- stay admissible and are
     never moved;
  4. two centers a few metres apart but separated by a building are NOT
     deduplicated as one basin;
  5. joint pair refinement never increases the exact variable-projected
     objective.

Runs at a short tfinal: every property above is independent of the time horizon,
and the production horizon would make this a multi-hour test.

Run:
  cd /path/to/Advection-Diffusion-PDECO
  python3 \
    "Applications/Application1/test scripts/test_feasibility_and_pair_refinement.py"
"""

from __future__ import annotations

import numpy as np

import _diag_common as dc
import sensitivity_initialization as sinit
from mesh_admissibility import MeshAdmissibility

ro = dc.ro
gt = dc.gt

TEST_TFINAL = 10.0
SIGMA2 = 1600.0

# Case 3b from the robustness sweep (the development log).
CASE3B_TRUE = np.array([260.0, 300.0, 1.55, 60.0, 180.0, 1.10])
CASE3B_INIT = np.array(
    [276.70079126628343, 299.23289027376404, 1.245466051878722,
     107.9552177193691, 252.99682653768105, 0.8116989255223073]
)

_FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        _FAILURES.append(message)
        print(f"    FAIL  {message}")


# ----------------------------------------------------------------- geometry --
def test_near_wall_sources_stay_admissible(admiss: MeshAdmissibility) -> None:
    print("[1] valid near-wall sources remain admissible and are never moved")
    points = {
        "Case 3b true s1": (260.0, 300.0),
        "Case 3b true s2": (60.0, 180.0),
        "Case 3b init s1 (8 mm from a wall)": (
            CASE3B_INIT[0], CASE3B_INIT[1]
        ),
        "Case 2b true s1": (200.0, 560.0),
        "Case 2b true s2": (120.0, 470.0),
    }
    for name, (x, y) in points.items():
        inside = admiss.is_point_in_domain(x, y)
        clearance = admiss.distance_to_boundary(x, y)
        point, was_projected, moved = admiss.project_into_domain(x, y)
        print(f"    {name:<36} in_domain={inside} "
              f"wall={clearance:8.4f} m  projected={was_projected}")
        check(inside, f"{name} must stay admissible")
        check(not was_projected, f"{name} must not be projected")
        check(moved == 0.0, f"{name} must not move")
        check(float(point[0]) == float(x) and float(point[1]) == float(y),
              f"{name} must be returned unchanged")
    check(admiss.validate_sources_in_domain(CASE3B_TRUE),
          "the Case 3b true control must remain admissible")
    # A near-wall clearance well under one Gaussian radius is exactly the case a
    # keep-out margin would have excluded; assert we did not introduce one.
    check(admiss.distance_to_boundary(260.0, 300.0) < 1.0,
          "Case 3b true s1 is expected to sit under 1 m from a wall")


def test_projection_is_admissible_and_continuous(admiss: MeshAdmissibility) -> None:
    print("[2] projection returns admissible points and is continuous at the wall")
    center = np.array([CASE3B_INIT[0], CASE3B_INIT[1]])
    _distance, foot = admiss.nearest_boundary_point(center[0], center[1])
    outward = foot - center
    outward = outward / float(np.hypot(*outward))
    previous = None
    for step in (1.0e-3, 1.0e-2, 0.1, 1.0, 5.0):
        trial = center + step * outward
        inside = admiss.is_point_in_domain(trial[0], trial[1])
        point, was_projected, moved = admiss.project_into_domain(trial[0], trial[1])
        if inside:
            continue
        print(f"    step {step:>6.3f} m outward -> projected, moved {moved:.6f} m, "
              f"admissible={admiss.is_point_in_domain(point[0], point[1])}")
        check(was_projected, f"trial at {step} m outside must be projected")
        check(admiss.is_point_in_domain(point[0], point[1]),
              f"projection of the {step} m trial must be admissible")
        check(moved <= step + 1.0e-6,
              "a projection may not move a point further than its violation")
        if previous is not None:
            check(moved >= previous - 1.0e-9,
                  "projection distance must grow with the violation")
        previous = moved
    # Continuity: the closer the trial is to the wall, the smaller the correction.
    tiny = center + 1.0e-6 * outward
    _p, was_projected, moved = admiss.project_into_domain(tiny[0], tiny[1])
    if was_projected:
        check(moved < 1.0e-3, "a micrometre violation must give a micrometre fix")


def test_building_separated_branches_not_deduplicated(
    admiss: MeshAdmissibility,
) -> None:
    print("[3] building-separated centers are not one basin")
    dedup_radius = max(2.0, 0.25 * float(np.sqrt(SIGMA2)))  # production rule: 10 m
    # Search the mesh for a real pair: both fluid, closer than the dedup radius,
    # but with a building between them.
    coord = np.asarray(admiss.coordinates)[:, :2]
    separated = None
    for i in range(0, coord.shape[0], 7):
        p = coord[i]
        if not admiss.is_point_in_domain(p[0], p[1]):
            continue
        near = np.flatnonzero(
            (np.abs(coord[:, 0] - p[0]) < dedup_radius)
            & (np.abs(coord[:, 1] - p[1]) < dedup_radius)
        )
        for j in near:
            q = coord[j]
            distance = float(np.hypot(*(q - p)))
            if not (1.0 < distance < dedup_radius):
                continue
            if not admiss.is_point_in_domain(q[0], q[1]):
                continue
            if not admiss.segment_in_domain(p, q, spacing=0.25):
                separated = (p.copy(), q.copy(), distance)
                break
        if separated is not None:
            break
    check(separated is not None,
          "expected to find a building-separated pair inside the dedup radius")
    if separated is None:
        return
    p, q, distance = separated
    print(f"    ({p[0]:.2f},{p[1]:.2f}) and ({q[0]:.2f},{q[1]:.2f}): "
          f"{distance:.2f} m apart, dedup radius {dedup_radius:.1f} m, "
          "segment leaves the fluid domain")
    check(
        sinit.centers_same_basin(p, q, dedup_radius, admiss, geometry_aware=False),
        "the distance-only rule is expected to merge this pair (documents the bug)",
    )
    check(
        not sinit.centers_same_basin(p, q, dedup_radius, admiss, geometry_aware=True),
        "the geometry-aware rule must NOT merge a building-separated pair",
    )
    check(
        not sinit.branches_same_basin([p], [q], dedup_radius, admiss, True),
        "branch dedup must keep building-separated branches distinct",
    )
    # A same-street pair must still be merged, or dedup would never fire.
    for shift in (1.0, 2.0, 3.0):
        r = p + np.array([shift, 0.0])
        if admiss.is_point_in_domain(r[0], r[1]) and admiss.segment_in_domain(p, r):
            check(
                sinit.centers_same_basin(p, r, dedup_radius, admiss, True),
                "a visible same-street pair must still be one basin",
            )
            print(f"    same-street control pair {shift:.0f} m apart: merged")
            break


# ---------------------------------------------------------------- evaluator --
def _configure_pde(mesh_data) -> None:
    velocity, summary = ro.load_velocity_field_file(
        str(ro.DEFAULT_VELOCITY_FIELD_FILE), mesh_data
    )
    ro.PDE["tfinal"] = TEST_TFINAL
    ro.PDE["sigma2"] = SIGMA2
    ro.PDE["velocity_nodes"] = velocity
    ro.PDE["velocity_field"] = summary


def test_no_objective_cliff(mesh_data, admiss: MeshAdmissibility) -> None:
    print("[4] no destructive objective jump when a trial center crosses a wall")
    target = gt.run_forward(mesh_data, ro.PDE, CASE3B_TRUE)
    center = np.array([CASE3B_INIT[0], CASE3B_INIT[1]])
    _distance, foot = admiss.nearest_boundary_point(center[0], center[1])
    outward = foot - center
    outward = outward / float(np.hypot(*outward))
    clearance = admiss.distance_to_boundary(center[0], center[1])
    print(f"    strong source sits {clearance * 1000.0:.2f} mm from the wall; "
          "stepping outward along the wall normal")

    steps = (0.0, 0.5 * clearance, 2.0 * clearance, 0.05, 0.5)
    values = {}
    for label in ("penalty", "project"):
        stats = ro.new_feasibility_stats(label)
        _c, _h, objective, gradient, _cb = ro.build_evaluator(
            mesh_data, target, "location_and_intensity",
            ro.MESH_DOMAIN_BOUND_CASE, None, CASE3B_INIT, admiss,
            feasibility_mode=label, feasibility_stats=stats,
        )
        row = []
        for step in steps:
            trial = CASE3B_INIT.copy()
            trial[0:2] = center + step * outward
            valid = admiss.validate_sources_in_domain(trial)
            row.append((step, valid, float(objective(trial))))
        values[label] = row
        print(f"    {label}:")
        for step, valid, value in row:
            print(f"      step {step:>9.6f} m  "
                  f"{'valid' if valid else 'INVALID':>7}  J = {value:.10e}")
        print(f"      stats: invalid={stats['invalid_evaluations']} "
              f"projected={stats['projected_evaluations']} "
              f"penalized={stats['penalized_evaluations']} "
              f"max move={stats['max_projection_distance']:.6g} m")
        if label == "project":
            check(stats["projected_evaluations"] > 0,
                  "project mode must actually project the invalid trials")
        else:
            check(stats["penalized_evaluations"] > 0,
                  "penalty mode must actually penalize the invalid trials")

    j0_penalty = values["penalty"][0][2]
    j0_project = values["project"][0][2]
    check(abs(j0_penalty - j0_project) <= 1.0e-12 * max(1.0, abs(j0_penalty)),
          "the two modes must agree exactly on admissible controls")

    invalid_penalty = [v for _s, valid, v in values["penalty"] if not valid]
    check(bool(invalid_penalty) and max(invalid_penalty) > 1.0e11,
          "penalty mode is expected to still show the ~1e12 cliff")

    for step, valid, value in values["project"]:
        check(np.isfinite(value), f"project mode must stay finite at step {step}")
        check(value < 10.0 * j0_project,
              f"project mode must not jump at step {step} "
              f"(J={value:.6e} vs J0={j0_project:.6e})")
    # Crossing the wall must be continuous: a step just inside and a step just
    # outside evaluate to nearly the same objective.
    inside_value = values["project"][1][2]
    outside_value = values["project"][2][2]
    relative = abs(outside_value - inside_value) / max(1.0, abs(inside_value))
    print(f"    relative objective change across the wall: {relative:.3e}")
    check(relative < 1.0e-3,
          "the projected objective must be continuous across the fluid boundary")


def test_projected_gradient_is_tangential(mesh_data, admiss: MeshAdmissibility) -> None:
    print("[5] projected gradient drops the component normal to the wall")
    target = gt.run_forward(mesh_data, ro.PDE, CASE3B_TRUE)
    center = np.array([CASE3B_INIT[0], CASE3B_INIT[1]])
    _distance, foot = admiss.nearest_boundary_point(center[0], center[1])
    outward = foot - center
    outward = outward / float(np.hypot(*outward))

    stats = ro.new_feasibility_stats("project")
    _c, _h, objective, gradient, _cb = ro.build_evaluator(
        mesh_data, target, "location_and_intensity",
        ro.MESH_DOMAIN_BOUND_CASE, None, CASE3B_INIT, admiss,
        feasibility_mode="project", feasibility_stats=stats,
    )
    trial = CASE3B_INIT.copy()
    trial[0:2] = center + 0.05 * outward
    check(not admiss.validate_sources_in_domain(trial),
          "the probe point must be outside the fluid domain")
    objective(trial)
    g = gradient(trial)
    normal_component = float(g[0:2] @ outward)
    print(f"    ||g|| = {np.linalg.norm(g):.6e}  "
          f"normal component = {normal_component:.6e}")
    check(np.all(np.isfinite(g)), "the projected gradient must be finite")
    check(abs(normal_component) <= 1.0e-8 * max(1.0, float(np.linalg.norm(g))),
          "the wall-normal gradient component must be removed")


def test_pair_refinement_never_worsens(mesh_data, pde, admiss) -> None:
    print("[6] joint pair refinement never increases the exact VP objective")
    observation = gt.run_forward(mesh_data, pde, CASE3B_TRUE)
    control, _diag, extras = sinit.build_sensitivity_initial_control(
        mesh_data, pde, observation, 2, dc.AMPLITUDE_BOUNDS, admiss,
        eta1=ro.ETA1, lam=ro.LAMBDA_REG,
        objective_time_rule=ro.OBJECTIVE_TIME_RULE,
        score="predicted_reduction", top_k=4, beam_width=2,
        refine_pair=True, pair_refine_max_fev=12,
        geometry_aware_branches=True, verbose=False,
    )
    records = extras["pair_refinement"]
    check(bool(records), "pair refinement must report per-branch records")
    for record in records:
        before = record["objective_before"]
        after = record["objective_after"]
        print(f"    branch {record['branch_index']}: J {before:.6e} -> {after:.6e}  "
              f"accepted={record['accepted']} evals={record['n_forward_evaluations']}  "
              f"moved {['%.2f m' % d for d in record['moved_m']]}")
        check(after <= before + 1.0e-9 * max(1.0, abs(before)),
              "pair refinement must not increase the exact objective")
        check(len(record["centers_after"]) == 2,
              "pair refinement must report both centers")
        check(len(record["wall_distance_after_m"]) == 2,
              "pair refinement must report a wall distance per source")
        for center in record["centers_after"]:
            check(admiss.is_point_in_domain(center[0], center[1]),
                  "a refined center must stay admissible")
    check(admiss.validate_sources_in_domain(control),
          "the produced control must be admissible")
    branch_controls = extras["branch_controls"]
    check(bool(branch_controls), "retained branches must be reported for multistart")
    objectives = [b["objective"] for b in branch_controls]
    check(objectives == sorted(objectives),
          "retained branches must be reported best-objective first")
    check(np.allclose(np.asarray(branch_controls[0]["control"]), control),
          "the returned control must be the best retained branch")
    print(f"    retained branches: {len(branch_controls)}  "
          f"objectives {['%.4e' % o for o in objectives]}")


def main() -> None:
    mesh_data, pde, admiss = dc.build_setup(
        ("left", "right", "bottom"), SIGMA2, tfinal=TEST_TFINAL
    )
    _configure_pde(mesh_data)
    print(f"mesh: {np.asarray(mesh_data['coord']).shape[0]} nodes, "
          f"{admiss.boundary_edge_count} boundary edges, "
          f"tfinal={TEST_TFINAL} sigma2={SIGMA2}\n")

    test_near_wall_sources_stay_admissible(admiss)
    test_projection_is_admissible_and_continuous(admiss)
    test_building_separated_branches_not_deduplicated(admiss)
    test_no_objective_cliff(mesh_data, admiss)
    test_projected_gradient_is_tangential(mesh_data, admiss)
    test_pair_refinement_never_worsens(mesh_data, pde, admiss)

    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} CHECK(S) FAILED:")
        for failure in _FAILURES:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("all feasibility / geometry / pair-refinement checks PASSED")


if __name__ == "__main__":
    main()
