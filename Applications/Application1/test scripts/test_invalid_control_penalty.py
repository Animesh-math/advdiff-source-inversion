"""Regression test for the Application1 invalid-control (out-of-domain) penalty.

Reproduces the scale failure that was observed in the sigma-continuation run
``right_adaptive_ampfloor_01_sigma2_1600_800_400_seed42``:

  * At broad continuation stages the legitimate objective was ~1e8.
  * L-BFGS-B stepped to a source center inside a building hole / outside the
    fluid mesh but still inside the rectangular coordinate bounds.
  * The old evaluator returned a flat ``J = 1e6`` with an exactly zero gradient.
  * Because ``1e6 << 1e8``, the invalid point looked like a dramatic
    improvement, and the zero gradient created a fake stationary point, so
    L-BFGS-B "converged" onto the invalid control and warm-started later stages
    from it.

This test drives ``build_evaluator`` with the forward/objective solve mocked out
so the "valid" objective is a controlled ~1e8 value (no PDE solve needed).  It
then asserts the two properties the fix must guarantee:

  1. An invalid control is never cheaper than a legitimate ~1e8 objective (at any
     stage), including a center that is inside the rectangular bounding box but
     inside a building hole.
  2. An invalid control does not produce a fake zero-gradient optimum: the
     gradient is finite, nonzero, and *restores* the offending center toward the
     fluid domain (moving along the negative gradient decreases the distance to
     the domain).

Run with the project environment from the repository root:

  python3 \
      "Applications/Application1/test_invalid_control_penalty.py"
"""

import json
import os as _os
import sys as _sys

import numpy as np

# This test/verification script lives in Application1/"test scripts"/; put the
# parent Application1 directory on sys.path so its local modules import cleanly.
_APP1_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _APP1_DIR not in _sys.path:
    _sys.path.insert(0, _APP1_DIR)

# Importing run_optimization first appends Src/Objective to sys.path, which
# is where the shared gradient_test module lives.
import run_optimization as ro
import gradient_test as gt
from mesh_admissibility import MeshAdmissibility
from mesh_data_config import BUILDING_METADATA_FILE, build_mesh_data
from multi_source_utils import (
    MESH_DOMAIN_BOUND_CASE,
    mode_fixed_values,
    pack_active,
)


# A representative legitimate Application1 objective at a broad continuation
# stage (sigma2=1600, tfinal=150 produced objectives ~9e7..1.5e8).
VALID_OBJECTIVE = 1.5e8


def _valid_control(admissibility: MeshAdmissibility) -> np.ndarray:
    """A two-source control whose centers are both inside the fluid domain."""
    control = np.array(
        [37.0278, 381.9172, 1.10, 267.0278, 201.9172, 0.85], dtype=np.float64
    )
    assert admissibility.validate_sources_in_domain(control), (
        "test setup error: default control is not admissible"
    )
    return control


def _building_interior_point(admissibility: MeshAdmissibility):
    """A point inside the rectangular bounding box but inside a building hole.

    Uses the committed building metadata centroids; most building footprints are
    convex enough that their centroid lies inside the hole (inadmissible) while
    still inside the coordinate bounding box.  Returns ``None`` if no such point
    is found (then the test relies on the outside-bbox case only).
    """
    xmin, xmax, ymin, ymax = admissibility.bounding_box
    metadata = json.loads(BUILDING_METADATA_FILE.read_text(encoding="utf-8"))
    for info in metadata.values():
        cx, cy = float(info["centroid"][0]), float(info["centroid"][1])
        inside_bbox = xmin <= cx <= xmax and ymin <= cy <= ymax
        if inside_bbox and not admissibility.is_point_in_domain(cx, cy):
            return cx, cy
    return None


def _build_mocked_evaluator(mesh_data, admissibility, mode):
    """Build an evaluator with the forward/objective solve mocked to a fixed
    valid objective, so no PDE solve is required.

    The evaluator resolves ``gt.run_forward`` / ``gt.compute_objective`` on the
    module at call time, so the mocks (installed by ``run`` for the whole test)
    take effect for any valid evaluation.  Invalid controls never reach the
    forward solve at all -- they take the invalid-iterate penalty branch.
    """
    fixed_values = mode_fixed_values(mode, z_ref_full=_valid_control(admissibility))
    z0_full = _valid_control(admissibility)
    _cache, _history, objective, gradient, _callback = ro.build_evaluator(
        mesh_data,
        target_state=None,
        mode=mode,
        bound_case=MESH_DOMAIN_BOUND_CASE,
        fixed_values=fixed_values,
        z0_full=z0_full,
        admissibility=admissibility,
    )
    return objective, gradient


def _check_invalid_center(objective, gradient, admissibility, mode, invalid_center,
                          valid_control, description):
    # First establish the valid objective scale in the evaluator.
    z_valid = pack_active(valid_control, mode)
    j_valid = objective(z_valid)
    assert np.isclose(j_valid, VALID_OBJECTIVE), (
        f"mocked valid objective changed: {j_valid} != {VALID_OBJECTIVE}"
    )

    # Build an invalid control by moving source 1's center out of the domain.
    invalid_control = valid_control.copy()
    invalid_control[0:2] = invalid_center
    assert not admissibility.validate_sources_in_domain(invalid_control), (
        f"test setup error: {description} center is admissible"
    )

    z_invalid = pack_active(invalid_control, mode)
    j_invalid = objective(z_invalid)
    g_invalid = np.asarray(gradient(z_invalid), dtype=np.float64)

    # Property 1: the invalid iterate must never look better than a legitimate
    # ~1e8 objective, with a large margin.
    assert np.isfinite(j_invalid), f"{description}: invalid objective is not finite"
    assert j_invalid > 100.0 * VALID_OBJECTIVE, (
        f"{description}: invalid objective {j_invalid:.6e} is not safely larger "
        f"than the valid objective {VALID_OBJECTIVE:.6e}"
    )
    assert j_invalid >= ro.INVALID_CONTROL_PENALTY_FLOOR, (
        f"{description}: invalid objective {j_invalid:.6e} is below the floor "
        f"{ro.INVALID_CONTROL_PENALTY_FLOOR:.6e}"
    )

    # Property 2: no fake zero-gradient optimum.
    assert np.all(np.isfinite(g_invalid)), f"{description}: gradient not finite"
    grad_norm = float(np.linalg.norm(g_invalid))
    assert grad_norm > 0.0, (
        f"{description}: invalid control produced a zero gradient (fake "
        "stationary point)"
    )

    # The gradient must *restore* the center: moving along -grad decreases the
    # squared distance to the fluid domain.
    center_grad = g_invalid[0:2]
    assert float(np.linalg.norm(center_grad)) > 0.0, (
        f"{description}: center gradient is zero"
    )
    dist_before, _ = admissibility.nearest_domain_distance_sq(*invalid_center)
    step = 1.0e-3 / float(np.linalg.norm(center_grad))
    moved = np.asarray(invalid_center, dtype=np.float64) - step * center_grad
    dist_after, _ = admissibility.nearest_domain_distance_sq(moved[0], moved[1])
    assert dist_after < dist_before, (
        f"{description}: negative gradient step did not move the center toward "
        f"the domain (dist_sq {dist_before:.6e} -> {dist_after:.6e})"
    )

    print(
        f"  {description}: J_valid={j_valid:.6e} J_invalid={j_invalid:.6e} "
        f"||g||={grad_norm:.6e} dist_sq {dist_before:.4e}->{dist_after:.4e} OK"
    )


def run() -> None:
    mesh_data = build_mesh_data()
    admissibility = MeshAdmissibility(mesh_data)
    valid_control = _valid_control(admissibility)

    xmin, xmax, ymin, ymax = admissibility.bounding_box
    # Well outside the rectangular bounding box.
    outside_bbox = (xmax + 100.0, ymax + 100.0)
    building_point = _building_interior_point(admissibility)
    if building_point is None:
        raise AssertionError(
            "regression test requires a building-interior point inside the "
            "bounding box (requirement 4); none was found"
        )

    original_run_forward = gt.run_forward
    original_compute_objective = gt.compute_objective

    def fake_run_forward(_mesh_data, _pde, _control):
        return np.zeros((1, mesh_data["coord"].shape[0]), dtype=np.float64)

    def fake_compute_objective(*_args, **_kwargs):
        return VALID_OBJECTIVE

    gt.run_forward = fake_run_forward
    gt.compute_objective = fake_compute_objective
    try:
        for mode in ("location_and_intensity", "location_only"):
            print(f"mode: {mode}")
            objective, gradient = _build_mocked_evaluator(
                mesh_data, admissibility, mode
            )
            _check_invalid_center(
                objective, gradient, admissibility, mode, outside_bbox,
                valid_control, "outside bounding box",
            )
            _check_invalid_center(
                objective, gradient, admissibility, mode, building_point,
                valid_control, "inside building hole (in bbox)",
            )
    finally:
        gt.run_forward = original_run_forward
        gt.compute_objective = original_compute_objective

    print("invalid-control penalty regression: PASS")


if __name__ == "__main__":
    run()
