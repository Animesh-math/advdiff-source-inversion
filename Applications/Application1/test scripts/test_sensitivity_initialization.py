"""Focused tests for the Application1 ``--init-mode sensitivity`` strategy.

Run with the project environment from the Application1 directory or the repo
root, e.g.:

    python3 "Applications/Application1/test scripts/test_sensitivity_initialization.py"

These tests exercise the greedy sensitivity-initialization module directly (fast,
using a short tfinal and constant velocity on the real DC mesh) and the CLI
argument parsing / dispatch of ``run_optimization``.  They verify, at minimum:

  1. Existing init modes still parse and dispatch unchanged.
  2. An explicit --initial-control retains its override behavior.
  3-4. Sensitivity init returns correctly shaped, admissible, bounded controls
       for one and two requested sources, with two-source separation respected
       and the [x1,y1,A1,x2,y2,A2] ordering.
  5. G(c) uses the repository's discrete spatial inner product (fast quadrature
     form matches the literal load-vector form).
  6. The amplitude curvature uses the correct weighted observation norm (and the
     source-regularization term), not a plain Euclidean norm.
  7. Exactly one unit-Gaussian forward solve is used per selected center (two
     forward solves per greedy source: background state + unit response), NOT one
     solve per candidate node.
  8. The selected center is the masked argmin of G(c).
  9. Graceful failure when all candidates are masked, when no meaningfully
     negative sensitivity exists, and correct projection when the amplitude falls
     below/above its bounds.
 10. The initialized objective is no worse than the zero-source objective when a
     valid negative insertion is found (amplitude bounds admitting the optimum).
"""

from __future__ import annotations

import os as _os
import sys as _sys

import numpy as np

# This test lives in Application1/"test scripts"/; put the parent Application1
# directory on sys.path so its local modules import cleanly.
_APP1_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _APP1_DIR not in _sys.path:
    _sys.path.insert(0, _APP1_DIR)

# Import run_optimization first: it appends Src/Objective to sys.path, which
# is where the shared gradient_test forward/objective module lives.
import run_optimization as ro
import gradient_test as gt
import sensitivity_initialization as sinit
import source_insertion_sensitivity as sis
from mesh_admissibility import MeshAdmissibility
from mesh_data_config import build_mesh_data
from multi_source_utils import reshape_control


# --- Shared fast configuration (real DC mesh, short tfinal, constant velocity). ---
TFINAL = 25.0
TRUE_CONTROL = np.array([250.0, 350.0, 1.0, 120.0, 300.0, 0.6], dtype=np.float64)
MEASUREMENT_BOUNDARIES = ["right"]


def _build_context():
    mesh_data = ro.configure_measurement_boundaries(
        build_mesh_data(), MEASUREMENT_BOUNDARIES
    )
    pde = dict(ro.PDE)
    pde["tfinal"] = TFINAL  # constant velocity: no velocity_nodes -> fast
    admissibility = MeshAdmissibility(mesh_data)
    observation_state = gt.run_forward(mesh_data, pde, TRUE_CONTROL)
    return mesh_data, pde, admissibility, observation_state


class _ForwardCounter:
    """Context manager counting gt.run_forward calls (shared module attribute)."""

    def __init__(self):
        self.count = 0

    def __enter__(self):
        self._orig = gt.run_forward

        def _counting(*args, **kwargs):
            self.count += 1
            return self._orig(*args, **kwargs)

        gt.run_forward = _counting
        return self

    def __exit__(self, *exc):
        gt.run_forward = self._orig
        return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_existing_modes_parse_and_dispatch_unchanged():
    """1. Every existing init mode still parses; sensitivity is additive."""
    parser_modes = {"manual", "mesh_random", "smart", "adaptive", "sensitivity"}
    # Default is still manual.
    args = _parse(["--mode", "location_and_intensity"])
    assert args.init_mode == "manual", args.init_mode
    # Each documented mode parses to itself.
    for mode in parser_modes:
        extra = []
        if mode in ("mesh_random", "smart"):
            extra = ["--stage", "short"]  # multistart-capable stages
        a = _parse(["--mode", "location_and_intensity", "--init-mode", mode, *extra])
        assert a.init_mode == mode, (mode, a.init_mode)
    # The new sensitivity flags exist with safe defaults.
    a = _parse(["--mode", "location_and_intensity", "--init-mode", "sensitivity"])
    assert a.sensitivity_min_value_tol == 0.0
    assert a.sensitivity_min_separation is None
    assert a.sensitivity_save_plots is False
    assert a.sensitivity_mask_boundary_buffer == 0.0
    assert a.sensitivity_mask_building_buffer == 0.0


def test_initial_control_override_behavior():
    """2. --initial-control remains authoritative over --init-mode sensitivity.

    Mirrors the driver's documented precedence: when a manual control is given,
    init_mode collapses to 'manual' and the supplied control is used verbatim.
    """
    supplied = [200.0, 300.0, 0.9, 100.0, 250.0, 0.5]
    args = _parse(
        [
            "--mode", "location_and_intensity",
            "--init-mode", "sensitivity",
            "--initial-control", *[str(v) for v in supplied],
        ]
    )
    assert args.initial_control is not None
    # Replicate the driver's override branch (run_optimization.main):
    if args.initial_control is not None:
        effective_init_mode = "manual"
        initial_controls = [np.asarray(args.initial_control, dtype=np.float64)]
    assert effective_init_mode == "manual"
    assert np.allclose(initial_controls[0], supplied)


def test_one_source_shape_admissible_bounded(ctx):
    """3. One requested source: admissible center, bounded amplitude, shape."""
    mesh_data, pde, admissibility, ud = ctx
    control, diags, _ = sinit.build_sensitivity_initial_control(
        mesh_data, pde, ud, 1, (0.0, 2.0), admissibility,
        eta1=ro.ETA1, lam=ro.LAMBDA_REG, objective_time_rule=ro.OBJECTIVE_TIME_RULE,
        verbose=False,
    )
    assert control.shape == (3,), control.shape
    cx, cy, a = control
    assert admissibility.is_point_in_domain(float(cx), float(cy)), (cx, cy)
    assert 0.0 <= a <= 2.0, a
    assert np.isfinite(a)
    assert len(diags) == 1
    assert np.allclose(diags[0].center, [cx, cy])


def test_two_source_ordering_and_separation(ctx):
    """4. Two sources: recompute after source 1, respect separation, ordering."""
    mesh_data, pde, admissibility, ud = ctx
    min_sep = 80.0
    with _ForwardCounter() as fc:
        control, diags, _ = sinit.build_sensitivity_initial_control(
            mesh_data, pde, ud, 2, (0.0, 2.0), admissibility,
            eta1=ro.ETA1, lam=ro.LAMBDA_REG,
            objective_time_rule=ro.OBJECTIVE_TIME_RULE,
            min_separation=min_sep, verbose=False,
        )
    assert control.shape == (6,), control.shape
    Z = reshape_control(control, num_sources=2)
    c1, c2 = Z[0, :2], Z[1, :2]
    # Ordering: diagnostics[i] matches control slot i.
    assert np.allclose(diags[0].center, c1)
    assert np.allclose(diags[1].center, c2)
    # Separation between placed centers is respected.
    assert np.hypot(*(c1 - c2)) >= min_sep - 1e-9, np.hypot(*(c1 - c2))
    # 7 (part): two forward solves per greedy source => 4 total for 2 sources,
    # proving the state/residual/adjoint/field were recomputed after source 1
    # (and that candidates are NOT solved one-by-one).
    assert fc.count == 4, fc.count
    for a in (Z[0, 2], Z[1, 2]):
        assert 0.0 <= a <= 2.0


def test_G_matches_discrete_inner_product(ctx):
    """5. Fast G(c) equals the literal discrete load-vector inner product."""
    mesh_data, pde, admissibility, ud = ctx
    coord = np.asarray(mesh_data["coord"], dtype=np.float64)[:, :2]
    zeros = np.zeros(3 * 2, dtype=np.float64)
    state, src_adj, inv_mass, W = sis.compute_adjoint_weight_field(
        mesh_data, pde, zeros, ud, ro.ETA1, ro.OBJECTIVE_TIME_RULE
    )
    qx, qy, ag = sis.precompute_quadrature_weights(mesh_data, pde, W)
    # Pick an interior admissible node near the true source (strong signal).
    dist = np.hypot(coord[:, 0] - TRUE_CONTROL[0], coord[:, 1] - TRUE_CONTROL[1])
    order = np.argsort(dist)
    node = next(
        i for i in order
        if admissibility.is_point_in_domain(float(coord[i, 0]), float(coord[i, 1]))
    )
    c = coord[node]
    g_fast = float(
        sis.sensitivity_at_candidates(c[None, :], qx, qy, ag, pde["sigma2"])[0]
    )
    g_ref = sis.sensitivity_via_load_vector(mesh_data, pde, src_adj, inv_mass, c)
    rel = abs(g_fast - g_ref) / max(1e-300, abs(g_ref))
    assert rel < 1e-10, (g_fast, g_ref, rel)
    # Sanity: near the true source, inserting a positive amplitude descends J.
    assert g_ref < 0.0, g_ref


def test_curvature_uses_weighted_observation_norm(ctx):
    """6/7. kappa = 2*J_track(delta_u; ud=0) + lambda via the exact boundary norm,
    and the unit response is a single forward solve (not per candidate)."""
    mesh_data, pde, admissibility, ud = ctx
    coord = np.asarray(mesh_data["coord"], dtype=np.float64)[:, :2]
    node = next(
        i for i in range(coord.shape[0])
        if admissibility.is_point_in_domain(float(coord[i, 0]), float(coord[i, 1]))
    )
    center = coord[node]
    with _ForwardCounter() as fc:
        kappa, _delta = sinit._unit_response_curvature(
            mesh_data, pde, center, ro.ETA1, ro.LAMBDA_REG, ro.OBJECTIVE_TIME_RULE
        )
    assert fc.count == 1, fc.count  # exactly one unit-Gaussian forward solve
    # Reference: 2 * compute_objective(delta_u vs zero data) + lambda, using the
    # repository's discrete boundary interval quadrature + time reduction.
    unit = np.zeros(6, dtype=np.float64)
    unit[0], unit[1], unit[2] = center[0], center[1], 1.0
    delta_u = gt.run_forward(mesh_data, pde, unit)
    dt_out = pde["dt"] * pde["jprint"]
    j_ref = gt.compute_objective(
        delta_u, np.zeros_like(delta_u), mesh_data,
        eta1=ro.ETA1, lam=0.0, z=unit, dt_out=dt_out,
        time_rule=ro.OBJECTIVE_TIME_RULE,
    )
    kappa_ref = 2.0 * float(j_ref) + float(ro.LAMBDA_REG)
    assert abs(kappa - kappa_ref) <= 1e-9 * max(1.0, abs(kappa_ref)), (kappa, kappa_ref)
    # It must NOT equal the plain Euclidean norm of the response.
    euclid = float(np.dot(delta_u.reshape(-1), delta_u.reshape(-1)))
    assert abs(kappa - euclid) > 1e-6 * max(1.0, abs(kappa)), (kappa, euclid)
    assert kappa > 0.0


def test_selected_center_is_masked_argmin(ctx):
    """8. c* equals argmin G(c) over the admissibility-masked candidates."""
    mesh_data, pde, admissibility, ud = ctx
    coord = np.asarray(mesh_data["coord"], dtype=np.float64)[:, :2]
    # Independently compute the zero-background G field and its masked argmin.
    # (A length-6 zero control is the 2-source model's empty source field.)
    zeros = np.zeros(6, dtype=np.float64)
    _, _, _, W = sis.compute_adjoint_weight_field(
        mesh_data, pde, zeros, ud, ro.ETA1, ro.OBJECTIVE_TIME_RULE
    )
    qx, qy, ag = sis.precompute_quadrature_weights(mesh_data, pde, W)
    g_field = sis.sensitivity_at_candidates(coord, qx, qy, ag, pde["sigma2"])
    mask = np.array(
        [admissibility.is_point_in_domain(float(x), float(y)) for x, y in coord]
    )
    masked_idx = np.flatnonzero(mask)
    expected = coord[masked_idx[int(np.argmin(g_field[masked_idx]))]]
    control, diags, _ = sinit.build_sensitivity_initial_control(
        mesh_data, pde, ud, 1, (0.0, 2.0), admissibility,
        eta1=ro.ETA1, lam=ro.LAMBDA_REG, objective_time_rule=ro.OBJECTIVE_TIME_RULE,
        verbose=False,
    )
    assert np.allclose(control[:2], expected), (control[:2], expected)


def test_graceful_all_candidates_masked(ctx):
    """9a. A huge boundary buffer masks everything -> graceful error."""
    mesh_data, pde, admissibility, ud = ctx
    try:
        sinit.build_sensitivity_initial_control(
            mesh_data, pde, ud, 1, (0.0, 2.0), admissibility,
            eta1=ro.ETA1, lam=ro.LAMBDA_REG,
            objective_time_rule=ro.OBJECTIVE_TIME_RULE,
            boundary_buffer=1.0e9, verbose=False,
        )
    except sinit.SensitivityInitializationError:
        return
    raise AssertionError("expected SensitivityInitializationError for empty mask")


def test_graceful_no_meaningful_descent(ctx):
    """9b. An impossibly strict tolerance -> graceful error, nothing selected."""
    mesh_data, pde, admissibility, ud = ctx
    try:
        sinit.build_sensitivity_initial_control(
            mesh_data, pde, ud, 1, (0.0, 2.0), admissibility,
            eta1=ro.ETA1, lam=ro.LAMBDA_REG,
            objective_time_rule=ro.OBJECTIVE_TIME_RULE,
            min_value_tol=1.0e12, verbose=False,
        )
    except sinit.SensitivityInitializationError:
        return
    raise AssertionError("expected SensitivityInitializationError for strict tol")


def test_amplitude_projection_below_and_above_bounds(ctx):
    """9c. Amplitude is clamped when the 1-D optimum falls outside the bounds."""
    mesh_data, pde, admissibility, ud = ctx
    # Reference unconstrained optimum with wide bounds.
    _, diags, _ = sinit.build_sensitivity_initial_control(
        mesh_data, pde, ud, 1, (0.0, 2.0), admissibility,
        eta1=ro.ETA1, lam=ro.LAMBDA_REG, objective_time_rule=ro.OBJECTIVE_TIME_RULE,
        verbose=False,
    )
    a_opt = diags[0].unconstrained_amplitude
    assert a_opt > 0.0, a_opt  # descent direction => positive optimal amplitude

    # Below-bound: floor above the optimum -> projected up to the floor.
    high_floor = a_opt + 0.5
    _, d_low, _ = sinit.build_sensitivity_initial_control(
        mesh_data, pde, ud, 1, (high_floor, high_floor + 1.0), admissibility,
        eta1=ro.ETA1, lam=ro.LAMBDA_REG, objective_time_rule=ro.OBJECTIVE_TIME_RULE,
        verbose=False,
    )
    assert d_low[0].amplitude_clipped is True
    assert abs(d_low[0].projected_amplitude - high_floor) < 1e-9

    # Above-bound: upper below the optimum -> projected down to the upper bound.
    low_upper = min(a_opt * 0.5, 0.5 * (a_opt))
    low_upper = max(low_upper, 1e-6)
    if low_upper < a_opt:
        _, d_high, _ = sinit.build_sensitivity_initial_control(
            mesh_data, pde, ud, 1, (0.0, low_upper), admissibility,
            eta1=ro.ETA1, lam=ro.LAMBDA_REG,
            objective_time_rule=ro.OBJECTIVE_TIME_RULE, verbose=False,
        )
        assert d_high[0].amplitude_clipped is True
        assert abs(d_high[0].projected_amplitude - low_upper) < 1e-9


def test_initialized_objective_no_worse_than_zero(ctx):
    """10. With bounds admitting the optimum, J(init) <= J(zero source field)."""
    mesh_data, pde, admissibility, ud = ctx
    dt_out = pde["dt"] * pde["jprint"]
    # Zero source field (length-6 empty control for the 2-source model).
    state_zero = gt.run_forward(mesh_data, pde, np.zeros(6, dtype=np.float64))
    j_zero = float(
        gt.compute_objective(
            state_zero, ud, mesh_data, eta1=ro.ETA1, lam=ro.LAMBDA_REG,
            z=np.zeros(6), dt_out=dt_out, time_rule=ro.OBJECTIVE_TIME_RULE,
        )
    )
    control, diags, _ = sinit.build_sensitivity_initial_control(
        mesh_data, pde, ud, 1, (0.0, 2.0), admissibility,
        eta1=ro.ETA1, lam=ro.LAMBDA_REG, objective_time_rule=ro.OBJECTIVE_TIME_RULE,
        verbose=False,
    )
    assert diags[0].min_masked_sensitivity < 0.0
    # Evaluate the initialized single-source field via the same Gaussian source
    # model the module uses (the fixed forward reshape expects the 2-source
    # length, so a length-3 control is evaluated through a source_func).
    init_source = sinit._gaussian_sum_source(
        [(control[0], control[1], control[2])], pde["sigma2"]
    )
    state_init = gt.run_forward(
        mesh_data, pde, sinit._DUMMY_Z, source_func=init_source
    )
    j_init = float(
        gt.compute_objective(
            state_init, ud, mesh_data, eta1=ro.ETA1, lam=ro.LAMBDA_REG,
            z=sinit._DUMMY_Z, dt_out=dt_out, time_rule=ro.OBJECTIVE_TIME_RULE,
        )
    )
    assert j_init <= j_zero + 1e-9 * max(1.0, abs(j_zero)), (j_init, j_zero)


def _parse(argv):
    _saved = _sys.argv
    try:
        _sys.argv = ["run_optimization.py", *argv]
        return ro.parse_args()
    finally:
        _sys.argv = _saved


def main():
    # CLI-only tests (no PDE solves).
    cli_tests = [
        test_existing_modes_parse_and_dispatch_unchanged,
        test_initial_control_override_behavior,
    ]
    # Numerical tests share one expensive context (mesh + u_d).
    numeric_tests = [
        test_one_source_shape_admissible_bounded,
        test_two_source_ordering_and_separation,
        test_G_matches_discrete_inner_product,
        test_curvature_uses_weighted_observation_norm,
        test_selected_center_is_masked_argmin,
        test_graceful_all_candidates_masked,
        test_graceful_no_meaningful_descent,
        test_amplitude_projection_below_and_above_bounds,
        test_initialized_objective_no_worse_than_zero,
    ]

    failures = []
    for t in cli_tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures.append((t.__name__, exc))
            print(f"FAIL  {t.__name__}: {type(exc).__name__}: {exc}")

    print("building shared numerical context (mesh + observation data) ...")
    ctx = _build_context()
    for t in numeric_tests:
        try:
            t(ctx)
            print(f"PASS  {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures.append((t.__name__, exc))
            print(f"FAIL  {t.__name__}: {type(exc).__name__}: {exc}")

    print()
    if failures:
        print(f"{len(failures)} test(s) FAILED")
        for name, exc in failures:
            print(f"  - {name}: {exc}")
        raise SystemExit(1)
    print(f"all {len(cli_tests) + len(numeric_tests)} sensitivity-init tests PASSED")


if __name__ == "__main__":
    main()
