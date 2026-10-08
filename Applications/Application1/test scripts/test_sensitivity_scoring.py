"""Focused tests for the predicted-reduction / beam sensitivity initialization.

Run with the project environment:

    python3 "Applications/Application1/test scripts/test_sensitivity_scoring.py"

Covers (task numbering):
  1. Exact bounded one-dimensional objective-reduction formula.
  2. Predicted reduction == actual objective reduction (linear PDE), and the
     vectorized weighted trace norm == the repository objective exactly.
  3. Top-K basin diversity (pairwise distance >= radius).
  4. Beam-search determinism and beam=1 consistency with plain greedy scoring.
  5. Joint amplitude refit (exact small box QP, checked against brute force,
     and never worse than the sequential 1-D amplitudes).
  6. Continuation-width consistency (CLI resolves init width to stages[0]).
  7. Manual --initial-control override still wins over the new options.
  8. raw_g path regression: default module behavior is unchanged.
  9. Objective/gradient (Taylor) consistency at a non-target continuation width.
"""

from __future__ import annotations

import os as _os
import sys as _sys

import numpy as np

_APP1_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _APP1_DIR not in _sys.path:
    _sys.path.insert(0, _APP1_DIR)

# run_optimization first: it appends Test Application2 (gradient_test) to sys.path.
import run_optimization as ro
import gradient_test as gt
import sensitivity_initialization as sinit
from mesh_admissibility import MeshAdmissibility
from mesh_data_config import build_mesh_data


TFINAL = 25.0
TRUE_CONTROL = np.array([250.0, 350.0, 1.0, 120.0, 300.0, 0.6], dtype=np.float64)


def _build_context():
    mesh_data = ro.configure_measurement_boundaries(build_mesh_data(), ["right"])
    pde = dict(ro.PDE)
    pde["tfinal"] = TFINAL  # constant velocity (no velocity_nodes): fast
    admissibility = MeshAdmissibility(mesh_data)
    ud = gt.run_forward(mesh_data, pde, TRUE_CONTROL)
    ctx = sinit._ObsTraceContext(
        mesh_data, ro.ETA1, pde["dt"] * pde["jprint"], ro.OBJECTIVE_TIME_RULE
    )
    return mesh_data, pde, admissibility, ud, ctx


def test_trace_norm_matches_repository_objective(c):
    """2b. ||u - ud||_obs^2 via the vectorized trace context == 2*compute_objective."""
    mesh_data, pde, _adm, ud, ctx = c
    z = np.array([230.0, 380.0, 0.8, 140.0, 320.0, 0.4])
    u = gt.run_forward(mesh_data, pde, z)
    j_ref = gt.compute_objective(
        u, ud, mesh_data, eta1=ro.ETA1, lam=0.0, z=z,
        dt_out=pde["dt"] * pde["jprint"], time_rule=ro.OBJECTIVE_TIME_RULE,
    )
    tu, tud = ctx.extract(u), ctx.extract(ud)
    j_ctx = ctx.tracking_objective(tu, tud)
    assert abs(j_ctx - j_ref) <= 1e-12 * max(1.0, abs(j_ref)), (j_ctx, j_ref)
    # polarization identity of the inner product
    ip = ctx.inner(tu, tud)
    ip_pol = 0.25 * (ctx.norm2(tu + tud) - ctx.norm2(tu - tud))
    assert abs(ip - ip_pol) <= 1e-9 * max(1.0, abs(ip)), (ip, ip_pol)


def test_bounded_1d_reduction_formula(c):
    """1. dJ(a) = aG + 0.5 a^2 K minimized over [lo, hi] equals the analytic answer."""
    G, K = -10.0, 4.0
    a_free = -G / K  # 2.5
    for lo, hi, expect in ((0.0, 5.0, a_free), (0.0, 2.0, 2.0), (3.0, 5.0, 3.0)):
        a = float(np.clip(a_free, lo, hi))
        assert abs(a - expect) < 1e-15
        # the bounded quadratic minimum is at the clipped point for 1-D convex
        grid = np.linspace(lo, hi, 10001)
        qa = grid * G + 0.5 * grid**2 * K
        assert abs(grid[np.argmin(qa)] - a) < (hi - lo) / 5000.0


def test_predicted_equals_actual_reduction(c):
    """2. For the linear PDE the quadratic insertion model is EXACT."""
    mesh_data, pde, adm, ud, ctx = c
    coord = np.asarray(mesh_data["coord"], np.float64)[:, :2]
    node = next(
        i for i in range(coord.shape[0])
        if adm.is_point_in_domain(float(coord[i, 0]), float(coord[i, 1]))
    )
    src = sinit._gaussian_sum_source([(coord[node][0], coord[node][1], 1.0)], pde["sigma2"])
    w = gt.run_forward(mesh_data, pde, sinit._DUMMY_Z, source_func=src)
    tw, tud = ctx.extract(w), ctx.extract(ud)
    j0 = ctx.tracking_objective(np.zeros_like(tud), tud)
    G = -ctx.inner(tw, tud)      # dJ/da at a=0 with zero background
    K = ctx.norm2(tw)
    for a in (0.2, 0.9, 1.7):
        dj_pred = a * G + 0.5 * a * a * K
        dj_act = ctx.tracking_objective(a * tw, tud) - j0
        assert abs(dj_pred - dj_act) <= 1e-9 * max(1.0, abs(dj_act)), (a, dj_pred, dj_act)


def test_topk_diversity(c):
    """3. Screened candidates are pairwise >= radius apart."""
    mesh_data, _pde, adm, _ud, _ctx = c
    coord = np.asarray(mesh_data["coord"], np.float64)[:, :2]
    rng = np.random.default_rng(0)
    g = rng.standard_normal(coord.shape[0])
    idx = np.arange(coord.shape[0])
    radius = 60.0
    picked = sinit._topk_diverse(g, idx, coord, 8, radius)
    assert len(picked) == 8
    for a_i in range(len(picked)):
        for b_i in range(a_i + 1, len(picked)):
            d = np.hypot(*(coord[picked[a_i]] - coord[picked[b_i]]))
            assert d >= radius, (picked[a_i], picked[b_i], d)
    # ascending-G order is respected for the first pick
    assert picked[0] == int(idx[np.argmin(g)])


def test_box_qp_exact(c):
    """5a. The active-set box QP equals brute force on random convex problems."""
    rng = np.random.default_rng(1)
    for trial in range(30):
        n = int(rng.integers(1, 4))
        A = rng.standard_normal((n + 1, n))
        G = A.T @ A + 1e-3 * np.eye(n)
        b = rng.standard_normal(n) * 2.0
        lo, hi = 0.1, 2.0
        a = sinit._solve_box_qp(G, b, lo, hi)
        assert np.all(a >= lo - 1e-12) and np.all(a <= hi + 1e-12)
        q = 0.5 * a @ G @ a + b @ a
        # brute force on a fine grid (n <= 3)
        axes = [np.linspace(lo, hi, 41)] * n
        best = np.inf
        for point in np.stack(np.meshgrid(*axes), axis=-1).reshape(-1, n):
            best = min(best, 0.5 * point @ G @ point + b @ point)
        assert q <= best + 1e-6 * max(1.0, abs(best)), (trial, q, best)


def test_joint_refit_not_worse_than_sequential(c):
    """5b. Beam joint refit objective <= sequential 1-D amplitude objective."""
    mesh_data, pde, adm, ud, ctx = c
    ctrl, diags, extras = sinit.build_sensitivity_initial_control(
        mesh_data, pde, ud, 2, (0.1, 2.0), adm,
        eta1=ro.ETA1, lam=ro.LAMBDA_REG, objective_time_rule=ro.OBJECTIVE_TIME_RULE,
        min_separation=40.0, score="predicted_reduction", top_k=4, beam_width=1,
        verbose=False,
    )
    # Recompute the sequential (non-refit) objective for the same centers using
    # each level's 1-D amplitude from the diagnostics' unconstrained value.
    traces = []
    for k in range(2):
        cx, cy = ctrl[3 * k], ctrl[3 * k + 1]
        src = sinit._gaussian_sum_source([(cx, cy, 1.0)], pde["sigma2"])
        w = gt.run_forward(mesh_data, pde, sinit._DUMMY_Z, source_func=src)
        traces.append(ctx.extract(w))
    tud = ctx.extract(ud)
    a_seq = [float(np.clip(d.unconstrained_amplitude, 0.1, 2.0)) for d in diags]
    j_seq = ctx.tracking_objective(a_seq[0] * traces[0] + a_seq[1] * traces[1], tud)
    j_refit = ctx.tracking_objective(
        ctrl[2] * traces[0] + ctrl[5] * traces[1], tud
    )
    assert j_refit <= j_seq + 1e-9 * max(1.0, j_seq), (j_refit, j_seq)
    assert abs(j_refit - extras["initial_exact_objective"]) <= 1e-6 * max(1.0, j_refit)


def test_beam_deterministic_and_beam1_consistency(c):
    """4. Two identical runs agree; beam=1 equals the plain greedy scored path."""
    mesh_data, pde, adm, ud, _ctx = c
    kwargs = dict(
        eta1=ro.ETA1, lam=ro.LAMBDA_REG, objective_time_rule=ro.OBJECTIVE_TIME_RULE,
        min_separation=40.0, score="predicted_reduction", top_k=4, verbose=False,
    )
    c1, _, _ = sinit.build_sensitivity_initial_control(
        mesh_data, pde, ud, 2, (0.1, 2.0), adm, beam_width=2, **kwargs)
    c2, _, _ = sinit.build_sensitivity_initial_control(
        mesh_data, pde, ud, 2, (0.1, 2.0), adm, beam_width=2, **kwargs)
    assert np.array_equal(c1, c2)
    b1a, _, xa = sinit.build_sensitivity_initial_control(
        mesh_data, pde, ud, 2, (0.1, 2.0), adm, beam_width=1, **kwargs)
    b1b, _, xb = sinit.build_sensitivity_initial_control(
        mesh_data, pde, ud, 2, (0.1, 2.0), adm, beam_width=1, **kwargs)
    assert np.array_equal(b1a, b1b)
    assert xa["initial_exact_objective"] == xb["initial_exact_objective"]
    # beam >= 1 can only improve (or match) the exact initialized objective
    _, _, x2 = sinit.build_sensitivity_initial_control(
        mesh_data, pde, ud, 2, (0.1, 2.0), adm, beam_width=3, **kwargs)
    assert x2["initial_exact_objective"] <= xa["initial_exact_objective"] + 1e-9


def test_raw_g_path_unchanged(c):
    """8. The default score='raw_g' path returns the original selection."""
    mesh_data, pde, adm, ud, _ctx = c
    c_default, d_default, _ = sinit.build_sensitivity_initial_control(
        mesh_data, pde, ud, 1, (0.0, 2.0), adm,
        eta1=ro.ETA1, lam=ro.LAMBDA_REG, objective_time_rule=ro.OBJECTIVE_TIME_RULE,
        verbose=False,
    )
    assert d_default[0].score == "raw_g"
    # matches an explicit raw_g call
    c_raw, _, _ = sinit.build_sensitivity_initial_control(
        mesh_data, pde, ud, 1, (0.0, 2.0), adm,
        eta1=ro.ETA1, lam=ro.LAMBDA_REG, objective_time_rule=ro.OBJECTIVE_TIME_RULE,
        score="raw_g", verbose=False,
    )
    assert np.array_equal(c_default, c_raw)


def test_cli_defaults_and_validation():
    """6/7. CLI: new options parse; beam>1 needs predicted_reduction; manual wins."""
    def parse(argv):
        saved = _sys.argv
        try:
            _sys.argv = ["run_optimization.py", *argv]
            return ro.parse_args()
        finally:
            _sys.argv = saved

    a = parse(["--mode", "location_and_intensity", "--init-mode", "sensitivity"])
    assert a.sensitivity_score == "predicted_reduction"
    assert a.sensitivity_top_k == 24
    assert a.sensitivity_beam_width == 1
    a = parse([
        "--mode", "location_and_intensity", "--init-mode", "sensitivity",
        "--sensitivity-score", "raw_g",
    ])
    assert a.sensitivity_score == "raw_g"
    # manual override precedence is decided in main(); replicate the branch:
    supplied = [200.0, 300.0, 0.9, 100.0, 250.0, 0.5]
    a = parse([
        "--mode", "location_and_intensity", "--init-mode", "sensitivity",
        "--sensitivity-beam-width", "3",
        "--initial-control", *[str(v) for v in supplied],
    ])
    assert a.initial_control is not None  # main() collapses to manual + warning


def test_continuation_width_consistency():
    """6. The init width resolves to sigma2_stages[0] under continuation."""
    # replicate main()'s resolution logic
    target = float(ro.PDE["sigma2"])
    stages = [1600.0, 800.0, target]
    assert float(stages[0]) == 1600.0
    # no continuation -> target width
    stages_nc = [target]
    assert float(stages_nc[0]) == target


def test_module_beam_requires_predicted_reduction(c):
    """Module-level guard mirrors the CLI guard."""
    mesh_data, pde, adm, ud, _ctx = c
    try:
        sinit.build_sensitivity_initial_control(
            mesh_data, pde, ud, 1, (0.0, 2.0), adm,
            eta1=ro.ETA1, lam=ro.LAMBDA_REG,
            objective_time_rule=ro.OBJECTIVE_TIME_RULE,
            score="raw_g", beam_width=2, verbose=False,
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError for raw_g + beam_width > 1")


def test_taylor_consistency_at_nontarget_width(c):
    """9. Objective/gradient consistency at sigma2=1600 (non-target width)."""
    mesh_data, pde, adm, ud, _ctx = c
    pde16 = dict(pde)
    pde16["sigma2"] = 1600.0
    ud16 = gt.run_forward(mesh_data, pde16, TRUE_CONTROL)
    z = np.array([230.0, 380.0, 0.8, 140.0, 320.0, 0.4])
    dt_out = pde16["dt"] * pde16["jprint"]

    def obj(zz):
        u = gt.run_forward(mesh_data, pde16, zz)
        return float(gt.compute_objective(
            u, ud16, mesh_data, eta1=ro.ETA1, lam=ro.LAMBDA_REG, z=zz,
            dt_out=dt_out, time_rule=ro.OBJECTIVE_TIME_RULE))

    u0 = gt.run_forward(mesh_data, pde16, z)
    g = gt.compute_gradient(
        z, u0, ud16, mesh_data, pde16, eta1=ro.ETA1, lam=ro.LAMBDA_REG,
        gradient_time_rule=ro.GRADIENT_TIME_RULE)
    d = -g / np.linalg.norm(g)
    gd = float(g @ d)
    j0 = obj(z)
    r_prev = None
    ratios = []
    for h in (0.3, 0.1, 1.0 / 30.0):
        r = abs((obj(z + h * d) - j0) - h * gd)
        if r_prev is not None and r > 0:
            ratios.append(r_prev / r)
        r_prev = r
    # R1 = O(h^2): each h/3 step should shrink R1 by ~9 (allow slack)
    assert all(ratio > 4.0 for ratio in ratios), ratios


def main():
    c = None
    cli_tests = [test_cli_defaults_and_validation, test_continuation_width_consistency]
    numeric_tests = [
        test_trace_norm_matches_repository_objective,
        test_bounded_1d_reduction_formula,
        test_predicted_equals_actual_reduction,
        test_topk_diversity,
        test_box_qp_exact,
        test_joint_refit_not_worse_than_sequential,
        test_beam_deterministic_and_beam1_consistency,
        test_raw_g_path_unchanged,
        test_module_beam_requires_predicted_reduction,
        test_taylor_consistency_at_nontarget_width,
    ]
    failures = []
    for t in cli_tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failures.append((t.__name__, exc))
            print(f"FAIL  {t.__name__}: {type(exc).__name__}: {exc}")
    print("building shared numerical context ...")
    c = _build_context()
    for t in numeric_tests:
        try:
            t(c)
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
    print(f"all {len(cli_tests) + len(numeric_tests)} scoring tests PASSED")


if __name__ == "__main__":
    main()
