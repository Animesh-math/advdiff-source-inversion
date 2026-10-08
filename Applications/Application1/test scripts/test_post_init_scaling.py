"""Focused tests for --post-init-scaling (gn-diag / on-stall) and maxcor.

What is proved here:

1. The new flags parse, default to the historical behaviour (scaling 'none',
   maxcor not passed), and reject bad values.
2. ``run_lbfgsb_polish(scaling="none")`` is exactly the plain ``Optimizer``
   L-BFGS-B run (same iterates, bit for bit).
3. On a badly scaled analytic quadratic, ``gn-diag`` reaches the minimizer in
   far fewer iterations than the unscaled run; results come back in the
   original variables, bounds are honoured, and the callback sees original
   variables.
4. ``on-stall`` returns the unscaled result unchanged when it converges before
   the threshold, and switches (counts spanning both phases) when it does not.
5. ``gn_diagonal_scaling`` floors D at max(D)/max_ratio.
6. ``hessian_utils.hessian_vector_product`` accepts an M = 3 control (the old
   guard only accepted M = 2).
7. Short driver runs (tfinal 6): gn-diag and on-stall exit 0 and record the
   scaling block; the default run records scaling 'none'; Newton methods reject
   the scaling flag.

Usage:
    PY=python3
    cd "Applications/Application1/test scripts"
    "$PY" test_post_init_scaling.py
"""

from __future__ import annotations

import json
import os as _os
import sys as _sys

import numpy as np

_APP1_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _APP1_DIR not in _sys.path:
    _sys.path.insert(0, _APP1_DIR)
_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)

import run_optimization as app1  # noqa: E402
import post_init_optimizers as pio  # noqa: E402
from test_post_init_method_flag import run_case  # noqa: E402

_Optimizer = app1.Optimizer

FAILURES: list[str] = []

# Badly scaled convex quadratic: curvatures 1 .. 1e6, like positions vs amplitudes.
CURV = np.array([1.0, 30.0, 4.0e5, 3.0, 60.0, 1.0e6])
XSTAR = np.array([100.0, 210.0, 1.3, 270.0, 60.0, 0.85])
X0 = XSTAR + np.array([0.3, -0.2, 0.01, 0.25, 0.1, -0.005])
BOUNDS = [(0.0, 400.0), (0.0, 700.0), (0.1, 2.0)] * 2
OPTIONS = {"ftol": 1.0e-14, "gtol": 1.0e-9, "maxls": 20}


def check(condition, message):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {message}")
    if not condition:
        FAILURES.append(message)
    return bool(condition)


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def quad_obj(z, *_):
    d = np.asarray(z) - XSTAR
    return float(0.5 * np.sum(CURV * d * d))


def quad_grad(z, *_):
    return CURV * (np.asarray(z) - XSTAR)


def quad_hess(_z):
    return np.diag(CURV)


def polish(scaling, max_iter=200, stall_iters=pio.DEFAULT_STALL_ITERS, record=None):
    seen = [] if record is None else record
    return pio.run_lbfgsb_polish(
        optimizer_cls=_Optimizer,
        objective=quad_obj,
        gradient=quad_grad,
        x0=X0,
        bounds=BOUNDS,
        max_iter=max_iter,
        tol=1.0e-8,
        options=dict(OPTIONS),
        callback=lambda z: seen.append(np.asarray(z, dtype=float).copy()),
        scaling=scaling,
        stall_iters=stall_iters,
        gn_hess=None if scaling == pio.SCALING_NONE else quad_hess,
    )


def test_argument_parsing():
    section("1. flag parsing and defaults")
    argv = _sys.argv
    try:
        _sys.argv = ["run_optimization.py", "--mode", "location_and_intensity"]
        args = app1.parse_args()
        check(args.post_init_scaling == "none", "default --post-init-scaling is 'none'")
        check(args.post_init_lbfgsb_maxcor is None, "default maxcor is not passed (None)")
        check(args.post_init_stall_iters == 150, "default --post-init-stall-iters is 150")
        for value in pio.POST_INIT_SCALINGS:
            _sys.argv = ["run_optimization.py", "--mode", "location_and_intensity",
                         "--post-init-scaling", value]
            check(app1.parse_args().post_init_scaling == value, f"'{value}' parses")
        _sys.argv = ["run_optimization.py", "--mode", "location_and_intensity",
                     "--post-init-scaling", "diag"]
        rejected = False
        try:
            app1.parse_args()
        except SystemExit:
            rejected = True
        check(rejected, "an unknown --post-init-scaling is rejected")
    finally:
        _sys.argv = argv


def test_none_is_plain_optimizer():
    section("2. scaling 'none' == plain Optimizer L-BFGS-B, bit for bit")
    plain_seen = []
    plain = _Optimizer(
        objective_function=quad_obj, gradient_function=quad_grad,
        optimization_method="L-BFGS-B", max_iter=200, tol=1.0e-8, bounds=BOUNDS,
        callback=lambda z: plain_seen.append(np.asarray(z, dtype=float).copy()),
        options=dict(OPTIONS),
    ).optimize(initial_guess=X0)
    seen = []
    result, record = polish("none", record=seen)
    check(record == {"scaling": "none"}, "record is just {'scaling': 'none'}")
    check(np.array_equal(result.x, plain.x) and result.nit == plain.nit,
          f"same final x and iteration count ({result.nit})")
    check(len(seen) == len(plain_seen) and all(np.array_equal(a, b)
                                                 for a, b in zip(seen, plain_seen)),
          "identical iterate sequence")


def test_gn_diag():
    global XSTAR
    section("3. gn-diag on a badly scaled quadratic")
    base, _ = polish("none")
    seen = []
    result, record = polish("gn-diag", record=seen)
    err = float(np.max(np.abs(result.x - XSTAR)))
    print(f"  unscaled: {base.nit} its ({base.message}); gn-diag: {result.nit} its, "
          f"max err {err:.2e}; cond {record['condition_number_unscaled']:.2e} -> "
          f"{record['condition_number_scaled']:.2e}")
    check(err < 1.0e-6, "gn-diag reaches the minimizer (result.x in original variables)")
    check(result.nit < base.nit, f"gn-diag needs fewer iterations ({result.nit} < {base.nit})")
    check(abs(record["condition_number_scaled"] - 1.0) < 1e-9,
          "diagonal Hessian scales to condition number 1")
    check(np.allclose(record["scale_vector"], np.sqrt(CURV)), "D = sqrt(diag H)")
    check(all(np.max(np.abs(z - XSTAR)) < 1.0 for z in seen),
          "callback receives original-variable iterates")
    check(np.allclose(result.jac, quad_grad(result.x), atol=1e-6),
          "result.jac is mapped back to the original gradient")

    # Active bound in original units: minimizer of A1 outside [0.1, 2.0] -> clamps.
    saved = XSTAR.copy()
    try:
        XSTAR = saved.copy(); XSTAR[2] = 0.05
        result_b, _ = polish("gn-diag")
        check(abs(result_b.x[2] - 0.1) < 1e-12, f"scaled bounds honoured (A1 -> {result_b.x[2]})")
    finally:
        XSTAR = saved


def test_on_stall():
    section("4. on-stall")
    base, _ = polish("none")
    result, record = polish("on-stall", stall_iters=base.nit + 5)
    check(record["switched"] is False, "converging before the threshold does not switch")
    check(np.array_equal(result.x, base.x) and result.nit == base.nit,
          "and returns the unscaled result unchanged")
    seen = []
    result, record = polish("on-stall", stall_iters=3, record=seen)
    check(record["switched"] is True and record["switch_iteration"] == 3,
          "switches after 3 unscaled iterations")
    check(result.nit == 3 + (len(seen) - 3), f"nit spans both phases ({result.nit})")
    check(float(np.max(np.abs(result.x - XSTAR))) < 1e-6, "and converges after switching")
    result, record = polish("on-stall", max_iter=10, stall_iters=10)
    check(record["switched"] is False and "note" in record,
          "stall_iters >= max_iter means unscaled only")


def test_floor():
    section("5. gn_diagonal_scaling floor")
    scale, record = pio.gn_diagonal_scaling(lambda _z: np.diag([1.0e8, 1.0e-6, 4.0]),
                                            np.zeros(3), max_ratio=1.0e3)
    check(np.allclose(scale, [1.0e4, 10.0, 10.0]), f"D floored at max/ratio: {scale}")
    check(record["n_floored"] == 2, "two entries floored")


def test_hessian_guard():
    section("6. hessian_utils accepts M = 3 controls")
    import hessian_utils as hu
    rejected_bad = False
    try:
        hu.hessian_vector_product(np.zeros(7), np.zeros(7), None, {}, None)
    except ValueError:
        rejected_bad = True
    check(rejected_bad, "length 7 (not 3M) is rejected")
    # length 9 passes the guard and fails only later (mesh_data=None), not on length
    try:
        hu.hessian_vector_product(np.zeros(9), np.zeros(9), None, {}, None)
        passed_guard = True
    except ValueError as exc:
        passed_guard = "length" not in str(exc)
    except Exception:  # noqa: BLE001 - anything past the guard
        passed_guard = True
    check(passed_guard, "length 9 (M = 3) passes the length guard")


def test_driver_runs():
    section("7. short driver runs")
    for name, extra in (
        ("test_post_init_scaling_gn_diag", ["--post-init-scaling", "gn-diag",
                                            "--post-init-lbfgsb-maxcor", "30"]),
        ("test_post_init_scaling_on_stall", ["--post-init-scaling", "on-stall",
                                             "--post-init-stall-iters", "1"]),
        ("test_post_init_scaling_default", []),
    ):
        completed, out = run_case(name, extra, max_iter="3")
        if not check(completed.returncode == 0, f"{name} exits 0"):
            continue
        summary = json.loads((out / "optimization_summary.json").read_text())
        post = summary["post_init"]
        stage = post["stages"][0]
        expected = extra[1] if extra else "none"
        check(post["scaling"] == expected and stage["scaling"]["scaling"] == expected,
              f"{name}: summary records scaling {expected!r}")
        if expected == "gn-diag":
            check(stage["lbfgsb_maxcor"] == 30 and "condition_number_scaled" in stage["scaling"],
                  "gn-diag: maxcor 30 and the conditioning record are stored")
            check("[PostInit] gn-diag scaling" in completed.stdout, "gn-diag: scaling logged")
        if expected == "on-stall":
            rec = stage["scaling"]
            check("switched" in rec, f"on-stall: switch decision recorded ({rec.get('switched')})")
        if expected == "none":
            check("[PostInit]" not in completed.stdout, "default: no scaling branch entered")
            check(stage["lbfgsb_maxcor"] == 10, "default: maxcor recorded as SciPy's 10")

    completed, _ = run_case("test_post_init_scaling_newton_reject",
                            ["--post-init-method", "gauss_newton",
                             "--post-init-scaling", "gn-diag"], max_iter="1")
    check(completed.returncode != 0 and "apply only to" in (completed.stderr + completed.stdout),
          "gauss_newton + --post-init-scaling is rejected")


def main():
    test_argument_parsing()
    test_none_is_plain_optimizer()
    test_gn_diag()
    test_on_stall()
    test_floor()
    test_hessian_guard()
    test_driver_runs()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for failure in FAILURES:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
