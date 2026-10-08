"""Focused tests for --post-init-method and the ParaView sensor export.

What is proved here, without running a full optimization:

1. ``--post-init-method`` parses, defaults to ``lbfgsb``, and rejects garbage.
2. The DEFAULT is unchanged: with no flag, and with an explicit ``lbfgsb``,
   ``run_single_optimization`` takes the historical ``Optimizer``/L-BFGS-B
   branch and never touches ``hessian_utils``.
3. ``gauss_newton`` / ``newton`` take the Newton-type branch, call
   ``hessian_utils.hessian_matrix`` with the matching Hessian mode
   (``gauss_newton`` / ``exact``), and record their diagnostics.
4. Every existing summary is still written, plus ``post_init_method`` in
   ``optimization_summary.json`` and ``post_init_summary.json`` for the
   Newton-type methods.
5. ``sensors.vtu`` is written for node-sensor runs, is readable by meshio,
   carries one point per sensor with the documented fields, and matches the
   sensor metadata already stored in ``optimization_summary.json``.
6. ``active_index_map`` agrees with ``pack_active`` for every mode, which is
   what makes the Hessian restriction correct in the non-full modes.
7. ``solve_counter`` counts exactly and restores the patched entry points.

The optimization runs use a SHORT ``--tfinal`` and a tiny iteration cap: the
point is the plumbing, not the physics (the physics comparison lives in
``diagnose_post_init_polish.py``).

Usage:
    PY="python3"
    cd "Applications/Application1/test scripts"
    "$PY" test_post_init_method_flag.py
"""

from __future__ import annotations

import json
import os as _os
import shutil
import subprocess
import sys as _sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_APP1_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _APP1_DIR not in _sys.path:
    _sys.path.insert(0, _APP1_DIR)

import run_optimization as app1
import post_init_optimizers as pio
from multi_source_utils import pack_active, restrict_gradient

APP_DIR = Path(_APP1_DIR)
PYTHON = _sys.executable

TRUE_CONTROL = ["162", "300", "1.40", "270", "600", "0.95"]
# The saved sensitivity initialization of node_sensors_right_bottom_top_c06.
INIT_CONTROL = [
    "270.0042837605424",
    "600.051691697479",
    "0.9510994778671144",
    "161.8662190981801",
    "300.00986383727934",
    "1.3940636135616904",
]
SENSOR_NODE_IDS = [
    "641,644,647,650,653,656,659,662,665,668,671,674,677",
    "582,585,588,591,594,596",
    "693,690,687,684,681,679",
]

FAILURES: list[str] = []


def check(condition, message):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {message}")
    if not condition:
        FAILURES.append(message)
    return bool(condition)


def section(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ---------------------------------------------------------------------------
# 1 / 6 / 7: pure-python unit checks
# ---------------------------------------------------------------------------
def test_argument_parsing():
    section("1. --post-init-method parsing and default")
    argv = _sys.argv
    try:
        _sys.argv = ["run_optimization.py", "--mode", "location_and_intensity"]
        args = app1.parse_args()
        check(
            args.post_init_method == pio.DEFAULT_POST_INIT_METHOD == "lbfgsb",
            f"default post_init_method is 'lbfgsb' (got {args.post_init_method!r})",
        )
        for method in ("lbfgsb", "gauss_newton", "newton"):
            _sys.argv = [
                "run_optimization.py",
                "--mode",
                "location_and_intensity",
                "--post-init-method",
                method,
            ]
            args = app1.parse_args()
            check(
                args.post_init_method == method,
                f"--post-init-method {method} parses to {method!r}",
            )
        _sys.argv = [
            "run_optimization.py",
            "--mode",
            "location_and_intensity",
            "--post-init-method",
            "bfgs_but_wrong",
        ]
        rejected = False
        try:
            app1.parse_args()
        except SystemExit:
            rejected = True
        check(rejected, "an unknown --post-init-method is rejected by argparse")
    finally:
        _sys.argv = argv

    rejected = False
    try:
        pio.validate_post_init_method("nope")
    except ValueError:
        rejected = True
    check(rejected, "validate_post_init_method rejects an unknown name")
    check(
        pio.is_newton_type("gauss_newton") and pio.is_newton_type("newton")
        and not pio.is_newton_type("lbfgsb"),
        "is_newton_type classifies the three methods correctly",
    )
    check(
        pio.HESSIAN_MODE_BY_METHOD["gauss_newton"] == "gauss_newton"
        and pio.HESSIAN_MODE_BY_METHOD["newton"] == "exact",
        "methods map to the hessian_utils modes 'gauss_newton' / 'exact'",
    )


def test_active_index_map():
    section("6. active_index_map matches pack_active for every mode")
    for num_sources in (1, 2, 3):
        z = np.arange(3 * num_sources, dtype=np.float64)
        for mode in ("location_only", "intensity_only", "location_and_intensity"):
            indices = pio.active_index_map(mode, num_sources)
            expected = pack_active(z, mode)
            check(
                np.array_equal(z[indices], expected),
                f"M={num_sources} mode={mode}: index map reproduces pack_active",
            )
            check(
                np.array_equal(z[indices], restrict_gradient(z, mode)),
                f"M={num_sources} mode={mode}: index map reproduces restrict_gradient",
            )


def test_solve_counter():
    section("7. solve_counter counts exactly and restores the entry points")

    # A namespace holding plain functions is the faithful analogue of the
    # gradient_test MODULE that solve_counter actually patches (identity of a
    # bound method is not stable, so a class instance would not test restoration).
    def _forward(*a, **k):
        return "fwd"

    def _adjoint(*a, **k):
        return "adj"

    fake = SimpleNamespace(
        run_forward=_forward, rk4_discrete_source_adjoint=_adjoint
    )
    original_forward = fake.run_forward
    original_adjoint = fake.rk4_discrete_source_adjoint
    with pio.solve_counter(fake) as counts:
        for _ in range(4):
            fake.run_forward()
        for _ in range(3):
            fake.rk4_discrete_source_adjoint()
        observed = dict(counts)
    check(observed == {"forward_solves": 4, "adjoint_solves": 3},
          f"counted 4 forward / 3 adjoint solves (got {observed})")
    check(
        fake.run_forward is original_forward
        and fake.rk4_discrete_source_adjoint is original_adjoint,
        "the patched entry points are restored on exit",
    )
    restored = {"forward_solves": 0, "adjoint_solves": 0}
    try:
        with pio.solve_counter(fake) as counts2:
            fake.run_forward()
            raise RuntimeError("boom")
    except RuntimeError:
        restored = dict(counts2)
    check(
        fake.run_forward is original_forward,
        "the entry points are restored even when the body raises",
    )
    check(restored["forward_solves"] == 1, "counts survive an exception")


# ---------------------------------------------------------------------------
# 2-5: end-to-end short runs
# ---------------------------------------------------------------------------
def run_case(case_name, extra_args, tfinal="6", max_iter="1"):
    out_dir = APP_DIR / "output" / case_name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    command = [
        PYTHON,
        str(APP_DIR / "run_optimization.py"),
        "--mode", "location_and_intensity",
        "--stage", "full",
        "--num-sources", "2",
        "--observation-mode", "nodes",
        "--sensor-weights", "equal",
        "--sensor-equal-weight", "1.0",
        "--sensor-node-ids", *SENSOR_NODE_IDS,
        "--measurement-boundaries", "right", "bottom", "top",
        "--feasibility-mode", "raw",
        "--amplitude-floor", "0.1",
        "--true-control", *TRUE_CONTROL,
        "--initial-control", *INIT_CONTROL,
        "--case-name", case_name,
        "--tfinal", tfinal,
        "--max-iter", max_iter,
        "--max-paraview-frames", "3",
        *extra_args,
    ]
    print("\n$ " + " ".join(f'"{c}"' if " " in c else c for c in command))
    completed = subprocess.run(
        command, cwd=str(APP_DIR), capture_output=True, text=True
    )
    if completed.returncode != 0:
        print(completed.stdout[-4000:])
        print(completed.stderr[-4000:])
    return completed, out_dir / "full"


def test_default_and_explicit_lbfgsb():
    section("2. default behaviour is unchanged (lbfgsb, no hessian_utils)")
    completed_default, dir_default = run_case("test_post_init_default", [])
    check(completed_default.returncode == 0, "default run (no flag) exits 0")
    completed_explicit, dir_explicit = run_case(
        "test_post_init_lbfgsb", ["--post-init-method", "lbfgsb"]
    )
    check(completed_explicit.returncode == 0, "explicit --post-init-method lbfgsb exits 0")

    for label, out in (("default", dir_default), ("explicit lbfgsb", dir_explicit)):
        summary_path = out / "optimization_summary.json"
        check(summary_path.exists(), f"{label}: optimization_summary.json written")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        check(
            summary["post_init_method"] == "lbfgsb",
            f"{label}: summary records post_init_method='lbfgsb'",
        )
        check(
            not (out / "post_init_summary.json").exists(),
            f"{label}: no post_init_summary.json for the lbfgsb default",
        )
        for name in (
            "optimization_history.csv",
            "source_fields.vtu",
        ):
            check((out / name).exists(), f"{label}: {name} still written")
        for name in ("true_state", "initial_state", "optimized_state", "source_paths"):
            check((out / name).is_dir(), f"{label}: {name}/ still written")

    check(
        "[Optimizer] method=L-BFGS-B" in completed_default.stdout,
        "default run went through the historical L-BFGS-B Optimizer",
    )
    check(
        "[PostInit]" not in completed_default.stdout,
        "default run never entered the Newton-type branch",
    )

    # Bit-for-bit: no flag and explicit lbfgsb must produce identical numerics.
    a = json.loads((dir_default / "optimization_summary.json").read_text())
    b = json.loads((dir_explicit / "optimization_summary.json").read_text())
    same = all(
        a[key] == b[key]
        for key in (
            "final_control",
            "final_objective",
            "initial_objective",
            "final_gradient_norm",
            "iterations",
            "message",
        )
    )
    check(same, "no-flag and explicit-lbfgsb runs agree bit-for-bit")
    return dir_default


def test_newton_type(method):
    section(f"3-4. --post-init-method {method}")
    completed, out = run_case(
        f"test_post_init_{method}", ["--post-init-method", method]
    )
    check(completed.returncode == 0, f"{method} run exits 0")
    if completed.returncode != 0:
        return

    check(
        "[PostInit] method=" + method in completed.stdout,
        f"{method}: the Newton-type branch ran",
    )
    check(
        "[Optimizer] method=L-BFGS-B" not in completed.stdout,
        f"{method}: L-BFGS-B was NOT used",
    )

    summary = json.loads((out / "optimization_summary.json").read_text())
    check(
        summary["post_init_method"] == method,
        f"{method}: optimization_summary.json records post_init_method",
    )
    check(
        summary["post_init"]["hessian_mode"] == pio.HESSIAN_MODE_BY_METHOD[method],
        f"{method}: summary records hessian_mode="
        f"{pio.HESSIAN_MODE_BY_METHOD[method]!r}",
    )
    check(
        summary["post_init"]["scipy_method"] == pio.NEWTON_SCIPY_METHOD,
        f"{method}: summary records scipy_method={pio.NEWTON_SCIPY_METHOD!r}",
    )

    post_init_path = out / "post_init_summary.json"
    check(post_init_path.exists(), f"{method}: post_init_summary.json written")
    if not post_init_path.exists():
        return
    post_init = json.loads(post_init_path.read_text())
    check(
        post_init["total_hessian_evaluations"] >= 1,
        f"{method}: at least one Hessian was assembled "
        f"({post_init['total_hessian_evaluations']})",
    )
    check(
        post_init["total_forward_solves"] > 0 and post_init["total_adjoint_solves"] > 0,
        f"{method}: solve counts recorded "
        f"({post_init['total_forward_solves']}f/"
        f"{post_init['total_adjoint_solves']}a)",
    )
    hessians = post_init["stages"][0]["hessian_evaluations"]
    check(len(hessians) >= 1, f"{method}: per-Hessian diagnostics persisted")
    if hessians:
        record = hessians[0]
        check(
            record["hessian_mode"] == pio.HESSIAN_MODE_BY_METHOD[method],
            f"{method}: hessian diagnostics tagged with the right mode",
        )
        check(
            len(record["eigenvalues"]) == 6,
            f"{method}: 6x6 active Hessian eigenvalues recorded "
            f"(got {len(record['eigenvalues'])})",
        )
    for name in ("optimization_history.csv", "source_fields.vtu"):
        check((out / name).exists(), f"{method}: {name} still written")


def test_sensor_vtu(out_dir):
    section("5. sensors.vtu ParaView point cloud")
    import meshio

    path = out_dir / "sensors.vtu"
    check(path.exists(), f"sensors.vtu written to {path}")
    if not path.exists():
        return

    summary = json.loads((out_dir / "optimization_summary.json").read_text())
    observation = summary["observation"]
    expected_ids = observation["sensor_node_ids"]
    expected_xy = observation["sensor_coordinates"]
    expected_w = observation["sensor_weights"]

    mesh = meshio.read(path)
    check(
        mesh.points.shape[0] == len(expected_ids),
        f"one point per sensor ({mesh.points.shape[0]} vs {len(expected_ids)})",
    )
    cell_types = {block.type for block in mesh.cells}
    check(cell_types == {"vertex"}, f"cells are vertex cells (got {cell_types})")

    for field in ("sensor_id", "node_id", "weight", "row_sum_weight", "x", "y"):
        check(field in mesh.point_data, f"point data has '{field}'")

    check(
        np.array_equal(
            mesh.point_data["node_id"].astype(int), np.asarray(expected_ids, dtype=int)
        ),
        "node_id field matches optimization_summary.json sensor_node_ids (same order)",
    )
    check(
        np.array_equal(
            mesh.point_data["sensor_id"].astype(int), np.arange(len(expected_ids))
        ),
        "sensor_id field is 0..Ns-1 in sensor order",
    )
    xy = np.asarray(expected_xy, dtype=np.float64)
    check(
        np.allclose(mesh.points[:, :2], xy, rtol=0, atol=0),
        "point coordinates match the summary sensor_coordinates exactly",
    )
    check(
        np.allclose(mesh.point_data["x"], xy[:, 0], rtol=0, atol=0)
        and np.allclose(mesh.point_data["y"], xy[:, 1], rtol=0, atol=0),
        "x / y fields match the point coordinates",
    )
    check(
        np.allclose(mesh.point_data["weight"], np.asarray(expected_w), rtol=0, atol=0),
        "weight field matches the summary sensor_weights",
    )
    check(
        np.allclose(mesh.points[:, 2], 0.0),
        "points are written in the z=0 plane of the 2-D mesh",
    )
    check(
        observation["observation_mode"] == "nodes"
        and observation["num_sensors"] == len(expected_ids),
        "sensor metadata is (still) in optimization_summary.json",
    )


def test_boundary_mode_has_no_sensor_file():
    section("5b. boundary observation mode writes no sensors.vtu")
    case = "test_post_init_boundary_no_sensors"
    out_dir = APP_DIR / "output" / case
    if out_dir.exists():
        shutil.rmtree(out_dir)
    command = [
        PYTHON,
        str(APP_DIR / "run_optimization.py"),
        "--mode", "location_and_intensity",
        "--stage", "full",
        "--num-sources", "2",
        "--measurement-boundaries", "right", "bottom", "top",
        "--feasibility-mode", "raw",
        "--amplitude-floor", "0.1",
        "--true-control", *TRUE_CONTROL,
        "--initial-control", *INIT_CONTROL,
        "--case-name", case,
        "--tfinal", "6",
        "--max-iter", "1",
        "--max-paraview-frames", "3",
    ]
    print("\n$ " + " ".join(command))
    completed = subprocess.run(command, cwd=str(APP_DIR), capture_output=True, text=True)
    if completed.returncode != 0:
        print(completed.stdout[-3000:])
        print(completed.stderr[-3000:])
    check(completed.returncode == 0, "boundary-mode run exits 0")
    full = out_dir / "full"
    check(
        not (full / "sensors.vtu").exists(),
        "no sensors.vtu in the default boundary observation mode",
    )
    if (full / "optimization_summary.json").exists():
        summary = json.loads((full / "optimization_summary.json").read_text())
        check(
            summary["observation"] == {"observation_mode": "boundary"},
            "boundary-mode summary observation record unchanged",
        )
        check(
            summary["post_init_method"] == "lbfgsb",
            "boundary-mode run still defaults to lbfgsb",
        )


def main():
    test_argument_parsing()
    test_active_index_map()
    test_solve_counter()
    default_dir = test_default_and_explicit_lbfgsb()
    test_sensor_vtu(default_dir)
    test_newton_type("gauss_newton")
    test_newton_type("newton")
    test_boundary_mode_has_no_sensor_file()

    section("RESULT")
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED:")
        for message in FAILURES:
            print(f"  - {message}")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
