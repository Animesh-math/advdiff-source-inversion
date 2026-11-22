"""Prove the default boundary observation mode is numerically unchanged.

The node-sensor extension touched two shared files on the objective/gradient
path: ``Src/Objective/gradient_test.py`` (formerly
``Applications/Test Application2/gradient_test.py``) and
``Src/Solvers/adv_diffusion_adjoint.py``.  Both were extended with an
``observation is None`` branch that must leave the boundary path untouched.

This test materializes the PRE-CHANGE versions of those two files from a git
revision (default ``HEAD``, where neither file had been modified), loads them
side by side with the current ones, and compares the reduced objective and the
reduced adjoint gradient at fixed controls in boundary mode.  They must agree
BIT-FOR-BIT, not merely to a tolerance.

Usage:
    PY="python3"
    cd "Applications/Application1/test scripts"
    "$PY" test_boundary_mode_unchanged.py --tfinal 10
    "$PY" test_boundary_mode_unchanged.py                 # production tfinal
"""

from __future__ import annotations

import argparse
import importlib.util
import os as _os
import shutil
import subprocess
import sys as _sys
import tempfile
from pathlib import Path

import numpy as np

_APP1_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _APP1_DIR not in _sys.path:
    _sys.path.insert(0, _APP1_DIR)

import run_optimization as app1


APP_DIR = Path(_APP1_DIR)
PROJECT_ROOT = APP_DIR.parents[1]
# The shared objective module moved from "Applications/Test Application2/" to
# "Src/Objective/"; a revision may hold it at either path.
LEGACY_GRADIENT_PATHS = (
    "Src/Objective/gradient_test.py",
    "Applications/Test Application2/gradient_test.py",
)
LEGACY_GRADIENT_DEST = "Src/Objective/gradient_test.py"
LEGACY_FILES = ("Src/Solvers/adv_diffusion_adjoint.py",)
MEASUREMENT_BOUNDARIES = ["left", "right", "bottom"]
CONTROLS = (
    ("true control", np.array([162.0, 300.0, 1.40, 270.0, 600.0, 0.95])),
    (
        "archived initialization",
        np.array(
            [
                270.07223071231783,
                599.9449540916377,
                0.9490975837420003,
                162.0128404133529,
                300.0415089706361,
                1.3987361385724204,
            ]
        ),
    ),
    ("perturbed control", np.array([200.0, 500.0, 1.25, 130.0, 330.0, 0.60])),
)


def materialize_legacy_tree(revision: str, destination: Path) -> Path:
    """Write the pre-change shared solver files into a mirrored temp tree."""
    (destination / "Src" / "Solvers").mkdir(parents=True, exist_ok=True)
    (destination / "Src" / "Objective").mkdir(parents=True, exist_ok=True)
    for name in sorted((PROJECT_ROOT / "Src" / "Solvers").glob("*.py")):
        shutil.copy2(name, destination / "Src" / "Solvers" / name.name)
    for rel in LEGACY_FILES:
        blob = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "show", f"{revision}:{rel}"],
            check=True,
            capture_output=True,
        ).stdout
        (destination / rel).write_bytes(blob)
    for rel in LEGACY_GRADIENT_PATHS:
        result = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "show", f"{revision}:{rel}"],
            capture_output=True,
        )
        if result.returncode == 0:
            (destination / LEGACY_GRADIENT_DEST).write_bytes(result.stdout)
            return destination / LEGACY_GRADIENT_DEST
    raise FileNotFoundError(
        f"gradient_test.py not found at {LEGACY_GRADIENT_PATHS} in revision {revision}"
    )


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("legacy_gradient_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(gt_module, mesh_data, pde, control, ud, state):
    j = gt_module.compute_objective(
        state,
        ud,
        mesh_data,
        eta1=app1.ETA1,
        lam=app1.LAMBDA_REG,
        z=control,
        dt_out=pde["dt"] * pde["jprint"],
        time_rule=app1.OBJECTIVE_TIME_RULE,
    )
    _r, _a, gradient, _raw = gt_module.compute_gradient_parts(
        control,
        state,
        ud,
        mesh_data,
        pde,
        eta1=app1.ETA1,
        lam=app1.LAMBDA_REG,
        gradient_time_rule=app1.GRADIENT_TIME_RULE,
    )
    return float(j), np.asarray(gradient, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument("--tfinal", type=float, default=None)
    args = parser.parse_args()

    mesh_data = app1.configure_measurement_boundaries(
        app1.build_mesh_data(), MEASUREMENT_BOUNDARIES
    )
    if mesh_data.get("observation") is not None:
        raise SystemExit("boundary mode must not attach an observation operator")

    pde = dict(app1.PDE)
    if args.tfinal is not None:
        pde["tfinal"] = float(args.tfinal)
    velocity_nodes, _summary = app1.load_velocity_field_file(
        str(app1.DEFAULT_VELOCITY_FIELD_FILE), mesh_data
    )
    pde["velocity_nodes"] = velocity_nodes

    with tempfile.TemporaryDirectory(prefix="pdeco_legacy_") as tmp:
        legacy_path = materialize_legacy_tree(args.revision, Path(tmp))
        legacy_gt = load_module(legacy_path)

        print("=== Boundary Mode Backward-Compatibility Check ===")
        print(f"legacy revision: {args.revision}")
        print(f"legacy files: {list(LEGACY_FILES)}")
        print(f"measurement boundaries: {MEASUREMENT_BOUNDARIES}")
        print(f"tfinal: {pde['tfinal']}  sigma2: {pde['sigma2']}")

        ud = app1.gt.run_forward(mesh_data, pde, CONTROLS[0][1])
        ud_legacy = legacy_gt.run_forward(mesh_data, pde, CONTROLS[0][1])
        if not np.array_equal(ud, ud_legacy):
            raise SystemExit("forward solves differ; the solver itself changed")
        print("forward solve: bit-identical")

        failures = 0
        for label, control in CONTROLS:
            state = (
                ud if np.array_equal(control, CONTROLS[0][1])
                else app1.gt.run_forward(mesh_data, pde, control)
            )
            j_new, g_new = evaluate(app1.gt, mesh_data, pde, control, ud, state)
            j_old, g_old = evaluate(legacy_gt, mesh_data, pde, control, ud, state)
            same_j = j_new == j_old
            same_g = np.array_equal(g_new, g_old)
            failures += int(not (same_j and same_g))
            print(f"\n--- {label} ---")
            print(f"legacy  J: {j_old:.17e}")
            print(f"current J: {j_new:.17e}")
            print(f"objective bit-identical: {same_j}")
            print(f"gradient bit-identical:  {same_g}")
            if not same_g:
                print(f"max |dg|: {np.max(np.abs(g_new - g_old)):.6e}")

        verdict = "PASS" if failures == 0 else "FAIL"
        print(f"\nboundary mode unchanged: {verdict}")
        if failures:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
