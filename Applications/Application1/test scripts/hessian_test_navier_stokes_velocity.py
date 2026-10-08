"""Hessian verification for Application1 with the Navier-Stokes velocity field."""

from __future__ import annotations

import csv
import json
import os as _os
import sys as _sys
from pathlib import Path

import numpy as np

# This test/verification script lives in Application1/"test scripts"/; put the
# parent Application1 directory on sys.path so its local modules import cleanly.
# (is_valid_control is imported from the sibling gradient test in this folder.)
_APP1_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _APP1_DIR not in _sys.path:
    _sys.path.insert(0, _APP1_DIR)

import hessian_utils
import run_optimization as app1
from gradient_test_navier_stokes_velocity import is_valid_control
from mesh_admissibility import MeshAdmissibility
from multi_source_utils import (
    INITIAL_CONTROL,
    MESH_DOMAIN_BOUND_CASE,
    TRUE_CONTROL,
    component_names,
)


# The Application1 directory (parent of this "test scripts" folder), so output
# artifacts land under Application1/output as before the move.
APP_DIR = Path(__file__).resolve().parent.parent
VELOCITY_FIELD_FILE = (
    APP_DIR / "output" / "navier_stokes" / "application1_navier_stokes_velocity_nodes.npz"
)
# Hessian base is the admissible Application1 INITIAL_CONTROL (DC meters).
TEST_CONTROL = INITIAL_CONTROL.copy()
MEASUREMENT_BOUNDARIES = ["right", "bottom", "top"]
MODE = "exact"
FD_EPSILONS = np.array([1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5], dtype=np.float64)


def output_dir_for(boundaries: list[str]) -> Path:
    name = "_".join(boundaries)
    return APP_DIR / "output" / "hessian_tests" / "navier_stokes_velocity" / name


def make_pde_with_velocity(mesh_data):
    pde = dict(app1.PDE)
    pde["objective_time_rule"] = app1.OBJECTIVE_TIME_RULE
    velocity_nodes, velocity_summary = app1.load_velocity_field_file(
        str(VELOCITY_FIELD_FILE), mesh_data
    )
    pde["velocity_nodes"] = velocity_nodes
    pde["velocity_field"] = velocity_summary
    return pde, velocity_summary


def gradient_at(control: np.ndarray, mesh_data, pde, target_state: np.ndarray):
    state = app1.gt.run_forward(mesh_data, pde, control)
    _g_reg, _g_adj, gradient, _g_adj_raw = app1.gt.compute_gradient_parts(
        control,
        state,
        target_state,
        mesh_data,
        pde,
        eta1=app1.ETA1,
        lam=app1.LAMBDA_REG,
        gradient_time_rule=app1.GRADIENT_TIME_RULE,
    )
    return gradient


def choose_direction(
    z: np.ndarray,
    admissibility: MeshAdmissibility,
    epsilons: np.ndarray,
    seed: int = 4,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    for _attempt in range(1000):
        direction = rng.normal(size=z.size)
        direction /= np.linalg.norm(direction)
        if all(
            is_valid_control(z + eps * direction, admissibility)
            and is_valid_control(z - eps * direction, admissibility)
            for eps in epsilons
        ):
            return direction
    raise RuntimeError("Could not find an admissible finite-difference direction.")


def write_fd_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "eps",
                "abs_error",
                "rel_error",
                "fd_hv_norm",
                "analytic_hv_norm",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_eigen_csv(path: Path, eigenvalues: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["index", "eigenvalue"])
        writer.writeheader()
        for index, value in enumerate(eigenvalues):
            writer.writerow({"index": index, "eigenvalue": float(value)})


def main() -> None:
    out_dir = output_dir_for(MEASUREMENT_BOUNDARIES)
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh_data = app1.configure_measurement_boundaries(
        app1.build_mesh_data(), MEASUREMENT_BOUNDARIES
    )
    admissibility = MeshAdmissibility(mesh_data)
    app1.validate_control_for_region(
        TRUE_CONTROL, "true/reference", MESH_DOMAIN_BOUND_CASE, admissibility
    )
    app1.validate_control_for_region(
        TEST_CONTROL, "Hessian base", MESH_DOMAIN_BOUND_CASE, admissibility
    )
    pde, velocity_summary = make_pde_with_velocity(mesh_data)

    print("=== Application1 Navier-Stokes Hessian Test ===")
    print(f"mode: {MODE}")
    print(f"measurement boundaries: {MEASUREMENT_BOUNDARIES}")
    print(f"velocity field file: {velocity_summary['path']}")
    print(f"test control z: {np.array2string(TEST_CONTROL, precision=16)}")
    print(f"components: {component_names()}")

    print("\nRunning target forward solve...")
    target_state = app1.gt.run_forward(mesh_data, pde, TRUE_CONTROL)
    print(f"target_state shape: {target_state.shape}")

    print("Running base forward solve and gradient...")
    base_state = app1.gt.run_forward(mesh_data, pde, TEST_CONTROL)
    j0 = app1.gt.compute_objective(
        base_state,
        target_state,
        mesh_data,
        eta1=app1.ETA1,
        lam=app1.LAMBDA_REG,
        z=TEST_CONTROL,
        dt_out=pde["dt"] * pde["jprint"],
        time_rule=app1.OBJECTIVE_TIME_RULE,
    )
    gradient = gradient_at(TEST_CONTROL, mesh_data, pde, target_state)
    print(f"J(z): {float(j0):.16e}")
    print(f"gradient norm: {np.linalg.norm(gradient):.16e}")

    print("\nFilling raw Hessian by coordinate Hessian-vector products...")
    H = hessian_utils.hessian_matrix(
        TEST_CONTROL,
        mesh_data,
        pde,
        target_state,
        eta1=app1.ETA1,
        lam=app1.LAMBDA_REG,
        mode=MODE,
    )
    H_sym = 0.5 * (H + H.T)
    eigenvalues = np.linalg.eigvalsh(H_sym)
    positive = np.abs(eigenvalues) > 1.0e-14
    condition_number = None
    if np.any(positive):
        nonzero_abs_eigs = np.abs(eigenvalues[positive])
        condition_number = float(np.max(nonzero_abs_eigs) / np.min(nonzero_abs_eigs))

    np.save(out_dir / "hessian_raw.npy", H)
    np.save(out_dir / "hessian_sym.npy", H_sym)
    write_eigen_csv(out_dir / "hessian_eigenvalues.csv", eigenvalues)

    raw_asym_abs = float(np.linalg.norm(H - H.T, ord="fro"))
    raw_norm = float(np.linalg.norm(H, ord="fro"))
    raw_asym_rel = raw_asym_abs / max(1.0, raw_norm)

    print(f"raw Hessian:\n{np.array2string(H, precision=10)}")
    print(f"symmetrized eigenvalues: {np.array2string(eigenvalues, precision=10)}")
    print(f"condition number from nonzero |eigenvalues|: {condition_number}")
    print(f"raw asymmetry abs: {raw_asym_abs:.16e}")
    print(f"raw asymmetry rel: {raw_asym_rel:.16e}")

    rng = np.random.default_rng(12)
    pair_rows = []
    max_pair_abs = 0.0
    max_pair_rel = 0.0
    for pair_index in range(5):
        p = rng.normal(size=TEST_CONTROL.size)
        q = rng.normal(size=TEST_CONTROL.size)
        p /= np.linalg.norm(p)
        q /= np.linalg.norm(q)
        pt_hq = float(p @ (H @ q))
        qt_hp = float(q @ (H @ p))
        abs_error = abs(pt_hq - qt_hp)
        rel_error = abs_error / max(1.0e-14, abs(pt_hq), abs(qt_hp))
        max_pair_abs = max(max_pair_abs, abs_error)
        max_pair_rel = max(max_pair_rel, rel_error)
        pair_rows.append(
            {
                "pair": pair_index,
                "pTHq": pt_hq,
                "qTHp": qt_hp,
                "abs_error": abs_error,
                "rel_error": rel_error,
            }
        )
    print(f"max pTHq symmetry abs error: {max_pair_abs:.16e}")
    print(f"max pTHq symmetry rel error: {max_pair_rel:.16e}")

    direction = choose_direction(TEST_CONTROL, admissibility, FD_EPSILONS)
    analytic_hv = hessian_utils.hessian_vector_product(
        TEST_CONTROL,
        direction,
        mesh_data,
        pde,
        target_state,
        eta1=app1.ETA1,
        lam=app1.LAMBDA_REG,
        mode=MODE,
        base_state=base_state,
    )
    print("\nFinite-difference Hessian-vector test:")
    print(f"direction: {np.array2string(direction, precision=16)}")
    print(f"analytic Hs: {np.array2string(analytic_hv, precision=16)}")

    fd_rows = []
    for eps in FD_EPSILONS:
        g_plus = gradient_at(
            TEST_CONTROL + eps * direction, mesh_data, pde, target_state
        )
        g_minus = gradient_at(
            TEST_CONTROL - eps * direction, mesh_data, pde, target_state
        )
        fd_hv = (g_plus - g_minus) / (2.0 * eps)
        abs_error = float(np.linalg.norm(fd_hv - analytic_hv))
        rel_error = abs_error / max(
            1.0e-14,
            float(np.linalg.norm(fd_hv)),
            float(np.linalg.norm(analytic_hv)),
        )
        fd_rows.append(
            {
                "eps": float(eps),
                "abs_error": abs_error,
                "rel_error": rel_error,
                "fd_hv_norm": float(np.linalg.norm(fd_hv)),
                "analytic_hv_norm": float(np.linalg.norm(analytic_hv)),
            }
        )
        print(
            f"eps={eps:.1e} abs_error={abs_error:.16e} "
            f"rel_error={rel_error:.16e}"
        )
    write_fd_csv(out_dir / "hessian_fd_test.csv", fd_rows)

    summary = {
        "mode": MODE,
        "velocity_field_file": velocity_summary["path"],
        "velocity_summary": velocity_summary,
        "measurement_boundaries": MEASUREMENT_BOUNDARIES,
        "true_control": TRUE_CONTROL.tolist(),
        "test_control": TEST_CONTROL.tolist(),
        "objective_time_rule": app1.OBJECTIVE_TIME_RULE,
        "gradient_time_rule": app1.GRADIENT_TIME_RULE,
        "objective": float(j0),
        "gradient": gradient.tolist(),
        "gradient_norm": float(np.linalg.norm(gradient)),
        "hessian_shape": list(H.shape),
        "hessian_fro_norm": raw_norm,
        "raw_asymmetry_abs": raw_asym_abs,
        "raw_asymmetry_rel": raw_asym_rel,
        "symmetry_pair_tests": pair_rows,
        "max_pair_abs_error": max_pair_abs,
        "max_pair_rel_error": max_pair_rel,
        "eigenvalues_sym": eigenvalues.tolist(),
        "condition_number_abs_eigenvalues": condition_number,
        "fd_direction": direction.tolist(),
        "analytic_hv": analytic_hv.tolist(),
        "fd_tests": fd_rows,
        "outputs": {
            "hessian_raw": str(out_dir / "hessian_raw.npy"),
            "hessian_sym": str(out_dir / "hessian_sym.npy"),
            "summary": str(out_dir / "hessian_summary.json"),
            "fd_test": str(out_dir / "hessian_fd_test.csv"),
            "eigenvalues": str(out_dir / "hessian_eigenvalues.csv"),
        },
        "limitations": (
            "Gauss-Newton mode omits base-adjoint source-second-derivative "
            "terms. This run uses exact mode, which includes those terms for "
            "finite-difference verification of the current discrete gradient."
        ),
    }
    (out_dir / "hessian_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print("\n=== Output ===")
    print(f"raw Hessian: {out_dir / 'hessian_raw.npy'}")
    print(f"sym Hessian: {out_dir / 'hessian_sym.npy'}")
    print(f"summary: {out_dir / 'hessian_summary.json'}")
    print(f"FD CSV: {out_dir / 'hessian_fd_test.csv'}")
    print(f"eigenvalues CSV: {out_dir / 'hessian_eigenvalues.csv'}")


if __name__ == "__main__":
    main()
