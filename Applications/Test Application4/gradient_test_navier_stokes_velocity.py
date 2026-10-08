"""Taylor gradient check for App4 with the Navier-Stokes nodal velocity field."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import run_optimization as app4
from mesh_admissibility import MeshAdmissibility
from multi_source_utils import MESH_DOMAIN_BOUND_CASE


APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "output" / "gradient_tests" / "navier_stokes_velocity"
VELOCITY_FIELD_FILE = (
    APP_DIR / "output" / "navier_stokes" / "app4_navier_stokes_velocity_nodes.npz"
)

TRUE_CONTROL = np.array([0.20, 0.88, 1.10, 1.30, 0.22, 0.85], dtype=np.float64)
TEST_CONTROL = np.array([0.65, 0.18, 0.60, 0.30, 0.35, 0.50], dtype=np.float64)
MEASUREMENT_BOUNDARIES = ["right", "inner_corner", "top"]
H_VALUES = np.array([10.0**(-power) for power in range(1, 9)], dtype=np.float64)


def make_pde_with_velocity(mesh_data):
    pde = dict(app4.PDE)
    velocity_nodes, velocity_summary = app4.load_velocity_field_file(
        str(VELOCITY_FIELD_FILE), mesh_data
    )
    pde["velocity_nodes"] = velocity_nodes
    pde["velocity_field"] = velocity_summary
    return pde, velocity_summary


def is_valid_control(control: np.ndarray, admissibility: MeshAdmissibility) -> bool:
    try:
        app4.validate_control_for_region(
            control, "Taylor candidate", MESH_DOMAIN_BOUND_CASE, admissibility
        )
    except ValueError:
        return False
    return True


def choose_direction(
    z: np.ndarray,
    admissibility: MeshAdmissibility,
    hs: np.ndarray,
    seed: int = 0,
) -> np.ndarray:
    """Choose a deterministic unit direction whose tested steps stay admissible."""
    rng = np.random.default_rng(seed)
    for _attempt in range(1000):
        direction = rng.normal(size=z.size)
        norm = np.linalg.norm(direction)
        if norm == 0.0:
            continue
        direction /= norm
        if all(is_valid_control(z + h * direction, admissibility) for h in hs):
            return direction
    raise RuntimeError(
        "Could not find an admissible Taylor direction after 1000 deterministic draws"
    )


def objective_at(control: np.ndarray, mesh_data, pde, target_state: np.ndarray) -> float:
    state = app4.gt.run_forward(mesh_data, pde, control)
    return float(
        app4.gt.compute_objective(
            state,
            target_state,
            mesh_data,
            eta1=app4.ETA1,
            lam=app4.LAMBDA_REG,
            z=control,
            dt_out=pde["dt"] * pde["jprint"],
            time_rule=app4.OBJECTIVE_TIME_RULE,
        )
    )


def estimate_slope(hs: np.ndarray, residuals: np.ndarray, start: int = 1, stop: int = 5):
    hs_window = hs[start:stop]
    residual_window = residuals[start:stop]
    mask = residual_window > 0.0
    if np.count_nonzero(mask) < 2:
        return None
    coeff = np.polyfit(np.log10(hs_window[mask]), np.log10(residual_window[mask]), 1)
    return float(coeff[0])


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "h",
                "J_perturbed",
                "R1",
                "R2",
                "R1_rate_from_previous",
                "R2_rate_from_previous",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_plot(path: Path, hs: np.ndarray, r1: np.ndarray, r2: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.loglog(hs, r1, "o-", label="R1 = |J(z+h d)-J(z)|")
    ax.loglog(hs, r2, "s-", label="R2 = |J(z+h d)-J(z)-h g.d|")

    ref_h = hs
    r1_ref = r1[0] * (ref_h / ref_h[0])
    r2_ref = max(r2[0], 1.0e-30) * (ref_h / ref_h[0]) ** 2
    ax.loglog(ref_h, r1_ref, "--", color="0.45", label="O(h) reference")
    ax.loglog(ref_h, r2_ref, ":", color="0.25", label="O(h^2) reference")

    ax.set_xlabel("h")
    ax.set_ylabel("residual")
    ax.set_title("App4 Navier-Stokes Velocity Gradient Taylor Test")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mesh_data = app4.configure_measurement_boundaries(
        app4.build_mesh_data(), MEASUREMENT_BOUNDARIES
    )
    admissibility = MeshAdmissibility(mesh_data)
    app4.validate_control_for_region(
        TRUE_CONTROL, "true/reference", MESH_DOMAIN_BOUND_CASE, admissibility
    )
    app4.validate_control_for_region(
        TEST_CONTROL, "Taylor base", MESH_DOMAIN_BOUND_CASE, admissibility
    )
    pde, velocity_summary = make_pde_with_velocity(mesh_data)

    print("=== App4 Navier-Stokes Velocity Gradient Test ===")
    print(f"velocity field file: {velocity_summary['path']}")
    print(
        "coordinate validation passed: "
        f"{velocity_summary['coordinate_validation_passed']}"
    )
    print(f"measurement boundaries: {MEASUREMENT_BOUNDARIES}")
    print(f"true control: {np.array2string(TRUE_CONTROL, precision=16)}")
    print(f"test control z: {np.array2string(TEST_CONTROL, precision=16)}")
    print(f"objective_time_rule: {app4.OBJECTIVE_TIME_RULE}")
    print(f"gradient_time_rule: {app4.GRADIENT_TIME_RULE}")

    print("\nRunning target forward solve at true control...")
    target_state = app4.gt.run_forward(mesh_data, pde, TRUE_CONTROL)
    print(f"target_state shape: {target_state.shape}")

    print("Running base forward solve and adjoint gradient at test control...")
    base_state = app4.gt.run_forward(mesh_data, pde, TEST_CONTROL)
    j0 = app4.gt.compute_objective(
        base_state,
        target_state,
        mesh_data,
        eta1=app4.ETA1,
        lam=app4.LAMBDA_REG,
        z=TEST_CONTROL,
        dt_out=pde["dt"] * pde["jprint"],
        time_rule=app4.OBJECTIVE_TIME_RULE,
    )
    _g_reg, _g_adj, gradient, _g_adj_raw = app4.gt.compute_gradient_parts(
        TEST_CONTROL,
        base_state,
        target_state,
        mesh_data,
        pde,
        eta1=app4.ETA1,
        lam=app4.LAMBDA_REG,
        gradient_time_rule=app4.GRADIENT_TIME_RULE,
    )

    direction = choose_direction(TEST_CONTROL, admissibility, H_VALUES, seed=0)
    gdotd = float(np.dot(gradient, direction))

    print(f"J(z): {float(j0):.16e}")
    print(f"gradient: {np.array2string(gradient, precision=16)}")
    print(f"gradient norm: {np.linalg.norm(gradient):.16e}")
    print(f"direction d: {np.array2string(direction, precision=16)}")
    print(f"gradient dot direction: {gdotd:.16e}")

    print("\n=== Taylor Table ===")
    print(f"{'h':>10} {'J(z+h d)':>22} {'R1':>18} {'R2':>18} {'rate R2':>10}")

    rows = []
    previous = None
    for h in H_VALUES:
        z_perturbed = TEST_CONTROL + h * direction
        if not is_valid_control(z_perturbed, admissibility):
            raise RuntimeError(f"Perturbed control left App4 domain at h={h}")
        jh = objective_at(z_perturbed, mesh_data, pde, target_state)
        r1 = abs(jh - j0)
        r2 = abs(jh - j0 - h * gdotd)
        if previous is None:
            r1_rate = None
            r2_rate = None
            rate_text = "-"
        else:
            prev_h, prev_r1, prev_r2 = previous
            r1_rate = float(np.log(prev_r1 / r1) / np.log(prev_h / h)) if r1 > 0 else None
            r2_rate = float(np.log(prev_r2 / r2) / np.log(prev_h / h)) if r2 > 0 else None
            rate_text = f"{r2_rate:.4f}" if r2_rate is not None else "-"
        rows.append(
            {
                "h": float(h),
                "J_perturbed": float(jh),
                "R1": float(r1),
                "R2": float(r2),
                "R1_rate_from_previous": r1_rate,
                "R2_rate_from_previous": r2_rate,
            }
        )
        print(f"{h:10.1e} {jh:22.14e} {r1:18.10e} {r2:18.10e} {rate_text:>10}")
        previous = (h, r1, r2)

    r1_values = np.asarray([row["R1"] for row in rows], dtype=np.float64)
    r2_values = np.asarray([row["R2"] for row in rows], dtype=np.float64)
    r2_middle_slope = estimate_slope(H_VALUES, r2_values, start=1, stop=5)

    csv_path = OUTPUT_DIR / "gradient_test_results.csv"
    json_path = OUTPUT_DIR / "gradient_test_summary.json"
    png_path = OUTPUT_DIR / "gradient_test_loglog.png"
    write_csv(csv_path, rows)
    write_plot(png_path, H_VALUES, r1_values, r2_values)

    summary = {
        "velocity_field_file": velocity_summary["path"],
        "velocity_summary": velocity_summary,
        "measurement_boundaries": MEASUREMENT_BOUNDARIES,
        "true_control": TRUE_CONTROL.tolist(),
        "test_control": TEST_CONTROL.tolist(),
        "objective_time_rule": app4.OBJECTIVE_TIME_RULE,
        "gradient_time_rule": app4.GRADIENT_TIME_RULE,
        "J_z": float(j0),
        "gradient": gradient.tolist(),
        "gradient_norm": float(np.linalg.norm(gradient)),
        "direction": direction.tolist(),
        "gradient_dot_direction": gdotd,
        "h_values": H_VALUES.tolist(),
        "results": rows,
        "estimated_R2_slope_middle_h_1e-2_to_1e-5": r2_middle_slope,
        "output_csv": str(csv_path),
        "output_json": str(json_path),
        "output_png": str(png_path),
    }
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\n=== Output ===")
    print(f"estimated R2 slope over h=1e-2..1e-5: {r2_middle_slope}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"PNG: {png_path}")


if __name__ == "__main__":
    main()
