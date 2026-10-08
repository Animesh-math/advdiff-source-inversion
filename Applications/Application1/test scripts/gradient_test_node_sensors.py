"""Taylor gradient test for Application1 in NODE-SENSOR observation mode.

Verifies that the RK4 discrete adjoint remains exact for the newly defined
discrete node-sensor objective

    J_h = (eta1 * dt / 2) * sum_n omega_n || C_s U^n - d_s^n ||_{W_s}^2 ,

whose state derivative is  eta1 * dt * omega_n * C_s^T W_s (C_s U^n - d_s^n).
A correct discrete adjoint gives R2 = |J(z+h d) - J(z) - h g.d| = O(h^2).

Two configurations are exercised:

  ``sparse``   a genuinely sparse set of interior/boundary mesh nodes with the
               ordinary equal point-sensor weighting (the real use case);
  ``boundary`` the boundary-equivalent node set (matrix weights), which must
               also pass because it is the same operator machinery.

Usage:
    PY="python3"
    cd "Applications/Application1/test scripts"
    "$PY" gradient_test_node_sensors.py                       # sparse sensors
    "$PY" gradient_test_node_sensors.py --config boundary
    "$PY" gradient_test_node_sensors.py --tfinal 60           # cheaper
"""

from __future__ import annotations

import argparse
import json
import os as _os
import sys as _sys
from pathlib import Path

import numpy as np

_APP1_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _APP1_DIR not in _sys.path:
    _sys.path.insert(0, _APP1_DIR)

import run_optimization as app1
from mesh_admissibility import MeshAdmissibility
from multi_source_utils import MESH_DOMAIN_BOUND_CASE


APP_DIR = Path(_APP1_DIR)
OUTPUT_DIR = APP_DIR / "output" / "node_observation" / "gradient_tests"

MEASUREMENT_BOUNDARIES = ["left", "right", "bottom"]
TRUE_CONTROL = np.array([162.0, 300.0, 1.40, 270.0, 600.0, 0.95], dtype=np.float64)
TEST_CONTROL = np.array([200.0, 500.0, 1.25, 130.0, 330.0, 0.60], dtype=np.float64)
H_VALUES = np.array([10.0 ** (-p) for p in range(1, 7)], dtype=np.float64)

# 18 sparse sensor locations spread over the DC street domain: the three
# observed outflow/inflow walls plus interior street corridors.  Chosen from
# geometry only (see inspect_mesh_nodes.py), snapped to mesh nodes at run time.
SPARSE_SENSOR_COORDINATES = (
    "10,60;10,300;10,560;"
    "295,80;295,300;295,520;295,660;"
    "60,10;150,10;250,10;"
    "100,250;150,400;200,150;"
    "120,320;90,430;260,430;200,620;150,690"
)


def build_mesh(config: str, coordinates: str, snap_tolerance: float):
    base = app1.configure_measurement_boundaries(
        app1.build_mesh_data(), MEASUREMENT_BOUNDARIES
    )
    if config == "sparse":
        args = argparse.Namespace(
            observation_mode="nodes",
            sensor_node_ids=None,
            sensor_coordinates=coordinates,
            sensor_weights="equal",
            sensor_equal_weight=1.0,
            sensor_snap_tolerance=snap_tolerance,
        )
    else:
        args = argparse.Namespace(
            observation_mode="nodes",
            sensor_node_ids=None,
            sensor_coordinates=None,
            sensor_weights="boundary_quadrature",
            sensor_equal_weight=1.0,
            sensor_snap_tolerance=snap_tolerance,
        )
    return app1.configure_observation(base, args)


def is_valid(control, admissibility) -> bool:
    try:
        app1.validate_control_for_region(
            control, "Taylor candidate", MESH_DOMAIN_BOUND_CASE, admissibility
        )
    except ValueError:
        return False
    return True


def choose_direction(z, admissibility, hs, seed: int = 0):
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        d = rng.normal(size=z.size)
        norm = np.linalg.norm(d)
        if norm == 0.0:
            continue
        d /= norm
        if all(is_valid(z + h * d, admissibility) for h in hs):
            return d
    raise RuntimeError("No admissible Taylor direction found in 1000 draws")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=("sparse", "boundary"), default="sparse")
    parser.add_argument("--tfinal", type=float, default=None)
    parser.add_argument(
        "--sensor-coordinates", default=SPARSE_SENSOR_COORDINATES
    )
    parser.add_argument(
        "--sensor-snap-tolerance",
        type=float,
        default=app1.DEFAULT_SENSOR_SNAP_TOLERANCE,
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mesh_data = build_mesh(
        args.config, args.sensor_coordinates, args.sensor_snap_tolerance
    )
    app1.print_observation_summary(mesh_data)
    observation = mesh_data["observation"]
    admissibility = MeshAdmissibility(mesh_data)

    pde = dict(app1.PDE)
    if args.tfinal is not None:
        pde["tfinal"] = float(args.tfinal)
    velocity_nodes, velocity_summary = app1.load_velocity_field_file(
        str(app1.DEFAULT_VELOCITY_FIELD_FILE), mesh_data
    )
    pde["velocity_nodes"] = velocity_nodes
    pde["velocity_field"] = velocity_summary

    for label, control in (("true", TRUE_CONTROL), ("test", TEST_CONTROL)):
        app1.validate_control_for_region(
            control, label, MESH_DOMAIN_BOUND_CASE, admissibility
        )

    print("\n=== Node-Sensor Gradient Taylor Test ===")
    print(f"configuration: {args.config}")
    print(f"tfinal: {pde['tfinal']}  sigma2: {pde['sigma2']}  eta1: {app1.ETA1}")
    print(f"true control: {np.array2string(TRUE_CONTROL, precision=10)}")
    print(f"test control: {np.array2string(TEST_CONTROL, precision=10)}")

    # Synthetic node observations d_s = C_s U_true, generated from the true
    # state at exactly the sensor nodes (the operator does the extraction).
    ud = app1.gt.run_forward(mesh_data, pde, TRUE_CONTROL)
    d_sensors = observation.observe(ud)
    print(f"synthetic sensor data shape: {d_sensors.shape}")
    print(
        "sensor data range: "
        f"[{float(np.min(d_sensors)):.6e}, {float(np.max(d_sensors)):.6e}]"
    )

    state0 = app1.gt.run_forward(mesh_data, pde, TEST_CONTROL)
    j0 = app1.gt.compute_objective(
        state0,
        ud,
        mesh_data,
        eta1=app1.ETA1,
        lam=app1.LAMBDA_REG,
        z=TEST_CONTROL,
        dt_out=pde["dt"] * pde["jprint"],
        time_rule=app1.OBJECTIVE_TIME_RULE,
    )
    _r, _a, gradient, _raw = app1.gt.compute_gradient_parts(
        TEST_CONTROL,
        state0,
        ud,
        mesh_data,
        pde,
        eta1=app1.ETA1,
        lam=app1.LAMBDA_REG,
        gradient_time_rule=app1.GRADIENT_TIME_RULE,
    )
    gradient = np.asarray(gradient, dtype=np.float64)
    direction = choose_direction(TEST_CONTROL, admissibility, H_VALUES, seed=0)
    gdotd = float(np.dot(gradient, direction))

    print(f"J(z): {float(j0):.16e}")
    print(f"gradient: {np.array2string(gradient, precision=12)}")
    print(f"gradient norm: {np.linalg.norm(gradient):.16e}")
    print(f"direction: {np.array2string(direction, precision=12)}")
    print(f"gradient dot direction: {gdotd:.16e}")

    print("\n=== Taylor Table ===")
    print(f"{'h':>10} {'J(z+h d)':>24} {'R1':>16} {'R2':>16} {'rate R2':>10}")
    rows = []
    previous = None
    for h in H_VALUES:
        zh = TEST_CONTROL + h * direction
        if not is_valid(zh, admissibility):
            raise RuntimeError(f"Perturbed control left the fluid domain at h={h}")
        jh = float(
            app1.gt.compute_objective(
                app1.gt.run_forward(mesh_data, pde, zh),
                ud,
                mesh_data,
                eta1=app1.ETA1,
                lam=app1.LAMBDA_REG,
                z=zh,
                dt_out=pde["dt"] * pde["jprint"],
                time_rule=app1.OBJECTIVE_TIME_RULE,
            )
        )
        r1 = abs(jh - float(j0))
        r2 = abs(jh - float(j0) - h * gdotd)
        rate = None
        if previous is not None and r2 > 0.0 and previous[2] > 0.0:
            rate = float(np.log(previous[2] / r2) / np.log(previous[0] / h))
        rows.append(
            {"h": float(h), "J": jh, "R1": r1, "R2": r2, "R2_rate": rate}
        )
        print(
            f"{h:10.1e} {jh:24.14e} {r1:16.8e} {r2:16.8e} "
            f"{(f'{rate:.4f}' if rate is not None else '-'):>10}"
        )
        previous = (h, r1, r2)

    rates = [row["R2_rate"] for row in rows if row["R2_rate"] is not None]
    best_window = None
    if len(rates) >= 3:
        best_window = max(
            float(np.mean(rates[i:i + 3])) for i in range(len(rates) - 2)
        )
    mean_rate = float(np.mean(rates)) if rates else float("nan")
    verdict = (
        "PASS"
        if (best_window is not None and best_window >= 1.8) or mean_rate >= 1.8
        else "FAIL"
    )
    print(f"\nmean R2 rate: {mean_rate:.6f}")
    if best_window is not None:
        print(f"best 3-rate window mean: {best_window:.6f}")
    print(f"second-order remainder verdict: {verdict}")

    summary = {
        "configuration": args.config,
        "observation": observation.metadata(),
        "measurement_boundaries": MEASUREMENT_BOUNDARIES,
        "pde": app1.serializable_pde_config(pde),
        "eta1": app1.ETA1,
        "true_control": TRUE_CONTROL.tolist(),
        "test_control": TEST_CONTROL.tolist(),
        "J_z": float(j0),
        "gradient": gradient.tolist(),
        "gradient_norm": float(np.linalg.norm(gradient)),
        "direction": direction.tolist(),
        "gradient_dot_direction": gdotd,
        "results": rows,
        "mean_R2_rate": mean_rate,
        "best_3_window_R2_rate": best_window,
        "verdict": verdict,
    }
    out_path = OUTPUT_DIR / f"node_gradient_test_{args.config}.json"
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"summary: {out_path}")
    if verdict != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
