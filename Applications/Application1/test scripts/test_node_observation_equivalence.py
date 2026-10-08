"""Boundary vs boundary-equivalent-node observation equivalence check.

Compares the EXISTING boundary observation operator against the
boundary-equivalent mesh-node observation operator (node set = every node of the
selected measurement boundaries; weights = the exact P1 boundary observation
matrix) at fixed controls, BEFORE any expensive optimization.

For every control it reports the boundary objective, the node-equivalent
objective, their absolute/relative difference, both reduced gradients and the
relative gradient difference.  A correctly constructed equivalent operator must
agree to floating-point roundoff (the two implementations differ only in
summation order: an edge/Gauss-point Python loop versus a sparse weighted
matvec).

Default controls are the archived final-sweep case c06
(``final_sweep_c06_nearwall_b13``): its true control, its archived selected
sensitivity initialization, and one nontrivial admissible perturbation.

Usage:
    PY="python3"
    cd "Applications/Application1/test scripts"
    "$PY" test_node_observation_equivalence.py                # archived config
    "$PY" test_node_observation_equivalence.py --tfinal 10    # fast smoke
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
OUTPUT_DIR = APP_DIR / "output" / "node_observation" / "equivalence"

# Archived reference case: output/final_sweep_c06_nearwall_b13/full/.
ARCHIVED_CASE = "final_sweep_c06_nearwall_b13"
MEASUREMENT_BOUNDARIES = ["left", "right", "bottom"]
TRUE_CONTROL = np.array([162.0, 300.0, 1.40, 270.0, 600.0, 0.95], dtype=np.float64)
ARCHIVED_INITIAL_CONTROL = np.array(
    [
        270.07223071231783,
        599.9449540916377,
        0.9490975837420003,
        162.0128404133529,
        300.0415089706361,
        1.3987361385724204,
    ],
    dtype=np.float64,
)
# One nontrivial admissible perturbation of the archived initialization
# (both centers moved tens of metres, both amplitudes changed).
PERTURBED_CONTROL = np.array(
    [200.0, 500.0, 1.25, 130.0, 330.0, 0.60], dtype=np.float64
)


def build_modes(args):
    """Return (boundary_mesh_data, node_mesh_data) for the same boundaries."""
    base = app1.configure_measurement_boundaries(
        app1.build_mesh_data(), MEASUREMENT_BOUNDARIES
    )
    boundary_args = argparse.Namespace(
        observation_mode="boundary",
        sensor_node_ids=None,
        sensor_coordinates=None,
        sensor_weights="equal",
        sensor_equal_weight=1.0,
        sensor_snap_tolerance=app1.DEFAULT_SENSOR_SNAP_TOLERANCE,
    )
    node_args = argparse.Namespace(
        observation_mode="nodes",
        sensor_node_ids=None,
        sensor_coordinates=None,
        sensor_weights="boundary_quadrature",
        sensor_equal_weight=1.0,
        sensor_snap_tolerance=app1.DEFAULT_SENSOR_SNAP_TOLERANCE,
    )
    return (
        app1.configure_observation(base, boundary_args),
        app1.configure_observation(base, node_args),
    )


def objective_and_gradient(mesh_data, pde, control, state, ud):
    j = app1.gt.compute_objective(
        state,
        ud,
        mesh_data,
        eta1=app1.ETA1,
        lam=app1.LAMBDA_REG,
        z=control,
        dt_out=pde["dt"] * pde["jprint"],
        time_rule=app1.OBJECTIVE_TIME_RULE,
    )
    _reg, _adj, gradient, _raw = app1.gt.compute_gradient_parts(
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
    parser.add_argument("--tfinal", type=float, default=None)
    parser.add_argument("--sigma2", type=float, default=None)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    boundary_mesh, node_mesh = build_modes(args)
    admissibility = MeshAdmissibility(boundary_mesh)

    pde = dict(app1.PDE)
    if args.tfinal is not None:
        pde["tfinal"] = float(args.tfinal)
    if args.sigma2 is not None:
        pde["sigma2"] = float(args.sigma2)
    velocity_nodes, velocity_summary = app1.load_velocity_field_file(
        str(app1.DEFAULT_VELOCITY_FIELD_FILE), boundary_mesh
    )
    pde["velocity_nodes"] = velocity_nodes
    pde["velocity_field"] = velocity_summary

    observation = node_mesh["observation"]
    boundary_edges = app1.measurement_observation_edges(boundary_mesh)

    print("=== Observation Equivalence Test ===")
    print(f"archived reference case: {ARCHIVED_CASE}")
    print(f"measurement boundaries: {MEASUREMENT_BOUNDARIES}")
    print(f"boundary edges: {len(boundary_edges)}")
    print(f"node-equivalent sensors: {observation.num_sensors}")
    print(f"weight representation: {observation.weight_kind}")
    print(f"weight convention: {observation.description}")
    print(f"tfinal: {pde['tfinal']}  sigma2: {pde['sigma2']}  eta1: {app1.ETA1}")
    print(f"objective_time_rule: {app1.OBJECTIVE_TIME_RULE}")

    controls = [
        ("archived true control", TRUE_CONTROL),
        ("archived selected initialization", ARCHIVED_INITIAL_CONTROL),
        ("perturbed admissible control", PERTURBED_CONTROL),
    ]
    for label, control in controls:
        app1.validate_control_for_region(
            control, label, MESH_DOMAIN_BOUND_CASE, admissibility
        )

    print("\nGenerating synthetic observations from the true state...")
    ud = app1.gt.run_forward(boundary_mesh, pde, TRUE_CONTROL)
    print(f"true state shape: {ud.shape}")
    print(
        "node-mode data d_s = C_s U_true extracted at the same "
        f"{observation.num_sensors} sensor nodes"
    )

    records = []
    worst_rel_j = 0.0
    worst_rel_g = 0.0
    for label, control in controls:
        print(f"\n--- {label} ---")
        print(f"control: {np.array2string(control, precision=10)}")
        state = ud if np.array_equal(control, TRUE_CONTROL) else (
            app1.gt.run_forward(boundary_mesh, pde, control)
        )
        j_b, g_b = objective_and_gradient(boundary_mesh, pde, control, state, ud)
        j_n, g_n = objective_and_gradient(node_mesh, pde, control, state, ud)

        dj = abs(j_n - j_b)
        rel_j = dj / max(abs(j_b), 1.0e-300)
        dg = float(np.linalg.norm(g_n - g_b))
        rel_g = dg / max(float(np.linalg.norm(g_b)), 1.0e-300)
        worst_rel_j = max(worst_rel_j, rel_j)
        worst_rel_g = max(worst_rel_g, rel_g)

        print(f"boundary objective:       {j_b:.16e}")
        print(f"node-equivalent objective:{j_n:.16e}")
        print(f"absolute difference:      {dj:.6e}")
        print(f"relative difference:      {rel_j:.6e}")
        print(f"boundary gradient:        {np.array2string(g_b, precision=12)}")
        print(f"node-equivalent gradient: {np.array2string(g_n, precision=12)}")
        print(f"gradient abs difference:  {dg:.6e}")
        print(f"gradient rel difference:  {rel_g:.6e}")

        records.append(
            {
                "label": label,
                "control": control.tolist(),
                "boundary_objective": j_b,
                "node_objective": j_n,
                "objective_abs_diff": dj,
                "objective_rel_diff": rel_j,
                "boundary_gradient": g_b.tolist(),
                "node_gradient": g_n.tolist(),
                "gradient_abs_diff": dg,
                "gradient_rel_diff": rel_g,
            }
        )

    tolerance = 1.0e-12
    verdict = "PASS" if max(worst_rel_j, worst_rel_g) <= tolerance else "FAIL"
    print("\n=== Verdict ===")
    print(f"worst relative objective difference: {worst_rel_j:.6e}")
    print(f"worst relative gradient difference:  {worst_rel_g:.6e}")
    print(f"tolerance: {tolerance:.1e}")
    print(f"equivalence: {verdict}")

    summary = {
        "archived_case": ARCHIVED_CASE,
        "measurement_boundaries": MEASUREMENT_BOUNDARIES,
        "boundary_edges": int(len(boundary_edges)),
        "observation": observation.metadata(),
        "pde": app1.serializable_pde_config(pde),
        "eta1": app1.ETA1,
        "objective_time_rule": app1.OBJECTIVE_TIME_RULE,
        "gradient_time_rule": app1.GRADIENT_TIME_RULE,
        "results": records,
        "worst_objective_rel_diff": worst_rel_j,
        "worst_gradient_rel_diff": worst_rel_g,
        "tolerance": tolerance,
        "verdict": verdict,
    }
    out_path = OUTPUT_DIR / (
        "equivalence_summary.json"
        if args.tfinal is None
        else f"equivalence_summary_tfinal{args.tfinal:g}.json"
    )
    out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"summary: {out_path}")

    if verdict != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
