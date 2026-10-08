"""Application1 driver: adjoint sensitivity-field restart for a partial recovery.

This is the controlled first experiment described in the task
``right_t150_start2_sensitivity_enrichment_test``.  It answers a single
question: can an adjoint-based source-insertion sensitivity field correctly
REINITIALIZE one DEAD source while RETAINING one ACTIVE source, for the
right-boundary-only, tfinal=150 partial-recovery endpoint of Start 2 of the
adaptive multistart experiment
``right_t150_adaptive_ampfloor_01_sigma2_1600_800_400_seed42_fixed``.

Pipeline (see ``source_insertion_sensitivity.py`` for the math and sign
convention):

  1. Build the SAME physical/observation configuration as that experiment
     (DC mesh + Navier-Stokes velocity, right-boundary measurements, tfinal=150,
     dt=0.1, k=10, sigma2=400, eta1=100, no regularization).
  2. Generate the synthetic observation u_d from the recorded true control
     (this is DATA of the inverse problem, exactly as the optimizer generates
     it; the true control is used for NOTHING else -- no candidate filtering,
     selection, classification, or initialization).
  3. z_bar = the Start-2 endpoint with the inactive amplitude set exactly to 0.
  4. ONE forward solve for z_bar, ONE adjoint solve -> adjoint weight field W.
  5. S(q) for every mesh vertex via cheap Gaussian-weighted quadrature sums.
  6. Admissible candidates = interior fluid vertices >= 40 m from the retained
     active source.  q_new = argmin S over admissible candidates, requiring
     S(q_new) < 0 (descent for a positive amplitude).
  7. Finite-difference validation of S at the selected minimum, a moderately
     negative point, and a near-zero/least-negative point.
  8. Sample one amplitude a_new ~ Uniform(0.3, 1.3) with a recorded seed
     (reusing the smart-amplitude uniform draw).
  9. z_restart = [q_new_x, q_new_y, a_new, <retained active source>].
 10. Run ONE ordinary full local optimization from z_restart via the existing
     driver (manual --initial-control override, same settings), then report.

Outputs land in ``output/right_t150_start2_sensitivity_enrichment_test/`` and do
NOT overwrite the original multistart outputs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:  # pragma: no cover
    plt = None

import meshio

# This verification/experiment script lives in Application1/"test scripts"/diagnostics/; put
# the parent Application1 directory on sys.path so its local modules import
# cleanly (and so the subprocess call below targets the right run_optimization).
import os as _os

_APP1_DIR = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _APP1_DIR not in sys.path:
    sys.path.insert(0, _APP1_DIR)

import run_optimization as ro
from mesh_data_config import build_mesh_data
from mesh_admissibility import MeshAdmissibility
import source_insertion_sensitivity as sis


# The Application1 directory (two levels above this "test scripts/diagnostics" folder), so outputs
# and the subprocessed run_optimization.py resolve under Application1 as before.
APP_DIR = Path(__file__).resolve().parents[2]
# Case name / output paths are resolved per run in main() from the selected
# measurement boundaries (default: right-only, reproducing the first experiment).
DEFAULT_MEASUREMENT_BOUNDARIES = ["right"]
CASE_NAME = "right_t150_start2_sensitivity_enrichment_test"
OUT_DIR = APP_DIR / "output" / CASE_NAME
SENS_DIR = OUT_DIR / "sensitivity_field"

# --- Fixed inputs from the original experiment (Start 2, right, tfinal=150) ---
# Recovered from
# output/right_t150_adaptive_ampfloor_01_sigma2_1600_800_400_seed42_fixed/full/
#   start_002/optimization_summary.json
TRUE_CONTROL = np.array([150.0, 600.0, 1.2, 50.0, 250.0, 0.7], dtype=np.float64)
START2_ENDPOINT = np.array(
    [0.0, 0.0, 0.1, 149.9975258526295, 599.9988264225241, 1.2000135865980774],
    dtype=np.float64,
)
# Baseline z_bar: inactive source-1 amplitude set EXACTLY to 0 (center irrelevant).
Z_BAR = np.array(
    [0.0, 0.0, 0.0, 149.9975258526295, 599.9988264225241, 1.2000135865980774],
    dtype=np.float64,
)
RETAINED_SOURCE_CENTER = np.array([149.9975258526295, 599.9988264225241], dtype=np.float64)

TFINAL = 150.0
TARGET_SIGMA2 = 400.0
SEPARATION_MIN_M = 40.0
AMPLITUDE_SAMPLE_LOW = 0.3
AMPLITUDE_SAMPLE_HIGH = 1.3
AMPLITUDE_SAMPLE_SEED = 20260720  # fixed, recorded for reproducibility

# Enriched-restart optimization settings (identical to the original experiment
# except for the measurement boundaries, which are selected per run).
OPT_SETTINGS = {
    "mode": "location_and_intensity",
    "tfinal": TFINAL,
    "amplitude_floor": 0.1,
    "sigma2_continuation": [1600.0, 800.0, 400.0],
    "max_iter": 150,
}
# Measurement boundaries for the current run (set in main()).
MEASUREMENT_BOUNDARIES = list(DEFAULT_MEASUREMENT_BOUNDARIES)


def build_pde(mesh_data):
    pde = dict(ro.PDE)
    pde["tfinal"] = TFINAL
    pde["sigma2"] = TARGET_SIGMA2
    velocity_nodes, velocity_summary = ro.load_velocity_field_file(
        str(ro.DEFAULT_VELOCITY_FIELD_FILE), mesh_data
    )
    pde["velocity_nodes"] = velocity_nodes
    pde["velocity_field"] = velocity_summary
    return pde


def classify_candidates(coord, admissibility, boundary_node_set):
    """Return (admissible_mask, reason) per vertex, matching the optimizer.

    A vertex is an admissible candidate iff it lies in a fluid triangle (the
    optimizer's authoritative admissibility, which excludes the 21 building
    holes and the exterior), is an INTERIOR node (not on the outer domain
    boundary), and is at least SEPARATION_MIN_M from the retained active source.
    No true-source information is used.
    """
    n = coord.shape[0]
    reason = np.array(["admissible"] * n, dtype=object)
    admissible = np.ones(n, dtype=bool)
    for i in range(n):
        x, y = float(coord[i, 0]), float(coord[i, 1])
        if not admissibility.is_point_in_domain(x, y):
            admissible[i] = False
            reason[i] = "outside_fluid_or_building_hole"
            continue
        if i in boundary_node_set:
            admissible[i] = False
            reason[i] = "outer_boundary_node"
            continue
        dist = float(np.hypot(x - RETAINED_SOURCE_CENTER[0], y - RETAINED_SOURCE_CENTER[1]))
        if dist < SEPARATION_MIN_M:
            admissible[i] = False
            reason[i] = "within_40m_of_retained_source"
            continue
    return admissible, reason


def finite_difference_validation(mesh_data, pde, ud_out, j_baseline, q, s_pred, eta1):
    """Compare S(q) with forward differences [J(z_bar + eps at q) - J(z_bar)]/eps."""
    dt_out = pde["dt"] * pde["jprint"]
    results = []
    for eps in (1.0e-3, 1.0e-2, 5.0e-2, 1.0e-1):
        z_eps = np.array(
            [q[0], q[1], eps, RETAINED_SOURCE_CENTER[0], RETAINED_SOURCE_CENTER[1],
             Z_BAR[5]],
            dtype=np.float64,
        )
        state_eps = sis.gt.run_forward(mesh_data, pde, z_eps)
        j_eps = sis.gt.compute_objective(
            state_eps, ud_out, mesh_data, eta1=eta1, lam=ro.LAMBDA_REG,
            z=z_eps, dt_out=dt_out, time_rule=ro.OBJECTIVE_TIME_RULE,
        )
        fd = (j_eps - j_baseline) / eps
        results.append({
            "eps": float(eps),
            "J_eps": float(j_eps),
            "fd_slope": float(fd),
            "S_pred": float(s_pred),
            "abs_diff": float(abs(fd - s_pred)),
            "sign_match": bool((fd < 0) == (s_pred < 0)),
        })
    return results


def write_vtu(coord, connections, sensitivity, admissible):
    SENS_DIR.mkdir(parents=True, exist_ok=True)
    points = np.asarray(coord, dtype=np.float64)
    if points.shape[1] == 2:
        points = np.column_stack([points, np.zeros(points.shape[0])])
    cells = [("triangle", np.asarray(connections, dtype=int))]
    raw = sensitivity.astype(np.float64)
    # Admissible-only view (NaN elsewhere) and a normalized-magnitude view purely
    # for visualization -- selection always uses the RAW field, never normalized.
    admissible_field = np.where(admissible, raw, np.nan)
    finite = raw[np.isfinite(raw)]
    scale = np.max(np.abs(finite)) if finite.size else 1.0
    normalized = raw / scale if scale > 0 else raw
    point_data = {
        "source_insertion_sensitivity": raw,
        "source_insertion_sensitivity_admissible": admissible_field,
        "source_insertion_sensitivity_normalized": normalized,
        "admissible": admissible.astype(np.float64),
    }
    path = SENS_DIR / "source_insertion_sensitivity.vtu"
    meshio.write(path, meshio.Mesh(points=points, cells=cells, point_data=point_data))
    return path


def write_csv(coord, sensitivity, admissible, reason):
    SENS_DIR.mkdir(parents=True, exist_ok=True)
    path = SENS_DIR / "source_insertion_sensitivity.csv"
    lines = ["index,x,y,sensitivity,admissible,exclusion_reason"]
    for i in range(coord.shape[0]):
        lines.append(
            f"{i},{coord[i,0]:.10e},{coord[i,1]:.10e},{sensitivity[i]:.16e},"
            f"{int(admissible[i])},{'' if admissible[i] else reason[i]}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_png(coord, connections, sensitivity, admissible, q_new, fd_points):
    if plt is None:
        return None
    SENS_DIR.mkdir(parents=True, exist_ok=True)
    import matplotlib.tri as mtri

    triang = mtri.Triangulation(coord[:, 0], coord[:, 1], connections)
    field = np.where(admissible, sensitivity, np.nan)
    fig, ax = plt.subplots(figsize=(6.4, 9.0))
    tpc = ax.tripcolor(triang, field, shading="gouraud", cmap="viridis")
    fig.colorbar(tpc, ax=ax, label="S(q)  (dJ/dA_new;  < 0 = descent)")
    # Retained active source (NOT a true-source marker).
    ax.plot(RETAINED_SOURCE_CENTER[0], RETAINED_SOURCE_CENTER[1], "o",
            color="white", markeredgecolor="black", markersize=11,
            label="retained active source")
    # Selected sensitivity minimum.
    ax.plot(q_new[0], q_new[1], "*", color="red", markeredgecolor="black",
            markersize=20, label="selected S minimum (q_new)")
    for p in fd_points:
        ax.plot(p["q"][0], p["q"][1], "s", color="orange",
                markeredgecolor="black", markersize=7)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m, local)")
    ax.set_ylabel("y (m, local)")
    ax.set_title("Source-insertion sensitivity S(q)\n"
                 f"{'+'.join(MEASUREMENT_BOUNDARIES)} boundary, tfinal=150, sigma2=400")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = SENS_DIR / "source_insertion_sensitivity.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def run_enriched_optimization(z_restart):
    """Invoke the existing optimizer with the enriched manual initial control."""
    py = sys.executable
    cmd = [
        py, str(APP_DIR / "run_optimization.py"),
        "--mode", OPT_SETTINGS["mode"],
        "--case-name", CASE_NAME,
        "--measurement-boundaries", *MEASUREMENT_BOUNDARIES,
        "--true-control", *[f"{v:.16g}" for v in TRUE_CONTROL],
        "--initial-control", *[f"{v:.16g}" for v in z_restart],
        "--init-mode", "manual",
        "--tfinal", f"{OPT_SETTINGS['tfinal']}",
        "--amplitude-floor", f"{OPT_SETTINGS['amplitude_floor']}",
        "--sigma2-continuation", *[f"{v:g}" for v in OPT_SETTINGS["sigma2_continuation"]],
        "--max-iter", f"{OPT_SETTINGS['max_iter']}",
        "--max-paraview-frames", "50",
    ]
    print("\n=== Enriched restart optimization command ===")
    print(" ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(APP_DIR))
    runtime = time.time() - t0
    return cmd, proc.returncode, runtime


def parse_args():
    parser = argparse.ArgumentParser(
        description="Adjoint source-insertion sensitivity restart for an "
                    "Application1 partial recovery (single controlled run)."
    )
    parser.add_argument(
        "--measurement-boundaries", nargs="+", default=DEFAULT_MEASUREMENT_BOUNDARIES,
        help="Observation boundaries, e.g. 'right' or 'right bottom'. "
             "Matches the optimizer's --measurement-boundaries.",
    )
    parser.add_argument(
        "--case-name", default=None,
        help="Output case folder name. Defaults to "
             "<boundaries>_t150_start2_sensitivity_enrichment_test.",
    )
    return parser.parse_args()


def main():
    global MEASUREMENT_BOUNDARIES, CASE_NAME, OUT_DIR, SENS_DIR

    args = parse_args()
    MEASUREMENT_BOUNDARIES = list(args.measurement_boundaries)
    if args.case_name is not None:
        CASE_NAME = args.case_name
    else:
        CASE_NAME = (
            "_".join(MEASUREMENT_BOUNDARIES)
            + "_t150_start2_sensitivity_enrichment_test"
        )
    OUT_DIR = APP_DIR / "output" / CASE_NAME
    SENS_DIR = OUT_DIR / "sensitivity_field"
    print(f"=== Case: {CASE_NAME} (boundaries: {MEASUREMENT_BOUNDARIES}) ===")

    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    mesh_data = build_mesh_data()
    # Wire in the selected measurement boundaries exactly as the optimizer does:
    # a single name sets mesh_data["measurement_boundary"]; multiple names build
    # a synthetic union boundary Gamma_obs so the tracking objective AND the RK4
    # discrete adjoint both integrate over the union (right+bottom here).
    mesh_data = ro.configure_measurement_boundaries(mesh_data, MEASUREMENT_BOUNDARIES)
    print(f"measurement boundaries: {list(mesh_data['selected_measurement_boundaries'])}")
    pde = build_pde(mesh_data)
    coord = np.asarray(mesh_data["coord"], dtype=np.float64)[:, :2]
    connections = np.asarray(mesh_data["connections"], dtype=int)
    admissibility = MeshAdmissibility(mesh_data)

    # --- Synthetic observation data u_d (DATA, generated from the true control,
    #     exactly as the optimizer does; used for nothing else). ---
    print("=== Generating synthetic observation u_d (data setup) ===")
    ud_out = sis.gt.run_forward(mesh_data, pde, TRUE_CONTROL)

    # --- Sensitivity map: ONE forward (z_bar) + ONE adjoint. ---
    print("=== Building sensitivity map (1 forward + 1 adjoint) ===")
    state_out, source_adjoint, inv_mass, W = sis.compute_adjoint_weight_field(
        mesh_data, pde, Z_BAR, ud_out, ro.ETA1, ro.OBJECTIVE_TIME_RULE
    )
    dt_out = pde["dt"] * pde["jprint"]
    j_baseline = float(sis.gt.compute_objective(
        state_out, ud_out, mesh_data, eta1=ro.ETA1, lam=ro.LAMBDA_REG,
        z=Z_BAR, dt_out=dt_out, time_rule=ro.OBJECTIVE_TIME_RULE,
    ))
    qx, qy, ag = sis.precompute_quadrature_weights(mesh_data, pde, W)
    sensitivity = sis.sensitivity_at_candidates(coord, qx, qy, ag, pde["sigma2"])

    # --- Admissibility (interior fluid vertices, >=40 m from retained source). ---
    boundary_names = ("left", "right", "top", "bottom")
    boundary_node_set = set()
    for name in boundary_names:
        nodes = mesh_data["boundary_nodes_by_name"].get(name)
        if nodes is not None:
            boundary_node_set.update(int(v) for v in np.asarray(nodes).reshape(-1))
    admissible, reason = classify_candidates(coord, admissibility, boundary_node_set)
    n_admissible = int(np.sum(admissible))

    # --- Selection: q_new = argmin S over admissible candidates. ---
    adm_idx = np.flatnonzero(admissible)
    adm_sens = sensitivity[adm_idx]
    order = np.argsort(adm_sens)  # ascending: most negative (descent) first
    imin = int(adm_idx[order[0]])
    q_new = coord[imin].copy()
    s_min = float(sensitivity[imin])
    descent_ok = bool(s_min < 0.0)

    # --- POST-RUN DIAGNOSTIC ONLY (never used for selection/initialization): ---
    # sensitivity value and rank at the admissible vertex nearest the physically
    # missing source (the true source NOT retained by z_bar).  This inspects
    # whether the residual-driven field even points toward the missing source.
    true_sources = TRUE_CONTROL.reshape(2, 3)[:, :2]
    d_to_retained = np.linalg.norm(true_sources - RETAINED_SOURCE_CENTER, axis=1)
    missing_source_xy = true_sources[int(np.argmax(d_to_retained))]
    adm_coord = coord[adm_idx]
    near_local = int(np.argmin(np.linalg.norm(adm_coord - missing_source_xy, axis=1)))
    i_missing = int(adm_idx[near_local])
    s_missing = float(sensitivity[i_missing])
    # Rank among admissible candidates (ascending S; rank 1 = most negative).
    ranks = np.empty(len(order), dtype=int)
    ranks[order] = np.arange(1, len(order) + 1)
    missing_rank = int(ranks[near_local])
    missing_diag = {
        "missing_source_reference_xy": missing_source_xy.tolist(),
        "nearest_admissible_vertex_index": i_missing,
        "nearest_admissible_vertex_xy": coord[i_missing].tolist(),
        "distance_to_missing_source_m": float(
            np.linalg.norm(coord[i_missing] - missing_source_xy)
        ),
        "sensitivity_at_nearest": s_missing,
        "rank_ascending_among_admissible": missing_rank,
        "num_admissible": int(len(order)),
        "percentile_more_negative": float(100.0 * (missing_rank - 1) / max(1, len(order) - 1)),
        "note": "Diagnostic only; the true control is NOT used for selection.",
    }

    # --- Pick FD validation points: selected min, moderately negative, least neg. ---
    imid = int(adm_idx[order[len(order) // 2]])         # moderately negative
    imax = int(adm_idx[order[-1]])                        # near-zero / least negative
    fd_targets = [
        {"label": "selected_minimum", "index": imin, "q": coord[imin], "s": s_min},
        {"label": "moderately_negative", "index": imid, "q": coord[imid],
         "s": float(sensitivity[imid])},
        {"label": "least_negative_or_positive", "index": imax, "q": coord[imax],
         "s": float(sensitivity[imax])},
    ]
    print("\n=== Finite-difference validation of S(q) ===")
    fd_all = []
    for tgt in fd_targets:
        fd_res = finite_difference_validation(
            mesh_data, pde, ud_out, j_baseline, tgt["q"], tgt["s"], ro.ETA1
        )
        print(f"[{tgt['label']}] q=({tgt['q'][0]:.2f},{tgt['q'][1]:.2f}) "
              f"S={tgt['s']:.6e}")
        for r in fd_res:
            print(f"    eps={r['eps']:.1e}  fd={r['fd_slope']:.6e}  "
                  f"S={r['S_pred']:.6e}  |diff|={r['abs_diff']:.3e}  "
                  f"sign_match={r['sign_match']}")
        fd_all.append({**tgt, "q": [float(tgt["q"][0]), float(tgt["q"][1])],
                       "fd": fd_res})

    # --- Internal exact-match cross-check (fast quadrature vs load vector). ---
    s_ref = sis.sensitivity_via_load_vector(mesh_data, pde, source_adjoint, inv_mass, q_new)
    exact_match_rel = abs(s_min - s_ref) / max(1.0e-300, abs(s_ref))

    # --- Sample one amplitude a_new ~ Uniform(0.3, 1.3), recorded seed. ---
    amp_rng = np.random.default_rng(AMPLITUDE_SAMPLE_SEED)
    a_new = float(amp_rng.uniform(AMPLITUDE_SAMPLE_LOW, AMPLITUDE_SAMPLE_HIGH))
    z_restart = np.array(
        [q_new[0], q_new[1], a_new,
         RETAINED_SOURCE_CENTER[0], RETAINED_SOURCE_CENTER[1], Z_BAR[5]],
        dtype=np.float64,
    )
    state_restart = sis.gt.run_forward(mesh_data, pde, z_restart)
    j_restart = float(sis.gt.compute_objective(
        state_restart, ud_out, mesh_data, eta1=ro.ETA1, lam=ro.LAMBDA_REG,
        z=z_restart, dt_out=dt_out, time_rule=ro.OBJECTIVE_TIME_RULE,
    ))

    print(f"\nbaseline objective J(z_bar) = {j_baseline:.10e}")
    print(f"selected q_new = ({q_new[0]:.6f}, {q_new[1]:.6f}), S(q_new) = {s_min:.6e}")
    print(f"exact-match (fast vs load-vector) rel error = {exact_match_rel:.3e}")
    print(f"descent sign OK (S<0): {descent_ok}")
    print(f"sampled a_new = {a_new:.10f} (seed {AMPLITUDE_SAMPLE_SEED})")
    print(f"z_restart = {z_restart.tolist()}")
    print(f"enriched initial objective J(z_restart) = {j_restart:.10e}")
    print("\n=== Missing-source diagnostic (post-run only, not used for selection) ===")
    print(f"missing source ref = ({missing_source_xy[0]:.2f}, {missing_source_xy[1]:.2f})")
    print(f"nearest admissible vertex = ({coord[i_missing][0]:.2f}, {coord[i_missing][1]:.2f}), "
          f"dist={missing_diag['distance_to_missing_source_m']:.2f} m")
    print(f"S there = {s_missing:.6e}  (rank {missing_rank} of {len(order)} ascending; "
          f"{missing_diag['percentile_more_negative']:.1f}% of admissible are more negative)")

    # --- Write sensitivity artifacts. ---
    vtu_path = write_vtu(coord, connections, sensitivity, admissible)
    csv_path = write_csv(coord, sensitivity, admissible, reason)
    png_path = write_png(coord, connections, sensitivity, admissible, q_new, fd_targets)

    # --- Partial summary written before optimization (in case it is long). ---
    summary = {
        "case_name": CASE_NAME,
        "description": "Adjoint sensitivity-field restart for the Start-2 "
                       "tfinal=150 partial recovery (one dead + one active source).",
        "physical_config": {
            "measurement_boundaries": list(mesh_data["selected_measurement_boundaries"]),
            "tfinal": TFINAL,
            "dt": pde["dt"], "k": pde["k"], "sigma2": TARGET_SIGMA2,
            "eta1": ro.ETA1, "lambda_reg": ro.LAMBDA_REG,
            "objective_time_rule": ro.OBJECTIVE_TIME_RULE,
            "velocity_mode": "nodal",
        },
        "baseline_control_z_bar": Z_BAR.tolist(),
        "start2_endpoint": START2_ENDPOINT.tolist(),
        "true_control_used_only_for_data": TRUE_CONTROL.tolist(),
        "baseline_objective": j_baseline,
        "solves_to_construct_map": {
            "forward_solves_for_z_bar": 1,
            "adjoint_solves": 1,
            "data_generation_forward_solves_for_u_d": 1,
            "note": "The per-candidate S(q) map uses NO additional PDE solves.",
        },
        "num_vertices": int(coord.shape[0]),
        "num_admissible_candidates": n_admissible,
        "separation_min_m": SEPARATION_MIN_M,
        "retained_source_center": RETAINED_SOURCE_CENTER.tolist(),
        "min_sensitivity": s_min,
        "min_sensitivity_location": q_new.tolist(),
        "min_sensitivity_vertex_index": imin,
        "descent_sign_ok": descent_ok,
        "exact_match_rel_error": float(exact_match_rel),
        "missing_source_diagnostic": missing_diag,
        "finite_difference_validation": fd_all,
        "amplitude_sample": {
            "distribution": "Uniform",
            "low": AMPLITUDE_SAMPLE_LOW, "high": AMPLITUDE_SAMPLE_HIGH,
            "seed": AMPLITUDE_SAMPLE_SEED, "a_new": a_new,
        },
        "z_restart": z_restart.tolist(),
        "enriched_initial_objective": j_restart,
        "artifacts": {
            "vtu": str(vtu_path), "csv": str(csv_path),
            "png": str(png_path) if png_path else None,
        },
        "optimization_settings": {
            **OPT_SETTINGS, "measurement_boundaries": list(MEASUREMENT_BOUNDARIES)
        },
    }
    summary_path = OUT_DIR / "sensitivity_enrichment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote partial summary: {summary_path}")

    if not descent_ok:
        print("ERROR: S(q_new) >= 0; no descent direction for a positive "
              "amplitude. Aborting restart.")
        return

    # --- Run the enriched restart optimization through the existing pipeline. ---
    cmd, returncode, opt_runtime = run_enriched_optimization(z_restart)

    # --- Read the optimizer's own summary and finalize. ---
    opt_summary_path = OUT_DIR / "full" / "optimization_summary.json"
    opt_result = None
    if opt_summary_path.exists():
        opt = json.loads(opt_summary_path.read_text(encoding="utf-8"))
        opt_result = {
            "success": opt.get("success"),
            "message": opt.get("message"),
            "iterations": opt.get("iterations"),
            "initial_objective": opt.get("initial_objective"),
            "final_objective": opt.get("final_objective"),
            "final_gradient_norm": opt.get("final_gradient_norm"),
            "final_control": opt.get("final_control"),
            "summary_path": str(opt_summary_path),
        }

    summary["enriched_restart_command"] = cmd
    summary["optimization_returncode"] = returncode
    summary["optimization_runtime_seconds"] = opt_runtime
    summary["optimization_result"] = opt_result
    summary["total_runtime_seconds"] = time.time() - t_start
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote final summary: {summary_path}")
    if opt_result is not None:
        print("\n=== Enriched restart optimizer result ===")
        for k, v in opt_result.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
