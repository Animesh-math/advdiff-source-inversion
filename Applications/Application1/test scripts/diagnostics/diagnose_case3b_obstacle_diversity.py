"""Case 3b sensitivity-initialization diagnostics ONLY (no L-BFGS-B).

Runs the production ``build_sensitivity_initial_control`` entry point at the
exact Case 3b configuration, with the obstacle-aware diversity screen either ON
or OFF, and records everything needed to answer:

  * which candidates the top-K diversity screen retains;
  * which of them survived ONLY because a wall separated them from a
    higher-ranked candidate (i.e. the Euclidean rule would have suppressed them);
  * their sensitivity ranks and scores;
  * the single selected sensitivity initialization and its exact objective.

The true control is used ONLY for post-hoc distance reporting and for the
plot-only overlay; it never enters candidate selection, masking, or scoring.

Usage:
  cd "Applications/Application1/test scripts/diagnostics"
  python3 diagnose_case3b_obstacle_diversity.py --obstacle-aware
  python3 diagnose_case3b_obstacle_diversity.py --no-obstacle-aware
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

import sys as _sys_dc
from pathlib import Path as _Path_dc
_sys_dc.path.insert(0, str(_Path_dc(__file__).resolve().parents[1]))  # _diag_common
import _diag_common as D
import run_optimization as ro
import sensitivity_initialization as sinit

# Case 3b production configuration (the development log, "App1 second-source
# re-selection, and the decisive Case 3b near-truth probe").
TRUE_CONTROL = np.array([260.0, 300.0, 1.55, 60.0, 180.0, 1.10])
BOUNDARIES = ("left", "right", "bottom")
SIGMA2_STAGES = (1600.0, 800.0, 400.0)
TFINAL = 150.0
AMPLITUDE_BOUNDS = (0.1, 2.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--obstacle-aware", dest="oa", action="store_true", default=True)
    ap.add_argument("--no-obstacle-aware", dest="oa", action="store_false")
    args = ap.parse_args()

    label = "obstacle_aware" if args.oa else "euclidean"
    out = D.APP_DIR / "output" / "case3b_init_diag" / label
    out.mkdir(parents=True, exist_ok=True)

    true_centers = TRUE_CONTROL.reshape(2, 3)[:, :2]
    print(f"### Case 3b sensitivity-init diagnostic: diversity = {label}")
    print(f"### true control (REPORTING ONLY): {TRUE_CONTROL.tolist()}")
    print(f"### boundaries {BOUNDARIES}  init sigma2 {SIGMA2_STAGES[0]}  "
          f"tfinal {TFINAL}  amplitude bounds {AMPLITUDE_BOUNDS}")

    mesh_data, pde, admiss = D.build_setup(BOUNDARIES, SIGMA2_STAGES[0], TFINAL)
    obs = D.true_state(mesh_data, pde, TRUE_CONTROL)

    t0 = time.time()
    control, diags, extras = sinit.build_sensitivity_initial_control(
        mesh_data, pde, obs, 2, AMPLITUDE_BOUNDS, admiss,
        eta1=ro.ETA1, lam=ro.LAMBDA_REG,
        objective_time_rule=ro.OBJECTIVE_TIME_RULE,
        min_separation=None,
        min_value_tol=0.0,
        save_plots=True,
        plot_dir=out,
        plot_context=f"{'+'.join(BOUNDARIES)}, sigma2={SIGMA2_STAGES[0]:g}, {label}",
        score="predicted_reduction",
        top_k=24,
        beam_width=3,
        refine_centers=True,
        refine_max_fev=60,
        coherence_cap=0.3,
        refine_pair=True,
        pair_refine_max_fev=60,
        geometry_aware_branches=True,
        obstacle_aware_diversity=args.oa,
        reselect_second=True,
        reselect_threshold=0.25,
        reselect_max_passes=2,
        true_centers_for_plots=true_centers,
    )
    wall = time.time() - t0

    c = np.asarray(control, dtype=np.float64).reshape(2, 3)
    # optimal permutation matching, reporting only
    perms = [(0, 1), (1, 0)]
    best_perm, best_err = None, None
    for perm in perms:
        err = sum(
            float(np.hypot(*(c[perm[i], :2] - true_centers[i]))) for i in range(2)
        )
        if best_err is None or err < best_err:
            best_perm, best_err = perm, err

    print(f"\n>>> selected initialization = {np.array2string(control, precision=6)}")
    print(f">>> exact init objective {extras['initial_exact_objective']:.6e} "
          f"(zero-source J0 {extras['zero_source_objective']:.6e})")
    for i in range(2):
        d = float(np.hypot(*(c[best_perm[i], :2] - true_centers[i])))
        print(f">>> init source matched to true source {i + 1} "
              f"({true_centers[i][0]:.0f},{true_centers[i][1]:.0f}): "
              f"({c[best_perm[i], 0]:.2f},{c[best_perm[i], 1]:.2f}) "
              f"a={c[best_perm[i], 2]:.4f}  distance {d:.2f} m")
    print(f">>> forwards {extras['n_forward_solves']}  "
          f"adjoints {extras['n_adjoint_solves']}  wall {wall / 60.0:.1f} min")

    div = extras.get("diversity", {})
    print(f"\n>>> diversity summary: {json.dumps({k: v for k, v in div.items() if k != 'per_screen'})}")
    for screen in div.get("per_screen", []):
        preserved = screen.get("preserved_xy") or []
        print(f"    screen '{screen['label']}' r={screen['radius_m']:.1f} m: "
              f"retained {screen['n_retained']}, suppressed same-region "
              f"{screen['n_suppressed_by_distance']}, obstacle-preserved "
              f"{screen['n_preserved_obstacle_separated']}")
        for px, py in preserved:
            dists = [float(np.hypot(px - t[0], py - t[1])) for t in true_centers]
            print(f"        obstacle-preserved ({px:7.2f},{py:7.2f})  "
                  f"distance to true s1 {dists[0]:7.2f} m, to true s2 {dists[1]:7.2f} m")

    payload = {
        "label": label,
        "obstacle_aware_diversity": bool(args.oa),
        "true_control_reporting_only": TRUE_CONTROL.tolist(),
        "boundaries": list(BOUNDARIES),
        "init_sigma2": SIGMA2_STAGES[0],
        "selected_control": [float(v) for v in control],
        "initial_exact_objective": float(extras["initial_exact_objective"]),
        "zero_source_objective": float(extras["zero_source_objective"]),
        "n_forward_solves": int(extras["n_forward_solves"]),
        "n_adjoint_solves": int(extras["n_adjoint_solves"]),
        "wall_seconds": float(wall),
        "diversity": div,
        "branch_controls": [
            {k: v for k, v in b.items() if k != "traces"}
            for b in extras.get("branch_controls", [])
        ],
        "pair_refinement": extras.get("pair_refinement"),
        "second_source_reselection": extras.get("second_source_reselection"),
        "plot_paths": extras.get("plot_paths"),
    }
    (out / "init_diagnostic.json").write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out / 'init_diagnostic.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
