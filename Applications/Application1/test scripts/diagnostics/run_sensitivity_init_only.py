"""Run ONLY the App1 sensitivity initialization (no L-BFGS-B) and record everything.

Calls the unmodified production entry point
``sensitivity_initialization.build_sensitivity_initial_control`` with exactly the
arguments ``run_optimization.py --init-mode sensitivity`` passes, for both
measurement-boundary configurations, and additionally for the candidate remedy
settings (which are existing function parameters, not new code).

Saves per-configuration: the selected initial control, the exact initial
objective, the per-source diagnostics, solve counts, plots, and timings.
"""

from __future__ import annotations

import json
import time

import numpy as np

import sys as _sys_dc
from pathlib import Path as _Path_dc
_sys_dc.path.insert(0, str(_Path_dc(__file__).resolve().parents[1]))  # _diag_common
import _diag_common as D
import run_optimization as ro
import sensitivity_initialization as sinit


OUT = D.APP_DIR / "output" / "sensitivity_init_only"
TRUE1 = np.array([250.0, 130.0])
TRUE2 = np.array([15.0, 580.0])

CASES = [
    # label, boundaries, top_k, beam, min_separation, screen_radius
    ("prod_lrb", "lrb", 24, 3, None, None),
    ("prod_lrbt", "lrbt", 24, 3, None, None),
    ("remedy_k48_r40_sep150_lrb", "lrb", 48, 3, 150.0, 40.0),
    ("remedy_k48_r40_sep150_lrbt", "lrbt", 48, 3, 150.0, 40.0),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    for label, bkey, top_k, beam, sep, radius in CASES:
        print(f"\n{'#' * 78}\n# {label}: boundaries={D.BOUNDARY_SETS[bkey]} "
              f"top_k={top_k} beam={beam} min_sep={sep} screen_radius={radius}\n{'#' * 78}")
        t0 = time.time()
        mesh_data, pde, admiss = D.build_setup(D.BOUNDARY_SETS[bkey], 1600.0)
        obs = D.true_state(mesh_data, pde)
        plot_dir = OUT / label
        plot_dir.mkdir(parents=True, exist_ok=True)
        control, diags, extras = sinit.build_sensitivity_initial_control(
            mesh_data, pde, obs, 2, D.AMPLITUDE_BOUNDS, admiss,
            eta1=ro.ETA1, lam=ro.LAMBDA_REG,
            objective_time_rule=ro.OBJECTIVE_TIME_RULE,
            min_separation=sep,
            min_value_tol=0.0,
            save_plots=True,
            plot_dir=plot_dir,
            plot_context=f"{'+'.join(D.BOUNDARY_SETS[bkey])}, sigma2=1600",
            score="predicted_reduction",
            top_k=top_k,
            beam_width=beam,
            screen_radius=radius,
            true_centers_for_plots=np.array([TRUE1, TRUE2]),
        )
        wall = time.time() - t0
        c = np.asarray(control).reshape(2, 3)
        d1 = float(min(np.hypot(*(c[i, :2] - TRUE1)) for i in range(2)))
        d2 = float(min(np.hypot(*(c[i, :2] - TRUE2)) for i in range(2)))
        print(f"\n>>> {label}: control = {np.array2string(control, precision=10)}")
        print(f">>> exact init objective {extras['initial_exact_objective']:.6e} "
              f"(J0 {extras['zero_source_objective']:.6e}, "
              f"{100*(1-extras['initial_exact_objective']/extras['zero_source_objective']):.4f}% explained)")
        print(f">>> nearest source to TRUE1: {d1:.2f} m; to TRUE2: {d2:.2f} m; "
              f"basin-correct = {d1 <= 60 and d2 <= 60}")
        print(f">>> forwards {extras['n_forward_solves']}  "
              f"adjoints {extras['n_adjoint_solves']}  wall {wall:.1f}s")
        results[label] = dict(
            boundaries=list(D.BOUNDARY_SETS[bkey]), top_k=top_k, beam_width=beam,
            min_separation=sep, screen_radius=radius,
            control=[float(v) for v in control],
            initial_exact_objective=float(extras["initial_exact_objective"]),
            zero_source_objective=float(extras["zero_source_objective"]),
            n_forward_solves=int(extras["n_forward_solves"]),
            n_adjoint_solves=int(extras["n_adjoint_solves"]),
            wall_seconds=wall,
            dist_true1=d1, dist_true2=d2,
            basin_correct=bool(d1 <= 60 and d2 <= 60),
            plot_paths=extras["plot_paths"],
            diagnostics=[d.to_dict() for d in diags],
        )
        with open(OUT / "init_only_results.json", "w") as fh:
            json.dump(results, fh, indent=1)

    print("\n" + "=" * 78)
    print(f"{'case':30s} {'control':>52s} {'J_init':>11} {'d1':>6} {'d2':>6} "
          f"{'fwd':>4} {'adj':>4} {'sec':>6}")
    for k, v in results.items():
        cs = "[" + ", ".join(f"{x:.2f}" for x in v["control"]) + "]"
        print(f"{k:30s} {cs:>52s} {v['initial_exact_objective']:11.4e} "
              f"{v['dist_true1']:6.1f} {v['dist_true2']:6.1f} "
              f"{v['n_forward_solves']:4d} {v['n_adjoint_solves']:4d} "
              f"{v['wall_seconds']:6.1f}")


if __name__ == "__main__":
    main()
