"""Compare post-initialization polish methods from a SAVED initialized control.

Motivation
----------
Run ``node_sensors_right_bottom_top_c06`` (node sensors on right+bottom+top,
sensitivity initialization) reached ``J = 1.639e-01`` at the initializer, i.e.
sub-metre position error, and then spent 75 L-BFGS-B iterations polishing.  This
script asks whether the second-order machinery already in
``Applications/Application1/hessian_utils.py`` closes that last gap in fewer
solves.

It reads the initialized control straight out of the run's
``sensitivity_init_summary.json`` (no re-initialization: that step cost 482
forward solves / ~20 minutes), evaluates J and g there, and then runs each
requested polish from that identical start under the identical evaluator,
bounds, feasibility mode and observation operator.

Usage
-----
    PY="python3"
    cd "Applications/Application1/test scripts/diagnostics"
    "$PY" diagnose_post_init_polish.py --tfinal 10 --max-iter 5   # smoke
    "$PY" diagnose_post_init_polish.py                            # production
"""

from __future__ import annotations

import argparse
import json
import os as _os
import sys as _sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_APP1_DIR = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
if _APP1_DIR not in _sys.path:
    _sys.path.insert(0, _APP1_DIR)

import run_optimization as app1
import hessian_utils
import post_init_optimizers as pio
from mesh_admissibility import MeshAdmissibility
from multi_source_utils import (
    infer_num_sources,
    mesh_domain_active_bounds_for_mode,
    mode_fixed_values,
    pack_active,
    unpack_active,
)


APP_DIR = Path(_APP1_DIR)

DEFAULT_CASE = "node_sensors_right_bottom_top_c06"
DEFAULT_TRUE_CONTROL = [162.0, 300.0, 1.40, 270.0, 600.0, 0.95]
DEFAULT_SENSOR_NODE_IDS = [
    "641,644,647,650,653,656,659,662,665,668,671,674,677",
    "582,585,588,591,594,596",
    "693,690,687,684,681,679",
]
DEFAULT_MEASUREMENT_BOUNDARIES = ["right", "bottom", "top"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-name", default=DEFAULT_CASE)
    parser.add_argument(
        "--init-summary",
        default=None,
        help="Path to sensitivity_init_summary.json (default: the case's full/ run).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(pio.POST_INIT_METHODS),
        choices=list(pio.POST_INIT_METHODS),
    )
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--tfinal", type=float, default=None)
    parser.add_argument("--amplitude-floor", type=float, default=0.1)
    parser.add_argument("--feasibility-mode", default="raw")
    parser.add_argument("--mode", default="location_and_intensity")
    parser.add_argument("--num-sources", type=int, default=2)
    parser.add_argument(
        "--true-control", type=float, nargs="+", default=DEFAULT_TRUE_CONTROL
    )
    parser.add_argument(
        "--hessian-fd-check",
        action="store_true",
        help=(
            "Also verify the reduced Hessian at the initialized control against "
            "central finite differences of the adjoint gradient. The 'exact' "
            "Hessian must match to the FD floor; the Gauss-Newton Hessian is "
            "expected to differ by exactly the dropped base-adjoint D^2 S term, "
            "and the size of that gap says how Newton-like GN is at this point."
        ),
    )
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def hessian_fd_check(mesh_data, control, true_state, epsilons=(1.0e-2, 1.0e-3, 1.0e-4)):
    """Central-difference check of H(z) d against dg/deps along d."""
    print(f"\n{'=' * 78}\n=== Hessian finite-difference check (node-sensor mode) ===\n{'=' * 78}")
    control = np.asarray(control, dtype=np.float64)
    direction = np.zeros_like(control)
    # A direction that excites both a position and an amplitude component of
    # both sources, scaled so the position and amplitude steps are comparable
    # relative to their own scales (metres vs O(1) amplitudes).
    direction[0] = 1.0
    direction[1] = -0.5
    direction[2] = 0.01
    direction[3] = -0.75
    direction[4] = 0.25
    direction[5] = -0.02

    def grad_at(z):
        state = app1.gt.run_forward(mesh_data, app1.PDE, z)
        _, _, g, _ = app1.gt.compute_gradient_parts(
            z,
            state,
            true_state,
            mesh_data,
            app1.PDE,
            eta1=app1.ETA1,
            lam=app1.LAMBDA_REG,
            gradient_time_rule=app1.GRADIENT_TIME_RULE,
        )
        return np.asarray(g, dtype=np.float64)

    records = []
    analytic = {}
    for hessian_mode in ("exact", "gauss_newton"):
        hv = hessian_utils.hessian_vector_product(
            control,
            direction,
            mesh_data,
            app1.PDE,
            true_state,
            eta1=app1.ETA1,
            lam=app1.LAMBDA_REG,
            mode=hessian_mode,
        )
        analytic[hessian_mode] = np.asarray(hv, dtype=np.float64)
        print(f"H_{hessian_mode} d = {np.array2string(analytic[hessian_mode], precision=8)}")

    gap = analytic["exact"] - analytic["gauss_newton"]
    print(
        f"||H_exact d - H_gn d|| / ||H_exact d|| = "
        f"{np.linalg.norm(gap) / np.linalg.norm(analytic['exact']):.6e}   "
        "(the dropped base-adjoint D^2 S term)"
    )

    for eps in epsilons:
        fd = (grad_at(control + eps * direction) - grad_at(control - eps * direction)) / (
            2.0 * eps
        )
        row = {"epsilon": float(eps), "fd": fd.tolist()}
        for hessian_mode, hv in analytic.items():
            rel = np.linalg.norm(hv - fd) / max(np.linalg.norm(fd), 1.0e-300)
            row[f"rel_error_{hessian_mode}"] = float(rel)
            print(f"  eps={eps:.1e}  relative error vs {hessian_mode:<12} = {rel:.6e}")
        records.append(row)

    return {
        "direction": direction.tolist(),
        "analytic": {k: v.tolist() for k, v in analytic.items()},
        "gauss_newton_gap_relative": float(
            np.linalg.norm(gap) / np.linalg.norm(analytic["exact"])
        ),
        "finite_differences": records,
    }


def build_problem(args):
    """Reproduce main()'s setup for the node-sensor case, without optimizing."""
    fake = SimpleNamespace(
        observation_mode="nodes",
        sensor_node_ids=DEFAULT_SENSOR_NODE_IDS,
        sensor_coordinates=None,
        sensor_weights="equal",
        sensor_equal_weight=1.0,
        sensor_snap_tolerance=app1.DEFAULT_SENSOR_SNAP_TOLERANCE,
    )
    mesh_data = app1.configure_measurement_boundaries(
        app1.build_mesh_data(), DEFAULT_MEASUREMENT_BOUNDARIES
    )
    mesh_data = app1.configure_observation(mesh_data, fake)
    app1.print_observation_summary(mesh_data)

    if args.tfinal is not None:
        app1.PDE["tfinal"] = float(args.tfinal)

    if app1.DEFAULT_VELOCITY_FIELD_FILE.exists():
        velocity_nodes, velocity_summary = app1.load_velocity_field_file(
            str(app1.DEFAULT_VELOCITY_FIELD_FILE), mesh_data
        )
        app1.PDE["velocity_nodes"] = velocity_nodes
        app1.PDE["velocity_field"] = velocity_summary
    else:
        raise SystemExit(
            f"missing velocity field {app1.DEFAULT_VELOCITY_FIELD_FILE}"
        )

    admissibility = MeshAdmissibility(mesh_data)
    return mesh_data, admissibility


def load_initial_control(args) -> np.ndarray:
    path = (
        Path(args.init_summary)
        if args.init_summary
        else APP_DIR / "output" / args.case_name / "full" / "sensitivity_init_summary.json"
    )
    if not path.exists():
        raise SystemExit(f"no initialization summary at {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    control = np.asarray(data["selected_control"], dtype=np.float64)
    print(f"\ninitialized control read from: {path}")
    print(f"  selected_control = {np.array2string(control, precision=16)}")
    print(f"  initializer J_exact = {data.get('initial_exact_objective')!r}")
    print(
        f"  initializer solve budget: {data.get('n_forward_solves')} forward, "
        f"{data.get('n_adjoint_solves')} adjoint, "
        f"{data.get('wall_time_seconds')} s"
    )
    return control


def report_match(true_control, control, label):
    match = app1.match_sources_by_permutation(true_control, control)
    print(f"  {label} permutation-matched recovery (permuted={match['labels_permuted']}):")
    for t in range(match["num_sources"]):
        f = match["permutation"][t]
        print(
            f"    true s{t + 1} <- s{f + 1}: position {match['position_errors_m'][t]:.6e} m  "
            f"amplitude {match['amplitude_errors'][t]:.6e}"
        )
    print(
        f"    max position {match['max_position_error_m']:.6e} m   "
        f"max amplitude {match['max_amplitude_error']:.6e}"
    )
    return match


def run_method(
    method,
    args,
    mesh_data,
    admissibility,
    true_control,
    true_state,
    initial_control,
    active_bounds,
):
    mode = args.mode
    num_sources = int(infer_num_sources(true_control))
    fixed_values = mode_fixed_values(mode, z_ref_full=initial_control)
    z0 = pack_active(initial_control, mode)

    feasibility_stats = app1.new_feasibility_stats(args.feasibility_mode)
    _cache, history, objective, gradient, callback = app1.build_evaluator(
        mesh_data,
        true_state,
        mode,
        app1.MESH_DOMAIN_BOUND_CASE,
        fixed_values,
        initial_control,
        admissibility,
        feasibility_mode=args.feasibility_mode,
        feasibility_stats=feasibility_stats,
    )

    hessian_records = []
    print(f"\n{'=' * 78}\n=== post-init method: {method} ===\n{'=' * 78}")
    started = time.perf_counter()
    with pio.solve_counter(app1.gt) as counts:
        j0 = float(objective(z0))
        g0 = np.asarray(gradient(z0), dtype=np.float64)
        print(f"J(z_init)      = {j0:.16e}")
        print(f"g(z_init)      = {np.array2string(g0, precision=8)}")
        print(f"||g(z_init)||  = {np.linalg.norm(g0):.16e}")
        setup_counts = dict(counts)

        if method == pio.LBFGSB:
            optimizer = app1.Optimizer(
                objective_function=objective,
                gradient_function=gradient,
                optimization_method="L-BFGS-B",
                max_iter=args.max_iter,
                tol=1.0e-8,
                bounds=active_bounds,
                callback=callback,
                options={"ftol": 1.0e-14, "gtol": 1.0e-9, "maxls": 20, "disp": False},
            )
            result = optimizer.optimize(
                initial_guess=z0,
                mesh_data=mesh_data,
                problem_data={"pde": app1.PDE, "mode": mode},
            )
        else:
            hess = pio.build_active_hessian(
                method=method,
                mode=mode,
                num_sources=num_sources,
                mesh_data=mesh_data,
                pde=app1.PDE,
                target_state=true_state,
                z_template=initial_control,
                fixed_values=fixed_values,
                eta1=app1.ETA1,
                lam=app1.LAMBDA_REG,
                unpack_active=unpack_active,
                hessian_module=hessian_utils,
                record=hessian_records,
            )
            result = pio.run_newton_type_polish(
                method=method,
                objective=objective,
                gradient=gradient,
                hess=hess,
                x0=z0,
                bounds=active_bounds,
                max_iter=args.max_iter,
                callback=callback,
                verbose=0,
            )
        z_final = np.asarray(result.x, dtype=np.float64)
        final_control = unpack_active(z_final, initial_control, mode, fixed_values)
        j_final = float(objective(z_final))
        g_final = np.asarray(gradient(z_final), dtype=np.float64)
        total_counts = dict(counts)
    wall = time.perf_counter() - started

    print(f"\nsuccess    = {result.success}")
    print(f"message    = {result.message}")
    print(f"iterations = {int(getattr(result, 'nit', -1))}")
    print(f"nfev/njev/nhev = {getattr(result, 'nfev', None)}/"
          f"{getattr(result, 'njev', None)}/{getattr(result, 'nhev', None)}")
    print(f"J0         = {j0:.16e}")
    print(f"Jf         = {j_final:.16e}")
    print(f"reduction  = {j0 - j_final:.16e}  (factor {j0 / j_final if j_final else float('inf'):.6e})")
    print(f"||g_f||    = {np.linalg.norm(g_final):.16e}")
    print(f"final control = {np.array2string(final_control, precision=12)}")
    match = report_match(true_control, final_control, "final")
    print(
        f"solves: forward={total_counts['forward_solves']} "
        f"adjoint={total_counts['adjoint_solves']}  "
        f"(of which setup/eval-at-start: {setup_counts['forward_solves']}f/"
        f"{setup_counts['adjoint_solves']}a)"
    )
    print(f"wall time  = {wall:.2f} s")
    if hessian_records:
        last = hessian_records[-1]
        print(
            f"last Hessian ({last['hessian_mode']}): spd={last['positive_definite']} "
            f"eig=[{last['min_eigenvalue']:.6e}, {last['max_eigenvalue']:.6e}] "
            f"cond={last['condition_number']!r}"
        )

    return {
        "method": method,
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(getattr(result, "nit", -1)),
        "nfev": int(getattr(result, "nfev", 0) or 0),
        "njev": int(getattr(result, "njev", 0) or 0),
        "nhev": int(getattr(result, "nhev", 0) or 0),
        "forward_solves": int(total_counts["forward_solves"]),
        "adjoint_solves": int(total_counts["adjoint_solves"]),
        "wall_time_seconds": float(wall),
        "initial_objective": j0,
        "initial_gradient_norm": float(np.linalg.norm(g0)),
        "final_objective": j_final,
        "final_gradient_norm": float(np.linalg.norm(g_final)),
        "final_control": final_control.tolist(),
        "recovery_permutation_matched": match,
        "callback_iterations": len(history),
        "hessian_diagnostics": hessian_records,
        "feasibility": dict(feasibility_stats),
    }


def main() -> None:
    args = parse_args()
    true_control = np.asarray(args.true_control, dtype=np.float64)
    mesh_data, admissibility = build_problem(args)
    initial_control = load_initial_control(args)

    num_sources = int(infer_num_sources(true_control))
    active_bounds = app1.apply_amplitude_floor(
        mesh_domain_active_bounds_for_mode(
            args.mode, admissibility.bounding_box, num_sources=num_sources
        ),
        args.mode,
        args.amplitude_floor,
        num_sources=num_sources,
    )

    print("\n=== Problem ===")
    print(f"tfinal={app1.PDE['tfinal']}  dt={app1.PDE['dt']}  sigma2={app1.PDE['sigma2']}")
    print(f"feasibility mode: {args.feasibility_mode}")
    print(f"true control: {np.array2string(true_control, precision=12)}")
    report_match(true_control, initial_control, "initialized")

    print("\ngenerating observation data (forward solve at the true control)...")
    true_state = app1.gt.run_forward(mesh_data, app1.PDE, true_control)

    fd_check = None
    if args.hessian_fd_check:
        fd_check = hessian_fd_check(mesh_data, initial_control, true_state)

    records = []
    for method in args.methods:
        records.append(
            run_method(
                method,
                args,
                mesh_data,
                admissibility,
                true_control,
                true_state,
                initial_control,
                active_bounds,
            )
        )

    print(f"\n\n{'=' * 100}")
    print("=== POST-INITIALIZATION POLISH COMPARISON (same start, same evaluator) ===")
    print(f"{'=' * 100}")
    header = (
        f"{'method':<14}{'iters':>7}{'fwd':>7}{'adj':>7}{'wall s':>10}"
        f"{'J_final':>16}{'max pos err m':>17}{'max amp err':>15}"
    )
    print(header)
    print("-" * len(header))
    for r in records:
        m = r["recovery_permutation_matched"]
        print(
            f"{r['method']:<14}{r['iterations']:>7}{r['forward_solves']:>7}"
            f"{r['adjoint_solves']:>7}{r['wall_time_seconds']:>10.2f}"
            f"{r['final_objective']:>16.4e}{m['max_position_error_m']:>17.4e}"
            f"{m['max_amplitude_error']:>15.4e}"
        )

    out_path = (
        Path(args.out)
        if args.out
        else APP_DIR / "output" / "post_init_diagnostics"
        / f"{args.case_name}_post_init_comparison.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "case_name": args.case_name,
                "true_control": true_control.tolist(),
                "initial_control": initial_control.tolist(),
                "initial_recovery_permutation_matched": (
                    app1.match_sources_by_permutation(true_control, initial_control)
                ),
                "mode": args.mode,
                "feasibility_mode": args.feasibility_mode,
                "max_iter": int(args.max_iter),
                "active_bounds": [list(b) for b in active_bounds],
                "pde": app1.serializable_pde_config(app1.PDE),
                "observation": mesh_data["observation"].metadata(),
                "hessian_fd_check": fd_check,
                "methods": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\ncomparison JSON: {out_path}")


if __name__ == "__main__":
    main()
