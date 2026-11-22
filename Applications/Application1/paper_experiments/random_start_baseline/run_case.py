"""Run ONE random-start baseline optimization through the UNMODIFIED App1 driver.

    PY run_case.py --exp-dir EXP --case-id ID -- <driver args>

Adapted from test scripts/mesh_study_run_case.py (no mesh / dt override here).
Only redirects ``run_optimization.OUTPUT_ROOT`` -> EXP/cases (driver output in
EXP/cases/ID/full) and counts/timestamps calls, delegating to the originals:

* every gt.run_forward / gt.rk4_discrete_source_adjoint call = one actual PDE
  solve, split into: setup -> data generation (forward solve of the TRUE
  control) -> driver pre-optimization (initial-state export) -> the driver's
  post-init solve_counter block = [objective at z0 + GN-diag construction
  (pio.gn_diagonal_scaling) + L-BFGS-B] -> driver post-processing (final
  state, exports);
* J0 = objective of the zero source against the generated data, via the
  driver's own gt.compute_objective at its first call (zero state; no solve).

After the driver returns (or fails) it writes run_instrumentation.json and a
compact run_result.json (controls, objectives, matched errors, counts, final
feasibility check, termination) and the marker DONE (exit status inside).
"""

from __future__ import annotations

import contextlib
import csv
import json
import os
import math
import platform
import sys
import time
import traceback
from pathlib import Path

import numpy as np

def _find_app1() -> Path:
    """Applications/Application1: $APP1_DIR, else the first parent holding the driver.

    Works both from the repository copy (paper_experiments/...) and from the
    copy placed in <EXP>/scripts/ when EXP is under Applications/Application1/output/.
    """
    env = os.environ.get("APP1_DIR")
    if env:
        return Path(env).resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "run_optimization.py").is_file() and (parent / "multi_source_utils.py").is_file():
            return parent
    raise SystemExit("cannot locate Applications/Application1; set APP1_DIR")


APP1 = _find_app1()
sys.path.insert(0, str(APP1))


def _arg_values(argv, flag):
    i = argv.index(flag)
    vals = []
    for t in argv[i + 1:]:
        if t.startswith("--"):
            break
        vals.append(t)
    return vals


def main() -> int:
    argv = sys.argv[1:]
    split = argv.index("--")
    own, driver_args = argv[:split], argv[split + 1:]
    opt = {own[i]: own[i + 1] for i in range(0, len(own), 2)}
    exp = Path(opt["--exp-dir"]).resolve()
    case_id = opt["--case-id"]
    case_dir = exp / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    true_control = np.asarray([float(v) for v in _arg_values(driver_args, "--true-control")])
    start_control = [float(v) for v in _arg_values(driver_args, "--initial-control")]

    p_start = time.perf_counter()
    rec: dict = {"case_id": case_id, "driver_argv": driver_args,
                 "start_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                 "host": platform.node(), "python": sys.executable,
                 "phases": {}, "local_opt_blocks": [], "gn_diag_blocks": []}
    status, err = 0, None
    counts = {"forward": 0, "adjoint": 0}
    marks: dict = {}
    adm_holder: dict = {}

    def snap():
        return dict(counts), time.perf_counter()

    try:
        import run_optimization as ro
        import post_init_optimizers as pio
        from mesh_admissibility import MeshAdmissibility
        gt = ro.gt
        ro.OUTPUT_ROOT = exp / "cases"
        rec["pde_defaults"] = {k: v for k, v in ro.PDE.items() if not isinstance(v, np.ndarray)}

        orig_fwd, orig_adj = gt.run_forward, gt.rk4_discrete_source_adjoint

        def counting_forward(mesh_data, pde, z, *a, **k):
            first_truth = ("data_start" not in marks and "single_start" not in marks
                           and k.get("source_func") is None and not a
                           and np.array_equal(np.asarray(z, dtype=float).reshape(-1), true_control))
            if first_truth:
                marks["data_start"] = snap()
                adm_holder["mesh_data"] = mesh_data
            counts["forward"] += 1
            try:
                return orig_fwd(mesh_data, pde, z, *a, **k)
            finally:
                if first_truth:
                    marks["data_end"] = snap()
                    rec["data_generation_pde"] = {k2: v for k2, v in pde.items()
                                                  if not isinstance(v, (np.ndarray, dict))}

        def counting_adjoint(*a, **k):
            counts["adjoint"] += 1
            return orig_adj(*a, **k)

        gt.run_forward = counting_forward
        gt.rk4_discrete_source_adjoint = counting_adjoint

        orig_obj = gt.compute_objective

        def j0_objective(state, target_state, mesh_data, *a, **k):
            if "J0" not in rec:
                try:
                    rec["J0"] = float(orig_obj(np.zeros_like(np.asarray(state)), target_state,
                                               mesh_data, *a, **{**k, "lam": 0.0}))
                    rec["J0_definition"] = ("objective of the zero source: zero state vs the "
                                            "generated data, driver's gt.compute_objective "
                                            "(eta1, time rule, sensor weights as in the run), "
                                            "no PDE solve")
                except Exception as exc:  # noqa: BLE001
                    rec["J0"] = None
                    rec["J0_error"] = repr(exc)
            return orig_obj(state, target_state, mesh_data, *a, **k)

        gt.compute_objective = j0_objective

        orig_single = ro.run_single_optimization

        def timed_single(*a, **k):
            marks["single_start"] = snap()
            try:
                return orig_single(*a, **k)
            finally:
                marks["single_end"] = snap()

        ro.run_single_optimization = timed_single

        orig_counter = pio.solve_counter

        @contextlib.contextmanager
        def timed_counter(gt_module):
            c0, t0 = snap()
            with orig_counter(gt_module) as inner:
                try:
                    yield inner
                finally:
                    c1, t1 = snap()
                    rec["local_opt_blocks"].append({
                        "seconds": t1 - t0, "forward_solves": c1["forward"] - c0["forward"],
                        "adjoint_solves": c1["adjoint"] - c0["adjoint"]})

        pio.solve_counter = timed_counter

        orig_gn = pio.gn_diagonal_scaling

        def timed_gn(*a, **k):
            c0, t0 = snap()
            try:
                return orig_gn(*a, **k)
            finally:
                c1, t1 = snap()
                rec["gn_diag_blocks"].append({
                    "seconds": t1 - t0, "forward_solves": c1["forward"] - c0["forward"],
                    "adjoint_solves": c1["adjoint"] - c0["adjoint"]})

        pio.gn_diagonal_scaling = timed_gn

        orig_parse = ro.parse_args

        def recording_parse(*a, **k):
            ns = orig_parse(*a, **k)
            rec["resolved_driver_args"] = {key: (val.tolist() if hasattr(val, "tolist") else val)
                                           for key, val in vars(ns).items()}
            (case_dir / "resolved_driver_args.json").write_text(
                json.dumps(rec["resolved_driver_args"], indent=2, default=str) + "\n")
            return ns

        ro.parse_args = recording_parse
        sys.argv = ["run_optimization.py"] + driver_args
        ro.main()
    except SystemExit as exc:
        status = int(exc.code) if isinstance(exc.code, int) else 1
        if status:
            err = f"SystemExit({exc.code})"
    except BaseException as exc:  # noqa: BLE001
        status = 1
        err = "".join(traceback.format_exception(exc))
        traceback.print_exc()
    finally:
        p_end = time.perf_counter()

        def delta(a, b):
            if a not in marks or b not in marks:
                return None
            (ca, ta), (cb, tb) = marks[a], marks[b]
            return {"seconds": tb - ta, "forward_solves": cb["forward"] - ca["forward"],
                    "adjoint_solves": cb["adjoint"] - ca["adjoint"]}

        def total(blocks):
            if not blocks:
                return None
            return {k: sum(b[k] for b in blocks) for k in ("seconds", "forward_solves", "adjoint_solves")}

        ph = rec["phases"]
        if "data_start" in marks:
            c, t = marks["data_start"]
            ph["setup_mesh_velocity_sensors"] = {"seconds": t - p_start, "forward_solves": c["forward"],
                                                 "adjoint_solves": c["adjoint"]}
        ph["data_generation"] = delta("data_start", "data_end")
        ph["between_data_and_optimization"] = delta("data_end", "single_start")
        single = delta("single_start", "single_end")
        ph["run_single_optimization_total"] = single
        lo = total(rec["local_opt_blocks"])
        gn = total(rec["gn_diag_blocks"])
        ph["post_init_total_gn_construction_plus_lbfgsb"] = lo
        ph["scaling_setup_gn_diag_construction"] = gn
        if lo and gn:
            ph["optimization_lbfgsb_excluding_gn_construction"] = {k: lo[k] - gn[k] for k in lo}
        if single and lo:
            ph["driver_pre_and_post_processing_output"] = {k: single[k] - lo[k] for k in single}
        rec["total_counted"] = {"forward_solves": counts["forward"], "adjoint_solves": counts["adjoint"]}
        rec["total_elapsed_seconds_in_process"] = p_end - p_start
        rec["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        rec["exit_status"] = status
        rec["error"] = err
        rec["notes"] = [
            "solve counts are actual gt.run_forward / gt.rk4_discrete_source_adjoint calls; "
            "optimizer nfev/njev count evaluator calls, some served from the evaluator cache",
            "optimization_lbfgsb_excluding_gn_construction includes the driver's objective "
            "evaluation at z0 (inside its solve_counter block), exactly as in the reference runs",
        ]
        (case_dir / "run_instrumentation.json").write_text(json.dumps(rec, indent=2, default=str) + "\n")

        # ---------------- compact per-run result ----------------
        res: dict = {"run_id": case_id, "exit_status": status, "error": err,
                     "start_control_from_argv": start_control,
                     "true_control": true_control.tolist(), "J0_zero_source": rec.get("J0")}
        try:
            summ_path = case_dir / "full" / "optimization_summary.json"
            if summ_path.exists():
                s = json.loads(summ_path.read_text())
                pi = s.get("post_init", {}) or {}
                stage = (pi.get("stages") or [{}])[0]
                scal = stage.get("scaling") or {}
                m = s.get("recovery_permutation_matched") or {}
                mi = s.get("initial_recovery_permutation_matched") or {}
                res.update({
                    "initial_control": s.get("initial_control"),
                    "start_control_matches_argv": bool(np.allclose(s.get("initial_control"), start_control,
                                                                   rtol=0, atol=0)),
                    "final_control": s.get("final_control"),
                    "initial_objective": s.get("initial_objective"),
                    "final_objective": s.get("final_objective"),
                    "final_gradient_norm": s.get("final_gradient_norm"),
                    "final_over_J0": (s["final_objective"] / rec["J0"]
                                      if rec.get("J0") else None),
                    "initial_over_J0": (s["initial_objective"] / rec["J0"]
                                        if rec.get("J0") else None),
                    "max_position_error_m": m.get("max_position_error_m"),
                    "max_amplitude_error": m.get("max_amplitude_error"),
                    "position_errors_m": m.get("position_errors_m"),
                    "amplitude_errors": m.get("amplitude_errors"),
                    "permutation": m.get("permutation"),
                    "initial_max_position_error_m": mi.get("max_position_error_m"),
                    "initial_max_amplitude_error": mi.get("max_amplitude_error"),
                    "success_flag": s.get("success"),
                    "message": s.get("message"),
                    "iterations": s.get("iterations"),
                    "n_objective_evaluations": pi.get("total_objective_evaluations"),
                    "n_gradient_evaluations": pi.get("total_gradient_evaluations"),
                    "lbfgsb_maxcor": pi.get("lbfgsb_maxcor"),
                    "scaling": pi.get("scaling"),
                    "gn_condition_unscaled": scal.get("condition_number_unscaled"),
                    "gn_condition_scaled": scal.get("condition_number_scaled"),
                    "gn_n_floored": scal.get("n_floored"),
                    "feasibility": s.get("feasibility"),
                    "pde": s.get("pde"),
                    "active_bounds": s.get("active_bounds"),
                })
                # Final feasibility check, independent of the driver: centers in the fluid mesh
                # and inside the L-BFGS-B box; amplitudes inside their box.
                md = adm_holder.get("mesh_data")
                if md is None:
                    from mesh_data_config import build_mesh_data
                    md = build_mesh_data()
                from mesh_admissibility import MeshAdmissibility
                adm = MeshAdmissibility(md)
                zf = np.asarray(s["final_control"], dtype=float)
                M = zf.size // 3
                inside = [bool(adm.is_point_in_domain(zf[3 * i], zf[3 * i + 1])) for i in range(M)]
                clear = [float(adm.signed_clearance(zf[3 * i], zf[3 * i + 1])) for i in range(M)]
                b = s.get("active_bounds")
                in_box = bool(all(lo_ - 1e-12 <= v <= hi_ + 1e-12 for v, (lo_, hi_) in zip(zf, b))) if b else None
                res["final_feasibility_check"] = {
                    "centers_in_fluid_mesh": inside, "all_centers_feasible": bool(all(inside)),
                    "signed_clearance_m": clear, "within_lbfgsb_box": in_box}
                finite = bool(np.all(np.isfinite(zf))) and all(
                    isinstance(s.get(k), (int, float)) and math.isfinite(s.get(k))
                    for k in ("final_objective", "final_gradient_norm"))
                res["numerical_flags"] = {
                    "finite_final_values": finite,
                    "abnormal_termination": "ABNORMAL" in str(s.get("message", "")).upper(),
                    "iteration_limit": ("ITERATION" in str(s.get("message", "")).upper()
                                        or "LIMIT" in str(s.get("message", "")).upper()),
                    "raw_infeasible_evaluations": (s.get("feasibility") or {}).get("raw_infeasible_evaluations"),
                }
            hist = case_dir / "full" / "optimization_history.csv"
            if hist.exists():
                with hist.open() as fh:
                    res["history_rows"] = sum(1 for _ in csv.reader(fh)) - 1
        except Exception as exc:  # noqa: BLE001
            res["result_extraction_error"] = "".join(traceback.format_exception(exc))
        res["pde_solves"] = {
            "data_generation": ph.get("data_generation"),
            "scaling_setup": ph.get("scaling_setup_gn_diag_construction"),
            "optimization": ph.get("optimization_lbfgsb_excluding_gn_construction"),
            "post_init_total": ph.get("post_init_total_gn_construction_plus_lbfgsb"),
            "driver_output_overhead": ph.get("driver_pre_and_post_processing_output"),
            "between_data_and_optimization": ph.get("between_data_and_optimization"),
            "total_counted": rec["total_counted"],
        }
        (case_dir / "run_result.json").write_text(json.dumps(res, indent=2, default=str) + "\n")
        (case_dir / "DONE").write_text(f"exit_status {status}\n{time.strftime('%F %T')}\n")
        print(f"[run_case] {case_id}: instrumentation + run_result written (status {status})", flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
