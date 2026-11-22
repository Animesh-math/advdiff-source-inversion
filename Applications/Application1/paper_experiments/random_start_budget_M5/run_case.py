"""Run ONE budget-limited random start through the UNMODIFIED App1 driver.

    PY run_case.py --exp-dir EXP --case-id rNN --budget-sweeps B --seed S -- <driver args>

Nothing in the solver, objective, GN-diag scaling or L-BFGS-B code is changed.
The wrapper:

* redirects run_optimization.OUTPUT_ROOT -> EXP/cases (driver output in EXP/cases/<id>/full);
* counts every gt.run_forward / gt.rk4_discrete_source_adjoint call (= one PDE sweep);
* ARMS a sweep budget for exactly the driver's post-init solve_counter block
  (objective at the start + GN-diag construction + every L-BFGS-B evaluation,
  line search included) -- the same block that holds the reference run's
  84 fwd / 82 adj local-refinement count;
* guard (PDE-evaluation level): before each solve inside the block,
    - GN-diag construction is only entered if its full cost (n+1 fwd, n adj)
      plus one complete evaluation (2 sweeps) fits in the remaining allowance;
    - during L-BFGS-B a forward (new objective evaluation) is only started if
      >= 2 sweeps remain (so its adjoint/gradient can be completed);
    - any other solve needs >= 1 sweep.
  If the guard refuses, BudgetExhausted is raised BEFORE the solve starts;
  the BudgetedOptimizer catches it and returns the best feasible fully
  evaluated point (objective and gradient computed, all centers in the fluid
  mesh) with message "BUDGET EXHAUSTED ..." and status 99 (never "success").
* the driver's post-processing after the block (final re-evaluation, state
  export) is outside the budget and reported separately, exactly like the
  reference run's 4 fwd / 2 adj overhead.

Outputs in EXP/cases/<id>/: run_instrumentation.json, evaluations.csv (every
L-BFGS-B evaluation with cumulative sweeps), iterates.csv (accepted iterates),
run_result.json (everything needed by the report), DONE.
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


class BudgetExhausted(Exception):
    pass


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
    budget = int(opt["--budget-sweeps"])
    seed = int(opt["--seed"])
    case_dir = exp / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    true_control = np.asarray([float(v) for v in _arg_values(driver_args, "--true-control")])
    start_control = [float(v) for v in _arg_values(driver_args, "--initial-control")]
    M = true_control.size // 3

    p_start = time.perf_counter()
    rec: dict = {"case_id": case_id, "seed": seed, "budget_sweeps": budget, "driver_argv": driver_args,
                 "start_time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "host": platform.node(),
                 "python": sys.executable, "phases": {}, "local_opt_blocks": [], "gn_diag_blocks": []}
    status, err = 0, None
    counts = {"forward": 0, "adjoint": 0}
    B = {"armed": False, "c0": None, "phase": None, "refusals": [], "budget_stop": False,
         "scale": None, "gn_cost": None}
    evals: list[dict] = []          # every L-BFGS-B evaluation (optimizer variables)
    iterates: list[dict] = []       # accepted iterates (callback)
    marks: dict = {}
    holder: dict = {}

    def snap():
        return dict(counts), time.perf_counter()

    def used():
        c0 = B["c0"]
        return (counts["forward"] - c0["forward"]) + (counts["adjoint"] - c0["adjoint"])

    def guard(kind: str):
        if not B["armed"]:
            return
        remaining = budget - used()
        need = 2 if (kind == "forward" and B["phase"] == "lbfgsb") else 1
        if remaining < need:
            B["refusals"].append({"kind": kind, "phase": B["phase"], "remaining": remaining, "needed": need,
                                  "used": used()})
            raise BudgetExhausted(f"{kind} refused in phase {B['phase']}: remaining {remaining} < {need}")

    try:
        import run_optimization as ro
        import post_init_optimizers as pio
        from mesh_admissibility import MeshAdmissibility
        from scipy.optimize import OptimizeResult
        gt = ro.gt
        ro.OUTPUT_ROOT = exp / "cases"

        orig_fwd, orig_adj = gt.run_forward, gt.rk4_discrete_source_adjoint

        def counting_forward(mesh_data, pde, z, *a, **k):
            first_truth = ("data_start" not in marks and "single_start" not in marks
                           and k.get("source_func") is None and not a
                           and np.array_equal(np.asarray(z, dtype=float).reshape(-1), true_control))
            if first_truth:
                marks["data_start"] = snap()
                holder["mesh_data"] = mesh_data
            guard("forward")
            counts["forward"] += 1
            try:
                return orig_fwd(mesh_data, pde, z, *a, **k)
            finally:
                if first_truth:
                    marks["data_end"] = snap()

        def counting_adjoint(*a, **k):
            guard("adjoint")
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
                except Exception as exc:  # noqa: BLE001
                    rec["J0"], rec["J0_error"] = None, repr(exc)
            return orig_obj(state, target_state, mesh_data, *a, **k)

        gt.compute_objective = j0_objective

        orig_single = ro.run_single_optimization

        def timed_single(*a, **k):
            marks["single_start"] = snap()
            holder["admissibility"] = a[4] if len(a) > 4 else k.get("admissibility")
            try:
                return orig_single(*a, **k)
            finally:
                marks["single_end"] = snap()

        ro.run_single_optimization = timed_single

        orig_counter = pio.solve_counter

        @contextlib.contextmanager
        def budgeted_counter(gt_module):
            c0, t0 = snap()
            B["armed"], B["c0"], B["phase"] = True, dict(counts), "objective_at_start"
            try:
                with orig_counter(gt_module) as inner:
                    yield inner
            finally:
                B["armed"], B["phase"] = False, None
                c1, t1 = snap()
                rec["local_opt_blocks"].append({
                    "seconds": t1 - t0, "forward_solves": c1["forward"] - c0["forward"],
                    "adjoint_solves": c1["adjoint"] - c0["adjoint"]})

        pio.solve_counter = budgeted_counter

        orig_gn = pio.gn_diagonal_scaling

        def budgeted_gn(hess, z_active, *a, **k):
            n = int(np.asarray(z_active).size)
            need = (n + 1) + n + 2
            if B["armed"] and budget - used() < need:
                B["refusals"].append({"kind": "gn_construction", "remaining": budget - used(), "needed": need})
                raise BudgetExhausted(f"GN-diag construction needs {need} sweeps incl. one evaluation")
            B["phase"] = "gn_diag_construction"
            c0, t0 = snap()
            try:
                scale, record = orig_gn(hess, z_active, *a, **k)
                B["scale"] = np.asarray(scale, dtype=np.float64).copy()
                return scale, record
            finally:
                c1, t1 = snap()
                blk = {"seconds": t1 - t0, "forward_solves": c1["forward"] - c0["forward"],
                       "adjoint_solves": c1["adjoint"] - c0["adjoint"]}
                rec["gn_diag_blocks"].append(blk)
                B["gn_cost"] = blk

        pio.gn_diagonal_scaling = budgeted_gn

        BaseOptimizer = ro.Optimizer

        class BudgetedOptimizer(BaseOptimizer):
            """Same L-BFGS-B call; adds evaluation logging and a clean budget stop."""

            def optimize(self, initial_guess, mesh_data=None, problem_data=None):
                adm = holder["admissibility"]
                scale = B["scale"] if B["scale"] is not None else 1.0
                f_inner, g_inner, cb_inner = self.objective_function, self.gradient_function, self.callback
                last = {}

                def control_of(y):
                    return np.asarray(y, dtype=np.float64) / scale

                def f_wrapped(y, *a):
                    yv = np.array(y, dtype=np.float64, copy=True)
                    val = f_inner(yv, *a)
                    z = control_of(yv)
                    feas = bool(adm.validate_sources_in_domain(z))
                    evals.append({"eval": len(evals) + 1, "y": yv, "control": z, "objective": float(val),
                                  "feasible": feas, "gradient_done": False,
                                  "fwd_used": counts["forward"] - B["c0"]["forward"],
                                  "adj_used": counts["adjoint"] - B["c0"]["adjoint"]})
                    last["y"] = yv
                    return val

                def g_wrapped(y, *a):
                    g = g_inner(y, *a)
                    yv = np.asarray(y, dtype=np.float64)
                    for e in reversed(evals):
                        if np.array_equal(e["y"], yv):
                            if not e["gradient_done"]:
                                e["gradient_done"] = True
                                e["grad"] = np.asarray(g, dtype=np.float64).copy()
                                e["fwd_used"] = counts["forward"] - B["c0"]["forward"]
                                e["adj_used"] = counts["adjoint"] - B["c0"]["adjoint"]
                            break
                    return g

                def cb_wrapped(y, *a):
                    yv = np.asarray(y, dtype=np.float64)
                    z = control_of(yv)
                    match = next((e for e in reversed(evals) if np.array_equal(e["y"], yv)), None)
                    iterates.append({"iteration": len(iterates) + 1,
                                     "objective": None if match is None else match["objective"],
                                     "control": z, "feasible": bool(adm.validate_sources_in_domain(z)),
                                     "sweeps_used": used()})
                    if cb_inner is not None:
                        return cb_inner(y, *a)
                    return None

                self.objective_function, self.gradient_function, self.callback = f_wrapped, g_wrapped, cb_wrapped
                B["phase"] = "lbfgsb"
                try:
                    res = super().optimize(initial_guess, mesh_data=mesh_data, problem_data=problem_data)
                    res.budget_stop = False
                    return res
                except BudgetExhausted as exc:
                    B["budget_stop"] = True
                    full = [e for e in evals if e["gradient_done"] and e["feasible"]
                            and math.isfinite(e["objective"])]
                    if not full:
                        raise RuntimeError("budget exhausted with no feasible fully evaluated point") from exc
                    best = min(full, key=lambda e: e["objective"])
                    B["best_eval"] = best["eval"]
                    res = OptimizeResult(
                        x=best["y"].copy(), fun=float(best["objective"]), jac=best["grad"].copy(),
                        nit=len(iterates), nfev=len(evals), njev=sum(e["gradient_done"] for e in evals),
                        status=99, success=False,
                        message=(f"BUDGET EXHAUSTED ({used()}/{budget} sweeps used; {exc}); returned best "
                                 f"feasible fully evaluated point = evaluation {best['eval']}"))
                    res.budget_stop = True
                    return res
                finally:
                    B["phase"] = "after_lbfgsb"
                    self.objective_function, self.gradient_function, self.callback = f_inner, g_inner, cb_inner

        ro.Optimizer = BudgetedOptimizer

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
        ph["data_generation"] = delta("data_start", "data_end")
        ph["before_budget_block_driver_smoke"] = None
        single = delta("single_start", "single_end")
        ph["run_single_optimization_total"] = single
        lo = total(rec["local_opt_blocks"])
        gn = total(rec["gn_diag_blocks"])
        ph["budgeted_block_total"] = lo
        ph["scaling_setup_gn_diag_construction"] = gn
        if lo and gn:
            ph["budgeted_block_excluding_gn"] = {k: lo[k] - gn[k] for k in lo}
        if single and lo:
            ph["driver_overhead_outside_budget"] = {k: single[k] - lo[k] for k in single}
        rec["total_counted"] = {"forward_solves": counts["forward"], "adjoint_solves": counts["adjoint"]}
        rec["budget"] = {"allowance_sweeps": budget,
                         "used_forward": lo["forward_solves"] if lo else None,
                         "used_adjoint": lo["adjoint_solves"] if lo else None,
                         "used_sweeps": (lo["forward_solves"] + lo["adjoint_solves"]) if lo else None,
                         "unused_sweeps": (budget - lo["forward_solves"] - lo["adjoint_solves"]) if lo else None,
                         "exceeded": bool(lo and lo["forward_solves"] + lo["adjoint_solves"] > budget),
                         "budget_stop": B["budget_stop"], "refusals": B["refusals"],
                         "best_eval_returned": B.get("best_eval")}
        rec["total_elapsed_seconds_in_process"] = p_end - p_start
        rec["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        rec["exit_status"] = status
        rec["error"] = err
        (case_dir / "run_instrumentation.json").write_text(json.dumps(rec, indent=2, default=str) + "\n")

        # ---- evaluation and iterate logs ----
        names = [f"{c}{i}" for i in range(1, M + 1) for c in ("x", "y", "A")]
        with (case_dir / "evaluations.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["eval", "objective", "feasible", "gradient_done", "fwd_used", "adj_used",
                        "sweeps_used"] + names)
            for e in evals:
                w.writerow([e["eval"], repr(e["objective"]), e["feasible"], e["gradient_done"], e["fwd_used"],
                            e["adj_used"], e["fwd_used"] + e["adj_used"]] + [repr(float(v)) for v in e["control"]])
        with (case_dir / "iterates.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["iteration", "objective", "feasible", "sweeps_used"] + names)
            for it in iterates:
                w.writerow([it["iteration"], repr(it["objective"]), it["feasible"], it["sweeps_used"]]
                           + [repr(float(v)) for v in it["control"]])

        # ---- compact result ----
        res: dict = {"run_id": case_id, "seed": seed, "budget_sweeps": budget, "exit_status": status,
                     "error": err, "start_control": start_control, "true_control": true_control.tolist(),
                     "J0_zero_source": rec.get("J0")}
        try:
            from mesh_admissibility import MeshAdmissibility
            import run_optimization as ro2
            md = holder.get("mesh_data")
            if md is None:
                from mesh_data_config import build_mesh_data
                md = build_mesh_data()
            adm = MeshAdmissibility(md)

            def feas_check(z):
                z = np.asarray(z, dtype=float)
                inside = [bool(adm.is_point_in_domain(z[3 * i], z[3 * i + 1])) for i in range(M)]
                return {"centers_in_fluid_mesh": inside, "all_centers_feasible": bool(all(inside)),
                        "signed_clearance_m": [float(adm.signed_clearance(z[3 * i], z[3 * i + 1]))
                                               for i in range(M)]}

            def matched(z):
                m = ro2.match_sources_by_permutation(true_control, np.asarray(z, dtype=float))
                return {k: m.get(k) for k in ("permutation", "position_errors_m", "amplitude_errors",
                                              "max_position_error_m", "max_amplitude_error")}

            summ_path = case_dir / "full" / "optimization_summary.json"
            s = json.loads(summ_path.read_text()) if summ_path.exists() else {}
            full_feas = [e for e in evals if e["gradient_done"] and e["feasible"] and math.isfinite(e["objective"])]
            best = min(full_feas, key=lambda e: e["objective"]) if full_feas else None
            last_it = iterates[-1] if iterates else None
            msg = str(s.get("message", ""))
            if status != 0:
                stop = "error"
            elif B["budget_stop"]:
                stop = "budget_exhausted"
            elif s.get("success"):
                stop = "optimizer_converged"
            elif "ITERATION" in msg.upper() or "LIMIT" in msg.upper():
                stop = "iteration_limit"
            elif "ABNORMAL" in msg.upper():
                stop = "abnormal_line_search"
            else:
                stop = "other_optimizer_stop"
            pi = s.get("post_init") or {}
            stage = (pi.get("stages") or [{}])[0]
            scal = stage.get("scaling") or {}
            res.update({
                "stop_reason": stop,
                "optimizer_message": msg, "optimizer_success_flag": s.get("success"),
                "iterations": s.get("iterations"), "n_evaluations": len(evals),
                "n_full_evaluations": sum(e["gradient_done"] for e in evals),
                "initial_control_driver": s.get("initial_control"),
                "start_matches_argv_exactly": bool(s.get("initial_control") is not None and
                                                   np.array_equal(s["initial_control"], start_control)),
                "initial_objective": s.get("initial_objective"),
                "final_control": s.get("final_control"), "final_objective": s.get("final_objective"),
                "final_gradient_norm": s.get("final_gradient_norm"),
                "final_feasibility": feas_check(s["final_control"]) if s.get("final_control") else None,
                "final_matched": matched(s["final_control"]) if s.get("final_control") else None,
                "best_feasible_eval": None if best is None else {
                    "eval": best["eval"], "objective": best["objective"], "control": best["control"].tolist(),
                    "sweeps_used_when_evaluated": best["fwd_used"] + best["adj_used"],
                    "feasibility": feas_check(best["control"]), "matched": matched(best["control"])},
                "last_accepted_iterate": None if last_it is None else {
                    "iteration": last_it["iteration"], "objective": last_it["objective"],
                    "control": last_it["control"].tolist(), "feasible": last_it["feasible"]},
                "objective_history_iterates": [it["objective"] for it in iterates],
                "initial_matched": matched(start_control),
                "gn_condition_unscaled": scal.get("condition_number_unscaled"),
                "gn_condition_scaled": scal.get("condition_number_scaled"),
                "gn_n_floored": scal.get("n_floored"), "lbfgsb_maxcor": pi.get("lbfgsb_maxcor"),
                "driver_feasibility_stats": s.get("feasibility"),
                "pde": s.get("pde"),
            })
            res["final_over_J0"] = (s["final_objective"] / rec["J0"]) if (rec.get("J0") and s.get("final_objective") is not None) else None
            gnc = gn or {"forward_solves": 0, "adjoint_solves": 0}
            res["pde_sweeps"] = {
                "budget_allowance": budget,
                "objective_at_start": {"forward": 1 if lo and lo["forward_solves"] > 0 else 0, "adjoint": 0},
                "scaling_setup_gn_diag": {"forward": gnc["forward_solves"], "adjoint": gnc["adjoint_solves"]},
                "lbfgsb": ({"forward": lo["forward_solves"] - gnc["forward_solves"] - 1,
                            "adjoint": lo["adjoint_solves"] - gnc["adjoint_solves"]} if lo else None),
                "budgeted_total": ({"forward": lo["forward_solves"], "adjoint": lo["adjoint_solves"],
                                    "sweeps": lo["forward_solves"] + lo["adjoint_solves"]} if lo else None),
                "unused_allowance": rec["budget"]["unused_sweeps"],
                "exceeded": rec["budget"]["exceeded"],
                "outside_budget_data_generation": ph.get("data_generation"),
                "outside_budget_driver_overhead": ph.get("driver_overhead_outside_budget"),
                "total_counted_in_process": rec["total_counted"],
            }
        except Exception as exc:  # noqa: BLE001
            res["result_extraction_error"] = "".join(traceback.format_exception(exc))
        (case_dir / "run_result.json").write_text(json.dumps(res, indent=2, default=str) + "\n")
        (case_dir / "DONE").write_text(f"exit_status {status}\n{time.strftime('%F %T')}\n")
        print(f"[run_case] {case_id}: written (status {status}, stop {res.get('stop_reason')})", flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
