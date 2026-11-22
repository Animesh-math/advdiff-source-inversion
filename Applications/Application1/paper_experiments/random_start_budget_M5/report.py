"""Report for the budget-matched five-source random-start benchmark.

    PY report.py --exp-dir EXP [--partial]

Writes EXP/report/: REPORT.md, comparison.{csv,json}, per_start.{csv,json},
reconstruction_comparison.{png,pdf}, reconstruction_per_start.{png,pdf},
best_objective_vs_sweeps.{png,pdf}. The overall random-start result is chosen
ONLY by the lowest data misfit (objective) among feasible fully evaluated
points; errors to the truth are computed afterwards.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import math
import sys
import time
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


def fmt(v, p=3):
    if v is None:
        return "-"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, float):
        if v == 0.0:
            return "0"
        if not math.isfinite(v):
            return str(v)
        return f"{v:.{p}g}" if (abs(v) >= 1e-3 and abs(v) < 1e5) else f"{v:.{p}e}"
    return str(v)


def md_table(header, rows):
    esc = lambda t: str(t).replace("|", "\\|")  # noqa: E731
    out = ["| " + " | ".join(esc(h) for h in header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(esc(fmt(c)) for c in r) + " |" for r in rows]
    return "\n".join(out)


def read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def draw_domain(ax, md):
    coord = np.asarray(md["coord"])
    segs = []
    for name, edges in md["boundary_edges_by_name"].items():
        e = np.asarray(edges).reshape(-1, 2)
        segs.append(coord[e][:, :, :2])
    from matplotlib.collections import LineCollection
    ax.add_collection(LineCollection(np.concatenate(segs), colors="0.35", linewidths=0.6))
    ax.set_xlim(coord[:, 0].min() - 5, coord[:, 0].max() + 5)
    ax.set_ylim(coord[:, 1].min() - 5, coord[:, 1].max() + 5)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--partial", action="store_true")
    a = ap.parse_args()
    exp = Path(a.exp_dir).resolve()
    rep = exp / "report"
    rep.mkdir(exist_ok=True)
    cfg = exp / "config"
    ref = json.loads((cfg / "reference.json").read_text())
    bud = json.loads((cfg / "budget.json").read_text())
    starts = json.loads((cfg / "starts.json").read_text())
    order = [s["run_id"] for s in starts["starts"]]

    runs = {}
    for s in starts["starts"]:
        p = exp / "cases" / s["run_id"] / "run_result.json"
        if p.exists():
            runs[s["run_id"]] = json.loads(p.read_text())
    finished = [r for r in order if r in runs]
    ok = [r for r in finished if runs[r].get("exit_status") == 0 and runs[r].get("best_feasible_eval")]
    J0 = next((runs[r].get("J0_zero_source") for r in ok if runs[r].get("J0_zero_source")), None)

    # ---------------- per-start table ----------------
    per = []
    for rid in order:
        s = next(x for x in starts["starts"] if x["run_id"] == rid)
        r = runs.get(rid)
        row = {"run": rid, "seed": s["seed"], "allowance_sweeps": s["budget_sweeps"],
               "initial_control": s["initial_control"], "status": "not finished"}
        if r is not None:
            ps = r.get("pde_sweeps") or {}
            bt = ps.get("budgeted_total") or {}
            gn = ps.get("scaling_setup_gn_diag") or {}
            lb = ps.get("lbfgsb") or {}
            be = r.get("best_feasible_eval") or {}
            bm = be.get("matched") or {}
            row.update({
                "status": "ok" if r.get("exit_status") == 0 else f"exit {r.get('exit_status')}",
                "stop_reason": r.get("stop_reason"),
                "optimizer_message": r.get("optimizer_message"),
                "iterations": r.get("iterations"),
                "evaluations": r.get("n_evaluations"),
                "fwd_used": bt.get("forward"), "adj_used": bt.get("adjoint"), "sweeps_used": bt.get("sweeps"),
                "unused_sweeps": ps.get("unused_allowance"), "exceeded": ps.get("exceeded"),
                "gn_fwd": gn.get("forward"), "gn_adj": gn.get("adjoint"),
                "start_obj_fwd": (ps.get("objective_at_start") or {}).get("forward"),
                "lbfgsb_fwd": lb.get("forward"), "lbfgsb_adj": lb.get("adjoint"),
                "outside_budget_overhead_fwd": (ps.get("outside_budget_driver_overhead") or {}).get("forward_solves"),
                "outside_budget_overhead_adj": (ps.get("outside_budget_driver_overhead") or {}).get("adjoint_solves"),
                "initial_objective": r.get("initial_objective"),
                "best_feasible_objective": be.get("objective"),
                "best_over_J0": (be.get("objective") / J0) if (J0 and be.get("objective") is not None) else None,
                "best_feasible_control": be.get("control"),
                "best_eval_index": be.get("eval"),
                "best_feasible": (be.get("feasibility") or {}).get("all_centers_feasible"),
                "max_center_err_m": bm.get("max_position_error_m"),
                "max_amp_err": bm.get("max_amplitude_error"),
                "center_errs_m": bm.get("position_errors_m"), "amp_errs": bm.get("amplitude_errors"),
                "final_control_driver": r.get("final_control"),
                "final_objective_driver": r.get("final_objective"),
                "final_feasible_driver": (r.get("final_feasibility") or {}).get("all_centers_feasible"),
                "last_accepted_iterate": r.get("last_accepted_iterate"),
                "gn_cond_unscaled": r.get("gn_condition_unscaled"), "gn_cond_scaled": r.get("gn_condition_scaled"),
                "raw_infeasible_evaluations": (r.get("driver_feasibility_stats") or {}).get("raw_infeasible_evaluations"),
            })
        per.append(row)
    (rep / "per_start.json").write_text(json.dumps(per, indent=2, default=str) + "\n")
    cols = ["run", "seed", "allowance_sweeps", "status", "stop_reason", "iterations", "evaluations", "fwd_used",
            "adj_used", "sweeps_used", "unused_sweeps", "exceeded", "gn_fwd", "gn_adj", "start_obj_fwd",
            "lbfgsb_fwd", "lbfgsb_adj", "outside_budget_overhead_fwd", "outside_budget_overhead_adj",
            "initial_objective", "best_feasible_objective", "best_over_J0", "best_feasible", "max_center_err_m",
            "max_amp_err", "raw_infeasible_evaluations", "gn_cond_unscaled", "gn_cond_scaled"]
    with (rep / "per_start.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols + ["initial_control", "best_feasible_control"])
        for row in per:
            w.writerow([row.get(c) for c in cols] + [json.dumps(row.get("initial_control")),
                                                      json.dumps(row.get("best_feasible_control"))])

    # ---------------- comparison ----------------
    rb = bud["reference_local_refinement"]
    ri = bud["reference_initialization"]
    rm = ref["recovery_permutation_matched"]
    ours = {"method": "sensitivity initialization + GN-diag L-BFGS-B (saved paper run)",
            "init_fwd": ri["forward"], "init_adj": ri["adjoint"],
            "local_fwd": rb["forward"], "local_adj": rb["adjoint"],
            "total_fwd": ri["forward"] + rb["forward"], "total_adj": ri["adjoint"] + rb["adjoint"],
            "total_sweeps": bud["total_budget"]["sweeps"],
            "gn_setup_fwd": 16, "gn_setup_adj": 15,
            "iterations": ref["iterations"], "objective": ref["final_objective"],
            "objective_over_J0": (ref["final_objective"] / J0) if J0 else None,
            "max_center_err_m": rm["max_position_error_m"], "max_amp_err": rm["max_amplitude_error"],
            "feasible": ref["feasibility"]["raw_infeasible_evaluations"] == 0,
            "stop": ref["message"], "control": ref["final_control"]}
    best = None
    if ok:
        best_id = min(ok, key=lambda r: runs[r]["best_feasible_eval"]["objective"])
        br = next(p for p in per if p["run"] == best_id)
        agg_f = sum(p["fwd_used"] or 0 for p in per if p["run"] in finished)
        agg_a = sum(p["adj_used"] or 0 for p in per if p["run"] in finished)
        agg_gn_f = sum(p["gn_fwd"] or 0 for p in per if p["run"] in finished)
        agg_gn_a = sum(p["gn_adj"] or 0 for p in per if p["run"] in finished)
        agg_unused = sum(p["unused_sweeps"] or 0 for p in per if p["run"] in finished)
        best = {"method": f"best of {len(finished)} random starts by lowest objective ({best_id})",
                "selected_run": best_id, "selection_rule": "lowest best-feasible objective (data misfit) only",
                "own_fwd": br["fwd_used"], "own_adj": br["adj_used"], "own_sweeps": br["sweeps_used"],
                "aggregate_fwd_all_starts": agg_f, "aggregate_adj_all_starts": agg_a,
                "aggregate_sweeps_all_starts": agg_f + agg_a,
                "aggregate_gn_setup_fwd": agg_gn_f, "aggregate_gn_setup_adj": agg_gn_a,
                "aggregate_unused_sweeps": agg_unused, "total_budget_sweeps": bud["total_budget"]["sweeps"],
                "iterations": br["iterations"], "aggregate_iterations": sum(p.get("iterations") or 0 for p in per),
                "objective": br["best_feasible_objective"], "objective_over_J0": br["best_over_J0"],
                "max_center_err_m": br["max_center_err_m"], "max_amp_err": br["max_amp_err"],
                "feasible": br["best_feasible"], "stop": br["stop_reason"], "control": br["best_feasible_control"],
                "n_finished": len(finished)}
    (rep / "comparison.json").write_text(json.dumps({"ours": ours, "best_random": best, "J0": J0,
                                                     "budget": bud}, indent=2, default=str) + "\n")
    with (rep / "comparison.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "forward_sweeps", "adjoint_sweeps", "total_sweeps", "init_fwd", "init_adj",
                    "local_or_setup+opt_fwd", "local_or_setup+opt_adj", "gn_setup_fwd", "gn_setup_adj",
                    "iterations", "objective", "objective_over_J0", "max_center_err_m", "max_amp_err",
                    "feasible", "stop"])
        w.writerow([ours["method"], ours["total_fwd"], ours["total_adj"], ours["total_sweeps"], ours["init_fwd"],
                    ours["init_adj"], ours["local_fwd"], ours["local_adj"], 16, 15, ours["iterations"],
                    ours["objective"], ours["objective_over_J0"], ours["max_center_err_m"], ours["max_amp_err"],
                    ours["feasible"], ours["stop"]])
        if best:
            w.writerow([best["method"] + " -- AGGREGATE cost of all starts", best["aggregate_fwd_all_starts"],
                        best["aggregate_adj_all_starts"], best["aggregate_sweeps_all_starts"], 0, 0,
                        best["aggregate_fwd_all_starts"], best["aggregate_adj_all_starts"],
                        best["aggregate_gn_setup_fwd"], best["aggregate_gn_setup_adj"],
                        best["aggregate_iterations"], best["objective"], best["objective_over_J0"],
                        best["max_center_err_m"], best["max_amp_err"], best["feasible"], best["stop"]])

    # ---------------- figures ----------------
    figs = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mesh_data_config import build_mesh_data
        md = build_mesh_data()
        tc = np.asarray(ref["true_control"]).reshape(-1, 3)

        # reconstruction comparison
        if best:
            fig, axes = plt.subplots(1, 2, figsize=(9, 9))
            for ax, ctrl, init, title in (
                    (axes[0], ours["control"], ref["initialized_control"],
                     f"Ours: sensitivity init + L-BFGS-B\n{ours['total_fwd']} fwd + {ours['total_adj']} adj"),
                    (axes[1], best["control"], next(p for p in per if p["run"] == best["selected_run"])["initial_control"],
                     f"Best random start ({best['selected_run']}, chosen by misfit)\n"
                     f"all {best['n_finished']} starts: {best['aggregate_fwd_all_starts']} fwd + "
                     f"{best['aggregate_adj_all_starts']} adj")):
                draw_domain(ax, md)
                c = np.asarray(ctrl).reshape(-1, 3)
                i0 = np.asarray(init).reshape(-1, 3)
                ax.scatter(tc[:, 0], tc[:, 1], s=260, marker="*", c="tab:green", edgecolor="k", label="true", zorder=3)
                ax.scatter(i0[:, 0], i0[:, 1], s=50, marker="s", facecolor="none", edgecolor="tab:blue",
                           label="start / initialized", zorder=3)
                ax.scatter(c[:, 0], c[:, 1], s=60, marker="o", c="tab:red", label="final", zorder=4)
                for k in range(len(c)):
                    ax.annotate(f"{c[k, 2]:.2f}", (c[k, 0], c[k, 1]), xytext=(4, 4), textcoords="offset points",
                                fontsize=7, color="tab:red")
                ax.set_title(title, fontsize=9)
            axes[0].legend(loc="lower left", fontsize=7)
            fig.tight_layout()
            for ext in ("png", "pdf"):
                fig.savefig(rep / f"reconstruction_comparison.{ext}", dpi=160)
            plt.close(fig)
            figs.append("reconstruction_comparison.png")

        # per-start reconstructions
        fig, axes = plt.subplots(2, 5, figsize=(17, 13))
        for ax, row in zip(axes.ravel(), per):
            draw_domain(ax, md)
            ax.scatter(tc[:, 0], tc[:, 1], s=160, marker="*", c="tab:green", edgecolor="k", zorder=3)
            i0 = np.asarray(row["initial_control"]).reshape(-1, 3)
            ax.scatter(i0[:, 0], i0[:, 1], s=30, marker="s", facecolor="none", edgecolor="tab:blue", zorder=3)
            if row.get("best_feasible_control"):
                c = np.asarray(row["best_feasible_control"]).reshape(-1, 3)
                ax.scatter(c[:, 0], c[:, 1], s=36, c="tab:red", zorder=4)
                for k in range(len(c)):
                    ax.plot([i0[k, 0], c[k, 0]], [i0[k, 1], c[k, 1]], color="tab:blue", lw=0.5, alpha=0.6)
            ax.set_title(f"{row['run']} seed {row['seed']}\n{row.get('stop_reason', '-')}; "
                         f"J/J0={fmt(row.get('best_over_J0'))}; max err {fmt(row.get('max_center_err_m'))} m",
                         fontsize=8)
            ax.set_xlabel("")
            ax.set_ylabel("")
        fig.suptitle("Random starts: start (blue squares) -> best feasible point (red); truth (green stars)")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(rep / f"reconstruction_per_start.{ext}", dpi=130)
        plt.close(fig)
        figs.append("reconstruction_per_start.png")

        # best objective vs cumulative sweeps
        fig, ax = plt.subplots(figsize=(9, 5))
        offset, xs, ys, bounds = 0, [], [], []
        best_so_far = math.inf
        for rid in order:
            ev_path = exp / "cases" / rid / "evaluations.csv"
            r = runs.get(rid)
            if r is None or not ev_path.exists():
                break
            for e in read_csv(ev_path):
                if e["gradient_done"] == "True" and e["feasible"] == "True":
                    best_so_far = min(best_so_far, float(e["objective"]))
                if math.isfinite(best_so_far):
                    xs.append(offset + int(e["sweeps_used"]))
                    ys.append(best_so_far / J0 if J0 else best_so_far)
            used_ = ((r.get("pde_sweeps") or {}).get("budgeted_total") or {}).get("sweeps") or 0
            offset += used_
            bounds.append((offset, rid))
        if xs:
            ax.step(xs, ys, where="post", color="tab:red", label="random starts: best feasible objective so far")
        for x, rid in bounds:
            ax.axvline(x, color="0.8", lw=0.6)
            ax.text(x, 1.02, rid, rotation=90, fontsize=6, transform=ax.get_xaxis_transform(), ha="right", va="bottom")
        x_init = ri["forward"] + ri["adjoint"]
        y0 = ref["initialized_objective"] / J0 if J0 else ref["initialized_objective"]
        y1 = ref["final_objective"] / J0 if J0 else ref["final_objective"]
        ax.plot([x_init, ours["total_sweeps"]], [y0, y1], "o--", color="tab:green",
                label="ours: after initialization -> after L-BFGS-B (endpoints only)")
        ax.axvline(bud["total_budget"]["sweeps"], color="k", ls=":", lw=1, label="total budget")
        ax.set_yscale("log")
        ax.set_xlabel("cumulative PDE sweeps (forward + adjoint), random starts concatenated in predetermined order r01..r10")
        ax.set_ylabel("objective / J0" if J0 else "objective")
        ax.set_title("Best objective vs cumulative PDE sweeps (order fixed before launch: r01, r02, ..., r10)",
                     fontsize=10)
        ax.legend(fontsize=7, loc="lower left")
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(rep / f"best_objective_vs_sweeps.{ext}", dpi=160)
        plt.close(fig)
        figs.append("best_objective_vs_sweeps.png")
    except Exception as exc:  # noqa: BLE001
        import traceback
        (rep / "figure_error.txt").write_text("".join(traceback.format_exception(exc)))

    # ---------------- REPORT.md ----------------
    L = []
    L.append("# Five-source random-start benchmark at matched PDE-sweep budget")
    L.append("")
    L.append(f"Generated {time.strftime('%F %T')}. {len(finished)}/{len(order)} starts finished"
             + (" (PARTIAL report)." if a.partial or len(finished) < len(order) else "."))
    L.append("")
    L.append("## Setup")
    L.append(f"- Reference (ours): `{ref['run_dir']}`. {ref['selection']}")
    st = ref.get("stalled_run_for_contrast") or {}
    L.append(f"- Not used: the stalled unscaled run `{st.get('path')}` ({st.get('message')}, "
             f"J = {fmt(st.get('final_objective'))}).")
    L.append(f"- Budget unit: {bud['unit']}. Reference counts from `{bud['source']}`:")
    L.append(f"  initialization {ri['forward']} fwd + {ri['adjoint']} adj; local refinement {rb['forward']} fwd + "
             f"{rb['adjoint']} adj (= 1 fwd objective at the initialized control + GN-diag 16 fwd/15 adj + "
             f"L-BFGS-B 67 fwd/67 adj). **Total {bud['total_budget']['forward']} fwd + "
             f"{bud['total_budget']['adjoint']} adj = {bud['total_budget']['sweeps']} sweeps.**")
    L.append(f"- {bud['paper_note']}")
    L.append(f"- Allocation: {bud['allocation_rule']}")
    L.append(f"- Enforcement: {bud['enforcement']}")
    L.append(f"- Excluded for both methods: data generation (1 fwd); driver reporting overhead "
             f"(reference 4 fwd/2 adj; per start in the table).")
    L.append(f"- Starts: {starts['sampler']['function']}; centers {starts['sampler']['centers']}; amplitudes "
             f"{starts['sampler']['amplitudes']}; {starts['sampler']['seed_rule']}. Truth never used.")
    L.append("- Optimizer per start: same driver path as the reference (GN-diag computed at each start and charged "
             "to that start, L-BFGS-B maxcor 10, ftol 1e-14, gtol 1e-9, maxls 20, raw feasibility, amplitude box "
             "[0.1, 2]); no sensitivity initialization, beam search or center refinement.")
    L.append(f"- J0 = objective of the zero source against the same synthetic data = {fmt(J0)}.")
    L.append("")
    L.append("## Comparison")
    if best:
        L.append(md_table(
            ["method", "fwd", "adj", "sweeps", "iterations", "objective", "J/J0", "max center err (m)",
             "max amp err", "feasible", "stop"],
            [["Ours (init + refinement)", ours["total_fwd"], ours["total_adj"], ours["total_sweeps"],
              ours["iterations"], ours["objective"], ours["objective_over_J0"], ours["max_center_err_m"],
              ours["max_amp_err"], ours["feasible"], "converged"],
             [f"Best random start {best['selected_run']} -- cost of ALL {best['n_finished']} starts",
              best["aggregate_fwd_all_starts"], best["aggregate_adj_all_starts"],
              best["aggregate_sweeps_all_starts"], f"{best['iterations']} (all: {best['aggregate_iterations']})",
              best["objective"], best["objective_over_J0"], best["max_center_err_m"], best["max_amp_err"],
              best["feasible"], best["stop"]]]))
        L.append("")
        L.append(f"Ours split: initialization {ours['init_fwd']} fwd / {ours['init_adj']} adj; local refinement "
                 f"{ours['local_fwd']} fwd / {ours['local_adj']} adj. Random starts: GN-diag setup "
                 f"{best['aggregate_gn_setup_fwd']} fwd / {best['aggregate_gn_setup_adj']} adj in total; unused "
                 f"allowance {best['aggregate_unused_sweeps']} sweeps (not reallocated). Best start's own cost "
                 f"{best['own_fwd']} fwd / {best['own_adj']} adj.")
        L.append("")
        L.append("The best random start is selected by the lowest objective only; the center/amplitude errors are "
                 "computed afterwards with the permutation that minimizes total center error "
                 "(`run_optimization.match_sources_by_permutation`).")
    else:
        L.append("No finished random start yet.")
    L.append("")
    L.append("## Per start")
    L.append(md_table(
        ["run", "seed", "allow", "used fwd/adj", "unused", "GN fwd/adj", "iters", "evals", "stop", "J start",
         "best J", "J/J0", "feasible", "max ctr err (m)", "max amp err", "overhead fwd/adj"],
        [[p["run"], p["seed"], p["allowance_sweeps"], f"{fmt(p.get('fwd_used'))}/{fmt(p.get('adj_used'))}",
          p.get("unused_sweeps"), f"{fmt(p.get('gn_fwd'))}/{fmt(p.get('gn_adj'))}", p.get("iterations"),
          p.get("evaluations"), p.get("stop_reason") or p["status"], p.get("initial_objective"),
          p.get("best_feasible_objective"), p.get("best_over_J0"), p.get("best_feasible"),
          p.get("max_center_err_m"), p.get("max_amp_err"),
          f"{fmt(p.get('outside_budget_overhead_fwd'))}/{fmt(p.get('outside_budget_overhead_adj'))}"]
         for p in per]))
    L.append("")
    L.append("`stop`: budget_exhausted = the PDE-sweep guard ended the run (not convergence); optimizer_converged = "
             "SciPy L-BFGS-B success. `best J` = best feasible fully evaluated point (objective + gradient), which "
             "is what the driver reports as final control on a budget stop. The `overhead` columns are driver "
             "re-evaluation/export solves outside the budget (excluded for both methods).")
    L.append("")
    L.append("## Figures")
    for f in figs:
        L.append(f"- `{f}` (+ .pdf)")
    L.append("- Cumulative plot order: r01, r02, ..., r10 (run index, fixed before launch). The reference curve shows "
             "only its two recorded endpoints (after initialization, after L-BFGS-B); per-sweep intermediate values "
             "were not recorded for the saved run.")
    L.append("")
    L.append("## Scope")
    L.append("- One five-source configuration, noiseless synthetic (inverse-crime) data, one sensor layout, ten "
             "random starts. This is a cost-matched comparison against a single saved run, not a statistical study.")
    (rep / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"report written: {len(finished)} finished, best {best['selected_run'] if best else None}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
