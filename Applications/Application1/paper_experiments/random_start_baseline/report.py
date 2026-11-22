"""Aggregate the random-start baseline: CSV/JSON, REPORT.md and compact plots.

    PY report.py --exp-dir EXP

Safe to run on a partial sweep (unfinished runs are listed as pending).
Recovery (truth-based, post hoc) and optimizer convergence are reported
separately. Nothing here changes any run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

CASE_ORDER = ["A_L1", "A_L2", "B_L1", "B_L2", "M5_S4"]
C_REC, C_NOT, C_REF = "#2a78d6", "#eb6834", "#222222"


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def conv_class(msg, exit_status):
    if exit_status not in (0, None):
        return "failed"
    m = str(msg or "").upper()
    if "ABNORMAL" in m:
        return "abnormal_line_search"
    if "CONVERGENCE" in m:
        return "converged"
    if "LIMIT" in m or "ITERATION" in m or "EXCEEDS" in m:
        return "iteration_or_eval_limit"
    return "other"


def fa(b, k):
    return None if not b else b.get(k)


def stats(vals):
    v = [x for x in vals if x is not None and isinstance(x, (int, float)) and math.isfinite(x)]
    if not v:
        return {"n": 0}
    a = np.asarray(v, dtype=float)
    return {"n": len(v), "min": float(a.min()), "q25": float(np.percentile(a, 25)),
            "median": float(np.median(a)), "q75": float(np.percentile(a, 75)), "max": float(a.max())}


def g(x, fmt=".3g"):
    if x is None:
        return "-"
    if isinstance(x, float):
        return format(x, fmt)
    return str(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    exp = Path(ap.parse_args().exp_dir).resolve()
    out = exp / "report"
    out.mkdir(exist_ok=True)
    crit = load(exp / "config/recovery_criterion.json")
    ref = load(exp / "config/reference.json")
    starts = load(exp / "config/starts.json")["starts"]
    PT, AT = crit["max_position_error_m"], crit["max_amplitude_error"]

    rows = []
    for s in starts:
        cdir = exp / "cases" / s["run_id"]
        r = load(cdir / "run_result.json")
        row = {"run_id": s["run_id"], "case": s["case"], "run_index": s["run_index"],
               "seed": s["seed"], "status": "pending", "initial_control": s["initial_control"]}
        if r is not None:
            ps = r.get("pde_solves", {})
            feas = (r.get("final_feasibility_check") or {}).get("all_centers_feasible")
            pe, ae = r.get("max_position_error_m"), r.get("max_amplitude_error")
            recovered = bool(feas and pe is not None and ae is not None and pe <= PT and ae <= AT)
            row.update({
                "status": "done", "exit_status": r.get("exit_status"),
                "error": (r.get("error") or "")[-300:] or None,
                "convergence": conv_class(r.get("message"), r.get("exit_status")),
                "message": r.get("message"), "success_flag": r.get("success_flag"),
                "iterations": r.get("iterations"),
                "n_objective_evaluations": r.get("n_objective_evaluations"),
                "n_gradient_evaluations": r.get("n_gradient_evaluations"),
                "J0": r.get("J0_zero_source"),
                "J_initial": r.get("initial_objective"), "J_final": r.get("final_objective"),
                "J_final_over_J0": r.get("final_over_J0"),
                "final_gradient_norm": r.get("final_gradient_norm"),
                "max_position_error_m": pe, "max_amplitude_error": ae,
                "initial_max_position_error_m": r.get("initial_max_position_error_m"),
                "final_centers_feasible": feas,
                "recovered": recovered,
                "raw_infeasible_evaluations": (r.get("numerical_flags") or {}).get("raw_infeasible_evaluations"),
                "finite_final_values": (r.get("numerical_flags") or {}).get("finite_final_values"),
                "gn_cond_unscaled": r.get("gn_condition_unscaled"),
                "gn_cond_scaled": r.get("gn_condition_scaled"),
                "lbfgsb_maxcor": r.get("lbfgsb_maxcor"),
                "fwd_setup": fa(ps.get("scaling_setup"), "forward_solves"),
                "adj_setup": fa(ps.get("scaling_setup"), "adjoint_solves"),
                "fwd_opt": fa(ps.get("optimization"), "forward_solves"),
                "adj_opt": fa(ps.get("optimization"), "adjoint_solves"),
                "fwd_data": fa(ps.get("data_generation"), "forward_solves"),
                "fwd_output_overhead": fa(ps.get("driver_output_overhead"), "forward_solves"),
                "adj_output_overhead": fa(ps.get("driver_output_overhead"), "adjoint_solves"),
                "final_control": r.get("final_control"),
                "start_matches_saved": (r.get("initial_control") == s["initial_control"])
                if r.get("initial_control") is not None else None,
            })
            fs = [row.get(k) for k in ("fwd_setup", "fwd_opt")]
            as_ = [row.get(k) for k in ("adj_setup", "adj_opt")]
            row["fwd_setup_plus_opt"] = sum(fs) if all(v is not None for v in fs) else None
            row["adj_setup_plus_opt"] = sum(as_) if all(v is not None for v in as_) else None
        rows.append(row)

    # ---------------- reference rows ----------------
    refrows = {}
    for case in CASE_ORDER:
        rf = ref[case]
        J0s = [r["J0"] for r in rows if r["case"] == case and r.get("J0")]
        J0 = float(np.median(J0s)) if J0s else None
        init = rf["phases_initializer_run"]["initialization"]
        post = rf["phases_post_init_run"]["local_optimization"]
        rm = rf["recovery_permutation_matched"]
        feas_ref = rf.get("feasibility") or {}
        refrows[case] = {
            "case": case, "J0": J0, "J_initial": rf["initial_objective"], "J_final": rf["final_objective"],
            "J_final_over_J0": rf["final_objective"] / J0 if J0 else None,
            "iterations": rf["iterations"], "message": rf["message"],
            "convergence": conv_class(rf["message"], 0),
            "max_position_error_m": rm["max_position_error_m"], "max_amplitude_error": rm["max_amplitude_error"],
            "recovered_by_same_criterion": rm["max_position_error_m"] <= PT and rm["max_amplitude_error"] <= AT,
            "raw_infeasible_evaluations": feas_ref.get("raw_infeasible_evaluations"),
            "fwd_init": init["forward_solves"], "adj_init": init["adjoint_solves"],
            "fwd_post_init_gn_plus_opt": post["forward_solves"], "adj_post_init_gn_plus_opt": post["adjoint_solves"],
            "fwd_total_init_setup_opt": init["forward_solves"] + post["forward_solves"],
            "adj_total_init_setup_opt": init["adjoint_solves"] + post["adjoint_solves"],
            "fwd_data": (rf["phases_initializer_run"].get("pre_init_setup_and_synthetic_data") or {}).get("forward_solves"),
            "fwd_output_overhead_post_run": (rf["phases_post_init_run"].get("driver_overhead_smoke_export_final_eval") or {}).get("forward_solves"),
            "adj_output_overhead_post_run": (rf["phases_post_init_run"].get("driver_overhead_smoke_export_final_eval") or {}).get("adjoint_solves"),
            "gn_cond": [(st.get("scaling") or {}).get("condition_number_unscaled") for st in rf["post_init_summary"]["stages"]]
            + [(st.get("scaling") or {}).get("condition_number_scaled") for st in rf["post_init_summary"]["stages"]],
            "max_iter": rf["max_iter"], "lbfgsb_maxcor": rf["lbfgsb_maxcor"],
            "post_init_run": rf["post_init_run_dir"], "initializer_run": rf["initializer_run_dir"],
        }

    # ---------------- per-case summary + best-of-k ----------------
    case_sum, bestk = {}, []
    for case in CASE_ORDER:
        cr = sorted([r for r in rows if r["case"] == case], key=lambda r: r["run_index"])
        done = [r for r in cr if r["status"] == "done"]
        case_sum[case] = {
            "finished": len(done), "planned": len(cr),
            "recovered": sum(1 for r in done if r.get("recovered")),
            "converged": sum(1 for r in done if r.get("convergence") == "converged"),
            "iteration_limit": sum(1 for r in done if r.get("convergence") == "iteration_or_eval_limit"),
            "abnormal": sum(1 for r in done if r.get("convergence") == "abnormal_line_search"),
            "failed": sum(1 for r in done if r.get("convergence") == "failed"),
            "converged_but_not_recovered": sum(1 for r in done if r.get("convergence") == "converged" and not r.get("recovered")),
            "final_infeasible": sum(1 for r in done if r.get("final_centers_feasible") is False),
            **{f"dist_{k}": stats([r.get(k) for r in done]) for k in (
                "max_position_error_m", "max_amplitude_error", "J_final_over_J0", "iterations",
                "fwd_setup_plus_opt", "adj_setup_plus_opt")},
            "sum_fwd_setup_plus_opt": sum(r.get("fwd_setup_plus_opt") or 0 for r in done),
            "sum_adj_setup_plus_opt": sum(r.get("adj_setup_plus_opt") or 0 for r in done),
        }
        best, cf, ca, anyrec, contiguous = None, 0, 0, False, True
        for r in cr:
            if r["status"] != "done":
                contiguous = False
            if not contiguous:
                break
            cf += r.get("fwd_setup_plus_opt") or 0
            ca += r.get("adj_setup_plus_opt") or 0
            if r.get("J_final") is not None and math.isfinite(r["J_final"]) and (best is None or r["J_final"] < best["J_final"]):
                best = r
            anyrec = anyrec or bool(r.get("recovered"))
            bestk.append({"case": case, "k": r["run_index"],
                          "best_J_final": best["J_final"] if best else None,
                          "best_J_final_over_J0": best.get("J_final_over_J0") if best else None,
                          "best_run_id": best["run_id"] if best else None,
                          "best_run_recovered": best.get("recovered") if best else None,
                          "best_run_max_position_error_m": best.get("max_position_error_m") if best else None,
                          "best_run_max_amplitude_error": best.get("max_amplitude_error") if best else None,
                          "any_of_first_k_recovered": anyrec,
                          "cum_fwd_setup_plus_opt": cf, "cum_adj_setup_plus_opt": ca})

    # ---------------- write CSV / JSON ----------------
    scalar_keys = ["run_id", "case", "run_index", "seed", "status", "exit_status", "convergence", "message",
                   "success_flag", "iterations", "n_objective_evaluations", "n_gradient_evaluations",
                   "J0", "J_initial", "J_final", "J_final_over_J0", "final_gradient_norm",
                   "max_position_error_m", "max_amplitude_error", "initial_max_position_error_m",
                   "final_centers_feasible", "recovered", "raw_infeasible_evaluations", "finite_final_values",
                   "gn_cond_unscaled", "gn_cond_scaled", "lbfgsb_maxcor",
                   "fwd_setup", "adj_setup", "fwd_opt", "adj_opt", "fwd_setup_plus_opt", "adj_setup_plus_opt",
                   "fwd_data", "fwd_output_overhead", "adj_output_overhead", "start_matches_saved",
                   "initial_control", "final_control", "error"]
    with (out / "runs.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=scalar_keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(r.get(k)) if isinstance(r.get(k), list) else r.get(k)) for k in scalar_keys})
    with (out / "best_of_k.csv").open("w", newline="") as fh:
        if bestk:
            w = csv.DictWriter(fh, fieldnames=list(bestk[0]))
            w.writeheader()
            w.writerows(bestk)
    with (out / "reference.csv").open("w", newline="") as fh:
        keys = list(next(iter(refrows.values())))
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for v in refrows.values():
            w.writerow({k: (json.dumps(v[k]) if isinstance(v[k], list) else v[k]) for k in keys})
    (out / "summary.json").write_text(json.dumps({
        "generated": time.strftime("%F %T"), "recovery_criterion": crit,
        "case_summary": case_sum, "reference": refrows, "best_of_k": bestk, "runs": rows}, indent=2, default=str) + "\n")

    # ---------------- plots ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                             "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6})

        def strip(ax, key, ylabel, logy=True, refkey=None, thresh=None):
            for i, case in enumerate(CASE_ORDER):
                d = [r for r in rows if r["case"] == case and r["status"] == "done" and r.get(key) is not None]
                jit = np.linspace(-0.18, 0.18, max(len(d), 1))
                for j, r in enumerate(d):
                    rec = r.get("recovered")
                    ax.plot(i + jit[j], max(r[key], 1e-16) if logy else r[key],
                            marker="o" if rec else "X", ms=6, ls="none",
                            color=C_REC if rec else C_NOT, mec="white", mew=0.8)
                rv = refrows[case].get(refkey or key)
                if rv is not None:
                    ax.plot(i + 0.33, max(rv, 1e-16) if logy else rv, marker="*", ms=11, color=C_REF, ls="none")
            if thresh is not None:
                ax.axhline(thresh, color="#888888", lw=1, ls="--")
            ax.set_xticks(range(len(CASE_ORDER)), CASE_ORDER)
            if logy:
                ax.set_yscale("log")
            ax.set_ylabel(ylabel)

        from matplotlib.lines import Line2D
        legend = [Line2D([], [], marker="o", ls="none", color=C_REC, label="random start, recovered"),
                  Line2D([], [], marker="X", ls="none", color=C_NOT, label="random start, not recovered"),
                  Line2D([], [], marker="*", ls="none", ms=10, color=C_REF, label="sensitivity-initialized (paper)")]

        fig, axs = plt.subplots(2, 3, figsize=(12, 6.5), constrained_layout=True)
        strip(axs[0, 0], "max_position_error_m", "max matched center error (m)", thresh=PT)
        strip(axs[0, 1], "max_amplitude_error", "max matched amplitude error", thresh=AT)
        strip(axs[0, 2], "J_final_over_J0", "final J / J(0)")
        strip(axs[1, 0], "iterations", "L-BFGS-B iterations", logy=False)
        strip(axs[1, 1], "fwd_setup_plus_opt", "forward solves (setup + opt)", refkey="fwd_total_init_setup_opt")
        strip(axs[1, 2], "adj_setup_plus_opt", "adjoint solves (setup + opt)", refkey="adj_total_init_setup_opt")
        axs[1, 1].set_title("reference star = init + setup + opt", fontsize=8)
        axs[1, 2].set_title("reference star = init + setup + opt", fontsize=8)
        fig.legend(handles=legend, loc="outside upper center", ncol=3, frameon=False)
        fig.savefig(out / "overview_distributions.png", dpi=150)
        fig.savefig(out / "overview_distributions.pdf")
        plt.close(fig)

        fig, axs = plt.subplots(1, 5, figsize=(15, 3.4), constrained_layout=True)
        for ax, case in zip(axs, CASE_ORDER):
            b = [x for x in bestk if x["case"] == case and x["best_J_final_over_J0"] is not None]
            if b:
                cx = [x["cum_fwd_setup_plus_opt"] + x["cum_adj_setup_plus_opt"] for x in b]
                cy = [max(x["best_J_final_over_J0"], 1e-22) for x in b]
                ax.step(cx, cy, where="post", color=C_REC, lw=2)
                for x, xx, yy in zip(b, cx, cy):
                    ax.plot(xx, yy, marker="o" if x["best_run_recovered"] else "X", ms=6, ls="none",
                            color=C_REC if x["best_run_recovered"] else C_NOT, mec="white", mew=0.8)
                    ax.annotate(str(x["k"]), (xx, yy), textcoords="offset points", xytext=(0, 6),
                                fontsize=7, ha="center", color="#555555")
            rr = refrows[case]
            if rr["J_final_over_J0"] is not None:
                ax.plot(rr["fwd_total_init_setup_opt"] + rr["adj_total_init_setup_opt"],
                        max(rr["J_final_over_J0"], 1e-22), marker="*", ms=12, color=C_REF, ls="none")
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.set_title(case)
            ax.set_xlabel("cumulative fwd + adj solves")
        axs[0].set_ylabel("best final J / J(0) after k starts")
        fig.legend(handles=legend, loc="outside upper center", ncol=3, frameon=False)
        fig.savefig(out / "best_of_k.png", dpi=150)
        fig.savefig(out / "best_of_k.pdf")
        plt.close(fig)
        plot_err = None
    except Exception as exc:  # noqa: BLE001
        plot_err = repr(exc)

    # ---------------- REPORT.md ----------------
    n_done = sum(1 for r in rows if r["status"] == "done")
    L = [f"# Random-start baseline (App1) — report", "",
         f"Generated {time.strftime('%F %T')} — {n_done}/{len(rows)} runs finished."
         + ("" if n_done == len(rows) else " **PARTIAL: unfinished runs are pending.**"), "",
         "Setup: `manifest.json`; starts and sampler: `config/starts.json`; per-run data: `runs.csv`, "
         "`summary.json`, `cases/<run>/`.", "",
         f"**Recovery criterion (fixed before launch):** {crit['definition']}. Matching: {crit['matching']}.",
         "Recovery is truth-based and post hoc. It is reported separately from optimizer convergence "
         "(the L-BFGS-B termination message).", "",
         "## Recovery and convergence per case", "",
         "| case | finished | recovered | converged | conv. but not recovered | iter/eval cap | abnormal | failed | final infeasible | reference recovered |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for c in CASE_ORDER:
        s = case_sum[c]
        L.append(f"| {c} | {s['finished']}/{s['planned']} | **{s['recovered']}/{s['finished']}** | {s['converged']} | "
                 f"{s['converged_but_not_recovered']} | {s['iteration_limit']} | {s['abnormal']} | {s['failed']} | "
                 f"{s['final_infeasible']} | {refrows[c]['recovered_by_same_criterion']} |")
    L += ["", "## Distributions over finished random starts (min / median / max)", "",
          "| case | max center err (m) | max amp err | J/J0 | iterations | fwd setup+opt | adj setup+opt |",
          "|---|---|---|---|---|---|---|"]
    for c in CASE_ORDER:
        s = case_sum[c]

        def mm(k):
            d = s[f"dist_{k}"]
            return "-" if not d.get("n") else f"{g(d['min'])} / {g(d['median'])} / {g(d['max'])}"
        L.append(f"| {c} | {mm('max_position_error_m')} | {mm('max_amplitude_error')} | {mm('J_final_over_J0')} | "
                 f"{mm('iterations')} | {mm('fwd_setup_plus_opt')} | {mm('adj_setup_plus_opt')} |")
    L += ["", "## Sensitivity-initialized reference (paper run) and its full cost", "",
          "Total = initialization + post-init block (GN-diag construction + L-BFGS-B). The GN share is not "
          "instrumented in the reference runs, so it stays inside the post-init column.",
          "Data generation (1 forward) and driver output overhead are excluded from the totals, for both methods.", "",
          "| case | iters (cap) | message | max center err (m) | max amp err | J/J0 | init fwd/adj | post-init fwd/adj | **total fwd/adj** | 10 random starts: setup+opt fwd/adj |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for c in CASE_ORDER:
        r, s = refrows[c], case_sum[c]
        L.append(f"| {c} | {r['iterations']} ({r['max_iter']}) | {r['message'][:40]} | {g(r['max_position_error_m'])} | "
                 f"{g(r['max_amplitude_error'])} | {g(r['J_final_over_J0'])} | {r['fwd_init']}/{r['adj_init']} | "
                 f"{r['fwd_post_init_gn_plus_opt']}/{r['adj_post_init_gn_plus_opt']} | "
                 f"**{r['fwd_total_init_setup_opt']}/{r['adj_total_init_setup_opt']}** | "
                 f"{s['sum_fwd_setup_plus_opt']}/{s['sum_adj_setup_plus_opt']} ({s['finished']} runs) |")
    L += ["", "## Best final objective after k starts (predetermined order r01..r10)", "",
          "The best start is chosen by objective value only. Its recovery status is evaluated afterwards. Costs are cumulative setup+opt solves.", ""]
    for c in CASE_ORDER:
        b = [x for x in bestk if x["case"] == c]
        if not b:
            continue
        L += [f"**{c}**", "", "| k | best J/J0 | best run | recovered | any of first k recovered | cum fwd | cum adj |",
              "|---|---|---|---|---|---|---|"]
        for x in b:
            L.append(f"| {x['k']} | {g(x['best_J_final_over_J0'])} | {x['best_run_id']} | {x['best_run_recovered']} | "
                     f"{x['any_of_first_k_recovered']} | {x['cum_fwd_setup_plus_opt']} | {x['cum_adj_setup_plus_opt']} |")
        L.append("")
    L += ["## Per-run table", "",
          "| run | seed | termination | iters | nfev/ngev | J0 → J_init → J_final | J/J0 | max center err (m) | max amp err | feasible | recovered | setup fwd/adj | opt fwd/adj |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["status"] != "done":
            L.append(f"| {r['run_id']} | {r['seed']} | pending | | | | | | | | | | |")
            continue
        L.append(f"| {r['run_id']} | {r['seed']} | {r['convergence']}: {str(r.get('message'))[:38]} | {r.get('iterations')} | "
                 f"{r.get('n_objective_evaluations')}/{r.get('n_gradient_evaluations')} | "
                 f"{g(r.get('J0'))} → {g(r.get('J_initial'))} → {g(r.get('J_final'))} | {g(r.get('J_final_over_J0'))} | "
                 f"{g(r.get('max_position_error_m'))} | {g(r.get('max_amplitude_error'))} | {r.get('final_centers_feasible')} | "
                 f"{r.get('recovered')} | {r.get('fwd_setup')}/{r.get('adj_setup')} | {r.get('fwd_opt')}/{r.get('adj_opt')} |")
    fails = [r for r in rows if r["status"] == "done" and (r.get("exit_status") or r.get("error"))]
    L += ["", "## Failures / exceptional terminations", ""]
    L += [f"- {r['run_id']}: exit {r.get('exit_status')} — {r.get('error')}" for r in fails] or ["- none"]
    L += ["", "## Interpretation caveats", "",
          "- A run that stops at the iteration cap is not proof that it could never recover with more iterations. "
          "Cap-limited runs are counted separately.",
          "- Do not infer a cost advantage from local-optimizer counts alone. The reference total includes the initializer's "
          "forward solves, which dominate its cost.",
          "- Noiseless inverse-crime data, one truth per case, ten starts per case.",
          "- The 2-source references are a chain: the sensitivity initializer in `sensor_layout_comparison_20261003_151233`, "
          "then a gn-diag restart from its saved control in `gn_diag_default_check_20261004_161718`. The identical start is verified. "
          "The unscaled polish of the initializer run is not part of the paper pipeline and is excluded.",
          "- Wall times are not compared (different thread counts and host load). Solve counts are the cost metric.",
          "", "Figures: `overview_distributions.png/pdf`, `best_of_k.png/pdf`."
          + (f" Plot error: {plot_err}" if plot_err else "")]
    (out / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"report written: {out} ({n_done}/{len(rows)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
