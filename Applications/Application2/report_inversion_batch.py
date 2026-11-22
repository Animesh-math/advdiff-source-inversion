"""Post-hoc report for an Application2 inversion batch (reads the true controls ONLY here, for scoring).

    PY report_inversion_batch.py --batch-dir <batch>            # full report (re-solves 2 coarse forwards per run)
    PY report_inversion_batch.py --batch-dir <batch> --no-traces

Inputs: <batch>/resolved_settings.json, <batch>/status/<run>.exit, <batch>/runs/<run>/{inversion_summary.json,
optimization_history.csv}.  The true controls (<dataset>/case_XX/truth/truth_record.json) are opened after the
optimizations have finished and are used only to compute recovery errors; they never feed back into a run.

Writes <batch>/REPORT.md, <batch>/report/{summary.json, summary.csv, figures}.

Classification (fixed before the runs, stated in the report):
  optimizer outcome   FAILED            non-zero exit status or no inversion_summary.json
                      ITERATION_LIMIT   L-BFGS-B stopped at --max-iter (or its evaluation limit)
                      CONVERGED         L-BFGS-B success flag (projected-gradient or relative-reduction test)
                      STOPPED_OTHER     any other termination (e.g. abnormal line search); message reported
  recovery            RECOVERED         outcome CONVERGED or STOPPED_OTHER, both final centres in the fluid mesh,
                                        and both permutation-matched centre errors <= one Gaussian e-folding radius
                                        sqrt(rho) = 0.20 m
                      NOT_RECOVERED     otherwise (amplitude errors are reported, not thresholded)
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import report_source_id as rsi  # noqa: E402  (draw_room)
import run_inversion as ri  # noqa: E402

CLR = {"clean": "#1f6fb4", "noisy": "#d1495b", "truth": "#222222"}


def read_exit(path: Path) -> dict:
    if not path.exists():
        return {"rc": None, "state": "NOT_FINISHED"}
    d = dict(kv.split("=", 1) for kv in path.read_text().split())
    d["rc"] = int(d["rc"])
    return d


def read_history(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None
    return np.array([[float(r[k]) for k in rows[0].keys()] for r in rows])


def match_sources(z, z_true):
    """Permutation of the estimated sources minimizing the summed squared centre error."""
    e, t = np.asarray(z).reshape(-1, 3), np.asarray(z_true).reshape(-1, 3)
    best = None
    for perm in itertools.permutations(range(len(t))):
        cost = sum(float(np.sum((e[p, :2] - t[i, :2]) ** 2)) for i, p in enumerate(perm))
        if best is None or cost < best[0]:
            best = (cost, perm)
    perm = best[1]
    out = []
    for i, p in enumerate(perm):
        out.append({"true_source": i + 1, "estimated_source": p + 1,
                    "true_center": t[i, :2].tolist(), "estimated_center": e[p, :2].tolist(),
                    "center_error_m": float(np.linalg.norm(e[p, :2] - t[i, :2])),
                    "true_amplitude": float(t[i, 2]), "estimated_amplitude": float(e[p, 2]),
                    "amplitude_error": float(e[p, 2] - t[i, 2]),
                    "amplitude_rel_error": float((e[p, 2] - t[i, 2]) / t[i, 2])})
    return {"permutation": [p + 1 for p in perm], "sources": out}


def classify(exit_rec, summary, max_iter):
    if exit_rec.get("rc") is None:
        return "NOT_FINISHED"
    if exit_rec["rc"] != 0 or summary is None:
        return "FAILED"
    res = summary["result"]
    msg = res["message"].upper()
    if res["iterations"] >= max_iter or "LIMIT" in msg or "TOTAL NO. OF" in msg:
        return "ITERATION_LIMIT"
    return "CONVERGED" if res["success"] else "STOPPED_OTHER"


def fmt(z, p=4):
    return "(" + ", ".join(f"{v:.{p}f}" for v in np.asarray(z, dtype=float)) + ")"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch-dir", type=Path, required=True)
    ap.add_argument("--no-traces", action="store_true", help="skip the post-hoc forward solves for trace fits")
    a = ap.parse_args(argv)
    B = a.batch_dir.resolve()
    settings = json.loads((B / "resolved_settings.json").read_text())
    dataset_dir = Path(settings["dataset_dir"])
    rep = B / "report"
    rep.mkdir(parents=True, exist_ok=True)
    radius = None
    rows, traces = [], {}

    for rs in settings["runs"]:
        run_id, case, kind = rs["run_id"], rs["case"], rs["data"]
        rdir = B / "runs" / run_id
        max_iter = int(rs["resolved_arguments"]["max_iter"])
        exit_rec = read_exit(B / "status" / f"{run_id}.exit")
        sp = rdir / "inversion_summary.json"
        summary = json.loads(sp.read_text()) if sp.exists() else None
        init_rec = json.loads((rdir / "initialization.json").read_text()) if (rdir / "initialization.json").exists() else None
        hist = read_history(rdir / "optimization_history.csv")
        truth = json.loads((dataset_dir / f"case_{case}" / "truth" / "truth_record.json").read_text())   # post hoc
        z_true = np.asarray(truth["true_control"], dtype=float)
        radius = float(np.sqrt(rs["transport"]["sigma2"]))
        outcome = classify(exit_rec, summary, max_iter)
        row = {"run_id": run_id, "case": case, "data": kind, "exit": exit_rec, "outcome": outcome,
               "true_control": z_true.tolist(), "max_iter": max_iter}
        if init_rec is not None:
            row["initial_control"] = init_rec["initial_control"]
            row["initialization_cost"] = init_rec["cost"]
            row["initializer_reported"] = {k: init_rec["initializer_reported"].get(k) for k in (
                "zero_source_objective", "initial_exact_objective", "n_forward_solves", "n_adjoint_solves",
                "n_candidate_traces")}
            row["initial_feasibility"] = init_rec["feasibility"]
            row["initial_match"] = match_sources(init_rec["initial_control"], z_true)
        if summary is not None:
            res = summary["result"]
            zf = np.asarray(res["final_control"], dtype=float)
            feas = res["feasibility"]
            match = match_sources(zf, z_true)
            row.update({
                "final_control": zf.tolist(), "objective_zero_source": summary["objective_zero_source"],
                "objective_initial": res["initial_objective"], "objective_final": res["final_objective"],
                "final_gradient_norm": res["final_gradient_norm"], "termination_message": res["message"],
                "success_flag": res["success"], "iterations": res["iterations"], "nfev": res["nfev"],
                "njev": res["njev"], "solve_accounting": summary["solve_accounting"],
                "final_feasibility": feas, "final_match": match,
                "scaling": {k: res["scaling"].get(k) for k in ("scaling", "condition_number_unscaled",
                                                               "condition_number_scaled", "diag") if k in res["scaling"]},
            })
            feasible = all(f["in_fluid_mesh"] for f in feas)
            ok_centers = max(s["center_error_m"] for s in match["sources"]) <= radius
            row["final_feasible"] = feasible
            row["recovery"] = ("RECOVERED" if outcome in ("CONVERGED", "STOPPED_OTHER") and feasible and ok_centers
                               else "NOT_RECOVERED")
        else:
            row["recovery"] = "NOT_RECOVERED" if outcome != "NOT_FINISHED" else "NOT_FINISHED"
        # noise-only objective (data property, not truth): reference level for the noisy fits
        if kind == "noisy":
            eps = np.load(dataset_dir / f"case_{case}" / "data" / "noise.npz")["noise"]
            row["objective_noise_only"] = 0.5 * rs["eta1"] * rs["transport"]["dt"] * float(np.sum(eps[1:] ** 2))
        row["history"] = hist.tolist() if hist is not None else None
        rows.append(row)

        if not a.no_traces and summary is not None:
            ds = ri.load_dataset(dataset_dir, case, kind)
            problem = ri.build_problem(ds, log=lambda m: None)
            gt, nid = problem["app"].gt, problem["observation"].node_ids
            tr = {"data": ds["observations"], "times": ds["times"]}
            for lbl, z in (("final", zf), ("initial", np.asarray(row.get("initial_control", zf)))):
                tr[lbl] = gt.run_forward(problem["mesh_data"], problem["pde"], z)[:, nid]
            r = tr["final"][1:] - tr["data"][1:]
            row["trace_fit"] = {
                "relative_residual_all": float(np.linalg.norm(r) / np.linalg.norm(tr["data"][1:])),
                "relative_residual_per_sensor": (np.linalg.norm(r, axis=0)
                                                 / np.maximum(np.linalg.norm(tr["data"][1:], axis=0), 1e-300)).tolist(),
                "data_rms_per_sensor": np.sqrt(np.mean(tr["data"][1:] ** 2, axis=0)).tolist(),
                "note": "post-hoc coarse forward solves at the initial and final controls; not counted in the "
                        "optimization solve accounting"}
            traces[run_id] = tr
            plot_traces(rep / f"fig_traces_{run_id}.png", run_id, tr, rs["sensor_labels"])

    plot_paths(rep, rows, settings, dataset_dir)
    plot_objective(rep / "fig_objective_history.png", rows)
    write_outputs(B, rep, rows, settings, radius, traces_done=not a.no_traces)
    print(f"report: {B / 'REPORT.md'}")
    return 0


def plot_traces(path, run_id, tr, labels):
    t = tr["times"]
    s = slice(None, None, 10)
    fig, axes = plt.subplots(4, 2, figsize=(11, 10), sharex=True)
    for j, ax in enumerate(axes.ravel()):
        ax.plot(t[s], tr["data"][s, j], color="#9a9a9a", lw=0.6, label="observed (fine mesh)")
        ax.plot(t[s], tr["initial"][s, j], color="#e09f3e", lw=1.0, ls="--", label="initial control (coarse)")
        ax.plot(t[s], tr["final"][s, j], color="#1f6fb4", lw=1.2, label="final control (coarse)")
        ax.set_title(labels[j], fontsize=9, loc="left")
        ax.tick_params(labelsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("t [s]")
    axes[0, 0].legend(fontsize=7, loc="best")
    fig.suptitle(f"{run_id}: sensor traces (every 10th sample plotted)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_paths(rep, rows, settings, dataset_dir):
    sensors = np.asarray(settings["runs"][0]["sensor_coordinates"])
    labels = settings["runs"][0]["sensor_labels"]
    for case in sorted({r["case"] for r in rows}):
        fig, ax = plt.subplots(figsize=(11, 6.6))
        rsi.draw_room(ax, labels=True)
        ax.scatter(sensors[:, 0], sensors[:, 1], marker="s", s=28, color="#2a9d8f", zorder=6, label="sensors (wall_8)")
        for (x, y), lb in zip(sensors, labels):
            ax.annotate(lb, (x, y), xytext=(3, 3), textcoords="offset points", fontsize=7, color="#2a9d8f")
        for r in rows:
            if r["case"] != case:
                continue
            c = CLR[r["data"]]
            h = np.asarray(r["history"]) if r.get("history") else None
            if h is not None:
                for i in range(len(r["true_control"]) // 3):
                    ax.plot(h[:, 3 + 3 * i], h[:, 4 + 3 * i], "-", color=c, lw=1.0, alpha=0.8, zorder=7)
            if "initial_control" in r:
                zi = np.asarray(r["initial_control"]).reshape(-1, 3)
                ax.scatter(zi[:, 0], zi[:, 1], marker="o", facecolor="none", edgecolor=c, s=60, zorder=8,
                           label=f"{r['data']}: initial")
            if "final_control" in r:
                zf = np.asarray(r["final_control"]).reshape(-1, 3)
                ax.scatter(zf[:, 0], zf[:, 1], marker="x", color=c, s=70, zorder=9,
                           label=f"{r['data']}: final ({r['outcome']}, {r['recovery']})")
        zt = np.asarray(rows[[r["case"] for r in rows].index(case)]["true_control"]).reshape(-1, 3)
        ax.scatter(zt[:, 0], zt[:, 1], marker="*", color=CLR["truth"], s=140, zorder=10,
                   label="true centres (scoring only)")
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title(f"case {case}: initial, L-BFGS-B path and final centres (clean vs noisy)")
        ax.legend(fontsize=7, loc="lower right")
        fig.tight_layout()
        fig.savefig(rep / f"fig_sources_paths_case_{case}.png", dpi=130)
        plt.close(fig)


def plot_objective(path, rows):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for r in rows:
        if r.get("history"):
            h = np.asarray(r["history"])
            ax.semilogy(h[:, 0], h[:, 1], color=CLR[r["data"]], ls={"12": "-", "13": "--", "23": ":"}[r["case"]],
                        label=r["run_id"])
    ax.set_xlabel("L-BFGS-B iteration")
    ax.set_ylabel("objective J")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_outputs(B, rep, rows, settings, radius, traces_done):
    slim = [{k: v for k, v in r.items() if k != "history"} for r in rows]
    (rep / "summary.json").write_text(json.dumps({"runs": slim, "settings_file": str(B / "resolved_settings.json")},
                                                 indent=2, default=float))
    with open(rep / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "exit_rc", "outcome", "recovery", "iterations", "J_zero", "J_initial", "J_final",
                    "init_fwd", "init_adj", "scal_fwd", "scal_adj", "opt_fwd", "opt_adj",
                    "center_err_1_m", "center_err_2_m", "amp_rel_err_1", "amp_rel_err_2", "final_feasible",
                    "termination_message"])
        for r in rows:
            sa = r.get("solve_accounting", {})
            m = r.get("final_match", {}).get("sources", [{}, {}])
            w.writerow([r["run_id"], r["exit"].get("rc"), r["outcome"], r["recovery"], r.get("iterations"),
                        r.get("objective_zero_source"), r.get("objective_initial"), r.get("objective_final"),
                        sa.get("initialization", {}).get("forward_solves"), sa.get("initialization", {}).get("adjoint_solves"),
                        sa.get("scaling_gn_diagonal", {}).get("forward_solves"), sa.get("scaling_gn_diagonal", {}).get("adjoint_solves"),
                        sa.get("optimization_lbfgsb", {}).get("forward_solves"), sa.get("optimization_lbfgsb", {}).get("adjoint_solves"),
                        m[0].get("center_error_m"), m[1].get("center_error_m"),
                        m[0].get("amplitude_rel_error"), m[1].get("amplitude_rel_error"),
                        r.get("final_feasible"), r.get("termination_message")])

    L = []
    ds = settings["runs"][0]
    L += [f"# Application2 source identification: first full inversion batch",
          "",
          f"Dataset `{settings['expected_dataset_version']}` (`{settings['dataset_dir']}`), sensor layout "
          f"`{settings['expected_layout']}` with {settings['expected_num_sensors']} sensors "
          f"({', '.join(ds['sensor_labels'])}), flow mesh/velocity sha256 `{ds['mesh']['mesh_sha256'][:12]}` / "
          f"`{ds['velocity']['velocity_sha256'][:12]}`. Coarse-mesh inversion of fine-mesh observations; two unknown "
          f"centres and amplitudes, prescribed width rho = {ds['transport']['sigma2']} m^2; W = I; amplitude bounds "
          f"{ds['admissible_set']['amplitude_bounds']}. Sensitivity initializer (OSQP, M = 2 preset, room-scale tolerance "
          f"shim) then GN-diagonal-scaled L-BFGS-B (maxcor 30, max-iter 200). Settings per run: `resolved_settings.json`.",
          "",
          "The true controls were read only by this report, after the runs, to score recovery.",
          "",
          "## Classification rules (fixed before the runs)",
          "",
          "- Optimizer outcome: `FAILED` (non-zero exit or no summary), `ITERATION_LIMIT` (stopped at max-iter or an "
          "evaluation limit), `CONVERGED` (L-BFGS-B success flag), `STOPPED_OTHER` (other termination, e.g. abnormal line "
          "search; see message), `NOT_FINISHED`.",
          f"- Recovery: `RECOVERED` iff the outcome is CONVERGED or STOPPED_OTHER, both final centres lie in the fluid "
          f"mesh, and both permutation-matched centre errors are <= one e-folding radius sqrt(rho) = {radius:.2f} m. "
          f"Amplitude errors are reported, not thresholded. An ITERATION_LIMIT or FAILED run is never counted as recovered.",
          "",
          "## 1. Run status",
          "",
          "| run | exit | outcome | recovery | L-BFGS-B its | nfev / njev | termination message |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['run_id']} | {r['exit'].get('rc')} | **{r['outcome']}** | **{r['recovery']}** | "
                 f"{r.get('iterations', '-')} | {r.get('nfev', '-')} / {r.get('njev', '-')} | "
                 f"{r.get('termination_message', '-')} |")
    L += ["", "## 2. Controls `[x1, y1, A1, x2, y2, A2]` and objectives", "",
          "| run | initial control | final control | true control | J(0) | J(initial) | J(final) | J(final)/J(0) | noise-only J |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        jf, j0 = r.get("objective_final"), r.get("objective_zero_source")
        L.append(f"| {r['run_id']} | {fmt(r['initial_control']) if 'initial_control' in r else '-'} | "
                 f"{fmt(r['final_control']) if 'final_control' in r else '-'} | {fmt(r['true_control'], 3)} | "
                 f"{j0:.4e} | {r['objective_initial']:.4e} | {jf:.4e} | {jf / j0:.3e} | "
                 f"{(format(r['objective_noise_only'], '.4e') if 'objective_noise_only' in r else 'n/a')} |"
                 if jf is not None else
                 f"| {r['run_id']} | - | - | {fmt(r['true_control'], 3)} | - | - | - | - | - |")
    L += ["", "Noise-only J = (eta1 dt/2) sum ||eps^n||^2 of the saved realization (clean runs: not applicable).", ""]
    L += ["## 3. Iterations and PDE solve counts", "",
          "Counts are actual forward / adjoint sweeps at the shared solver entry points (cache hits cost 0). "
          "Optimization includes the objective/gradient at the initial control and the final re-evaluation.", "",
          "| run | init fwd | init adj | GN-diag scaling fwd | scaling adj | L-BFGS-B fwd | L-BFGS-B adj | total fwd | total adj | L-BFGS-B its |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        sa = r.get("solve_accounting")
        if not sa:
            L.append(f"| {r['run_id']} | - | - | - | - | - | - | - | - | - |")
            continue
        i, s, o, t = sa["initialization"], sa["scaling_gn_diagonal"], sa["optimization_lbfgsb"], sa["total"]
        L.append(f"| {r['run_id']} | {i['forward_solves']} | {i['adjoint_solves']} | {s['forward_solves']} | "
                 f"{s['adjoint_solves']} | {o['forward_solves']} | {o['adjoint_solves']} | {t['forward_solves']} | "
                 f"{t['adjoint_solves']} | {r['iterations']} |")
    L += ["", "## 4. Final feasibility and permutation-matched recovery errors", "",
          "| run | matched (true->est) | centre err 1 [m] | centre err 2 [m] | A err 1 (rel) | A err 2 (rel) | "
          "final in fluid mesh | clearances [m] | initial centre errs [m] |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "final_match" not in r:
            L.append(f"| {r['run_id']} | - | - | - | - | - | - | - | - |")
            continue
        s = r["final_match"]["sources"]
        ini = r.get("initial_match", {}).get("sources")
        L.append(f"| {r['run_id']} | {r['final_match']['permutation']} | {s[0]['center_error_m']:.4f} | "
                 f"{s[1]['center_error_m']:.4f} | {s[0]['amplitude_error']:+.4f} ({s[0]['amplitude_rel_error']:+.2%}) | "
                 f"{s[1]['amplitude_error']:+.4f} ({s[1]['amplitude_rel_error']:+.2%}) | "
                 f"{all(f['in_fluid_mesh'] for f in r['final_feasibility'])} | "
                 f"{', '.join(f'{f['signed_clearance_m']:.3f}' for f in r['final_feasibility'])} | "
                 f"{', '.join(f'{x['center_error_m']:.3f}' for x in ini) if ini else '-'} |")
    L += ["", "## 5. Clean versus noisy", "",
          "| case | clean outcome / recovery | noisy outcome / recovery | max centre err clean / noisy [m] | "
          "|z_clean - z_noisy| centres [m] | amplitudes clean | amplitudes noisy |",
          "|---|---|---|---|---|---|---|"]
    by = {(r["case"], r["data"]): r for r in rows}
    for case in sorted({r["case"] for r in rows}):
        c, n = by.get((case, "clean")), by.get((case, "noisy"))
        if not c or not n or "final_match" not in c or "final_match" not in n:
            L.append(f"| {case} | {c and c['outcome']} | {n and n['outcome']} | - | - | - | - |")
            continue
        mc = max(s["center_error_m"] for s in c["final_match"]["sources"])
        mn = max(s["center_error_m"] for s in n["final_match"]["sources"])
        dc = [np.linalg.norm(np.asarray(a["estimated_center"]) - np.asarray(b["estimated_center"]))
              for a, b in zip(c["final_match"]["sources"], n["final_match"]["sources"])]
        L.append(f"| {case} | {c['outcome']} / {c['recovery']} | {n['outcome']} / {n['recovery']} | {mc:.4f} / {mn:.4f} | "
                 f"{', '.join(f'{d:.4f}' for d in dc)} | "
                 f"{', '.join(f'{s['estimated_amplitude']:.4f}' for s in c['final_match']['sources'])} | "
                 f"{', '.join(f'{s['estimated_amplitude']:.4f}' for s in n['final_match']['sources'])} |")
    L += ["", "Centres are paired by the true source each one is matched to.", ""]
    if traces_done:
        L += ["## 6. Sensor-trace fits at the final control", "",
              "Relative residual ||C u(z_final) - d|| / ||d|| over n >= 1 (post-hoc coarse forward solves, not counted above).",
              "", "| run | all sensors | " + " | ".join(ds["sensor_labels"]) + " |",
              "|---|---|" + "---|" * len(ds["sensor_labels"])]
        for r in rows:
            tf = r.get("trace_fit")
            if tf:
                L.append(f"| {r['run_id']} | {tf['relative_residual_all']:.3e} | "
                         + " | ".join(f"{v:.2e}" for v in tf["relative_residual_per_sensor"]) + " |")
        L += ["", "Data RMS per sensor (first run of each case) shows which sensors carry signal: "]
        for r in rows:
            if r.get("trace_fit") and r["data"] == "clean":
                L.append(f"- case {r['case']}: " + ", ".join(
                    f"{lb} {v:.3g}" for lb, v in zip(ds["sensor_labels"], r["trace_fit"]["data_rms_per_sensor"])))
        L.append("")
    L += ["## Figures", ""]
    for p in sorted(rep.glob("fig_*.png")):
        L.append(f"- `report/{p.name}`")
    L += ["", "## Failures and limits", ""]
    bad = [r for r in rows if r["outcome"] in ("FAILED", "ITERATION_LIMIT", "NOT_FINISHED")]
    if not bad:
        L.append("None: every run exited 0 and stopped before the iteration limit.")
    for r in bad:
        log = f"logs/{r['run_id']}.log"
        L.append(f"- **{r['run_id']}: {r['outcome']}** (exit {r['exit'].get('rc')}; see `{log}`)"
                 + (f"; stopped at {r.get('iterations')} iterations: {r.get('termination_message')}" if r["outcome"] == "ITERATION_LIMIT" else ""))
    (B / "REPORT.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
