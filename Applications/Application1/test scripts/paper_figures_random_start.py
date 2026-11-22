"""Paper figures and tables for the App1 random-start batches (post hoc; no run output is changed).

    PY "Applications/Application1/test scripts/paper_figures_random_start.py" [--skip-traces]

Batches:
  output/random_start_baseline_20261114_224229     10 random starts x A_L1, A_L2, B_L1, B_L2, M5_S4
  output/random_start_budget_M5_20261116_135850    five sources, ten starts at the reference's PDE-sweep budget
References: the sensitivity-initialized runs named in each batch's config/reference.json.

Everything is read from saved CSV/JSON output.  Recovery flags and errors are the batches' own values; path colours use
the same minimum-total-centre-error matching, and the recomputed error is checked against the saved one.  The sensor-trace
figures need post-hoc forward solves (4 per case + 1), cached in <batch>/paper_figures/data/traces_<case>.npz and not part of
any reported solve count.

Writes <batch>/paper_figures/: figures (PDF + 300 dpi PNG), tables/ (booktabs .tex + .csv), tables_preview.pdf, FIGURES.md.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402
from matplotlib.ticker import MaxNLocator, ScalarFormatter  # noqa: E402

APP1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP1))
OUTPUT = APP1 / "output"
# Folder holding the two batch folders (and the reference runs' batch folders).
# Default: the live output/ tree.  To regenerate the paper tables/figures from
# the shipped results, copy paper_experiments/reference_results somewhere and
# set RANDOM_START_RESULTS_ROOT to that copy (outputs go to <batch>/paper_figures/).
RESULTS_ROOT = (Path(os.environ["RANDOM_START_RESULTS_ROOT"]).resolve()
                if os.environ.get("RANDOM_START_RESULTS_ROOT") else OUTPUT)
BASE = RESULTS_ROOT / "random_start_baseline_20261114_224229"
BUDGET = RESULTS_ROOT / "random_start_budget_M5_20261116_135850"


def _run_dir(recorded: str) -> Path:
    """A recorded run folder, re-rooted at RESULTS_ROOT when it is not present as recorded."""
    p = Path(recorded)
    if p.exists():
        return p
    text = str(recorded).replace("\\", "/")
    if "/output/" in text or text.startswith("output/"):
        tail = text.split("output/", 1)[1] if text.startswith("output/") else text.split("/output/", 1)[1]
        cand = RESULTS_ROOT / tail
        if cand.exists():
            return cand
    return p

CASES = ["A_L1", "A_L2", "B_L1", "B_L2", "M5_S4"]
NAME = {"A_L1": "A-L1", "A_L2": "A-L2", "B_L1": "B-L1", "B_L2": "B-L2", "M5_S4": "M5"}
SRC = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
CASE_CLR = dict(zip(CASES, SRC))
REC, NOT, REF, SENSOR = "#0072B2", "#D55E00", "#111111", "#555555"


def set_style():
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 9,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.direction": "in",
        "ytick.direction": "in", "xtick.major.size": 3, "ytick.major.size": 3, "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5, "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------ data ------------------------------------------
def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def history(path):
    return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)


def match(z, zt):
    """perm[i] = estimated source matched to true source i (minimum total centre error), plus centre errors."""
    e, t = np.asarray(z, float).reshape(-1, 3), np.asarray(zt, float).reshape(-1, 3)
    d = np.linalg.norm(e[:, None, :2] - t[None, :, :2], axis=2)
    perm = min(itertools.permutations(range(len(t))), key=lambda p: sum(d[q, i] for i, q in enumerate(p)))
    return list(perm), np.array([d[q, i] for i, q in enumerate(perm)])


def end_at(ctrls, final):
    """Iterate controls ending at the reported final control."""
    final = np.asarray(final, float)
    hit = np.flatnonzero(np.all(np.isclose(ctrls, final, rtol=1e-9, atol=1e-9), axis=1))
    return ctrls[: hit[-1] + 1] if hit.size else np.vstack([ctrls, final])


def load_base():
    cases = {c["case"]: c for c in json.loads((BASE / "config/cases.json").read_text())}
    refcfg = json.loads((BASE / "config/reference.json").read_text())
    summary = json.loads((BASE / "report/summary.json").read_text())
    refcsv = {r["case"]: r for r in read_csv(BASE / "report/reference.csv")}
    runs = read_csv(BASE / "report/runs.csv")
    bok = read_csv(BASE / "report/best_of_k.csv")
    data = {}
    for c in CASES:
        zt = np.asarray(cases[c]["true_control"], float)
        h = history(_run_dir(refcfg[c]["post_init_run_dir"]) / "full/optimization_history.csv")
        rr = [r for r in runs if r["case"] == c]
        for r in rr:
            r["z0"], r["zf"] = np.asarray(json.loads(r["initial_control"])), np.asarray(json.loads(r["final_control"]))
            r["ok"] = r["recovered"] == "True"
            r["ctrls"] = end_at(history(BASE / "cases" / r["run_id"] / "full/optimization_history.csv")[:, 3:], r["zf"])
            err = match(r["zf"], zt)[1].max()
            if not np.isclose(err, float(r["max_position_error_m"]), rtol=1e-6, atol=1e-7):
                raise SystemExit(f"{r['run_id']}: matched error {err} differs from the saved value")
        best = next(b for b in bok if b["case"] == c and b["k"] == "10")["best_run_id"]
        flags = cases[c]["common_driver_flags"]
        data[c] = {"cfg": cases[c], "zt": zt, "ref_hist": h, "ref": refcsv[c], "runs": rr,
                   "best": next(r for r in rr if r["run_id"] == best), "summary": summary["case_summary"][c],
                   "bok": [b for b in bok if b["case"] == c], "argv": refcfg[c]["post_init_driver_argv"],
                   "nodes": [int(i) for i in flags[flags.index("--sensor-node-ids") + 1].split(",")]}
    return data


def load_budget():
    per = read_csv(BUDGET / "report/per_start.csv")
    comp = read_csv(BUDGET / "report/comparison.csv")
    best_id = re.search(r"\((r\d+)\)", comp[1]["method"]).group(1)
    zt = None
    for r in per:
        r["z0"], r["zf"] = np.asarray(json.loads(r["initial_control"])), np.asarray(json.loads(r["best_feasible_control"]))
        it = read_csv(BUDGET / "cases" / r["run"] / "iterates.csv")
        cols = [k for k in it[0] if re.fullmatch(r"[xyA]\d", k)]
        z = np.array([[float(row[k]) for k in cols] for row in it])
        r["ctrls"] = end_at(np.vstack([r["z0"], z]), r["zf"])
        r["evals"] = read_csv(BUDGET / "cases" / r["run"] / "evaluations.csv")
    return {"per": per, "comp": comp, "best": next(r for r in per if r["run"] == best_id), "zt": zt}


# ------------------------------------------------------------------ drawing ---------------------------------------
def chain_loops(edges):
    adj = {}
    for a, b in edges:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))
    seen, loops = set(), []
    for start in adj:
        if start in seen:
            continue
        loop, prev, cur = [start], None, start
        seen.add(start)
        while True:
            nxt = [n for n in adj[cur] if n != prev]
            if not nxt or nxt[0] == start or nxt[0] in seen:
                break
            prev, cur = cur, nxt[0]
            loop.append(cur)
            seen.add(cur)
        loops.append(loop)
    return loops


def draw_domain(ax, md, sensors=None, labels=True):
    xy = np.asarray(md["coord"])[:, :2]
    outer = []
    for name, edges in md["boundary_edges_by_name"].items():
        e = np.asarray(edges, int).reshape(-1, 2)
        if name.startswith("building"):
            for loop in chain_loops(e):
                ax.add_patch(Polygon(xy[loop], closed=True, fc="0.88", ec="0.35", lw=0.45, zorder=1))
        else:
            outer.append(xy[e])
    ax.add_collection(LineCollection(np.concatenate(outer), colors="0.2", linewidths=0.7, zorder=2))
    if sensors is not None:
        ax.scatter(sensors[:, 0], sensors[:, 1], marker="^", s=7, color=SENSOR, linewidths=0, zorder=3)
    ax.set_xlim(xy[:, 0].min() - 8, xy[:, 0].max() + 8)
    ax.set_ylim(xy[:, 1].min() - 8, xy[:, 1].max() + 8)
    ax.set_aspect("equal")
    if labels:
        ax.set_xlabel("$x$ [m]")
        ax.set_ylabel("$y$ [m]")


def draw_paths(ax, ctrls, zt, lw=0.9, s0=20, s1=26, dots=False):
    perm, _ = match(ctrls[-1], zt)
    for i, p in enumerate(perm):
        c = SRC[i % len(SRC)]
        q = ctrls[:, [3 * p, 3 * p + 1]]
        ax.plot(q[:, 0], q[:, 1], "-", color=c, lw=lw, marker="." if dots else None, ms=2.5, zorder=5)
        ax.scatter(*q[0], marker="o", s=s0, facecolor="white", edgecolor=c, linewidths=0.8, zorder=6)
        ax.scatter(*q[-1], marker="X", s=s1, color=c, edgecolor="white", linewidths=0.3, zorder=7)
    return perm


def draw_truth(ax, zt, s=50):
    t = zt.reshape(-1, 3)
    ax.scatter(t[:, 0], t[:, 1], marker="*", s=s, color=REF, edgecolor="white", linewidths=0.35, zorder=8)


def path_legend(fig, y=0.0, extra=()):
    h = [Line2D([], [], marker="o", ls="", mfc="white", mec="#333333", ms=5, label="start"),
         Line2D([], [], color="#333333", lw=1, label="L-BFGS-B path"),
         Line2D([], [], marker="X", ls="", color="#333333", ms=5.5, label="final estimate"),
         Line2D([], [], marker="*", ls="", color=REF, ms=8, label="true source"),
         Line2D([], [], marker="^", ls="", color=SENSOR, ms=3.5, label="sensor"), *extra]
    fig.legend(handles=h, loc="lower center", ncol=len(h), bbox_to_anchor=(0.5, y), columnspacing=1.2, handletextpad=0.4)


def fig_reconstruction(out, name, md, sensors, zt, ref_ctrls, rnd_ctrls, ttl_ref, ttl_rnd):
    t = zt.reshape(-1, 3)
    m = len(t)
    fig = plt.figure(figsize=(7.2, 6.5))
    gs = fig.add_gridspec(m, 3, width_ratios=[1, 1, 0.66], left=0.07, right=0.985, top=0.93, bottom=0.115,
                          wspace=0.2, hspace=0.42)
    a, b = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[:, 1])
    for ax, ctrls, ttl in ((a, ref_ctrls, ttl_ref), (b, rnd_ctrls, ttl_rnd)):
        draw_domain(ax, md, sensors)
        draw_paths(ax, ctrls, zt)
        draw_truth(ax, zt)
        ax.set_title(ttl, loc="left", fontsize=8.5)
    b.set_ylabel("")
    perm, _ = match(ref_ctrls[-1], zt)
    for i, p in enumerate(perm):
        az = fig.add_subplot(gs[i, 2])
        q = ref_ctrls[:, [3 * p, 3 * p + 1]]
        draw_domain(az, md, labels=False)
        c = SRC[i % len(SRC)]
        az.plot(q[:, 0], q[:, 1], "-", color=c, lw=0.9, marker=".", ms=3, zorder=5)
        az.scatter(*q[0], marker="o", s=20, facecolor="white", edgecolor=c, linewidths=0.8, zorder=6)
        az.scatter(*q[-1], marker="X", s=26, color=c, edgecolor="white", linewidths=0.3, zorder=7)
        draw_truth(az, t[i], s=42)
        pts = np.vstack([q, t[i, :2]])
        mid, half = 0.5 * (pts.min(0) + pts.max(0)), max(0.62 * np.ptp(pts, axis=0).max(), 1.0)
        az.set_xlim(mid[0] - half, mid[0] + half)
        az.set_ylim(mid[1] - half, mid[1] + half)
        for axis in (az.xaxis, az.yaxis):
            axis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]))
            axis.set_major_formatter(ScalarFormatter(useOffset=False))
        az.tick_params(labelsize=6.5, pad=1.5)
        az.set_title(f"(c{i + 1}) start error {np.hypot(*(q[0] - t[i, :2])):.1f} m", loc="left", fontsize=7, pad=2)
    path_legend(fig)
    save(fig, out, name)


def fig_starts_grid(out, name, md, sensors, zt, items):
    fig, axes = plt.subplots(2, 5, figsize=(7.2, 6.75), sharex=True, sharey=True)
    for ax, (ttl, ctrls, ok) in zip(axes.ravel(), items):
        draw_domain(ax, md, sensors, labels=False)
        draw_paths(ax, ctrls, zt, lw=0.6, s0=9, s1=13)
        draw_truth(ax, zt, s=24)
        ax.set_title(ttl, fontsize=7.5, pad=2, color=REC if ok else "#222222")
        ax.tick_params(labelsize=6.5)
    for ax in axes[1]:
        ax.set_xlabel("$x$ [m]", fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel("$y$ [m]", fontsize=8)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.965, bottom=0.095, wspace=0.08, hspace=0.1)
    path_legend(fig)
    save(fig, out, name)


def strip(ax, vals, oks, x0, rng):
    x = x0 + rng.uniform(-0.2, 0.12, len(vals))
    for flag, col, mk, s in ((False, NOT, "x", 12), (True, REC, "o", 15)):
        m = oks == flag
        ax.scatter(x[m], vals[m], marker=mk, s=s, color=col, linewidths=0.8, zorder=3)


def fig_recovery(out, data):
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75))
    spec = (("max_position_error_m", "largest centre error [m]", 1.0), ("max_amplitude_error", "largest amplitude error", 0.01),
            ("J_final_over_J0", r"$J_{\mathrm{final}}/J(0)$", None))
    for ax, (key, yl, lim), tag in zip(axes, spec, "abc"):
        for k, c in enumerate(CASES):
            d = data[c]
            strip(ax, np.array([float(r[key]) for r in d["runs"]]), np.array([r["ok"] for r in d["runs"]]), k, rng)
            ax.scatter([k + 0.3], [float(d["ref"][key])], marker="*", s=50, color=REF, linewidths=0, zorder=4)
        if lim:
            ax.axhline(lim, color="0.45", ls="--", lw=0.7)
        ax.set_yscale("log")
        ax.set_xticks(range(5), [NAME[c] for c in CASES])
        ax.set_ylabel(yl)
        ax.set_title(f"({tag})", loc="left")
        ax.grid(axis="y", alpha=0.25, lw=0.4)
    h = [Line2D([], [], marker="o", ls="", color=REC, ms=4, label="random start, recovered"),
         Line2D([], [], marker="x", ls="", color=NOT, ms=4, label="random start, not recovered"),
         Line2D([], [], marker="*", ls="", color=REF, ms=8, label="sensitivity initialization"),
         Line2D([], [], color="0.45", ls="--", lw=0.7, label="recovery threshold")]
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.legend(handles=h, loc="lower center", ncol=4)
    save(fig, out, "recovery_overview")


def fig_cost(out, data):
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    for k, c in enumerate(CASES):
        d = data[c]
        ours = int(d["ref"]["fwd_total_init_setup_opt"]) + int(d["ref"]["adj_total_init_setup_opt"])
        rnd = sum(int(r["fwd_setup_plus_opt"]) + int(r["adj_setup_plus_opt"]) for r in d["runs"])
        ax.plot([k, k], [ours, rnd], color="0.75", lw=0.8, zorder=1)
        ax.scatter([k], [ours], marker="*", s=70, color=REF, linewidths=0, zorder=3)
        ax.scatter([k], [rnd], marker="s", s=24, color=NOT, linewidths=0, zorder=3)
        for v in (ours, rnd):
            ax.annotate(str(v), (k, v), xytext=(6, 0), textcoords="offset points", va="center", fontsize=7.5)
    ax.set_yscale("log")
    ax.set_ylim(300, 22000)
    ax.set_xlim(-0.4, 4.6)
    ax.set_xticks(range(5), [NAME[c] for c in CASES])
    ax.set_ylabel("PDE solves (forward + adjoint)")
    ax.grid(axis="y", alpha=0.25, lw=0.4, which="major")
    ax.legend(handles=[Line2D([], [], marker="*", ls="", color=REF, ms=8, label="sensitivity initialization + L-BFGS-B"),
                       Line2D([], [], marker="s", ls="", color=NOT, ms=4.5, label="ten random starts (all)")],
              loc="upper left", fontsize=7.5)
    save(fig, out, "solve_cost")


def fig_best_of_k(out, data):
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.5), sharey=True)
    for ax, c, tag in zip(axes, CASES, "abcde"):
        d = data[c]
        x = np.array([int(b["cum_fwd_setup_plus_opt"]) + int(b["cum_adj_setup_plus_opt"]) for b in d["bok"]])
        y = np.array([float(b["best_J_final_over_J0"]) for b in d["bok"]])
        ok = np.array([b["best_run_recovered"] == "True" for b in d["bok"]])
        ax.step(x, y, where="post", color=NOT, lw=0.9)
        ax.scatter(x[~ok], y[~ok], marker="x", s=11, color=NOT, linewidths=0.7, zorder=3)
        ax.scatter(x[ok], y[ok], marker="o", s=12, color=REC, zorder=3)
        tot = int(d["ref"]["fwd_total_init_setup_opt"]) + int(d["ref"]["adj_total_init_setup_opt"])
        ax.scatter([tot], [float(d["ref"]["J_final_over_J0"])], marker="*", s=55, color=REF, linewidths=0, zorder=4)
        ax.set_yscale("log")
        ax.set_title(f"({tag}) {NAME[c]}", loc="left")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=2, integer=True))
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(alpha=0.25, lw=0.4)
    axes[0].set_ylabel("best $J/J(0)$ so far")
    fig.supxlabel("cumulative PDE solves (forward + adjoint)", fontsize=9, y=0.17)
    h = [Line2D([], [], marker="o", ls="", color=REC, ms=4, label="best start recovered"),
         Line2D([], [], marker="x", ls="", color=NOT, ms=4, label="best start not recovered"),
         Line2D([], [], marker="*", ls="", color=REF, ms=8, label="sensitivity initialization")]
    fig.tight_layout(rect=(0, 0.14, 1, 1), w_pad=0.3)
    fig.legend(handles=h, loc="lower center", ncol=3)
    save(fig, out, "best_of_k")


def fig_ref_convergence(out, data):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    for c in CASES:
        d = data[c]
        h = d["ref_hist"]
        st = dict(color=CASE_CLR[c], lw=1.1, label=NAME[c])
        axes[0].semilogy(h[:, 0], h[:, 1] / float(d["ref"]["J0"]), **st)
        axes[1].semilogy(h[:, 0], h[:, 2], **st)
        axes[2].semilogy(h[:, 0], [match(z, d["zt"])[1].max() for z in h[:, 3:]], **st)
    for ax, yl, tag in zip(axes, (r"$J/J(0)$", r"$\|\nabla J\|_2$", "largest centre error [m]"), "abc"):
        ax.set_xlabel("L-BFGS-B iteration")
        ax.set_ylabel(yl)
        ax.set_title(f"({tag})", loc="left")
        ax.grid(alpha=0.25, lw=0.4)
    hs, ls = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.legend(hs, ls, loc="lower center", ncol=5)
    save(fig, out, "reference_convergence")


def fig_budget_curve(out, data, bud):
    d = data["M5_S4"]
    J0 = float(d["ref"]["J0"])
    offset, xs, ys, best, bounds = 0, [], [], np.inf, []
    for r in bud["per"]:
        for e in r["evals"]:
            if e["gradient_done"] == "True" and e["feasible"] == "True":
                best = min(best, float(e["objective"]))
            if np.isfinite(best):
                xs.append(offset + int(e["sweeps_used"]))
                ys.append(best / J0)
        offset += int(r["sweeps_used"])
        bounds.append(offset)
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    for b in bounds[:-1]:
        ax.axvline(b, color="0.88", lw=0.5, zorder=0)
    ax.step(xs, ys, where="post", color=NOT, lw=1.1, label="best of random starts so far")
    xi = int(d["ref"]["fwd_init"]) + int(d["ref"]["adj_init"])
    xt = int(d["ref"]["fwd_total_init_setup_opt"]) + int(d["ref"]["adj_total_init_setup_opt"])
    ax.plot([xi, xt], [float(d["ref"]["J_initial"]) / J0, float(d["ref"]["J_final_over_J0"])], "--", color=REF, lw=0.9,
            marker="*", ms=8, label="sensitivity initialization: after init., after L-BFGS-B")
    ax.set_yscale("log")
    ax.set_xlabel("cumulative PDE solves (forward + adjoint)")
    ax.set_ylabel("$J/J(0)$")
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(loc="center left", fontsize=7.5)
    save(fig, out, "budget_objective_vs_solves")


# ------------------------------------------------------------------ traces ----------------------------------------
def solve_traces(case, argv, controls, out):
    cache = out / "data" / f"traces_{case}.npz"
    if cache.exists():
        with np.load(cache) as f:
            if all(k in f and np.array_equal(f["z_" + k], z) for k, z in controls.items() if "z_" + k in f) \
                    and all("z_" + k in f for k in controls):
                return {k: f[k] for k in [*controls, "times"]}
    import run_optimization as ro
    from mesh_data_config import build_mesh_data
    sys.argv = ["run_optimization.py", *argv]
    args = ro.parse_args()
    pde = dict(ro.PDE)
    if args.tfinal is not None:
        pde["tfinal"] = args.tfinal
    md = ro.configure_measurement_boundaries(build_mesh_data(), args.measurement_boundaries)
    md = ro.configure_observation(md, args)
    pde["velocity_nodes"], pde["velocity_field"] = ro.load_velocity_field_file(args.velocity_field_file, md)
    tr = {k: md["observation"].observe(ro.gt.run_forward(md, pde, np.asarray(z, float))) for k, z in controls.items()}
    tr["times"] = np.arange(next(iter(tr.values())).shape[0]) * pde["dt"] * pde["jprint"]
    np.savez_compressed(cache, **tr, **{"z_" + k: np.asarray(z, float) for k, z in controls.items()})
    set_style()
    print(f"traces solved: {case}", flush=True)
    return tr


def trace_axes(axes, tr, nodes, keys, ylabel, by_misfit=False):
    score = tr["rnd"] - tr["truth"] if by_misfit else tr["truth"]
    order = np.argsort(-np.sqrt(np.mean(score[1:] ** 2, axis=0)))[: len(axes)]
    stride = max(1, tr["truth"].shape[0] // 400)
    t = tr["times"][::stride]
    style = {"truth": ("#9A9A9A", 2.0, "-"), "init": ("#E69F00", 0.9, "--"), "ref": (REC, 0.9, "-"), "rnd": (NOT, 0.9, "-")}
    for ax, j in zip(axes, order):
        for k in keys:
            c, lw, ls = style[k]
            ax.plot(t, tr[k][::stride, j], color=c, lw=lw, ls=ls)
        ax.set_title(f"node {nodes[j]}", fontsize=8, pad=2)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel(ylabel)


TRACE_LEGEND = [Line2D([], [], color="#9A9A9A", lw=2, label="data (true control)"),
                Line2D([], [], color="#E69F00", lw=1, ls="--", label="sensitivity initialization"),
                Line2D([], [], color=REC, lw=1, label="final, sensitivity initialization"),
                Line2D([], [], color=NOT, lw=1, label="final, best random start")]


def fig_traces(out, data, traces):
    fig, axes = plt.subplots(5, 4, figsize=(7.2, 7.4), sharex=True)
    for row, c in zip(axes, CASES):
        trace_axes(row, traces[c], data[c]["nodes"], ("truth", "rnd", "init", "ref"), f"{NAME[c]}\n$u$")
    for ax in axes[-1]:
        ax.set_xlabel("$t$ [s]")
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=0.5)
    fig.legend(handles=TRACE_LEGEND, loc="lower center", ncol=4, columnspacing=1.2)
    save(fig, out, "sensor_traces")


def fig_budget_traces(out, data, tr):
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.25))
    tb = dict(tr, rnd=tr["budget_best"])
    trace_axes(axes, tb, data["M5_S4"]["nodes"], ("truth", "rnd", "init", "ref"), "$u$", by_misfit=True)
    for ax in axes:
        ax.set_xlabel("$t$ [s]")
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.legend(handles=TRACE_LEGEND, loc="lower center", ncol=4, columnspacing=1.2)
    save(fig, out, "budget_sensor_traces")


# ------------------------------------------------------------------ tables ----------------------------------------
def sci(v, d=2):
    v = float(v)
    if v == 0:
        return "0"
    m, e = f"{v:.{d}e}".split("e")
    return rf"${m}\times 10^{{{int(e)}}}$"


def sig(v):
    v = float(v)
    return f"{v:.3g}" if 1e-2 <= abs(v) < 1e4 else sci(v, 2)


def txt(v):
    return str(v).replace("_", r"\_")


def table(tdir, name, caption, spec, cols, rows, note=None, longtable=False):
    """cols: (LaTeX header, CSV header, formatter or None).  rows hold raw values, so the CSV keeps full precision."""
    with open(tdir / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([c[1] for c in cols])
        w.writerows(rows)
    cell = lambda fn, v: "--" if v is None else (fn(v) if fn else txt(v))
    body = [" & ".join(cell(c[2], v) for c, v in zip(cols, r)) + r" \\" for r in rows]
    head = " & ".join(c[0] for c in cols) + r" \\"
    if longtable:
        L = [r"{\scriptsize", rf"\begin{{longtable}}{{{spec}}}", rf"\caption{{{caption}}}\label{{tab:{name}}} \\",
             r"\toprule", head, r"\midrule", r"\endfirsthead", r"\toprule", head, r"\midrule", r"\endhead",
             r"\bottomrule", r"\endfoot", *body, r"\end{longtable}", "}"]
        if note:
            L.append(rf"\noindent{{\scriptsize {note}}}")
    else:
        L = [r"\begin{table}[htbp]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{tab:{name}}}", r"\footnotesize",
             rf"\begin{{tabular}}{{{spec}}}", r"\toprule", head, r"\midrule", *body, r"\bottomrule", r"\end{tabular}"]
        if note:
            L += [r"\par\smallskip", rf"\parbox{{0.92\linewidth}}{{\scriptsize {note}}}"]
        L.append(r"\end{table}")
    (tdir / f"{name}.tex").write_text("\n".join(L) + "\n")


def base_tables(tdir, data):
    I = lambda v: str(int(v))
    rows = []
    for c in CASES:
        t = data[c]["zt"].reshape(-1, 3)
        rows.append([NAME[c], len(t), len(data[c]["nodes"]),
                     "; ".join(f"({x:g}, {y:g}, {a:g})" for x, y, a in t)])
    table(tdir, "table_cases", r"Test cases: number of sources $M$, number of sensors and true sources $(x, y, A)$.",
          "lccp{0.62\\linewidth}", [("Case", "case", None), ("$M$", "num_sources", I), ("Sensors", "num_sensors", I),
                                    ("True sources $(x\\,[\\mathrm{m}], y\\,[\\mathrm{m}], A)$", "true_sources", None)], rows)

    rows = []
    for c in CASES:
        r = data[c]["ref"]
        rows.append([NAME[c], int(r["iterations"]), float(r["J_final_over_J0"]), float(r["max_position_error_m"]),
                     float(r["max_amplitude_error"]), int(r["fwd_init"]), int(r["adj_init"]),
                     int(r["fwd_post_init_gn_plus_opt"]), int(r["adj_post_init_gn_plus_opt"]),
                     int(r["fwd_total_init_setup_opt"]), int(r["adj_total_init_setup_opt"])])
    table(tdir, "table_reference", "Sensitivity initialization followed by GN-diagonal-scaled L-BFGS-B: recovery and PDE solves.",
          "lrcccrrrrrr", [("Case", "case", None), ("Iter.", "iterations", I), ("$J/J(0)$", "J_final_over_J0", sci),
                          ("Centre err. [m]", "max_center_error_m", sci), ("Amp. err.", "max_amplitude_error", sci),
                          ("Init. F", "init_forward", I), ("Init. A", "init_adjoint", I), ("Local F", "local_forward", I),
                          ("Local A", "local_adjoint", I), ("Total F", "total_forward", I), ("Total A", "total_adjoint", I)],
          rows, "F, A: forward and adjoint solves. Local = GN-diagonal construction plus L-BFGS-B. Errors are the largest over "
                "the sources after permutation matching.")

    rows = []
    for c in CASES:
        s, r = data[c]["summary"], data[c]["ref"]
        rows.append([NAME[c], f"{s['recovered']}/{s['planned']}", s["converged"], s["converged_but_not_recovered"],
                     s["iteration_limit"], s["final_infeasible"], "yes" if r["recovered_by_same_criterion"] == "True" else "no"])
    table(tdir, "table_recovery", "Ten random starts per case: recovery and optimizer outcome.", "lcccccc",
          [("Case", "case", None), ("Recovered", "recovered", None), ("Converged", "converged", I),
           ("Conv., not rec.", "converged_not_recovered", I), ("Iter. limit", "iteration_limit", I),
           ("Final infeasible", "final_infeasible", I), ("Sens. init. rec.", "sensitivity_init_recovered", None)], rows,
          r"Recovered: all final centres in the fluid, largest matched centre error $\le 1$ m and largest matched amplitude "
          r"error $\le 0.01$ (fixed before the runs).")

    rows = []
    for c in CASES:
        for key, lab in (("max_position_error_m", "Largest centre error [m]"), ("max_amplitude_error", "Largest amplitude error"),
                         ("J_final_over_J0", "$J/J(0)$"), ("iterations", "L-BFGS-B iterations")):
            v = np.array([float(r[key]) for r in data[c]["runs"]])
            rows.append([NAME[c], lab, v.min(), float(np.median(v)), v.max()])
    D = lambda v: f"{v:g}" if v >= 1 and float(2 * v).is_integer() else sig(v)
    table(tdir, "table_distributions", "Random starts: minimum, median and maximum over the ten starts of each case.",
          "llccc", [("Case", "case", None), ("Quantity", "quantity", lambda v: v), ("Min", "min", D), ("Median", "median", D),
                    ("Max", "max", D)], rows)

    rows = []
    for c in CASES:
        d = data[c]
        rf, ra = int(d["ref"]["fwd_total_init_setup_opt"]), int(d["ref"]["adj_total_init_setup_opt"])
        sf = sum(int(r["fwd_setup_plus_opt"]) for r in d["runs"])
        sa = sum(int(r["adj_setup_plus_opt"]) for r in d["runs"])
        first = next((b for b in d["bok"] if b["any_of_first_k_recovered"] == "True"), None)
        rows.append([NAME[c], rf, ra, rf + ra, sf, sa, sf + sa, int(first["k"]) if first else None,
                     int(first["cum_fwd_setup_plus_opt"]) + int(first["cum_adj_setup_plus_opt"]) if first else None])
    table(tdir, "table_cost", "PDE solves: sensitivity initialization against ten random starts.", "lrrrrrrcr",
          [("Case", "case", None), ("Sens. F", "sens_forward", I), ("Sens. A", "sens_adjoint", I), ("Sens. total", "sens_total", I),
           ("Random F", "random_forward", I), ("Random A", "random_adjoint", I), ("Random total", "random_total", I),
           ("First rec. $k$", "first_recovered_k", I), ("Solves to $k$", "solves_through_k", I)], rows,
          "Random totals cover all ten starts (GN-diagonal setup plus L-BFGS-B). First rec. $k$: first start, in the fixed order "
          "r01--r10, that recovers the sources; a dash means none did. Identifying it uses the true sources.")

    rows = []
    for c in CASES:
        for r in data[c]["runs"]:
            rows.append([f"{NAME[c]} {r['run_id'][-3:]}", int(r["iterations"]), r["convergence"].replace("iteration_or_eval_limit", "limit"),
                         float(r["J_final_over_J0"]), float(r["max_position_error_m"]), float(r["max_amplitude_error"]),
                         "yes" if r["final_centers_feasible"] == "True" else "no", "yes" if r["ok"] else "no",
                         int(r["fwd_setup_plus_opt"]), int(r["adj_setup_plus_opt"])])
    table(tdir, "table_all_runs", "All fifty random-start runs.", "lrlcccccrr",
          [("Run", "run_id", None), ("Iter.", "iterations", I), ("Stop", "stop", None), ("$J/J(0)$", "J_final_over_J0", sig),
           ("Centre err. [m]", "max_center_error_m", sig), ("Amp. err.", "max_amplitude_error", sig),
           ("Feasible", "final_feasible", None), ("Recovered", "recovered", None), ("F", "forward", I), ("A", "adjoint", I)],
          rows, longtable=True)


def budget_tables(tdir, data, bud):
    I = lambda v: str(int(v))
    label = {"sensitivity": "Sensitivity init. + L-BFGS-B", "best": rf"Best random start ({bud['best']['run']}; solves of all 10)"}
    rows = []
    for key, r in zip(("sensitivity", "best"), bud["comp"]):
        rows.append([label[key], int(r["forward_sweeps"]), int(r["adjoint_sweeps"]), int(r["total_sweeps"]),
                     float(r["objective_over_J0"]), float(r["max_center_err_m"]), float(r["max_amp_err"]),
                     "converged" if key == "sensitivity" else "budget"])
    table(tdir, "table_budget_comparison", "Five-source case at a matched PDE-solve budget.", "lrrrcccl",
          [("Method", "method", None), ("F", "forward", I), ("A", "adjoint", I), ("Total", "total", I),
           ("$J/J(0)$", "J_over_J0", sig), ("Centre err. [m]", "max_center_error_m", sig), ("Amp. err.", "max_amplitude_error", sig),
           ("Stop", "stop", None)], rows, "The best random start is the one with the lowest objective; errors are evaluated afterwards.")

    rows = [[r["run"], int(r["allowance_sweeps"]), int(r["fwd_used"]), int(r["adj_used"]), int(r["iterations"]),
             float(r["best_over_J0"]), float(r["max_center_err_m"]), float(r["max_amp_err"])] for r in bud["per"]]
    table(tdir, "table_budget_starts", "Five-source case: the ten budgeted random starts (all stopped on the budget).", "lrrrrccc",
          [("Start", "run", None), ("Allowance", "allowance", I), ("F", "forward", I), ("A", "adjoint", I), ("Iter.", "iterations", I),
           ("Best $J/J(0)$", "best_J_over_J0", sig), ("Centre err. [m]", "max_center_error_m", sig),
           ("Amp. err.", "max_amplitude_error", sig)], rows)

    d = data["M5_S4"]
    t = d["zt"].reshape(-1, 3)
    z0, zf = d["ref_hist"][0, 3:], d["ref_hist"][-1, 3:]
    pf, ef = match(zf, d["zt"])
    p0, e0 = match(z0, d["zt"])
    pb, eb = match(bud["best"]["zf"], d["zt"])
    zf3, zb3 = zf.reshape(-1, 3), bud["best"]["zf"].reshape(-1, 3)
    rows = [[i + 1, t[i, 0], t[i, 1], t[i, 2], e0[i], ef[i], abs(zf3[pf[i], 2] - t[i, 2]), zb3[pb[i], 0], zb3[pb[i], 1],
             zb3[pb[i], 2], eb[i]] for i in range(len(t))]
    F = lambda v: f"{v:g}"
    table(tdir, "table_budget_sources", rf"Five-source case, source by source: sensitivity initialization and best random start ({bud['best']['run']}).",
          "c" * 11, [("Src.", "source", I), ("$x$", "true_x", F), ("$y$", "true_y", F), ("$A$", "true_A", F),
                     ("Init. err. [m]", "sens_init_center_error_m", lambda v: f"{v:.2f}"), ("Final err. [m]", "sens_final_center_error_m", sci),
                     ("Amp. err.", "sens_final_amplitude_error", sci), ("Rand. $x$", "random_x", lambda v: f"{v:.1f}"),
                     ("Rand. $y$", "random_y", lambda v: f"{v:.1f}"), ("Rand. $A$", "random_A", lambda v: f"{v:.2f}"),
                     ("Rand. err. [m]", "random_center_error_m", lambda v: f"{v:.3g}")], rows,
          "Estimates are paired with true sources by the permutation with the smallest total centre error, so an estimate can "
          "lie close to a true source other than the one it is paired with.")


def preview(tdir, out):
    if not shutil.which("pdflatex"):
        print("pdflatex not found: no tables_preview.pdf")
        return
    tex = [r"\documentclass[10pt]{article}", r"\usepackage[landscape,margin=1.5cm]{geometry}",
           r"\usepackage{booktabs,longtable,amsmath}", r"\begin{document}"]
    tex += [(tdir / p.name).read_text() + "\n\\clearpage" for p in sorted(tdir.glob("table_*.tex"))]
    tex.append(r"\end{document}")
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "tables_preview.tex").write_text("\n".join(tex))
        for _ in range(2):
            res = subprocess.run(["pdflatex", "-interaction=nonstopmode", "tables_preview.tex"], cwd=tmp, capture_output=True, text=True)
        if not (Path(tmp) / "tables_preview.pdf").exists() or res.returncode:
            raise SystemExit("LaTeX failed:\n" + res.stdout[-2500:])
        shutil.copy(Path(tmp) / "tables_preview.pdf", out / "tables_preview.pdf")


NOTES_BASE = """# App1 random-start baseline: paper figures and tables

Figures are vector PDF plus 300 dpi PNG. Tables are in `tables/` as booktabs LaTeX (needs `booktabs`; `table_all_runs` also needs
`longtable`) and full-precision CSV. `tables_preview.pdf` shows every table.

- `reconstruction_<case>`: sensitivity-initialized run against the random start with the lowest final objective. Colours identify
  the matched true source. The right column zooms on the sensitivity-initialized path to each source. Iterates may cross
  buildings because the optimizer runs with raw feasibility; only final centres are required to lie in the fluid.
- `starts_<case>`: all ten random starts of the case; a blue title marks a recovered start.
- `recovery_overview`: final errors and objectives of all random starts and of the sensitivity-initialized runs.
- `best_of_k`: best objective so far against cumulative solves, starts taken in the fixed order r01..r10.
- `solve_cost`: total forward + adjoint solves. The sensitivity total includes the initializer.
- `reference_convergence`: L-BFGS-B history of the sensitivity-initialized runs (iteration 0 = initialized control).
- `sensor_traces`: the four strongest sensors of each case. Traces are post-hoc forward solves, not counted in any solve total.

Scope: noiseless inverse-crime data, one truth per case, ten starts per case; the original (`p = 0`) velocity field.
"""

NOTES_BUDGET = """# App1 five-source matched-budget benchmark: paper figures and tables

Figures are vector PDF plus 300 dpi PNG. Tables are in `tables/` as booktabs LaTeX and full-precision CSV; see `tables_preview.pdf`.

- `budget_reconstruction`: sensitivity-initialized run against the best random start by objective. The random start's path ends at
  its best feasible, fully evaluated point, which is what the driver reports on a budget stop.
- `budget_starts`: all ten budgeted starts.
- `budget_objective_vs_solves`: best objective so far with the starts concatenated in the order r01..r10 (light vertical lines
  separate starts). The sensitivity curve has only its two recorded points: after initialization and after L-BFGS-B.
- `budget_sensor_traces`: the four sensors where the best random start has the largest misfit (chosen to show the difference,
  not a typical sensor); post-hoc forward solves, not counted in the budget.

Scope: one five-source configuration, noiseless inverse-crime data, ten starts. A cost-matched comparison, not a statistical study.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-traces", action="store_true")
    a = ap.parse_args()
    from mesh_data_config import build_mesh_data
    set_style()
    md = build_mesh_data()
    xy = np.asarray(md["coord"])[:, :2]
    data, bud = load_base(), load_budget()
    obase, obud = BASE / "paper_figures", BUDGET / "paper_figures"
    for o in (obase, obud):
        (o / "tables").mkdir(parents=True, exist_ok=True)
        (o / "data").mkdir(parents=True, exist_ok=True)

    for c in CASES:
        d = data[c]
        s, b = xy[d["nodes"]], d["best"]
        r = d["ref"]
        status = "recovered" if b["ok"] else "not recovered"
        fig_reconstruction(obase, f"reconstruction_{c}", md, s, d["zt"], d["ref_hist"][:, 3:], b["ctrls"],
                           f"(a) Sensitivity initialization\n{int(r['fwd_total_init_setup_opt']) + int(r['adj_total_init_setup_opt'])} PDE solves",
                           f"(b) Best random start ({b['run_id'][-3:]})\n{status}, $J/J(0)=${sig(b['J_final_over_J0'])}")
        items = [(f"{q['run_id'][-3:]}: " + ("recovered" if q["ok"] else f"{float(q['max_position_error_m']):.0f} m"), q["ctrls"], q["ok"])
                 for q in d["runs"]]
        fig_starts_grid(obase, f"starts_{c}", md, s, d["zt"], items)
    fig_recovery(obase, data)
    fig_cost(obase, data)
    fig_best_of_k(obase, data)
    fig_ref_convergence(obase, data)

    m5 = data["M5_S4"]
    sm5 = xy[m5["nodes"]]
    br = bud["best"]
    total = int(m5["ref"]["fwd_total_init_setup_opt"]) + int(m5["ref"]["adj_total_init_setup_opt"])
    used = sum(int(r["sweeps_used"]) for r in bud["per"])
    fig_reconstruction(obud, "budget_reconstruction", md, sm5, m5["zt"], m5["ref_hist"][:, 3:], br["ctrls"],
                       f"(a) Sensitivity initialization\n{total} PDE solves",
                       f"(b) Best of 10 random starts ({br['run']})\n{used} PDE solves in total")
    fig_starts_grid(obud, "budget_starts", md, sm5, m5["zt"],
                    [(f"{r['run']}: {float(r['max_center_err_m']):.0f} m", r["ctrls"], False) for r in bud["per"]])
    fig_budget_curve(obud, data, bud)

    if not a.skip_traces:
        traces = {}
        for c in CASES:
            d = data[c]
            ctl = {"truth": d["zt"], "init": d["ref_hist"][0, 3:], "ref": d["ref_hist"][-1, 3:], "rnd": d["best"]["zf"]}
            if c == "M5_S4":
                ctl["budget_best"] = br["zf"]
            traces[c] = solve_traces(c, d["argv"], ctl, obase)
        fig_traces(obase, data, traces)
        fig_budget_traces(obud, data, traces["M5_S4"])

    base_tables(obase / "tables", data)
    budget_tables(obud / "tables", data, bud)
    preview(obase / "tables", obase)
    preview(obud / "tables", obud)
    (obase / "FIGURES.md").write_text(NOTES_BASE)
    (obud / "FIGURES.md").write_text(NOTES_BUDGET)
    print(f"written: {obase}\nwritten: {obud}")


if __name__ == "__main__":
    main()
