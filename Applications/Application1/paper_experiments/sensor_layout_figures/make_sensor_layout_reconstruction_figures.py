"""Sensor-layout reconstruction figures from the selected gn-diag runs.

Reads only saved outputs of the six P_<cfg>_<layout>__gn-diag cases in this
study (optimization_summary.json, source_paths.csv, optimization_history.csv)
plus the App1 DC mesh geometry; no PDE solves or optimization.

Writes, without touching any older figure:
  cases/<case>/reconstruction_<case>.{png,pdf}         per-case map + zooms
  overview_sensor_layout_reconstructions.{png,pdf}     2 x 3 overview (study root)

Run with the project environment (needs meshio), e.g.
  python make_sensor_layout_reconstruction_figures.py --study-dir <gn-diag batch folder>
"""
from __future__ import annotations

import json
import sys
from itertools import permutations
from pathlib import Path

import numpy as np

import os  # noqa: E402

# Applications/Application1 (this file lives in paper_experiments/sensor_layout_figures/).
APP = Path(os.environ["APP1_DIR"]).resolve() if os.environ.get("APP1_DIR") else Path(__file__).resolve().parents[2]
# Folder holding cases/<case>/full/: the original gn-diag batch by default, or
# --study-dir / REFERENCE_RUNS_DIR (runs regenerated with paper_experiments/run_reference.py).
STUDY = APP / "output" / "gn_diag_default_check_20261004_161718"
sys.path.insert(0, str(APP))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Polygon, Rectangle  # noqa: E402

import mesh_data_config as mdc  # noqa: E402

CONFIGS = ["A", "B"]
LAYOUTS = [("L1_boundary", "boundary sensors"),
           ("L2_mixed", "mixed sensors"),
           ("L3_weak", "weak sensors")]
CASE_FMT = "P_{cfg}_{layout}__gn-diag"
if os.environ.get("REFERENCE_RUNS_DIR"):
    STUDY = Path(os.environ["REFERENCE_RUNS_DIR"]).resolve()
    CASE_FMT = "restart_{cfg}_{layout}"

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Okabe-Ito colour-blind-safe palette, one colour per *true* source
SRC_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
BUILDING_FACE = "0.86"
BUILDING_EDGE = "0.25"
DOMAIN_EDGE = "0.35"
SENSOR_COLOR = "0.15"
ZOOM_MAX_EXTENT_M = 120.0  # beyond this the matched estimate is not shown in the zoom
ZOOM_FALLBACK_HALF_M = 25.0


# ----------------------------------------------------------------- data
def edge_loops(edges):
    """Chain an unordered list of (a, b) node edges into closed node loops."""
    adj = {}
    for a, b in edges:
        adj.setdefault(int(a), []).append(int(b))
        adj.setdefault(int(b), []).append(int(a))
    used = set()
    loops = []
    for start in adj:
        for nxt in adj[start]:
            if (start, nxt) in used:
                continue
            loop = [start]
            cur = nxt
            used.update({(start, nxt), (nxt, start)})
            while cur != start:
                loop.append(cur)
                cand = [n for n in adj[cur] if (cur, n) not in used]
                if not cand:
                    break
                used.update({(cur, cand[0]), (cand[0], cur)})
                cur = cand[0]
            loops.append(loop)
    return loops


def match_min_total_position_error(true_c, cand_c):
    """Same rule as run_optimization.match_sources_by_permutation:
    permutation[i] = candidate matched to true source i, minimizing the summed
    centre distance."""
    best, best_cost = None, np.inf
    for perm in permutations(range(len(true_c))):
        cost = sum(np.hypot(*(cand_c[perm[i], :2] - true_c[i, :2])) for i in range(len(true_c)))
        if cost < best_cost:
            best, best_cost = perm, cost
    pos = [float(np.hypot(*(cand_c[best[i], :2] - true_c[i, :2]))) for i in range(len(true_c))]
    amp = [float(abs(cand_c[best[i], 2] - true_c[i, 2])) for i in range(len(true_c))]
    return list(best), pos, amp


def load_case(case, md):
    full = STUDY / "cases" / case / "full"
    summ = json.loads((full / "optimization_summary.json").read_text())
    paths = np.genfromtxt(full / "source_paths" / "source_paths.csv", delimiter=",", names=True)
    hist = np.genfromtxt(full / "optimization_history.csv", delimiter=",", names=True)

    true_c = np.asarray(summ["true_control"], float).reshape(-1, 3)
    init_c = np.asarray(summ["initial_control"], float).reshape(-1, 3)
    final_c = np.asarray(summ["final_control"], float).reshape(-1, 3)
    M = true_c.shape[0]
    sensors = np.asarray(summ["observation"]["sensor_coordinates"], float)
    ids = summ["observation"]["sensor_node_ids"]
    assert len(sensors) == summ["observation"]["num_sensors"] == len(ids)
    assert np.allclose(md["coord"][ids, :2], sensors, atol=1e-9, rtol=0), case

    traj = []
    for sid in range(1, M + 1):
        m = paths["source_id"] == sid
        o = np.argsort(paths["iter"][m])
        traj.append(np.column_stack([paths["iter"][m][o], paths["x"][m][o],
                                     paths["y"][m][o], paths["A"][m][o]]))

    # --- verification: trajectories start/end at saved controls ----------
    n_it = int(summ["iterations"])
    for j, p in enumerate(traj):
        assert p[0, 0] == 0 and p[-1, 0] == n_it and len(p) == n_it + 1, (case, j)
        assert np.abs(p[0, 1:] - init_c[j]).max() < 1e-9, (case, j, "start")
        assert np.abs(p[-1, 1:] - final_c[j]).max() < 1e-9, (case, j, "end")
    # history file agrees with paths and with the summary objective values
    assert len(hist) == n_it + 1
    for j in range(M):
        hx = np.column_stack([hist[f"x{j + 1}"], hist[f"y{j + 1}"], hist[f"A{j + 1}"]])
        assert np.abs(hx - traj[j][:, 1:]).max() < 1e-9, (case, j, "history")
    assert np.isclose(hist["objective"][0], summ["initial_objective"], rtol=1e-12)
    assert np.isclose(hist["objective"][-1], summ["final_objective"], rtol=1e-12)

    # --- verification: label matching agrees with the reported errors ------
    rec = summ["recovery_permutation_matched"]
    perm, pos, amp = match_min_total_position_error(true_c, final_c)
    assert perm == list(rec["permutation"]), (case, perm, rec["permutation"])
    assert np.allclose(pos, rec["position_errors_m"], rtol=1e-10, atol=1e-12), case
    assert np.allclose(amp, rec["amplitude_errors"], rtol=1e-10, atol=1e-12), case

    return dict(case=case, summ=summ, true=true_c, init=init_c, final=final_c,
                sensors=sensors, traj=traj, perm=perm, pos_err=rec["position_errors_m"],
                amp_err=rec["amplitude_errors"], M=M)


def status_text(summ):
    """Optimizer status taken from the summary message (no 'converged' claim
    for an iteration-limit stop)."""
    msg = summ["message"]
    n = summ["iterations"]
    if "ITERATIONS REACHED LIMIT" in msg:
        return f"L-BFGS-B stopped at the iteration limit ({n} iterations)"
    if "RELATIVE REDUCTION OF F" in msg:
        return f"L-BFGS-B stopped after {n} iterations (relative-reduction criterion)"
    if "PROJ" in msg and "GRAD" in msg:
        return f"L-BFGS-B stopped after {n} iterations (projected-gradient criterion)"
    return f"L-BFGS-B stopped after {n} iterations ({msg.strip().lower()})"


def sci(v):
    if v >= 0.1:
        return f"{v:.3g}"
    return f"{v:.1e}".replace("e-0", "e-").replace("e+0", "e+")


# ----------------------------------------------------------------- drawing
def draw_geometry(ax, md):
    xy = md["coord"][:, :2]
    for name, edges in md["boundary_edges_by_name"].items():
        if name.startswith("building"):
            for loop in edge_loops(edges):
                ax.add_patch(Polygon(xy[loop], closed=True, facecolor=BUILDING_FACE,
                                     edgecolor=BUILDING_EDGE, lw=0.6, zorder=1))
        else:
            for a, b in np.asarray(edges):
                ax.plot(xy[[a, b], 0], xy[[a, b], 1], color=DOMAIN_EDGE, lw=0.9,
                        zorder=1, solid_capstyle="butt")


def draw_case(ax, c, *, detail):
    """Sensors, paths, true / initial / final centres. Each estimate is coloured
    by the true source it is matched to (summary permutation)."""
    s = c["sensors"]
    ax.plot(s[:, 0], s[:, 1], "v", color=SENSOR_COLOR, ms=5 if detail else 3.6,
            mec="white", mew=0.4, ls="none", zorder=3)
    handles = {}
    for i in range(c["M"]):
        col = SRC_COLORS[i]
        p = c["traj"][c["perm"][i]]
        ax.plot(p[:, 1], p[:, 2], "-", color=col, lw=1.2 if detail else 1.0, zorder=4)
        if detail:
            ax.plot(p[1:-1, 1], p[1:-1, 2], ".", color=col, ms=2.5, zorder=4)
        ax.plot(*c["true"][i, :2], "*", mfc="none", mec="black",
                ms=16 if detail else 10, mew=0.9, zorder=6)
        ax.plot(p[0, 1], p[0, 2], "o", mfc="white", mec=col,
                ms=7 if detail else 4.5, mew=1.3, zorder=5)
        (h,) = ax.plot(p[-1, 1], p[-1, 2], "X", color=col, mec="black", mew=0.4,
                       ms=7 if detail else 5, zorder=7)
        handles[i] = h
    return handles


def legend_handles(n_sources, n_sensors=None):
    sens = "Sensors" if n_sensors is None else f"Sensors ({n_sensors})"
    return [
        Line2D([], [], marker="*", ls="none", mfc="none", mec="black", ms=11, mew=0.9,
               label="True center"),
        Line2D([], [], marker="o", ls="none", mfc="white", mec="0.35", ms=6, mew=1.3,
               label="Initialization"),
        Line2D([], [], marker="X", ls="none", color="0.35", mec="black", mew=0.4, ms=7,
               label="Recovered center"),
        Line2D([], [], color="0.35", lw=1.2, marker=".", ms=3, label="L-BFGS-B path"),
        Line2D([], [], marker="v", ls="none", color=SENSOR_COLOR, mec="white", mew=0.4,
               ms=5, label=sens),
        Patch(fc=BUILDING_FACE, ec=BUILDING_EDGE, lw=0.6, label="Buildings"),
    ] + [Patch(fc=SRC_COLORS[i], ec="none", label=f"Source S{i + 1}")
         for i in range(n_sources)]


def set_domain(ax, summ):
    lx, ly = summ["pde"]["lx"], summ["pde"]["ly"]
    ax.set_xlim(-12, lx + 12)
    ax.set_ylim(-12, ly + 12)
    ax.set_aspect("equal")


def zoom_window(c, i):
    """Square window centred on true source i that contains the matched
    estimate's whole path; None if that path wanders too far to zoom on."""
    p = c["traj"][c["perm"][i]]
    ext = float(np.max(np.abs(p[:, 1:3] - c["true"][i, :2])))
    if ext > ZOOM_MAX_EXTENT_M:
        return ZOOM_FALLBACK_HALF_M, False
    return max(0.3, 1.25 * ext), True


def case_figure(c, md, title, out_stem):
    summ, M = c["summ"], c["M"]
    fig = plt.figure(figsize=(6.4, 6.6))
    gs = fig.add_gridspec(M, 2, width_ratios=[1.0, 1.0], wspace=0.12, hspace=0.55,
                          left=0.08, right=0.975, top=0.885, bottom=0.21)
    ax = fig.add_subplot(gs[:, 0])
    zooms = [fig.add_subplot(gs[i, 1]) for i in range(M)]

    plotted = {}
    for k, a in enumerate([ax] + zooms):
        draw_geometry(a, md)
        h = draw_case(a, c, detail=k > 0)
        if k == 0:
            plotted = h
    set_domain(ax, summ)
    ax.set_xlabel("$x$ (m)")
    ax.set_ylabel("$y$ (m)")
    ax.set_title("Full domain")

    for i, az in enumerate(zooms):
        tx, ty = c["true"][i, :2]
        half, contains = zoom_window(c, i)
        az.set_xlim(tx - half, tx + half)
        az.set_ylim(ty - half, ty + half)
        az.set_aspect("equal")
        az.locator_params(nbins=4)
        az.set_title(f"Source S{i + 1} (true $A$ = {c['true'][i, 2]:g})",
                     color=SRC_COLORS[i], fontweight="bold")
        for sp in az.spines.values():
            sp.set_edgecolor(SRC_COLORS[i])
            sp.set_linestyle("--")
            sp.set_linewidth(0.9)
        ax.add_patch(Rectangle((tx - half, ty - half), 2 * half, 2 * half, fill=False,
                               ec=SRC_COLORS[i], lw=0.8, ls="--", zorder=5))
        ax.annotate(f"S{i + 1}", (tx, ty), xytext=(-6, 9) if tx > 150 else (8, 0),
                    textcoords="offset points", va="center",
                    ha="right" if tx > 150 else "left", fontsize=8.5,
                    fontweight="bold", color=SRC_COLORS[i], zorder=8)
        txt = (f"position error {sci(c['pos_err'][i])} m, "
               f"amplitude error {sci(c['amp_err'][i])}")
        if not contains:
            j = c["perm"][i]
            fx, fy = c["final"][j, :2]
            txt += f"\nmatched estimate outside view, at ({fx:.1f}, {fy:.1f}) m"
        az.set_xlabel(txt, fontsize=7, color="0.2")

    fig.legend(handles=legend_handles(M, len(c["sensors"])), loc="lower center",
               ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.045), columnspacing=1.4,
               handletextpad=0.5)
    fig.suptitle(title, fontsize=11, y=0.975)
    sub = (f"{status_text(summ)};  $J$: {sci(summ['initial_objective'])} "
           f"$\\rightarrow$ {sci(summ['final_objective'])}")
    fig.text(0.5, 0.925, sub, ha="center", fontsize=8, color="0.25")
    nbad = summ.get("feasibility", {}).get("raw_infeasible_iterates", 0) or 0
    if nbad:
        fig.text(0.5, 0.012,
                 f"Raw feasibility mode: {nbad} of {summ['iterations']} path iterates "
                 f"have a center inside a building.", ha="center", fontsize=7,
                 color="0.3", style="italic")

    # verify plotted final markers are the saved controls
    for i, h in plotted.items():
        xy = np.array(h.get_data()).ravel()
        assert np.array_equal(xy, c["final"][c["perm"][i], :2])

    fig.savefig(out_stem.with_suffix(".pdf"))
    fig.savefig(out_stem.with_suffix(".png"), dpi=300)
    plt.close(fig)


def overview_figure(cases, md, out_stem):
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 9.4))
    fig.subplots_adjust(left=0.085, right=0.99, top=0.925, bottom=0.115,
                        wspace=0.12, hspace=0.20)
    for r, cfg in enumerate(CONFIGS):
        for k, (lay, lay_label) in enumerate(LAYOUTS):
            ax = axes[r, k]
            c = cases[(cfg, lay)]
            draw_geometry(ax, md)
            h = draw_case(ax, c, detail=False)
            for i, hh in h.items():
                assert np.array_equal(np.array(hh.get_data()).ravel(),
                                      c["final"][c["perm"][i], :2])
            set_domain(ax, c["summ"])
            limit = "ITERATIONS REACHED LIMIT" in c["summ"]["message"]
            it = f"{c['summ']['iterations']} it." + (" (limit)" if limit else "")
            ax.set_title(f"Configuration {cfg} — {lay_label}\n"
                         f"{it}, max. error {sci(max(c['pos_err']))} m",
                         fontsize=8)
            ax.tick_params(labelsize=6.5)
            if k == 0:
                ax.set_ylabel("$y$ (m)")
            else:
                ax.set_yticklabels([])
            if r == 1:
                ax.set_xlabel("$x$ (m)")
    n_src = max(c["M"] for c in cases.values())
    fig.legend(handles=legend_handles(n_src), loc="lower center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, 0.0), columnspacing=1.4,
               handletextpad=0.5, fontsize=7.5)
    fig.suptitle("Sensor-layout reconstructions", fontsize=11, y=0.985)
    fig.savefig(out_stem.with_suffix(".pdf"))
    fig.savefig(out_stem.with_suffix(".png"), dpi=300)
    plt.close(fig)


def main():
    md = mdc.build_mesh_data()
    cases = {}
    for cfg in CONFIGS:
        for lay, lay_label in LAYOUTS:
            name = CASE_FMT.format(cfg=cfg, layout=lay)
            c = load_case(name, md)
            cases[(cfg, lay)] = c
            out = STUDY / "cases" / name / f"reconstruction_{name}"
            case_figure(c, md, f"Configuration {cfg} — {lay_label}", out)
            print(f"{name}: verified (paths start/end at saved controls, history "
                  f"consistent, matching {c['perm']} reproduces reported errors)")
            print(f"    {status_text(c['summ'])}; pos. errors "
                  f"{[sci(e) for e in c['pos_err']]} m; amp. errors "
                  f"{[sci(e) for e in c['amp_err']]}")
            print(f"    wrote {out}.pdf / .png")
    out = STUDY / "overview_sensor_layout_reconstructions"
    overview_figure(cases, md, out)
    print(f"wrote {out}.pdf / .png")


if __name__ == "__main__":
    if "--study-dir" in sys.argv:
        STUDY = Path(sys.argv[sys.argv.index("--study-dir") + 1]).resolve()
    main()
