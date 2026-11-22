"""Publication figures for a finished Application2 inversion batch (post hoc; no run output is changed).

    cd Applications/Application2
    PY paper_figures_inversion_batch.py --batch-dir <batch>

Reads <batch>/report/summary.json (written by report_inversion_batch.py, including the post-hoc true controls),
<batch>/resolved_settings.json and <batch>/runs/<run>/optimization_history.csv.  The sensor-trace figures need two
coarse forward solves per run (initial and final control); these are not part of the optimization solve accounting
and are cached in <batch>/paper_figures/data/traces_<run>.npz.

Writes <batch>/paper_figures/ (each figure as vector PDF and 300 dpi PNG):
  problem_setup                 room, openings, sensors and the three release locations
  reconstruction_case_<c>       room map plus one zoom per source: initial, L-BFGS-B path, final, truth (clean, noisy)
  convergence_history           J/J(0), gradient norm and largest centre error against iteration
  traces_case_<c>               strongest four sensors: data, initial-control and final-control traces (clean, noisy)
  FIGURES.md, data/source_paths_<run>.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402
from matplotlib.ticker import FixedLocator, FormatStrFormatter, MaxNLocator, NullFormatter, ScalarFormatter  # noqa: E402

import report_inversion_batch as rib  # noqa: E402
import report_source_id as rsi  # noqa: E402  (draw_room)
import room_geometry_config as geo  # noqa: E402
import run_inversion as ri  # noqa: E402

CLEAN, NOISY, TRUTH, SENSOR = "#0072B2", "#D55E00", "#111111", "#00795C"
KIND = {"clean": CLEAN, "noisy": NOISY}
CASE_CLR = {"12": "#0072B2", "13": "#D55E00", "23": "#009E73"}
SENSOR_OFFSET = {"S1": (5, 3, "left"), "S2": (-5, -8, "right"), "S3": (5, 3, "left"), "S4": (-5, -8, "right"),
                 "S5": (5, 3, "left"), "S6": (-5, -8, "right"), "S7": (5, 3, "left"), "S8": (-5, -7, "right")}
TRACE_STRIDE = 20


def set_style():
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 9, "axes.titlesize": 9,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.direction": "in",
        "ytick.direction": "in", "xtick.major.size": 3, "ytick.major.size": 3, "legend.frameon": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def save(fig, out: Path, name: str):
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=300)
    plt.close(fig)


def load_runs(B: Path):
    summary = json.loads((B / "report" / "summary.json").read_text())["runs"]
    runs = {}
    for r in summary:
        h = rib.read_history(B / "runs" / r["run_id"] / "optimization_history.csv")
        if h is None or "final_control" not in r:
            raise SystemExit(f"{r['run_id']}: missing history or final control")
        for label, row, z in (("first", h[0], r["initial_control"]), ("last", h[-1], r["final_control"])):
            if not np.allclose(row[3:], z, rtol=1e-8, atol=1e-8):
                raise SystemExit(f"{r['run_id']}: {label} history row differs from the recorded control")
        r = dict(r, history=h, perm=[p - 1 for p in r["final_match"]["permutation"]])
        runs[(str(r["case"]), r["data"])] = r
    return runs


def room(ax, sensors=None, labels=None, annotate=True):
    rsi.draw_room(ax, labels=False)
    if annotate:
        for name, (x0, x1, y0, y1) in geo.SOLIDS.items():
            if name.startswith("bed_"):
                ax.text(0.5 * (x0 + x1), 0.5 * (y0 + y1), f"Bed {name[-1]}", ha="center", va="center", fontsize=8,
                        color="#444444", zorder=4)
        ax.text(4.5, 6.1, "supply vents", ha="center", va="bottom", fontsize=8, color="#444444")
        ax.text(np.mean(geo.DOOR_X_RANGE), -0.1, "door outlet", ha="center", va="top", fontsize=8, color="#444444")
        ax.text(10.85, 0.3, "exhaust outlet", ha="right", va="center", fontsize=8, color="#444444")
    if sensors is not None:
        ax.scatter(sensors[:, 0], sensors[:, 1], marker="s", s=14, color=SENSOR, zorder=6, linewidths=0)
        for (x, y), lb in zip(sensors, labels):
            dx, dy, ha = SENSOR_OFFSET.get(lb, (4, 3, "left"))
            ax.annotate(lb.replace("S", "S$_{") + "}$", (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha,
                        fontsize=7.5, color=SENSOR, zorder=6)
    ax.set_xlabel("$x$ [m]")
    ax.set_ylabel("$y$ [m]")
    ax.set_xticks(range(0, 12))
    ax.set_yticks(range(0, 7))
    ax.tick_params(top=False, right=False)


def source_path(r, true_index):
    p = r["perm"][true_index]
    return r["history"][:, [3 + 3 * p, 4 + 3 * p]]


def center_errors(r_hist, z_true):
    return np.array([max(s["center_error_m"] for s in rib.match_sources(row[3:], z_true)["sources"]) for row in r_hist])


def fig_setup(out, runs, sensors, labels, radius):
    releases = {}
    for r in runs.values():
        for x, y, a in np.asarray(r["true_control"]).reshape(-1, 3):
            releases[(float(x), float(y))] = float(a)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    room(ax, sensors, labels)
    for k, ((x, y), a) in enumerate(sorted(releases.items()), start=1):
        ax.add_patch(Circle((x, y), radius, facecolor=NOISY, alpha=0.20, edgecolor=NOISY, lw=0.7, zorder=5))
        ax.scatter([x], [y], marker="*", s=70, color=TRUTH, edgecolors="white", linewidths=0.4, zorder=7)
        ax.annotate(f"R$_{k}$\n$A={a:g}$", (x, y), xytext=(0, -12), textcoords="offset points", ha="center",
                    va="top", fontsize=8, linespacing=1.0, zorder=7)
    h = [Line2D([], [], marker="s", ls="", color=SENSOR, ms=4, label="sensors"),
         Line2D([], [], marker="*", ls="", color=TRUTH, ms=8, label="release centres"),
         Line2D([], [], marker="o", ls="", mfc=NOISY, mec=NOISY, alpha=0.3, ms=9, label=r"radius $\sqrt{\rho}$")]
    fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, out, "problem_setup")


def zoom_window(pts):
    pts = np.vstack(pts)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    c = 0.5 * (lo + hi)
    return c, max(0.62 * float(np.max(hi - lo)), 0.03)


def draw_paths(ax, paths, scale=1.0):
    for kind, p in paths.items():
        col = KIND[kind]
        ax.plot(p[:, 0], p[:, 1], "-", color=col, lw=0.9, marker=".", ms=2.8, zorder=7)
        ax.scatter(*p[0], marker="o", s=38 * scale, facecolor="white", edgecolor=col, linewidths=1.0, zorder=8)
        ax.scatter(*p[-1], marker="X", s=42 * scale, color=col, edgecolor="white", linewidths=0.35, zorder=9)


def add_path_inset(az, truth, paths, mid, half):
    """Magnify the iterates when they span under a quarter of the zoom window (the final offset dominates it)."""
    pts = np.vstack(list(paths.values()))
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    extent = float(np.max(hi - lo))
    if extent >= 0.5 * half:
        return
    origin = mid - half
    occupied = np.vstack([pts, truth]) - origin
    occupied = occupied / (2 * half)
    corners = [(0.25, 0.25), (0.25, 0.75), (0.75, 0.25), (0.75, 0.75)]
    cx, cy = max(corners, key=lambda q: float(np.min(np.linalg.norm(occupied - np.array(q), axis=1))))
    ia = az.inset_axes([cx - 0.21, cy - 0.21, 0.42, 0.42], zorder=20)
    draw_paths(ia, paths, scale=0.75)
    c = 0.5 * (lo + hi)
    h = max(0.68 * extent, 5e-4)
    ia.set_xlim(c[0] - h, c[0] + h)
    ia.set_ylim(c[1] - h, c[1] + h)
    ia.set_aspect("equal")
    ia.set_xticks([])
    ia.set_yticks([])
    inside = (pts - (c - h)) / (2 * h)
    tx, ty = max([(0.04, 0.04), (0.04, 0.96), (0.96, 0.04), (0.96, 0.96)],
                 key=lambda q: float(np.min(np.linalg.norm((inside - np.array(q)) * [0.45, 1.0], axis=1))))
    ia.text(tx, ty, f"width {2 * h:.3f} m", transform=ia.transAxes, fontsize=7, color="#333333",
            ha="left" if tx < 0.5 else "right", va="bottom" if ty < 0.5 else "top")
    az.indicate_inset_zoom(ia, edgecolor="#555555", lw=0.6, alpha=0.9)


def fig_reconstruction(out, case, runs, sensors, labels):
    c, n = runs[(case, "clean")], runs[(case, "noisy")]
    zt = np.asarray(c["true_control"]).reshape(-1, 3)
    fig = plt.figure(figsize=(7.0, 7.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.28, 1.0], hspace=0.3, wspace=0.26)
    ax = fig.add_subplot(gs[0, :])
    room(ax, sensors, labels)
    ax.set_title("(a) Room, sensors and zoom windows", loc="left")
    for i in range(len(zt)):
        az = fig.add_subplot(gs[1, i])
        paths = {r["data"]: source_path(r, i) for r in (c, n)}
        mid, half = zoom_window([zt[i:i + 1, :2], *paths.values()])
        room(az, annotate=False)
        draw_paths(az, paths)
        az.scatter(*zt[i, :2], marker="*", s=105, color=TRUTH, edgecolor="white", linewidths=0.5, zorder=10)
        az.set_xlim(mid[0] - half, mid[0] + half)
        az.set_ylim(mid[1] - half, mid[1] + half)
        for axis in (az.xaxis, az.yaxis):
            axis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
            axis.set_major_formatter(ScalarFormatter(useOffset=False))
        add_path_inset(az, zt[i, :2], paths, mid, half)
        ec, en = (r["final_match"]["sources"][i]["center_error_m"] for r in (c, n))
        tag = "bc"[i]
        az.set_title(f"({tag}) Source at $({zt[i, 0]:.4g},\\,{zt[i, 1]:.4g})$, $A={zt[i, 2]:g}$\n"
                     f"centre error: {ec:.3f} m clean, {en:.3f} m noisy", loc="left", fontsize=8.5)
        ax.add_patch(Rectangle(mid - half, 2 * half, 2 * half, fill=False, ec="#222222", lw=0.8, zorder=8))
        ax.scatter(*zt[i, :2], marker="*", s=42, color=TRUTH, edgecolor="white", linewidths=0.3, zorder=9)
        ax.annotate(f"({tag})", mid, xytext=(0, -15), textcoords="offset points", ha="center", fontsize=8.5, zorder=9)
    h = [Line2D([], [], color=CLEAN, lw=1.1, marker=".", ms=4, label="path, clean data"),
         Line2D([], [], color=NOISY, lw=1.1, marker=".", ms=4, label="path, noisy data"),
         Line2D([], [], marker="o", ls="", mfc="white", mec="#333333", ms=6, label="initialization"),
         Line2D([], [], marker="X", ls="", color="#333333", ms=6, label="final estimate"),
         Line2D([], [], marker="*", ls="", color=TRUTH, ms=9, label="true centre"),
         Line2D([], [], marker="s", ls="", color=SENSOR, ms=4, label="sensor")]
    fig.legend(handles=h, loc="lower center", ncol=6, columnspacing=1.0, handletextpad=0.35,
               bbox_to_anchor=(0.5, 0.015))
    save(fig, out, f"reconstruction_case_{case}")


def fig_convergence(out, runs):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65))
    for (case, kind), r in runs.items():
        h = r["history"]
        st = dict(color=CASE_CLR[case], ls="-" if kind == "clean" else "--", lw=1.15)
        axes[0].semilogy(h[:, 0], h[:, 1] / r["objective_zero_source"], **st)
        axes[1].semilogy(h[:, 0], h[:, 2], **st)
        axes[2].semilogy(h[:, 0], center_errors(h, r["true_control"]), **st)
    for ax, yl, tag in zip(axes, (r"$J/J(0)$", r"$\|\nabla J\|_2$", "largest centre error [m]"), "abc"):
        ax.set_xlabel("L-BFGS-B iteration")
        ax.set_ylabel(yl)
        ax.set_title(f"({tag})", loc="left")
        ax.grid(alpha=0.25, lw=0.4, which="major")
    axes[2].set_ylim(0.018, 0.36)
    axes[2].yaxis.set_major_locator(FixedLocator([0.02, 0.05, 0.1, 0.2, 0.3]))
    axes[2].yaxis.set_major_formatter(FormatStrFormatter("%g"))
    axes[2].yaxis.set_minor_formatter(NullFormatter())
    h = [Line2D([], [], color=CASE_CLR[c], lw=1.4, label=f"case {c}") for c in sorted(CASE_CLR)]
    h += [Line2D([], [], color="#333333", lw=1.15, ls="-", label="clean"),
          Line2D([], [], color="#333333", lw=1.15, ls="--", label="noisy")]
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.legend(handles=h, loc="lower center", ncol=5)
    save(fig, out, "convergence_history")


def load_traces(B, out, settings, runs):
    dataset_dir = Path(settings["dataset_dir"])
    if not dataset_dir.is_absolute() and not dataset_dir.exists():
        # shipped copies record repo-relative paths
        dataset_dir = Path(__file__).resolve().parents[2] / dataset_dir
    traces = {}
    for (case, kind), r in runs.items():
        cache = out / "data" / f"traces_{r['run_id']}.npz"
        zi, zf = np.asarray(r["initial_control"], float), np.asarray(r["final_control"], float)
        if cache.exists():
            with np.load(cache) as d:
                if np.array_equal(d["z_initial"], zi) and np.array_equal(d["z_final"], zf):
                    traces[(case, kind)] = {k: d[k] for k in ("times", "data", "initial", "final")}
                    continue
        ds = ri.load_dataset(dataset_dir, case, kind)
        problem = ri.build_problem(ds, log=lambda m: None)
        gt, nid = problem["app"].gt, problem["observation"].node_ids
        tr = {"times": ds["times"], "data": ds["observations"],
              "initial": gt.run_forward(problem["mesh_data"], problem["pde"], zi)[:, nid],
              "final": gt.run_forward(problem["mesh_data"], problem["pde"], zf)[:, nid]}
        np.savez_compressed(cache, z_initial=zi, z_final=zf, **tr)
        traces[(case, kind)] = tr
        print(f"traces solved: {r['run_id']}", flush=True)
    return traces


def fig_traces(out, case, traces, labels):
    clean = traces[(case, "clean")]
    order = np.argsort(-np.sqrt(np.mean(clean["data"][1:] ** 2, axis=0)))[:4]
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 3.6), sharex=True, sharey="col")
    s = slice(None, None, TRACE_STRIDE)
    for row, kind in enumerate(("clean", "noisy")):
        tr = traces[(case, kind)]
        t = tr["times"][s]
        for col, j in enumerate(order):
            ax = axes[row, col]
            ax.plot(t, tr["data"][s, j], color="#9A9A9A", lw=1.6 if kind == "clean" else 0.45, rasterized=kind == "noisy")
            ax.plot(t, tr["initial"][s, j], color="#E69F00", lw=0.9, ls="--")
            ax.plot(t, tr["final"][s, j], color=CLEAN, lw=0.9)
            if row == 0:
                ax.set_title(f"S$_{{{labels[j][1:]}}}$")
            else:
                ax.set_xlabel("$t$ [s]")
            ax.set_xticks([0, 100, 200, 300])
        axes[row, 0].set_ylabel(f"{kind} data\n$u$ [tracer units]")
    h = [Line2D([], [], color="#9A9A9A", lw=1.6, label="observations (fine mesh)"),
         Line2D([], [], color="#E69F00", lw=1.0, ls="--", label="initialization (coarse mesh)"),
         Line2D([], [], color=CLEAN, lw=1.0, label="final estimate (coarse mesh)")]
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.legend(handles=h, loc="lower center", ncol=3)
    save(fig, out, f"traces_case_{case}")


def write_paths(out, runs):
    for r in runs.values():
        with open(out / "data" / f"source_paths_{r['run_id']}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["iteration", "true_source", "x", "y", "amplitude"])
            for row in r["history"]:
                for i, p in enumerate(r["perm"]):
                    w.writerow([int(row[0]), i + 1, row[3 + 3 * p], row[4 + 3 * p], row[5 + 3 * p]])


def write_notes(out, cases):
    L = ["# Application2 inversion paper figures", "",
         "Each figure is saved as a vector PDF and a 300 dpi PNG. Suggested captions:", "",
         "- `problem_setup`: room geometry, supply vents, door and exhaust outlets, eight wall-mounted sensors, and the "
         "three release locations with amplitudes; shaded discs have radius sqrt(rho) = 0.20 m.",
         "- `reconstruction_case_<c>`: recovery of the two sources. Panel (a) locates the zoom windows; the zooms show the "
         "sensitivity initialization, the L-BFGS-B iterates, the final estimate and the true centre for clean and noisy "
         "data. Zoom scales differ between panels; an inset magnifies the iterates when they are short compared with "
         "the distance to the true centre.",
         "- `convergence_history`: objective relative to the zero-source objective, unscaled gradient norm and largest "
         "permutation-matched centre error at each accepted L-BFGS-B iterate (iteration 0 = initialization).",
         "- `traces_case_<c>`: the four sensors with the largest clean-data RMS. Model traces are post-hoc coarse-mesh "
         f"forward solves (not counted in the optimization solve totals); every {TRACE_STRIDE}th time sample is plotted.",
         "", "Cases: " + ", ".join(cases) + ". True controls are used only for scoring and plotting, after the runs.", ""]
    (out / "FIGURES.md").write_text("\n".join(L))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch-dir", type=Path, required=True)
    ap.add_argument("--dataset-dir", type=Path, default=None,
                    help="dataset folder (default: the one recorded in <batch>/resolved_settings.json)")
    a = ap.parse_args(argv)
    B = a.batch_dir.resolve()
    settings = json.loads((B / "resolved_settings.json").read_text())
    if a.dataset_dir is not None:
        settings["dataset_dir"] = str(a.dataset_dir.resolve())
    out = B / "paper_figures"
    (out / "data").mkdir(parents=True, exist_ok=True)
    set_style()
    runs = load_runs(B)
    rs = settings["runs"][0]
    sensors, labels = np.asarray(rs["sensor_coordinates"], float), rs["sensor_labels"]
    radius = float(np.sqrt(rs["transport"]["sigma2"]))
    cases = sorted({c for c, _ in runs})
    write_paths(out, runs)
    fig_setup(out, runs, sensors, labels, radius)
    for case in cases:
        fig_reconstruction(out, case, runs, sensors, labels)
    fig_convergence(out, runs)
    traces = load_traces(B, out, settings, runs)
    for case in cases:
        fig_traces(out, case, traces, labels)
    write_notes(out, cases)
    print(f"figures: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
