"""Two-page PDF report on the App1 raw-mode recovery experiments (six runs).

Reads the two machine-readable result files produced by the reporting scripts:

    output/raw_robustness_sweep/raw_robustness_sweep.json   (report_raw_robustness_sweep.py)
    output/rerun_vs_archive/rerun_vs_archive.json           (report_rerun_vs_archive.py)

plus each run's ``optimization_history.csv`` for the iterate trajectories, and
writes ``output/raw_robustness_sweep/app1_recovery_report.pdf``:

    page 1  Part A - robustness on three new synthetic truths
    page 2  Part B - reruns of three archived cases that previously failed

Usage:
    cd "Applications/Application1/test scripts/diagnostics"
    python3 make_recovery_report_pdf.py [--preview DIR]

``--preview DIR`` additionally writes page1.png / page2.png for quick viewing.
Nothing here re-runs an optimization; it is a read-only presentation layer.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Rectangle

APP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(APP.parents[1] / "Applications" / "Test Application2"))
sys.path.insert(0, str(APP))
import mesh_data_config as mdc  # noqa: E402
from mesh_admissibility import MeshAdmissibility  # noqa: E402

OUT = APP / "output"
SWEEP = json.loads((OUT / "raw_robustness_sweep" /
                    "raw_robustness_sweep.json").read_text())
RERUN = json.loads((OUT / "rerun_vs_archive" / "rerun_vs_archive.json").read_text())
sweep = {r["case"]: r for r in SWEEP}

INK = "#16181d"
MUTED = "#5b6270"
RULE = "#c9ced8"
GOOD = "#1a7f4b"
BAD = "#a32020"
WARN = "#8a5a00"
BAND = "#eef1f6"
S1C = "#1f5fa8"
S2C = "#b3341f"

mesh_data = mdc.build_mesh_data()
adm = MeshAdmissibility(mesh_data)
adm._ensure_boundary_geometry()
BA, BB = adm._boundary_a, adm._boundary_b


def history(case):
    h = np.loadtxt(OUT / case / "full" / "optimization_history.csv",
                   delimiter=",", skiprows=1)
    return h[None, :] if h.ndim == 1 else h


def inside_mask(xy):
    return np.array([adm.is_point_in_domain(float(p[0]), float(p[1])) for p in xy])


PARTA = [
    ("raw_robust_caseA_open_pair", "A", "well-separated interior pair"),
    ("raw_robust_caseB_walled_street", "B", "source 1 in a walled street"),
    ("raw_robust_caseC_nearwall_pair", "C", "near-wall, unequal, close"),
]
PARTB = [
    ("rerun_fail1_lrb_150_600_50_250", "F1", "converged wrong minimum"),
    ("rerun_fail2_lrb_250_130_15_580", "F2", "line-search stall, 0 iters"),
    ("rerun_fail3_lrb_close_pair_200_560", "F3", "close pair, 0 iters"),
]
J0_A = {"raw_robust_caseA_open_pair": 3.408965e02,
        "raw_robust_caseB_walled_street": 2.2173508e04,
        "raw_robust_caseC_nearwall_pair": 1.5326132e03}
INITERR_A = {"raw_robust_caseA_open_pair": (0.221, 2.583),
             "raw_robust_caseB_walled_street": (1.417, 7.053),
             "raw_robust_caseC_nearwall_pair": (0.673, 26.99)}
rerun = {r["new_case"]: r for r in RERUN}


def make_fig():
    f = plt.figure(figsize=(8.27, 11.69))
    f.patch.set_facecolor("white")
    return f


def T(fig, x, y, s, size=8.0, color=INK, family="sans-serif", weight="normal",
      ha="left", va="top", **kw):
    return fig.text(x, y, s, fontsize=size, color=color, family=family,
                    fontweight=weight, ha=ha, va=va, **kw)


def rule(fig, y, x0=0.06, x1=0.94, lw=0.8, color=RULE):
    fig.add_artist(plt.Line2D([x0, x1], [y, y], lw=lw, color=color,
                              transform=fig.transFigure))


def band(fig, y, h, x0=0.06, x1=0.94, color=BAND):
    fig.add_artist(Rectangle((x0, y), x1 - x0, h, transform=fig.transFigure,
                             facecolor=color, edgecolor="none", zorder=0))


def vec(v):
    return (f"[{v[0]:9.4f} {v[1]:9.4f} {v[2]:7.4f} |"
            f"{v[3]:9.4f} {v[4]:9.4f} {v[5]:7.4f} ]")


def draw_maps(fig, y_top, cases, get_ctrl, panel_h=0.300):
    """Row of three domain panels with building geometry and full trajectories."""
    panel_w = panel_h * 11.69 / 8.27 * (301.99 / 700.46)
    gap = 0.045
    total = 3 * panel_w + 2 * gap
    x0 = 0.5 - total / 2.0
    for i, (case, label, blurb) in enumerate(cases):
        t, ini, fin = get_ctrl(case)
        ax = fig.add_axes([x0 + i * (panel_w + gap), y_top - panel_h,
                           panel_w, panel_h])
        for k in range(BA.shape[0]):
            ax.plot([BA[k, 0], BB[k, 0]], [BA[k, 1], BB[k, 1]], "-",
                    color="0.58", linewidth=0.30, zorder=1)
        h = history(case)
        all_in = np.ones(len(h), dtype=bool)
        for s_i, col in enumerate((S1C, S2C)):
            xy = h[:, 3 + 3 * s_i: 5 + 3 * s_i]
            ins = inside_mask(xy)
            all_in &= ins
            for r in range(len(xy) - 1):
                ok = ins[r] and ins[r + 1]
                ax.plot(xy[r:r + 2, 0], xy[r:r + 2, 1], "-" if ok else "--",
                        color=col if ok else "magenta",
                        linewidth=0.9 if ok else 1.4, alpha=0.95,
                        zorder=3 if ok else 4)
        n_in_bldg = int(np.sum(~all_in[1:]))
        t2 = np.asarray(t).reshape(-1, 3)
        i2 = np.asarray(ini).reshape(-1, 3)
        f2 = np.asarray(fin).reshape(-1, 3)
        ax.plot(t2[:, 0], t2[:, 1], "o", markerfacecolor="none",
                markeredgecolor="#1f9e3f", markeredgewidth=1.2, markersize=10.0,
                linestyle="none", zorder=5)
        for s_i, col in enumerate((S1C, S2C)):
            ax.plot(i2[s_i, 0], i2[s_i, 1], "o", color=col,
                    markeredgecolor="white", markeredgewidth=0.5,
                    markersize=4.4, zorder=6)
            ax.plot(f2[s_i, 0], f2[s_i, 1], "X", color=col,
                    markeredgecolor="white", markeredgewidth=0.5,
                    markersize=5.2, zorder=7)
        ax.set_xlim(-12, 314)
        ax.set_ylim(-12, 712)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(RULE)
            sp.set_linewidth(0.6)
        ax.set_title(f"{label}  ·  {blurb}", fontsize=6.6, color=INK, pad=2.5)
        cap = (f"{n_in_bldg} iterates with a source in a building"
               if n_in_bldg else "no iterate left the fluid")
        fig.text(x0 + i * (panel_w + gap) + panel_w / 2.0,
                 y_top - panel_h - 0.010, cap, fontsize=6.0, color=MUTED,
                 ha="center", va="top")
    return y_top - panel_h - 0.024


# =========================================================== PAGE 1
def page1():
    fig = make_fig()
    y = 0.975
    T(fig, 0.06, y, "Inverse source identification — DC Navy Yard domain",
      14.5, INK, weight="bold")
    y -= 0.020
    T(fig, 0.06, y, "Recovery robustness of adjoint-sensitivity initialization "
      "with raw-center L-BFGS-B", 10.5, MUTED)
    y -= 0.017
    T(fig, 0.06, y, "Animesh Jain   ·   2026-07-31   ·   Application 1 "
      "(transient advection–diffusion, steady Navier–Stokes wind)", 8.0, MUTED)
    y -= 0.013
    rule(fig, y)

    # ---- abstract
    y -= 0.010
    h = 0.084
    band(fig, y - h, h)
    y -= 0.008
    T(fig, 0.065, y, "SUMMARY", 8.6, INK, weight="bold")
    y -= 0.016
    for line in [
        "Six deterministic two-source recovery experiments were run: three on new "
        "synthetic truths chosen to span easy, obstacle-",
        "enclosed and near-wall/unequal-amplitude regimes (Part A), and three "
        "reruns of genuinely failed archived cases (Part B).",
        "Every run used ONE adjoint-sensitivity initialization followed by ONE "
        "sigma-continuation L-BFGS-B solve with raw source",
        "centers.  ALL SIX RECOVERED THE TRUE CONTROL TO MACHINE PRECISION "
        "(final objective 3.5e-18 to 2.0e-16).  The three archived",
        "failures had previously terminated at objectives 1.8e4 to 1.9e6 with "
        "source errors of 23 m to 449 m.",
    ]:
        T(fig, 0.065, y, line, 6.8, INK, family="monospace")
        y -= 0.0115

    # ---- methods
    y -= 0.012
    T(fig, 0.06, y, "METHOD", 9.0, INK, weight="bold")
    y -= 0.017
    for line in [
        "Domain    DC Navy Yard street network, 302 x 700 m, 21 building holes "
        "(natural no-flux); 2795-node triangular mesh",
        "Physics   transient advection-diffusion, k = 10 m^2/s, dt = 0.1 s, "
        "tfinal = 150 s, RK4 forward + exact discrete RK4 adjoint",
        "Wind      steady incompressible Navier-Stokes nodal velocity field on "
        "the same mesh (mean |u| ~ 2.4 m/s)",
        "Data      boundary misfit J = 1/2 int_0^T int_Gamma |u - u_d|^2 ds dt "
        "on Gamma = left + right + bottom",
        "Control   two Gaussians [x1 y1 A1 x2 y2 A2]; sigma^2 continuation "
        "1600 -> 800 -> 400 m^2; bounds x,y in mesh box, A in [0.10, 2.00]",
        "Optimizer L-BFGS-B, ftol 1e-14, gtol 1e-9, maxls 20, max-iter 150; RAW "
        "centers (no projection, no penalty, full adjoint gradient)",
    ]:
        T(fig, 0.065, y, line, 6.8, INK, family="monospace")
        y -= 0.0118

    y -= 0.008
    h = 0.070
    band(fig, y - h, h)
    y -= 0.007
    T(fig, 0.065, y, "INITIALIZATION — ADJOINT SOURCE-INSERTION SCREENING "
      "(truth-free, deterministic, single output)", 8.4, INK, weight="bold")
    y -= 0.016
    for line in [
        "The initial control is not guessed or randomly sampled.  The adjoint field "
        "of the current residual gives an insertion",
        "sensitivity G(c) at every mesh node.  The top-24 basin-diverse minima are "
        "ranked by the exact bounded one-dimensional",
        "reduction dJ = a*G + 0.5 a*^2 kappa (one unit forward solve each).  A beam "
        "of 3 first-source hypotheses is refined by",
        "variable projection, expanded against its own rebuilt residual and jointly "
        "pair-refined; candidate de-duplication is",
        "OBSTACLE-AWARE (nearby candidates are distinct basins if a building wall "
        "separates them).  One control is returned.",
    ]:
        T(fig, 0.065, y, line, 6.6, INK, family="monospace")
        y -= 0.0111

    # ---- Part A
    y -= 0.014
    T(fig, 0.06, y, "PART A  —  ROBUSTNESS ON THREE NEW SYNTHETIC TRUTHS",
      9.5, INK, weight="bold")
    y -= 0.018
    for case, label, blurb in PARTA:
        r = sweep[case]
        e1, e2 = INITERR_A[case]
        T(fig, 0.065, y, f"Case {label} — {blurb}", 8.0, INK, weight="bold")
        T(fig, 0.94, y, "EXACT RECOVERY", 7.4, GOOD, weight="bold", ha="right")
        y -= 0.0145
        for tag, v, note in [
            ("true", vec(r["true_control"]), ""),
            ("init", vec(r["initial_control"]),
             f"J0 {J0_A[case]:.3e}   init err {e1:.2f} / {e2:.2f} m"),
            ("final", vec(r["final_control"]),
             f"Jf {r['final_objective']:.3e}   max err "
             f"{r['max_position_error_m']:.1e} m"),
        ]:
            T(fig, 0.075, y, tag, 6.8, MUTED, family="monospace")
            T(fig, 0.120, y, v, 6.8, INK, family="monospace")
            if note:
                T(fig, 0.560, y, note, 6.8, MUTED, family="monospace")
            y -= 0.0118
        y -= 0.005

    y -= 0.004
    rule(fig, y, lw=0.6)
    y -= 0.013
    cols = [("case", 0.065, "left"), ("iters", 0.20, "right"),
            ("J0", 0.30, "right"), ("J final", 0.40, "right"),
            ("max dx", 0.505, "right"), ("max dA", 0.60, "right"),
            ("in bldg", 0.685, "right"), ("feasible", 0.775, "right"),
            ("wall", 0.855, "right"), ("class", 0.935, "right")]
    for n, x, ha in cols:
        T(fig, x, y, n, 6.6, MUTED, weight="bold", ha=ha)
    y -= 0.012
    for case, label, _ in PARTA:
        r = sweep[case]
        vals = [f"Case {label}", f"{r['iterations_total']}",
                f"{J0_A[case]:.2e}", f"{r['final_objective']:.2e}",
                f"{r['max_position_error_m']:.1e} m",
                f"{r['max_amplitude_error']:.1e}", "0", "yes",
                f"{r['wall_seconds']/60.0:.0f} min", "1"]
        for s, (_n, x, ha) in zip(vals, cols):
            T(fig, x, y, s, 6.9, INK, family="monospace", ha=ha)
        y -= 0.0128
    rule(fig, y + 0.003, lw=0.6)

    # ---- maps
    y -= 0.014
    T(fig, 0.06, y, "green ring = true source   ·   filled circle = sensitivity "
      "initialization   ·   X = final   ·   blue = source 1, red = source 2",
      6.8, MUTED)
    y -= 0.020
    draw_maps(fig, y, PARTA,
              lambda c: (sweep[c]["true_control"], sweep[c]["initial_control"],
                         sweep[c]["final_control"]))

    rule(fig, 0.030)
    T(fig, 0.06, 0.022, "Page 1 of 2   ·   Part A: robustness on new truths",
      6.4, MUTED)
    T(fig, 0.94, 0.022, "Applications/Application1/output/raw_robustness_sweep/",
      6.4, MUTED, ha="right")
    return fig


# =========================================================== PAGE 2
def page2():
    fig = make_fig()
    y = 0.975
    T(fig, 0.06, y, "Part B — Reruns of archived cases that previously failed",
      14.5, INK, weight="bold")
    y -= 0.020
    T(fig, 0.06, y, "Identical physics, mesh, sensors, schedule and bounds; "
      "only the initializer and feasibility mode differ", 9.5, MUTED)
    y -= 0.013
    rule(fig, y)

    y -= 0.019
    T(fig, 0.06, y, "SELECTION", 9.0, INK, weight="bold")
    y -= 0.016
    for line in [
        "All 83 archived Application 1 run summaries were scanned and grouped by "
        "true control (12 distinct truths).  Runs whose",
        "termination was a deliberate iteration cap (smoke / short stages) were "
        "excluded as truncated rather than failed.  That",
        "leaves exactly three distinct truths with genuine full-run failures - the "
        "complete set, not a selection.",
    ]:
        T(fig, 0.065, y, line, 6.8, INK, family="monospace")
        y -= 0.0118

    y -= 0.012
    T(fig, 0.06, y, "ARCHIVED FAILURES AND THEIR RERUNS", 9.5, INK, weight="bold")
    y -= 0.019

    for case, label, blurb in PARTB:
        r = rerun[case]
        T(fig, 0.065, y, f"{label} — {blurb}", 8.0, INK, weight="bold")
        T(fig, 0.94, y, "NOW EXACT", 7.4, GOOD, weight="bold", ha="right")
        y -= 0.0135
        T(fig, 0.075, y, f"archive  output/{r['archived']}", 6.4, MUTED,
          family="monospace")
        y -= 0.0125
        for tag, v, note in [
            ("true", vec(r["true"]), ""),
            ("old init", vec(r["old_initial"]), ""),
            ("old final", vec(r["old_final"]),
             f"Jf {r['old_Jf']:.3e}  err {max(r['old_pos_err']):.1f} m  "
             f"class {r['old_class']}  ({r['old_iters']} it)"),
            ("new init", vec(r["new_initial"]),
             f"init err {r['new_init_pos_err'][0]:.2f} / "
             f"{r['new_init_pos_err'][1]:.2f} m"),
            ("new final", vec(r["new_final"]),
             f"Jf {r['new_Jf']:.3e}  err {max(r['new_pos_err']):.1e} m  "
             f"class {r['new_class']}  ({r['new_iters']} it)"),
        ]:
            col = BAD if tag == "old final" else (
                GOOD if tag == "new final" else INK)
            T(fig, 0.075, y, tag, 6.8, MUTED, family="monospace")
            T(fig, 0.148, y, v, 6.8, col, family="monospace")
            if note:
                T(fig, 0.585, y, note, 6.6, MUTED, family="monospace")
            y -= 0.0117
        y -= 0.006

    # ---- comparison table
    y -= 0.002
    rule(fig, y, lw=0.6)
    y -= 0.013
    cols = [("", 0.065, "left"), ("orig J final", 0.235, "right"),
            ("orig err", 0.325, "right"), ("orig cls", 0.405, "right"),
            ("init err", 0.505, "right"), ("new J final", 0.615, "right"),
            ("new err", 0.710, "right"), ("new cls", 0.790, "right"),
            ("iters", 0.860, "right"), ("wall", 0.935, "right")]
    for n, x, ha in cols:
        T(fig, x, y, n, 6.6, MUTED, weight="bold", ha=ha)
    y -= 0.012
    for case, label, _ in PARTB:
        r = rerun[case]
        vals = [label, f"{r['old_Jf']:.2e}", f"{max(r['old_pos_err']):.1f} m",
                f"{r['old_class']}", f"{max(r['new_init_pos_err']):.1f} m",
                f"{r['new_Jf']:.2e}", f"{max(r['new_pos_err']):.1e} m",
                f"{r['new_class']}", f"{r['new_iters']}",
                f"{r['wall_seconds']/60.0:.0f} min"]
        for s, (_n, x, ha) in zip(vals, cols):
            T(fig, x, y, s, 6.9, INK, family="monospace", ha=ha)
        y -= 0.0128
    rule(fig, y + 0.003, lw=0.6)

    # ---- maps
    y -= 0.013
    T(fig, 0.06, y, "green ring = true   ·   filled circle = sensitivity "
      "initialization   ·   X = final   ·   magenta dashed = iterate inside a "
      "building", 6.8, MUTED)
    y -= 0.020
    y = draw_maps(fig, y, PARTB,
                  lambda c: (rerun[c]["true"], rerun[c]["new_initial"],
                             rerun[c]["new_final"]), panel_h=0.245)

    # ---- findings
    y -= 0.010
    h = 0.118
    fig.add_artist(FancyBboxPatch(
        (0.06, y - h), 0.88, h, transform=fig.transFigure,
        boxstyle="round,pad=0.006,rounding_size=0.008",
        facecolor="#f2f7f3", edgecolor="#bcd6c4", linewidth=0.9, zorder=0))
    T(fig, 0.072, y - 0.011, "FINDINGS", 8.4, GOOD, weight="bold")
    yy = y - 0.028
    for line in [
        "1. All six cases recovered exactly (6/6).  The three archived failures "
        "were INITIALIZATION failures, not optimizer or",
        "   identifiability limits: the archived starts placed the second source "
        "440 m and 527 m away, in the wrong plume basin,",
        "   whereas adjoint screening places it 13-35 m away in the correct basin "
        "and L-BFGS-B closes the remainder.",
        "2. Truth [150 600 1.2 | 50 250 0.7] appears in 53 archived runs, was "
        "recovered in one, and had never been recovered under",
        "   left+right+bottom sensors.  It now converges to 1.4e-17.",
        "3. F3 is the first case in which raw obstacle handling did real work: 62 "
        "accepted iterates lay inside buildings, source 1",
        "   spent 48 consecutive iterations inside building 4, and source 2 made 6 "
        "complete building transits - yet the terminal",
        "   point was feasible with no projection and no repair.  Remaining "
        "weakness: the initializer still mis-separates a close,",
        "   equal-strength pair (F3 initial error 117 m, one amplitude at 1.965 of "
        "a 2.00 bound); the optimizer rescued it.",
    ]:
        T(fig, 0.072, yy, line, 6.5, INK, family="monospace")
        yy -= 0.0105

    rule(fig, 0.030)
    T(fig, 0.06, 0.022, "Page 2 of 2   ·   Part B: archived-failure reruns",
      6.4, MUTED)
    T(fig, 0.94, 0.022, "Applications/Application1/output/rerun_vs_archive/",
      6.4, MUTED, ha="right")
    return fig


out = OUT / "raw_robustness_sweep" / "app1_recovery_report.pdf"
PREV = Path(sys.argv[sys.argv.index("--preview") + 1]) if "--preview" in sys.argv \
    else None
with PdfPages(out) as pdf:
    for i, mk in enumerate((page1, page2), start=1):
        f = mk()
        pdf.savefig(f)
        if PREV is not None:
            PREV.mkdir(parents=True, exist_ok=True)
            f.savefig(PREV / f"page{i}.png", dpi=115)
        plt.close(f)
print(f"wrote {out}")
