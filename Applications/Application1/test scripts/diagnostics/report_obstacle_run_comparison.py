"""Read-only comparison report + trajectory plots for App1 optimization runs.

Usage:
  python3 report_runs.py <case_name> [<case_name> ...]

For each case reads output/<case>/full/{optimization_summary.json,
optimization_history.csv} and reports final control, objective, termination,
per-source errors after optimal permutation, projection accounting, wall-pinning
detection, amplitude saturation, and final feasibility.  Also writes a combined
trajectory plot with the building geometry overlaid.
"""

from __future__ import annotations

import json
import sys
from itertools import permutations
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parents[2]
# Src/Objective (gradient_test) on the path; APP inserted last so it is FIRST
# and Application1's mesh_data_config / multi_source_utils win.
sys.path.insert(0, str(APP.parents[1] / "Src" / "Objective"))
sys.path.insert(0, str(APP))

import mesh_data_config as mdc  # noqa: E402
from mesh_admissibility import MeshAdmissibility  # noqa: E402

PIN_TOL = 1.0e-6      # metres: "nearly unchanged" position
AMP_UPPER = 2.0
AMP_TOL = 1.0e-9


def load(case):
    d = APP / "output" / case / "full"
    summary = json.loads((d / "optimization_summary.json").read_text())
    hist = np.loadtxt(d / "optimization_history.csv", delimiter=",", skiprows=1)
    if hist.ndim == 1:
        hist = hist[None, :]
    return summary, hist, d


def match_sources(final, true):
    f = np.asarray(final).reshape(-1, 3)
    t = np.asarray(true).reshape(-1, 3)
    best, best_cost = None, None
    for perm in permutations(range(len(t))):
        cost = sum(float(np.hypot(*(f[perm[i], :2] - t[i, :2]))) for i in range(len(t)))
        if best_cost is None or cost < best_cost:
            best, best_cost = perm, cost
    return best


def pin_runs(xy, tol=PIN_TOL):
    """Longest run of consecutive iterations with position change < tol.

    Returns (max_run_length, start_index, end_index) in history-row indices.
    """
    if len(xy) < 2:
        return 0, 0, 0
    step = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    best = (0, 0, 0)
    run, start = 0, 0
    for i, s in enumerate(step):
        if s < tol:
            if run == 0:
                start = i
            run += 1
            if run > best[0]:
                best = (run, start, i + 1)
        else:
            run = 0
    return best


def report(case, admiss):
    summary, hist, d = load(case)
    true = np.asarray(summary["true_control"], dtype=np.float64)
    final = np.asarray(summary["final_control"], dtype=np.float64)
    init = np.asarray(summary["initial_control"], dtype=np.float64)
    perm = match_sources(final, true)
    f = final.reshape(-1, 3)
    t = true.reshape(-1, 3)

    print("\n" + "=" * 78)
    print(f"CASE  {case}")
    print("=" * 78)
    print(f"  feasibility mode      : {(summary.get('feasibility') or {}).get('feasibility_mode')}")
    print(f"  measurement boundaries: {summary.get('measurement_boundaries')}")
    print(f"  sigma2 continuation   : {summary.get('sigma2_continuation')}")
    print(f"  initial control       : {np.array2string(init, precision=6)}")
    print(f"  FINAL control         : {np.array2string(final, precision=8)}")
    print(f"  initial objective     : {summary['initial_objective']:.10e}")
    print(f"  FINAL objective       : {summary['final_objective']:.10e}")
    print(f"  final gradient norm   : {summary['final_gradient_norm']:.6e}")
    print(f"  iterations            : {summary['iterations']}")
    print(f"  TERMINATION           : {summary['message']}")
    print(f"  success flag          : {summary.get('success')}")

    print("  per-source errors (optimal permutation):")
    for i in range(len(t)):
        s = f[perm[i]]
        dpos = float(np.hypot(*(s[:2] - t[i, :2])))
        damp = float(abs(s[2] - t[i, 2]))
        inside = admiss.is_point_in_domain(float(s[0]), float(s[1]))
        clear = admiss.distance_to_boundary(float(s[0]), float(s[1]))
        outside_d = 0.0
        if not inside:
            outside_d = float(np.sqrt(admiss.nearest_domain_distance_sq(
                float(s[0]), float(s[1]))[0]))
        print(f"    true s{i+1} ({t[i,0]:.1f},{t[i,1]:.1f}) A={t[i,2]:.4f}"
              f"  <-  ({s[0]:.4f},{s[1]:.4f}) A={s[2]:.6f}"
              f"   dpos={dpos:9.4f} m  dA={damp:.6f}"
              f"   in_fluid={inside}  wall_clearance={clear:.4f} m"
              + (f"  OUTSIDE by {outside_d:.4f} m" if not inside else ""))

    feas = summary.get("feasibility") or {}
    print(f"  projection accounting : invalid evals={feas.get('invalid_evaluations')}"
          f"  projected evals={feas.get('projected_evaluations')}"
          f"  penalized evals={feas.get('penalized_evaluations')}"
          f"  projected source events={feas.get('projected_source_events')}"
          f"  max projection distance={feas.get('max_projection_distance')}")

    n_sources = len(t)
    print("  wall-pinning / saturation over the combined history "
          f"({hist.shape[0]} recorded iterates):")
    for s_i in range(n_sources):
        xy = hist[:, 3 + 3 * s_i: 5 + 3 * s_i]
        amp = hist[:, 5 + 3 * s_i]
        run, i0, i1 = pin_runs(xy)
        sat = int(np.sum(amp >= AMP_UPPER - AMP_TOL))
        print(f"    source {s_i+1}: longest frozen-position run = {run} consecutive "
              f"iterations (history rows {i0}..{i1}), "
              f"position there ({xy[i0,0]:.4f},{xy[i0,1]:.4f})")
        clear0 = admiss.distance_to_boundary(float(xy[i0, 0]), float(xy[i0, 1]))
        print(f"              wall clearance at that position = {clear0:.6f} m; "
              f"amplitude at upper bound {AMP_UPPER} in {sat} iterates "
              f"(final A={amp[-1]:.6f})")
        if run >= 3:
            lo = max(0, i0 - 3)
            hi = min(hist.shape[0], i0 + 4)
            print(f"              onset window rows {lo}..{hi-1}:")
            for r in range(lo, hi):
                mark = " <-- freeze starts" if r == i0 else ""
                print(f"                it={int(hist[r,0]):4d} J={hist[r,1]:.6e} "
                      f"|g|={hist[r,2]:.4e} "
                      f"s{s_i+1}=({hist[r,3+3*s_i]:.5f},{hist[r,4+3*s_i]:.5f}) "
                      f"A={hist[r,5+3*s_i]:.5f}{mark}")

    all_in = all(
        admiss.is_point_in_domain(float(f[i, 0]), float(f[i, 1]))
        for i in range(n_sources)
    )
    print(f"  FINAL FEASIBILITY     : {'FEASIBLE' if all_in else 'INFEASIBLE'} "
          f"(all source centers in the fluid mesh: {all_in})")
    return summary, hist, perm


def plot(cases, admiss, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    admiss._ensure_boundary_geometry()
    a, b = admiss._boundary_a, admiss._boundary_b

    fig, axes = plt.subplots(1, len(cases), figsize=(7.0 * len(cases), 10.0),
                             squeeze=False)
    for ax, case in zip(axes[0], cases):
        summary, hist, _ = load(case)
        true = np.asarray(summary["true_control"]).reshape(-1, 3)
        for k in range(a.shape[0]):
            ax.plot([a[k, 0], b[k, 0]], [a[k, 1], b[k, 1]], "-",
                    color="0.35", linewidth=0.7, zorder=1)
        colors = ["tab:blue", "tab:red"]
        for s_i in range(true.shape[0]):
            xy = hist[:, 3 + 3 * s_i: 5 + 3 * s_i]
            ax.plot(xy[:, 0], xy[:, 1], "-", color=colors[s_i], linewidth=1.4,
                    alpha=0.85, zorder=3, label=f"source {s_i+1} path")
            ax.plot(xy[0, 0], xy[0, 1], "o", color=colors[s_i],
                    markeredgecolor="black", markersize=10, zorder=4,
                    label=f"source {s_i+1} init")
            ax.plot(xy[-1, 0], xy[-1, 1], "X", color=colors[s_i],
                    markeredgecolor="black", markersize=14, zorder=5,
                    label=f"source {s_i+1} final")
            inside = np.array([
                admiss.is_point_in_domain(float(p[0]), float(p[1])) for p in xy
            ])
            if np.any(~inside):
                ax.plot(xy[~inside, 0], xy[~inside, 1], "v", color="magenta",
                        markeredgecolor="black", markersize=7, linestyle="none",
                        zorder=6,
                        label="iterate INSIDE a building" if s_i == 0 else None)
        ax.plot(true[:, 0], true[:, 1], "P", color="lime", markeredgecolor="black",
                markersize=17, linestyle="none", zorder=7, label="TRUE source")
        mode = (summary.get("feasibility") or {}).get("feasibility_mode")
        ax.set_title(f"{case}\nfeasibility={mode}  "
                     f"Jf={summary['final_objective']:.3e}", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m, local)")
        ax.set_ylabel("y (m, local)")
        ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nwrote {out_path}")

    # zoomed panel on the contested region
    fig2, axes2 = plt.subplots(1, len(cases), figsize=(6.5 * len(cases), 6.0),
                               squeeze=False)
    for ax, case in zip(axes2[0], cases):
        summary, hist, _ = load(case)
        true = np.asarray(summary["true_control"]).reshape(-1, 3)
        for k in range(a.shape[0]):
            ax.plot([a[k, 0], b[k, 0]], [a[k, 1], b[k, 1]], "-",
                    color="0.35", linewidth=0.9, zorder=1)
        colors = ["tab:blue", "tab:red"]
        for s_i in range(true.shape[0]):
            xy = hist[:, 3 + 3 * s_i: 5 + 3 * s_i]
            ax.plot(xy[:, 0], xy[:, 1], ".-", color=colors[s_i], linewidth=1.2,
                    markersize=3, zorder=3)
            ax.plot(xy[-1, 0], xy[-1, 1], "X", color=colors[s_i],
                    markeredgecolor="black", markersize=13, zorder=5)
            inside = np.array([
                admiss.is_point_in_domain(float(p[0]), float(p[1])) for p in xy
            ])
            if np.any(~inside):
                ax.plot(xy[~inside, 0], xy[~inside, 1], "v", color="magenta",
                        markeredgecolor="black", markersize=6, linestyle="none",
                        zorder=6)
        ax.plot(true[:, 0], true[:, 1], "P", color="lime", markeredgecolor="black",
                markersize=15, linestyle="none", zorder=7)
        ax.set_xlim(180.0, 300.0)
        ax.set_ylim(240.0, 360.0)
        ax.set_aspect("equal")
        ax.set_title(f"{case} (zoom: strong-source street)", fontsize=10)
    fig2.tight_layout()
    zoom_path = str(out_path).replace(".png", "_zoom.png")
    fig2.savefig(zoom_path, dpi=150)
    print(f"wrote {zoom_path}")


def main():
    cases = sys.argv[1:]
    if not cases:
        print(__doc__)
        return 1
    mesh_data = mdc.build_mesh_data()
    admiss = MeshAdmissibility(mesh_data)
    for case in cases:
        report(case, admiss)
    out = APP / "output" / "obstacle_run_comparison" / "trajectories.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plot(cases, admiss, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
