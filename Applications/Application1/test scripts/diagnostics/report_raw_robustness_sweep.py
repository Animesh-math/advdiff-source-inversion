"""Read-only report for the raw-mode (``--feasibility-mode raw``) robustness sweep.

Usage:
  python3 report_raw_robustness_sweep.py <case_name> [<case_name> ...] \
      [--master-log PATH] [--out-dir DIR]

For each case it reads
``output/<case>/full/{optimization_summary.json, sigma_continuation_summary.json,
optimization_history.csv, sensitivity_init_summary.json}`` and reports every
field required by the raw-mode robustness protocol: true / initial / final
control, objectives, termination, iteration and solve accounting, per-source
position and amplitude errors after optimal label permutation, building
incursions (which source entered which building, whether it crossed completely
through, longest infeasible run), terminal feasibility and clearance, amplitude
bound activity, and line-search / convergence warnings.

The terminal point is reported EXACTLY as returned by L-BFGS-B; nothing is
projected here.

It also writes one trajectory figure per case (building geometry, true centers,
initialized centers, complete per-source iterate trajectories with consistent
source identity, final centers, and infeasible trajectory segments drawn in a
distinct style), plus a combined comparison table.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
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

AMP_LOWER = 0.1
AMP_UPPER = 2.0
AMP_BOUND_TOL = 1.0e-6
PIN_TOL = 1.0e-6

EXACT_POS_M = 1.0
EXACT_AMP = 0.02
ACCEPTABLE_POS_M = 20.0
ACCEPTABLE_AMP = 0.15

CLASSES = {
    1: "exact or near-exact recovery",
    2: "acceptable recovery",
    3: "wrong feasible local minimum",
    4: "infeasible terminal solution",
    5: "optimizer failure",
}


# --------------------------------------------------------------------------- IO
def load_case(case):
    d = APP / "output" / case / "full"
    summary = json.loads((d / "optimization_summary.json").read_text())
    hist = np.loadtxt(d / "optimization_history.csv", delimiter=",", skiprows=1)
    if hist.ndim == 1:
        hist = hist[None, :]
    stages = None
    p = d / "sigma_continuation_summary.json"
    if p.exists():
        stages = json.loads(p.read_text())
    init = None
    p = d / "sensitivity_init_summary.json"
    if p.exists():
        init = json.loads(p.read_text())
    return summary, hist, stages, init, d


def wall_clock_from_master_log(master_log, case):
    """(seconds, start_iso, end_iso) parsed from the sweep driver's markers."""
    if master_log is None or not Path(master_log).exists():
        return None, None, None
    text = Path(master_log).read_text(errors="replace")
    s = re.search(rf"START {re.escape(case)} (\S+)", text)
    e = re.search(rf"END {re.escape(case)} rc=\d+ (\S+)", text)
    if not (s and e):
        return None, (s.group(1) if s else None), (e.group(1) if e else None)
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    t0 = datetime.strptime(s.group(1), fmt)
    t1 = datetime.strptime(e.group(1), fmt)
    return (t1 - t0).total_seconds(), s.group(1), e.group(1)


def scan_log_warnings(log_path):
    """Line-search / convergence / numerical warnings from the run log."""
    if log_path is None or not Path(log_path).exists():
        return None
    patterns = {
        "ABNORMAL_TERMINATION_IN_LNSRCH": r"ABNORMAL_TERMINATION_IN_LNSRCH",
        "ITERATION LIMIT": r"TOTAL NO\. of ITERATIONS REACHED LIMIT",
        "RuntimeWarning": r"RuntimeWarning",
        "LineSearchWarning": r"LineSearchWarning",
        "overflow/invalid": r"(overflow encountered|invalid value encountered)",
        "ConvergenceWarning": r"ConvergenceWarning",
    }
    text = Path(log_path).read_text(errors="replace")
    return {k: len(re.findall(v, text)) for k, v in patterns.items()}


# ------------------------------------------------------------------- geometry
class BuildingIndex:
    """Attribute an inadmissible point to the building whose wall is nearest."""

    def __init__(self, mesh_data):
        coord = np.asarray(mesh_data["coord"], dtype=np.float64)[:, :2]
        self.names, self.a, self.b, self.owner = [], [], [], []
        for name, edges in mesh_data["boundary_edges_by_name"].items():
            e = np.asarray(edges)
            if e.ndim != 2 or e.shape[0] == 0:
                continue
            idx = len(self.names)
            self.names.append(name)
            self.a.append(coord[e[:, 0]])
            self.b.append(coord[e[:, 1]])
            self.owner.append(np.full(e.shape[0], idx))
        self.a = np.vstack(self.a)
        self.b = np.vstack(self.b)
        self.owner = np.concatenate(self.owner)
        self.ab = self.b - self.a
        self.len_sq = np.einsum("ij,ij->i", self.ab, self.ab)
        self.len_sq[self.len_sq == 0.0] = 1.0

    def nearest(self, x, y):
        p = np.array([float(x), float(y)])
        t = np.clip(np.einsum("ij,ij->i", p - self.a, self.ab) / self.len_sq, 0.0, 1.0)
        feet = self.a + t[:, None] * self.ab
        d = np.hypot(*(feet - p).T)
        k = int(np.argmin(d))
        return self.names[self.owner[k]], float(d[k])


def infeasible_runs(inside):
    """Maximal runs of consecutive False entries as (start, end) index pairs."""
    runs, i, n = [], 0, len(inside)
    while i < n:
        if not inside[i]:
            j = i
            while j + 1 < n and not inside[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def stage_boundaries(hist, stages):
    """History row index at which each continuation stage ends."""
    if not stages:
        return []
    bounds = []
    for st in stages.get("stages", []):
        target = np.asarray(st["stage_final_control"], dtype=np.float64)
        d = np.linalg.norm(hist[:, 3:] - target[None, :], axis=1)
        bounds.append(int(np.argmin(d)))
    return bounds


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
    if len(xy) < 2:
        return 0, 0, 0
    step = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
    best, run, start = (0, 0, 0), 0, 0
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


# --------------------------------------------------------------------- report
def analyse(case, admiss, bidx, master_log, log_dir):
    summary, hist, stages, initsum, d = load_case(case)
    true = np.asarray(summary["true_control"], dtype=np.float64)
    final = np.asarray(summary["final_control"], dtype=np.float64)
    init = np.asarray(summary["initial_control"], dtype=np.float64)
    t = true.reshape(-1, 3)
    f = final.reshape(-1, 3)
    n_src = t.shape[0]
    perm = match_sources(final, true)
    permuted = tuple(perm) != tuple(range(n_src))

    feas = summary.get("feasibility") or {}
    log_path = Path(log_dir) / f"{case}.log" if log_dir else None
    secs, t0, t1 = wall_clock_from_master_log(master_log, case)
    warns = scan_log_warnings(log_path)

    rec = {
        "case": case,
        "true_control": true.tolist(),
        "initial_control": init.tolist(),
        "final_control": final.tolist(),
        "initial_objective": float(summary["initial_objective"]),
        "final_objective": float(summary["final_objective"]),
        "final_gradient_norm": float(summary["final_gradient_norm"]),
        "success": bool(summary.get("success")),
        "message": summary.get("message"),
        "labels_permuted": permuted,
        "permutation": list(perm),
        "wall_seconds": secs,
        "wall_start": t0,
        "wall_end": t1,
        "warnings": warns,
        "feasibility": feas,
        "sensitivity_init": {
            "initial_exact_objective": (initsum or {}).get("initial_exact_objective"),
            "zero_source_objective": (initsum or {}).get("zero_source_objective"),
            "n_forward_solves": (initsum or {}).get("n_forward_solves"),
            "n_adjoint_solves": (initsum or {}).get("n_adjoint_solves"),
        },
        "sources": [],
        "stages": [],
    }

    print("\n" + "=" * 90)
    print(f"CASE  {case}")
    print("=" * 90)
    print(f"  feasibility mode        : {feas.get('feasibility_mode')}")
    print(f"  measurement boundaries  : {summary.get('measurement_boundaries')}")
    print(f"  sigma2 continuation     : {summary.get('sigma2_continuation')}")
    print(f"  amplitude floor / bounds: {summary.get('amplitude_floor')} / "
          f"[{AMP_LOWER}, {AMP_UPPER}]")
    print(f"  random seed             : {summary.get('random_seed')}")
    print(f"  TRUE control            : {np.array2string(true, precision=6)}")
    print(f"  SENSITIVITY init control: {np.array2string(init, precision=8)}")
    if initsum:
        print(f"    init exact J          : {initsum.get('initial_exact_objective'):.10e}"
              f"   (zero-source J {initsum.get('zero_source_objective'):.6e})")
        print(f"    init solve budget     : {initsum.get('n_forward_solves')} forward, "
              f"{initsum.get('n_adjoint_solves')} adjoint")
    print(f"  initial objective (stg1): {summary['initial_objective']:.10e}")
    print(f"  FINAL control (RAW, unprojected): {np.array2string(final, precision=8)}")
    print(f"  FINAL objective         : {summary['final_objective']:.10e}")
    print(f"  final gradient norm     : {summary['final_gradient_norm']:.6e}")
    print(f"  success flag            : {summary.get('success')}")
    print(f"  termination message     : {summary.get('message')}")
    if secs is not None:
        print(f"  wall-clock              : {secs / 60.0:.1f} min  ({t0} -> {t1})")

    total_iters = 0
    if stages:
        print("  continuation stages:")
        for st in stages["stages"]:
            total_iters += int(st["iterations"])
            print(f"    stage {st['stage_index']} sigma2={st['sigma2']:>7.1f}: "
                  f"J {st['stage_initial_objective']:.6e} -> "
                  f"{st['stage_final_objective']:.6e}  "
                  f"iters={st['iterations']:4d}  success={st['success']}  "
                  f"{st['message']}")
            rec["stages"].append({
                "stage_index": st["stage_index"], "sigma2": st["sigma2"],
                "iterations": st["iterations"], "success": st["success"],
                "message": st["message"],
                "J0": st["stage_initial_objective"],
                "Jf": st["stage_final_objective"],
            })
    print(f"  ITERATIONS (sum over stages) : {total_iters}   "
          f"(recorded iterates in history: {hist.shape[0] - 1})")
    rec["iterations_total"] = total_iters
    rec["history_rows"] = int(hist.shape[0])

    # Solve accounting.  The driver does not persist L-BFGS-B nfev/njev; every
    # recorded iterate costs at least one forward and one adjoint solve (the
    # evaluator caches the forward state for the gradient), and line-search
    # trials add further forward solves that are not counted individually.
    n_invalid_ev = feas.get("invalid_evaluations")
    print(f"  solve accounting        : sensitivity init "
          f"{(initsum or {}).get('n_forward_solves')} forward / "
          f"{(initsum or {}).get('n_adjoint_solves')} adjoint (exact); "
          f"optimization >= {hist.shape[0] - 1} forward + "
          f"{hist.shape[0] - 1} adjoint (per accepted iterate; L-BFGS-B "
          f"nfev/njev are not persisted by the driver)")
    print(f"  raw-mode evaluation counters: invalid evals={n_invalid_ev}, "
          f"raw infeasible evals={feas.get('raw_infeasible_evaluations')}, "
          f"raw infeasible iterates={feas.get('raw_infeasible_iterates')}, "
          f"max distance outside fluid={feas.get('raw_max_outside_distance')}, "
          f"projections={feas.get('projected_evaluations')}, "
          f"penalties={feas.get('penalized_evaluations')}")

    # ------------------------------------------------- per-source diagnostics
    print(f"  label permutation needed: {permuted}  (final index -> true index: "
          + ", ".join(f"f{perm[i] + 1}->t{i + 1}" for i in range(n_src)) + ")")
    print("  per-source recovery (after optimal permutation):")
    max_pos_err = 0.0
    max_amp_err = 0.0
    any_final_outside = False
    for i in range(n_src):
        s = f[perm[i]]
        dpos = float(np.hypot(*(s[:2] - t[i, :2])))
        damp = float(abs(s[2] - t[i, 2]))
        max_pos_err = max(max_pos_err, dpos)
        max_amp_err = max(max_amp_err, damp)
        inside = admiss.is_point_in_domain(float(s[0]), float(s[1]))
        clear = admiss.distance_to_boundary(float(s[0]), float(s[1]))
        bname, bdist = bidx.nearest(float(s[0]), float(s[1]))
        outside_d = 0.0
        if not inside:
            any_final_outside = True
            outside_d = float(np.sqrt(
                admiss.nearest_domain_distance_sq(float(s[0]), float(s[1]))[0]))
        at_lo = abs(s[2] - AMP_LOWER) <= AMP_BOUND_TOL
        at_hi = abs(s[2] - AMP_UPPER) <= AMP_BOUND_TOL
        print(f"    true s{i+1} ({t[i,0]:8.3f},{t[i,1]:8.3f}) A={t[i,2]:.4f}"
              f"  <-  final ({s[0]:11.5f},{s[1]:11.5f}) A={s[2]:.6f}")
        print(f"          position error {dpos:10.4f} m    amplitude error {damp:.6f}"
              f"    amplitude at bound: {'LOWER' if at_lo else ('UPPER' if at_hi else 'no')}")
        print(f"          final in fluid = {inside}"
              + (f"  (OUTSIDE by {outside_d:.4f} m)" if not inside else "")
              + f"   clearance to nearest wall = {clear:.4f} m"
              f"   (nearest boundary: {bname} at {bdist:.4f} m)")
        rec["sources"].append({
            "true": t[i].tolist(), "final": s.tolist(),
            "position_error_m": dpos, "amplitude_error": damp,
            "final_in_fluid": bool(inside), "final_outside_by_m": outside_d,
            "final_wall_clearance_m": float(clear),
            "nearest_boundary": bname, "nearest_boundary_distance_m": bdist,
            "amplitude_at_lower_bound": bool(at_lo),
            "amplitude_at_upper_bound": bool(at_hi),
        })
    rec["max_position_error_m"] = max_pos_err
    rec["max_amplitude_error"] = max_amp_err
    rec["final_feasible"] = not any_final_outside

    # ------------------------------------------------ trajectory feasibility
    print("  trajectory feasibility (per source, over the full concatenated history):")
    traj = {}
    any_entered = False
    any_crossed = False
    longest_infeasible = 0
    for s_i in range(n_src):
        xy = hist[:, 3 + 3 * s_i: 5 + 3 * s_i]
        amp = hist[:, 5 + 3 * s_i]
        inside = np.array([admiss.is_point_in_domain(float(p[0]), float(p[1]))
                           for p in xy])
        traj[s_i] = (xy, inside)
        runs = infeasible_runs(inside)
        n_out = int(np.sum(~inside))
        sat_hi = int(np.sum(amp >= AMP_UPPER - AMP_BOUND_TOL))
        sat_lo = int(np.sum(amp <= AMP_LOWER + AMP_BOUND_TOL))
        pin, p0, p1 = pin_runs(xy)
        entered = {}
        crossings = []
        for (i0, i1) in runs:
            longest_infeasible = max(longest_infeasible, i1 - i0 + 1)
            for r in range(i0, i1 + 1):
                bname, _ = bidx.nearest(float(xy[r, 0]), float(xy[r, 1]))
                entered[bname] = entered.get(bname, 0) + 1
            # complete transit: feasible before AND after, and the two feasible
            # endpoints are separated by a wall (straight path leaves the fluid)
            if i0 - 1 >= 0 and i1 + 1 < len(inside):
                p, q = xy[i0 - 1], xy[i1 + 1]
                if not admiss.segment_in_domain(p, q, spacing=2.0):
                    bname, _ = bidx.nearest(float(xy[(i0 + i1) // 2, 0]),
                                            float(xy[(i0 + i1) // 2, 1]))
                    crossings.append({
                        "rows": [int(i0), int(i1)],
                        "iterations": [int(hist[i0, 0]), int(hist[i1, 0])],
                        "from": p.tolist(), "to": q.tolist(),
                        "building": bname,
                        "run_length": int(i1 - i0 + 1),
                    })
        if n_out:
            any_entered = True
        if crossings:
            any_crossed = True
        print(f"    source {s_i+1}: {n_out}/{len(inside)} iterates outside the fluid "
              f"mesh; {len(runs)} infeasible run(s); longest run "
              f"{max([r[1]-r[0]+1 for r in runs], default=0)} consecutive iterations")
        if entered:
            print("        buildings entered (iterate counts by nearest wall): "
                  + ", ".join(f"{k}:{v}" for k, v in sorted(entered.items())))
        for c in crossings:
            print(f"        COMPLETE TRANSIT of {c['building']}: iterations "
                  f"{c['iterations'][0]}..{c['iterations'][1]} "
                  f"({c['run_length']} infeasible iterates) from "
                  f"({c['from'][0]:.2f},{c['from'][1]:.2f}) to "
                  f"({c['to'][0]:.2f},{c['to'][1]:.2f}) -- the two feasible "
                  f"endpoints are NOT connected by a fluid straight line")
        if not crossings and n_out:
            print("        no complete transit detected (excursions returned to the "
                  "same side, or the run touches the trajectory endpoints)")
        print(f"        longest frozen-position run = {pin} consecutive iterations "
              f"at ({xy[p0,0]:.5f},{xy[p0,1]:.5f}), wall clearance "
              f"{admiss.distance_to_boundary(float(xy[p0,0]), float(xy[p0,1])):.6f} m")
        print(f"        amplitude at UPPER bound in {sat_hi} iterates, at LOWER bound "
              f"in {sat_lo} iterates; final A={amp[-1]:.6f}")
        rec["sources"][s_i].update({
            "iterates_outside": n_out,
            "n_infeasible_runs": len(runs),
            "longest_infeasible_run": max([r[1] - r[0] + 1 for r in runs], default=0),
            "buildings_entered": entered,
            "complete_transits": crossings,
            "longest_frozen_run": int(pin),
            "amplitude_upper_bound_iterates": sat_hi,
            "amplitude_lower_bound_iterates": sat_lo,
        })
    rec["any_iterate_entered_building"] = any_entered
    rec["any_complete_transit"] = any_crossed
    rec["longest_infeasible_run"] = longest_infeasible

    print(f"  FINAL FEASIBILITY (raw terminal point, NOT projected): "
          f"{'FEASIBLE' if rec['final_feasible'] else 'INFEASIBLE'}")
    if warns:
        print("  log warnings            : "
              + ", ".join(f"{k}={v}" for k, v in warns.items() if v))
        if not any(warns.values()):
            print("  log warnings            : none")

    # ---------------------------------------------------------- classification
    stage_msgs = [s["message"] for s in rec["stages"]]
    lnsrch = any("ABNORMAL_TERMINATION_IN_LNSRCH" in m for m in stage_msgs)
    if not rec["final_feasible"]:
        cls = 4
    elif max_pos_err <= EXACT_POS_M and max_amp_err <= EXACT_AMP:
        cls = 1
    elif max_pos_err <= ACCEPTABLE_POS_M and max_amp_err <= ACCEPTABLE_AMP:
        cls = 2
    elif lnsrch and rec["final_objective"] > 0.1 * rec["initial_objective"]:
        cls = 5
    else:
        cls = 3
    rec["classification"] = cls
    rec["classification_label"] = CLASSES[cls]
    rec["lnsrch_termination_any_stage"] = lnsrch
    print(f"  CLASSIFICATION          : {cls} - {CLASSES[cls]}")

    rec["_hist"] = hist
    rec["_traj"] = traj
    rec["_stage_bounds"] = stage_boundaries(hist, stages)
    return rec


# ----------------------------------------------------------------------- plot
def plot_case(rec, admiss, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    admiss._ensure_boundary_geometry()
    a, b = admiss._boundary_a, admiss._boundary_b
    hist = rec["_hist"]
    traj = rec["_traj"]
    true = np.asarray(rec["true_control"]).reshape(-1, 3)
    init = np.asarray(rec["initial_control"]).reshape(-1, 3)
    final = np.asarray(rec["final_control"]).reshape(-1, 3)

    fig, axes = plt.subplots(1, 2, figsize=(16.0, 12.0),
                             gridspec_kw={"width_ratios": [1.0, 1.0]})
    colors = ["tab:blue", "tab:red", "tab:green", "tab:purple"]

    for ax_i, ax in enumerate(axes):
        for k in range(a.shape[0]):
            ax.plot([a[k, 0], b[k, 0]], [a[k, 1], b[k, 1]], "-",
                    color="0.35", linewidth=0.7, zorder=1)
        for s_i, (xy, inside) in traj.items():
            c = colors[s_i % len(colors)]
            # feasible segments (solid) vs infeasible segments (dashed magenta):
            # a segment is "infeasible" if either endpoint is outside the fluid.
            for r in range(len(xy) - 1):
                seg_ok = inside[r] and inside[r + 1]
                ax.plot(xy[r:r + 2, 0], xy[r:r + 2, 1],
                        "-" if seg_ok else "--",
                        color=c if seg_ok else "magenta",
                        linewidth=1.5 if seg_ok else 2.2,
                        alpha=0.9, zorder=3 if seg_ok else 4)
            ax.plot(xy[~inside, 0], xy[~inside, 1], "v", color="magenta",
                    markeredgecolor="black", markersize=6, linestyle="none",
                    zorder=6,
                    label="iterate INSIDE a building" if (s_i == 0 and ax_i == 0)
                    else None)
            ax.plot(init[s_i, 0], init[s_i, 1], "o", color=c,
                    markeredgecolor="black", markersize=12, zorder=7,
                    label=f"source {s_i+1} sensitivity init" if ax_i == 0 else None)
            ax.plot(final[s_i, 0], final[s_i, 1], "X", color=c,
                    markeredgecolor="black", markersize=16, zorder=8,
                    label=f"source {s_i+1} FINAL (raw)" if ax_i == 0 else None)
            ax.plot([], [], "-", color=c, linewidth=1.5,
                    label=f"source {s_i+1} trajectory" if ax_i == 0 else None)
        ax.plot(true[:, 0], true[:, 1], "P", color="lime", markeredgecolor="black",
                markersize=18, linestyle="none", zorder=9,
                label="TRUE source" if ax_i == 0 else None)
        ax.set_aspect("equal")
        ax.set_xlabel("x (m, local)")
        ax.set_ylabel("y (m, local)")

    axes[0].set_title(
        f"{rec['case']}\nfeasibility=raw   J {rec['initial_objective']:.3e} -> "
        f"{rec['final_objective']:.3e}   class {rec['classification']}: "
        f"{rec['classification_label']}", fontsize=11)
    axes[0].legend(loc="upper right", fontsize=8)

    # zoom on the union of true / init / final / trajectory extent
    pts = np.vstack([true[:, :2], init[:, :2], final[:, :2]]
                    + [xy for xy, _ in traj.values()])
    pad = 40.0
    axes[1].set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
    axes[1].set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)
    axes[1].set_title("zoom on true / init / final / trajectory extent", fontsize=11)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="+")
    ap.add_argument("--master-log", default=None)
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (
        APP / "output" / "raw_robustness_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh_data = mdc.build_mesh_data()
    admiss = MeshAdmissibility(mesh_data)
    bidx = BuildingIndex(mesh_data)

    recs = []
    for case in args.cases:
        rec = analyse(case, admiss, bidx, args.master_log, args.log_dir)
        plot_case(rec, admiss, out_dir / f"trajectory_{case}.png")
        recs.append(rec)

    # ------------------------------------------------------ comparison table
    print("\n" + "=" * 130)
    print("COMPARISON TABLE")
    print("=" * 130)
    hdr = (f"{'case':<34}{'J_init':>12}{'J_final':>12}{'iters':>7}"
           f"{'maxPosErr':>11}{'maxAmpErr':>11}{'inBldg':>8}{'transit':>9}"
           f"{'feasEnd':>9}{'class':>7}{'min':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in recs:
        print(f"{r['case']:<34}{r['initial_objective']:>12.3e}"
              f"{r['final_objective']:>12.3e}{r['iterations_total']:>7d}"
              f"{r['max_position_error_m']:>11.3f}{r['max_amplitude_error']:>11.4f}"
              f"{str(r['any_iterate_entered_building']):>8}"
              f"{str(r['any_complete_transit']):>9}"
              f"{str(r['final_feasible']):>9}{r['classification']:>7d}"
              + (f"{r['wall_seconds']/60.0:>7.1f}" if r["wall_seconds"] else
                 f"{'n/a':>7}"))
    print("-" * len(hdr))
    for r in recs:
        print(f"  {r['case']}: {r['classification']} - {r['classification_label']}"
              f"   termination: {r['message']}")

    payload = [{k: v for k, v in r.items() if not k.startswith("_")} for r in recs]
    (out_dir / "raw_robustness_sweep.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n")
    print(f"\nwrote {out_dir / 'raw_robustness_sweep.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
