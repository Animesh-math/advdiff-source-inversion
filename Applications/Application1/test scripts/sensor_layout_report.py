"""Signals, per-case figures/summaries, and the final report for the
sensor-layout comparison (read-only w.r.t. the solver; post-processing only).

Subcommands:
  signals --exp-dir DIR             observation signal magnitude per layout and
                                    per true source (3 diagnostic forward solves
                                    per source configuration, recorded separately)
  case    --exp-dir DIR --case-id ID   figures + case_result.json for one case,
                                    then refresh summary_partial.{json,csv}
  final   --exp-dir DIR             summary.{json,csv}, REPORT.md, overview figure
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_APP1_DIR = Path(__file__).resolve().parents[1]
if str(_APP1_DIR) not in sys.path:
    sys.path.insert(0, str(_APP1_DIR))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import run_optimization as ro  # noqa: E402
from mesh_admissibility import MeshAdmissibility  # noqa: E402
from mesh_data_config import build_mesh_data  # noqa: E402

gt = ro.gt
CASE_ORDER = ["A_L1_boundary", "A_L2_mixed", "A_L3_weak",
              "B_L1_boundary", "B_L2_mixed", "B_L3_weak"]


def _load(p: Path):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _geometry():
    md = build_mesh_data()
    v, _ = ro.load_velocity_field_file(str(ro.DEFAULT_VELOCITY_FIELD_FILE), md)
    return md, np.asarray(md["coord"], dtype=float), v


# --------------------------------------------------------------------------
# signals
# --------------------------------------------------------------------------
def _signal_metrics(states, ids, dt_out):
    out = {}
    for key, state in states.items():
        d = np.asarray(state)[:, ids]
        series = np.einsum("ns,ns->n", d, d)
        j0 = 0.5 * ro.ETA1 * gt._time_reduce(series, dt_out, ro.OBJECTIVE_TIME_RULE)
        per_sensor_max = np.abs(d).max(axis=0)
        out[key] = {
            "J_zero_control": float(j0),
            "peak_abs": float(np.abs(d).max()),
            "rms_over_sensors_and_time": float(np.sqrt(np.mean(d * d))),
            "time_integrated_sum_sq": float(np.sum(series) * dt_out),
            "per_sensor_max_abs": per_sensor_max.tolist(),
            "sensors_above_1e-3": int(np.sum(per_sensor_max > 1e-3)),
            "sensors_above_1e-6": int(np.sum(per_sensor_max > 1e-6)),
        }
    return out


def cmd_signals(exp: Path) -> int:
    layouts = _load(exp / "layouts.json")
    md, coord, v = _geometry()
    pde = dict(ro.PDE)
    pde["velocity_nodes"] = v
    dt_out = pde["dt"] * pde["jprint"]
    t0 = time.perf_counter()
    nfwd = 0
    result = {"note": "diagnostic forward solves for signal magnitude only; NOT part "
                      "of synthetic-data generation, initialization or optimization",
              "eta1": ro.ETA1, "time_rule": ro.OBJECTIVE_TIME_RULE, "cases": {}}
    for cfg, c in layouts["source_configs"].items():
        z = np.asarray(c["true_control"], dtype=float)
        states = {"both_sources": gt.run_forward(md, pde, z)}
        nfwd += 1
        for j in range(z.size // 3):
            zj = z.copy()
            for k in range(z.size // 3):
                if k != j:
                    zj[3 * k + 2] = 0.0
            states[f"source_{j + 1}_only"] = gt.run_forward(md, pde, zj)
            nfwd += 1
        for case in layouts["cases"]:
            if case["source_config"] != cfg:
                continue
            m = _signal_metrics(states, case["sensor_node_ids"], dt_out)
            result["cases"][case["case_id"]] = m
            (exp / "cases" / case["case_id"] / "observation_signal.json").write_text(
                json.dumps(m, indent=2) + "\n")
    for cid, m in result["cases"].items():
        ref = result["cases"].get(cid[:2] + "L1_boundary")
        if ref:
            m["J_zero_ratio_to_L1_boundary"] = (
                m["both_sources"]["J_zero_control"] / ref["both_sources"]["J_zero_control"])
    result["diagnostic_forward_solves"] = nfwd
    result["seconds"] = time.perf_counter() - t0
    (exp / "observation_signals.json").write_text(json.dumps(result, indent=2) + "\n")
    for cid, m in result["cases"].items():
        print(f"{cid:16s} J0(both)={m['both_sources']['J_zero_control']:.4e} "
              + " ".join(f"J0({k})={m[k]['J_zero_control']:.3e}"
                         for k in m if k.startswith("source_")))
    return 0


# --------------------------------------------------------------------------
# per-case figures
# --------------------------------------------------------------------------
def _draw_domain(ax, md, coord, v, alpha=0.55):
    speed = np.linalg.norm(v, axis=1)
    ax.tricontourf(coord[:, 0], coord[:, 1], md["connections"], speed, levels=20,
                   cmap="Blues", alpha=alpha)
    sl = slice(None, None, 6)
    ax.quiver(coord[sl, 0], coord[sl, 1], v[sl, 0], v[sl, 1], color="0.45",
              scale=190, width=0.0015, alpha=0.6)
    for name in md["building_boundary_names"]:
        for a, b in np.asarray(md["boundary_edges_by_name"][name]):
            ax.plot(coord[[a, b], 0], coord[[a, b], 1], color="k", lw=0.9)


def _read_paths(full: Path):
    p = full / "source_paths" / "source_paths.csv"
    if not p.exists():
        return None
    arr = np.genfromtxt(p, delimiter=",", names=True)
    paths = {}
    for sid in np.unique(arr["source_id"]).astype(int):
        sel = arr[arr["source_id"] == sid]
        paths[sid] = np.column_stack([sel["iter"], sel["x"], sel["y"], sel["A"]])
    return paths


def _read_history(full: Path):
    p = full / "optimization_history.csv"
    if not p.exists():
        return None
    return np.genfromtxt(p, delimiter=",", names=True)


def case_figures(exp, case_id, md, coord, v, summary, init_summary, cfg):
    full = exp / "cases" / case_id / "full"
    figs = {}
    true = np.asarray(cfg["true_control"]).reshape(-1, 3)
    sensors = np.asarray([s["actual_xy"] for s in cfg["sensors"]])
    init = np.asarray(summary["initial_control"]).reshape(-1, 3) if summary else None
    final = np.asarray(summary["final_control"]).reshape(-1, 3) if summary else None
    paths = _read_paths(full)
    colors = ["tab:green", "tab:orange", "tab:purple", "tab:brown"]

    # --- reconstruction map + zooms -------------------------------------
    nsrc = true.shape[0]
    fig = plt.figure(figsize=(13, 12.5))
    gs = fig.add_gridspec(nsrc, 2, width_ratios=[1.25, 1.0])
    ax = fig.add_subplot(gs[:, 0])
    zooms = [fig.add_subplot(gs[i, 1]) for i in range(nsrc)]
    for k, a in enumerate([ax] + zooms):
        _draw_domain(a, md, coord, v)
        a.plot(sensors[:, 0], sensors[:, 1], "s", color="gold", mec="k", ms=5 if k else 6,
               label="sensor" if k == 0 else None)
        if init_summary:
            for br in init_summary.get("branch_controls") or []:
                c = np.asarray(br["centers"])
                a.plot(c[:, 0], c[:, 1], "o", mfc="none", mec="0.5", ms=6,
                       label="retained init branches" if (k == 0 and br["branch_index"] == 0) else None)
        if paths:
            for sid, p in paths.items():
                a.plot(p[:, 1], p[:, 2], "-", color=colors[(sid - 1) % 4], lw=1.4, alpha=0.9,
                       label=f"L-BFGS-B path, source {sid}" if k == 0 else None)
                a.plot(p[:, 1], p[:, 2], ".", color=colors[(sid - 1) % 4], ms=2.5)
        a.plot(true[:, 0], true[:, 1], "*", color="red", mec="k", ms=18,
               label="true source" if k == 0 else None)
        if init is not None:
            a.plot(init[:, 0], init[:, 1], "o", color="tab:blue", mec="k", ms=9,
                   label="sensitivity initialization" if k == 0 else None)
        if final is not None:
            a.plot(final[:, 0], final[:, 1], "X", color="lime", mec="k", ms=11,
                   label="final reconstruction" if k == 0 else None)
        a.set_aspect("equal")
    ax.set_xlim(-5, 307)
    ax.set_ylim(-5, 705)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8)
    for i, a in enumerate(zooms):
        cx, cy = true[i, :2]
        pts = [true[i, :2]]
        if final is not None:
            pts += list(final[:, :2])
        if init is not None:
            pts += list(init[:, :2])
        r = 45.0
        a.set_xlim(cx - r, cx + r)
        a.set_ylim(cy - r, cy + r)
        a.set_title(f"zoom on true source {i + 1} ({cx:g}, {cy:g}), A={true[i, 2]:g}", fontsize=9)
    ttl = f"{case_id}: true / initialization / recovered sources and optimization paths"
    if summary:
        ttl += (f"\nJ_init={summary['initial_objective']:.3e}  J_final={summary['final_objective']:.3e}"
                f"  iters={summary['iterations']}  {summary['message']}")
    else:
        ttl += "\n(no optimization summary: run failed before producing a control)"
    fig.suptitle(ttl, fontsize=10)
    p = exp / "cases" / case_id / f"reconstruction_{case_id}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figs["reconstruction_map"] = str(p)

    # --- convergence / source movement history --------------------------
    hist = _read_history(full)
    if hist is not None and summary is not None:
        match = ro.match_sources_by_permutation(np.asarray(cfg["true_control"]),
                                                np.asarray(summary["final_control"]))
        perm = match["permutation"]
        it = hist["iteration"]
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        axs[0, 0].semilogy(it, np.maximum(hist["objective"], 1e-300), "k.-", ms=3)
        axs[0, 0].set_title("objective J")
        axs[0, 1].semilogy(it, np.maximum(hist["gradient_norm"], 1e-300), "k.-", ms=3)
        axs[0, 1].set_title("gradient norm")
        for i in range(nsrc):
            j = perm[i] + 1
            x, y, A = hist[f"x{j}"], hist[f"y{j}"], hist[f"A{j}"]
            err = np.hypot(x - true[i, 0], y - true[i, 1])
            axs[1, 0].semilogy(it, np.maximum(err, 1e-12), ".-", ms=3, color=colors[i],
                               label=f"true S{i + 1} <- est. {j}")
            axs[1, 1].plot(it, A, ".-", ms=3, color=colors[i], label=f"est. {j} (true S{i + 1})")
            axs[1, 1].axhline(true[i, 2], color=colors[i], ls="--", lw=1)
        axs[1, 0].set_title("position error vs matched truth (m)")
        axs[1, 1].set_title("amplitudes (dashed = truth)")
        for a in axs.flat:
            a.set_xlabel("L-BFGS-B iteration")
            a.grid(alpha=0.3)
        axs[1, 0].legend(fontsize=8)
        axs[1, 1].legend(fontsize=8)
        fig.suptitle(f"{case_id}: local optimization history (final label permutation {perm})")
        fig.tight_layout()
        p = exp / "cases" / case_id / f"convergence_{case_id}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        figs["convergence_history"] = str(p)
    sens_pngs = sorted((full / "sensitivity_init").glob("*.png")) if full.exists() else []
    figs["sensitivity_init"] = [str(x) for x in sens_pngs]
    figs["layout"] = cfg.get("layout_figure")
    return figs


# --------------------------------------------------------------------------
# per-case result
# --------------------------------------------------------------------------
def build_case_result(exp: Path, case_id: str, md=None, coord=None, v=None):
    case_dir = exp / "cases" / case_id
    full = case_dir / "full"
    cfg = _load(case_dir / "case_config.json")
    status = _load(case_dir / "batch_status.json") or {}
    instr = _load(case_dir / "run_instrumentation.json") or {}
    summary = _load(full / "optimization_summary.json")
    init = _load(full / "sensitivity_init_summary.json")
    signal = _load(case_dir / "observation_signal.json")
    res = {"case_id": case_id, "source_config": cfg["source_config"],
           "layout": cfg["layout"], "true_control": cfg["true_control"],
           "exit_status": status.get("exit_status", instr.get("exit_status")),
           "start_time": status.get("start_time"), "end_time": status.get("end_time"),
           "batch_wall_seconds": status.get("wall_seconds"),
           "error": instr.get("error"), "log": status.get("log")}
    res["initialization"] = None if init is None else {
        "selected_control": init.get("selected_control"),
        "objective_after_initialization_exact": init.get("initial_exact_objective"),
        "zero_source_objective": init.get("zero_source_objective"),
        "initializer_self_reported_forward_solves": init.get("n_forward_solves"),
        "initializer_self_reported_adjoint_solves": init.get("n_adjoint_solves"),
        "candidate_traces": init.get("n_candidate_traces"),
        "initializer_self_reported_wall_seconds": init.get("wall_time_seconds"),
        "amplitude_solver": (init.get("options") or {}).get("amplitude_solver"),
    }
    if summary is not None:
        pi = summary.get("post_init") or {}
        msg = str(summary.get("message"))
        tc = np.asarray(summary["true_control"])
        fc = np.asarray(summary["final_control"])
        ic = np.asarray(summary["initial_control"])
        match = ro.match_sources_by_permutation(tc, fc)
        imatch = ro.match_sources_by_permutation(tc, ic)
        adm = MeshAdmissibility(md if md is not None else build_mesh_data())
        term = []
        for x, y, a in fc.reshape(-1, 3):
            term.append({"x": float(x), "y": float(y), "A": float(a),
                         "fluid_admissible": bool(adm.is_point_in_domain(float(x), float(y))),
                         "wall_clearance_m": float(adm.distance_to_boundary(float(x), float(y)))})
        lo = summary.get("active_bounds")
        amp_at_bound = [bool(abs(t["A"] - b[0]) < 1e-9 or abs(t["A"] - b[1]) < 1e-9)
                        for t, b in zip(term, [lo[3 * i + 2] for i in range(len(term))])]
        res["optimization"] = {
            "objective_after_initialization": summary["initial_objective"],
            "objective_after_local_optimization": summary["final_objective"],
            "final_gradient_norm": summary["final_gradient_norm"],
            "iterations": summary["iterations"],
            "objective_evaluations_nfev": pi.get("total_objective_evaluations"),
            "gradient_evaluations_njev": pi.get("total_gradient_evaluations"),
            "success_flag": summary["success"],
            "stopping_reason": msg,
            "budget_limited": ("LIMIT" in msg.upper()) or ("EXCEED" in msg.upper()),
            "post_init_forward_solves_driver_counter": pi.get("total_forward_solves"),
            "post_init_adjoint_solves_driver_counter": pi.get("total_adjoint_solves"),
            "initial_control": summary["initial_control"],
            "final_control": summary["final_control"],
        }
        res["recovery"] = {
            "permutation": match["permutation"],
            "position_errors_m": match["position_errors_m"],
            "amplitude_errors": match["amplitude_errors"],
            "max_position_error_m": match["max_position_error_m"],
            "max_amplitude_error": match["max_amplitude_error"],
            "init_position_errors_m": imatch["position_errors_m"],
            "init_amplitude_errors": imatch["amplitude_errors"],
            "init_max_position_error_m": imatch["max_position_error_m"],
            "terminal_sources": term,
            "terminal_all_fluid_admissible": all(t["fluid_admissible"] for t in term),
            "terminal_amplitude_at_bound": amp_at_bound,
            "raw_feasibility_record": summary.get("feasibility"),
        }
    else:
        res["optimization"] = None
        res["recovery"] = None
    ph = instr.get("phases") or {}
    res["pde_solves"] = {
        "synthetic_data_and_setup": ph.get("pre_init_setup_and_synthetic_data"),
        "initialization": ph.get("initialization"),
        "local_optimization": ph.get("local_optimization"),
        "driver_overhead_smoke_export_final_eval": ph.get("driver_overhead_smoke_export_final_eval"),
        "total_counted": instr.get("total_counted"),
        "note": "counts = actual gt.run_forward / gt.rk4_discrete_source_adjoint calls "
                "(cached evaluator responses excluded); 'seconds' = wall time of that phase",
    }
    res["timing_seconds"] = {
        "initialization": (ph.get("initialization") or {}).get("seconds"),
        "local_optimization": (ph.get("local_optimization") or {}).get("seconds"),
        "driver_process_total": instr.get("total_elapsed_seconds_in_process"),
        "batch_wall_total": status.get("wall_seconds"),
    }
    if signal:
        res["observation_signal"] = {k: {kk: vv for kk, vv in val.items()
                                         if kk != "per_sensor_max_abs"}
                                     for k, val in signal.items() if isinstance(val, dict)}
    if md is not None:
        try:
            res["figures"] = case_figures(exp, case_id, md, coord, v, summary, init, cfg)
        except Exception as exc:  # noqa: BLE001
            res["figures"] = {"error": f"figure generation failed: {exc!r}"}
    return res


def _flat_row(r):
    o, rec, ts = r.get("optimization") or {}, r.get("recovery") or {}, r.get("timing_seconds") or {}
    ps = r.get("pde_solves") or {}
    sig = (r.get("observation_signal") or {})

    def g(d, k):
        return (d or {}).get(k)
    row = {
        "case_id": r["case_id"], "source_config": r["source_config"], "layout": r["layout"],
        "exit_status": r.get("exit_status"),
        "J_after_init": o.get("objective_after_initialization"),
        "J_final": o.get("objective_after_local_optimization"),
        "grad_norm_final": o.get("final_gradient_norm"),
        "iterations": o.get("iterations"), "nfev": o.get("objective_evaluations_nfev"),
        "njev": o.get("gradient_evaluations_njev"),
        "stopping_reason": o.get("stopping_reason"), "budget_limited": o.get("budget_limited"),
        "init_fwd": g(ps.get("initialization"), "forward_solves"),
        "init_adj": g(ps.get("initialization"), "adjoint_solves"),
        "opt_fwd": g(ps.get("local_optimization"), "forward_solves"),
        "opt_adj": g(ps.get("local_optimization"), "adjoint_solves"),
        "synthetic_fwd": g(ps.get("synthetic_data_and_setup"), "forward_solves"),
        "overhead_fwd": g(ps.get("driver_overhead_smoke_export_final_eval"), "forward_solves"),
        "overhead_adj": g(ps.get("driver_overhead_smoke_export_final_eval"), "adjoint_solves"),
        "init_seconds": ts.get("initialization"), "opt_seconds": ts.get("local_optimization"),
        "total_seconds": ts.get("batch_wall_total"),
        "init_max_pos_err_m": rec.get("init_max_position_error_m"),
        "max_pos_err_m": rec.get("max_position_error_m"),
        "max_amp_err": rec.get("max_amplitude_error"),
        "pos_errs_m": rec.get("position_errors_m"), "amp_errs": rec.get("amplitude_errors"),
        "terminal_feasible": rec.get("terminal_all_fluid_admissible"),
        "J0_both": g(sig.get("both_sources"), "J_zero_control"),
        "J0_source1": g(sig.get("source_1_only"), "J_zero_control"),
        "J0_source2": g(sig.get("source_2_only"), "J_zero_control"),
        "peak_signal_both": g(sig.get("both_sources"), "peak_abs"),
        "peak_signal_s1": g(sig.get("source_1_only"), "peak_abs"),
        "peak_signal_s2": g(sig.get("source_2_only"), "peak_abs"),
        "error": (r.get("error") or "")[-300:],
    }
    return row


def _write_table(exp, results, stem):
    (exp / f"{stem}.json").write_text(json.dumps(results, indent=2, default=str) + "\n")
    rows = [_flat_row(r) for r in results]
    if rows:
        with open(exp / f"{stem}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for row in rows:
                w.writerow(row)


def cmd_case(exp: Path, case_id: str) -> int:
    md, coord, v = _geometry()
    case_dir = exp / "cases" / case_id
    full = case_dir / "full"
    if (full / "optimization_summary.json").exists():
        status = _load(case_dir / "batch_status.json") or {}
        subprocess.run([sys.executable, str(Path(__file__).parent / "diagnostics" / "report_m_source_run.py"),
                        str(full), "--log", str(status.get("log", "")),
                        "--out", str(case_dir / "recovery_report.txt")],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    res = build_case_result(exp, case_id, md, coord, v)
    (case_dir / "case_result.json").write_text(json.dumps(res, indent=2, default=str) + "\n")
    done = [build_case_result(exp, c) if c != case_id else res
            for c in CASE_ORDER if (exp / "cases" / c / "batch_status.json").exists()]
    _write_table(exp, done, "summary_partial")
    print(f"case {case_id}: result written; partial summary has {len(done)} case(s)")
    return 0


def _fmt(x, f="{:.3e}"):
    if x is None:
        return "n/a"
    try:
        return f.format(x)
    except Exception:  # noqa: BLE001
        return str(x)


def cmd_final(exp: Path) -> int:
    manifest = _load(exp / "manifest.json") or {}
    results = []
    for c in CASE_ORDER:
        p = exp / "cases" / c / "case_result.json"
        r = _load(p)
        if r is None:
            r = {"case_id": c, "source_config": c[0], "layout": c[2:], "exit_status": "not run",
                 "optimization": None, "recovery": None}
        results.append(r)
    _write_table(exp, results, "summary")

    # overview figure: 2 x 3 reconstruction maps
    try:
        import matplotlib.image as mpimg
        fig, axs = plt.subplots(2, 3, figsize=(18, 17))
        for a, c in zip(axs.flat, CASE_ORDER):
            p = exp / "cases" / c / f"reconstruction_{c}.png"
            a.axis("off")
            if p.exists():
                a.imshow(mpimg.imread(p))
            a.set_title(c)
        fig.tight_layout()
        fig.savefig(exp / "overview_reconstructions.png", dpi=110)
        plt.close(fig)
        fig, axs = plt.subplots(2, 3, figsize=(15, 15))
        for a, c in zip(axs.flat, CASE_ORDER):
            p = exp / "cases" / c / f"layout_{c}.png"
            a.axis("off")
            if p.exists():
                a.imshow(mpimg.imread(p))
        fig.tight_layout()
        fig.savefig(exp / "overview_layouts.png", dpi=110)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        print(f"overview figure failed: {exc!r}")

    L = []
    L.append("# Sensor-layout comparison: Application1 (DC Navy Yard)\n")
    L.append(f"Experiment directory: `{exp}`  ")
    L.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')} (automatic, no analysis)  ")
    L.append(f"Commit: `{manifest.get('git', {}).get('commit')}`  host: "
             f"`{manifest.get('environment', {}).get('hostname')}`\n")
    L.append("## Settings (identical for all six cases except sensor placement)\n")
    L.append("```text")
    L.append(" ".join(manifest.get("common_driver_args", [])))
    L.append("```")
    for k, val in (manifest.get("physical_settings") or {}).items():
        L.append(f"- {k}: {val}")
    L.append("")
    L.append("## Source configurations\n")
    for cfg, c in (manifest.get("source_configs") or {}).items():
        L.append(f"- **{cfg}**: {c['label']} — truth `{c['true_control']}` ({c['documented_as']})")
    L.append("\n## Results\n")
    L.append("| case | exit | J after init | J final | iters | nfev/njev | stop | init fwd/adj | opt fwd/adj | init s | opt s | total s | init max pos err (m) | max pos err (m) | max amp err | terminal feasible | J0(both) | J0(S1) | J0(S2) |")
    L.append("|" + "---|" * 19)
    for r in results:
        row = _flat_row(r)
        L.append("| " + " | ".join([
            row["case_id"], str(row["exit_status"]), _fmt(row["J_after_init"]), _fmt(row["J_final"]),
            str(row["iterations"]), f"{row['nfev']}/{row['njev']}",
            (row["stopping_reason"] or "n/a").replace("|", "/"),
            f"{row['init_fwd']}/{row['init_adj']}", f"{row['opt_fwd']}/{row['opt_adj']}",
            _fmt(row["init_seconds"], "{:.0f}"), _fmt(row["opt_seconds"], "{:.0f}"),
            _fmt(row["total_seconds"], "{:.0f}"),
            _fmt(row["init_max_pos_err_m"], "{:.3g}"), _fmt(row["max_pos_err_m"], "{:.3g}"),
            _fmt(row["max_amp_err"], "{:.3g}"), str(row["terminal_feasible"]),
            _fmt(row["J0_both"]), _fmt(row["J0_source1"]), _fmt(row["J0_source2"]),
        ]) + " |")
    L.append("")
    L.append("Solve counts are actual PDE solves (forward/adjoint) counted at "
             "`gt.run_forward` / `gt.rk4_discrete_source_adjoint`; nfev/njev are optimizer "
             "evaluation counts (cached responses do not trigger solves). Synthetic data "
             "generation and driver overhead (forward smoke/export solves, final evaluation) "
             "are listed per case in `summary.json` under `pde_solves`. J0 = objective at the "
             "zero control for that layout (observation signal energy, eta1-weighted), per "
             "source from `observation_signals.json` (diagnostic solves, separate).\n")
    L.append("## Failures / budget-limited terminations\n")
    flagged = False
    for r in results:
        o = r.get("optimization") or {}
        if r.get("exit_status") not in (0, "0"):
            flagged = True
            L.append(f"- **{r['case_id']}**: exit status {r.get('exit_status')}; "
                     f"error tail: `{(r.get('error') or 'see log')[-400:]}`; log `{r.get('log')}`")
        elif o.get("budget_limited"):
            flagged = True
            L.append(f"- **{r['case_id']}**: budget-limited termination: {o.get('stopping_reason')}")
    if not flagged:
        L.append("- none")
    L.append("\n## Figures\n")
    L.append("- overview: `overview_reconstructions.png`, `overview_layouts.png`")
    for r in results:
        f = r.get("figures") or {}
        L.append(f"- {r['case_id']}: layout `{f.get('layout')}`, reconstruction "
                 f"`{f.get('reconstruction_map')}`, convergence `{f.get('convergence_history')}`, "
                 f"sensitivity-init {len(f.get('sensitivity_init') or [])} PNG(s) in "
                 f"`cases/{r['case_id']}/full/sensitivity_init/`")
    L.append("\nPer-case: `cases/<id>/case_result.json`, `recovery_report.txt`, "
             "`run_instrumentation.json`, `batch_status.json`, `run.log`, `full/` (driver outputs, "
             "ParaView states, `source_paths/`, histories).")
    (exp / "REPORT.md").write_text("\n".join(L) + "\n")
    print(f"final report: {exp / 'REPORT.md'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["signals", "case", "final"])
    ap.add_argument("--exp-dir", required=True)
    ap.add_argument("--case-id")
    a = ap.parse_args()
    exp = Path(a.exp_dir).resolve()
    if a.command == "signals":
        return cmd_signals(exp)
    if a.command == "case":
        return cmd_case(exp, a.case_id)
    return cmd_final(exp)


if __name__ == "__main__":
    sys.exit(main())
