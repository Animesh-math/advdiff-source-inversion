"""Sweep-level report for the final Application1 robustness sweep.

Read-only.  Reuses the validated per-case analysis in
``report_raw_robustness_sweep.py`` and adds

  * per-continuation-stage detail (input/output control, per-stage errors after
    optimal source matching, amplitude-bound activity, in-building iterates,
    stage-end feasibility),
  * initializer stage detail parsed from the archived run log (top-K screen,
    coarse -> refined first-source centers, coherence rejections, kept branches),
  * the six-label recovery classification defined for this sweep,
  * one global CSV + JSON + Markdown report.

Usage:
  python3 report_final_sweep.py [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from itertools import permutations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
APP = HERE.parents[1]
sys.path.insert(0, str(APP.parents[1] / "Src" / "Objective"))
sys.path.insert(0, str(APP))
sys.path.insert(0, str(HERE))

import mesh_data_config as mdc  # noqa: E402
from mesh_admissibility import MeshAdmissibility  # noqa: E402
import report_raw_robustness_sweep as rrs  # noqa: E402

SWEEP = APP / "output" / "final_sweep"
LOGS = SWEEP / "logs"

# ---------------------------------------------------------------- thresholds
# Mesh context: median edge length ~7.5 m, median fluid wall clearance 9.76 m,
# target Gaussian e-folding radius sqrt(400) = 20 m.
PERFECT_POS_M = 1.0        # sub-mesh-edge: the optimizer resolved the center
PERFECT_AMP = 1.0e-3
PERFECT_OBJ = 1.0e-10      # "near numerical precision" (exact runs land ~1e-16)
NEAREXACT_POS_M = 5.0      # well inside one mesh edge of the truth
NEAREXACT_AMP = 1.0e-2
ACCEPTABLE_POS_M = 20.0    # one Gaussian e-folding radius at the target width
ACCEPTABLE_AMP = 0.15
AMP_LOWER, AMP_UPPER, AMP_TOL = 0.1, 2.0, 1.0e-6
# "correct broad basin" for the initializer: within one Gaussian radius at the
# INITIALIZATION width (sqrt(1600) = 40 m) of the truth.
BASIN_M = 40.0

LABELS = [
    "PERFECT RECOVERY",
    "NEAR-EXACT RECOVERY",
    "ACCEPTABLE RECOVERY",
    "WRONG FEASIBLE LOCAL MINIMUM",
    "INFEASIBLE TERMINAL SOLUTION",
    "OPTIMIZER FAILURE",
    # Added during the sweep: a run can abort inside the initializer, producing
    # no control at all.  That is neither a recovery nor an optimizer outcome,
    # so it needs its own honest label rather than being folded into one.
    "INITIALIZATION FAILURE",
]


def build_master_log() -> Path:
    """Rewrite progress.log into the marker format report_raw_* expects."""
    src = SWEEP / "progress.log"
    out = SWEEP / "master_log.txt"
    lines = []
    for line in src.read_text(errors="replace").splitlines():
        p = line.split()
        if line.startswith("SWEEP_CASE_START") and len(p) >= 4:
            lines.append(f"START {p[2]} {p[3]}")
        elif line.startswith("SWEEP_CASE_END") and len(p) >= 6:
            rc = p[3]
            lines.append(f"END {p[2]} {rc} {p[-1]}")
    out.write_text("\n".join(lines) + "\n")
    return out


def match_perm(got: np.ndarray, true: np.ndarray):
    g = np.asarray(got, dtype=np.float64).reshape(-1, 3)
    t = np.asarray(true, dtype=np.float64).reshape(-1, 3)
    best, cost = None, None
    for perm in permutations(range(len(t))):
        c = sum(float(np.hypot(*(g[perm[i], :2] - t[i, :2]))) for i in range(len(t)))
        if cost is None or c < cost:
            best, cost = perm, c
    pos = np.array([float(np.hypot(*(g[best[i], :2] - t[i, :2]))) for i in range(len(t))])
    amp = np.array([abs(float(g[best[i], 2] - t[i, 2])) for i in range(len(t))])
    return best, pos, amp


def classify(rec) -> str:
    pos = rec["max_position_error_m"]
    amp = rec["max_amplitude_error"]
    jf = rec["final_objective"]
    if not rec["final_feasible"]:
        return "INFEASIBLE TERMINAL SOLUTION"
    if pos <= PERFECT_POS_M and amp <= PERFECT_AMP and jf <= PERFECT_OBJ:
        return "PERFECT RECOVERY"
    if pos <= NEAREXACT_POS_M and amp <= NEAREXACT_AMP:
        return "NEAR-EXACT RECOVERY"
    separated = rec.get("sources_separated", True)
    if pos <= ACCEPTABLE_POS_M and amp <= ACCEPTABLE_AMP and separated:
        return "ACCEPTABLE RECOVERY"
    if rec.get("lnsrch_termination_any_stage") and jf > 0.1 * rec["initial_objective"]:
        return "OPTIMIZER FAILURE"
    return "WRONG FEASIBLE LOCAL MINIMUM"


def parse_init_log(log_path: Path) -> dict:
    """Initializer stage detail that lives only in the run log."""
    if not log_path.exists():
        return {}
    text = log_path.read_text(errors="replace")
    # Only the initialization section (ends at the first optimizer banner).
    cut = text.find("sensitivity greedy initial control")
    head = text[:cut] if cut > 0 else text
    refine = [
        {"coarse": [float(m[0]), float(m[1])], "refined": [float(m[2]), float(m[3])],
         "J_before": float(m[4]), "J_after": float(m[5]), "a1": float(m[6]),
         "evals": int(m[7])}
        for m in re.findall(
            r"refine branch: \(([-\d.]+),([-\d.]+)\) -> \(([-\d.]+),([-\d.]+)\)\s+"
            r"J ([\d.eE+-]+) -> ([\d.eE+-]+)\s+a1=([\d.eE+-]+)\s+refine evals=(\d+)",
            head)
    ]
    coh = [
        {"n_rejected": int(m[1]), "cap": float(m[0]), "detail": m[2].strip()}
        for m in re.findall(r"coherence cap ([\d.]+): rejected (\d+) candidate\(s\) (.*)",
                            head)
    ]
    coh_all = [
        {"n_rejected": int(m[0]), "cap": float(m[1])}
        for m in re.findall(
            r"all (\d+) screened second-source candidate\(s\) rejected by coherence cap "
            r"([\d.]+)", head)
    ]
    kept = [
        {"level": int(m[0]), "branch": int(m[1]), "centers": m[2], "amps": m[3],
         "J": float(m[4])}
        for m in re.findall(
            r"\[level (\d+)\] kept branch (\d+): centers \[(.*?)\] amps \[(.*?)\] "
            r"J_exact=([\d.eE+-]+)", head)
    ]
    cand = re.findall(
        r"cand \(\s*([-\d.]+),\s*([-\d.]+)\) G=\s*([-\d.eE+]+) kappa=([\d.eE+-]+) "
        r"a1d=([\d.eE+-]+) dJ_pred=\s*([-\d.eE+]+) rank=(\d+)", head)
    levels = re.findall(r"\[level (\d+)\] .*?J_before=([\d.eE+-]+)", head)
    dedup = re.findall(r"deduplicated (\d+) refined branch\(es\); (\d+) distinct", head)
    return {
        "first_source_refinements": refine,
        "coherence_rejections": coh,
        "coherence_rejected_all": coh_all,
        "kept_branches_per_level": kept,
        "n_screened_candidates_logged": len(cand),
        "topk_best_candidates": [
            {"center": [float(c[0]), float(c[1])], "G": float(c[2]),
             "kappa": float(c[3]), "a_1d": float(c[4]), "dJ_pred": float(c[5]),
             "rank": int(c[6])}
            for c in cand[:8]
        ],
        "levels_seen": len(levels),
        "dedup_events": [{"removed": int(a), "remaining": int(b)} for a, b in dedup],
    }


def augment(rec, case_meta, admiss) -> None:
    """Per-stage detail, initializer detail, basin flags, final label."""
    case = rec["case"]
    d = APP / "output" / case / "full"
    true = np.asarray(rec["true_control"], dtype=np.float64)
    init = np.asarray(rec["initial_control"], dtype=np.float64)
    hist = rec["_hist"]
    bounds = rec["_stage_bounds"]

    rec["case_id"] = case_meta["id"]
    rec["category"] = case_meta["category"]

    # ------------------------------------------------ initialization accuracy
    perm, pos, amp = match_perm(init, true)
    rec["init_permutation"] = list(perm)
    rec["init_position_errors_m"] = pos.tolist()
    rec["init_amplitude_errors"] = amp.tolist()
    rec["max_init_position_error_m"] = float(pos.max())
    rec["max_init_amplitude_error"] = float(amp.max())
    t = true.reshape(-1, 3)
    g = init.reshape(-1, 3)
    vis = []
    for i, p in enumerate(perm):
        try:
            vis.append(bool(admiss.segment_in_domain(g[p, :2], t[i, :2], spacing=2.0)))
        except Exception:
            vis.append(None)
    rec["init_same_street_as_truth"] = vis
    rec["init_both_in_correct_basin"] = bool(
        pos.max() <= BASIN_M and all(v for v in vis if v is not None))
    rec["init_sources_in_basin"] = int(np.sum(pos <= BASIN_M))

    # final sources separated (not collapsed onto each other)
    f = np.asarray(rec["final_control"], dtype=np.float64).reshape(-1, 3)
    true_sep = float(np.hypot(*(t[0, :2] - t[1, :2])))
    got_sep = float(np.hypot(*(f[0, :2] - f[1, :2])))
    rec["true_separation_m"] = true_sep
    rec["final_separation_m"] = got_sep
    rec["sources_separated"] = bool(got_sep >= 0.5 * true_sep)

    # ------------------------------------------------------------- per stage
    stage_file = d / "sigma_continuation_summary.json"
    stages_detail = []
    if stage_file.exists():
        sj = json.loads(stage_file.read_text())
        prev_control = init
        for k, st in enumerate(sj.get("stages", [])):
            out_ctrl = np.asarray(st["stage_final_control"], dtype=np.float64)
            in_ctrl = np.asarray(
                st.get("stage_initial_control", prev_control), dtype=np.float64)
            p2, pos2, amp2 = match_perm(out_ctrl, true)
            row0 = 0 if k == 0 else (bounds[k - 1] if k - 1 < len(bounds) else 0)
            row1 = bounds[k] if k < len(bounds) else hist.shape[0] - 1
            seg = hist[row0:row1 + 1]
            entered = 0
            for s_i in range(t.shape[0]):
                xy = seg[:, 3 + 3 * s_i:5 + 3 * s_i]
                entered += int(sum(
                    0 if admiss.is_point_in_domain(float(q[0]), float(q[1])) else 1
                    for q in xy))
            oc = out_ctrl.reshape(-1, 3)
            feasible_end = all(
                admiss.is_point_in_domain(float(s[0]), float(s[1])) for s in oc)
            at_bound = [
                ("LOWER" if abs(s[2] - AMP_LOWER) <= AMP_TOL else
                 ("UPPER" if abs(s[2] - AMP_UPPER) <= AMP_TOL else "no"))
                for s in oc
            ]
            stages_detail.append({
                "stage_index": st["stage_index"],
                "sigma2": st["sigma2"],
                "input_control": in_ctrl.tolist(),
                "output_control": out_ctrl.tolist(),
                "J_start": st["stage_initial_objective"],
                "J_end": st["stage_final_objective"],
                "iterations": st["iterations"],
                "success": st["success"],
                "message": st["message"],
                "position_errors_m": pos2.tolist(),
                "amplitude_errors": amp2.tolist(),
                "max_position_error_m": float(pos2.max()),
                "max_amplitude_error": float(amp2.max()),
                "amplitude_at_bound": at_bound,
                "in_building_iterates": entered,
                "stage_end_feasible": bool(feasible_end),
                "history_rows": [int(row0), int(row1)],
            })
            prev_control = out_ctrl
    rec["stages_detail"] = stages_detail

    # ------------------------------------------------------ initializer stages
    rec["init_log_detail"] = parse_init_log(LOGS / f"{case}.log")
    isum = d / "sensitivity_init_summary.json"
    rec["init_summary"] = json.loads(isum.read_text()) if isum.exists() else {}
    plots = sorted((d / "sensitivity_init").glob("*.png")) if (d / "sensitivity_init").exists() else []
    rec["sensitivity_plot_files"] = [str(p.relative_to(APP)) for p in plots]

    rec["label"] = classify(rec)


def aborted_record(name: str, case_meta: dict) -> dict:
    """Stub record for a case that never produced an optimization summary."""
    log = LOGS / f"{name}.log"
    text = log.read_text(errors="replace") if log.exists() else ""
    exc = ""
    m = re.search(r"^(\w*Error|\w+\.\w+Error): (.*)$", text, re.M)
    if m:
        exc = f"{m.group(1)}: {m.group(2)}"
    secs = None
    prog = (SWEEP / "progress.log").read_text(errors="replace")
    m2 = re.search(rf"SWEEP_CASE_END \S+ {re.escape(name)} rc=(\d+) seconds=(\d+)", prog)
    rc = int(m2.group(1)) if m2 else None
    if m2:
        secs = float(m2.group(2))
    fwd = re.findall(r"forward solves so far: (\d+)", text)
    rec = {
        "case": name,
        "case_id": case_meta["id"],
        "category": case_meta["category"],
        "true_control": case_meta["true"],
        "initial_control": None,
        "final_control": None,
        "initial_objective": float("nan"),
        "final_objective": float("nan"),
        "max_position_error_m": float("nan"),
        "max_amplitude_error": float("nan"),
        "max_init_position_error_m": float("nan"),
        "max_init_amplitude_error": float("nan"),
        "iterations_total": 0,
        "stages_detail": [],
        "sources": [],
        "final_feasible": None,
        "any_iterate_entered_building": None,
        "any_complete_transit": None,
        "success": False,
        "message": exc or "run aborted before an optimization summary was written",
        "return_code": rc,
        "wall_seconds": secs,
        "aborted": True,
        "abort_exception": exc,
        "abort_stage": ("second-source selection (coherence screen)"
                        if "coherence cap" in exc else "initializer"),
        "init_summary": {},
        "init_log_detail": parse_init_log(log),
        "sensitivity_plot_files": [],
        "init_both_in_correct_basin": None,
        "label": "INITIALIZATION FAILURE",
        "feasibility": {},
        "last_forward_count": (int(fwd[-1]) if fwd else None),
    }
    return rec


def write_csv(recs, path: Path) -> None:
    cols = [
        "case_id", "case", "category", "true_control", "initial_control",
        "max_init_position_error_m", "max_init_amplitude_error",
        "initialized_exact_objective", "stage1_initial_objective",
        "final_control", "max_final_position_error_m", "max_final_amplitude_error",
        "final_objective", "iterations_by_stage", "iterations_total",
        "raw_infeasible_iterates", "any_iterate_entered_building",
        "any_complete_transit", "final_feasible", "min_final_clearance_m",
        "amplitude_on_bound", "success", "message", "wall_minutes", "label",
    ]
    def num(v, fmt=".6e"):
        try:
            if v is None:
                return ""
            f = float(v)
            return "" if np.isnan(f) else format(f, fmt)
        except (TypeError, ValueError):
            return ""

    def ctrl(v, fmt=".6f"):
        return "" if v is None else " ".join(format(float(x), fmt) for x in v)

    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in recs:
            srcs = r.get("sources") or []
            w.writerow([
                r["case_id"], r["case"], r["category"],
                ctrl(r["true_control"], "g"),
                ctrl(r["initial_control"], ".4f"),
                num(r.get("max_init_position_error_m"), ".3f"),
                num(r.get("max_init_amplitude_error"), ".4f"),
                num((r.get("init_summary") or {}).get("initial_exact_objective")),
                num(r.get("initial_objective")),
                ctrl(r["final_control"]),
                num(r.get("max_position_error_m"), ".4f"),
                num(r.get("max_amplitude_error"), ".6f"),
                num(r.get("final_objective")),
                "/".join(str(s["iterations"]) for s in r.get("stages_detail") or []),
                r.get("iterations_total"),
                (r.get("feasibility") or {}).get("raw_infeasible_iterates"),
                r.get("any_iterate_entered_building"),
                r.get("any_complete_transit"),
                r.get("final_feasible"),
                (f"{min(s['final_wall_clearance_m'] for s in srcs):.4f}"
                 if srcs else ""),
                ";".join(
                    "UPPER" if s["amplitude_at_upper_bound"] else
                    ("LOWER" if s["amplitude_at_lower_bound"] else "no")
                    for s in srcs),
                r.get("success"), r.get("message"),
                (f"{r['wall_seconds'] / 60.0:.1f}" if r.get("wall_seconds") else ""),
                r["label"],
            ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(SWEEP / "report"))
    ap.add_argument("--cases", nargs="*", default=None)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    master = build_master_log()
    meta = {c["name"]: c for c in json.loads((SWEEP / "cases.json").read_text())}
    names = args.cases or [c["name"] for c in
                           json.loads((SWEEP / "cases.json").read_text())]

    mesh_data = mdc.build_mesh_data()
    admiss = MeshAdmissibility(mesh_data)
    bidx = rrs.BuildingIndex(mesh_data)

    recs = []
    for name in names:
        d = APP / "output" / name / "full" / "optimization_summary.json"
        if not d.exists():
            rec = aborted_record(name, meta[name])
            print("\n" + "=" * 90)
            print(f"CASE  {name}")
            print("=" * 90)
            print(f"  ABORTED: {rec['message']}")
            print(f"  abort stage             : {rec['abort_stage']}")
            print(f"  return code / wall      : {rec['return_code']} / "
                  f"{(rec['wall_seconds'] or 0) / 60.0:.1f} min")
            ref = rec["init_log_detail"].get("first_source_refinements") or []
            for r0 in ref:
                print(f"  first-source refinement : ({r0['coarse'][0]:.1f},"
                      f"{r0['coarse'][1]:.1f}) -> ({r0['refined'][0]:.1f},"
                      f"{r0['refined'][1]:.1f})  J {r0['J_before']:.4e} -> "
                      f"{r0['J_after']:.4e}")
            recs.append(rec)
            continue
        rec = rrs.analyse(name, admiss, bidx, str(master), str(LOGS))
        augment(rec, meta[name], admiss)
        rrs.plot_case(rec, admiss, out / f"trajectory_{rec['case_id']}_{name}.png")
        recs.append(rec)

    write_csv(recs, out / "final_sweep_results.csv")
    payload = []
    for r in recs:
        c = {k: v for k, v in r.items() if not k.startswith("_")}
        payload.append(c)
    (out / "final_sweep_results.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n")

    print("\n\n" + "=" * 90)
    print("SWEEP SUMMARY")
    print("=" * 90)
    for r in recs:
        print(f"{r['case_id']}  {r['case']:<40} {r['label']:<30} "
              f"pos={r['max_position_error_m']:9.4f} m  "
              f"amp={r['max_amplitude_error']:.6f}  Jf={r['final_objective']:.3e}")
    n = len(recs)
    if n:
        cnt = {lab: sum(1 for r in recs if r["label"] == lab) for lab in LABELS}
        print("\nlabel counts: " + ", ".join(f"{k}={v}" for k, v in cnt.items() if v))
        print(f"initializer both sources in correct basin: "
              f"{sum(1 for r in recs if r['init_both_in_correct_basin'])}/{n}")
        print(f"raw obstacle crossing (any in-building iterate): "
              f"{sum(1 for r in recs if r['any_iterate_entered_building'])}/{n}")
        print(f"complete building transits: "
              f"{sum(1 for r in recs if r['any_complete_transit'])}/{n}")
        print(f"final feasible: {sum(1 for r in recs if r['final_feasible'])}/{n}")
    print(f"\nwrote {out / 'final_sweep_results.csv'}")
    print(f"wrote {out / 'final_sweep_results.json'}")


if __name__ == "__main__":
    main()
