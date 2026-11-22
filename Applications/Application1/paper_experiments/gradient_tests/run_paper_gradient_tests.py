"""Taylor-remainder gradient tests for the three published App1 configurations.

Gradient verification only: no initialization, center search or L-BFGS-B.

For each case the production discrete objective J_h and its RK4 discrete-adjoint
gradient are obtained from ``run_optimization.build_evaluator`` (the evaluator the
driver hands to L-BFGS-B) in PHYSICAL control coordinates
[x1, y1, A1, ..., xM, yM, AM], mode location_and_intensity, bound case
mesh_domain, feasibility mode raw -- exactly as in the source runs.  The model
settings (mesh, sensor node ids, velocity file, tfinal, sigma2, ...) are read
from each source run's resolved_driver_args.json / optimization_summary.json.

Test point z: the saved L-BFGS-B starting control of the source run
(optimization_summary.json "initial_control", = history iteration 0).

Direction q = D r / ||r||_2 with r ~ N(0, I) from numpy default_rng(seed) and
D = diag(CENTER_SCALE, CENTER_SCALE, AMPLITUDE_SCALE, ...) so that both center
and amplitude components are exercised at comparable objective scale.  Seeds
are tried from SEED0 upward; a draw is rejected (and recorded) if any
z + eps q leaves the admissible set (center outside the fluid mesh or
amplitude outside [amplitude_floor, AMPLITUDE_BOUNDS[1]]) or if
|g.q| < MIN_DIRECTIONAL_FRACTION * ||D g||_2.

    E0(eps) = |J(z + eps q) - J(z)|
    E1(eps) = |J(z + eps q) - J(z) - eps g.q|

Usage:  python run_paper_gradient_tests.py --out-dir DIR [--cases ...]
        python run_paper_gradient_tests.py --out-dir DIR --plot-only
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

def _find_app1() -> Path:
    env = os.environ.get("APP1_DIR")
    if env:
        return Path(env).resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "run_optimization.py").is_file() and (parent / "multi_source_utils.py").is_file():
            return parent
    raise SystemExit("cannot locate Applications/Application1; set APP1_DIR")


APP1 = _find_app1()
REPO = APP1.parents[1]
OUT_ROOT = APP1 / "output"
sys.path.insert(0, str(APP1))

CASES = {
    "B_L1": {
        "label": "Configuration B, layout L1 (25 boundary sensors)",
        "short_label": "B–L1: 2 sources, 25 boundary sensors",
        "run_dir": OUT_ROOT / "gn_diag_default_check_20261004_161718" / "cases"
        / "P_B_L1_boundary__gn-diag",
        "companion_none_run": OUT_ROOT / "gn_diag_default_check_20261004_161718"
        / "cases" / "P_B_L1_boundary__none",
    },
    "B_L2": {
        "label": "Configuration B, layout L2 (10 boundary + 15 interior sensors)",
        "short_label": "B–L2: 2 sources, 10 boundary + 15 interior",
        "run_dir": OUT_ROOT / "gn_diag_default_check_20261004_161718" / "cases"
        / "P_B_L2_mixed__gn-diag",
        "companion_none_run": OUT_ROOT / "gn_diag_default_check_20261004_161718"
        / "cases" / "P_B_L2_mixed__none",
    },
    "M5_S4": {
        "label": "Five-source case M5_S4 (31 sensors)",
        "short_label": "M5: 5 sources, 31 sensors",
        "run_dir": OUT_ROOT / "post_init_scaling_validation_20261004_125952"
        / "cases" / "M5_S4_gndiag",
        "companion_none_run": None,
    },
}

# Runs regenerated with paper_experiments/run_reference.py (REFERENCE_RUNS_DIR=<its
# --out-dir>) can replace the original batch folders; the "none" companion runs
# are then absent and their optional cross-check is skipped.
if os.environ.get("REFERENCE_RUNS_DIR"):
    _RR = Path(os.environ["REFERENCE_RUNS_DIR"]).resolve() / "cases"
    for _cid, _rid in (("B_L1", "restart_B_L1_boundary"), ("B_L2", "restart_B_L2_mixed"),
                       ("M5_S4", "M5_S4_gndiag")):
        CASES[_cid]["run_dir"] = _RR / _rid
        CASES[_cid]["companion_none_run"] = None

EXPECTED = {
    "B_L1": {"true_control": [100.0, 210.0, 1.30, 270.0, 60.0, 0.85], "num_sensors": 25},
    "B_L2": {"true_control": [100.0, 210.0, 1.30, 270.0, 60.0, 0.85], "num_sensors": 25},
    "M5_S4": {
        "true_control": [110.0, 660.0, 0.90, 270.0, 600.0, 1.35, 200.0, 425.0, 0.70,
                         100.0, 210.0, 1.60, 275.0, 120.0, 1.10],
        "num_sensors": 31,
    },
}

# eps_k = 10^(-k/2), k = 0..16  (1 down to 1e-8, half-decade ratio)
EPS = np.array([10.0 ** (-k / 2.0) for k in range(17)], dtype=np.float64)
CENTER_SCALE = 10.0       # metres per unit eps in each center coordinate
AMPLITUDE_SCALE = 0.1     # amplitude units per unit eps
SEED0 = 0
MIN_DIRECTIONAL_FRACTION = 0.1
MAX_DRAWS = 200


def log(msg: str) -> None:
    print(msg, flush=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def recover_settings(case_id: str) -> dict:
    spec = CASES[case_id]
    run = spec["run_dir"]
    args = load_json(run / "resolved_driver_args.json")
    summ = load_json(run / "full" / "optimization_summary.json")
    with open(run / "full" / "optimization_history.csv", encoding="utf-8") as fh:
        row0 = next(csv.DictReader(fh))
    instr_path = run / "run_instrumentation.json"
    instr = load_json(instr_path) if instr_path.exists() else None
    rec = {
        "case_id": case_id,
        "run_dir": str(run),
        "driver_args": args,
        "summary_pde": summ["pde"],
        "summary_observation": summ["observation"],
        "initial_control": summ["initial_control"],
        "input_initial_control": summ.get("input_initial_control"),
        "true_control": summ["true_control"],
        "final_control": summ["final_control"],
        "recorded_initial_objective": summ["initial_objective"],
        "recorded_history_iter0_objective": float(row0["objective"]),
        "recorded_history_iter0_gradient_norm": float(row0["gradient_norm"]),
        "amplitude_floor": summ["amplitude_floor"],
        "measurement_boundaries": summ["measurement_boundaries"],
        "feasibility_mode": args["feasibility_mode"],
        "bound_case": args["bound_case"],
        "mode": args["mode"],
        "num_sources": summ["num_sources"],
        "target_sigma2": summ["target_sigma2"],
        "sigma2_continuation": summ["sigma2_continuation"],
        "building_boundary_condition": summ.get("building_boundary_condition"),
        "instrumentation_argv": instr.get("driver_argv") if instr else None,
    }
    comp = spec.get("companion_none_run")
    if comp is not None and (Path(comp) / "full" / "optimization_history.csv").exists():
        with open(Path(comp) / "full" / "optimization_history.csv", encoding="utf-8") as fh:
            crow = next(csv.DictReader(fh))
        rec["companion_none_run"] = str(comp)
        rec["companion_none_iter0_gradient_norm_physical"] = float(crow["gradient_norm"])
        rec["companion_none_iter0_objective"] = float(crow["objective"])
    return rec


def run_case(case_id: str, out_dir: Path) -> dict:
    import run_optimization as ro
    from mesh_admissibility import MeshAdmissibility
    from multi_source_utils import AMPLITUDE_BOUNDS, reshape_control

    t0 = time.time()
    rec = recover_settings(case_id)
    a = rec["driver_args"]
    exp = EXPECTED[case_id]

    # --- verify against the brief -------------------------------------------
    checks = {}
    checks["true_control_matches_brief"] = bool(
        np.allclose(rec["true_control"], exp["true_control"], atol=1e-12)
    )
    checks["num_sensors_matches_brief"] = (
        rec["summary_observation"]["num_sensors"] == exp["num_sensors"]
    )
    checks["sigma2_is_400"] = float(rec["summary_pde"]["sigma2"]) == 400.0
    checks["no_sigma2_continuation"] = rec["sigma2_continuation"] is None
    checks["mode_location_and_intensity"] = a["mode"] == "location_and_intensity"
    checks["observation_mode_nodes"] = a["observation_mode"] == "nodes"
    checks["sensor_weights_equal_1"] = (
        a["sensor_weights"] == "equal" and float(a["sensor_equal_weight"]) == 1.0
    )
    checks["constant_velocity_false"] = not a["constant_velocity"]
    log(f"[{case_id}] brief/record checks: {checks}")
    if not all(checks.values()):
        raise RuntimeError(f"{case_id}: record does not match brief: {checks}")

    # --- rebuild the model exactly as the driver did -----------------------
    if a["tfinal"] is not None:
        ro.PDE["tfinal"] = float(a["tfinal"])
    mesh_data = ro.configure_measurement_boundaries(
        ro.build_mesh_data(), a["measurement_boundaries"]
    )
    obs_args = argparse.Namespace(
        observation_mode=a["observation_mode"],
        sensor_node_ids=a["sensor_node_ids"],
        sensor_coordinates=a["sensor_coordinates"],
        sensor_weights=a["sensor_weights"],
        sensor_equal_weight=a["sensor_equal_weight"],
        sensor_snap_tolerance=a["sensor_snap_tolerance"],
    )
    mesh_data = ro.configure_observation(mesh_data, obs_args)
    velocity_nodes, velocity_summary = ro.load_velocity_field_file(
        a["velocity_field_file"], mesh_data
    )
    ro.PDE["velocity_nodes"] = velocity_nodes
    ro.PDE["velocity_field"] = velocity_summary
    admissibility = MeshAdmissibility(mesh_data)

    obs_meta = mesh_data["observation"].metadata()
    pde_now = ro.serializable_pde_config(ro.PDE)
    rec_pde = rec["summary_pde"]
    for key in ("k", "dt", "tfinal", "jprint", "ngauss", "sigma2"):
        checks[f"pde_{key}_matches_record"] = float(pde_now[key]) == float(rec_pde[key])
    checks["velocity_mode_matches_record"] = pde_now.get("velocity_mode") == rec_pde.get("velocity_mode")
    checks["velocity_shape_matches_record"] = (
        list(velocity_summary["velocity_nodes_shape"])
        == list(rec_pde["velocity_field"]["velocity_nodes_shape"])
    )
    checks["mesh_nodes_match_record"] = (
        len(mesh_data["coord"]) == rec_pde["velocity_field"]["coords_shape"][0]
    )
    checks["sensor_ids_match_record"] = (
        list(obs_meta["sensor_node_ids"]) == list(rec["summary_observation"]["sensor_node_ids"])
    )
    checks["sensor_coords_match_record"] = bool(
        np.allclose(obs_meta["sensor_coordinates"],
                    rec["summary_observation"]["sensor_coordinates"], atol=1e-9)
    )
    checks["sensor_weights_match_record"] = (
        list(obs_meta["sensor_weights"]) == list(rec["summary_observation"]["sensor_weights"])
    )
    log(f"[{case_id}] model checks: {checks}")
    if not all(checks.values()):
        raise RuntimeError(f"{case_id}: rebuilt model differs from record: {checks}")

    z = np.asarray(rec["initial_control"], dtype=np.float64)
    true_control = np.asarray(rec["true_control"], dtype=np.float64)
    M = int(rec["num_sources"])
    amp_lo = float(rec["amplitude_floor"]) if rec["amplitude_floor"] is not None else AMPLITUDE_BOUNDS[0]
    amp_hi = float(AMPLITUDE_BOUNDS[1])

    # Synthetic data exactly as the driver: forward solve at the true control.
    true_state = ro.gt.run_forward(mesh_data, ro.PDE, true_control)
    feas = ro.new_feasibility_stats(a["feasibility_mode"])
    _cache, _hist, objective, gradient, _cb = ro.build_evaluator(
        mesh_data, true_state, a["mode"], a["bound_case"], {}, z.copy(),
        admissibility, feasibility_mode=a["feasibility_mode"], feasibility_stats=feas,
    )
    j0 = float(objective(z.copy()))
    g = np.asarray(gradient(z.copy()), dtype=np.float64)
    gnorm = float(np.linalg.norm(g))
    log(f"[{case_id}] J(z) = {j0:.16e}  (recorded {rec['recorded_initial_objective']:.16e})")
    log(f"[{case_id}] ||g|| = {gnorm:.16e}")
    log(f"[{case_id}] g = {np.array2string(g, precision=10)}")
    repro = {
        "J_z_recomputed": j0,
        "J_z_recorded": rec["recorded_initial_objective"],
        "J_z_rel_diff": abs(j0 - rec["recorded_initial_objective"]) / abs(rec["recorded_initial_objective"]),
        "grad_norm_recomputed": gnorm,
    }
    if "companion_none_iter0_gradient_norm_physical" in rec:
        ref = rec["companion_none_iter0_gradient_norm_physical"]
        repro["grad_norm_recorded_physical_none_run"] = ref
        repro["grad_norm_rel_diff_vs_none_run"] = abs(gnorm - ref) / ref
    log(f"[{case_id}] reproduction: {repro}")
    if repro["J_z_rel_diff"] > 1e-10:
        raise RuntimeError(f"{case_id}: J(z) does not reproduce the record: {repro}")

    # --- reproducible admissible direction ---------------------------------
    scale = np.tile([CENTER_SCALE, CENTER_SCALE, AMPLITUDE_SCALE], M)
    dg_norm = float(np.linalg.norm(scale * g))

    def admissible(c: np.ndarray) -> bool:
        if not admissibility.validate_sources_in_domain(c):
            return False
        amps = reshape_control(c)[:, 2]
        return bool(np.all(amps >= amp_lo) and np.all(amps <= amp_hi))

    rejected = []
    q = r = None
    seed_used = None
    for seed in range(SEED0, SEED0 + MAX_DRAWS):
        rr = np.random.default_rng(seed).normal(size=z.size)
        rr /= np.linalg.norm(rr)
        qq = scale * rr
        bad_eps = [float(e) for e in EPS if not admissible(z + e * qq)]
        gq = float(g @ qq)
        if bad_eps:
            rejected.append({"seed": seed, "reason": "inadmissible", "eps": bad_eps})
            continue
        if abs(gq) < MIN_DIRECTIONAL_FRACTION * dg_norm:
            rejected.append({"seed": seed, "reason": "small |g.q|", "g_dot_q": gq,
                             "threshold": MIN_DIRECTIONAL_FRACTION * dg_norm})
            continue
        q, r, seed_used = qq, rr, seed
        break
    if q is None:
        raise RuntimeError(f"{case_id}: no admissible direction in {MAX_DRAWS} draws")
    gq = float(g @ q)
    center_mask = np.tile([True, True, False], M)
    gq_center = float(g[center_mask] @ q[center_mask])
    gq_amp = float(g[~center_mask] @ q[~center_mask])
    log(f"[{case_id}] seed={seed_used} rejected={len(rejected)} g.q={gq:.6e} "
        f"(centers {gq_center:.6e}, amplitudes {gq_amp:.6e})")

    # --- Taylor sweep -------------------------------------------------------
    rows = []
    for e in EPS:
        ze = z + e * q
        je = float(objective(ze.copy()))
        diff = je - j0
        e0 = abs(diff)
        e1 = abs(diff - e * gq)
        rows.append({"eps": float(e), "J_eps": je, "J_eps_minus_J": diff,
                     "E0": e0, "E1": e1, "E0_over_J": e0 / j0, "E1_over_J": e1 / j0})
        log(f"[{case_id}] eps={e:9.3e} J={je:.16e} E0={e0:.6e} E1={e1:.6e}")
    for i, row in enumerate(rows):
        if i == 0:
            row["E0_rate"] = row["E1_rate"] = None
            continue
        p = rows[i - 1]
        ln = np.log(p["eps"] / row["eps"])
        row["E0_rate"] = float(np.log(p["E0"] / row["E0"]) / ln) if row["E0"] > 0 and p["E0"] > 0 else None
        row["E1_rate"] = float(np.log(p["E1"] / row["E1"]) / ln) if row["E1"] > 0 and p["E1"] > 0 else None

    analysis = analyse(rows, j0, gq)
    log(f"[{case_id}] analysis: {json.dumps(analysis)}")
    if feas["raw_infeasible_evaluations"] or feas["invalid_evaluations"]:
        log(f"[{case_id}] WARNING feasibility stats: {feas}")

    result = {
        "case_id": case_id,
        "label": CASES[case_id]["label"],
        "short_label": CASES[case_id]["short_label"],
        "source_run": rec["run_dir"],
        "recovered_settings": rec,
        "checks": checks,
        "reproduction": repro,
        "rebuilt_pde": pde_now,
        "rebuilt_observation": obs_meta,
        "objective": {"eta1": ro.ETA1, "lambda_reg": ro.LAMBDA_REG,
                      "objective_time_rule": ro.OBJECTIVE_TIME_RULE,
                      "gradient_time_rule_arg": ro.GRADIENT_TIME_RULE,
                      "evaluator": "run_optimization.build_evaluator (production)",
                      "feasibility_mode": a["feasibility_mode"],
                      "bound_case": a["bound_case"], "mode": a["mode"]},
        "control_ordering": "[x1, y1, A1, ..., xM, yM, AM] physical (m, m, amplitude)",
        "z": z.tolist(),
        "true_control": true_control.tolist(),
        "J_z": j0,
        "gradient": g.tolist(),
        "gradient_norm": gnorm,
        "direction": {
            "q": q.tolist(), "r_unit": r.tolist(), "seed": seed_used,
            "rng": "numpy.random.default_rng(seed).normal(size=3M); r /= ||r||_2",
            "scaling_D": scale.tolist(),
            "construction": "q = D r",
            "center_scale": CENTER_SCALE, "amplitude_scale": AMPLITUDE_SCALE,
            "min_directional_fraction": MIN_DIRECTIONAL_FRACTION,
            "rejected_draws": rejected,
            "admissible_set": f"centers in fluid mesh; amplitudes in [{amp_lo}, {amp_hi}]",
            "max_center_step_m": float(EPS[0] * np.max(np.abs(q[center_mask]))),
            "max_amplitude_step": float(EPS[0] * np.max(np.abs(q[~center_mask]))),
        },
        "g_dot_q": gq, "g_dot_q_centers": gq_center, "g_dot_q_amplitudes": gq_amp,
        "eps": EPS.tolist(),
        "rows": rows,
        "analysis": analysis,
        "feasibility_stats": feas,
        "elapsed_s": time.time() - t0,
        "python": sys.version, "platform": platform.platform(),
    }
    case_dir = out_dir / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with open(case_dir / "taylor_table.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    np.savez(case_dir / "control_direction.npz", z=z, q=q, r=r, D=scale, g=g,
             eps=EPS, E0=np.array([x["E0"] for x in rows]),
             E1=np.array([x["E1"] for x in rows]), J_z=j0, seed=seed_used)
    return result


def analyse(rows: list[dict], j0: float, gq: float) -> dict:
    """Resolved range = longest contiguous run of consecutive E1 rates in
    [1.8, 2.2]; first-order range likewise for E0 rates in [0.9, 1.1].
    Points below the resolved E1 range are classified, not discarded."""
    def longest(rates, lo, hi):
        best, cur = (None, None), None
        for i, rt in enumerate(rates):
            ok = rt is not None and lo <= rt <= hi
            if ok:
                cur = (cur[0], i) if cur else (i, i)
                if best[0] is None or cur[1] - cur[0] > best[1] - best[0]:
                    best = cur
            else:
                cur = None
        return best

    eps = np.array([x["eps"] for x in rows])
    e0 = np.array([x["E0"] for x in rows])
    e1 = np.array([x["E1"] for x in rows])
    r1 = [x["E1_rate"] for x in rows]
    r0 = [x["E0_rate"] for x in rows]

    def fit(arr, i0, i1):
        # rates index i compares points i-1 and i
        sl = slice(i0 - 1, i1 + 1)
        return float(np.polyfit(np.log(eps[sl]), np.log(arr[sl]), 1)[0])

    out = {}
    a, b = longest(r1, 1.8, 2.2)
    if a is not None:
        out["E1_resolved_eps_range"] = [float(eps[a - 1]), float(eps[b])]
        out["E1_resolved_num_points"] = int(b - a + 2)
        out["E1_fitted_slope_resolved"] = fit(e1, a, b)
        below = [{"eps": float(eps[i]), "E1": float(e1[i]), "E1_rate": r1[i]}
                 for i in range(b + 1, len(rows))]
        above = [{"eps": float(eps[i]), "E1": float(e1[i]), "E1_rate": r1[i]}
                 for i in range(0, a - 1)]
        out["E1_points_below_resolved_range"] = below
        out["E1_points_above_resolved_range"] = above
        out["cancellation_onset_eps"] = float(eps[b + 1]) if b + 1 < len(rows) else None
    else:
        out["E1_resolved_eps_range"] = None
    a0, b0 = longest(r0, 0.9, 1.1)
    if a0 is not None:
        out["E0_first_order_eps_range"] = [float(eps[a0 - 1]), float(eps[b0])]
        out["E0_fitted_slope"] = fit(e0, a0, b0)
    out["E0_rate_all"] = r0
    out["E1_rate_all"] = r1
    out["roundoff_floor_estimate"] = float(np.finfo(float).eps * abs(j0))
    out["roundoff_floor_note"] = "machine eps * |J(z)|; E1 cannot resolve below a small multiple of this"
    out["second_order_observed"] = bool(
        a is not None and (b - a + 2) >= 4 and abs(out["E1_fitted_slope_resolved"] - 2.0) <= 0.1
    )
    out["first_order_observed"] = bool(
        a0 is not None and abs(out["E0_fitted_slope"] - 1.0) <= 0.1
    )
    return out


def plot(out_dir: Path, case_ids: list[str]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 12, "axes.labelsize": 13, "axes.titlesize": 12,
                         "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11,
                         "mathtext.fontset": "cm"})
    results = [load_json(out_dir / "cases" / c / "result.json") for c in case_ids]
    figdir = out_dir / "figures"
    figdir.mkdir(exist_ok=True)
    c0, c1 = "#1f5fa8", "#c4452b"

    def draw(ax, res, ylabel=True):
        eps = np.array(res["eps"])
        n0 = np.array([x["E0_over_J"] for x in res["rows"]])
        n1 = np.array([x["E1_over_J"] for x in res["rows"]])
        h0 = ax.loglog(eps, n0, "o-", color=c0, ms=5, lw=1.6, label=r"$E_0(\varepsilon)/J_h(z)$")[0]
        h1 = ax.loglog(eps, n1, "s-", color=c1, ms=5, lw=1.6, label=r"$E_1(\varepsilon)/J_h(z)$")[0]
        an = res["analysis"]
        # slope-1 reference: alongside E0 over its smallest-eps decades (first-order
        # regime); slope-2 reference: alongside E1 over the middle of its range.
        e_1 = eps[-9:]
        r1 = 0.15 * n0[-1] * (e_1 / eps[-1])
        e_2 = eps[3:12]
        r2 = 0.08 * n1[3] * (e_2 / eps[3]) ** 2
        h2 = ax.loglog(e_1, r1, "--", color="0.35", lw=1.3, label="slope 1")[0]
        h3 = ax.loglog(e_2, r2, ":", color="0.1", lw=1.6, label="slope 2")[0]
        ax.set_xlabel(r"$\varepsilon$")
        if ylabel:
            ax.set_ylabel("normalized Taylor remainder")
        ax.set_title(res["short_label"])
        ax.grid(True, which="major", ls=":", lw=0.6, color="0.75")
        return [h0, h1, h2, h3]

    fig, axes = plt.subplots(1, len(results), figsize=(5.0 * len(results), 4.6), sharey=False)
    axes = np.atleast_1d(axes)
    for k, (ax, res) in enumerate(zip(axes, results)):
        handles = draw(ax, res, ylabel=(k == 0))
    fig.legend(handles, [h.get_label() for h in handles], loc="lower center",
               ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext in ("pdf", "png"):
        fig.savefig(figdir / f"taylor_gradient_tests_3panel.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    for res in results:
        fig, ax = plt.subplots(figsize=(5.4, 4.6))
        handles = draw(ax, res)
        ax.legend(handles, [h.get_label() for h in handles], loc="lower right", frameon=False)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(figdir / f"taylor_gradient_test_{res['case_id']}.{ext}", dpi=300,
                        bbox_inches="tight")
        plt.close(fig)

    # combined tables
    with open(out_dir / "all_cases_taylor_table.csv", "w", newline="", encoding="utf-8") as fh:
        w = None
        for res in results:
            for row in res["rows"]:
                d = {"case_id": res["case_id"], "J_z": res["J_z"], "g_dot_q": res["g_dot_q"], **row}
                if w is None:
                    w = csv.DictWriter(fh, fieldnames=list(d.keys()))
                    w.writeheader()
                w.writerow(d)
    summary = {res["case_id"]: {
        "source_run": res["source_run"], "J_z": res["J_z"], "gradient_norm": res["gradient_norm"],
        "g_dot_q": res["g_dot_q"], "g_dot_q_centers": res["g_dot_q_centers"],
        "g_dot_q_amplitudes": res["g_dot_q_amplitudes"], "seed": res["direction"]["seed"],
        "analysis": {k: v for k, v in res["analysis"].items()
                     if k not in ("E0_rate_all", "E1_rate_all")},
    } for res in results}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cases", nargs="+", default=list(CASES))
    p.add_argument("--plot-only", action="store_true")
    args = p.parse_args()
    out_dir = Path(args.out_dir).resolve()
    if not args.plot_only:
        for c in args.cases:
            log(f"===== case {c} start {time.strftime('%Y-%m-%d %H:%M:%S')} =====")
            res = run_case(c, out_dir)
            log(f"===== case {c} done in {res['elapsed_s']:.1f}s =====")
    plot(out_dir, args.cases)
    log("plots written; ALL DONE")


if __name__ == "__main__":
    main()
