"""Focused validation of ``--amplitude-solver osqp`` against the enumeration solver.

Scope: the JOINT AMPLITUDE FIT ONLY.  For fixed source centers the fit is the
box-constrained QP

    min_{A_min <= a <= A_max}   0.5 a^T H a - a^T h,                        (*)
    H[i, j] = < delta_u_{c_i}, delta_u_{c_j} >_obs + lambda delta_ij,
    h[i]    = < delta_u_{c_i}, u_d >_obs,

in the repository's exact weighted boundary/time observation inner product.
Both solvers are handed byte-identical ``H`` and ``h``.

This script does NOT run sensitivity initialization over the mesh, a center
search, a sweep, or the L-BFGS-B reconstruction.  Part B pays for exactly THREE
production forward solves (one for the data ``u_d``, one per fixed-center unit
Gaussian) and caches the three extracted observation traces, so re-runs pay
nothing.  Part D is a plumbing check on a deliberately tiny 2-candidate,
short-``tfinal`` configuration; it proves the selection reaches every joint-fit
call path and is not a physics result.

Checks:
  A. Solver-level agreement on random convex QPs, including rank-deficient H:
     objective, bound feasibility, projected-gradient optimality.
  A2. The 3^n source-count limit is confined to the enumeration solver; OSQP
     solves n = 8, 9, 10 (the intended amplitude-problem sizes).
  A3. Badly scaled amplitude QPs of the REAL H = T'T, h = T'u_d structure
     (response magnitudes spanning 1e-12..1e5, 30% near-duplicate responses,
     noiseless and 1% noise): every ACCEPTED solution is first-order optimal
     and no worse than enumeration, and every refusal is loud and confined to
     extreme conditioning.
  B. The REAL Application1 two-source case: one common (H, h) solved both ways,
     compared on amplitudes, the common quadratic (*), the full tracking
     objective, bound violation, projected-gradient optimality, cond(H), and
     the OSQP termination status.  Run for both the raw amplitude bounds
     (0.0, 2.0) and the production floored bounds (0.1, 2.0).
  C. Failure handling: a bad solver name is rejected, and the missing-OSQP path
     produces an actionable error (simulated by hiding the import) rather than
     a silent fallback to enumeration.
  D. CLI plumbing: ``--amplitude-solver`` defaults to osqp, accepts both
     choices, rejects garbage, and the selection reaches ``_solve_box_qp`` from
     the level expansion, the variable-projected center-refinement trials, and
     the second-source re-selection.

Usage:
    PY="python3"
    cd "Applications/Application1/test scripts"
    "$PY" test_amplitude_solver_osqp.py

    # optional
    "$PY" test_amplitude_solver_osqp.py --no-cache     # ignore the trace cache
    "$PY" test_amplitude_solver_osqp.py --skip-plumbing
"""

from __future__ import annotations

import argparse
import builtins
import json
import os as _os
import subprocess
import sys as _sys
import time
from pathlib import Path

import numpy as np

_APP1_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
if _APP1_DIR not in _sys.path:
    _sys.path.insert(0, _APP1_DIR)

# run_optimization first: it appends Src/Objective (gradient_test) to sys.path.
import run_optimization as ro
import gradient_test as gt
import sensitivity_initialization as sinit
from mesh_admissibility import MeshAdmissibility
from mesh_data_config import build_mesh_data
from multi_source_utils import AMPLITUDE_BOUNDS, TRUE_CONTROL

APP_DIR = Path(_APP1_DIR)
PYTHON = _sys.executable
REPORT_DIR = APP_DIR / "output" / "amplitude_solver_validation"
CACHE_PATH = REPORT_DIR / "two_source_unit_traces.npz"

# ---------------------------------------------------------------------------
# The two-source case under test: the repository's canonical Application1
# reference control (multi_source_utils.TRUE_CONTROL) observed on the
# production measurement-boundary preset recorded in the saved sweep summaries
# (output/final_sweep_*/full/optimization_summary.json -> left+right+bottom,
# amplitude_floor 0.1, bound_case mesh_domain).
# ---------------------------------------------------------------------------
MEASUREMENT_BOUNDARIES = ["left", "right", "bottom"]
PRODUCTION_AMPLITUDE_FLOOR = 0.1

# Explicit numerical tolerances. The gates are the OBJECTIVE and the
# OPTIMALITY residual, because H can be ill-conditioned (two coherent
# observation traces) and then the minimizer need not be unique in a.
TOL_OBJ_REL = 1.0e-9          # relative difference in the common quadratic (*)
TOL_TRACK_REL = 1.0e-9        # relative difference in the full tracking objective
TOL_BOUND_ABS = 1.0e-12       # absolute bound violation allowed of either solver
TOL_OPT_REL = 1.0e-8          # projected-gradient residual / max(1, ||h||_inf)
TOL_AMP_ABS = 1.0e-6          # reported; only a gate when cond(H) is benign
COND_WELL_POSED = 1.0e10      # above this, amplitudes may legitimately differ

FAILURES: list[str] = []
RECORDS: dict = {}


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def check(condition, message) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {message}")
    if not condition:
        FAILURES.append(message)
    return bool(condition)


# ---------------------------------------------------------------------------
# Shared metrics
# ---------------------------------------------------------------------------
def quadratic(H: np.ndarray, h: np.ndarray, a: np.ndarray) -> float:
    """The common objective 0.5 a^T H a - a^T h of (*)."""
    return float(0.5 * a @ H @ a - h @ a)


def bound_violation(a: np.ndarray, lo: float, hi: float) -> float:
    return float(max(np.max(lo - a), np.max(a - hi), 0.0))


def projected_gradient_residual(
    H: np.ndarray, h: np.ndarray, a: np.ndarray, lo: float, hi: float
) -> float:
    """||a - P_[lo,hi](a - grad)||_inf, the box-constrained first-order residual.

    Zero exactly at a KKT point of (*): an interior component needs a zero
    gradient, a component at ``lo`` needs grad >= 0, one at ``hi`` needs
    grad <= 0.
    """
    grad = H @ a - h
    return float(np.max(np.abs(a - np.clip(a - grad, lo, hi))))


def solve_both(H: np.ndarray, h: np.ndarray, lo: float, hi: float) -> dict:
    """Solve the SAME (H, h, lo, hi) with both solvers and measure everything."""
    b = -h  # _solve_box_qp minimizes 0.5 a^T G a + b^T a
    t0 = time.perf_counter()
    a_enum = sinit._solve_box_qp(
        H, b, lo, hi, solver=sinit.AMPLITUDE_SOLVER_ENUMERATION
    )
    t_enum = time.perf_counter() - t0
    t0 = time.perf_counter()
    a_osqp = sinit._solve_box_qp(H, b, lo, hi, solver=sinit.AMPLITUDE_SOLVER_OSQP)
    t_osqp = time.perf_counter() - t0

    q_enum = quadratic(H, h, a_enum)
    q_osqp = quadratic(H, h, a_osqp)
    scale_q = max(1.0, abs(q_enum))
    scale_g = max(1.0, float(np.max(np.abs(h))))
    return {
        "n": int(h.size),
        "bounds": [float(lo), float(hi)],
        "cond_H": float(np.linalg.cond(H)),
        "amplitudes_enumeration": [float(v) for v in a_enum],
        "amplitudes_osqp": [float(v) for v in a_osqp],
        "amplitude_max_abs_diff": float(np.max(np.abs(a_enum - a_osqp))),
        "quadratic_enumeration": q_enum,
        "quadratic_osqp": q_osqp,
        "quadratic_abs_diff": abs(q_osqp - q_enum),
        "quadratic_rel_diff": abs(q_osqp - q_enum) / scale_q,
        "bound_violation_enumeration": bound_violation(a_enum, lo, hi),
        "bound_violation_osqp": bound_violation(a_osqp, lo, hi),
        "optimality_enumeration": projected_gradient_residual(H, h, a_enum, lo, hi),
        "optimality_osqp": projected_gradient_residual(H, h, a_osqp, lo, hi),
        "optimality_scale": scale_g,
        "seconds_enumeration": t_enum,
        "seconds_osqp": t_osqp,
        "_a_enum": a_enum,
        "_a_osqp": a_osqp,
    }


def gate_comparison(result: dict, label: str) -> None:
    """Apply the documented tolerances to one solve_both() result."""
    check(
        result["quadratic_rel_diff"] <= TOL_OBJ_REL,
        f"{label}: common quadratic agrees "
        f"(rel diff {result['quadratic_rel_diff']:.3e} <= {TOL_OBJ_REL:.0e})",
    )
    check(
        result["bound_violation_osqp"] <= TOL_BOUND_ABS,
        f"{label}: OSQP respects the amplitude bounds "
        f"(violation {result['bound_violation_osqp']:.3e} <= {TOL_BOUND_ABS:.0e})",
    )
    check(
        result["bound_violation_enumeration"] <= TOL_BOUND_ABS,
        f"{label}: enumeration respects the amplitude bounds "
        f"(violation {result['bound_violation_enumeration']:.3e})",
    )
    opt_rel = result["optimality_osqp"] / result["optimality_scale"]
    opt_rel_enum = result["optimality_enumeration"] / result["optimality_scale"]
    check(
        opt_rel <= TOL_OPT_REL,
        f"{label}: OSQP is first-order optimal on the box "
        f"(scaled residual {opt_rel:.3e} <= {TOL_OPT_REL:.0e})",
    )
    check(
        opt_rel_enum <= TOL_OPT_REL,
        f"{label}: enumeration is first-order optimal on the box "
        f"(scaled residual {opt_rel_enum:.3e})",
    )
    if result["cond_H"] <= COND_WELL_POSED:
        check(
            result["amplitude_max_abs_diff"] <= TOL_AMP_ABS,
            f"{label}: amplitudes agree "
            f"(max |da| {result['amplitude_max_abs_diff']:.3e} <= "
            f"{TOL_AMP_ABS:.0e}; cond(H) = {result['cond_H']:.3e})",
        )
    else:
        print(
            f"  [NOTE] {label}: cond(H) = {result['cond_H']:.3e} > "
            f"{COND_WELL_POSED:.0e}; the minimizer in a need not be unique, so "
            f"the amplitude difference ({result['amplitude_max_abs_diff']:.3e}) "
            "is reported but not gated. Objective and optimality are gated."
        )


# ---------------------------------------------------------------------------
# A. Solver-level agreement on random convex QPs
# ---------------------------------------------------------------------------
def test_random_qps() -> dict:
    section("A. Random convex box QPs: enumeration vs OSQP")
    rng = np.random.default_rng(20260930)
    worst = {
        "quadratic_rel_diff": 0.0,
        "amplitude_max_abs_diff": 0.0,
        "bound_violation_osqp": 0.0,
        "optimality_osqp_scaled": 0.0,
    }
    n_trials = 0
    n_rank_deficient = 0
    for _ in range(240):
        n = int(rng.integers(1, 7))
        lo, hi = 0.1, 2.0
        if rng.random() < 0.25:
            # Rank-deficient H: coherent observation traces. The QP is still
            # convex but the minimizer in a may be non-unique.
            rank = max(1, n - 1)
            A = rng.standard_normal((rank, n))
            n_rank_deficient += 1
        else:
            A = rng.standard_normal((n + 2, n))
        H = A.T @ A
        h = rng.standard_normal(n) * 2.0
        res = solve_both(H, h, lo, hi)
        n_trials += 1
        worst["quadratic_rel_diff"] = max(
            worst["quadratic_rel_diff"], res["quadratic_rel_diff"]
        )
        worst["amplitude_max_abs_diff"] = max(
            worst["amplitude_max_abs_diff"], res["amplitude_max_abs_diff"]
        )
        worst["bound_violation_osqp"] = max(
            worst["bound_violation_osqp"], res["bound_violation_osqp"]
        )
        worst["optimality_osqp_scaled"] = max(
            worst["optimality_osqp_scaled"],
            res["optimality_osqp"] / res["optimality_scale"],
        )
    print(f"  {n_trials} random QPs ({n_rank_deficient} rank-deficient), n in [1, 6]")
    for key, value in worst.items():
        print(f"    worst {key}: {value:.3e}")
    check(
        worst["quadratic_rel_diff"] <= TOL_OBJ_REL,
        f"random QPs: worst relative objective difference "
        f"{worst['quadratic_rel_diff']:.3e} <= {TOL_OBJ_REL:.0e}",
    )
    check(
        worst["bound_violation_osqp"] <= TOL_BOUND_ABS,
        f"random QPs: worst OSQP bound violation "
        f"{worst['bound_violation_osqp']:.3e} <= {TOL_BOUND_ABS:.0e}",
    )
    check(
        worst["optimality_osqp_scaled"] <= TOL_OPT_REL,
        f"random QPs: worst OSQP scaled optimality residual "
        f"{worst['optimality_osqp_scaled']:.3e} <= {TOL_OPT_REL:.0e}",
    )
    return {"n_trials": n_trials, "n_rank_deficient": n_rank_deficient, "worst": worst}


# ---------------------------------------------------------------------------
# A3. Badly scaled Gram matrices with the REAL H = T'T, h = T'u_d structure
# ---------------------------------------------------------------------------
#: Acceptance thresholds for the ill-conditioning stress batch. These bound the
#: residual of ACCEPTED solutions and the rate of loud refusals; they are not
#: claims about the production case (cond(H) = 73 there).
STRESS_MAX_ACCEPTED_KKT = 1.0e-8
STRESS_MAX_REFUSAL_RATE = 0.02
STRESS_MIN_REFUSAL_COND = 1.0e18
#: Objective-gap threshold for the STRESS batch only -- deliberately NOT the
#: production TOL_OBJ_REL. cond(H) reaches ~1e35 here, where a 1e-9 relative
#: objective gap is numerical noise rather than a worse answer. Measured over
#: 4498 stress problems (noiseless, 1% and 5% noise): worst gap 1.6e-09, p99.9
#: 1.2e-09, median ~1e-13. 1e-7 is ~60x above that worst case.
STRESS_MAX_OBJECTIVE_GAP = 1.0e-7


def test_illconditioned_stress() -> dict:
    """OSQP on badly scaled amplitude QPs of the real structure.

    Builds ``n`` synthetic unit responses whose magnitudes span 1e-12 to 1e5 --
    an unobserved source sitting next to a strongly observed one -- forms
    ``H = T'T`` and ``h = T'u_d`` with ``u_d = T a_true``, and (30% of trials)
    makes two responses near-duplicates so H is numerically rank deficient.
    This is the regime the production `tfinal = 8 s` case lives in, pushed much
    further than any real configuration.

    What is asserted: every ACCEPTED solution is first-order optimal and no
    worse than enumeration's objective, refusals are loud and confined to
    extreme conditioning, and the refusal rate stays low.
    """
    section("A3. Ill-conditioned stress batch (real H = T'T, h = T'u_d structure)")
    record = {}
    for label, noise in (("noiseless", 0.0), ("1% noise", 0.01)):
        rng = np.random.default_rng(23)
        refusals, residuals, accepted_conds, objective_gaps = [], [], [], []
        trials = 400
        t0 = time.perf_counter()
        for _ in range(trials):
            n = int(rng.integers(2, 11))
            m = 60
            scales = 10.0 ** rng.uniform(-12, 5, size=n)
            T = rng.standard_normal((m, n)) * scales
            if rng.random() < 0.3:
                T[:, -1] = T[:, 0] * (1.0 + 1.0e-14)  # near-duplicate responses
            a_true = rng.uniform(0.1, 2.0, size=n)
            ud = T @ a_true
            if noise:
                ud = ud + noise * np.linalg.norm(ud) * rng.standard_normal(m) / np.sqrt(m)
            H, h = T.T @ T, T.T @ ud
            lo, hi = 0.1, 2.0
            cond = float(np.linalg.cond(H))
            try:
                a = sinit._solve_box_qp(
                    H, -h, lo, hi, solver=sinit.AMPLITUDE_SOLVER_OSQP
                )
            except sinit.SensitivityInitializationError as exc:
                refusals.append({"n": n, "cond_H": cond, "error": str(exc)[:200]})
                continue
            accepted_conds.append(cond)
            residuals.append(
                projected_gradient_residual(H, h, a, lo, hi)
                / max(1.0, float(np.max(np.abs(h))))
            )
            if n <= sinit._ENUMERATION_MAX_SOURCES:
                a_e = sinit._solve_box_qp(
                    H, -h, lo, hi, solver=sinit.AMPLITUDE_SOLVER_ENUMERATION
                )
                q_e, q_o = quadratic(H, h, a_e), quadratic(H, h, a)
                objective_gaps.append((q_o - q_e) / max(1.0, abs(q_e)))
        seconds = time.perf_counter() - t0
        res = np.asarray(residuals)
        gaps = np.asarray(objective_gaps)
        rate = len(refusals) / trials
        entry = {
            "trials": trials,
            "seconds": seconds,
            "n_refused": len(refusals),
            "refusal_rate": rate,
            "accepted_kkt_max": float(res.max()) if res.size else None,
            "accepted_kkt_median": float(np.median(res)) if res.size else None,
            "max_cond_accepted": (
                float(np.max(accepted_conds)) if accepted_conds else None
            ),
            "worst_objective_gap_vs_enumeration": (
                float(gaps.max()) if gaps.size else None
            ),
            "refusals": refusals[:10],
        }
        record[label] = entry
        print(f"  {label}: {trials} trials in {seconds:.1f} s, "
              f"{len(refusals)} refused ({rate:.1%})")
        print(f"    accepted scaled box-KKT residual: max "
              f"{entry['accepted_kkt_max']:.3e}, median "
              f"{entry['accepted_kkt_median']:.3e}")
        print(f"    worst objective gap vs enumeration (n <= 8): "
              f"{entry['worst_objective_gap_vs_enumeration']:.3e}")
        for ref in refusals[:3]:
            print(f"    refused n={ref['n']} cond={ref['cond_H']:.2e}")

        check(
            entry["accepted_kkt_max"] <= STRESS_MAX_ACCEPTED_KKT,
            f"stress/{label}: every ACCEPTED solution is first-order optimal "
            f"(worst {entry['accepted_kkt_max']:.3e} <= "
            f"{STRESS_MAX_ACCEPTED_KKT:.0e})",
        )
        check(
            gaps.size > 0
            and entry["worst_objective_gap_vs_enumeration"]
            <= STRESS_MAX_OBJECTIVE_GAP,
            f"stress/{label}: no accepted solution is worse than enumeration "
            f"(worst gap {entry['worst_objective_gap_vs_enumeration']:.3e} <= "
            f"{STRESS_MAX_OBJECTIVE_GAP:.0e})",
        )
        check(
            rate <= STRESS_MAX_REFUSAL_RATE,
            f"stress/{label}: refusal rate {rate:.1%} <= "
            f"{STRESS_MAX_REFUSAL_RATE:.0%}",
        )
        check(
            all(r["cond_H"] >= STRESS_MIN_REFUSAL_COND for r in refusals),
            f"stress/{label}: every refusal is at cond(H) >= "
            f"{STRESS_MIN_REFUSAL_COND:.0e} (extreme ill-conditioning), "
            f"not at ordinary conditioning",
        )
        check(
            all("--amplitude-solver enumeration" in r["error"] for r in refusals),
            f"stress/{label}: every refusal is loud and names the enumeration "
            f"alternative ({len(refusals)} refusal(s))",
        )
    return record


# ---------------------------------------------------------------------------
# A2. The enumeration source-count cap is enumeration-specific
# ---------------------------------------------------------------------------
def test_source_count_limit() -> dict:
    section("A2. 3^n limit confined to enumeration; OSQP covers n = 8, 9, 10")
    rng = np.random.default_rng(7)
    record = {"enumeration_max_sources": int(sinit._ENUMERATION_MAX_SOURCES)}
    sizes = {}
    for n in (8, 9, 10):
        A = rng.standard_normal((n + 3, n))
        H = A.T @ A
        h = rng.standard_normal(n) * 2.0
        lo, hi = 0.1, 2.0
        t0 = time.perf_counter()
        a = sinit._solve_box_qp(H, -h, lo, hi, solver=sinit.AMPLITUDE_SOLVER_OSQP)
        seconds = time.perf_counter() - t0
        opt = projected_gradient_residual(H, h, a, lo, hi) / max(
            1.0, float(np.max(np.abs(h)))
        )
        viol = bound_violation(a, lo, hi)
        sizes[n] = {
            "seconds_osqp": seconds,
            "optimality_scaled": opt,
            "bound_violation": viol,
        }
        check(
            viol <= TOL_BOUND_ABS and opt <= TOL_OPT_REL,
            f"OSQP solves n={n} ({seconds * 1e3:.2f} ms, scaled optimality "
            f"{opt:.3e}, bound violation {viol:.3e})",
        )
        if n > sinit._ENUMERATION_MAX_SOURCES:
            raised = False
            try:
                sinit._solve_box_qp(
                    H, -h, lo, hi, solver=sinit.AMPLITUDE_SOLVER_ENUMERATION
                )
            except ValueError as exc:
                raised = "osqp" in str(exc)
            check(raised, f"enumeration refuses n={n} and names the osqp solver")
    record["osqp_by_size"] = sizes
    return record


# ---------------------------------------------------------------------------
# B. The real Application1 two-source case
# ---------------------------------------------------------------------------
def build_two_source_problem(
    use_cache: bool = True, tfinal: float | None = None, cache_name: str | None = None
) -> dict:
    """Build the ONE common (H, h) of the two-source case.

    Pays three forward solves the first time (u_d plus one unit Gaussian per
    fixed center) and caches the three extracted observation traces.  No
    initialization, no center search, no optimization.

    ``tfinal`` overrides the production final time; this is used only to build
    the deliberately UNOBSERVABLE companion case of check B2.
    """
    centers = np.asarray(
        [[TRUE_CONTROL[0], TRUE_CONTROL[1]], [TRUE_CONTROL[3], TRUE_CONTROL[4]]],
        dtype=np.float64,
    )
    mesh_data = ro.configure_measurement_boundaries(
        build_mesh_data(), list(MEASUREMENT_BOUNDARIES)
    )
    # Observation configuration: the DEFAULT boundary-integral misfit (the
    # production and committed default); no observation operator is attached.
    mesh_data = ro.configure_observation(
        mesh_data, _boundary_observation_args()
    )
    pde = dict(ro.PDE)
    if tfinal is not None:
        pde["tfinal"] = float(tfinal)
    velocity_summary = None
    if ro.DEFAULT_VELOCITY_FIELD_FILE.exists():
        velocity_nodes, velocity_summary = ro.load_velocity_field_file(
            str(ro.DEFAULT_VELOCITY_FIELD_FILE), mesh_data
        )
        pde["velocity_nodes"] = velocity_nodes
        pde["velocity_field"] = velocity_summary
    ctx = sinit._ObsTraceContext(
        mesh_data, ro.ETA1, pde["dt"] * pde["jprint"], ro.OBJECTIVE_TIME_RULE
    )

    config = {
        "true_control": [float(v) for v in TRUE_CONTROL],
        "fixed_centers": centers.tolist(),
        "true_amplitudes": [float(TRUE_CONTROL[2]), float(TRUE_CONTROL[5])],
        "amplitude_bounds_raw": [float(v) for v in AMPLITUDE_BOUNDS],
        "amplitude_floor_production": PRODUCTION_AMPLITUDE_FLOOR,
        "observation_mode": "boundary",
        "measurement_boundaries": list(MEASUREMENT_BOUNDARIES),
        "selected_measurement_boundaries": list(
            mesh_data.get("selected_measurement_boundaries", ())
        ),
        "eta1": float(ro.ETA1),
        "lambda_reg": float(ro.LAMBDA_REG),
        "objective_time_rule": ro.OBJECTIVE_TIME_RULE,
        "sigma2": float(pde["sigma2"]),
        "k": float(pde["k"]),
        "dt": float(pde["dt"]),
        "tfinal": float(pde["tfinal"]),
        "jprint": int(pde["jprint"]),
        "ngauss": int(pde["ngauss"]),
        "velocity_mode": "nodal" if velocity_summary is not None else "constant",
        "n_observed_boundary_nodes": int(np.asarray(ctx.bnodes).size),
        "mesh_nodes": int(len(mesh_data["coord"])),
    }
    cache_key = json.dumps(config, sort_keys=True)
    cache_path = REPORT_DIR / (cache_name or CACHE_PATH.name)

    traces = None
    if use_cache and cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as data:
            if str(data["cache_key"]) == cache_key:
                traces = {
                    "ud": data["ud"],
                    "t1": data["t1"],
                    "t2": data["t2"],
                }
                print(f"  reusing cached unit responses: {cache_path}")
            else:
                print("  cached unit responses do not match this configuration")
    if traces is None:
        print("  running 3 fixed-center forward solves (u_d + one unit Gaussian each)")
        t0 = time.perf_counter()
        ud = ctx.extract(gt.run_forward(mesh_data, pde, np.asarray(TRUE_CONTROL)))
        unit = []
        for cx, cy in centers:
            src = sinit._gaussian_sum_source([(float(cx), float(cy), 1.0)], pde["sigma2"])
            unit.append(
                ctx.extract(
                    gt.run_forward(mesh_data, pde, sinit._DUMMY_Z, source_func=src)
                )
            )
        print(f"  3 forward solves in {time.perf_counter() - t0:.1f} s")
        traces = {"ud": ud, "t1": unit[0], "t2": unit[1]}
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, cache_key=cache_key, **traces)
        print(f"  cached unit responses: {cache_path}")

    ud = traces["ud"]
    unit_traces = [traces["t1"], traces["t2"]]

    # H and h EXACTLY as joint_refit_traces builds them.
    lam = float(ro.LAMBDA_REG)
    H = np.empty((2, 2), dtype=np.float64)
    h = np.empty(2, dtype=np.float64)
    for i in range(2):
        for j in range(i, 2):
            H[i, j] = H[j, i] = ctx.inner(unit_traces[i], unit_traces[j])
        h[i] = ctx.inner(unit_traces[i], ud)
    H = H + lam * np.eye(2)

    coherence = sinit._coherence_from_norms(
        ctx.inner(unit_traces[0], unit_traces[1]),
        ctx.norm2(unit_traces[0]),
        ctx.norm2(unit_traces[1]),
    )
    print("  fixed centers (m): "
          f"({centers[0][0]:.4f}, {centers[0][1]:.4f}) and "
          f"({centers[1][0]:.4f}, {centers[1][1]:.4f})")
    print(f"  true amplitudes: {config['true_amplitudes']}")
    print(f"  observation: boundary misfit on {config['selected_measurement_boundaries']}"
          f", eta1 = {config['eta1']:g}, time rule {config['objective_time_rule']}")
    print(f"  sigma2 = {config['sigma2']:g} m^2, k = {config['k']:g} m^2/s, "
          f"dt = {config['dt']:g} s, tfinal = {config['tfinal']:g} s, "
          f"velocity {config['velocity_mode']}")
    print(f"  lambda (Tikhonov) = {lam:g}")
    print(f"  H = {np.array2string(H, precision=12)}")
    print(f"  h = {np.array2string(h, precision=12)}")
    print(f"  cond(H) = {np.linalg.cond(H):.6e}, trace coherence = {coherence:.6f}")

    config["H"] = H.tolist()
    config["h"] = h.tolist()
    config["cond_H"] = float(np.linalg.cond(H))
    config["trace_coherence"] = float(coherence)
    return {
        "config": config,
        "H": H,
        "h": h,
        "ud": ud,
        "unit_traces": unit_traces,
        "ctx": ctx,
    }


UNOBSERVABLE_TFINAL = 8.0


def compare_unobservable_case(problem: dict) -> dict:
    """B2. Non-unique amplitudes: one source not yet observed at all.

    At a short final time the upstream source's plume has not reached any
    measurement boundary, so its diagonal of H is ~1e-23 and the objective is
    FLAT in that amplitude over the whole box.  The minimizer in ``a`` is then
    non-unique and the two solvers legitimately return different amplitudes;
    what must agree is the objective and the first-order optimality, which is
    exactly what is gated here.
    """
    H, h, ud, ctx = problem["H"], problem["h"], problem["ud"], problem["ctx"]
    unit_traces = problem["unit_traces"]
    lo, hi = PRODUCTION_AMPLITUDE_FLOOR, float(AMPLITUDE_BOUNDS[1])
    label = f"unobservable case (tfinal = {UNOBSERVABLE_TFINAL:g} s)"
    print(f"\n  -- {label} --")
    res = solve_both(H, h, lo, hi)

    def tracking(a: np.ndarray) -> float:
        model = np.zeros_like(ud)
        for amp, trace in zip(a, unit_traces):
            model = model + float(amp) * trace
        return float(ctx.tracking_objective(model, ud))

    j_zero = tracking(np.zeros(len(unit_traces)))
    res["tracking_enumeration"] = tracking(res["_a_enum"])
    res["tracking_osqp"] = tracking(res["_a_osqp"])
    res["tracking_zero_amplitude"] = j_zero
    res["tracking_abs_diff"] = abs(res["tracking_osqp"] - res["tracking_enumeration"])
    res["tracking_diff_over_j_zero"] = res["tracking_abs_diff"] / max(j_zero, 1e-300)
    res["H_diagonal"] = [float(H[0, 0]), float(H[1, 1])]
    res["osqp_status"] = _osqp_status(H, -h, lo, hi)
    res["optimality_enumeration_scaled"] = (
        res["optimality_enumeration"] / res["optimality_scale"]
    )
    res["optimality_osqp_scaled"] = res["optimality_osqp"] / res["optimality_scale"]

    print(f"     diag(H)                        {res['H_diagonal'][0]:.6e}, "
          f"{res['H_diagonal'][1]:.6e}")
    print(f"     cond(H)                        {res['cond_H']:.6e}")
    print(f"     a_enumeration = {np.array2string(res['_a_enum'], precision=16)}")
    print(f"     a_osqp        = {np.array2string(res['_a_osqp'], precision=16)}")
    print(f"     max |da|                       "
          f"{res['amplitude_max_abs_diff']:.6e}   <-- NON-UNIQUE, not a defect")
    print(f"     quadratic  abs / rel diff      {res['quadratic_abs_diff']:.3e}"
          f" / {res['quadratic_rel_diff']:.3e}")
    print(f"     tracking J enum / osqp         "
          f"{res['tracking_enumeration']:.6e} / {res['tracking_osqp']:.6e}"
          f"  (J(a=0) = {j_zero:.6e})")
    print(f"     tracking J diff / J(0)         "
          f"{res['tracking_diff_over_j_zero']:.3e}")
    print(f"     proj-grad residual scaled      "
          f"{res['optimality_enumeration_scaled']:.3e} / "
          f"{res['optimality_osqp_scaled']:.3e}")
    print(f"     OSQP status                    {res['osqp_status']['status']}")

    check(
        res["cond_H"] > COND_WELL_POSED,
        f"{label}: H is genuinely rank-deficient (cond {res['cond_H']:.3e} > "
        f"{COND_WELL_POSED:.0e}), so amplitudes need not match",
    )
    gate_comparison(res, label)
    check(
        res["tracking_diff_over_j_zero"] <= TOL_TRACK_REL,
        f"{label}: tracking objective agrees relative to J(a=0) "
        f"({res['tracking_diff_over_j_zero']:.3e} <= {TOL_TRACK_REL:.0e})",
    )
    check(
        res["osqp_status"]["status"] in sinit.OSQP_ACCEPTED_STATUSES,
        f"{label}: OSQP terminated with an accepted status "
        f"({res['osqp_status']['status']!r})",
    )
    return {label: {k: v for k, v in res.items() if not k.startswith("_")}}


def _boundary_observation_args():
    """The default (boundary) observation configuration, as argparse supplies it."""
    from types import SimpleNamespace

    return SimpleNamespace(
        observation_mode=ro.DEFAULT_OBSERVATION_MODE,
        sensor_node_ids=None,
        sensor_coordinates=None,
        sensor_weights=ro.DEFAULT_SENSOR_WEIGHT_MODE,
        sensor_equal_weight=ro.DEFAULT_SENSOR_EQUAL_WEIGHT,
        sensor_snap_tolerance=ro.DEFAULT_SENSOR_SNAP_TOLERANCE,
    )


def compare_two_source(problem: dict) -> dict:
    H, h, ud, ctx = problem["H"], problem["h"], problem["ud"], problem["ctx"]
    unit_traces = problem["unit_traces"]
    results = {}
    for label, (lo, hi) in (
        ("raw bounds (0.0, 2.0)", (float(AMPLITUDE_BOUNDS[0]), float(AMPLITUDE_BOUNDS[1]))),
        (
            f"production bounds ({PRODUCTION_AMPLITUDE_FLOOR}, {AMPLITUDE_BOUNDS[1]})",
            (PRODUCTION_AMPLITUDE_FLOOR, float(AMPLITUDE_BOUNDS[1])),
        ),
    ):
        print(f"\n  -- {label} --")
        res = solve_both(H, h, lo, hi)

        # Full tracking objective of each fitted amplitude vector, through the
        # repository's own objective (lambda = 0 here, so no regularization
        # term is added; it is reported for provenance).
        def tracking(a: np.ndarray) -> float:
            model = np.zeros_like(ud)
            for amp, trace in zip(a, unit_traces):
                model = model + float(amp) * trace
            return float(ctx.tracking_objective(model, ud))

        j_enum = tracking(res["_a_enum"])
        j_osqp = tracking(res["_a_osqp"])
        # J(a = 0) = 0.5 ||u_d||^2_obs is the physically meaningful scale: at the
        # exact fixed centers the fit drives J to the numerical floor, so a bare
        # ratio of two ~1e-20 numbers is meaningless.
        j_zero = tracking(np.zeros(len(unit_traces)))
        scale_j = max(1.0, abs(j_enum))
        res["tracking_enumeration"] = j_enum
        res["tracking_osqp"] = j_osqp
        res["tracking_zero_amplitude"] = j_zero
        res["tracking_abs_diff"] = abs(j_osqp - j_enum)
        res["tracking_rel_diff"] = abs(j_osqp - j_enum) / scale_j
        res["tracking_diff_over_j_zero"] = abs(j_osqp - j_enum) / max(j_zero, 1.0e-300)
        res["optimality_enumeration_scaled"] = (
            res["optimality_enumeration"] / res["optimality_scale"]
        )
        res["optimality_osqp_scaled"] = (
            res["optimality_osqp"] / res["optimality_scale"]
        )

        # OSQP termination status, measured on this exact problem.
        res["osqp_status"] = _osqp_status(H, -h, lo, hi)

        print(f"     a_enumeration = {np.array2string(res['_a_enum'], precision=16)}")
        print(f"     a_osqp        = {np.array2string(res['_a_osqp'], precision=16)}")
        print(f"     max |da|                       {res['amplitude_max_abs_diff']:.6e}")
        print(f"     quadratic  enum / osqp         {res['quadratic_enumeration']:.16e}"
              f" / {res['quadratic_osqp']:.16e}")
        print(f"     quadratic  abs / rel diff      {res['quadratic_abs_diff']:.3e}"
              f" / {res['quadratic_rel_diff']:.3e}")
        print(f"     tracking J enum / osqp         {j_enum:.16e} / {j_osqp:.16e}")
        print(f"     tracking J(a=0) reference      {j_zero:.16e}")
        print(f"     tracking J abs diff / / J(0)   {res['tracking_abs_diff']:.3e}"
              f" / {res['tracking_diff_over_j_zero']:.3e}")
        print(f"     bound violation enum / osqp    "
              f"{res['bound_violation_enumeration']:.3e} / "
              f"{res['bound_violation_osqp']:.3e}")
        print(f"     proj-grad residual abs         "
              f"{res['optimality_enumeration']:.3e} / {res['optimality_osqp']:.3e}"
              f"  (scale ||h||_inf = {res['optimality_scale']:.3e})")
        print(f"     proj-grad residual scaled      "
              f"{res['optimality_enumeration_scaled']:.3e} / "
              f"{res['optimality_osqp_scaled']:.3e}")
        print(f"     OSQP status                    {res['osqp_status']}")
        print(f"     wall time enum / osqp (ms)     "
              f"{res['seconds_enumeration'] * 1e3:.3f} / "
              f"{res['seconds_osqp'] * 1e3:.3f}")

        gate_comparison(res, label)
        check(
            res["tracking_rel_diff"] <= TOL_TRACK_REL,
            f"{label}: full tracking objective agrees "
            f"(rel diff {res['tracking_rel_diff']:.3e} <= {TOL_TRACK_REL:.0e})",
        )
        check(
            res["osqp_status"]["status"] in sinit.OSQP_ACCEPTED_STATUSES,
            f"{label}: OSQP terminated with an accepted status "
            f"({res['osqp_status']['status']!r})",
        )
        results[label] = {k: v for k, v in res.items() if not k.startswith("_")}
    return results


def _osqp_status(H: np.ndarray, b: np.ndarray, lo: float, hi: float) -> dict:
    """Re-solve with OSQP directly to record the raw termination record."""
    osqp, sp = sinit._require_osqp()
    n = b.size
    with sinit._suppressed_stdout_fd(sinit.OSQP_SUPPRESS_SOLVER_STDOUT):
        problem = osqp.OSQP()
        problem.setup(
            P=sp.triu(sp.csc_matrix(0.5 * (H + H.T)), format="csc"),
            q=np.asarray(b, dtype=np.float64),
            A=sp.identity(n, format="csc"),
            l=np.full(n, float(lo)),
            u=np.full(n, float(hi)),
            **sinit._osqp_settings(),
        )
        result = problem.solve()
    info = result.info
    return {
        "status": str(info.status),
        "status_val": int(info.status_val),
        "iter": int(info.iter),
        "obj_val": float(info.obj_val),
        "primal_residual": float(getattr(info, "prim_res", float("nan"))),
        "dual_residual": float(getattr(info, "dual_res", float("nan"))),
        "osqp_version": str(__import__("osqp").__version__),
    }


# ---------------------------------------------------------------------------
# C. Failure handling
# ---------------------------------------------------------------------------
def test_failure_handling() -> dict:
    section("C. Failure handling: no silent fallback, actionable missing-OSQP error")
    record = {}

    rejected = False
    try:
        sinit.validate_amplitude_solver("scipy")
    except ValueError as exc:
        rejected = "scipy" in str(exc)
    check(rejected, "an unknown solver name is rejected by validate_amplitude_solver")

    # Simulate OSQP being absent: hide the import and confirm the error names
    # the install command and the enumeration escape hatch, and that nothing
    # falls back to enumeration.
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "osqp" or name.startswith("osqp."):
            raise ImportError("No module named 'osqp'")
        return real_import(name, *args, **kwargs)

    saved_module = _sys.modules.pop("osqp", None)
    builtins.__import__ = blocked_import
    message = ""
    kind = None
    try:
        sinit._solve_box_qp(
            np.eye(2), -np.ones(2), 0.0, 2.0, solver=sinit.AMPLITUDE_SOLVER_OSQP
        )
    except sinit.SensitivityInitializationError as exc:
        kind = "SensitivityInitializationError"
        message = str(exc)
    except Exception as exc:  # pragma: no cover - diagnostic
        kind = type(exc).__name__
        message = str(exc)
    finally:
        builtins.__import__ = real_import
        if saved_module is not None:
            _sys.modules["osqp"] = saved_module

    record["missing_osqp_error_type"] = kind
    record["missing_osqp_error_message"] = message
    check(
        kind == "SensitivityInitializationError",
        f"a missing OSQP raises SensitivityInitializationError (got {kind})",
    )
    check(
        "pip install" in message and "--amplitude-solver enumeration" in message,
        "the missing-OSQP error names the install command and the enumeration option",
    )
    check(
        "check_inverse_env" in message,
        "the missing-OSQP error points at tools/check_inverse_env.py",
    )

    # availability probe round-trips
    check(
        sinit.amplitude_solver_available(sinit.AMPLITUDE_SOLVER_ENUMERATION),
        "enumeration is always available (no optional dependency)",
    )
    record["osqp_available"] = bool(
        sinit.amplitude_solver_available(sinit.AMPLITUDE_SOLVER_OSQP)
    )
    check(record["osqp_available"], "osqp is available in this interpreter")
    return record


# ---------------------------------------------------------------------------
# D. CLI plumbing
# ---------------------------------------------------------------------------
def test_cli_parsing() -> dict:
    section("D1. CLI: --amplitude-solver parsing and default")
    record = {}
    saved = _sys.argv
    try:
        base = [
            "run_optimization.py",
            "--mode",
            "location_and_intensity",
            "--stage",
            "smoke",
        ]
        _sys.argv = list(base)
        args = ro.parse_args()
        record["default"] = str(args.amplitude_solver)
        check(
            args.amplitude_solver == sinit.AMPLITUDE_SOLVER_OSQP,
            f"the CLI default joint amplitude solver is osqp (got "
            f"{args.amplitude_solver!r})",
        )
        for name in sinit.AMPLITUDE_SOLVERS:
            _sys.argv = base + ["--amplitude-solver", name]
            args = ro.parse_args()
            check(
                args.amplitude_solver == name,
                f"--amplitude-solver {name} parses to {name!r}",
            )
    finally:
        _sys.argv = saved

    completed = subprocess.run(
        [
            PYTHON,
            str(APP_DIR / "run_optimization.py"),
            "--mode",
            "location_and_intensity",
            "--stage",
            "smoke",
            "--amplitude-solver",
            "nonexistent",
        ],
        cwd=str(APP_DIR),
        capture_output=True,
        text=True,
    )
    record["bad_choice_returncode"] = completed.returncode
    check(
        completed.returncode != 0
        and "invalid choice" in (completed.stderr + completed.stdout),
        "an invalid --amplitude-solver is rejected by argparse before any solve",
    )

    # End-to-end dependency behavior, through the real driver. A shim directory
    # on PYTHONPATH makes `import osqp` fail, so this exercises the startup
    # validation without uninstalling anything. Neither invocation runs an
    # optimization: the osqp one dies in validation, and the enumeration one is
    # given a deliberately invalid --amplitude-floor so it stops at the NEXT
    # validation step, which proves the solver validation let it through.
    shim = Path(_os.environ.get("TMPDIR", "/tmp")) / "app1_amplitude_solver_no_osqp"
    shim.mkdir(parents=True, exist_ok=True)
    (shim / "osqp.py").write_text(
        "raise ImportError(\"No module named 'osqp'\")\n", encoding="utf-8"
    )
    env = dict(_os.environ)
    env["PYTHONPATH"] = str(shim) + _os.pathsep + env.get("PYTHONPATH", "")
    base_cmd = [
        PYTHON,
        str(APP_DIR / "run_optimization.py"),
        "--mode",
        "location_and_intensity",
        "--stage",
        "smoke",
    ]

    missing = subprocess.run(
        base_cmd, cwd=str(APP_DIR), capture_output=True, text=True, env=env
    )
    text = missing.stdout + missing.stderr
    record["missing_osqp_cli_returncode"] = missing.returncode
    check(
        missing.returncode != 0
        and "pip install 'osqp>=1.0'" in text
        and "--amplitude-solver enumeration" in text
        and "check_inverse_env.py" in text,
        "with OSQP unavailable the DEFAULT run fails at startup with the "
        "install command, the checker, and the enumeration alternative",
    )
    check(
        "Optimization" not in text and "iterations" not in text,
        "the missing-OSQP failure happens before any optimization output",
    )

    fallback = subprocess.run(
        base_cmd + ["--amplitude-solver", "enumeration", "--amplitude-floor", "99"],
        cwd=str(APP_DIR),
        capture_output=True,
        text=True,
        env=env,
    )
    text = fallback.stdout + fallback.stderr
    record["enumeration_without_osqp_returncode"] = fallback.returncode
    check(
        "--amplitude-floor must lie in" in text,
        "with OSQP unavailable, --amplitude-solver enumeration still passes "
        "solver validation (it reaches the next validation step)",
    )
    return record


def test_cli_reaches_solver() -> dict:
    """Prove the selection reaches EVERY joint-fit call path.

    Deliberately tiny: a 2-node candidate set at the two fixed centers and a
    short tfinal.  This is a plumbing check, not a physics result, and it never
    runs the full optimization.
    """
    section("D2. The selection reaches every joint amplitude-fit call path")
    centers = np.asarray(
        [[TRUE_CONTROL[0], TRUE_CONTROL[1]], [TRUE_CONTROL[3], TRUE_CONTROL[4]]],
        dtype=np.float64,
    )
    mesh_data = ro.configure_measurement_boundaries(
        build_mesh_data(), list(MEASUREMENT_BOUNDARIES)
    )
    mesh_data = ro.configure_observation(mesh_data, _boundary_observation_args())
    pde = dict(ro.PDE)
    pde["tfinal"] = UNOBSERVABLE_TFINAL  # plumbing only: cheap forward solves
    admissibility = MeshAdmissibility(mesh_data)
    ud = gt.run_forward(mesh_data, pde, np.asarray(TRUE_CONTROL))

    record = {}
    real_solve = sinit._solve_box_qp
    for requested in sinit.AMPLITUDE_SOLVERS:
        seen: list[str] = []

        def spy(G, b, lo, hi, *, solver=sinit.AMPLITUDE_SOLVER_ENUMERATION):
            seen.append(str(solver))
            return real_solve(G, b, lo, hi, solver=solver)

        sinit._solve_box_qp = spy
        try:
            control, _diag, extras = sinit.build_sensitivity_initial_control(
                mesh_data,
                pde,
                ud,
                2,
                (PRODUCTION_AMPLITUDE_FLOOR, float(AMPLITUDE_BOUNDS[1])),
                admissibility,
                eta1=ro.ETA1,
                lam=ro.LAMBDA_REG,
                objective_time_rule=ro.OBJECTIVE_TIME_RULE,
                candidate_xy=centers,
                score="predicted_reduction",
                top_k=2,
                beam_width=1,
                refine_centers=True,
                refine_max_fev=3,
                refine_pair=True,
                pair_refine_max_fev=3,
                reselect_second=True,
                reselect_max_passes=1,
                reselect_refine_max_fev=3,
                reselect_threshold=0.0,  # force the re-selection pass to run
                verbose=False,
                amplitude_solver=requested,
            )
        finally:
            sinit._solve_box_qp = real_solve

        record[requested] = {
            "n_joint_fits": len(seen),
            "distinct_solvers_seen": sorted(set(seen)),
            "extras_amplitude_solver": extras.get("amplitude_solver"),
            "control": [float(v) for v in control],
        }
        print(f"  --amplitude-solver {requested}: {len(seen)} joint amplitude fits, "
              f"solvers seen {sorted(set(seen))}")
        check(
            len(seen) > 0,
            f"--amplitude-solver {requested}: the joint fit was actually exercised "
            f"({len(seen)} calls)",
        )
        check(
            set(seen) == {requested},
            f"--amplitude-solver {requested}: every joint fit used {requested!r} "
            f"(saw {sorted(set(seen))})",
        )
        check(
            extras.get("amplitude_solver") == requested,
            f"--amplitude-solver {requested}: recorded in the initializer extras",
        )
    # The two selections solve the same QP, so the CENTERS must coincide. The
    # AMPLITUDES need not: at this deliberately short tfinal the upstream
    # source has not reached any measurement boundary, H is rank-deficient to
    # ~1e-23 and its amplitude is unidentifiable. That exact (H, h) is compared
    # properly in check B2; here only the plumbing is at stake.
    ctrl_enum = np.asarray(record["enumeration"]["control"]).reshape(-1, 3)
    ctrl_osqp = np.asarray(record["osqp"]["control"]).reshape(-1, 3)
    center_diff = float(np.max(np.abs(ctrl_enum[:, :2] - ctrl_osqp[:, :2])))
    amp_diff = float(np.max(np.abs(ctrl_enum[:, 2] - ctrl_osqp[:, 2])))
    record["plumbing_center_max_abs_diff_m"] = center_diff
    record["plumbing_amplitude_max_abs_diff"] = amp_diff
    print(f"  plumbing run: center max |diff| {center_diff:.3e} m, "
          f"amplitude max |diff| {amp_diff:.3e}")
    check(
        center_diff <= 1.0e-9,
        f"plumbing run: both selections place identical centers "
        f"(max |diff| {center_diff:.3e} m <= 1e-09)",
    )
    if amp_diff > TOL_AMP_ABS:
        print(
            f"  [NOTE] the plumbing run's amplitude difference "
            f"({amp_diff:.3e}) is the unidentifiable-amplitude case of check "
            f"B2 (tfinal = {UNOBSERVABLE_TFINAL:g} s), not a solver "
            "disagreement: the objective there is flat in that amplitude."
        )
    return record


# ---------------------------------------------------------------------------
def write_report(payload: dict) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORT_DIR / "amplitude_solver_comparison.json"
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\nmachine-readable comparison: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore the cached unit responses and re-run the 3 forward solves.",
    )
    parser.add_argument(
        "--skip-plumbing",
        action="store_true",
        help="Skip check D2 (the only check that runs forward solves beyond part B).",
    )
    opts = parser.parse_args()

    print(f"python: {PYTHON}")
    print(f"numpy: {np.__version__}")
    try:
        import osqp

        print(f"osqp: {osqp.__version__}")
    except ImportError:
        print("osqp: NOT INSTALLED")

    RECORDS["python"] = PYTHON
    RECORDS["tolerances"] = {
        "TOL_OBJ_REL": TOL_OBJ_REL,
        "TOL_TRACK_REL": TOL_TRACK_REL,
        "TOL_BOUND_ABS": TOL_BOUND_ABS,
        "TOL_OPT_REL": TOL_OPT_REL,
        "TOL_AMP_ABS": TOL_AMP_ABS,
        "COND_WELL_POSED": COND_WELL_POSED,
        "STRESS_MAX_ACCEPTED_KKT": STRESS_MAX_ACCEPTED_KKT,
        "STRESS_MAX_OBJECTIVE_GAP": STRESS_MAX_OBJECTIVE_GAP,
        "STRESS_MAX_REFUSAL_RATE": STRESS_MAX_REFUSAL_RATE,
        "STRESS_MIN_REFUSAL_COND": STRESS_MIN_REFUSAL_COND,
    }
    RECORDS["osqp_settings"] = {
        "eps_abs": sinit.OSQP_EPS_ABS,
        "eps_rel": sinit.OSQP_EPS_REL,
        "max_iter": sinit.OSQP_MAX_ITER,
        "accepted_statuses": list(sinit.OSQP_ACCEPTED_STATUSES),
        "bound_tol": sinit.OSQP_BOUND_TOL,
        "kkt_tol": sinit.OSQP_KKT_TOL,
    }

    RECORDS["random_qps"] = test_random_qps()
    RECORDS["illconditioned_stress"] = test_illconditioned_stress()
    RECORDS["source_count_limit"] = test_source_count_limit()

    section("B. Application1 two-source case: one common (H, h)")
    problem = build_two_source_problem(use_cache=not opts.no_cache)
    RECORDS["two_source_case"] = problem["config"]
    RECORDS["two_source_comparison"] = compare_two_source(problem)

    section("B2. Same centers, unobservable first source: non-unique amplitudes")
    degenerate = build_two_source_problem(
        use_cache=not opts.no_cache,
        tfinal=UNOBSERVABLE_TFINAL,
        cache_name=f"two_source_unit_traces_tfinal{UNOBSERVABLE_TFINAL:g}.npz",
    )
    RECORDS["unobservable_case"] = degenerate["config"]
    RECORDS["unobservable_comparison"] = compare_unobservable_case(degenerate)

    RECORDS["failure_handling"] = test_failure_handling()
    RECORDS["cli"] = test_cli_parsing()
    if opts.skip_plumbing:
        print("\n=== D2 skipped (--skip-plumbing) ===")
    else:
        RECORDS["cli_reaches_solver"] = test_cli_reaches_solver()

    RECORDS["failures"] = list(FAILURES)
    write_report(RECORDS)

    section("RESULT")
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED:")
        for message in FAILURES:
            print(f"  - {message}")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
