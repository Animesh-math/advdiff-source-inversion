"""Freeze the budget-matched five-source random-start benchmark BEFORE any PDE solve.

    PY prepare.py --exp-dir EXP

1. Reads the paper's successful five-source GN-diag run
   (post_init_scaling_validation_20261004_125952/cases/M5_S4_gndiag) and checks
   that it is the converged gn-diag run, not the earlier stalled unscaled run
   (five_source_M5_S4sensors_20261003_204754, iteration limit 500).
2. Verifies the reference PDE-sweep counts from the saved records and derives
   the total budget = initialization + local refinement (GN-diag construction,
   the driver's objective at the initialized control, and every L-BFGS-B
   evaluation including line search). One forward or one adjoint sweep = 1 unit.
3. Splits the budget equally over 10 starts; the integer remainder goes to
   r01, r02, ... (one extra sweep each), in the predetermined run order.
4. Samples 10 feasible five-source starts with the production sampler
   MeshAdmissibility.sample_control_in_domain (the --init-mode mesh_random
   sampler), one np.random.default_rng(seed) per start. Truth is never used.
5. Writes config/*.json, commands.tsv, manifest.json and SHA256 sums.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

def _find_app1() -> Path:
    """Applications/Application1: $APP1_DIR, else the first parent holding the driver.

    Works both from the repository copy (paper_experiments/...) and from the
    copy placed in <EXP>/scripts/ when EXP is under Applications/Application1/output/.
    """
    env = os.environ.get("APP1_DIR")
    if env:
        return Path(env).resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "run_optimization.py").is_file() and (parent / "multi_source_utils.py").is_file():
            return parent
    raise SystemExit("cannot locate Applications/Application1; set APP1_DIR")


APP1 = _find_app1()
sys.path.insert(0, str(APP1))

from mesh_admissibility import MeshAdmissibility  # noqa: E402
from mesh_data_config import build_mesh_data  # noqa: E402
import multi_source_utils as msu  # noqa: E402

PY = os.environ.get("PY", sys.executable)  # interpreter written into commands.tsv
OUT = APP1 / "output"
REF_DIR = OUT / "post_init_scaling_validation_20261004_125952" / "cases" / "M5_S4_gndiag"
if os.environ.get("REFERENCE_RUNS_DIR"):  # run regenerated with paper_experiments/run_reference.py
    REF_DIR = Path(os.environ["REFERENCE_RUNS_DIR"]).resolve() / "cases" / "M5_S4_gndiag"
STALLED_DIR = OUT / "five_source_M5_S4sensors_20261003_204754"
N_STARTS = 10
SEED_BASE = 2026111600  # seed(r) = SEED_BASE + r, r = 1..10
INIT_FLAGS = {"--init-mode", "--initial-control", "--case-name", "--max-paraview-frames"}
# Expected GN-diag construction cost for 15 active parameters, measured in
# random_start_baseline_20261114_224229 (every M5 run: 16 fwd / 15 adj) and
# consistent with the mesh study (6 parameters: 7 fwd / 6 adj): n+1 fwd, n adj.
GN_FWD_EXPECTED, GN_ADJ_EXPECTED = 16, 15


def parse_flags(argv):
    out, cur = [], None
    for tok in argv:
        if tok.startswith("--"):
            cur = [tok, []]
            out.append(cur)
        else:
            cur[1].append(tok)
    return out


def flatten(flags):
    res = []
    for f, vals in flags:
        res.append(f)
        res.extend(vals)
    return res


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", required=True)
    exp = Path(ap.parse_args().exp_dir).resolve()
    cfg = exp / "config"
    cfg.mkdir(exist_ok=True)

    inst = json.loads((REF_DIR / "run_instrumentation.json").read_text())
    summ = json.loads((REF_DIR / "full" / "optimization_summary.json").read_text())
    argv = inst["driver_argv"]
    flags = parse_flags(argv)
    fd = dict(flags)

    # ---- identity checks: this is the successful GN-diag five-source run ----
    checks = {
        "num_sources_5": fd["--num-sources"] == ["5"],
        "post_init_scaling_gn_diag": fd.get("--post-init-scaling") == ["gn-diag"],
        "init_mode_sensitivity": fd.get("--init-mode") == ["sensitivity"],
        "converged_message": "CONVERGENCE" in summ["message"],
        "success_flag": bool(summ["success"]),
        "final_objective_below_1e-10": summ["final_objective"] < 1e-10,
        "max_position_error_below_1e-6_m": summ["recovery_permutation_matched"]["max_position_error_m"] < 1e-6,
        "no_raw_infeasible_evaluations": summ["feasibility"]["raw_infeasible_evaluations"] == 0,
    }
    stalled = {}
    stalled_summ = STALLED_DIR / "full" / "optimization_summary.json"
    if not stalled_summ.exists():
        cand = sorted(STALLED_DIR.rglob("optimization_summary.json"))
        stalled_summ = cand[0] if cand else None
    if stalled_summ is not None and Path(stalled_summ).exists():
        s2 = json.loads(Path(stalled_summ).read_text())
        stalled = {"path": str(stalled_summ), "message": s2.get("message"),
                   "iterations": s2.get("iterations"), "final_objective": s2.get("final_objective"),
                   "post_init_scaling": (s2.get("post_init") or {}).get("scaling"),
                   "max_position_error_m": (s2.get("recovery_permutation_matched") or {}).get("max_position_error_m")}
    assert all(checks.values()), checks

    # ---- verified reference counts ----
    ph = inst["phases"]
    init_f, init_a = ph["initialization"]["forward_solves"], ph["initialization"]["adjoint_solves"]
    loc_f, loc_a = ph["local_optimization"]["forward_solves"], ph["local_optimization"]["adjoint_solves"]
    pi = summ["post_init"]
    stage = pi["stages"][0]
    nfev, njev = stage["n_objective_evaluations"], stage["n_gradient_evaluations"]
    # Decomposition of the local block (driver code path run_single_optimization):
    # 1 fwd objective at the initialized control (fresh stage evaluator) +
    # GN construction (n+1 fwd, n adj) + nfev fwd / njev adj for L-BFGS-B.
    decomposition_ok = (loc_f == 1 + GN_FWD_EXPECTED + nfev) and (loc_a == GN_ADJ_EXPECTED + njev)
    assert (init_f, init_a) == (3069, 12), (init_f, init_a)
    assert (loc_f, loc_a) == (84, 82), (loc_f, loc_a)
    assert decomposition_ok, (loc_f, loc_a, nfev, njev)
    total_f, total_a = init_f + loc_f, init_a + loc_a
    budget = total_f + total_a
    base, rem = divmod(budget, N_STARTS)
    alloc = [base + (1 if r <= rem else 0) for r in range(1, N_STARTS + 1)]
    assert sum(alloc) == budget

    reference = {
        "run_dir": str(REF_DIR),
        "selection": ("paper five-source result: M = 5, 31 node sensors (L1 boundary 25 + S4 interior 6), "
                      "sensitivity initialization + L-BFGS-B with --post-init-scaling gn-diag; converged. "
                      "NOT the earlier stalled unscaled run five_source_M5_S4sensors_20261003_204754."),
        "identity_checks": checks,
        "stalled_run_for_contrast": stalled,
        "driver_argv": argv,
        "true_control": [float(v) for v in fd["--true-control"]],
        "sensor_node_ids": fd["--sensor-node-ids"][0],
        "initialized_control": summ["initial_control"],
        "initialized_objective": summ["initial_objective"],
        "final_control": summ["final_control"],
        "final_objective": summ["final_objective"],
        "final_gradient_norm": summ["final_gradient_norm"],
        "iterations": summ["iterations"],
        "message": summ["message"],
        "success": summ["success"],
        "recovery_permutation_matched": summ["recovery_permutation_matched"],
        "feasibility": summ["feasibility"],
        "post_init": pi,
        "pde": summ["pde"],
        "active_bounds": summ["active_bounds"],
        "instrumentation_phases": ph,
        "git_commit_of_reference_run": (REF_DIR.parents[1] / "git_commit.txt").read_text().strip()
        if (REF_DIR.parents[1] / "git_commit.txt").exists() else None,
    }
    budget_rec = {
        "unit": "PDE sweep: one complete forward solve OR one complete adjoint solve = 1 unit",
        "source": str(REF_DIR / "run_instrumentation.json") + " (actual gt.run_forward / gt.rk4_discrete_source_adjoint calls)",
        "reference_initialization": {"forward": init_f, "adjoint": init_a, "sweeps": init_f + init_a},
        "reference_local_refinement": {
            "forward": loc_f, "adjoint": loc_a, "sweeps": loc_f + loc_a,
            "decomposition": {
                "objective_at_initialized_control": {"forward": 1, "adjoint": 0},
                "gn_diag_construction": {"forward": GN_FWD_EXPECTED, "adjoint": GN_ADJ_EXPECTED},
                "lbfgsb_evaluations_incl_line_search": {"forward": nfev, "adjoint": njev,
                                                        "nfev": nfev, "njev": njev,
                                                        "iterations": summ["iterations"]},
                "check": "84 = 1 + 16 + 67 and 82 = 15 + 67 (verified)",
            },
        },
        "total_budget": {"forward": total_f, "adjoint": total_a, "sweeps": budget},
        "paper_note": "the paper's 3,069 forward sweeps are the initialization forward count only; the "
                      "initialization also used 12 adjoint sweeps and local refinement 84 fwd + 82 adj.",
        "excluded_for_both_methods": {
            "synthetic_data_generation": "1 forward solve of the true control (needed by any method)",
            "driver_reporting_overhead": "reference: 4 fwd / 2 adj (forward smoke/export at the start control, "
                                         "objective/gradient smoke test, final re-evaluation/state export); "
                                         "these do not feed the optimizer. Measured and reported separately "
                                         "for every random start too.",
        },
        "n_starts": N_STARTS,
        "per_start_allocation_sweeps": {f"r{r:02d}": a for r, a in enumerate(alloc, 1)},
        "allocation_rule": f"floor({budget}/{N_STARTS}) = {base} each; remainder {rem} -> +1 sweep to r01..r{rem:02d} "
                           "(predetermined run order). No reallocation of unused allowance.",
        "enforcement": (
            "PDE-evaluation level inside the driver's budgeted block (objective at the start, GN-diag "
            "construction, L-BFGS-B): before every forward or adjoint solve the guard checks the remaining "
            "allowance; during L-BFGS-B a new objective evaluation (forward) is only started when at least 2 "
            "sweeps remain, so that its gradient (adjoint) can also be completed. Otherwise the run stops "
            "cleanly (no solve started) and returns the best feasible fully evaluated point (objective AND "
            "gradient computed, all source centers in the fluid mesh). The GN-diag construction is only "
            "started if its full cost plus one evaluation fits."),
    }

    # ---- starts ----
    mesh_data = build_mesh_data()
    adm = MeshAdmissibility(mesh_data)
    M = 5
    floor = float(fd["--amplitude-floor"][0])
    amp_bounds = (floor, float(msu.AMPLITUDE_BOUNDS[1]))
    assert list(summ["active_bounds"][2]) == list(amp_bounds), (summ["active_bounds"][2], amp_bounds)
    kept = [(f, v) for f, v in flags if f not in INIT_FLAGS and not f.startswith("--sensitivity-")]
    dropped = [(f, v) for f, v in flags if f in INIT_FLAGS or f.startswith("--sensitivity-")]
    starts, commands = [], []
    for r in range(1, N_STARTS + 1):
        seed = SEED_BASE + r
        rng = np.random.default_rng(seed)
        z0 = adm.sample_control_in_domain(M, amp_bounds, rng=rng, min_separation=None)
        z0 = [float(v) for v in z0]
        feas = all(adm.is_point_in_domain(z0[3 * i], z0[3 * i + 1]) for i in range(M))
        amp_ok = all(amp_bounds[0] <= z0[3 * i + 2] <= amp_bounds[1] for i in range(M))
        box = summ["active_bounds"]
        box_ok = all(lo <= v <= hi for v, (lo, hi) in zip(z0, box))
        assert feas and amp_ok and box_ok, (r, z0)
        rid = f"r{r:02d}"
        starts.append({"run_id": rid, "run_index": r, "seed": seed, "budget_sweeps": alloc[r - 1],
                       "initial_control": z0, "initial_control_repr": [repr(v) for v in z0],
                       "feasible": feas})
        dargv = flatten(kept) + ["--init-mode", "manual", "--initial-control", *[repr(v) for v in z0],
                                 "--case-name", rid, "--max-paraview-frames", "1"]
        cmd = [PY, str(exp / "scripts/run_case.py"), "--exp-dir", str(exp), "--case-id", rid,
               "--budget-sweeps", str(alloc[r - 1]), "--seed", str(seed), "--"] + dargv
        commands.append((rid, " ".join(f"'{t}'" if " " in t else t for t in cmd)))

    sampler = {
        "function": "mesh_admissibility.MeshAdmissibility.sample_control_in_domain(num_sources=5, "
                    "amplitude_bounds, rng=np.random.default_rng(seed), min_separation=None) "
                    "(the sampler behind --init-mode mesh_random)",
        "centers": "independent, uniform with respect to area over the meshed fluid region (triangle chosen "
                   "with probability proportional to area, then uniform in the triangle); building holes "
                   "have zero probability; point-in-mesh check guards round-off",
        "amplitudes": f"independent uniform on [{amp_bounds[0]}, {amp_bounds[1]}] = production L-BFGS-B amplitude box",
        "min_separation": None,
        "draw_order": "all 5 centers first, then 5 amplitudes, from the same rng",
        "seed_rule": f"seed = {SEED_BASE} + r, r = 1..10 (fresh seeds, not those of random_start_baseline_20261114_224229)",
        "truth_used": False,
        "resampling": "none; every sampled start is run",
        "fluid_mesh_area_m2": float(np.sum(adm.triangle_areas)),
        "mesh_triangles": int(len(mesh_data["connections"])),
    }
    case_cfg = {
        "common_driver_flags": flatten(kept),
        "removed_from_reference_argv": flatten(dropped),
        "added_per_start": ["--init-mode", "manual", "--initial-control", "<start>", "--case-name", "<rNN>",
                            "--max-paraview-frames", "1 (export only)"],
        "what_is_identical_to_reference": [
            "mesh DC_mesh/dc_navy_yard_mesh.msh", "velocity NPZ output/navier_stokes/application1_navier_stokes_velocity_nodes.npz (read only)",
            "31 sensor node ids, equal weights 1.0", "true control (synthetic noiseless data regenerated by the same forward model)",
            "sigma2 = 400 m^2 (no continuation)", "dt 0.1, tfinal 150", "mesh_domain bounds, amplitude floor 0.1 (box [0.1, 2])",
            "raw feasibility mode", "L-BFGS-B, --post-init-scaling gn-diag (max ratio 1e4), maxcor 10 (SciPy default, as reference), "
            "ftol 1e-14, gtol 1e-9, maxls 20, --max-iter 500 (not binding under the budget)"],
        "not_used": ["sensitivity initialization", "beam search", "center/pair refinement", "second-source reselection",
                     "sigma continuation", "truth for start selection or stopping"],
    }
    (cfg / "reference.json").write_text(json.dumps(reference, indent=2, default=str) + "\n")
    (cfg / "budget.json").write_text(json.dumps(budget_rec, indent=2) + "\n")
    (cfg / "starts.json").write_text(json.dumps({"sampler": sampler, "starts": starts}, indent=2) + "\n")
    (cfg / "case_config.json").write_text(json.dumps(case_cfg, indent=2) + "\n")
    with (cfg / "starts.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run_id", "seed", "budget_sweeps"] + [f"{c}{i}" for i in range(1, 6) for c in ("x", "y", "A")])
        for s in starts:
            w.writerow([s["run_id"], s["seed"], s["budget_sweeps"]] + s["initial_control"])
    (exp / "commands.tsv").write_text("".join(f"{rid}\t{cmd}\n" for rid, cmd in commands))
    sums = {p: sha(cfg / p) for p in ("reference.json", "budget.json", "starts.json", "case_config.json")}
    sums["commands.tsv"] = sha(exp / "commands.tsv")
    (cfg / "SHA256SUMS.json").write_text(json.dumps(sums, indent=2) + "\n")
    print(json.dumps({"budget": budget_rec["total_budget"], "alloc": alloc, "seeds": [s["seed"] for s in starts]}))
    print(f"prepared {len(starts)} starts at {time.strftime('%F %T')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
