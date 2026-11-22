"""Freeze the random-start baseline BEFORE any optimization.

    PY prepare.py --exp-dir EXP

* reads the paper's reference runs (driver argv + resolved args + summaries),
* derives each case's random-start driver argv from the reference argv by
  changing ONLY the initialization flags (and the ParaView frame count, which
  is export-only),
* samples all 50 starting controls with the production mesh_random sampler
  (MeshAdmissibility.sample_control_in_domain) from fixed recorded seeds,
* writes config/{cases.json, starts.json, starts.csv, recovery_criterion.json,
  reference.json}, commands.tsv and manifest.json.

No PDE solve happens here. The true sources are never used to accept or
reject a start (they are only copied into the driver argv, as in the
reference runs).
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
GN = OUT / "gn_diag_default_check_20261004_161718"
SL = OUT / "sensor_layout_comparison_20261003_151233"
PS = OUT / "post_init_scaling_validation_20261004_125952"

# Paper cases, in the predetermined order. Each reference = the selected
# (paper) sensitivity-initialized run. For the two-source cases the paper run
# is a two-step chain: the sensitivity initializer of the sensor-layout
# comparison, then a gn-diag L-BFGS-B restart from its saved initialized
# control (gn_diag_default_check). For M5 it is one run.
CASES = [
    {"case": "A_L1", "label": "A-L1: two sources, 25 boundary sensors",
     "post_dir": GN / "cases/P_A_L1_boundary__gn-diag", "init_dir": SL / "cases/A_L1_boundary"},
    {"case": "A_L2", "label": "A-L2: two sources, mixed sensors (10 boundary + 15 interior)",
     "post_dir": GN / "cases/P_A_L2_mixed__gn-diag", "init_dir": SL / "cases/A_L2_mixed"},
    {"case": "B_L1", "label": "B-L1: two sources, 25 boundary sensors",
     "post_dir": GN / "cases/P_B_L1_boundary__gn-diag", "init_dir": SL / "cases/B_L1_boundary"},
    {"case": "B_L2", "label": "B-L2: two sources, mixed sensors (10 boundary + 15 interior)",
     "post_dir": GN / "cases/P_B_L2_mixed__gn-diag", "init_dir": SL / "cases/B_L2_mixed"},
    {"case": "M5_S4", "label": "M5: five sources, 31 sensors (25 boundary L1 + 6 interior S4)",
     "post_dir": PS / "cases/M5_S4_gndiag", "init_dir": PS / "cases/M5_S4_gndiag"},
]
# Alternative reference location: runs regenerated with
# paper_experiments/run_reference.py (REFERENCE_RUNS_DIR=<its --out-dir>).
if os.environ.get("REFERENCE_RUNS_DIR"):
    _RR = Path(os.environ["REFERENCE_RUNS_DIR"]).resolve() / "cases"
    _LAYOUT = {"A_L1": "A_L1_boundary", "A_L2": "A_L2_mixed", "B_L1": "B_L1_boundary", "B_L2": "B_L2_mixed"}
    for _c in CASES:
        if _c["case"] in _LAYOUT:
            _c["post_dir"] = _RR / f"restart_{_LAYOUT[_c['case']]}"
            _c["init_dir"] = _RR / f"init_{_LAYOUT[_c['case']]}"
        else:
            _c["post_dir"] = _c["init_dir"] = _RR / "M5_S4_gndiag"
RUNS_PER_CASE = 10
SEED_BASE = 20261114
# seed(case_index k = 1..5, run r = 1..10) = SEED_BASE + 1000*k + r
RECOVERY = {
    "definition": ("recovered = all final centers inside the fluid mesh (raw feasibility) AND "
                   "max matched center error <= 1.0 m AND max matched amplitude error <= 0.01"),
    "max_position_error_m": 1.0,
    "max_amplitude_error": 0.01,
    "require_feasible_centers": True,
    "matching": ("run_optimization.match_sources_by_permutation: label permutation minimizing the "
                 "TOTAL center position error; amplitude errors taken under that same permutation "
                 "(the convention of optimization_summary.json recovery_permutation_matched)"),
    "recorded_before_launch": True,
    "note": ("used only for post-hoc reporting; it never stops or alters an optimization. "
             "Continuous errors are stored for every run so other thresholds can be applied."),
}

# Flags that encode HOW the initial control is obtained, or are export-only.
INIT_FLAGS = {"--init-mode", "--initial-control", "--case-name", "--max-paraview-frames"}


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

    mesh_data = build_mesh_data()
    adm = MeshAdmissibility(mesh_data)
    n_tri = int(len(mesh_data["connections"]))
    n_sample_tri = int(len(adm._sample_triangle_indices))
    fluid_area = float(np.sum(adm.triangle_areas))

    cases_out, starts, commands, reference = [], [], [], {}
    for k, c in enumerate(CASES, start=1):
        post_inst = json.loads((c["post_dir"] / "run_instrumentation.json").read_text())
        post_sum = json.loads((c["post_dir"] / "full/optimization_summary.json").read_text())
        init_inst = json.loads((c["init_dir"] / "run_instrumentation.json").read_text())
        init_sum = json.loads((c["init_dir"] / "full/optimization_summary.json").read_text())
        ref_argv = post_inst["driver_argv"]
        flags = parse_flags(ref_argv)
        names = [f for f, _ in flags]
        assert "--post-init-scaling" in names and dict(flags)["--post-init-scaling"] == ["gn-diag"], c
        kept = [(f, v) for f, v in flags
                if f not in INIT_FLAGS and not f.startswith("--sensitivity-")]
        dropped = [(f, v) for f, v in flags
                   if f in INIT_FLAGS or f.startswith("--sensitivity-")]
        kd = dict(kept)
        M = int(kd["--num-sources"][0])
        true_control = [float(v) for v in kd["--true-control"]]
        assert len(true_control) == 3 * M
        floor = float(kd["--amplitude-floor"][0])
        amp_bounds = (floor, float(msu.AMPLITUDE_BOUNDS[1]))
        # Same initialized control in the reference chain (init run -> restart)?
        ref_init_control = post_sum["initial_control"]
        chain_dx = float(np.max(np.abs(np.asarray(init_sum["initial_control"]) - np.asarray(ref_init_control))))

        reference[c["case"]] = {
            "label": c["label"],
            "post_init_run_dir": str(c["post_dir"]),
            "initializer_run_dir": str(c["init_dir"]),
            "same_run": c["post_dir"] == c["init_dir"],
            "initialized_control": ref_init_control,
            "init_run_initialized_control_max_abs_diff_vs_restart_start": chain_dx,
            "post_init_driver_argv": ref_argv,
            "initializer_driver_argv": init_inst["driver_argv"],
            "phases_initializer_run": init_inst["phases"],
            "phases_post_init_run": post_inst["phases"],
            "post_init_summary": post_sum["post_init"],
            "initial_objective": post_sum["initial_objective"],
            "final_objective": post_sum["final_objective"],
            "final_gradient_norm": post_sum["final_gradient_norm"],
            "iterations": post_sum["iterations"],
            "message": post_sum["message"],
            "final_control": post_sum["final_control"],
            "recovery_permutation_matched": post_sum["recovery_permutation_matched"],
            "feasibility": post_sum.get("feasibility"),
            "max_iter": int(kd["--max-iter"][0]),
            "lbfgsb_maxcor": post_sum["post_init"].get("lbfgsb_maxcor"),
            "pde": post_sum["pde"],
        }

        case_rec = {
            "case": c["case"], "case_index": k, "label": c["label"], "num_sources": M,
            "true_control": true_control,
            "reference_post_init_run": str(c["post_dir"]),
            "reference_initializer_run": str(c["init_dir"]),
            "common_driver_flags": flatten(kept),
            "removed_from_reference_argv": flatten(dropped),
            "added_per_run": ["--init-mode", "manual", "--initial-control", "<sampled start>",
                              "--case-name", "<run id>", "--max-paraview-frames", "1"],
            "amplitude_sampling_interval": list(amp_bounds),
            "lbfgsb_amplitude_bounds": list(amp_bounds),
            "max_iter": int(kd["--max-iter"][0]),
            "reference_pde": post_sum["pde"],
            "reference_active_bounds": post_sum["active_bounds"],
            "reference_observation": {k2: post_sum["observation"].get(k2) for k2 in
                                      ("num_sensors", "sensor_node_ids", "sensor_coordinates",
                                       "sensor_weights", "weight_mode")},
        }
        cases_out.append(case_rec)

        for r in range(1, RUNS_PER_CASE + 1):
            seed = SEED_BASE + 1000 * k + r
            rng = np.random.default_rng(seed)
            z0 = adm.sample_control_in_domain(M, amp_bounds, rng=rng, min_separation=None)
            ok = all(adm.is_point_in_domain(float(z0[3 * i]), float(z0[3 * i + 1])) for i in range(M))
            assert ok and all(amp_bounds[0] <= z0[3 * i + 2] <= amp_bounds[1] for i in range(M))
            run_id = f"{c['case']}__r{r:02d}"
            z0l = [float(v) for v in z0]
            starts.append({"run_id": run_id, "case": c["case"], "run_index": r, "seed": seed,
                           "num_sources": M, "initial_control": z0l,
                           "initial_control_repr": [repr(v) for v in z0l]})
            argv = flatten(kept) + ["--init-mode", "manual", "--initial-control",
                                    *[repr(v) for v in z0l], "--case-name", run_id,
                                    "--max-paraview-frames", "1"]
            cmd = [PY, str(exp / "scripts/run_case.py"), "--exp-dir", str(exp), "--case-id", run_id,
                   "--"] + argv
            commands.append((run_id, " ".join(f"'{t}'" if " " in t else t for t in cmd)))

    sampler = {
        "function": "mesh_admissibility.MeshAdmissibility.sample_control_in_domain "
                    "(the sampler behind --init-mode mesh_random), called directly with a per-run "
                    "np.random.default_rng(seed); the driver's mesh_random path instead draws all "
                    "starts from one rng(--random-seed) sequentially -- same distribution, "
                    "independent per-run streams here",
        "spatial_distribution": (
            "uniform with respect to AREA over the meshed fluid region: a triangle is chosen with "
            "probability proportional to its area, then a point uniform in that triangle "
            "(barycentric sqrt construction); followed by a point-in-mesh check (100 retries) "
            "that only guards round-off. Building holes are not meshed, so they get zero "
            "probability. Triangles with area <= tol^2 are excluded (none in practice)."),
        "mesh_triangles": n_tri, "sampling_triangles": n_sample_tri,
        "fluid_mesh_area_m2": fluid_area,
        "centers_sampled_independently": True,
        "min_separation": None,
        "separation_note": "no separation restriction (min_separation=None -> 0.0; the "
                           "driver's mesh_random default when --min-source-separation is not given)",
        "amplitudes": "independent uniform on [amplitude floor, upper bound] = the L-BFGS-B amplitude box",
        "draw_order": "all M centers first, then M amplitudes, from the same rng",
        "truth_used_for_acceptance": False,
        "resampling_or_replacement": "none; every sampled start is run",
        "seed_rule": f"seed = {SEED_BASE} + 1000*case_index + run_index (case_index 1..5 in case order, run_index 1..10)",
    }

    (cfg / "cases.json").write_text(json.dumps(cases_out, indent=2) + "\n")
    (cfg / "reference.json").write_text(json.dumps(reference, indent=2, default=str) + "\n")
    (cfg / "starts.json").write_text(json.dumps({"sampler": sampler, "starts": starts}, indent=2) + "\n")
    with (cfg / "starts.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run_id", "case", "run_index", "seed", "initial_control"])
        for s in starts:
            w.writerow([s["run_id"], s["case"], s["run_index"], s["seed"], json.dumps(s["initial_control"])])
    (cfg / "recovery_criterion.json").write_text(json.dumps(RECOVERY, indent=2) + "\n")
    (exp / "commands.tsv").write_text("".join(f"{rid}\t{cmd}\n" for rid, cmd in commands))
    sums = {p: sha(cfg / p) for p in ("cases.json", "starts.json", "recovery_criterion.json", "reference.json")}
    sums["commands.tsv"] = sha(exp / "commands.tsv")
    (cfg / "SHA256SUMS.json").write_text(json.dumps(sums, indent=2) + "\n")
    print(f"prepared {len(starts)} starts for {len(cases_out)} cases; "
          f"mesh triangles {n_tri}, fluid area {fluid_area:.1f} m^2; written {time.strftime('%F %T')}")
    for k, v in reference.items():
        print(f"  {k}: ref init chain max|dz| = {v['init_run_initialized_control_max_abs_diff_vs_restart_start']:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
